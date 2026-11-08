import asyncio
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING
from zipfile import BadZipFile, ZipFile

from adapter.cache.ttl import AsyncTTLCache, CacheBusyError
from port.exceptions.recommendation import RecommendationUnavailableError
from port.recommendation import ModelStatus, RankedMovie, RankedRecommendations

if TYPE_CHECKING:
    from ml.model import ItemModel


class ArtifactRecommendationEngine:
    def __init__(self, path: Path | None, *, max_bytes: int = 512 * 1024 * 1024) -> None:
        self._path = path
        self._model: ItemModel | None = None
        self._manifest: dict = {}
        self._status: ModelStatus = "not_configured" if path is None else "unavailable"
        self._max_bytes = max_bytes
        self._load_task: asyncio.Task[None] | None = None
        self._cache = AsyncTTLCache(max_entries=64, max_pending=8, concurrency=2)
        self._lock = asyncio.Lock()

    async def status(self) -> ModelStatus:
        if self._path is None:
            return "not_configured"
        async with self._lock:
            if self._load_task is None:
                self._load_task = asyncio.create_task(self._initialize())
        await asyncio.shield(self._load_task)
        return self._status

    async def _initialize(self) -> None:
        try:
            await asyncio.to_thread(self._load)
            self._status = "ready"
        except (ImportError, OSError, ValueError, KeyError, TypeError, BadZipFile):
            self._status = "unavailable"

    async def close(self) -> None:
        await self._cache.close()
        if self._load_task is not None:
            await asyncio.gather(self._load_task, return_exceptions=True)

    def _load(self) -> None:
        from ml.model import ItemModel

        files = [
            self._path / name
            for name in ("manifest.json", "movies.json", "similarities.npz", "popularity.npy")
        ]
        if sum(path.stat().st_size for path in files) > self._max_bytes:
            raise ValueError("Artifact exceeds memory budget")
        with ZipFile(self._path / "similarities.npz") as archive:
            expanded = sum(info.file_size for info in archive.infolist())
        if expanded + sum(path.stat().st_size for path in files if path.suffix != ".npz") > self._max_bytes:
            raise ValueError("Expanded artifact exceeds memory budget")
        model, manifest = ItemModel.load(self._path)
        if manifest["strategy"] not in ("item_knn", "popularity"):
            raise ValueError("Invalid artifact strategy")
        if not isinstance(manifest.get("model_version"), str) or not manifest["model_version"].strip():
            raise ValueError("Invalid artifact version")
        self._model, self._manifest = model, manifest

    async def recommend(self, ratings: Mapping[int, float], limit: int) -> RankedRecommendations:
        if await self.status() != "ready":
            raise RecommendationUnavailableError
        try:
            return await self._cache.get_or_load(
                (tuple(sorted(ratings.items())), limit),
                lambda: asyncio.to_thread(self._recommend, dict(ratings), limit),
                ttl=30,
            )
        except CacheBusyError as error:
            raise RecommendationUnavailableError from error

    def _recommend(self, ratings: dict[int, float], limit: int) -> RankedRecommendations:
        if self._model is None:
            raise RecommendationUnavailableError
        model = self._model
        items, strategy = model.recommend(ratings, limit=limit, strategy=self._manifest["strategy"])
        known = {key: value for key, value in ratings.items() if key in model.by_tmdb}
        reason = None
        if strategy == "popularity":
            reason = (
                "baseline_selected"
                if self._manifest["strategy"] == "popularity"
                else (
                    "no_known_positive_ratings"
                    if not any(value >= 4 for value in known.values())
                    else "no_positive_candidates"
                )
            )
        return RankedRecommendations(
            tuple(
                RankedMovie(
                    item.tmdb_id,
                    item.title,
                    "Animation" in model.films[model.by_tmdb[item.tmdb_id]].genres.split("|"),
                    item.score,
                )
                for item in items
            ),
            strategy,
            self._manifest["model_version"],
            len(known),
            reason,
        )

    async def explore(self, ratings: Mapping[int, float], limit: int, seed: int) -> RankedRecommendations:
        if await self.status() != "ready":
            raise RecommendationUnavailableError
        try:
            return await self._cache.get_or_load(
                ("explore", tuple(sorted(ratings.items())), limit, seed),
                lambda: asyncio.to_thread(self._explore, dict(ratings), limit, seed),
                ttl=30,
            )
        except CacheBusyError as error:
            raise RecommendationUnavailableError from error

    def _explore(self, ratings: dict[int, float], limit: int, seed: int) -> RankedRecommendations:
        if self._model is None:
            raise RecommendationUnavailableError
        model = self._model
        items = model.explore(ratings, limit=limit, seed=seed)
        return RankedRecommendations(
            tuple(
                RankedMovie(
                    item.tmdb_id,
                    item.title,
                    "Animation" in model.films[model.by_tmdb[item.tmdb_id]].genres.split("|"),
                    item.score,
                    exploratory=broad,
                )
                for item, broad in items
            ),
            "item_knn" if any(not broad for _, broad in items) else "exploration",
            self._manifest["model_version"],
            sum(key in model.by_tmdb for key in ratings),
            None if any(not broad for _, broad in items) else "no_positive_candidates",
            mode="surprise",
            seed=seed,
        )
