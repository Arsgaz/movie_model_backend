from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy import sparse

from ml.dataset import Film, Observation, digest, write_json


@dataclass(frozen=True)
class Parameters:
    neighbors: int = 80
    shrinkage: float = 10.0
    min_support: int = 2
    max_user_positives: int = 0

    def __post_init__(self) -> None:
        if self.neighbors < 1 or self.shrinkage < 0 or self.min_support < 1 or self.max_user_positives < 0:
            raise ValueError("Invalid item model parameters")


@dataclass(frozen=True)
class Recommendation:
    movie_id: int
    tmdb_id: int
    title: str
    score: float


class ItemModel:
    """Sparse cosine similarity of positive feedback, weighted by co-rating support."""

    def __init__(
        self,
        films: Sequence[Film],
        similarities: sparse.csr_matrix,
        popularity: np.ndarray,
        parameters: Parameters,
    ) -> None:
        self.films = tuple(films)
        self.similarities, self.popularity, self.parameters = similarities, popularity, parameters
        self.by_movie = {film.movie_id: index for index, film in enumerate(self.films)}
        self.by_tmdb = {film.tmdb_id: index for index, film in enumerate(self.films)}
        similarities.check_format(full_check=True)
        n = len(self.films)
        if len(self.by_movie) != n or len(self.by_tmdb) != n:
            raise ValueError("Duplicate model movie identity")
        if similarities.shape != (n, n) or popularity.shape != (n,):
            raise ValueError("Inconsistent model dimensions")
        if (
            not np.isfinite(similarities.data).all()
            or not np.isfinite(popularity).all()
            or (similarities.data < 0).any()
            or (popularity < 0).any()
        ):
            raise ValueError("Invalid model weights")

    @classmethod
    def fit(
        cls,
        films: Sequence[Film],
        ratings: Sequence[Observation],
        parameters: Parameters,
        *,
        block_size: int = 128,
        progress: Callable[[str], None] | None = None,
    ) -> "ItemModel":
        from ml.large_dataset import DiskRatings

        if block_size < 1:
            raise ValueError("Block size must be positive")
        if isinstance(ratings, DiskRatings):
            user = ratings.column("user")
            movie = ratings.column("movie")
            value = ratings.column("value")
            timestamp = ratings.column("time")
        else:
            user = np.array([row.user_id for row in ratings], dtype=np.int32)
            movie = np.array([row.movie_id for row in ratings], dtype=np.int32)
            value = np.array([row.rating for row in ratings], dtype=np.float32)
            timestamp = np.array([row.timestamp for row in ratings], dtype=np.int64)
        active = np.unique(movie)
        active_set = set(active.tolist())
        catalog = sorted((film for film in films if film.movie_id in active_set), key=lambda f: f.movie_id)
        if not catalog:
            raise ValueError("Training set is empty")
        if len(catalog) != len(active):
            raise ValueError("Ratings reference an unknown movie")
        positive = np.flatnonzero(value >= 4)
        popularity = np.bincount(np.searchsorted(active, movie[positive]), minlength=len(catalog)).astype(
            np.float64
        )
        if parameters.max_user_positives and len(positive):
            # Sort only user groups when needed; all real DiskRatings views preserve source user ordering.
            if (np.diff(user[positive]) < 0).any():
                positive = positive[np.argsort(user[positive], kind="stable")]
            boundaries = np.r_[0, np.flatnonzero(np.diff(user[positive])) + 1, len(positive)]
            retained = []
            for first, last in zip(boundaries[:-1], boundaries[1:], strict=True):
                group = positive[first:last]
                if len(group) > parameters.max_user_positives:
                    order = np.lexsort((movie[group], -timestamp[group]))[: parameters.max_user_positives]
                    group = group[order]
                retained.append(group)
            positive = np.concatenate(retained)
            del retained, boundaries
        matrix = sparse.csr_matrix(
            (
                np.ones(len(positive), dtype=np.float32),
                (user[positive], np.searchsorted(active, movie[positive]).astype(np.int32)),
            ),
            shape=(int(user.max()) + 1, len(catalog)),
        )
        del user, movie, value, timestamp, positive
        norms = np.sqrt(np.asarray(matrix.sum(axis=0)).ravel())
        transpose = matrix.T.tocsr()
        n = len(catalog)
        data = np.empty(n * parameters.neighbors, dtype=np.float32)
        indices = np.empty(n * parameters.neighbors, dtype=np.int32)
        indptr = np.zeros(n + 1, dtype=np.int32)
        offset = 0
        for first in range(0, n, block_size):
            last = min(first + block_size, n)
            block = (transpose[first:last] @ matrix).tocsr()
            for local, index in enumerate(range(first, last)):
                start, end = block.indptr[local : local + 2]
                other, support = block.indices[start:end], block.data[start:end]
                valid = (other != index) & (support >= parameters.min_support)
                other, support = other[valid], support[valid]
                weights = support / (norms[index] * norms[other]) * support / (support + parameters.shrinkage)
                if len(weights) > parameters.neighbors:
                    cutoff = np.partition(weights, len(weights) - parameters.neighbors)[-parameters.neighbors]
                    keep = weights >= cutoff
                    other, weights = other[keep], weights[keep]
                order = np.lexsort((other, -weights))[: parameters.neighbors]
                count = len(order)
                data[offset : offset + count] = weights[order]
                indices[offset : offset + count] = other[order]
                offset += count
                indptr[index + 1] = offset
            if progress is not None and (first % 4096 == 0 or last == n):
                progress(f"Neighbors: {last:,}/{n:,} movies; {offset:,} retained edges")
        similarity = sparse.csr_matrix((data[:offset].copy(), indices[:offset].copy(), indptr), shape=(n, n))
        return cls(catalog, similarity, popularity, parameters)

    def recommend(
        self, ratings: Mapping[int, float], *, limit: int = 10, strategy: str = "item_knn"
    ) -> tuple[list[Recommendation], str]:
        """Input keys are TMDB movie IDs of a new profile, never MovieLens user IDs."""
        if strategy not in ("item_knn", "popularity") or not 1 <= limit <= 100:
            raise ValueError("Invalid recommendation request")
        if any(not np.isfinite(value) or not 0.5 <= value <= 5 for value in ratings.values()):
            raise ValueError("Invalid profile rating")
        known = {self.by_tmdb[key]: value for key, value in ratings.items() if key in self.by_tmdb}
        scores = np.zeros(len(self.films))
        available = np.ones(len(self.films), dtype=bool)
        available[list(known)] = False
        personalized = strategy == "item_knn" and any(value >= 4 for value in known.values())
        if personalized:
            # Neutral is 3. Low ratings reduce ranks; every already rated movie is excluded.
            indices = np.fromiter(known, dtype=np.int32)
            weights = np.array([known[index] - 3 for index in indices], dtype=np.float64)
            profile = sparse.csr_matrix(
                (weights, (np.zeros(len(indices), dtype=np.int32), indices)), shape=(1, len(self.films))
            )
            scores = (profile @ self.similarities).toarray().ravel()
            personalized = bool(np.any(scores[available] > 0))
        if not personalized:
            scores = self.popularity.copy()
        candidates = np.flatnonzero(available)
        # Zero-score items provide a deterministic popularity tail when neighbors are sparse.
        order = np.lexsort((candidates, -self.popularity[candidates], -scores[candidates]))[:limit]
        selected = candidates[order]
        return [
            Recommendation(
                self.films[i].movie_id, self.films[i].tmdb_id, self.films[i].title, float(scores[i])
            )
            for i in selected
        ], ("item_knn" if personalized else "popularity")

    def explore(
        self, ratings: Mapping[int, float], *, limit: int, seed: int
    ) -> list[tuple[Recommendation, bool]]:
        if not 1 <= limit <= 100 or not 0 <= seed <= 2**32 - 1:
            raise ValueError("Invalid exploration request")
        if any(not np.isfinite(value) or not 0.5 <= value <= 5 for value in ratings.values()):
            raise ValueError("Invalid profile rating")
        known = {self.by_tmdb[key]: value for key, value in ratings.items() if key in self.by_tmdb}
        scores = np.zeros(len(self.films))
        if known:
            indices = np.fromiter(known, dtype=np.int32)
            weights = np.array([known[index] - 3 for index in indices], dtype=np.float64)
            profile = sparse.csr_matrix(
                (weights, (np.zeros(len(indices), dtype=np.int32), indices)), shape=(1, len(self.films))
            )
            scores = (profile @ self.similarities).toarray().ravel()
        available = np.ones(len(self.films), dtype=bool)
        available[list(known)] = False
        # Negative affinity is not used for exploration; all zero-affinity catalog items remain eligible.
        available &= scores >= 0
        related = np.flatnonzero(available & (scores > 0))
        rng = np.random.default_rng(seed)
        selected: list[tuple[int, bool]] = []
        number = min(len(related), max(1, int(limit * 0.7)))
        if number:
            weights = (1 + scores[related] / scores[related].max()) / np.sqrt(
                1 + np.log1p(self.popularity[related])
            )
            picks = rng.choice(related, number, replace=False, p=weights / weights.sum())
            selected.extend((int(index), False) for index in picks)
            available[picks] = False
        broad = np.flatnonzero(available)
        number = min(limit - len(selected), len(broad))
        if number:
            # Full eligible artifact catalog, not just the old popularity top-N.
            weights = 1 / np.sqrt(1 + np.log1p(self.popularity[broad]))
            picks = rng.choice(broad, number, replace=False, p=weights / weights.sum())
            selected.extend((int(index), True) for index in picks)
        return [
            (
                Recommendation(
                    self.films[index].movie_id,
                    self.films[index].tmdb_id,
                    self.films[index].title,
                    float(scores[index]),
                ),
                broad,
            )
            for index, broad in selected
        ]

    def save(self, directory: Path, metadata: dict) -> dict:
        directory.mkdir(parents=True, exist_ok=False)
        sparse.save_npz(directory / "similarities.npz", self.similarities)
        np.save(directory / "popularity.npy", self.popularity, allow_pickle=False)
        write_json(directory / "movies.json", [asdict(film) for film in self.films])
        manifest = {
            **metadata,
            "format_version": 1,
            "parameters": asdict(self.parameters),
            "movies": len(self.films),
            "edges": self.similarities.nnz,
            "files": {
                name: digest(directory / name)
                for name in ["movies.json", "similarities.npz", "popularity.npy"]
            },
        }
        write_json(directory / "manifest.json", manifest)
        return manifest

    @classmethod
    def load(cls, directory: Path) -> tuple["ItemModel", dict]:
        import json

        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        if manifest["format_version"] != 1:
            raise ValueError("Unsupported model format")
        for name in ("movies.json", "similarities.npz", "popularity.npy"):
            if digest(directory / name) != manifest["files"][name]:
                raise ValueError(f"Model checksum mismatch: {name}")
        films = [Film(**row) for row in json.loads((directory / "movies.json").read_text(encoding="utf-8"))]
        model = cls(
            films,
            sparse.load_npz(directory / "similarities.npz").tocsr(),
            np.load(directory / "popularity.npy", allow_pickle=False, mmap_mode="r"),
            Parameters(**manifest["parameters"]),
        )
        return model, manifest
