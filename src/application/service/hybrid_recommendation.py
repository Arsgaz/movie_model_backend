import asyncio
import math
from dataclasses import dataclass, replace

from port.content_recommendation import ContentRecommendationProvider, MoviePreference
from port.exceptions.movie_metadata import (
    MovieMetadataNotConfiguredError,
    MovieMetadataNotFoundError,
    MovieMetadataUnavailableError,
)
from port.exceptions.recommendation import RecommendationUnavailableError
from port.recommendation import RankedMovie, RankedRecommendations, RecommendationEngine, SourceStatus


@dataclass(frozen=True, slots=True)
class RecommendationPolicy:
    candidate_limit: int = 40
    source_timeout: float = 10
    collaborative_weight: float = 1.0
    content_weight: float = 1.0
    rank_constant: int = 60

    def __post_init__(self) -> None:
        if not 20 <= self.candidate_limit <= 100 or self.rank_constant < 1:
            raise ValueError("Invalid recommendation budget")
        if any(
            not math.isfinite(value) or value <= 0
            for value in (self.source_timeout, self.collaborative_weight, self.content_weight)
        ):
            raise ValueError("Recommendation deadlines and weights must be positive and finite")


SOURCE_ERRORS = (
    RecommendationUnavailableError,
    MovieMetadataNotConfiguredError,
    MovieMetadataNotFoundError,
    MovieMetadataUnavailableError,
    TimeoutError,
)


class HybridRecommender:
    def __init__(
        self,
        engine: RecommendationEngine,
        content: ContentRecommendationProvider,
        policy: RecommendationPolicy,
    ) -> None:
        self._engine, self._content, self._policy = engine, content, policy

    async def recommend(self, preferences: tuple[MoviePreference, ...], limit: int) -> RankedRecommendations:
        if not 1 <= limit <= 50:
            raise ValueError("Invalid recommendation limit")
        ratings = {entry.movie.tmdb_id: entry.value for entry in preferences}
        excluded = frozenset(ratings)
        budget = max(self._policy.candidate_limit, min(100, limit * 2))

        async def collaborative():
            async with asyncio.timeout(self._policy.source_timeout):
                status = await self._engine.status()
                if status != "ready":
                    return status, None
                return status, await self._engine.recommend(ratings, budget)

        async def content():
            async with asyncio.timeout(self._policy.source_timeout):
                return await self._content.recommend(preferences, budget)

        collab, textual = await asyncio.gather(collaborative(), content(), return_exceptions=True)
        for result in (collab, textual):
            if isinstance(result, BaseException) and not isinstance(result, SOURCE_ERRORS):
                raise result
        statuses: list[SourceStatus] = []
        lists: list[tuple[str, float, tuple[RankedMovie, ...]]] = []
        model_version, known_ratings, fallback = None, 0, None
        baseline: tuple[RankedMovie, ...] | None = None
        if isinstance(collab, BaseException):
            statuses.append(SourceStatus("collaborative", "unavailable"))
            fallback = "model_unavailable"
        else:
            status, result = collab
            statuses.append(SourceStatus("collaborative", status))
            if result is not None:
                model_version, known_ratings = result.model_version, result.known_ratings
                if result.strategy == "item_knn":
                    lists.append(("collaborative", self._policy.collaborative_weight, result.items))
                else:
                    baseline = result.items
                    fallback = result.fallback_reason
            else:
                fallback = "model_not_configured" if status == "not_configured" else "model_unavailable"
        if isinstance(textual, BaseException):
            statuses.append(SourceStatus("content", "unavailable"))
        else:
            statuses.append(SourceStatus("content", "partial" if textual.partial else "ready"))
            if textual.items:
                lists.append(("content", self._policy.content_weight, textual.items))

        # Fuse ranks rather than incomparable raw model, lexical and popularity scores.
        scores: dict[int, float] = {}
        movies: dict[int, RankedMovie] = {}
        origins: dict[int, list[str]] = {}
        active_sources = set()
        for name, weight, candidates in lists:
            distinct = set()
            for candidate in candidates[:budget]:
                identity = candidate.tmdb_id
                if identity in excluded or identity in distinct or candidate.score <= 0:
                    continue
                distinct.add(identity)
                active_sources.add(name)
                scores[identity] = scores.get(identity, 0) + weight / (
                    self._policy.rank_constant + len(distinct)
                )
                if identity not in movies or candidate.metadata is not None:
                    movies[identity] = candidate
                origins.setdefault(identity, []).append(name)
        ordered = sorted(scores, key=lambda identity: (-scores[identity], identity))[:limit]
        items = [
            replace(movies[identity], score=scores[identity], sources=tuple(origins[identity]))
            for identity in ordered
        ]

        if len(items) < limit:
            try:
                async with asyncio.timeout(self._policy.source_timeout):
                    popular = await self._content.popular(excluded, budget)
                baseline = popular.items
                statuses.append(SourceStatus("popularity", "partial" if popular.partial else "ready"))
            except SOURCE_ERRORS:
                # A legacy model's popular list is only an offline backup for fresh discovery.
                statuses.append(
                    SourceStatus("popularity", "partial" if baseline is not None else "unavailable")
                )
            selected = excluded | {item.tmdb_id for item in items}
            for candidate in (baseline or ())[:budget]:
                if candidate.tmdb_id in selected:
                    continue
                # The fallback tail follows personalized results, regardless of raw score scale.
                items.append(replace(candidate, score=0, sources=("popularity",)))
                selected = selected | {candidate.tmdb_id}
                if len(items) == limit:
                    break
        if (
            not items
            and baseline is None
            and (isinstance(textual, BaseException) or not any(entry.value >= 4 for entry in preferences))
        ):
            raise RecommendationUnavailableError
        if not items and all(status.status in ("unavailable", "not_configured") for status in statuses):
            raise RecommendationUnavailableError
        strategy = (
            "hybrid"
            if len(active_sources) > 1
            else "item_knn"
            if "collaborative" in active_sources
            else ("content" if "content" in active_sources else "popularity")
        )
        return RankedRecommendations(
            tuple(items),
            strategy,
            model_version,
            known_ratings,
            None if active_sources else fallback or "no_positive_candidates",
            tuple(statuses),
            any(status.status in ("unavailable", "partial") for status in statuses),
        )
