import asyncio
import math
import random
from dataclasses import replace

from application.service.hybrid_recommendation import SOURCE_ERRORS, RecommendationPolicy
from port.clock import Clock
from port.content_recommendation import ContentRecommendationProvider, MoviePreference
from port.exceptions.recommendation import RecommendationUnavailableError
from port.random_seed import RandomSeedGenerator
from port.recommendation import RankedRecommendations, RecommendationEngine, SourceStatus


class SurpriseRecommender:
    def __init__(
        self,
        engine: RecommendationEngine,
        content: ContentRecommendationProvider,
        policy: RecommendationPolicy,
        clock: Clock,
        seeds: RandomSeedGenerator,
    ) -> None:
        self._engine, self._content, self._policy = engine, content, policy
        self._clock, self._seeds = clock, seeds

    async def recommend(
        self, preferences: tuple[MoviePreference, ...], limit: int, seed: int | None = None
    ) -> RankedRecommendations:
        if not 1 <= limit <= 20 or (seed is not None and not 0 <= seed < 2**32):
            raise ValueError("Invalid exploration request")
        seed = self._seeds.new() if seed is None else seed
        ratings = {entry.movie.tmdb_id: entry.value for entry in preferences}

        async def collaborative():
            async with asyncio.timeout(self._policy.source_timeout):
                status = await self._engine.status()
                result = (
                    await self._engine.explore(ratings, self._policy.candidate_limit, seed)
                    if status == "ready"
                    else None
                )
                return status, result

        async def content():
            async with asyncio.timeout(self._policy.source_timeout):
                return await self._content.explore(
                    preferences, self._policy.candidate_limit, seed, self._clock.now().year
                )

        collab, textual = await asyncio.gather(collaborative(), content(), return_exceptions=True)
        for result in (collab, textual):
            if isinstance(result, BaseException) and not isinstance(result, SOURCE_ERRORS):
                raise result
        statuses, candidates = [], {}
        version, known = None, 0
        if isinstance(collab, BaseException):
            statuses.append(SourceStatus("collaborative", "unavailable"))
        else:
            status, result = collab
            statuses.append(SourceStatus("collaborative", status))
            if result is not None:
                version, known = result.model_version, result.known_ratings
                for item in result.items:
                    if item.tmdb_id not in ratings:
                        candidates[item.tmdb_id] = replace(item, sources=("collaborative",))
        if isinstance(textual, BaseException):
            statuses.append(SourceStatus("content", "unavailable"))
        else:
            statuses.append(SourceStatus("content", "partial" if textual.partial else "ready"))
            for item in textual.items:
                if item.tmdb_id in ratings:
                    continue
                previous = candidates.get(item.tmdb_id)
                candidates[item.tmdb_id] = replace(
                    item,
                    sources=("collaborative", "content") if previous else ("content",),
                    exploratory=item.exploratory and (previous.exploratory if previous else True),
                )
        if not candidates and all(status.status in ("not_configured", "unavailable") for status in statuses):
            raise RecommendationUnavailableError
        rng = random.Random(seed)

        def sample(pool, number):
            # Weighted sampling without replacement. Scores from different sources are not compared.
            ordered = sorted(pool, key=lambda item: item.tmdb_id)
            weighted = []
            for item in ordered:
                popularity = max(0, item.metadata.popularity) if item.metadata else 0
                weight = 1 / math.sqrt(1 + math.log1p(popularity))
                weighted.append((-math.log(max(rng.random(), 1e-15)) / weight, item))
            return [item for _, item in sorted(weighted, key=lambda pair: pair[0])[:number]]

        related = [item for item in candidates.values() if not item.exploratory]
        broad = [item for item in candidates.values() if item.exploratory]
        items = sample(related, max(1, int(limit * 0.7))) + sample(broad, limit - max(1, int(limit * 0.7)))
        selected = {item.tmdb_id for item in items}
        items += sample(
            [item for item in candidates.values() if item.tmdb_id not in selected], limit - len(items)
        )
        rng.shuffle(items)
        active = {source for item in items if not item.exploratory for source in item.sources}
        strategy = (
            "hybrid"
            if len(active) > 1
            else "item_knn"
            if "collaborative" in active
            else "content"
            if active
            else "exploration"
        )
        return RankedRecommendations(
            tuple(items),
            strategy,
            version,
            known,
            None if active else "no_positive_candidates",
            tuple(statuses),
            any(status.status in ("partial", "unavailable") for status in statuses),
            "surprise",
            seed,
        )
