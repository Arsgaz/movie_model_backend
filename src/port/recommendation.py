from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal, Protocol

from port.movie_metadata import MovieMetadata

ModelStatus = Literal["not_configured", "ready", "unavailable"]


@dataclass(frozen=True, slots=True)
class RankedMovie:
    tmdb_id: int
    title: str
    animation: bool
    score: float
    metadata: MovieMetadata | None = None
    sources: tuple[str, ...] = ()
    exploratory: bool = False


@dataclass(frozen=True, slots=True)
class SourceStatus:
    source: Literal["collaborative", "content", "popularity"]
    status: Literal["not_configured", "ready", "unavailable", "partial"]


@dataclass(frozen=True, slots=True)
class RankedRecommendations:
    items: tuple[RankedMovie, ...]
    strategy: Literal["item_knn", "popularity", "content", "hybrid", "exploration"]
    model_version: str | None
    known_ratings: int
    fallback_reason: str | None
    sources: tuple[SourceStatus, ...] = ()
    degraded: bool = False
    mode: Literal["personalized", "surprise"] = "personalized"
    seed: int | None = None


class RecommendationEngine(Protocol):
    async def status(self) -> ModelStatus: ...

    async def recommend(self, ratings: Mapping[int, float], limit: int) -> RankedRecommendations: ...

    async def explore(self, ratings: Mapping[int, float], limit: int, seed: int) -> RankedRecommendations: ...
