from dataclasses import dataclass
from typing import Literal
from uuid import UUID

from port.movie_metadata import MovieMetadata
from port.recommendation import SourceStatus


@dataclass(frozen=True, slots=True)
class GetRecommendationsQuery:
    profile_id: UUID
    limit: int = 10
    mode: Literal["personalized", "surprise"] = "personalized"
    seed: int | None = None


@dataclass(frozen=True, slots=True)
class RecommendationItemDTO:
    movie: MovieMetadata
    score: float
    sources: tuple[str, ...] = ()
    exploratory: bool = False


@dataclass(frozen=True, slots=True)
class RecommendationsDTO:
    items: tuple[RecommendationItemDTO, ...]
    strategy: str
    model_version: str | None
    known_ratings: int
    movie_ratings: int
    fallback_reason: str | None
    metadata_partial: bool
    sources: tuple[SourceStatus, ...] = ()
    degraded: bool = False
    mode: Literal["personalized", "surprise"] = "personalized"
    seed: int | None = None
