from dataclasses import dataclass
from typing import Protocol

from port.movie_metadata import MovieMetadata
from port.recommendation import RankedMovie


@dataclass(frozen=True, slots=True)
class MoviePreference:
    movie: MovieMetadata
    value: int


@dataclass(frozen=True, slots=True)
class ContentCandidates:
    items: tuple[RankedMovie, ...]
    partial: bool = False


class ContentRecommendationProvider(Protocol):
    async def recommend(self, preferences: tuple[MoviePreference, ...], limit: int) -> ContentCandidates: ...

    async def popular(self, excluded: frozenset[int], limit: int) -> ContentCandidates: ...

    async def explore(
        self, preferences: tuple[MoviePreference, ...], limit: int, seed: int, year: int
    ) -> ContentCandidates: ...
