from dataclasses import dataclass
from datetime import date
from typing import Protocol

from domain.entities.genre import Genre
from domain.value_objects.media_type import MediaType


@dataclass(frozen=True, slots=True)
class MovieMetadata:
    tmdb_id: int
    media_type: MediaType
    title: str
    original_title: str
    overview: str
    poster_path: str | None
    backdrop_path: str | None
    release_date: date | None
    vote_average: float
    vote_count: int
    popularity: float
    genres: tuple[Genre, ...]


@dataclass(frozen=True, slots=True)
class MetadataPage:
    items: tuple[MovieMetadata, ...]
    page: int
    total_pages: int
    total_results: int


class MovieMetadataProvider(Protocol):
    async def get_page(
        self,
        *,
        media_type: MediaType,
        query: str,
        genre_id: int | None,
        page: int,
    ) -> MetadataPage: ...

    async def get_movie(self, *, media_type: MediaType, tmdb_id: int) -> MovieMetadata: ...

    async def get_exploration_page(
        self,
        *,
        genre_id: int | None,
        year: int,
        page: int,
    ) -> MetadataPage: ...
