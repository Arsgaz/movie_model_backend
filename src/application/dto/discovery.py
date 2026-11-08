from dataclasses import dataclass

from domain.value_objects.media_type import MediaType
from port.movie_metadata import MovieMetadata


@dataclass(frozen=True, slots=True)
class DiscoverMoviesQuery:
    media_type: MediaType = MediaType.MOVIE
    query: str = ""
    genre_id: int | None = None
    page: int = 1

    def __post_init__(self) -> None:
        if not 1 <= self.page <= 500 or len(self.query) > 150:
            raise ValueError("Invalid discovery search or page")
        if self.genre_id is not None and self.genre_id <= 0:
            raise ValueError("Invalid genre ID")


@dataclass(frozen=True, slots=True)
class DiscoverMovieQuery:
    media_type: MediaType
    tmdb_id: int

    def __post_init__(self) -> None:
        if self.tmdb_id <= 0:
            raise ValueError("Invalid TMDB ID")


@dataclass(frozen=True, slots=True)
class DiscoveryPageDTO:
    items: tuple[MovieMetadata, ...]
    page: int
    total_pages: int
    total_results: int | None
