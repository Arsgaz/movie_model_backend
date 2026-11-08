from dataclasses import dataclass

from application.dto.movie import MovieDTO
from domain.value_objects.media_type import MediaType


@dataclass(frozen=True, slots=True)
class ImportCatalogPageCommand:
    media_type: MediaType = MediaType.MOVIE
    query: str = ""
    genre_id: int | None = None
    page: int = 1

    def __post_init__(self) -> None:
        if not 1 <= self.page <= 500 or len(self.query) > 150:
            raise ValueError("Invalid TMDB pagination or search")
        if self.genre_id is not None and self.genre_id <= 0:
            raise ValueError("Invalid genre ID")


@dataclass(frozen=True, slots=True)
class ImportMovieCommand:
    media_type: MediaType
    tmdb_id: int

    def __post_init__(self) -> None:
        if self.tmdb_id <= 0:
            raise ValueError("Invalid TMDB ID")


@dataclass(frozen=True, slots=True)
class CatalogImportDTO:
    items: tuple[MovieDTO, ...]
    page: int
    imported_count: int
    source_total_results: int
    source_total_pages: int
