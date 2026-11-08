from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from uuid import UUID

from application.dto.movie import MovieDTO
from domain.value_objects.media_type import MediaType


@dataclass(frozen=True, slots=True)
class MovieReference:
    movie_id: UUID | None = None
    media_type: MediaType | None = None
    tmdb_id: int | None = None

    def __post_init__(self) -> None:
        if self.movie_id is not None:
            if self.media_type is not None or self.tmdb_id is not None:
                raise ValueError("Choose one movie identity")
        elif self.media_type is None or self.tmdb_id is None or self.tmdb_id <= 0:
            raise ValueError("Provide a valid external movie identity")


class LibraryOperation(StrEnum):
    FAVORITE = "favorite"
    UNFAVORITE = "unfavorite"
    RATE = "rate"
    UNRATE = "unrate"


@dataclass(frozen=True, slots=True)
class ChangeLibraryCommand:
    profile_id: UUID
    operation: LibraryOperation
    movie: MovieReference
    value: int | None = None


@dataclass(frozen=True, slots=True)
class FavoriteDTO:
    movie: MovieDTO
    created_at: datetime


@dataclass(frozen=True, slots=True)
class RatingDTO:
    movie: MovieDTO
    value: int
    updated_at: datetime


@dataclass(frozen=True, slots=True)
class LibraryDTO:
    favorites: tuple[FavoriteDTO, ...]
    ratings: tuple[RatingDTO, ...]


@dataclass(frozen=True, slots=True)
class GetLibraryQuery:
    profile_id: UUID


@dataclass(frozen=True, slots=True)
class LegacyEntry:
    media_type: MediaType
    tmdb_id: int
    favorite: bool = False
    rating: int | None = None
    rated_at: datetime | None = None


@dataclass(frozen=True, slots=True)
class ImportLibraryCommand:
    profile_id: UUID
    entries: tuple[LegacyEntry, ...]

    def __post_init__(self) -> None:
        if not 1 <= len(self.entries) <= 20:
            raise ValueError("Import 1 to 20 records per batch")
