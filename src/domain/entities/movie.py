import math
from dataclasses import dataclass
from datetime import date, datetime
from uuid import UUID

from domain.entities.genre import Genre
from domain.exceptions.movie import InvalidMovieError
from domain.value_objects.media_type import MediaType


@dataclass(frozen=True, slots=True)
class Movie:
    id: UUID
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
    updated_at: datetime
    movielens_id: int | None = None

    def __post_init__(self) -> None:
        if self.tmdb_id <= 0 or not self.title.strip():
            raise InvalidMovieError("A movie needs a positive TMDB ID and a title")
        if not isinstance(self.media_type, MediaType):
            raise InvalidMovieError("Invalid media type")
        if not math.isfinite(self.vote_average) or not 0 <= self.vote_average <= 10:
            raise InvalidMovieError("Invalid TMDB rating")
        if self.vote_count < 0 or not math.isfinite(self.popularity) or self.popularity < 0:
            raise InvalidMovieError("Invalid popularity or vote count")
        if any(genre.tmdb_id <= 0 or not genre.name.strip() for genre in self.genres):
            raise InvalidMovieError("Invalid genre")
        if self.movielens_id is not None and (self.movielens_id <= 0 or self.media_type != MediaType.MOVIE):
            raise InvalidMovieError("MovieLens mapping is only available for movies")
