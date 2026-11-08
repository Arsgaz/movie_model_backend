from dataclasses import dataclass
from datetime import date, datetime
from uuid import UUID

from domain.entities.movie import Movie
from domain.value_objects.media_type import MediaType


@dataclass(frozen=True, slots=True)
class GenreDTO:
    tmdb_id: int
    name: str


@dataclass(frozen=True, slots=True)
class MovieDTO:
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
    genres: tuple[GenreDTO, ...]
    updated_at: datetime
    movielens_id: int | None

    @classmethod
    def from_entity(cls, movie: Movie) -> "MovieDTO":
        return cls(
            id=movie.id,
            tmdb_id=movie.tmdb_id,
            media_type=movie.media_type,
            title=movie.title,
            original_title=movie.original_title,
            overview=movie.overview,
            poster_path=movie.poster_path,
            backdrop_path=movie.backdrop_path,
            release_date=movie.release_date,
            vote_average=movie.vote_average,
            vote_count=movie.vote_count,
            popularity=movie.popularity,
            genres=tuple(GenreDTO(tmdb_id=genre.tmdb_id, name=genre.name) for genre in movie.genres),
            updated_at=movie.updated_at,
            movielens_id=movie.movielens_id,
        )


@dataclass(frozen=True, slots=True)
class ListMoviesQuery:
    query: str = ""
    media_type: MediaType = MediaType.MOVIE
    genre_id: int | None = None
    page: int = 1
    page_size: int = 20

    def __post_init__(self) -> None:
        if self.page < 1 or not 1 <= self.page_size <= 100 or len(self.query) > 150:
            raise ValueError("Invalid catalog pagination or search")
        if self.genre_id is not None and self.genre_id <= 0:
            raise ValueError("Invalid genre ID")


@dataclass(frozen=True, slots=True)
class GetMovieQuery:
    movie_id: UUID


@dataclass(frozen=True, slots=True)
class MovieListDTO:
    items: tuple[MovieDTO, ...]
    page: int
    page_size: int
    total_results: int
    total_pages: int


@dataclass(frozen=True, slots=True)
class GetMovieByTMDBQuery:
    media_type: MediaType
    tmdb_id: int
