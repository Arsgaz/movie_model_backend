from datetime import date, datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict

from domain.value_objects.media_type import MediaType


class GenreResponseDTO(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    tmdb_id: int
    name: str


class MovieResponseDTO(BaseModel):
    model_config = ConfigDict(from_attributes=True)
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
    genres: list[GenreResponseDTO]
    updated_at: datetime
    movielens_id: int | None


class MovieListResponseDTO(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    items: list[MovieResponseDTO]
    page: int
    page_size: int
    total_results: int
    total_pages: int
