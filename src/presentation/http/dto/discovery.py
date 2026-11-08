from datetime import date

from pydantic import BaseModel, ConfigDict

from domain.value_objects.media_type import MediaType
from presentation.http.dto.movie import GenreResponseDTO


class MoviePreviewResponseDTO(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: None = None
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


class DiscoveryPageResponseDTO(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    items: list[MoviePreviewResponseDTO]
    page: int
    total_pages: int
    total_results: int | None
