from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from domain.value_objects.media_type import MediaType
from presentation.http.dto.movie import MovieResponseDTO


class ImportCatalogRequestDTO(BaseModel):
    model_config = ConfigDict(extra="forbid")
    media_type: MediaType = MediaType.MOVIE
    q: Annotated[str, StringConstraints(strip_whitespace=True, max_length=150)] = ""
    genre_id: int | None = Field(default=None, gt=0)
    page: int = Field(default=1, ge=1, le=500)


class CatalogImportResponseDTO(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    items: list[MovieResponseDTO]
    page: int
    imported_count: int
    source_total_results: int
    source_total_pages: int
