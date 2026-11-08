from datetime import datetime
from typing import Annotated, Self

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, model_validator

from domain.value_objects.media_type import MediaType
from presentation.http.dto.movie import MovieResponseDTO

RatingValue = Annotated[int, Field(strict=True, ge=1, le=5)]


class ExternalMovieRequestDTO(BaseModel):
    model_config = ConfigDict(extra="forbid")
    media_type: MediaType
    tmdb_id: int = Field(gt=0, strict=True)


class RatingRequestDTO(ExternalMovieRequestDTO):
    value: RatingValue


class RatingValueRequestDTO(BaseModel):
    model_config = ConfigDict(extra="forbid")
    value: RatingValue


class FavoriteResponseDTO(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    movie: MovieResponseDTO
    created_at: datetime


class RatingResponseDTO(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    movie: MovieResponseDTO
    value: int
    updated_at: datetime


class LibraryResponseDTO(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    favorites: list[FavoriteResponseDTO]
    ratings: list[RatingResponseDTO]


class LegacyEntryRequestDTO(ExternalMovieRequestDTO):
    favorite: bool = False
    rating: RatingValue | None = None
    rated_at: AwareDatetime | None = None

    @model_validator(mode="after")
    def require_action(self) -> Self:
        if not self.favorite and self.rating is None:
            raise ValueError("Provide favorite or rating")
        return self


class ImportLibraryRequestDTO(BaseModel):
    model_config = ConfigDict(extra="forbid")
    entries: list[LegacyEntryRequestDTO] = Field(min_length=1, max_length=20)


class ImportLibraryResponseDTO(BaseModel):
    processed: int
