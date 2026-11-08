from datetime import date
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, field_validator

PositiveID = Annotated[int, Field(strict=True, gt=0)]
ImagePath = Annotated[str, StringConstraints(pattern=r"^/[A-Za-z0-9_./-]+$", max_length=255)]


class TMDBGenreDTO(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: PositiveID
    name: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=120)]


class TMDBGenreListDTO(BaseModel):
    genres: list[TMDBGenreDTO]


class TMDBMovieDTO(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: PositiveID
    title: str | None = None
    name: str | None = None
    original_title: str | None = None
    original_name: str | None = None
    overview: str | None = None
    poster_path: ImagePath | None = None
    backdrop_path: ImagePath | None = None
    release_date: date | None = None
    first_air_date: date | None = None
    vote_average: float = Field(default=0, ge=0, le=10, allow_inf_nan=False)
    vote_count: int = Field(default=0, ge=0)
    popularity: float = Field(default=0, ge=0, allow_inf_nan=False)
    genre_ids: list[PositiveID] | None = None
    genres: list[TMDBGenreDTO] | None = None

    @field_validator("release_date", "first_air_date", mode="before")
    @classmethod
    def empty_date(cls, value: object) -> object:
        return None if value == "" else value


class TMDBPageDTO(BaseModel):
    page: int = Field(ge=1)
    total_pages: int = Field(ge=0)
    total_results: int = Field(ge=0)
    results: list[TMDBMovieDTO] = Field(max_length=100)
