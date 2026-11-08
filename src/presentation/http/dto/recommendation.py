from typing import Literal

from pydantic import BaseModel, ConfigDict

from presentation.http.dto.discovery import MoviePreviewResponseDTO


class RecommendationItemResponseDTO(BaseModel):
    model_config = ConfigDict(from_attributes=True, allow_inf_nan=False)
    movie: MoviePreviewResponseDTO
    score: float
    sources: list[Literal["collaborative", "content", "popularity"]]
    exploratory: bool


class SourceStatusResponseDTO(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    source: Literal["collaborative", "content", "popularity"]
    status: Literal["not_configured", "ready", "unavailable", "partial"]


class RecommendationsResponseDTO(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    items: list[RecommendationItemResponseDTO]
    strategy: Literal["item_knn", "popularity", "content", "hybrid", "exploration"]
    model_version: str | None
    known_ratings: int
    movie_ratings: int
    fallback_reason: (
        Literal[
            "baseline_selected",
            "no_known_positive_ratings",
            "no_positive_candidates",
            "model_not_configured",
            "model_unavailable",
        ]
        | None
    )
    metadata_partial: bool
    sources: list[SourceStatusResponseDTO]
    degraded: bool
    mode: Literal["personalized", "surprise"]
    seed: int | None
