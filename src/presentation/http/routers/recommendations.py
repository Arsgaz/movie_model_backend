from typing import Annotated, Literal
from uuid import UUID

from dishka.integrations.fastapi import DishkaRoute, FromDishka
from fastapi import APIRouter, HTTPException, Query

from application.dto.recommendation import GetRecommendationsQuery
from application.query.get_recommendations import GetRecommendationsHandler
from presentation.http.dto.recommendation import RecommendationsResponseDTO

router = APIRouter(prefix="/api/v1/profiles", tags=["recommendations"], route_class=DishkaRoute)


@router.get("/{profile_id}/recommendations", response_model=RecommendationsResponseDTO)
async def recommendations(
    profile_id: UUID,
    handler: FromDishka[GetRecommendationsHandler],
    limit: Annotated[int, Query(ge=1, le=50)] = 10,
    mode: Literal["personalized", "surprise"] = "personalized",
    seed: Annotated[int | None, Query(ge=0, le=2**32 - 1)] = None,
) -> RecommendationsResponseDTO:
    if mode == "surprise" and limit > 20:
        raise HTTPException(status_code=422, detail="Surprise limit must be between 1 and 20")
    return RecommendationsResponseDTO.model_validate(
        await handler(GetRecommendationsQuery(profile_id, limit, mode, seed))
    )
