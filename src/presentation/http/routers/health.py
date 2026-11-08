from dishka.integrations.fastapi import DishkaRoute, FromDishka
from fastapi import APIRouter, Response, status

from application.query.check_readiness import CheckReadinessHandler
from presentation.http.dto.health import HealthResponseDTO, ReadinessResponseDTO

router = APIRouter(tags=["operations"], route_class=DishkaRoute)


@router.get("/health", response_model=HealthResponseDTO)
async def health() -> HealthResponseDTO:
    return HealthResponseDTO()


@router.get(
    "/ready",
    response_model=ReadinessResponseDTO,
    responses={503: {"model": ReadinessResponseDTO, "description": "Database or schema is not ready"}},
)
async def ready(response: Response, handler: FromDishka[CheckReadinessHandler]) -> ReadinessResponseDTO:
    result = await handler()
    if not result.database_ready:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return ReadinessResponseDTO(
        status="ready" if result.database_ready else "not_ready",
        model=result.model,
        database="ready" if result.database_ready else "unavailable",
    )
