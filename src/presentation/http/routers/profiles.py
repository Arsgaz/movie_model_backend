from typing import Annotated
from uuid import UUID

from dishka.integrations.fastapi import DishkaRoute, FromDishka
from fastapi import APIRouter, Query, Response, status

from application.command.create_profile import CreateProfileHandler
from application.dto.profile import CreateProfileCommand, GetProfileQuery, ListProfilesQuery
from application.query.get_profile import GetProfileHandler
from application.query.list_profiles import ListProfilesHandler
from presentation.http.dto.error import ErrorResponseDTO
from presentation.http.dto.profile import CreateProfileRequestDTO, ProfileListResponseDTO, ProfileResponseDTO

router = APIRouter(
    prefix="/api/v1/profiles",
    tags=["profiles"],
    route_class=DishkaRoute,
    responses={503: {"model": ErrorResponseDTO, "description": "Database is unavailable"}},
)


@router.post("", status_code=status.HTTP_201_CREATED, response_model=ProfileResponseDTO)
async def create_profile(
    body: CreateProfileRequestDTO,
    response: Response,
    handler: FromDishka[CreateProfileHandler],
) -> ProfileResponseDTO:
    result = await handler(CreateProfileCommand(name=body.name))
    response.headers["Location"] = f"/api/v1/profiles/{result.id}"
    return ProfileResponseDTO.model_validate(result)


@router.get("", response_model=ProfileListResponseDTO)
async def list_profiles(
    handler: FromDishka[ListProfilesHandler],
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> ProfileListResponseDTO:
    result = await handler(ListProfilesQuery(limit=limit, offset=offset))
    return ProfileListResponseDTO.model_validate(result)


@router.get(
    "/{profile_id}",
    response_model=ProfileResponseDTO,
    responses={404: {"model": ErrorResponseDTO, "description": "Profile not found"}},
)
async def get_profile(profile_id: UUID, handler: FromDishka[GetProfileHandler]) -> ProfileResponseDTO:
    return ProfileResponseDTO.model_validate(await handler(GetProfileQuery(profile_id=profile_id)))
