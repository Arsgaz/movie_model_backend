from uuid import UUID

from dishka.integrations.fastapi import DishkaRoute, FromDishka
from fastapi import APIRouter, Response

from application.command.change_library import ChangeLibraryHandler
from application.command.import_library import ImportLibraryHandler
from application.dto.library import (
    ChangeLibraryCommand,
    GetLibraryQuery,
    ImportLibraryCommand,
    LegacyEntry,
    LibraryOperation,
    MovieReference,
)
from application.query.get_library import GetLibraryHandler
from presentation.http.dto.error import ErrorResponseDTO
from presentation.http.dto.library import (
    ExternalMovieRequestDTO,
    FavoriteResponseDTO,
    ImportLibraryRequestDTO,
    ImportLibraryResponseDTO,
    LibraryResponseDTO,
    RatingRequestDTO,
    RatingResponseDTO,
    RatingValueRequestDTO,
)

router = APIRouter(
    prefix="/api/v1/profiles/{profile_id}",
    tags=["library"],
    route_class=DishkaRoute,
    responses={code: {"model": ErrorResponseDTO} for code in (404, 502, 503)},
)


@router.get("/library", response_model=LibraryResponseDTO)
async def get_library(profile_id: UUID, handler: FromDishka[GetLibraryHandler]) -> LibraryResponseDTO:
    return LibraryResponseDTO.model_validate(await handler(GetLibraryQuery(profile_id)))


@router.put("/favorites", response_model=FavoriteResponseDTO)
async def add_external_favorite(
    profile_id: UUID, body: ExternalMovieRequestDTO, handler: FromDishka[ChangeLibraryHandler]
) -> FavoriteResponseDTO:
    result = await handler(
        ChangeLibraryCommand(
            profile_id,
            LibraryOperation.FAVORITE,
            MovieReference(media_type=body.media_type, tmdb_id=body.tmdb_id),
        )
    )
    return FavoriteResponseDTO.model_validate(result)


@router.put("/favorites/{movie_id}", response_model=FavoriteResponseDTO)
async def add_favorite(
    profile_id: UUID, movie_id: UUID, handler: FromDishka[ChangeLibraryHandler]
) -> FavoriteResponseDTO:
    result = await handler(
        ChangeLibraryCommand(profile_id, LibraryOperation.FAVORITE, MovieReference(movie_id=movie_id))
    )
    return FavoriteResponseDTO.model_validate(result)


@router.delete("/favorites/{movie_id}", status_code=204)
async def remove_favorite(
    profile_id: UUID, movie_id: UUID, handler: FromDishka[ChangeLibraryHandler]
) -> Response:
    await handler(
        ChangeLibraryCommand(profile_id, LibraryOperation.UNFAVORITE, MovieReference(movie_id=movie_id))
    )
    return Response(status_code=204)


@router.put("/ratings", response_model=RatingResponseDTO)
async def rate_external_movie(
    profile_id: UUID, body: RatingRequestDTO, handler: FromDishka[ChangeLibraryHandler]
) -> RatingResponseDTO:
    result = await handler(
        ChangeLibraryCommand(
            profile_id,
            LibraryOperation.RATE,
            MovieReference(media_type=body.media_type, tmdb_id=body.tmdb_id),
            body.value,
        )
    )
    return RatingResponseDTO.model_validate(result)


@router.put("/ratings/{movie_id}", response_model=RatingResponseDTO)
async def rate_movie(
    profile_id: UUID, movie_id: UUID, body: RatingValueRequestDTO, handler: FromDishka[ChangeLibraryHandler]
) -> RatingResponseDTO:
    result = await handler(
        ChangeLibraryCommand(profile_id, LibraryOperation.RATE, MovieReference(movie_id=movie_id), body.value)
    )
    return RatingResponseDTO.model_validate(result)


@router.delete("/ratings/{movie_id}", status_code=204)
async def remove_rating(
    profile_id: UUID, movie_id: UUID, handler: FromDishka[ChangeLibraryHandler]
) -> Response:
    await handler(
        ChangeLibraryCommand(profile_id, LibraryOperation.UNRATE, MovieReference(movie_id=movie_id))
    )
    return Response(status_code=204)


@router.post("/library/import", response_model=ImportLibraryResponseDTO)
async def import_library(
    profile_id: UUID, body: ImportLibraryRequestDTO, handler: FromDishka[ImportLibraryHandler]
) -> ImportLibraryResponseDTO:
    processed = await handler(
        ImportLibraryCommand(
            profile_id,
            tuple(
                LegacyEntry(
                    media_type=entry.media_type,
                    tmdb_id=entry.tmdb_id,
                    favorite=entry.favorite,
                    rating=entry.rating,
                    rated_at=entry.rated_at,
                )
                for entry in body.entries
            ),
        )
    )
    return ImportLibraryResponseDTO(processed=processed)
