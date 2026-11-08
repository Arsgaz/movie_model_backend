from typing import Annotated

from dishka.integrations.fastapi import DishkaRoute, FromDishka
from fastapi import APIRouter, Path

from application.command.import_catalog import ImportCatalogHandler
from application.command.import_movie import ImportMovieHandler
from application.dto.catalog import ImportCatalogPageCommand, ImportMovieCommand
from domain.value_objects.media_type import MediaType
from presentation.http.dto.catalog import CatalogImportResponseDTO, ImportCatalogRequestDTO
from presentation.http.dto.error import ErrorResponseDTO
from presentation.http.dto.movie import MovieResponseDTO

router = APIRouter(
    prefix="/api/v1/catalog",
    tags=["catalog import"],
    route_class=DishkaRoute,
    responses={status: {"model": ErrorResponseDTO} for status in (404, 502, 503)},
)


@router.post("/import", response_model=CatalogImportResponseDTO)
async def import_page(
    body: ImportCatalogRequestDTO, handler: FromDishka[ImportCatalogHandler]
) -> CatalogImportResponseDTO:
    result = await handler(
        ImportCatalogPageCommand(
            media_type=body.media_type,
            query=body.q,
            genre_id=body.genre_id,
            page=body.page,
        )
    )
    return CatalogImportResponseDTO.model_validate(result)


@router.post("/import/{media_type}/{tmdb_id}", response_model=MovieResponseDTO)
async def import_movie(
    media_type: MediaType,
    tmdb_id: Annotated[int, Path(gt=0)],
    handler: FromDishka[ImportMovieHandler],
) -> MovieResponseDTO:
    return MovieResponseDTO.model_validate(
        await handler(ImportMovieCommand(media_type=media_type, tmdb_id=tmdb_id))
    )
