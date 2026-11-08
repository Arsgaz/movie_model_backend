from typing import Annotated

from dishka.integrations.fastapi import DishkaRoute, FromDishka
from fastapi import APIRouter, Path, Query

from application.dto.discovery import DiscoverMovieQuery, DiscoverMoviesQuery
from application.query.discover_movies import DiscoverMovieHandler, DiscoverMoviesHandler
from domain.value_objects.media_type import MediaType
from presentation.http.dto.discovery import DiscoveryPageResponseDTO, MoviePreviewResponseDTO
from presentation.http.dto.error import ErrorResponseDTO

router = APIRouter(
    prefix="/api/v1/discovery/movies",
    tags=["discovery"],
    route_class=DishkaRoute,
    responses={code: {"model": ErrorResponseDTO} for code in (404, 502, 503)},
)


@router.get("", response_model=DiscoveryPageResponseDTO)
async def discover_movies(
    handler: FromDishka[DiscoverMoviesHandler],
    q: Annotated[str, Query(max_length=150)] = "",
    media_type: MediaType = MediaType.MOVIE,
    genre_id: Annotated[int | None, Query(gt=0)] = None,
    page: Annotated[int, Query(ge=1, le=500)] = 1,
) -> DiscoveryPageResponseDTO:
    return DiscoveryPageResponseDTO.model_validate(
        await handler(
            DiscoverMoviesQuery(
                media_type=media_type,
                query=q,
                genre_id=genre_id,
                page=page,
            )
        )
    )


@router.get("/{media_type}/{tmdb_id}", response_model=MoviePreviewResponseDTO)
async def discover_movie(
    media_type: MediaType,
    tmdb_id: Annotated[int, Path(gt=0)],
    handler: FromDishka[DiscoverMovieHandler],
) -> MoviePreviewResponseDTO:
    return MoviePreviewResponseDTO.model_validate(
        await handler(DiscoverMovieQuery(media_type=media_type, tmdb_id=tmdb_id))
    )
