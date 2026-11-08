from typing import Annotated
from uuid import UUID

from dishka.integrations.fastapi import DishkaRoute, FromDishka
from fastapi import APIRouter, Path, Query

from application.dto.movie import GetMovieByTMDBQuery, GetMovieQuery, ListMoviesQuery
from application.query.get_movie import GetMovieHandler
from application.query.get_movie_by_tmdb import GetMovieByTMDBHandler
from application.query.list_movies import ListMoviesHandler
from domain.value_objects.media_type import MediaType
from presentation.http.dto.error import ErrorResponseDTO
from presentation.http.dto.movie import MovieListResponseDTO, MovieResponseDTO

router = APIRouter(
    prefix="/api/v1/movies",
    tags=["movies"],
    route_class=DishkaRoute,
    responses={503: {"model": ErrorResponseDTO}},
)


@router.get("", response_model=MovieListResponseDTO)
async def list_movies(
    handler: FromDishka[ListMoviesHandler],
    q: Annotated[str, Query(max_length=150)] = "",
    media_type: MediaType = MediaType.MOVIE,
    genre_id: Annotated[int | None, Query(gt=0)] = None,
    page: Annotated[int, Query(ge=1)] = 1,
    page_size: Annotated[int, Query(ge=1, le=100)] = 20,
) -> MovieListResponseDTO:
    result = await handler(
        ListMoviesQuery(query=q, media_type=media_type, genre_id=genre_id, page=page, page_size=page_size)
    )
    return MovieListResponseDTO.model_validate(result)


@router.get("/{movie_id}", response_model=MovieResponseDTO, responses={404: {"model": ErrorResponseDTO}})
async def get_movie(movie_id: UUID, handler: FromDishka[GetMovieHandler]) -> MovieResponseDTO:
    return MovieResponseDTO.model_validate(await handler(GetMovieQuery(movie_id=movie_id)))


@router.get(
    "/by-tmdb/{media_type}/{tmdb_id}",
    response_model=MovieResponseDTO,
    responses={404: {"model": ErrorResponseDTO}},
)
async def get_movie_by_tmdb(
    media_type: MediaType,
    tmdb_id: Annotated[int, Path(gt=0)],
    handler: FromDishka[GetMovieByTMDBHandler],
) -> MovieResponseDTO:
    return MovieResponseDTO.model_validate(
        await handler(GetMovieByTMDBQuery(media_type=media_type, tmdb_id=tmdb_id))
    )
