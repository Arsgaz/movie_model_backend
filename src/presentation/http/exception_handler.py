from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from sqlalchemy.exc import SQLAlchemyError

from application.exceptions.movie import MovieNotFoundError
from application.exceptions.profile import ProfileNotFoundError
from domain.exceptions.profile import InvalidProfileNameError
from domain.exceptions.rating import InvalidRatingError
from port.exceptions.movie_metadata import (
    MovieMetadataNotConfiguredError,
    MovieMetadataNotFoundError,
    MovieMetadataUnavailableError,
)
from port.exceptions.recommendation import RecommendationUnavailableError
from presentation.http.dto.error import ErrorDetailDTO, ErrorResponseDTO


def _error_response(status_code: int, code: str, message: str) -> JSONResponse:
    body = ErrorResponseDTO(error=ErrorDetailDTO(code=code, message=message))
    return JSONResponse(status_code=status_code, content=body.model_dump(mode="json"))


async def profile_not_found_exception_handler(request: Request, error: ProfileNotFoundError) -> JSONResponse:
    return _error_response(404, "profile_not_found", "Profile not found")


async def invalid_profile_name_exception_handler(
    request: Request,
    error: InvalidProfileNameError,
) -> JSONResponse:
    return _error_response(422, "invalid_profile_name", "Name must contain 1 to 80 characters")


async def database_exception_handler(request: Request, error: SQLAlchemyError) -> JSONResponse:
    return _error_response(503, "database_unavailable", "Database is temporarily unavailable")


def setup_exception_handlers(app: FastAPI) -> None:
    app.add_exception_handler(ProfileNotFoundError, profile_not_found_exception_handler)
    app.add_exception_handler(InvalidProfileNameError, invalid_profile_name_exception_handler)
    app.add_exception_handler(InvalidRatingError, invalid_rating_exception_handler)
    app.add_exception_handler(RecommendationUnavailableError, recommendation_unavailable_exception_handler)
    app.add_exception_handler(SQLAlchemyError, database_exception_handler)
    app.add_exception_handler(MovieNotFoundError, movie_not_found_exception_handler)
    app.add_exception_handler(MovieMetadataNotFoundError, movie_not_found_exception_handler)
    app.add_exception_handler(MovieMetadataUnavailableError, metadata_unavailable_exception_handler)
    app.add_exception_handler(MovieMetadataNotConfiguredError, metadata_not_configured_exception_handler)


async def movie_not_found_exception_handler(request: Request, error: Exception) -> JSONResponse:
    return _error_response(404, "movie_not_found", "Movie not found")


async def metadata_unavailable_exception_handler(
    request: Request, error: MovieMetadataUnavailableError
) -> JSONResponse:
    return _error_response(502, "metadata_unavailable", "Movie metadata provider is temporarily unavailable")


async def metadata_not_configured_exception_handler(
    request: Request, error: MovieMetadataNotConfiguredError
) -> JSONResponse:
    return _error_response(503, "catalog_not_configured", "Movie metadata provider is not configured")


async def invalid_rating_exception_handler(request: Request, error: InvalidRatingError) -> JSONResponse:
    return _error_response(422, "invalid_rating", "Rating must be an integer from 1 to 5")


async def recommendation_unavailable_exception_handler(
    request: Request, error: RecommendationUnavailableError
) -> JSONResponse:
    return _error_response(503, "recommendation_unavailable", "Recommendation model is not available")
