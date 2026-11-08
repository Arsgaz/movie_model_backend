from application.base_interactor import Interactor
from application.dto.movie import GetMovieQuery, MovieDTO
from application.exceptions.movie import MovieNotFoundError
from port.repositories.movies import MovieRepository


class GetMovieHandler(Interactor[GetMovieQuery, MovieDTO]):
    def __init__(self, movies: MovieRepository) -> None:
        self._movies = movies

    async def __call__(self, data: GetMovieQuery) -> MovieDTO:
        movie = await self._movies.get(data.movie_id)
        if movie is None:
            raise MovieNotFoundError
        return MovieDTO.from_entity(movie)
