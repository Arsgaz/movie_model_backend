from application.base_interactor import Interactor
from application.dto.movie import GetMovieByTMDBQuery, MovieDTO
from application.exceptions.movie import MovieNotFoundError
from port.repositories.movies import MovieRepository


class GetMovieByTMDBHandler(Interactor[GetMovieByTMDBQuery, MovieDTO]):
    def __init__(self, movies: MovieRepository) -> None:
        self._movies = movies

    async def __call__(self, data: GetMovieByTMDBQuery) -> MovieDTO:
        movie = await self._movies.get_by_tmdb(media_type=data.media_type, tmdb_id=data.tmdb_id)
        if movie is None:
            raise MovieNotFoundError
        return MovieDTO.from_entity(movie)
