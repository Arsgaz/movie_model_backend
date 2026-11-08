from application.base_interactor import Interactor
from application.dto.movie import ListMoviesQuery, MovieDTO, MovieListDTO
from port.repositories.movies import MovieRepository


class ListMoviesHandler(Interactor[ListMoviesQuery, MovieListDTO]):
    def __init__(self, movies: MovieRepository) -> None:
        self._movies = movies

    async def __call__(self, data: ListMoviesQuery) -> MovieListDTO:
        movies, total = await self._movies.list(
            query=data.query.strip(),
            media_type=data.media_type,
            genre_id=data.genre_id,
            page=data.page,
            page_size=data.page_size,
        )
        return MovieListDTO(
            items=tuple(MovieDTO.from_entity(movie) for movie in movies),
            page=data.page,
            page_size=data.page_size,
            total_results=total,
            total_pages=(total + data.page_size - 1) // data.page_size,
        )
