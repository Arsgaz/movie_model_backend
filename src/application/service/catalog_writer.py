from application.dto.movie import MovieDTO
from application.service.movie_factory import movie_from_metadata
from port.clock import Clock
from port.id_generator import IdGenerator
from port.movie_metadata import MovieMetadata
from port.repositories.movies import MovieRepository
from port.unit_of_work import UnitOfWork


class CatalogWriter:
    """Persist an imported batch in one transaction."""

    def __init__(self, movies: MovieRepository, uow: UnitOfWork, clock: Clock, ids: IdGenerator) -> None:
        self._movies = movies
        self._uow = uow
        self._clock = clock
        self._ids = ids

    async def save(self, metadata: tuple[MovieMetadata, ...]) -> tuple[MovieDTO, ...]:
        now = self._clock.now()
        saved = []
        try:
            unique = {(item.media_type, item.tmdb_id): item for item in metadata}
            for item in unique.values():
                movie = movie_from_metadata(item, self._ids.new(), now)
                saved.append(MovieDTO.from_entity(await self._movies.upsert(movie)))
            await self._uow.commit()
        except Exception:
            await self._uow.rollback()
            raise
        return tuple(saved)
