import asyncio
from dataclasses import dataclass

from application.dto.library import MovieReference
from application.exceptions.movie import MovieNotFoundError
from application.service.movie_factory import movie_from_metadata
from domain.entities.movie import Movie
from port.clock import Clock
from port.id_generator import IdGenerator
from port.movie_metadata import MovieMetadataProvider
from port.repositories.movies import MovieRepository
from port.unit_of_work import UnitOfWork


@dataclass(frozen=True, slots=True)
class ResolvedMovie:
    movie: Movie
    requires_save: bool


class MovieResolver:
    def __init__(
        self,
        movies: MovieRepository,
        metadata: MovieMetadataProvider,
        uow: UnitOfWork,
        clock: Clock,
        ids: IdGenerator,
    ) -> None:
        self._movies, self._metadata, self._uow = movies, metadata, uow
        self._clock, self._ids = clock, ids

    async def prepare(self, references: tuple[MovieReference, ...]) -> tuple[ResolvedMovie, ...]:
        existing = []
        for reference in references:
            if reference.movie_id is not None:
                movie = await self._movies.get(reference.movie_id)
                if movie is None:
                    raise MovieNotFoundError
            else:
                movie = await self._movies.get_by_tmdb(
                    media_type=reference.media_type, tmdb_id=reference.tmdb_id
                )
            existing.append(movie)
        # End the read phase before waiting for external I/O; no write transaction is held.
        await self._uow.rollback()

        async def prepare_one(reference: MovieReference, movie: Movie | None) -> ResolvedMovie:
            if movie is not None:
                return ResolvedMovie(movie, False)
            metadata = await self._metadata.get_movie(
                media_type=reference.media_type, tmdb_id=reference.tmdb_id
            )
            return ResolvedMovie(movie_from_metadata(metadata, self._ids.new(), self._clock.now()), True)

        return tuple(
            await asyncio.gather(
                *(prepare_one(ref, movie) for ref, movie in zip(references, existing, strict=True))
            )
        )

    async def persist(self, resolved: ResolvedMovie) -> Movie:
        return await self._movies.upsert(resolved.movie) if resolved.requires_save else resolved.movie
