from dataclasses import replace
from datetime import UTC

from application.base_interactor import Interactor
from application.dto.library import ImportLibraryCommand, MovieReference
from application.exceptions.profile import ProfileNotFoundError
from application.service.movie_resolver import MovieResolver
from domain.entities.favorite import Favorite
from domain.entities.rating import Rating
from port.clock import Clock
from port.repositories.library import LibraryRepository
from port.repositories.profiles import ProfileRepository
from port.unit_of_work import UnitOfWork


class ImportLibraryHandler(Interactor[ImportLibraryCommand, int]):
    def __init__(
        self,
        profiles: ProfileRepository,
        library: LibraryRepository,
        resolver: MovieResolver,
        clock: Clock,
        uow: UnitOfWork,
    ) -> None:
        self._profiles, self._library, self._resolver = profiles, library, resolver
        self._clock, self._uow = clock, uow

    async def __call__(self, data: ImportLibraryCommand) -> int:
        if await self._profiles.get(data.profile_id) is None:
            raise ProfileNotFoundError
        unique = {}
        for entry in data.entries:
            key = (entry.media_type, entry.tmdb_id)
            previous = unique.get(key)
            unique[key] = (
                entry
                if previous is None
                else replace(
                    entry,
                    favorite=entry.favorite or previous.favorite,
                    rating=entry.rating if entry.rating is not None else previous.rating,
                    rated_at=entry.rated_at if entry.rating is not None else previous.rated_at,
                )
            )
        entries = tuple(unique.values())
        prepared = await self._resolver.prepare(
            tuple(MovieReference(media_type=entry.media_type, tmdb_id=entry.tmdb_id) for entry in entries)
        )
        try:
            for entry, resolved in zip(entries, prepared, strict=True):
                movie = await self._resolver.persist(resolved)
                if entry.favorite:
                    await self._library.put_favorite(Favorite(data.profile_id, movie.id, self._clock.now()))
                if entry.rating is not None:
                    when = entry.rated_at.astimezone(UTC) if entry.rated_at else self._clock.now()
                    await self._library.put_rating(
                        Rating(data.profile_id, movie.id, entry.rating, when), overwrite=False
                    )
            await self._uow.commit()
        except Exception:
            await self._uow.rollback()
            raise
        return len(entries)
