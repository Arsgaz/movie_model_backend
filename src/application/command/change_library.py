from application.base_interactor import Interactor
from application.dto.library import ChangeLibraryCommand, FavoriteDTO, LibraryOperation, RatingDTO
from application.dto.movie import MovieDTO
from application.exceptions.profile import ProfileNotFoundError
from application.service.movie_resolver import MovieResolver
from domain.entities.favorite import Favorite
from domain.entities.rating import Rating
from domain.exceptions.rating import InvalidRatingError
from port.clock import Clock
from port.repositories.library import LibraryRepository
from port.repositories.profiles import ProfileRepository
from port.unit_of_work import UnitOfWork


class ChangeLibraryHandler(Interactor[ChangeLibraryCommand, FavoriteDTO | RatingDTO | None]):
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

    async def __call__(self, data: ChangeLibraryCommand) -> FavoriteDTO | RatingDTO | None:
        if data.operation == LibraryOperation.RATE and (
            type(data.value) is not int or not 1 <= data.value <= 5
        ):
            raise InvalidRatingError
        if await self._profiles.get(data.profile_id) is None:
            raise ProfileNotFoundError
        if data.operation in (LibraryOperation.UNFAVORITE, LibraryOperation.UNRATE):
            if data.movie.movie_id is None:
                raise ValueError("Deletion needs a local movie UUID")
            try:
                remove = (
                    self._library.remove_favorite
                    if data.operation == LibraryOperation.UNFAVORITE
                    else self._library.remove_rating
                )
                await remove(profile_id=data.profile_id, movie_id=data.movie.movie_id)
                await self._uow.commit()
            except Exception:
                await self._uow.rollback()
                raise
            return None
        prepared = (await self._resolver.prepare((data.movie,)))[0]
        try:
            movie = await self._resolver.persist(prepared)
            if data.operation == LibraryOperation.FAVORITE:
                favorite = await self._library.put_favorite(
                    Favorite(data.profile_id, movie.id, self._clock.now())
                )
                result = FavoriteDTO(MovieDTO.from_entity(movie), favorite.created_at)
            elif data.operation == LibraryOperation.RATE:
                rating = await self._library.put_rating(
                    Rating(data.profile_id, movie.id, data.value, self._clock.now())
                )
                result = RatingDTO(MovieDTO.from_entity(movie), rating.value, rating.updated_at)
            else:
                raise ValueError("Invalid library operation")
            await self._uow.commit()
            return result
        except Exception:
            await self._uow.rollback()
            raise
