from application.base_interactor import Interactor
from application.dto.library import FavoriteDTO, GetLibraryQuery, LibraryDTO, RatingDTO
from application.dto.movie import MovieDTO
from application.exceptions.profile import ProfileNotFoundError
from port.repositories.library import LibraryRepository
from port.repositories.profiles import ProfileRepository


class GetLibraryHandler(Interactor[GetLibraryQuery, LibraryDTO]):
    def __init__(self, profiles: ProfileRepository, library: LibraryRepository) -> None:
        self._profiles, self._library = profiles, library

    async def __call__(self, data: GetLibraryQuery) -> LibraryDTO:
        if await self._profiles.get(data.profile_id) is None:
            raise ProfileNotFoundError
        favorites = await self._library.favorites(data.profile_id)
        ratings = await self._library.ratings(data.profile_id)
        return LibraryDTO(
            favorites=tuple(
                FavoriteDTO(MovieDTO.from_entity(movie), favorite.created_at) for favorite, movie in favorites
            ),
            ratings=tuple(
                RatingDTO(MovieDTO.from_entity(movie), rating.value, rating.updated_at)
                for rating, movie in ratings
            ),
        )
