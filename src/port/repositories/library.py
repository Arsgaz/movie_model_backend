from typing import Protocol
from uuid import UUID

from domain.entities.favorite import Favorite
from domain.entities.movie import Movie
from domain.entities.rating import Rating


class LibraryRepository(Protocol):
    async def put_favorite(self, favorite: Favorite) -> Favorite: ...

    async def put_rating(self, rating: Rating, *, overwrite: bool = True) -> Rating: ...

    async def remove_favorite(self, *, profile_id: UUID, movie_id: UUID) -> None: ...

    async def remove_rating(self, *, profile_id: UUID, movie_id: UUID) -> None: ...

    async def favorites(self, profile_id: UUID) -> list[tuple[Favorite, Movie]]: ...

    async def ratings(self, profile_id: UUID) -> list[tuple[Rating, Movie]]: ...
