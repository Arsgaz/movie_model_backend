from typing import Protocol
from uuid import UUID

from domain.entities.movie import Movie
from domain.value_objects.media_type import MediaType


class MovieRepository(Protocol):
    async def upsert(self, movie: Movie) -> Movie: ...

    async def get(self, movie_id: UUID) -> Movie | None: ...

    async def get_by_tmdb(self, *, media_type: MediaType, tmdb_id: int) -> Movie | None: ...

    async def list(
        self,
        *,
        query: str,
        media_type: MediaType,
        genre_id: int | None,
        page: int,
        page_size: int,
    ) -> tuple[list[Movie], int]: ...
