from adapter.cache.ttl import AsyncTTLCache, CacheBusyError
from domain.value_objects.media_type import MediaType
from port.exceptions.movie_metadata import MovieMetadataUnavailableError
from port.movie_metadata import MetadataPage, MovieMetadata, MovieMetadataProvider


class CachedMovieMetadataProvider:
    def __init__(self, source: MovieMetadataProvider, *, ttl: float, max_entries: int) -> None:
        self._source = source
        self._ttl = ttl
        self._pages: AsyncTTLCache[tuple, MetadataPage] = AsyncTTLCache(
            max_entries=max_entries,
        )
        self._movies: AsyncTTLCache[tuple[MediaType, int], MovieMetadata] = AsyncTTLCache(
            max_entries=max_entries,
        )

    async def get_page(
        self,
        *,
        media_type: MediaType,
        query: str,
        genre_id: int | None,
        page: int,
    ) -> MetadataPage:
        query = " ".join(query.split())
        key = (media_type, query.casefold(), genre_id, page)
        try:
            return await self._pages.get_or_load(
                key,
                lambda: self._source.get_page(
                    media_type=media_type, query=query, genre_id=genre_id, page=page
                ),
                ttl=self._ttl,
            )
        except CacheBusyError:
            raise MovieMetadataUnavailableError from None

    async def get_exploration_page(self, *, genre_id: int | None, year: int, page: int) -> MetadataPage:
        try:
            return await self._pages.get_or_load(
                ("explore", genre_id, year, page),
                lambda: self._source.get_exploration_page(genre_id=genre_id, year=year, page=page),
                ttl=self._ttl,
            )
        except CacheBusyError:
            raise MovieMetadataUnavailableError from None

    async def get_movie(self, *, media_type: MediaType, tmdb_id: int) -> MovieMetadata:
        try:
            return await self._movies.get_or_load(
                (media_type, tmdb_id),
                lambda: self._source.get_movie(media_type=media_type, tmdb_id=tmdb_id),
                ttl=self._ttl,
            )
        except CacheBusyError:
            raise MovieMetadataUnavailableError from None

    async def close(self) -> None:
        await self._pages.close()
        await self._movies.close()
