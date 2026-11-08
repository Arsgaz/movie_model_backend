from collections.abc import AsyncIterator

import httpx2
from dishka import Provider, Scope, provide

from adapter.tmdb.cached_metadata import CachedMovieMetadataProvider
from adapter.tmdb.movie_metadata import TMDBMovieMetadataProvider
from config.settings import Settings
from port.movie_metadata import MovieMetadataProvider


class MetadataProvider(Provider):
    scope = Scope.APP

    @provide
    async def http_client(self, settings: Settings) -> AsyncIterator[httpx2.AsyncClient]:
        async with httpx2.AsyncClient(
            base_url="https://api.themoviedb.org/3/",
            timeout=settings.tmdb_timeout,
            follow_redirects=False,
            limits=httpx2.Limits(max_connections=10, max_keepalive_connections=5),
        ) as client:
            yield client

    @provide
    async def movie_metadata(
        self, client: httpx2.AsyncClient, settings: Settings
    ) -> AsyncIterator[MovieMetadataProvider]:
        source = TMDBMovieMetadataProvider(client, settings)
        cached = CachedMovieMetadataProvider(
            source, ttl=settings.tmdb_cache_ttl, max_entries=settings.tmdb_cache_entries
        )
        try:
            yield cached
        finally:
            await cached.close()
            await source.close()
