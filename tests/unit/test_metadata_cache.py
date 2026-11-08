import asyncio

import pytest

from adapter.cache.ttl import AsyncTTLCache, CacheBusyError
from adapter.tmdb.cached_metadata import CachedMovieMetadataProvider
from domain.value_objects.media_type import MediaType
from port.exceptions.movie_metadata import MovieMetadataUnavailableError
from tests.factories import FakeMetadata


async def test_cache_expiration_and_lru_bound() -> None:
    clock = [0.0]
    cache = AsyncTTLCache[str, int](max_entries=2, clock=lambda: clock[0])
    calls = 0

    async def load() -> int:
        nonlocal calls
        calls += 1
        return calls

    try:
        assert await cache.get_or_load("a", load, ttl=10) == 1
        assert await cache.get_or_load("b", load, ttl=10) == 2
        assert await cache.get_or_load("a", load, ttl=10) == 1
        assert await cache.get_or_load("c", load, ttl=10) == 3
        assert await cache.get_or_load("b", load, ttl=10) == 4
        clock[0] = 11
        assert await cache.get_or_load("b", load, ttl=10) == 5
    finally:
        await cache.close()


async def test_identical_concurrent_requests_share_one_load() -> None:
    cache = AsyncTTLCache[str, int](max_entries=2)
    started, release = asyncio.Event(), asyncio.Event()
    calls = 0

    async def load() -> int:
        nonlocal calls
        calls += 1
        started.set()
        await release.wait()
        return 42

    tasks = [asyncio.create_task(cache.get_or_load("same", load, ttl=10)) for _ in range(10)]
    await started.wait()
    release.set()
    assert await asyncio.gather(*tasks) == [42] * 10
    assert calls == 1
    await cache.close()


async def test_cancelled_waiter_does_not_cancel_shared_load() -> None:
    cache = AsyncTTLCache[str, int](max_entries=2)
    started, release = asyncio.Event(), asyncio.Event()

    async def load() -> int:
        started.set()
        await release.wait()
        return 42

    first = asyncio.create_task(cache.get_or_load("same", load, ttl=10))
    second = asyncio.create_task(cache.get_or_load("same", load, ttl=10))
    await started.wait()
    first.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first
    release.set()
    assert await second == 42
    await cache.close()


async def test_failures_are_not_cached() -> None:
    cache = AsyncTTLCache[str, int](max_entries=2)
    calls = 0

    async def load() -> int:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise MovieMetadataUnavailableError
        return 42

    with pytest.raises(MovieMetadataUnavailableError):
        await cache.get_or_load("same", load, ttl=10)
    assert await cache.get_or_load("same", load, ttl=10) == 42
    assert calls == 2
    await cache.close()


async def test_pending_limit_and_shutdown() -> None:
    cache = AsyncTTLCache[str, int](max_entries=2, max_pending=1)
    started, release = asyncio.Event(), asyncio.Event()

    async def load() -> int:
        started.set()
        await release.wait()
        return 42

    first = asyncio.create_task(cache.get_or_load("a", load, ttl=10))
    await started.wait()
    with pytest.raises(CacheBusyError):
        await cache.get_or_load("b", load, ttl=10)
    await cache.close()
    with pytest.raises(asyncio.CancelledError):
        await first


async def test_metadata_cache_keys_include_media_page_and_genre() -> None:
    source = FakeMetadata()
    cached = CachedMovieMetadataProvider(source, ttl=300, max_entries=4)
    try:
        for query in (" ФИЛЬМ  тест ", "фильм тест"):
            await cached.get_page(media_type=MediaType.MOVIE, query=query, genre_id=None, page=1)
        assert len(source.calls) == 1
        await cached.get_page(media_type=MediaType.MOVIE, query="фильм тест", genre_id=16, page=1)
        await cached.get_page(media_type=MediaType.MOVIE, query="фильм тест", genre_id=None, page=2)
        await cached.get_page(media_type=MediaType.TV, query="фильм тест", genre_id=None, page=1)
        assert len(source.calls) == 4
        await cached.get_movie(media_type=MediaType.MOVIE, tmdb_id=123)
        await cached.get_movie(media_type=MediaType.MOVIE, tmdb_id=123)
        assert len(source.calls) == 5
    finally:
        await cached.close()


async def test_exploration_cache_keys_include_year_genre_and_page():
    source = FakeMetadata()
    cache = CachedMovieMetadataProvider(source, ttl=300, max_entries=10)
    try:
        await asyncio.gather(*(cache.get_exploration_page(genre_id=16, year=2000, page=1) for _ in range(4)))
        assert len(source.calls) == 1
        await cache.get_exploration_page(genre_id=16, year=2001, page=1)
        await cache.get_exploration_page(genre_id=None, year=2000, page=1)
        await cache.get_exploration_page(genre_id=16, year=2000, page=2)
        await cache.get_page(media_type=MediaType.MOVIE, query="", genre_id=16, page=1)
        assert len(source.calls) == 5
    finally:
        await cache.close()
