import asyncio
from collections import OrderedDict
from collections.abc import Awaitable, Callable, Hashable
from time import monotonic


class CacheBusyError(Exception):
    pass


class AsyncTTLCache[K: Hashable, V]:
    """Bounded process-local cache with shared in-flight loads and cancellation isolation."""

    def __init__(
        self,
        *,
        max_entries: int,
        max_pending: int = 32,
        concurrency: int = 4,
        clock: Callable[[], float] = monotonic,
    ) -> None:
        self._max_entries = max_entries
        self._max_pending = max_pending
        self._clock = clock
        self._entries: OrderedDict[K, tuple[float, V]] = OrderedDict()
        self._pending: dict[K, asyncio.Task[V]] = {}
        self._semaphore = asyncio.Semaphore(concurrency)

    async def get_or_load(self, key: K, loader: Callable[[], Awaitable[V]], *, ttl: float) -> V:
        cached = self._entries.get(key)
        if cached is not None:
            expires, value = cached
            if expires > self._clock():
                self._entries.move_to_end(key)
                return value
            del self._entries[key]
        task = self._pending.get(key)
        if task is None:
            if len(self._pending) >= self._max_pending:
                raise CacheBusyError
            task = asyncio.create_task(self._load(key, loader, ttl))
            self._pending[key] = task
            # Retrieve exceptions even if all waiting requests have disconnected.
            task.add_done_callback(lambda done: None if done.cancelled() else done.exception())
        return await asyncio.shield(task)

    async def _load(self, key: K, loader: Callable[[], Awaitable[V]], ttl: float) -> V:
        try:
            async with self._semaphore:
                value = await loader()
            now = self._clock()
            for expired in [key for key, (expires, _) in self._entries.items() if expires <= now]:
                del self._entries[expired]
            self._entries[key] = (now + ttl, value)
            self._entries.move_to_end(key)
            while len(self._entries) > self._max_entries:
                self._entries.popitem(last=False)
            return value
        finally:
            self._pending.pop(key, None)

    async def close(self) -> None:
        pending = tuple(self._pending.values())
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
        self._pending.clear()
        self._entries.clear()
