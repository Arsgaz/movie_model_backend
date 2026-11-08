from typing import Protocol


class DatabaseProbe(Protocol):
    async def is_ready(self) -> bool: ...
