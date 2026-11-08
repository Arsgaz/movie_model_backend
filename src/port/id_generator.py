from typing import Protocol
from uuid import UUID


class IdGenerator(Protocol):
    def new(self) -> UUID: ...
