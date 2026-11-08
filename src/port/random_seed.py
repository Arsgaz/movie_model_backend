from typing import Protocol


class RandomSeedGenerator(Protocol):
    def new(self) -> int: ...
