from typing import Protocol
from uuid import UUID

from domain.entities.profile import Profile


class ProfileRepository(Protocol):
    async def add(self, profile: Profile) -> None: ...

    async def get(self, profile_id: UUID) -> Profile | None: ...

    async def list(self, *, limit: int, offset: int) -> list[Profile]: ...
