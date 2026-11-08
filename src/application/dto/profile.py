from dataclasses import dataclass
from datetime import datetime
from uuid import UUID


@dataclass(frozen=True, slots=True)
class CreateProfileCommand:
    name: str


@dataclass(frozen=True, slots=True)
class GetProfileQuery:
    profile_id: UUID


@dataclass(frozen=True, slots=True)
class ListProfilesQuery:
    limit: int = 50
    offset: int = 0

    def __post_init__(self) -> None:
        if not 1 <= self.limit <= 100 or self.offset < 0:
            raise ValueError("Invalid pagination")


@dataclass(frozen=True, slots=True)
class ProfileDTO:
    id: UUID
    name: str
    created_at: datetime


@dataclass(frozen=True, slots=True)
class ProfileListDTO:
    items: tuple[ProfileDTO, ...]
    limit: int
    offset: int
