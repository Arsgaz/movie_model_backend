from dataclasses import dataclass
from datetime import datetime
from uuid import UUID

from domain.exceptions.profile import InvalidProfileNameError


@dataclass(frozen=True, slots=True)
class Profile:
    id: UUID
    name: str
    created_at: datetime

    def __post_init__(self) -> None:
        name = self.name.strip()
        if not 1 <= len(name) <= 80:
            raise InvalidProfileNameError("Profile name must contain 1 to 80 characters")
        object.__setattr__(self, "name", name)
