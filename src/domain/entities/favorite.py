from dataclasses import dataclass
from datetime import datetime
from uuid import UUID


@dataclass(frozen=True, slots=True)
class Favorite:
    profile_id: UUID
    movie_id: UUID
    created_at: datetime
