from dataclasses import dataclass
from datetime import datetime
from uuid import UUID

from domain.exceptions.rating import InvalidRatingError


@dataclass(frozen=True, slots=True)
class Rating:
    profile_id: UUID
    movie_id: UUID
    value: int
    updated_at: datetime

    def __post_init__(self) -> None:
        if type(self.value) is not int or not 1 <= self.value <= 5:
            raise InvalidRatingError("Rating must be an integer from 1 to 5")
