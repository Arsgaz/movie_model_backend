from datetime import datetime
from typing import Annotated
from uuid import UUID

from pydantic import BaseModel, ConfigDict, StringConstraints


class CreateProfileRequestDTO(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=80)]


class ProfileResponseDTO(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: UUID
    name: str
    created_at: datetime


class ProfileListResponseDTO(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    items: list[ProfileResponseDTO]
    limit: int
    offset: int
