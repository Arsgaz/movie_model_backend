from typing import Literal

from pydantic import BaseModel


class HealthResponseDTO(BaseModel):
    status: Literal["ok"] = "ok"


class ReadinessResponseDTO(BaseModel):
    status: Literal["ready", "not_ready"]
    database: Literal["ready", "unavailable"]
    model: Literal["not_configured", "ready", "unavailable"] = "not_configured"
