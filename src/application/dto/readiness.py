from dataclasses import dataclass

from port.recommendation import ModelStatus


@dataclass(frozen=True, slots=True)
class ReadinessDTO:
    database_ready: bool
    model: ModelStatus = "not_configured"
