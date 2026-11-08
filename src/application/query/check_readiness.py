from application.base_interactor import Interactor
from application.dto.readiness import ReadinessDTO
from port.database_probe import DatabaseProbe
from port.recommendation import RecommendationEngine


class CheckReadinessHandler(Interactor[None, ReadinessDTO]):
    def __init__(self, database: DatabaseProbe, engine: RecommendationEngine) -> None:
        self._database, self._engine = database, engine

    async def __call__(self, data: None = None) -> ReadinessDTO:
        return ReadinessDTO(database_ready=await self._database.is_ready(), model=await self._engine.status())
