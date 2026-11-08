from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from adapter.database.converters.profile import to_entity, to_model
from adapter.database.models.profile import ProfileModel
from domain.entities.profile import Profile


class SQLAlchemyProfileRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def add(self, profile: Profile) -> None:
        self._session.add(to_model(profile))
        await self._session.flush()

    async def get(self, profile_id: UUID) -> Profile | None:
        model = await self._session.get(ProfileModel, profile_id)
        return to_entity(model) if model is not None else None

    async def list(self, *, limit: int, offset: int) -> list[Profile]:
        statement = (
            select(ProfileModel)
            .order_by(ProfileModel.created_at, ProfileModel.id)
            .limit(limit)
            .offset(offset)
        )
        models = (await self._session.scalars(statement)).all()
        return [to_entity(model) for model in models]
