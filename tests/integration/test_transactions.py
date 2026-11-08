from datetime import UTC, datetime
from uuid import uuid4

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker

from adapter.database.engine import create_engine
from adapter.database.repositories.profiles import SQLAlchemyProfileRepository
from adapter.database.unit_of_work import SQLAlchemyUnitOfWork
from adapter.system.clock import SystemClock
from adapter.system.id_generator import UUIDGenerator
from application.command.create_profile import CreateProfileHandler
from application.dto.profile import CreateProfileCommand
from config.settings import Settings
from domain.entities.profile import Profile


async def test_write_failure_rolls_back_flushed_data(migrated_settings: Settings) -> None:
    engine = create_engine(migrated_settings.database_url.get_secret_value())
    factory = async_sessionmaker(engine, expire_on_commit=False)

    class FailingRepository(SQLAlchemyProfileRepository):
        async def add(self, profile: Profile) -> None:
            await super().add(profile)
            raise RuntimeError("Failure after flush")

    try:
        async with factory() as session:
            handler = CreateProfileHandler(
                FailingRepository(session),
                SQLAlchemyUnitOfWork(session),
                SystemClock(),
                UUIDGenerator(),
            )
            with pytest.raises(RuntimeError, match="Failure after flush"):
                await handler(CreateProfileCommand(name="Should not persist"))
            assert not session.in_transaction()
        async with factory() as session:
            assert await SQLAlchemyProfileRepository(session).list(limit=10, offset=0) == []
    finally:
        await engine.dispose()


async def test_closing_session_discards_uncommitted_writes(migrated_settings: Settings) -> None:
    engine = create_engine(migrated_settings.database_url.get_secret_value())
    factory = async_sessionmaker(engine, expire_on_commit=False)
    try:
        async with factory() as session:
            await SQLAlchemyProfileRepository(session).add(
                Profile(id=uuid4(), name="Uncommitted", created_at=datetime.now(UTC)),
            )
        async with factory() as session:
            assert await SQLAlchemyProfileRepository(session).list(limit=10, offset=0) == []
            assert (await session.execute(text("PRAGMA foreign_keys"))).scalar() == 1
    finally:
        await engine.dispose()
