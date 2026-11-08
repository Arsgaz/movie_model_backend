from collections.abc import AsyncIterator

from alembic.script import ScriptDirectory
from dishka import Provider, Scope, provide
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from adapter.database.database_probe import SQLAlchemyDatabaseProbe
from adapter.database.engine import create_engine
from adapter.database.migrations import migration_config
from adapter.database.unit_of_work import SQLAlchemyUnitOfWork
from config.settings import Settings
from port.database_probe import DatabaseProbe
from port.unit_of_work import UnitOfWork


class DatabaseProvider(Provider):
    @provide(scope=Scope.APP)
    async def engine(self, settings: Settings) -> AsyncIterator[AsyncEngine]:
        engine = create_engine(settings.database_url.get_secret_value())
        try:
            yield engine
        finally:
            await engine.dispose()

    @provide(scope=Scope.APP)
    def session_factory(self, engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
        return async_sessionmaker(engine, expire_on_commit=False)

    @provide(scope=Scope.REQUEST)
    async def session(self, factory: async_sessionmaker[AsyncSession]) -> AsyncIterator[AsyncSession]:
        async with factory() as session:
            yield session
        # Session.close rolls back any uncommitted transaction.

    @provide(scope=Scope.REQUEST)
    def unit_of_work(self, session: AsyncSession) -> UnitOfWork:
        return SQLAlchemyUnitOfWork(session)

    @provide(scope=Scope.APP)
    def database_probe(self, engine: AsyncEngine, settings: Settings) -> DatabaseProbe:
        heads = frozenset(ScriptDirectory.from_config(migration_config(settings)).get_heads())
        return SQLAlchemyDatabaseProbe(engine, heads)
