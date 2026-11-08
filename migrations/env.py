import asyncio
from logging.config import fileConfig

from alembic import context
from sqlalchemy.engine import Connection

from adapter.database.engine import create_engine
from adapter.database.models import Base
from config.settings import Settings

config = context.config
if config.config_file_name:
    fileConfig(config.config_file_name)
target_metadata = Base.metadata


def get_database_url() -> str:
    # A supplied settings object makes tests independent of the developer environment.
    value = config.attributes.get("database_url")
    return value if value is not None else Settings().database_url.get_secret_value()


def run_migrations_offline() -> None:
    context.configure(
        url=get_database_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        render_as_batch=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_with_connection(connection: Connection) -> None:
    context.configure(
        connection=connection, target_metadata=target_metadata, render_as_batch=True, compare_type=True
    )
    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    engine = create_engine(get_database_url())
    try:
        async with engine.connect() as connection:
            await connection.run_sync(run_with_connection)
    finally:
        await engine.dispose()


if context.is_offline_mode():
    run_migrations_offline()
elif (connection := config.attributes.get("connection")) is not None:
    run_with_connection(connection)
else:
    asyncio.run(run_async_migrations())
