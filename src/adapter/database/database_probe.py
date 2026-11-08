import asyncio

from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncEngine


class SQLAlchemyDatabaseProbe:
    def __init__(self, engine: AsyncEngine, expected_heads: frozenset[str]) -> None:
        self._engine = engine
        self._expected_heads = expected_heads

    async def is_ready(self) -> bool:
        try:
            async with asyncio.timeout(3), self._engine.connect() as connection:
                versions = await connection.execute(text("SELECT version_num FROM alembic_version"))
                if frozenset(versions.scalars()) != self._expected_heads:
                    return False
                # A matching stamp alone does not prove the required tables exist.
                await connection.execute(text("SELECT id, name, created_at FROM profiles LIMIT 0"))
                await connection.execute(
                    text("SELECT id, tmdb_id, media_type, movielens_id FROM movies LIMIT 0")
                )
                await connection.execute(text("SELECT media_type, tmdb_id, name FROM genres LIMIT 0"))
                await connection.execute(
                    text("SELECT movie_id, media_type, genre_tmdb_id FROM movie_genres LIMIT 0")
                )
                await connection.execute(
                    text("SELECT profile_id, movie_id, value, updated_at FROM ratings LIMIT 0")
                )
                await connection.execute(
                    text("SELECT profile_id, movie_id, created_at FROM favorites LIMIT 0")
                )
                return True
        except (SQLAlchemyError, OSError, TimeoutError):
            return False
