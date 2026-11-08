import asyncio

import pytest
from sqlalchemy import select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import async_sessionmaker

from adapter.database.engine import create_engine
from adapter.database.models.movie import GenreModel, MovieModel, movie_genres
from adapter.database.repositories.movies import SQLAlchemyMovieRepository
from adapter.database.unit_of_work import SQLAlchemyUnitOfWork
from adapter.system.clock import SystemClock
from adapter.system.id_generator import UUIDGenerator
from application.service.catalog_writer import CatalogWriter
from config.settings import Settings
from domain.entities.genre import Genre
from domain.entities.movie import Movie
from domain.value_objects.media_type import MediaType
from tests.factories import metadata


def writer(session, repository=None):
    return CatalogWriter(
        repository or SQLAlchemyMovieRepository(session),
        SQLAlchemyUnitOfWork(session),
        SystemClock(),
        UUIDGenerator(),
    )


async def test_import_preserves_uuid_and_movielens_mapping(migrated_settings: Settings) -> None:
    engine = create_engine(migrated_settings.database_url.get_secret_value())
    factory = async_sessionmaker(engine, expire_on_commit=False)
    try:
        async with factory() as session:
            original = (await writer(session).save((metadata(),)))[0]
            await session.execute(
                update(MovieModel).where(MovieModel.id == original.id).values(movielens_id=42)
            )
            await session.commit()
        async with factory() as session:
            updated = (await writer(session).save((metadata(title="Новое название"),)))[0]
            assert updated.id == original.id and updated.movielens_id == 42
            assert updated.title == "Новое название"
    finally:
        await engine.dispose()


async def test_failed_batch_restores_existing_movie_and_genres(migrated_settings: Settings) -> None:
    engine = create_engine(migrated_settings.database_url.get_secret_value())
    factory = async_sessionmaker(engine, expire_on_commit=False)

    class FailingRepository(SQLAlchemyMovieRepository):
        async def upsert(self, movie: Movie) -> Movie:
            result = await super().upsert(movie)
            if movie.tmdb_id == 999:
                raise RuntimeError("Failure after writing the second movie")
            return result

    try:
        async with factory() as session:
            original = (await writer(session).save((metadata(),)))[0]
        async with factory() as session:
            with pytest.raises(RuntimeError):
                await writer(session, FailingRepository(session)).save(
                    (
                        metadata(title="Should roll back", genres=(Genre(tmdb_id=99, name="New genre"),)),
                        metadata(tmdb_id=999),
                    )
                )
            assert not session.in_transaction()
        async with factory() as session:
            saved = await SQLAlchemyMovieRepository(session).get(original.id)
            assert saved.title == "Тайна Коко"
            assert {genre.tmdb_id for genre in saved.genres} == {12, 16}
            assert len((await session.scalars(select(MovieModel))).all()) == 1
            assert await session.get(GenreModel, ("movie", 99)) is None
    finally:
        await engine.dispose()


async def test_concurrent_imports_share_identity(migrated_settings: Settings) -> None:
    engine = create_engine(migrated_settings.database_url.get_secret_value())
    factory = async_sessionmaker(engine, expire_on_commit=False)

    async def save_one():
        async with factory() as session:
            return (await writer(session).save((metadata(),)))[0]

    try:
        first, second = await asyncio.gather(save_one(), save_one())
        assert first.id == second.id
        async with factory() as session:
            assert len((await session.scalars(select(MovieModel))).all()) == 1
    finally:
        await engine.dispose()


async def test_database_rejects_cross_media_genre_link(migrated_settings: Settings) -> None:
    engine = create_engine(migrated_settings.database_url.get_secret_value())
    factory = async_sessionmaker(engine, expire_on_commit=False)
    try:
        async with factory() as session:
            movie = (await writer(session).save((metadata(),)))[0]
            await writer(session).save((metadata(media_type=MediaType.TV),))
            with pytest.raises(IntegrityError):
                await session.execute(
                    movie_genres.insert().values(movie_id=movie.id, media_type="tv", genre_tmdb_id=16)
                )
            await session.rollback()
    finally:
        await engine.dispose()
