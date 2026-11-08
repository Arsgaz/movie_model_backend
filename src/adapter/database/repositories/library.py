from datetime import UTC, datetime
from uuid import UUID

from sqlalchemy import delete, select
from sqlalchemy.dialects.sqlite import insert
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from adapter.database.converters.movie import to_entity
from adapter.database.models.library import FavoriteModel, RatingModel
from adapter.database.models.movie import MovieModel
from domain.entities.favorite import Favorite
from domain.entities.movie import Movie
from domain.entities.rating import Rating


def utc(value: datetime) -> datetime:
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


class SQLAlchemyLibraryRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def put_favorite(self, favorite: Favorite) -> Favorite:
        statement = (
            insert(FavoriteModel)
            .values(
                profile_id=favorite.profile_id,
                movie_id=favorite.movie_id,
                created_at=favorite.created_at,
            )
            .on_conflict_do_nothing(index_elements=[FavoriteModel.profile_id, FavoriteModel.movie_id])
        )
        await self._session.execute(statement)
        model = await self._session.get(FavoriteModel, (favorite.profile_id, favorite.movie_id))
        return Favorite(
            profile_id=model.profile_id, movie_id=model.movie_id, created_at=utc(model.created_at)
        )

    async def put_rating(self, rating: Rating, *, overwrite: bool = True) -> Rating:
        statement = insert(RatingModel).values(
            profile_id=rating.profile_id,
            movie_id=rating.movie_id,
            value=rating.value,
            updated_at=rating.updated_at,
        )
        if overwrite:
            statement = statement.on_conflict_do_update(
                index_elements=[RatingModel.profile_id, RatingModel.movie_id],
                set_={"value": statement.excluded.value, "updated_at": statement.excluded.updated_at},
            )
        else:
            statement = statement.on_conflict_do_nothing(
                index_elements=[RatingModel.profile_id, RatingModel.movie_id]
            )
        await self._session.execute(statement)
        model = (
            await self._session.scalars(
                select(RatingModel)
                .where(
                    RatingModel.profile_id == rating.profile_id,
                    RatingModel.movie_id == rating.movie_id,
                )
                .execution_options(populate_existing=True)
            )
        ).one()
        return Rating(
            profile_id=model.profile_id,
            movie_id=model.movie_id,
            value=model.value,
            updated_at=utc(model.updated_at),
        )

    async def remove_favorite(self, *, profile_id: UUID, movie_id: UUID) -> None:
        await self._session.execute(
            delete(FavoriteModel).where(
                FavoriteModel.profile_id == profile_id,
                FavoriteModel.movie_id == movie_id,
            )
        )

    async def remove_rating(self, *, profile_id: UUID, movie_id: UUID) -> None:
        await self._session.execute(
            delete(RatingModel).where(
                RatingModel.profile_id == profile_id,
                RatingModel.movie_id == movie_id,
            )
        )

    async def favorites(self, profile_id: UUID) -> list[tuple[Favorite, Movie]]:
        rows = (
            await self._session.execute(
                select(FavoriteModel, MovieModel)
                .join(
                    MovieModel,
                    FavoriteModel.movie_id == MovieModel.id,
                )
                .where(FavoriteModel.profile_id == profile_id)
                .options(selectinload(MovieModel.genres))
                .order_by(
                    FavoriteModel.created_at.desc(),
                    MovieModel.id,
                )
            )
        ).all()
        return [
            (
                Favorite(profile_id=f.profile_id, movie_id=f.movie_id, created_at=utc(f.created_at)),
                to_entity(m),
            )
            for f, m in rows
        ]

    async def ratings(self, profile_id: UUID) -> list[tuple[Rating, Movie]]:
        rows = (
            await self._session.execute(
                select(RatingModel, MovieModel)
                .join(
                    MovieModel,
                    RatingModel.movie_id == MovieModel.id,
                )
                .where(RatingModel.profile_id == profile_id)
                .options(selectinload(MovieModel.genres))
                .order_by(
                    RatingModel.updated_at.desc(),
                    MovieModel.id,
                )
            )
        ).all()
        return [
            (
                Rating(
                    profile_id=r.profile_id, movie_id=r.movie_id, value=r.value, updated_at=utc(r.updated_at)
                ),
                to_entity(m),
            )
            for r, m in rows
        ]
