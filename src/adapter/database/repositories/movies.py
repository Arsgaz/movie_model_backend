from dataclasses import replace
from uuid import UUID

from sqlalchemy import delete, func, select
from sqlalchemy.dialects.sqlite import insert
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from adapter.database.converters.movie import to_entity
from adapter.database.models.movie import GenreModel, MovieModel, movie_genres
from domain.entities.movie import Movie
from domain.value_objects.media_type import MediaType


class SQLAlchemyMovieRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def upsert(self, movie: Movie) -> Movie:
        statement = insert(MovieModel).values(
            id=movie.id,
            tmdb_id=movie.tmdb_id,
            media_type=movie.media_type.value,
            movielens_id=movie.movielens_id,
            title=movie.title,
            original_title=movie.original_title,
            search_text=f"{movie.title}\n{movie.original_title}".casefold(),
            overview=movie.overview,
            poster_path=movie.poster_path,
            backdrop_path=movie.backdrop_path,
            release_date=movie.release_date,
            vote_average=movie.vote_average,
            vote_count=movie.vote_count,
            popularity=movie.popularity,
            updated_at=movie.updated_at,
        )
        # Reimport updates metadata, preserving our identity and independently maintained ML mapping.
        updates = {
            column.name: statement.excluded[column.name]
            for column in MovieModel.__table__.columns
            if column.name not in {"id", "media_type", "tmdb_id", "movielens_id"}
        }
        statement = statement.on_conflict_do_update(
            index_elements=[MovieModel.media_type, MovieModel.tmdb_id],
            set_=updates,
        ).returning(MovieModel.id, MovieModel.movielens_id)
        movie_id, movielens_id = (await self._session.execute(statement)).one()
        await self._session.execute(delete(movie_genres).where(movie_genres.c.movie_id == movie_id))
        genres = {genre.tmdb_id: genre for genre in movie.genres}
        for genre in genres.values():
            upsert_genre = insert(GenreModel).values(
                media_type=movie.media_type.value,
                tmdb_id=genre.tmdb_id,
                name=genre.name,
            )
            await self._session.execute(
                upsert_genre.on_conflict_do_update(
                    index_elements=[GenreModel.media_type, GenreModel.tmdb_id],
                    set_={"name": upsert_genre.excluded.name},
                )
            )
            await self._session.execute(
                insert(movie_genres).values(
                    movie_id=movie_id,
                    media_type=movie.media_type.value,
                    genre_tmdb_id=genre.tmdb_id,
                )
            )
        return replace(
            movie,
            id=movie_id,
            movielens_id=movielens_id,
            genres=tuple(sorted(genres.values(), key=lambda genre: genre.tmdb_id)),
        )

    async def get(self, movie_id: UUID) -> Movie | None:
        statement = (
            select(MovieModel).where(MovieModel.id == movie_id).options(selectinload(MovieModel.genres))
        )
        model = (
            await self._session.scalars(statement.execution_options(populate_existing=True))
        ).one_or_none()
        return to_entity(model) if model is not None else None

    async def get_by_tmdb(self, *, media_type: MediaType, tmdb_id: int) -> Movie | None:
        statement = (
            select(MovieModel)
            .where(
                MovieModel.media_type == media_type.value,
                MovieModel.tmdb_id == tmdb_id,
            )
            .options(selectinload(MovieModel.genres))
        )
        model = (await self._session.scalars(statement)).one_or_none()
        return to_entity(model) if model is not None else None

    async def list(
        self,
        *,
        query: str,
        media_type: MediaType,
        genre_id: int | None,
        page: int,
        page_size: int,
    ) -> tuple[list[Movie], int]:
        criteria = [MovieModel.media_type == media_type.value]
        if query:
            criteria.append(MovieModel.search_text.contains(query.casefold(), autoescape=True))
        if genre_id is not None:
            criteria.append(MovieModel.genres.any(GenreModel.tmdb_id == genre_id))
        total = (
            await self._session.scalar(select(func.count()).select_from(MovieModel).where(*criteria))
        ) or 0
        statement = (
            select(MovieModel)
            .where(*criteria)
            .options(selectinload(MovieModel.genres))
            .order_by(MovieModel.popularity.desc(), MovieModel.id)
            .offset((page - 1) * page_size)
            .limit(page_size)
        )
        models = (await self._session.scalars(statement)).all()
        return [to_entity(model) for model in models], total
