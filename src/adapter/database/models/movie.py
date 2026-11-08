from datetime import date, datetime
from uuid import UUID

from sqlalchemy import (
    CheckConstraint,
    Column,
    Date,
    DateTime,
    Float,
    ForeignKeyConstraint,
    Integer,
    String,
    Table,
    Text,
    UniqueConstraint,
    Uuid,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from adapter.database.models.base import Base


class GenreModel(Base):
    __tablename__ = "genres"
    __table_args__ = (
        CheckConstraint("tmdb_id > 0", name="tmdb_id_positive"),
        CheckConstraint("media_type IN ('movie', 'tv')", name="media_type"),
    )
    media_type: Mapped[str] = mapped_column(String(5), primary_key=True)
    tmdb_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(120), nullable=False)


movie_genres = Table(
    "movie_genres",
    Base.metadata,
    Column("movie_id", Uuid, primary_key=True),
    Column("media_type", String(5), nullable=False),
    Column("genre_tmdb_id", Integer, primary_key=True),
    ForeignKeyConstraint(["movie_id", "media_type"], ["movies.id", "movies.media_type"], ondelete="CASCADE"),
    ForeignKeyConstraint(
        ["media_type", "genre_tmdb_id"], ["genres.media_type", "genres.tmdb_id"], ondelete="CASCADE"
    ),
)


class MovieModel(Base):
    __tablename__ = "movies"
    __table_args__ = (
        UniqueConstraint("media_type", "tmdb_id"),
        UniqueConstraint("id", "media_type"),
        CheckConstraint("tmdb_id > 0", name="tmdb_id_positive"),
        CheckConstraint("media_type IN ('movie', 'tv')", name="media_type"),
        CheckConstraint("length(trim(title)) > 0", name="title_not_empty"),
        CheckConstraint("vote_average BETWEEN 0 AND 10", name="rating_range"),
        CheckConstraint("vote_count >= 0 AND popularity >= 0", name="nonnegative_metrics"),
        CheckConstraint(
            "movielens_id IS NULL OR (media_type = 'movie' AND movielens_id > 0)", name="movielens_mapping"
        ),
    )

    id: Mapped[UUID] = mapped_column(Uuid, primary_key=True)
    tmdb_id: Mapped[int] = mapped_column(Integer, nullable=False)
    media_type: Mapped[str] = mapped_column(String(5), nullable=False)
    movielens_id: Mapped[int | None] = mapped_column(Integer, unique=True)
    title: Mapped[str] = mapped_column(String(500), nullable=False)
    original_title: Mapped[str] = mapped_column(String(500), nullable=False)
    search_text: Mapped[str] = mapped_column(Text, nullable=False)
    overview: Mapped[str] = mapped_column(Text, nullable=False)
    poster_path: Mapped[str | None] = mapped_column(String(255))
    backdrop_path: Mapped[str | None] = mapped_column(String(255))
    release_date: Mapped[date | None] = mapped_column(Date)
    vote_average: Mapped[float] = mapped_column(Float, nullable=False)
    vote_count: Mapped[int] = mapped_column(Integer, nullable=False)
    popularity: Mapped[float] = mapped_column(Float, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    genres: Mapped[list[GenreModel]] = relationship(
        secondary=movie_genres,
        viewonly=True,
        lazy="raise",
        order_by=GenreModel.tmdb_id,
    )
