from datetime import datetime
from uuid import UUID

from sqlalchemy import CheckConstraint, DateTime, ForeignKey, Integer, Uuid
from sqlalchemy.orm import Mapped, mapped_column

from adapter.database.models.base import Base


class FavoriteModel(Base):
    __tablename__ = "favorites"
    profile_id: Mapped[UUID] = mapped_column(
        Uuid, ForeignKey("profiles.id", ondelete="CASCADE"), primary_key=True
    )
    movie_id: Mapped[UUID] = mapped_column(
        Uuid, ForeignKey("movies.id", ondelete="CASCADE"), primary_key=True
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class RatingModel(Base):
    __tablename__ = "ratings"
    __table_args__ = (
        CheckConstraint("value BETWEEN 1 AND 5 AND value = CAST(value AS INTEGER)", name="value_range"),
    )
    profile_id: Mapped[UUID] = mapped_column(
        Uuid, ForeignKey("profiles.id", ondelete="CASCADE"), primary_key=True
    )
    movie_id: Mapped[UUID] = mapped_column(
        Uuid, ForeignKey("movies.id", ondelete="CASCADE"), primary_key=True
    )
    value: Mapped[int] = mapped_column(Integer, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
