from datetime import datetime
from uuid import UUID

from sqlalchemy import CheckConstraint, DateTime, String, Uuid
from sqlalchemy.orm import Mapped, mapped_column

from adapter.database.models.base import Base


class ProfileModel(Base):
    __tablename__ = "profiles"
    __table_args__ = (CheckConstraint("length(trim(name)) BETWEEN 1 AND 80", name="name_length"),)

    id: Mapped[UUID] = mapped_column(Uuid, primary_key=True)
    name: Mapped[str] = mapped_column(String(80), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
