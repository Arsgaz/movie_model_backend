"""Create profiles.

Revision ID: 0001_profiles
"""

import sqlalchemy as sa
from alembic import op

revision = "0001_profiles"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "profiles",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("name", sa.String(length=80), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint("length(trim(name)) BETWEEN 1 AND 80", name=op.f("ck_profiles_name_length")),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_profiles")),
    )


def downgrade() -> None:
    op.drop_table("profiles")
