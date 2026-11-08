from pathlib import Path

import pytest
from alembic import command

from adapter.database.migrations import migration_config
from config.settings import Settings


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        _env_file=None,
        tmdb_api_key=None,
        database_url=f"sqlite+aiosqlite:///{tmp_path / 'test.sqlite3'}",
        cors_origins=["http://localhost:3000"],
    )


@pytest.fixture
def migrated_settings(settings: Settings) -> Settings:
    command.upgrade(migration_config(settings), "head")
    return settings
