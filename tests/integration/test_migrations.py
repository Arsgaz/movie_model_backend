import sqlite3

from alembic import command
from fastapi.testclient import TestClient
from sqlalchemy.engine import make_url

from adapter.database.migrations import migration_config
from config.settings import Settings
from main import create_app


def test_migration_roundtrip_and_metadata_match(settings: Settings) -> None:
    config = migration_config(settings)
    command.upgrade(config, "head")
    command.check(config)
    command.downgrade(config, "base")
    with sqlite3.connect(make_url(settings.database_url.get_secret_value()).database) as connection:
        assert connection.execute("SELECT name FROM sqlite_master WHERE name='profiles'").fetchall() == []
    command.upgrade(config, "head")
    with TestClient(create_app(settings)) as client:
        assert client.get("/ready").status_code == 200


def test_readiness_rejects_stale_migration_version(migrated_settings: Settings) -> None:
    with sqlite3.connect(make_url(migrated_settings.database_url.get_secret_value()).database) as connection:
        connection.execute("UPDATE alembic_version SET version_num='outdated'")
    with TestClient(create_app(migrated_settings)) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/ready").status_code == 503


def test_readiness_rejects_a_stamp_without_required_table(migrated_settings: Settings) -> None:
    with sqlite3.connect(make_url(migrated_settings.database_url.get_secret_value()).database) as connection:
        connection.execute("DROP TABLE profiles")
    with TestClient(create_app(migrated_settings)) as client:
        assert client.get("/ready").status_code == 503
