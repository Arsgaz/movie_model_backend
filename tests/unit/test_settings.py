import pytest
from pydantic import ValidationError

from config.settings import Settings


def test_database_secret_is_not_exposed_in_settings_repr() -> None:
    settings = Settings(_env_file=None, database_url="sqlite+aiosqlite:///private-name.sqlite3")
    assert "private-name" not in repr(settings)


@pytest.mark.parametrize("value", ["invalid", "sqlite:///db.sqlite3", "postgresql+asyncpg://localhost/db"])
def test_settings_reject_unsupported_database_drivers(value: str) -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, database_url=value)


def test_cors_requires_explicit_origins() -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, cors_origins=["*"])


def test_empty_runtime_model_path_disables_artifact(monkeypatch) -> None:
    monkeypatch.setenv("MOVIE_RECOMMENDATION_MODEL_PATH", "")
    settings = Settings(_env_file=None)
    assert settings.recommendation_model_path is None
