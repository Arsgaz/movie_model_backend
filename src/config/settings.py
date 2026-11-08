from pathlib import Path

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy.engine import make_url

from config.paths import PROJECT_ROOT


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="MOVIE_", env_file=PROJECT_ROOT / ".env", extra="ignore")

    recommendation_model_path: Path | None = None
    recommendation_artifact_max_mb: int = Field(default=512, ge=1, le=2048)
    recommendation_candidate_limit: int = Field(default=40, ge=20, le=100)
    recommendation_source_timeout: float = Field(default=10, ge=1, le=20)
    recommendation_collaborative_weight: float = Field(default=1, gt=0, le=10, allow_inf_nan=False)
    recommendation_content_weight: float = Field(default=1, gt=0, le=10, allow_inf_nan=False)
    app_name: str = "Movie Model API"
    database_url: SecretStr = SecretStr("sqlite+aiosqlite:///./data/movie_model.sqlite3")
    tmdb_api_key: SecretStr | None = None
    tmdb_cache_ttl: float = Field(default=300, ge=1, le=86400)
    tmdb_cache_entries: int = Field(default=128, ge=1, le=1000)
    tmdb_timeout: float = Field(default=8, ge=1, le=30)
    cors_origins: list[str] = Field(
        default_factory=lambda: [
            "http://localhost:3000",
            "http://127.0.0.1:3000",
            "http://127.0.0.1:3100",
        ]
    )

    @field_validator("recommendation_model_path", mode="before")
    @classmethod
    def empty_model_path(cls, value: object) -> object:
        # An empty runtime variable explicitly disables the optional artifact.
        return None if isinstance(value, str) and not value.strip() else value

    @field_validator("database_url")
    @classmethod
    def validate_database_url(cls, value: SecretStr) -> SecretStr:
        try:
            url = make_url(value.get_secret_value())
            if url.drivername != "sqlite+aiosqlite" or not url.database:
                raise ValueError
        except Exception as error:
            raise ValueError("Use a SQLite URL with the sqlite+aiosqlite driver") from error
        return value

    @field_validator("cors_origins")
    @classmethod
    def validate_origins(cls, value: list[str]) -> list[str]:
        if "*" in value:
            raise ValueError("List explicit allowed frontend origins")
        return value
