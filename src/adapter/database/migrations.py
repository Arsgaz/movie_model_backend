from alembic.config import Config

from config.paths import PROJECT_ROOT
from config.settings import Settings


def migration_config(settings: Settings) -> Config:
    config = Config(str(PROJECT_ROOT / "alembic.ini"))
    config.attributes["database_url"] = settings.database_url.get_secret_value()
    return config
