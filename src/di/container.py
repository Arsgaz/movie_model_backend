from dishka import AsyncContainer, Provider, make_async_container

from config.settings import Settings
from di.providers.configuration import ConfigurationProvider
from di.providers.database import DatabaseProvider
from di.providers.interactors import InteractorProvider
from di.providers.metadata import MetadataProvider
from di.providers.recommendation import RecommendationProvider
from di.providers.repositories import RepositoryProvider
from di.providers.system import SystemProvider


def create_container(settings: Settings, *additional_providers: Provider) -> AsyncContainer:
    return make_async_container(
        ConfigurationProvider(settings),
        DatabaseProvider(),
        RepositoryProvider(),
        SystemProvider(),
        MetadataProvider(),
        RecommendationProvider(),
        InteractorProvider(),
        *additional_providers,
    )
