from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from dishka import AsyncContainer
from dishka.integrations.fastapi import setup_dishka
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config.settings import Settings
from di.container import create_container
from presentation.http.exception_handler import setup_exception_handlers
from presentation.http.routers.catalog import router as catalog_router
from presentation.http.routers.discovery import router as discovery_router
from presentation.http.routers.health import router as health_router
from presentation.http.routers.library import router as library_router
from presentation.http.routers.movies import router as movies_router
from presentation.http.routers.profiles import router as profiles_router
from presentation.http.routers.recommendations import router as recommendations_router


def create_app(settings: Settings | None = None, *, container: AsyncContainer | None = None) -> FastAPI:
    settings = settings if settings is not None else Settings()
    container = container if container is not None else create_container(settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        try:
            yield
        finally:
            await container.close()

    app = FastAPI(title=settings.app_name, version="0.1.0", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["GET", "POST", "PUT", "DELETE"],
        allow_headers=["Content-Type"],
        allow_credentials=False,
    )
    setup_exception_handlers(app)
    app.include_router(health_router)
    app.include_router(profiles_router)
    app.include_router(library_router)
    app.include_router(recommendations_router)
    app.include_router(movies_router)
    app.include_router(catalog_router)
    app.include_router(discovery_router)
    setup_dishka(container=container, app=app)
    return app
