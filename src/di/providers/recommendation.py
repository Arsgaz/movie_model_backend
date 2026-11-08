from collections.abc import AsyncIterator

from dishka import Provider, Scope, provide

from adapter.recommendation.item_engine import ArtifactRecommendationEngine
from application.service.content_recommendation import MetadataContentRecommender
from application.service.hybrid_recommendation import HybridRecommender, RecommendationPolicy
from application.service.surprise_recommendation import SurpriseRecommender
from config.paths import PROJECT_ROOT
from config.settings import Settings
from port.content_recommendation import ContentRecommendationProvider
from port.movie_metadata import MovieMetadataProvider
from port.recommendation import RecommendationEngine


class RecommendationProvider(Provider):
    @provide(scope=Scope.APP)
    async def recommendation_engine(self, settings: Settings) -> AsyncIterator[RecommendationEngine]:
        path = settings.recommendation_model_path
        if path is not None and not path.is_absolute():
            path = PROJECT_ROOT / path
        engine = ArtifactRecommendationEngine(
            path, max_bytes=settings.recommendation_artifact_max_mb * 1024 * 1024
        )
        try:
            yield engine
        finally:
            await engine.close()

    @provide(scope=Scope.APP)
    def content_recommendations(self, metadata: MovieMetadataProvider) -> ContentRecommendationProvider:
        return MetadataContentRecommender(metadata)

    @provide(scope=Scope.APP)
    def policy(self, settings: Settings) -> RecommendationPolicy:
        return RecommendationPolicy(
            candidate_limit=settings.recommendation_candidate_limit,
            source_timeout=settings.recommendation_source_timeout,
            collaborative_weight=settings.recommendation_collaborative_weight,
            content_weight=settings.recommendation_content_weight,
        )

    surprise = provide(SurpriseRecommender, scope=Scope.APP)

    hybrid = provide(HybridRecommender, scope=Scope.APP)
