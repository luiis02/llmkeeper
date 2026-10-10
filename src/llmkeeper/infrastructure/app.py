from contextlib import asynccontextmanager
from fastapi import FastAPI, Depends
from llmkeeper.infrastructure.config import Settings
from llmkeeper.infrastructure.container import Container
from llmkeeper.adapters.outbound.embedding_sentence_transformers import (
    SentenceTransformerEmbeddingAdapter,
)
from llmkeeper.adapters.inbound.api import create_router


_container: Container | None = None


def get_container() -> Container:
    if _container is None:
        raise RuntimeError("Container not initialized")
    return _container


def create_app(settings: Settings | None = None) -> FastAPI:
    if settings is None:
        settings = Settings()

    global _container

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        global _container
        embedding_provider = SentenceTransformerEmbeddingAdapter(
            model_name=settings.embedding_model_name
        )
        _container = Container(settings=settings, embedding_provider=embedding_provider)

        router = create_router(
            create_field_uc=_container.create_field_uc(),
            validate_message_uc=_container.validate_message_uc(),
            adjust_strictness_uc=_container.adjust_strictness_uc(),
            create_session_uc=_container.create_session_uc(),
            append_message_uc=_container.append_message_uc(),
            get_session_uc=_container.get_session_uc(),
        )
        app.include_router(router)

        yield
        _container = None

    app = FastAPI(
        title="LLMKeeper",
        description="Local topic-drift guard for LLM conversations",
        version="0.1.0",
        lifespan=lifespan,
    )

    return app
