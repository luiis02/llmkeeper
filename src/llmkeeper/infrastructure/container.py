from llmkeeper.infrastructure.config import Settings
from llmkeeper.adapters.outbound.repository_memory import (
    InMemoryFieldRepository,
    InMemorySessionRepository,
)
from llmkeeper.application.use_cases import (
    CreateField,
    ValidateMessage,
    AdjustStrictness,
    CreateSession,
    AppendMessageToSession,
    GetSession,
)


class Container:
    def __init__(self, settings: Settings, embedding_provider):
        self.settings = settings
        self.embedding_provider = embedding_provider
        self.field_repository = InMemoryFieldRepository()
        self.session_repository = InMemorySessionRepository()

    def create_field_uc(self) -> CreateField:
        return CreateField(
            embeddings=self.embedding_provider,
            fields=self.field_repository,
        )

    def validate_message_uc(self) -> ValidateMessage:
        return ValidateMessage(
            embeddings=self.embedding_provider,
            fields=self.field_repository,
        )

    def adjust_strictness_uc(self) -> AdjustStrictness:
        return AdjustStrictness(fields=self.field_repository)

    def create_session_uc(self) -> CreateSession:
        return CreateSession(
            fields=self.field_repository,
            sessions=self.session_repository,
        )

    def append_message_uc(self) -> AppendMessageToSession:
        return AppendMessageToSession(
            embeddings=self.embedding_provider,
            fields=self.field_repository,
            sessions=self.session_repository,
        )

    def get_session_uc(self) -> GetSession:
        return GetSession(sessions=self.session_repository)
