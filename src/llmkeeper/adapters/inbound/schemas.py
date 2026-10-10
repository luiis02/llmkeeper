from pydantic import BaseModel, Field
from llmkeeper.domain.models import StrictnessLevel


class CreateFieldRequest(BaseModel):
    system_prompt: str
    extra_topics: list[str] | None = None
    strictness_preset: StrictnessLevel | None = None
    strictness_value: float | None = Field(None, ge=0.0, le=1.0)


class FieldResponse(BaseModel):
    field_id: str
    system_prompt: str
    num_anchors: int
    dispersion: float
    threshold: float
    preset: str | None
    alpha: float
    top_k: int


class ValidationRequestSimple(BaseModel):
    text: str


class ValidationResponse(BaseModel):
    is_valid: bool
    score: float
    threshold: float
    nearest_anchor_text: str


class AdjustStrictnessRequest(BaseModel):
    preset: StrictnessLevel | None = None
    custom_value: float | None = Field(None, ge=0.0, le=1.0)


class CreateSessionRequest(BaseModel):
    patience: int = 3
    rolling_window: int = 5


class SessionResponse(BaseModel):
    session_id: str
    field_id: str
    patience: int
    rolling_window: int
    status: str
    rolling_average: float
    num_messages: int


class MessageRequest(BaseModel):
    text: str


class MessageResponseWithSession(BaseModel):
    score: float
    is_valid: bool
    threshold: float
    rolling_average: float
    status: str
    consecutive_off: int
