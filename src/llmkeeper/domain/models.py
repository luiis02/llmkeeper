from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from uuid import uuid4
import numpy as np


Vector = np.ndarray


@dataclass(frozen=True)
class Embedding:
    vector: Vector
    dim: int

    @staticmethod
    def from_list(values: list[float]) -> "Embedding":
        arr = np.asarray(values, dtype=np.float32)
        return Embedding(vector=arr, dim=arr.shape[0])

    def to_list(self) -> list[float]:
        return self.vector.tolist()

    def cosine_similarity(self, other: "Embedding") -> float:
        a, b = self.vector, other.vector
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na == 0 or nb == 0:
            return 0.0
        return float(np.dot(a, b) / (na * nb))


@dataclass(frozen=True)
class Anchor:
    text: str
    embedding: Embedding


class StrictnessLevel(str, Enum):
    MUY_ESTRICTO = "muy_estricto"
    ESTRICTO = "estricto"
    MODERADO = "moderado"
    PERMISIVO = "permisivo"
    MUY_PERMISIVO = "muy_permisivo"

    @property
    def threshold(self) -> float:
        return {
            StrictnessLevel.MUY_ESTRICTO: 0.32,
            StrictnessLevel.ESTRICTO: 0.27,
            StrictnessLevel.MODERADO: 0.22,
            StrictnessLevel.PERMISIVO: 0.17,
            StrictnessLevel.MUY_PERMISIVO: 0.12,
        }[self]


@dataclass(frozen=True)
class Strictness:
    threshold: float
    preset: StrictnessLevel | None = None

    @staticmethod
    def from_preset(level: StrictnessLevel) -> "Strictness":
        return Strictness(threshold=level.threshold, preset=level)

    @staticmethod
    def from_custom(value: float) -> "Strictness":
        if not (0.0 <= value <= 1.0):
            raise ValueError("threshold must be in [0, 1]")
        return Strictness(threshold=value, preset=None)


@dataclass
class TopicField:
    field_id: str
    system_prompt: str
    anchors: list[Anchor]
    centroid: Embedding
    dispersion: float
    strictness: Strictness
    alpha: float = 0.7
    top_k: int = 3
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @staticmethod
    def new_id() -> str:
        return str(uuid4())


@dataclass(frozen=True)
class ValidationResult:
    is_valid: bool
    score: float
    threshold: float
    nearest_anchor_text: str


class SessionStatus(str, Enum):
    ON_TOPIC = "on_topic"
    WARNING = "warning"
    OFF_TOPIC = "off_topic"


@dataclass
class MessageScore:
    text: str
    score: float
    is_valid: bool
    threshold: float = 0.0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class Session:
    session_id: str
    field_id: str
    patience: int
    rolling_window: int
    history: list[MessageScore] = field(default_factory=list)
    consecutive_off: int = 0
    status: SessionStatus = SessionStatus.ON_TOPIC

    @staticmethod
    def new_id() -> str:
        return str(uuid4())

    def rolling_average(self) -> float:
        if not self.history:
            return 1.0
        window = self.history[-self.rolling_window:]
        return sum(m.score for m in window) / len(window)
