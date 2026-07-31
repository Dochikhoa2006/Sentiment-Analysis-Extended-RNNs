"""Stable inference API for application-review sentiment."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from sentiment_analyzer.embeddings import EmbeddingVectorizer
from sentiment_analyzer.serialization import load_model, load_vectorizer

SENTIMENT_LABELS = (
    "strongly dissatisfied",
    "dissatisfied",
    "neutral",
    "satisfied",
    "strongly satisfied",
)


@dataclass(frozen=True)
class Prediction:
    rating: int
    sentiment: str
    confidence: float
    probabilities: tuple[float, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class SentimentPredictor:
    def __init__(
        self,
        vectorizer: EmbeddingVectorizer,
        model: Any,
        *,
        sequence_length: int = 150,
    ) -> None:
        self.vectorizer = vectorizer
        self.model = model
        self.sequence_length = sequence_length

    @classmethod
    def from_artifacts(
        cls,
        vectorizer_path: Path,
        model_path: Path,
        *,
        sequence_length: int = 150,
    ) -> SentimentPredictor:
        return cls(
            load_vectorizer(vectorizer_path),
            load_model(model_path),
            sequence_length=sequence_length,
        )

    def predict(self, review: str) -> Prediction:
        if not review.strip():
            raise ValueError("review must not be empty")
        features = self.vectorizer.transform([review], self.sequence_length)
        keras_model = getattr(self.model, "model", self.model)
        probabilities = np.asarray(keras_model.predict(features, verbose=0))[0]
        class_index = int(np.argmax(probabilities))
        values = tuple(float(value) for value in probabilities)
        return Prediction(
            rating=class_index + 1,
            sentiment=SENTIMENT_LABELS[class_index],
            confidence=values[class_index],
            probabilities=values,
        )
