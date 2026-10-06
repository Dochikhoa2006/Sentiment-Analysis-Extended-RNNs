"""Stable inference API for application-review sentiment."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass
from itertools import islice
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
        return next(self.predict_many([review], batch_size=1))

    def predict_many(
        self, reviews: Iterable[str], *, batch_size: int = 128
    ) -> Iterator[Prediction]:
        """Predict reviews in input order, vectorizing at most one batch at a time."""

        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        source = iter(reviews)
        while batch := list(islice(source, batch_size)):
            if any(not isinstance(review, str) or not review.strip() for review in batch):
                raise ValueError("review must not be empty")
            features = self.vectorizer.transform(batch, self.sequence_length)
            keras_model = getattr(self.model, "model", self.model)
            scores = np.asarray(keras_model.predict(features, verbose=0))
            if scores.shape != (len(batch), len(SENTIMENT_LABELS)):
                raise ValueError("model returned an unexpected prediction shape")
            if not np.all(np.isfinite(scores)):
                raise ValueError("model returned non-finite prediction scores")
            for probabilities in scores:
                yield self._result(probabilities)

    @staticmethod
    def _result(probabilities: np.ndarray) -> Prediction:
        class_index = int(np.argmax(probabilities))
        values = tuple(float(value) for value in probabilities)
        return Prediction(
            rating=class_index + 1,
            sentiment=SENTIMENT_LABELS[class_index],
            confidence=values[class_index],
            probabilities=values,
        )
