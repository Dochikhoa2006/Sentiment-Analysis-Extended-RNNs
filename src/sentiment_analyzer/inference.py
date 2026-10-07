"""Stable inference API for application-review sentiment."""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass
from itertools import islice
from pathlib import Path
from typing import Any

import numpy as np

from sentiment_analyzer.embeddings import EmbeddingVectorizer
from sentiment_analyzer.serialization import artifact_sha256, load_model, load_vectorizer

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
        sequence_length: int | None = None,
    ) -> SentimentPredictor:
        metadata_path = model_path.with_suffix(".metadata.json")
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            hashes = metadata.get("artifacts", {})
            for name, path in (("model", model_path), ("vectorizer", vectorizer_path)):
                expected = hashes.get(f"{name}_sha256")
                if expected is not None and artifact_sha256(path) != expected:
                    raise ValueError(f"{name} artifact does not match training metadata: {path}")
            config = metadata.get("config", {})
            trained_length = config.get("sequence_length")
            if trained_length is not None and sequence_length not in (None, trained_length):
                raise ValueError("sequence_length does not match the trained model")
            if sequence_length is None:
                sequence_length = trained_length
        vectorizer = load_vectorizer(vectorizer_path)
        if metadata_path.exists():
            dimension = config.get("embedding_dimension")
            if dimension is not None and vectorizer.dimension != dimension:
                raise ValueError("vectorizer dimension does not match the trained model")
        model = load_model(model_path)
        effective_length = sequence_length or 150
        keras_model = getattr(model, "model", model)
        input_shape = getattr(keras_model, "input_shape", None)
        if input_shape is not None and (
            len(input_shape) != 3
            or input_shape[1] not in (None, effective_length)
            or input_shape[2] not in (None, vectorizer.dimension)
        ):
            raise ValueError("model input shape does not match the vectorizer and sequence length")
        output_shape = getattr(keras_model, "output_shape", None)
        if output_shape is not None and (
            len(output_shape) != 2 or output_shape[1] != len(SENTIMENT_LABELS)
        ):
            raise ValueError("model output shape does not match the sentiment labels")
        return cls(vectorizer, model, sequence_length=effective_length)

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
            if (
                np.any(scores < 0)
                or np.any(scores > 1)
                or not np.allclose(scores.sum(axis=1), 1.0, rtol=1e-5, atol=1e-5)
            ):
                raise ValueError("model returned invalid probability distributions")
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
