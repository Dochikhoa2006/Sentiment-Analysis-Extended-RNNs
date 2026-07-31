"""FastText training and fixed-length review vectorization."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from sentiment_analyzer.text import tokenize


class _TokenizedCorpus:
    def __init__(self, texts: Sequence[str]) -> None:
        self.texts = texts

    def __iter__(self):
        for text in self.texts:
            yield tokenize(text)


class EmbeddingVectorizer:
    """Train FastText embeddings and convert reviews to padded float32 matrices."""

    def __init__(
        self,
        vector_size: int = 130,
        *,
        workers: int = 1,
        seed: int = 100,
        epochs: int = 5,
    ) -> None:
        if vector_size < 1:
            raise ValueError("vector_size must be positive")
        self.vector_size = vector_size
        # Kept for compatibility with artifacts produced by the original project.
        self.token_vector_dimension = vector_size
        self.workers = workers
        self.seed = seed
        self.epochs = epochs
        self.model: Any | None = None

    @property
    def dimension(self) -> int:
        return int(getattr(self, "vector_size", self.token_vector_dimension))

    def fit(self, texts: Sequence[str]) -> EmbeddingVectorizer:
        """Fit a FastText model on a repeatable sequence of review strings."""

        from gensim.models import FastText

        if len(texts) == 0:
            raise ValueError("cannot train embeddings on an empty corpus")
        corpus = _TokenizedCorpus(texts)
        self.model = FastText(
            sentences=corpus,
            vector_size=self.dimension,
            workers=self.workers,
            seed=self.seed,
            epochs=self.epochs,
        )
        return self

    def write_corpus(self, texts: Iterable[str], destination: Path) -> None:
        """Persist normalized text for inspection without using it as model state."""

        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", encoding="utf-8") as stream:
            for text in texts:
                stream.write(" ".join(tokenize(text)) + "\n")

    def transform_one(self, text: str, sequence_length: int = 150) -> np.ndarray:
        """Return one zero-padded ``[sequence_length, vector_size]`` matrix."""

        if self.model is None:
            raise RuntimeError("the vectorizer must be fitted before transform")
        if sequence_length < 1:
            raise ValueError("sequence_length must be positive")

        matrix = np.zeros((sequence_length, self.dimension), dtype=np.float32)
        for index, token in enumerate(tokenize(text)[:sequence_length]):
            matrix[index] = np.asarray(self.model.wv[token], dtype=np.float32)
        return matrix

    def transform(self, texts: Iterable[str], sequence_length: int = 150) -> np.ndarray:
        matrices = [self.transform_one(text, sequence_length) for text in texts]
        if not matrices:
            return np.empty((0, sequence_length, self.dimension), dtype=np.float32)
        return np.stack(matrices)

    def save(self, destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, destination)

    @classmethod
    def load(cls, source: Path) -> EmbeddingVectorizer:
        vectorizer = joblib.load(source)
        if not hasattr(vectorizer, "transform"):
            raise TypeError(f"{source} does not contain a compatible vectorizer")
        return vectorizer

    # Original API aliases keep historical artifacts usable after the refactor.
    def get_vector_dimension(self) -> int:
        return self.dimension

    def tokenize(self, text: str) -> list[str]:
        return tokenize(text)

    def build_vocabulary(self, texts: Sequence[str]) -> None:
        self.fit(texts)

    def vectorize(self, text: str, model_input_length: int) -> list[np.ndarray]:
        return list(self.transform_one(text, model_input_length))

    def X_setup(self, review_text: Iterable[str], input_length: int) -> np.ndarray:
        return self.transform(review_text, input_length)


Embedding_Vector = EmbeddingVectorizer
