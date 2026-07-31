"""Memory-bounded batches for vectorizing reviews during model execution."""

from __future__ import annotations

from typing import Any

import numpy as np

from sentiment_analyzer.embeddings import EmbeddingVectorizer


def _sequence_base():
    try:
        from tensorflow.keras.utils import Sequence
    except ImportError as exc:
        raise RuntimeError("Install ML dependencies with: pip install -e '.[ml]'") from exc
    return Sequence


class ReviewSequence(_sequence_base()):
    """Vectorize only the current batch instead of allocating the full 15+ GB tensor."""

    def __init__(
        self,
        texts: Any,
        vectorizer: EmbeddingVectorizer,
        sequence_length: int,
        *,
        labels: Any | None = None,
        batch_size: int = 128,
        shuffle: bool = False,
        seed: int = 100,
    ) -> None:
        super().__init__()
        self.texts = np.asarray(texts, dtype=object)
        self.labels = None if labels is None else np.asarray(labels, dtype=np.int64)
        if self.labels is not None and len(self.texts) != len(self.labels):
            raise ValueError("texts and labels must contain the same number of rows")
        self.vectorizer = vectorizer
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.rng = np.random.default_rng(seed)
        self.indices = np.arange(len(self.texts))
        self.on_epoch_end()

    def __len__(self) -> int:
        return int(np.ceil(len(self.indices) / self.batch_size))

    def __getitem__(self, index: int):
        selected = self.indices[index * self.batch_size : (index + 1) * self.batch_size]
        features = self.vectorizer.transform(self.texts[selected], self.sequence_length)
        if self.labels is None:
            return features
        return features, self.labels[selected]

    def on_epoch_end(self) -> None:
        if self.shuffle:
            self.rng.shuffle(self.indices)
