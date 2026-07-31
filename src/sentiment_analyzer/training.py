"""Embedding and final-model training workflows."""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from sentiment_analyzer.batching import ReviewSequence
from sentiment_analyzer.config import ModelConfig
from sentiment_analyzer.data import load_dataset
from sentiment_analyzer.embeddings import EmbeddingVectorizer
from sentiment_analyzer.modeling import build_model
from sentiment_analyzer.serialization import load_vectorizer


def train_embeddings(
    dataset_path: Path,
    destination: Path,
    *,
    corpus_path: Path | None = None,
    vector_size: int = 130,
    workers: int | None = None,
    seed: int = 100,
    epochs: int = 5,
) -> EmbeddingVectorizer:
    dataset = load_dataset(dataset_path)
    texts = dataset["review"].astype(str).tolist()
    vectorizer = EmbeddingVectorizer(
        vector_size,
        workers=workers or max(1, (os.cpu_count() or 2) // 2),
        seed=seed,
        epochs=epochs,
    ).fit(texts)
    if corpus_path is not None:
        vectorizer.write_corpus(texts, corpus_path)
    vectorizer.save(destination)
    return vectorizer


def _class_weights(labels: np.ndarray) -> dict[int, float]:
    from sklearn.utils.class_weight import compute_class_weight

    classes = np.unique(labels)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=labels)
    return {int(label): float(weight) for label, weight in zip(classes, weights, strict=True)}


def train_final_model(
    dataset_path: Path,
    vectorizer_path: Path,
    destination: Path,
    *,
    config: ModelConfig | None = None,
    batch_size: int = 128,
    epochs: int = 5,
    validation_fraction: float = 0.1,
    balance_classes: bool = True,
) -> tuple[Any, dict[str, Any]]:
    from sklearn.model_selection import train_test_split
    from tensorflow.keras.callbacks import EarlyStopping

    model_config = config or ModelConfig()
    dataset = load_dataset(dataset_path)
    vectorizer = load_vectorizer(vectorizer_path)
    if vectorizer.dimension != model_config.embedding_dimension:
        raise ValueError(
            "vectorizer dimension does not match model config: "
            f"{vectorizer.dimension} != {model_config.embedding_dimension}"
        )

    texts = dataset["review"].astype(str).to_numpy()
    labels = dataset["star"].astype(np.int64).to_numpy()
    train_texts, validation_texts, train_labels, validation_labels = train_test_split(
        texts,
        labels,
        test_size=validation_fraction,
        random_state=model_config.seed,
        stratify=labels,
    )
    train_batches = ReviewSequence(
        train_texts,
        vectorizer,
        model_config.sequence_length,
        labels=train_labels,
        batch_size=batch_size,
        shuffle=True,
        seed=model_config.seed,
    )
    validation_batches = ReviewSequence(
        validation_texts,
        vectorizer,
        model_config.sequence_length,
        labels=validation_labels,
        batch_size=batch_size,
    )

    model = build_model(model_config)
    history = model.fit(
        train_batches,
        validation_data=validation_batches,
        epochs=epochs,
        class_weight=_class_weights(train_labels) if balance_classes else None,
        callbacks=[EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True)],
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    model.save(destination)

    metadata = {
        "created_at": datetime.now(UTC).isoformat(),
        "config": model_config.to_dict(),
        "training_rows": int(len(train_labels)),
        "validation_rows": int(len(validation_labels)),
        "class_weights_enabled": balance_classes,
        "history": {
            key: [float(value) for value in values] for key, values in history.history.items()
        },
    }
    metadata_path = destination.with_suffix(".metadata.json")
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return model, metadata
