"""Embedding and final-model training workflows."""

from __future__ import annotations

import json
import os
from contextlib import ExitStack
from datetime import UTC, datetime
from itertools import combinations
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np

from sentiment_analyzer.batching import ReviewSequence
from sentiment_analyzer.config import ModelConfig
from sentiment_analyzer.data import load_dataset
from sentiment_analyzer.embeddings import EmbeddingVectorizer
from sentiment_analyzer.modeling import build_model
from sentiment_analyzer.serialization import artifact_sha256, load_vectorizer


def _validate_distinct_paths(**paths: Path) -> None:
    """Reject artifact destinations that could overwrite an input or another output."""

    for (left_name, left), (right_name, right) in combinations(paths.items(), 2):
        if left.resolve() == right.resolve() or (
            left.exists() and right.exists() and left.samefile(right)
        ):
            raise ValueError(f"{left_name} and {right_name} must refer to different files")


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
    paths = {"dataset": dataset_path, "vectorizer": destination}
    if corpus_path is not None:
        paths["corpus"] = corpus_path
    _validate_distinct_paths(**paths)
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
    reuse_vectorizer: bool = False,
) -> tuple[Any, dict[str, Any]]:
    from sklearn.model_selection import train_test_split
    from tensorflow.keras.callbacks import EarlyStopping

    model_config = config or ModelConfig()
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    if batch_size < 1 or epochs < 1:
        raise ValueError("batch_size and epochs must be positive")
    metadata_path = destination.with_suffix(".metadata.json")
    _validate_distinct_paths(
        dataset=dataset_path,
        vectorizer=vectorizer_path,
        model=destination,
        metadata=metadata_path,
    )
    dataset = load_dataset(dataset_path)
    texts = dataset["review"].astype(str).to_numpy()
    labels = dataset["star"].astype(np.int64).to_numpy()
    train_texts, validation_texts, train_labels, validation_labels = train_test_split(
        texts,
        labels,
        test_size=validation_fraction,
        random_state=model_config.seed,
        stratify=labels,
    )
    if reuse_vectorizer:
        vectorizer = load_vectorizer(vectorizer_path)
        if vectorizer.dimension != model_config.embedding_dimension:
            raise ValueError(
                "vectorizer dimension does not match model config: "
                f"{vectorizer.dimension} != {model_config.embedding_dimension}"
            )
    else:
        vectorizer = EmbeddingVectorizer(
            model_config.embedding_dimension,
            workers=max(1, (os.cpu_count() or 2) // 2),
            seed=model_config.seed,
        ).fit(train_texts)
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
    with ExitStack() as stack:
        model_dir = Path(stack.enter_context(TemporaryDirectory(dir=destination.parent)))
        staged_model = model_dir / destination.name
        model.save(staged_model)
        if reuse_vectorizer:
            staged_vectorizer = vectorizer_path
        else:
            vectorizer_path.parent.mkdir(parents=True, exist_ok=True)
            vectorizer_dir = Path(
                stack.enter_context(TemporaryDirectory(dir=vectorizer_path.parent))
            )
            staged_vectorizer = vectorizer_dir / vectorizer_path.name
            vectorizer.save(staged_vectorizer)

        metadata = {
            "created_at": datetime.now(UTC).isoformat(),
            "config": model_config.to_dict(),
            "training_rows": int(len(train_labels)),
            "validation_rows": int(len(validation_labels)),
            "class_weights_enabled": balance_classes,
            "vectorizer_fitted_on_training_split": not reuse_vectorizer,
            "artifacts": {
                "model_sha256": artifact_sha256(staged_model),
                "vectorizer_sha256": artifact_sha256(staged_vectorizer),
            },
            "history": {
                key: [float(value) for value in values] for key, values in history.history.items()
            },
        }
        staged_metadata = model_dir / metadata_path.name
        staged_metadata.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

        # Publish metadata last so readers can detect any pair changed mid-publication.
        if not reuse_vectorizer:
            os.replace(staged_vectorizer, vectorizer_path)
        os.replace(staged_model, destination)
        os.replace(staged_metadata, metadata_path)
    return model, metadata
