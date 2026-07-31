"""Leakage-resistant stratified cross-validation for BiLSTM and BiGRU models."""

from __future__ import annotations

import json
import os
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from sentiment_analyzer.batching import ReviewSequence
from sentiment_analyzer.config import ModelConfig
from sentiment_analyzer.data import load_dataset
from sentiment_analyzer.embeddings import EmbeddingVectorizer
from sentiment_analyzer.modeling import build_model
from sentiment_analyzer.training import _class_weights


def confidence_interval(values: list[float], confidence: float = 0.95) -> tuple[float, float]:
    """Return the sample mean and Student-t margin of error."""

    from scipy import stats

    samples = np.asarray(values, dtype=np.float64)
    if len(samples) < 2:
        return float(samples.mean()), 0.0
    margin = stats.sem(samples) * stats.t.ppf((1 + confidence) / 2, len(samples) - 1)
    return float(samples.mean()), float(margin)


def cross_validate(
    dataset_path: Path,
    output_dir: Path,
    *,
    architectures: tuple[str, ...] = ("lstm", "gru"),
    folds: int = 5,
    epochs: int = 1,
    batch_size: int = 128,
    config: ModelConfig | None = None,
) -> dict[str, Any]:
    """Evaluate each architecture while fitting embeddings only on each training fold."""

    import tensorflow as tf
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
    from sklearn.model_selection import StratifiedKFold

    base_config = config or ModelConfig()
    dataset = load_dataset(dataset_path)
    texts = dataset["review"].astype(str).to_numpy()
    labels = dataset["star"].astype(np.int64).to_numpy()
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=base_config.seed)
    splits = list(splitter.split(texts, labels))
    results: dict[str, Any] = {}

    for architecture in architectures:
        architecture_config = replace(base_config, architecture=architecture)
        fold_metrics: list[dict[str, float]] = []
        aggregate_matrix = np.zeros(
            (architecture_config.number_of_classes, architecture_config.number_of_classes),
            dtype=np.int64,
        )

        for fold_number, (train_index, test_index) in enumerate(splits, start=1):
            train_texts, test_texts = texts[train_index], texts[test_index]
            train_labels, test_labels = labels[train_index], labels[test_index]
            vectorizer = EmbeddingVectorizer(
                architecture_config.embedding_dimension,
                workers=max(1, (os.cpu_count() or 2) // 2),
                seed=architecture_config.seed + fold_number,
            ).fit(train_texts)
            train_batches = ReviewSequence(
                train_texts,
                vectorizer,
                architecture_config.sequence_length,
                labels=train_labels,
                batch_size=batch_size,
                shuffle=True,
                seed=architecture_config.seed + fold_number,
            )
            test_batches = ReviewSequence(
                test_texts,
                vectorizer,
                architecture_config.sequence_length,
                batch_size=batch_size,
            )
            model = build_model(architecture_config)
            model.fit(
                train_batches,
                epochs=epochs,
                class_weight=_class_weights(train_labels),
                verbose=1,
            )
            predictions = np.argmax(model.predict(test_batches, verbose=0), axis=1)
            accuracy = float(accuracy_score(test_labels, predictions))
            macro_f1 = float(f1_score(test_labels, predictions, average="macro"))
            fold_metrics.append({"fold": fold_number, "accuracy": accuracy, "macro_f1": macro_f1})
            aggregate_matrix += confusion_matrix(
                test_labels,
                predictions,
                labels=np.arange(architecture_config.number_of_classes),
            )
            tf.keras.backend.clear_session()

        accuracies = [fold["accuracy"] for fold in fold_metrics]
        macro_scores = [fold["macro_f1"] for fold in fold_metrics]
        accuracy_mean, accuracy_margin = confidence_interval(accuracies)
        macro_mean, macro_margin = confidence_interval(macro_scores)
        results[architecture] = {
            "folds": fold_metrics,
            "accuracy": {"mean": accuracy_mean, "margin_95": accuracy_margin},
            "macro_f1": {"mean": macro_mean, "margin_95": macro_margin},
            "confusion_matrix": aggregate_matrix.tolist(),
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    _plot_results(results, output_dir / "model_comparison.png")
    return results


def _plot_results(results: dict[str, Any], destination: Path) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    architectures = list(results)
    column_count = len(architectures) + 1
    figure, axes = plt.subplots(1, column_count, figsize=(6 * column_count, 5))
    if not isinstance(axes, np.ndarray):
        axes = np.asarray([axes])

    for axis, architecture in zip(axes, architectures, strict=False):
        sns.heatmap(
            np.asarray(results[architecture]["confusion_matrix"]),
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=range(1, 6),
            yticklabels=range(1, 6),
            ax=axis,
        )
        axis.set(title=f"Bi{architecture.upper()}", xlabel="Predicted rating", ylabel="True rating")

    comparison_axis = axes[-1]
    means = [results[name]["accuracy"]["mean"] for name in architectures]
    margins = [results[name]["accuracy"]["margin_95"] for name in architectures]
    comparison_axis.bar(architectures, means, yerr=margins, capsize=6, color=["#2563eb", "#16a34a"])
    comparison_axis.set(title="Accuracy with 95% CI", ylabel="Accuracy", ylim=(0, 1))
    figure.tight_layout()
    figure.savefig(destination, dpi=160, bbox_inches="tight")
    plt.close(figure)
