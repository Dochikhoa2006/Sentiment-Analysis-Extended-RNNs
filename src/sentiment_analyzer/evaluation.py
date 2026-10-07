"""Leakage-resistant stratified cross-validation for BiLSTM and BiGRU models."""

from __future__ import annotations

import json
import os
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
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
    if not architectures or len(set(architectures)) != len(architectures):
        raise ValueError("architectures must contain unique model names")
    if folds < 2 or epochs < 1 or batch_size < 1:
        raise ValueError("folds must be at least 2; epochs and batch_size must be positive")
    configurations = {name: replace(base_config, architecture=name) for name in architectures}
    dataset = load_dataset(dataset_path)
    texts = dataset["review"].astype(str).to_numpy()
    labels = dataset["star"].astype(np.int64).to_numpy()
    class_labels = np.arange(base_config.number_of_classes)
    class_counts = np.bincount(labels, minlength=base_config.number_of_classes)
    if len(class_counts) != base_config.number_of_classes:
        raise ValueError("dataset labels exceed the configured number of classes")
    if np.any(class_counts < folds):
        counts = {int(label + 1): int(count) for label, count in enumerate(class_counts)}
        raise ValueError(
            f"each rating needs at least {folds} reviews for stratified validation; "
            f"rating counts: {counts}"
        )
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=base_config.seed)
    splits = list(splitter.split(texts, labels))
    fold_metrics: dict[str, list[dict[str, float]]] = {name: [] for name in architectures}
    matrices = {
        name: np.zeros(
            (base_config.number_of_classes, base_config.number_of_classes), dtype=np.int64
        )
        for name in architectures
    }

    for fold_number, (train_index, test_index) in enumerate(splits, start=1):
        train_texts, test_texts = texts[train_index], texts[test_index]
        train_labels, test_labels = labels[train_index], labels[test_index]
        vectorizer = EmbeddingVectorizer(
            base_config.embedding_dimension,
            workers=max(1, (os.cpu_count() or 2) // 2),
            seed=base_config.seed + fold_number,
        ).fit(train_texts)
        for architecture, architecture_config in configurations.items():
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
            macro_f1 = float(
                f1_score(
                    test_labels,
                    predictions,
                    labels=class_labels,
                    average="macro",
                    zero_division=0,
                )
            )
            fold_metrics[architecture].append(
                {"fold": fold_number, "accuracy": accuracy, "macro_f1": macro_f1}
            )
            matrices[architecture] += confusion_matrix(
                test_labels,
                predictions,
                labels=class_labels,
            )
            tf.keras.backend.clear_session()

    results: dict[str, Any] = {}
    for architecture in architectures:
        metrics = fold_metrics[architecture]
        accuracies = [fold["accuracy"] for fold in metrics]
        macro_scores = [fold["macro_f1"] for fold in metrics]
        accuracy_mean, accuracy_margin = confidence_interval(accuracies)
        macro_mean, macro_margin = confidence_interval(macro_scores)
        results[architecture] = {
            "rating_counts": {
                str(label + 1): int(count) for label, count in enumerate(class_counts)
            },
            "folds": metrics,
            "accuracy": {"mean": accuracy_mean, "margin_95": accuracy_margin},
            "macro_f1": {"mean": macro_mean, "margin_95": macro_margin},
            "confusion_matrix": matrices[architecture].tolist(),
        }

    _publish_results(results, output_dir)
    return results


def _publish_results(results: dict[str, Any], output_dir: Path) -> None:
    """Generate both outputs before replacing files from an earlier evaluation."""

    output_dir.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output_dir) as directory:
        staged = Path(directory)
        metrics = staged / "metrics.json"
        plot = staged / "model_comparison.png"
        metrics.write_text(json.dumps(results, indent=2), encoding="utf-8")
        _plot_results(results, plot)
        os.replace(plot, output_dir / plot.name)
        os.replace(metrics, output_dir / metrics.name)


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
