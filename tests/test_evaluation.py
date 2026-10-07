import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from sentiment_analyzer import evaluation


def test_cross_validation_reuses_fold_embeddings_across_architectures(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    labels = np.repeat(np.arange(5), 4)
    dataset = pd.DataFrame({"review": [f"review {i}" for i in range(20)], "star": labels})
    monkeypatch.setattr(evaluation, "load_dataset", lambda path: dataset)

    fitted: list[object] = []
    used: list[tuple[str, object]] = []

    class Vectorizer:
        def __init__(self, dimension: int, *, workers: int, seed: int) -> None:
            self.seed = seed

        def fit(self, texts: np.ndarray) -> "Vectorizer":
            fitted.append(self)
            return self

    class Batches:
        def __init__(self, texts, vectorizer, sequence_length, *, labels=None, **kwargs):
            inferred = [int(text.split()[1]) // 4 for text in texts]
            self.labels = labels if labels is not None else np.asarray(inferred)
            self.vectorizer = vectorizer

    class Model:
        def __init__(self, name: str) -> None:
            self.name = name

        def fit(self, batches, **kwargs) -> None:
            used.append((self.name, batches.vectorizer))

        def predict(self, batches, verbose: int = 0) -> np.ndarray:
            return np.eye(5)[batches.labels]

    monkeypatch.setattr(evaluation, "EmbeddingVectorizer", Vectorizer)
    monkeypatch.setattr(evaluation, "ReviewSequence", Batches)
    monkeypatch.setattr(evaluation, "build_model", lambda config: Model(config.architecture))
    monkeypatch.setattr(evaluation, "_plot_results", lambda results, path: None)
    monkeypatch.setattr(evaluation, "_class_weights", lambda labels: {})
    monkeypatch.setitem(
        sys.modules,
        "tensorflow",
        SimpleNamespace(keras=SimpleNamespace(backend=SimpleNamespace(clear_session=lambda: None))),
    )

    results = evaluation.cross_validate(tmp_path / "dataset", tmp_path, folds=2)

    assert len(fitted) == 2
    assert used == [
        ("lstm", fitted[0]),
        ("gru", fitted[0]),
        ("lstm", fitted[1]),
        ("gru", fitted[1]),
    ]
    assert all(len(results[name]["folds"]) == 2 for name in ("lstm", "gru"))
    assert all(results[name]["accuracy"]["mean"] == 1 for name in ("lstm", "gru"))
    assert results["gru"]["rating_counts"] == dict.fromkeys("12345", 4)


def test_cross_validation_rejects_duplicate_architectures(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unique"):
        evaluation.cross_validate(tmp_path, tmp_path, architectures=("gru", "gru"))


@pytest.mark.parametrize("counts", [(4, 4, 4, 4, 0), (4, 4, 4, 4, 1)])
def test_cross_validation_rejects_insufficient_class_coverage(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, counts: tuple[int, ...]
) -> None:
    labels = np.repeat(np.arange(5), counts)
    dataset = pd.DataFrame(
        {"review": [f"review {i}" for i in range(len(labels))], "star": labels}
    )
    monkeypatch.setattr(evaluation, "load_dataset", lambda path: dataset)
    monkeypatch.setitem(
        sys.modules,
        "tensorflow",
        SimpleNamespace(keras=SimpleNamespace(backend=SimpleNamespace(clear_session=lambda: None))),
    )

    with pytest.raises(ValueError, match="each rating needs at least 2 reviews"):
        evaluation.cross_validate(tmp_path / "dataset", tmp_path, folds=2)
