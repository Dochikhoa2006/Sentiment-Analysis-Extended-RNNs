from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sentiment_analyzer import training
from sentiment_analyzer.config import ModelConfig


def test_final_training_fits_vectorizer_only_on_training_reviews(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    dataset = pd.DataFrame(
        {"review": [f"review {i}" for i in range(20)], "star": np.repeat(np.arange(5), 4)}
    )
    monkeypatch.setattr(training, "load_dataset", lambda path: dataset)
    fitted_texts: list[str] = []

    class Vectorizer:
        def __init__(self, dimension: int, *, workers: int, seed: int) -> None:
            self.dimension = dimension

        def fit(self, texts: np.ndarray) -> "Vectorizer":
            fitted_texts.extend(texts)
            return self

        def save(self, path: Path) -> None:
            path.write_bytes(b"vectorizer")

    class Batches:
        def __init__(self, texts, vectorizer, sequence_length, **kwargs) -> None:
            self.texts = texts

    class Model:
        def fit(self, batches, *, validation_data, **kwargs):
            assert set(fitted_texts) == set(batches.texts)
            assert not set(fitted_texts).intersection(validation_data.texts)
            return type("History", (), {"history": {"loss": [0.5]}})()

        def save(self, path: Path) -> None:
            path.write_bytes(b"model")

    monkeypatch.setattr(training, "EmbeddingVectorizer", Vectorizer)
    monkeypatch.setattr(training, "ReviewSequence", Batches)
    monkeypatch.setattr(training, "build_model", lambda config: Model())
    monkeypatch.setattr(training, "_class_weights", lambda labels: {})

    _, metadata = training.train_final_model(
        tmp_path / "dataset",
        tmp_path / "vectorizer.joblib",
        tmp_path / "model.keras",
        config=ModelConfig(embedding_dimension=2),
        validation_fraction=0.25,
    )

    assert len(fitted_texts) == metadata["training_rows"] == 15
    assert metadata["vectorizer_fitted_on_training_split"] is True
    assert (tmp_path / "vectorizer.joblib").exists()


def test_final_training_rejects_invalid_validation_fraction(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="validation_fraction"):
        training.train_final_model(tmp_path, tmp_path, tmp_path, validation_fraction=1)


def test_failed_training_preserves_existing_artifacts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    dataset = pd.DataFrame(
        {"review": [f"review {i}" for i in range(20)], "star": np.repeat(np.arange(5), 4)}
    )
    monkeypatch.setattr(training, "load_dataset", lambda path: dataset)
    vectorizer_path = tmp_path / "vectorizer.joblib"
    model_path = tmp_path / "model.keras"
    metadata_path = tmp_path / "model.metadata.json"
    for path in (vectorizer_path, model_path, metadata_path):
        path.write_bytes(b"previous")

    class Vectorizer:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def fit(self, texts):
            return self

        def save(self, path: Path) -> None:
            path.write_bytes(b"new")

    class Model:
        def fit(self, *args, **kwargs):
            raise RuntimeError("training failed")

    monkeypatch.setattr(training, "EmbeddingVectorizer", Vectorizer)
    monkeypatch.setattr(training, "ReviewSequence", lambda *args, **kwargs: object())
    monkeypatch.setattr(training, "build_model", lambda config: Model())
    monkeypatch.setattr(training, "_class_weights", lambda labels: {})

    with pytest.raises(RuntimeError, match="training failed"):
        training.train_final_model(
            tmp_path / "dataset", vectorizer_path, model_path, validation_fraction=0.25
        )
    paths = (vectorizer_path, model_path, metadata_path)
    assert all(path.read_bytes() == b"previous" for path in paths)


@pytest.mark.parametrize(
    "left,right", list(combinations(("dataset", "vectorizer", "model", "metadata"), 2))
)
def test_final_training_rejects_overlapping_paths_before_loading(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, left: str, right: str
) -> None:
    paths = {
        "dataset": tmp_path / "dataset.joblib",
        "vectorizer": tmp_path / "vectorizer.joblib",
        "model": tmp_path / "model.keras",
        "metadata": tmp_path / "model.metadata.json",
    }
    if left == "model" and right == "metadata":
        paths[left].symlink_to(paths[right])
    elif right == "metadata":
        paths[left] = paths[right]
    else:
        paths[right] = paths[left]

    def unexpected_load(path):
        pytest.fail("overlapping paths must be rejected before loading the dataset")

    monkeypatch.setattr(training, "load_dataset", unexpected_load)
    with pytest.raises(ValueError, match="different files"):
        training.train_final_model(paths["dataset"], paths["vectorizer"], paths["model"])


@pytest.mark.parametrize("left,right", list(combinations(("dataset", "vectorizer", "corpus"), 2)))
def test_embedding_training_rejects_overlapping_destinations(
    tmp_path: Path, left: str, right: str
) -> None:
    paths = {name: tmp_path / name for name in ("dataset", "vectorizer", "corpus")}
    paths[right] = paths[left]
    with pytest.raises(ValueError, match="different files"):
        training.train_embeddings(
            paths["dataset"], paths["vectorizer"], corpus_path=paths["corpus"]
        )


@pytest.mark.parametrize("link_type", ["symlink", "hardlink"])
def test_embedding_training_rejects_aliases_of_dataset(tmp_path: Path, link_type: str) -> None:
    dataset = tmp_path / "dataset.joblib"
    dataset.write_bytes(b"original dataset")
    alias = tmp_path / "vectorizer.joblib"
    if link_type == "symlink":
        alias.symlink_to(dataset)
    else:
        alias.hardlink_to(dataset)

    with pytest.raises(ValueError, match="different files"):
        training.train_embeddings(dataset, alias)
    assert dataset.read_bytes() == b"original dataset"
