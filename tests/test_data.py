import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from sentiment_analyzer.data import _write_atomically, download_dataset, validate_dataset


def test_validate_dataset_accepts_prepared_rows() -> None:
    validate_dataset(pd.DataFrame({"review": ["Useful", "Needs work"], "star": [4, 1]}))


@pytest.mark.parametrize("review", [None, "", "  ", 12])
def test_validate_dataset_rejects_invalid_reviews(review: object) -> None:
    with pytest.raises(ValueError, match="reviews"):
        validate_dataset(pd.DataFrame({"review": [review], "star": [4]}))


@pytest.mark.parametrize("star", [None, 1.5, -1, 5, True, "4"])
def test_validate_dataset_rejects_invalid_labels(star: object) -> None:
    with pytest.raises(ValueError, match="star labels"):
        validate_dataset(pd.DataFrame({"review": ["Useful"], "star": [star]}))


def test_atomic_dataset_write_preserves_existing_file_on_failure(tmp_path: Path) -> None:
    destination = tmp_path / "reviews.joblib"
    destination.write_bytes(b"previous")

    def failing_writer(staged: Path) -> None:
        staged.write_bytes(b"partial")
        raise RuntimeError("write failed")

    with pytest.raises(RuntimeError, match="write failed"):
        _write_atomically(destination, failing_writer)

    assert destination.read_bytes() == b"previous"
    assert list(tmp_path.iterdir()) == [destination]


def test_download_publishes_complete_dataset(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    destination = tmp_path / "nested" / "reviews.parquet"
    destination.parent.mkdir()
    destination.write_bytes(b"previous")

    class Dataset:
        def to_parquet(self, path: str) -> None:
            assert Path(path) != destination
            Path(path).write_bytes(b"complete")

    monkeypatch.setitem(
        sys.modules, "datasets", SimpleNamespace(load_dataset=lambda *a, **k: Dataset())
    )

    assert download_dataset(destination) == destination
    assert destination.read_bytes() == b"complete"


def test_download_keeps_previous_dataset_on_write_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    destination = tmp_path / "reviews.parquet"
    destination.write_bytes(b"previous")

    class Dataset:
        def to_parquet(self, path: str) -> None:
            Path(path).write_bytes(b"partial")
            raise OSError("disk full")

    monkeypatch.setitem(
        sys.modules, "datasets", SimpleNamespace(load_dataset=lambda *a, **k: Dataset())
    )

    with pytest.raises(OSError, match="disk full"):
        download_dataset(destination)
    assert destination.read_bytes() == b"previous"
