from pathlib import Path

import pytest

from sentiment_analyzer.config import ModelConfig, ProjectPaths


def test_model_config_rejects_unknown_architecture() -> None:
    with pytest.raises(ValueError, match="architecture"):
        ModelConfig(architecture="transformer")


def test_project_paths_are_rooted_in_requested_directory(tmp_path: Path) -> None:
    paths = ProjectPaths.defaults(tmp_path)

    assert paths.raw_dataset == tmp_path / "data/raw/reviews.parquet"
    assert paths.model == tmp_path / "artifacts/sentiment_bigru.keras"
