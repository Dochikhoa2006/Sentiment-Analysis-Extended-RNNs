"""Central configuration and repository path conventions."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModelConfig:
    """Architecture and vectorization settings shared by training and inference."""

    sequence_length: int = 150
    embedding_dimension: int = 130
    recurrent_units: tuple[int, int] = (128, 64)
    number_of_classes: int = 5
    architecture: str = "gru"
    seed: int = 100

    def __post_init__(self) -> None:
        if self.architecture not in {"gru", "lstm"}:
            raise ValueError("architecture must be either 'gru' or 'lstm'")
        if self.sequence_length < 1 or self.embedding_dimension < 1:
            raise ValueError("sequence and embedding dimensions must be positive")
        if self.number_of_classes < 2:
            raise ValueError("number_of_classes must be at least 2")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ProjectPaths:
    """Resolved locations for local data and generated artifacts."""

    root: Path
    raw_dataset: Path
    processed_dataset: Path
    corpus: Path
    vectorizer: Path
    model: Path
    evaluation_dir: Path

    @classmethod
    def defaults(cls, root: Path | None = None) -> ProjectPaths:
        repository_root = root or Path(__file__).resolve().parents[2]
        return cls(
            root=repository_root,
            raw_dataset=Path(
                os.getenv(
                    "SENTIMENT_RAW_DATASET_PATH",
                    repository_root / "data/raw/reviews.parquet",
                )
            ),
            processed_dataset=Path(
                os.getenv(
                    "SENTIMENT_DATASET_PATH",
                    repository_root / "data/processed/reviews.joblib",
                )
            ),
            corpus=repository_root / "data/interim/processed_reviews.txt",
            vectorizer=Path(
                os.getenv(
                    "SENTIMENT_VECTORIZER_PATH",
                    repository_root / "artifacts/fasttext_vectorizer.joblib",
                )
            ),
            model=Path(
                os.getenv(
                    "SENTIMENT_MODEL_PATH",
                    repository_root / "artifacts/sentiment_bigru.keras",
                )
            ),
            evaluation_dir=repository_root / "artifacts/evaluation",
        )

    def create_runtime_directories(self) -> None:
        for path in (
            self.raw_dataset.parent,
            self.processed_dataset.parent,
            self.corpus.parent,
            self.vectorizer.parent,
            self.model.parent,
            self.evaluation_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)
