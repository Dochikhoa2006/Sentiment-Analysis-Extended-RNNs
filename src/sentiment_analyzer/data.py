"""Dataset download, validation, and preprocessing."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib

DATASET_ID = "LocalDoc/application_reviews"
REQUIRED_RAW_COLUMNS = {"review", "star", "date", "package_name"}
REQUIRED_MODEL_COLUMNS = {"review", "star"}


def download_dataset(destination: Path, dataset_id: str = DATASET_ID) -> Path:
    """Download the public Hugging Face training split as Parquet."""

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Install the data dependencies with: pip install -e '.[data]'") from exc

    destination.parent.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset(dataset_id, split="train")
    dataset.to_parquet(str(destination))
    return destination


def prepare_dataset(source: Path, destination: Path) -> Any:
    """Clean raw reviews with Spark and serialize the two modeling columns."""

    try:
        from pyspark.sql import SparkSession
        from pyspark.sql import functions as sql
    except ImportError as exc:
        raise RuntimeError("Install the data dependencies with: pip install -e '.[data]'") from exc

    if not source.exists():
        raise FileNotFoundError(f"raw dataset not found: {source}")

    spark = SparkSession.builder.master("local[*]").appName("review-preprocessing").getOrCreate()
    try:
        frame = spark.read.parquet(str(source))
        missing = REQUIRED_RAW_COLUMNS.difference(frame.columns)
        if missing:
            raise ValueError(f"raw dataset is missing columns: {sorted(missing)}")

        cleaned = (
            frame.select("review", "star")
            .dropna(subset=["review", "star"])
            .filter(sql.length(sql.trim(sql.col("review"))) > 0)
            .filter(sql.col("star").between(1, 5))
            .withColumn("star", sql.col("star").cast("int") - 1)
        )
        result = cleaned.toPandas()
    finally:
        spark.stop()

    validate_dataset(result)
    destination.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(result, destination)
    return result


def load_dataset(source: Path) -> Any:
    if not source.exists():
        raise FileNotFoundError(
            f"processed dataset not found: {source}. Run 'sentiment-analyzer prepare' first."
        )
    dataset = joblib.load(source)
    validate_dataset(dataset)
    return dataset


def validate_dataset(dataset: Any) -> None:
    if not hasattr(dataset, "columns"):
        raise TypeError("processed dataset must be a pandas DataFrame")
    missing = REQUIRED_MODEL_COLUMNS.difference(dataset.columns)
    if missing:
        raise ValueError(f"processed dataset is missing columns: {sorted(missing)}")
    if len(dataset) == 0:
        raise ValueError("processed dataset is empty")
    labels = set(int(value) for value in dataset["star"].unique())
    if not labels.issubset(set(range(5))):
        raise ValueError("star labels must be zero-based integers in the range 0..4")
