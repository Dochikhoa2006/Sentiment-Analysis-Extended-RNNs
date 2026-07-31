"""Artifact loading, including compatibility with the original pickle files."""

from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from sentiment_analyzer.embeddings import EmbeddingVectorizer


class LegacyBidirectionalRNN:
    """Compatibility shell for models serialized by ``Cross_Validation.py``."""

    def make_predictions(self, features: np.ndarray) -> np.ndarray:
        probabilities = self.model.predict(features, verbose=0)
        return np.argmax(probabilities, axis=1)


def _register_legacy_modules() -> None:
    # ``Second_Preprocess.py`` was originally executed as a script, so its pickle
    # records ``__main__.Embedding_Vector`` rather than an importable package path.
    main_module = sys.modules.get("__main__")
    if main_module is not None and not hasattr(main_module, "Embedding_Vector"):
        main_module.Embedding_Vector = EmbeddingVectorizer

    preprocess = types.ModuleType("Second_Preprocess")
    preprocess.Embedding_Vector = EmbeddingVectorizer
    sys.modules.setdefault("Second_Preprocess", preprocess)

    validation = types.ModuleType("Cross_Validation")
    validation.Bidirectional_Extended_RNNs = LegacyBidirectionalRNN
    sys.modules.setdefault("Cross_Validation", validation)


def load_vectorizer(source: Path) -> EmbeddingVectorizer:
    if not source.exists():
        raise FileNotFoundError(
            f"vectorizer not found: {source}. Run 'sentiment-analyzer embeddings' first."
        )
    _register_legacy_modules()
    vectorizer = joblib.load(source)
    if not hasattr(vectorizer, "transform"):
        raise TypeError(f"{source} does not contain a compatible vectorizer")
    return vectorizer


def load_model(source: Path) -> Any:
    if not source.exists():
        raise FileNotFoundError(f"model not found: {source}. Run 'sentiment-analyzer train' first.")
    if source.suffix == ".keras":
        try:
            from tensorflow.keras.models import load_model as keras_load_model
        except ImportError as exc:
            message = "Install inference dependencies with: pip install -e '.[inference]'"
            raise RuntimeError(message) from exc
        return keras_load_model(source)

    _register_legacy_modules()
    return joblib.load(source)
