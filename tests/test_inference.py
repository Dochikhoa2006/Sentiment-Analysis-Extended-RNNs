import numpy as np
import pytest

from sentiment_analyzer.embeddings import EmbeddingVectorizer
from sentiment_analyzer.inference import SentimentPredictor


class _WordVectors:
    def __getitem__(self, token: str) -> np.ndarray:
        return np.ones(2, dtype=np.float32)


class _FastTextStub:
    wv = _WordVectors()


class _ModelStub:
    def predict(self, features: np.ndarray, verbose: int = 0) -> np.ndarray:
        assert features.shape == (1, 3, 2)
        return np.asarray([[0.01, 0.02, 0.05, 0.12, 0.80]], dtype=np.float32)


def _predictor() -> SentimentPredictor:
    vectorizer = EmbeddingVectorizer(vector_size=2)
    vectorizer.model = _FastTextStub()
    return SentimentPredictor(vectorizer, _ModelStub(), sequence_length=3)


def test_predict_returns_model_class_instead_of_second_argmax() -> None:
    result = _predictor().predict("Excellent update")

    assert result.rating == 5
    assert result.sentiment == "strongly satisfied"
    assert result.confidence == pytest.approx(0.8)


def test_predict_rejects_empty_review() -> None:
    with pytest.raises(ValueError, match="empty"):
        _predictor().predict("   ")
