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


def test_predict_many_preserves_order_and_bounds_batch_size() -> None:
    class Model:
        sizes: list[int] = []

        def predict(self, features: np.ndarray, verbose: int = 0) -> np.ndarray:
            self.sizes.append(len(features))
            scores = np.zeros((len(features), 5), dtype=np.float32)
            scores[:, 3] = 1.0
            return scores

    model = Model()
    predictor = _predictor()
    predictor.model = model
    results = list(predictor.predict_many((f"review {n}" for n in range(5)), batch_size=2))

    assert model.sizes == [2, 2, 1]
    assert [result.rating for result in results] == [4] * 5


def test_predict_many_rejects_bad_model_output() -> None:
    class Model:
        def predict(self, features: np.ndarray, verbose: int = 0) -> np.ndarray:
            return np.full((len(features), 4), 0.25)

    predictor = _predictor()
    predictor.model = Model()
    with pytest.raises(ValueError, match="shape"):
        list(predictor.predict_many(["review"]))


@pytest.mark.parametrize(
    "scores",
    [
        [0.1, 0.1, 0.1, 0.1, 0.1],
        [-0.1, 0.1, 0.1, 0.1, 0.8],
        [0.0, 0.0, 0.0, 0.0, 1.1],
        [0.0, 0.0, 0.0, 0.0, float("nan")],
    ],
)
def test_predict_many_rejects_invalid_probability_distributions(scores: list[float]) -> None:
    class Model:
        def predict(self, features: np.ndarray, verbose: int = 0) -> np.ndarray:
            return np.asarray([scores] * len(features))

    predictor = _predictor()
    predictor.model = Model()
    with pytest.raises(ValueError, match="probability|non-finite"):
        list(predictor.predict_many(["review"]))


def test_predict_many_rejects_invalid_batch_size() -> None:
    with pytest.raises(ValueError, match="batch_size"):
        list(_predictor().predict_many(["review"], batch_size=0))
