import numpy as np

from sentiment_analyzer.embeddings import EmbeddingVectorizer


class _WordVectors:
    def __getitem__(self, token: str) -> np.ndarray:
        return np.full(3, len(token), dtype=np.float32)


class _FastTextStub:
    wv = _WordVectors()


def test_transform_is_padded_truncated_and_float32() -> None:
    vectorizer = EmbeddingVectorizer(vector_size=3)
    vectorizer.model = _FastTextStub()

    result = vectorizer.transform(["one two three", "four"], sequence_length=2)

    assert result.shape == (2, 2, 3)
    assert result.dtype == np.float32
    np.testing.assert_array_equal(result[0, 0], [3, 3, 3])
    np.testing.assert_array_equal(result[1, 1], [0, 0, 0])
