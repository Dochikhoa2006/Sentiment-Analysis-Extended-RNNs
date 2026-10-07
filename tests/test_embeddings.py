import numpy as np
import pytest

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


def test_batch_matches_individual_reviews_for_single_pass_input() -> None:
    vectorizer = EmbeddingVectorizer(vector_size=3)
    vectorizer.model = _FastTextStub()
    texts = ["one two three", "four", "", "café works!"]
    expected = np.stack([vectorizer.transform_one(text, 2) for text in texts])

    result = vectorizer.transform((text for text in texts), sequence_length=2)

    np.testing.assert_array_equal(result, expected)
    assert result.flags.c_contiguous
    result[0, 0, 0] = -1
    assert result[1, 0, 0] == 4


def test_empty_batch_has_correct_shape_and_dtype() -> None:
    vectorizer = EmbeddingVectorizer(vector_size=3)
    vectorizer.model = _FastTextStub()

    result = vectorizer.transform(iter(()), sequence_length=2)

    assert result.shape == (0, 2, 3)
    assert result.dtype == np.float32


def test_empty_batch_validates_transform_preconditions() -> None:
    vectorizer = EmbeddingVectorizer(vector_size=3)
    with pytest.raises(RuntimeError, match="fitted"):
        vectorizer.transform([])

    vectorizer.model = _FastTextStub()
    with pytest.raises(ValueError, match="positive"):
        vectorizer.transform([], sequence_length=0)
