import numpy as np
import pytest

from sentiment_analyzer.batching import ReviewSequence


class _Vectorizer:
    def transform(self, texts, sequence_length: int) -> np.ndarray:
        return np.zeros((len(texts), sequence_length, 2), dtype=np.float32)


def test_review_sequence_bounds_and_partial_last_batch() -> None:
    batches = ReviewSequence(["a", "b", "c"], _Vectorizer(), 3, batch_size=2)

    assert len(batches) == 2
    assert batches[0].shape == (2, 3, 2)
    assert batches[1].shape == (1, 3, 2)
    with pytest.raises(IndexError, match="out of range"):
        batches[2]


@pytest.mark.parametrize("batch_size, sequence_length", [(0, 3), (2, 0)])
def test_review_sequence_rejects_invalid_sizes(batch_size: int, sequence_length: int) -> None:
    with pytest.raises(ValueError, match="positive"):
        ReviewSequence(["review"], _Vectorizer(), sequence_length, batch_size=batch_size)
