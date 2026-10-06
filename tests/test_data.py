import pandas as pd
import pytest

from sentiment_analyzer.data import validate_dataset


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
