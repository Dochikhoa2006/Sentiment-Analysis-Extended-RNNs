import pytest

from sentiment_analyzer.text import tokenize


def test_tokenize_preserves_unicode_and_internal_punctuation() -> None:
    assert tokenize("  ÇOX yaxşı — user-friendly!  ") == [
        "çox",
        "yaxşı",
        "—",
        "user-friendly",
        "!",
    ]


def test_tokenize_rejects_non_string_input() -> None:
    with pytest.raises(TypeError, match="string"):
        tokenize(None)  # type: ignore[arg-type]
