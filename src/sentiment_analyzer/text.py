"""Unicode-aware text normalization and tokenization."""

from __future__ import annotations

import re
import unicodedata

# Preserve Unicode letters and internal apostrophes, underscores, and hyphens.
TOKEN_PATTERN = re.compile(r"\w+(?:['’_-]\w+)*|[^\w\s]+", flags=re.UNICODE)


def tokenize(text: str) -> list[str]:
    """Normalize a review and return stable, Unicode-aware tokens."""

    if not isinstance(text, str):
        raise TypeError("text must be a string")
    normalized = unicodedata.normalize("NFKC", text).casefold().strip()
    return TOKEN_PATTERN.findall(normalized)
