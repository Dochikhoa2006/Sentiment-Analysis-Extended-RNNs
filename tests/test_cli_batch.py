import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from sentiment_analyzer import cli, inference


class _Predictor:
    def predict_many(self, reviews, *, batch_size: int):
        for review in reviews:
            yield SimpleNamespace(to_dict=lambda review=review: {"review": review})


def _run(monkeypatch: pytest.MonkeyPatch, source: Path, output: Path) -> None:
    monkeypatch.setattr(inference.SentimentPredictor, "from_artifacts", lambda *args: _Predictor())
    cli.main(["predict-batch", str(source), "--output", str(output), "--batch-size", "1"])


def test_batch_cli_writes_complete_jsonl(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    source = tmp_path / "reviews.txt"
    output = tmp_path / "results.jsonl"
    source.write_text("first\nsecond\n", encoding="utf-8")

    _run(monkeypatch, source, output)

    assert [json.loads(line) for line in output.read_text().splitlines()] == [
        {"review": "first"},
        {"review": "second"},
    ]


def test_batch_cli_keeps_existing_output_on_invalid_review(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source = tmp_path / "reviews.txt"
    output = tmp_path / "results.jsonl"
    source.write_text("first\n \n", encoding="utf-8")
    output.write_text("previous\n", encoding="utf-8")

    with pytest.raises(ValueError, match="input line 2"):
        _run(monkeypatch, source, output)

    assert output.read_text(encoding="utf-8") == "previous\n"


def test_batch_cli_rejects_same_input_and_output(tmp_path: Path) -> None:
    source = tmp_path / "reviews.txt"
    source.write_text("review\n", encoding="utf-8")

    with pytest.raises(ValueError, match="differ"):
        cli.main(["predict-batch", str(source), "--output", str(source)])
