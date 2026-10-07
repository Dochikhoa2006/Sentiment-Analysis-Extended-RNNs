import json
from pathlib import Path

import pytest

from sentiment_analyzer import inference
from sentiment_analyzer.serialization import artifact_sha256


def _artifacts(tmp_path: Path) -> tuple[Path, Path]:
    model = tmp_path / "model.keras"
    vectorizer = tmp_path / "vectors.joblib"
    model.write_bytes(b"model")
    vectorizer.write_bytes(b"vectors")
    model.with_suffix(".metadata.json").write_text(
        json.dumps(
            {
                "config": {"sequence_length": 12, "embedding_dimension": 2},
                "artifacts": {
                    "model_sha256": artifact_sha256(model),
                    "vectorizer_sha256": artifact_sha256(vectorizer),
                },
            }
        ),
        encoding="utf-8",
    )
    return model, vectorizer


def test_inference_uses_trained_length_and_verifies_artifacts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    model, vectorizer = _artifacts(tmp_path)
    monkeypatch.setattr(inference, "load_model", lambda path: object())
    stub = type("Vectorizer", (), {"dimension": 2})()
    monkeypatch.setattr(inference, "load_vectorizer", lambda path: stub)

    predictor = inference.SentimentPredictor.from_artifacts(vectorizer, model)
    assert predictor.sequence_length == 12

    vectorizer.write_bytes(b"different")
    with pytest.raises(ValueError, match="vectorizer artifact"):
        inference.SentimentPredictor.from_artifacts(vectorizer, model)


def test_inference_rejects_wrong_sequence_length(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    model, vectorizer = _artifacts(tmp_path)
    with pytest.raises(ValueError, match="sequence_length"):
        inference.SentimentPredictor.from_artifacts(vectorizer, model, sequence_length=150)


@pytest.mark.parametrize(
    ("input_shape", "output_shape", "message"),
    [
        ((None, 13, 2), (None, 5), "input shape"),
        ((None, 12, 3), (None, 5), "input shape"),
        ((None, 12, 2), (None, 4), "output shape"),
    ],
)
def test_inference_rejects_incompatible_model_shapes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    input_shape: tuple[int | None, ...],
    output_shape: tuple[int | None, ...],
    message: str,
) -> None:
    model, vectorizer = _artifacts(tmp_path)
    stub = type("Vectorizer", (), {"dimension": 2})()
    monkeypatch.setattr(inference, "load_vectorizer", lambda path: stub)
    incompatible = type("Model", (), {"input_shape": input_shape, "output_shape": output_shape})()
    monkeypatch.setattr(inference, "load_model", lambda path: incompatible)

    with pytest.raises(ValueError, match=message):
        inference.SentimentPredictor.from_artifacts(vectorizer, model)
