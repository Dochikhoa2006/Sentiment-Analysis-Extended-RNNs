# Artifact directory

This directory stores generated model products and is ignored by Git except for this guide.

| Artifact | Producer | Purpose |
|---|---|---|
| `fasttext_vectorizer.joblib` | `sentiment-analyzer embeddings` | Token/subword representation |
| `sentiment_bigru.keras` | `sentiment-analyzer train` | Native TensorFlow classifier |
| `sentiment_bigru.metadata.json` | `sentiment-analyzer train` | Configuration and training history |
| `evaluation/metrics.json` | `sentiment-analyzer evaluate` | Per-fold and aggregate metrics |
| `evaluation/model_comparison.png` | `sentiment-analyzer evaluate` | Confusion matrices and accuracy CI |

The loader supports original Joblib model artifacts for local migration, but new training runs use
TensorFlow's `.keras` format.

> [!WARNING]
> Joblib and pickle-based artifacts can execute code while loading. Never load an artifact from an
> untrusted source. Prefer artifacts you trained yourself and record their checksum before transfer.

Large release artifacts should be published through a versioned model registry or GitHub Release,
not committed to the repository. Check the dataset's CC BY-NC 4.0 terms before redistribution.

