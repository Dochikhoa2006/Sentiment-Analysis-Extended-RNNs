# Architecture

## Design goals

The codebase separates data preparation, representation learning, neural modeling, evaluation, and
inference so each stage can be tested or replaced independently. Generated data and model files are
local artifacts rather than source-controlled dependencies.

## Data flow

| Stage | Input | Output | Module |
|---|---|---|---|
| Download | Hugging Face dataset ID | Raw Parquet | `data.py` |
| Prepare | Raw review records | Validated `review`, zero-based `star` frame | `data.py` |
| Explore | Raw review records | Curated PNG figures | `eda.py` |
| Embed | Clean review strings | Fitted FastText vectorizer | `embeddings.py` |
| Train | Text, labels, vectorizer | TensorFlow `.keras` model and metadata | `training.py` |
| Evaluate | Text and labels | Metrics JSON and comparison plot | `evaluation.py` |
| Infer | Review text and artifacts | Rating, sentiment, confidence, probabilities | `inference.py` |

## Core decisions

### Subword embeddings

FastText derives representations from character n-grams, which is useful for short reviews with
misspellings, inflections, abbreviations, and words absent from the training vocabulary. Reviews are
normalized with Unicode NFKC and case-folding before tokenization.

### Bounded-memory vectorization

The cleaned dataset contains 197,595 reviews. A dense tensor for every review at 150 tokens and 130
float32 values per token would occupy approximately 14.4 GiB before training overhead. `ReviewSequence`
constructs only the active batch and keeps memory proportional to batch size.

### Leakage-resistant evaluation

FastText is fitted independently inside each cross-validation training fold. The held-out fold does
not influence its vocabulary or learned subword representations. `StratifiedKFold` preserves the
severe label imbalance across folds, while macro-F1 exposes performance on rare classes.

### Artifact format

New neural models use TensorFlow's native `.keras` format rather than pickling application classes.
The vectorizer remains a Joblib artifact because it wraps a Gensim model. Compatibility shims allow
local artifacts created by the original flat scripts to be loaded during migration.

## Runtime boundaries

- Core text and embedding utilities require NumPy, Gensim, and Joblib.
- Inference adds TensorFlow.
- Training and evaluation add pandas, scikit-learn, and SciPy.
- Download, Spark preprocessing, and plotting are isolated behind the `data` extra.

This split keeps the Docker inference image independent of Java, Spark, and visualization libraries.

## Reproducibility

Architecture settings and seeds live in `ModelConfig`. Final training writes a metadata JSON file
next to the model. Evaluation writes machine-readable per-fold metrics and aggregate confusion
matrices. Exact neural results can still vary slightly across hardware and TensorFlow kernels.

