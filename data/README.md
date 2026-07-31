# Data directory

Data files are generated locally and intentionally excluded from Git.

```text
data/
├── raw/reviews.parquet              # Downloaded source split
├── interim/processed_reviews.txt    # Optional inspectable FastText corpus
└── processed/reviews.joblib         # Clean review/label DataFrame
```

## Source

- Dataset: [`LocalDoc/application_reviews`](https://huggingface.co/datasets/LocalDoc/application_reviews)
- Split: `train`
- Source license: CC BY-NC 4.0
- Source size: approximately 198,000 rows

Expected raw fields are `review`, `star`, `date`, and `package_name`. The preparation stage removes
missing or empty reviews, validates ratings, selects the modeling fields, and shifts ratings from
the human-facing range 1–5 to class indices 0–4.

## Rebuild

```bash
sentiment-analyzer download
sentiment-analyzer prepare
sentiment-analyzer embeddings
```

Do not commit raw reviews or serialized datasets. Besides repository size, review text can contain
personal or offensive content and remains governed by the source license.

