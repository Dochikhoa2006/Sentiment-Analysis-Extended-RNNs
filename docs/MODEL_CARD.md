# Model card: Bidirectional GRU app-review classifier

## Model overview

| Field | Value |
|---|---|
| Task | Five-class review sentiment/rating classification |
| Input | One application review, truncated to 150 tokens |
| Output | Probabilities for star ratings 1 through 5 |
| Representation | 130-dimensional FastText subword vectors |
| Selected architecture | Two stacked bidirectional GRU layers (128 and 64 units) |
| Alternative evaluated | Two stacked bidirectional LSTM layers (128 and 64 units) |
| Framework | TensorFlow/Keras |

## Intended use

The model is a research and portfolio demonstration for aggregate review analytics, model
comparison, and NLP experimentation. Appropriate uses include exploring broad sentiment trends,
prioritizing samples for human review, and learning how subword embeddings interact with recurrent
architectures.

It is not intended for automated moderation, decisions about individuals, safety-critical systems,
or any workflow where a wrong sentiment label has a material consequence.

## Training data

The source is the `LocalDoc/application_reviews` dataset on Hugging Face. It contains roughly 198k
rows across 622 application package names and is licensed under CC BY-NC 4.0. The modeling target is
the user-provided star rating shifted from 1–5 to class indices 0–4.

The local cleaned snapshot contains 197,595 non-empty reviews:

| Rating | Reviews | Share |
|---:|---:|---:|
| 1 | 24,964 | 12.63% |
| 2 | 3,901 | 1.97% |
| 3 | 5,871 | 2.97% |
| 4 | 9,225 | 4.67% |
| 5 | 153,634 | 77.75% |

## Evaluation

The retained legacy plot indicates approximately 83.27% accuracy for BiLSTM and 83.42% for BiGRU.
The raw fold-level arrays were not persisted, so these values are visual estimates rather than
precision claims. They are not presented as results from the refactored evaluation pipeline.

The current protocol uses:

- Stratified K-fold splits with a deterministic split seed.
- FastText fitting on the training portion of each fold only.
- Balanced class weights during recurrent-network training.
- Accuracy, macro-F1, 95% Student-t confidence intervals, and aggregate confusion matrices.
- True labels on confusion-matrix rows and predicted labels on columns.

New results are written to `artifacts/evaluation/metrics.json` and should be recorded here only after
the full experiment has completed in a documented environment.

## Limitations and risks

- **Proxy target:** star ratings are not direct sentiment annotations. A positive review can carry a
  low rating and vice versa.
- **Class imbalance:** a trivial five-star predictor would already achieve about 77.8% accuracy.
  Accuracy alone therefore overstates useful model quality.
- **Language and domain coverage:** quality will vary across languages, applications, slang, and
  review lengths represented unevenly in the source data.
- **Ordinal structure:** cross-entropy treats all class errors categorically; predicting 1 instead
  of 2 is penalized the same way as predicting 1 instead of 5.
- **Temporal drift:** applications and user vocabulary change. There is no guarantee that a model
  trained on the current snapshot will remain calibrated.
- **Confidence is not certainty:** softmax output is not automatically calibrated and should not be
  interpreted as a verified probability without calibration analysis.
- **Sensitive text:** source reviews may include personal or offensive content.

## Recommended follow-up work

1. Add a temporal or application-grouped holdout to measure out-of-domain generalization.
2. Report per-class precision, recall, F1, and ordinal metrics such as quadratic weighted kappa.
3. Compare against TF-IDF linear and compact transformer baselines.
4. Calibrate probabilities on a dedicated validation set.
5. Add experiment tracking and publish immutable model/data version identifiers.

## License

The dataset uses CC BY-NC 4.0. Review the dataset terms before distributing trained artifacts or
using them outside non-commercial research and demonstration contexts.
