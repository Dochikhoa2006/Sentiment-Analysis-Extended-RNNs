"""Command-line entry point for the full ML workflow."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from sentiment_analyzer.config import ModelConfig, ProjectPaths


def _path(value: str) -> Path:
    return Path(value).expanduser().resolve()


def build_parser() -> argparse.ArgumentParser:
    paths = ProjectPaths.defaults()
    parser = argparse.ArgumentParser(
        prog="sentiment-analyzer",
        description="Train and run five-class app-review sentiment models.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    download = subparsers.add_parser("download", help="download the public source dataset")
    download.add_argument("--output", type=_path, default=paths.raw_dataset)

    prepare = subparsers.add_parser("prepare", help="clean raw Parquet data with Spark")
    prepare.add_argument("--input", type=_path, default=paths.raw_dataset)
    prepare.add_argument("--output", type=_path, default=paths.processed_dataset)

    eda = subparsers.add_parser("eda", help="regenerate exploratory analysis figures")
    eda.add_argument("--input", type=_path, default=paths.raw_dataset)
    eda.add_argument("--output-dir", type=_path, default=paths.root / "docs/assets")

    embeddings = subparsers.add_parser("embeddings", help="train the FastText vectorizer")
    embeddings.add_argument("--dataset", type=_path, default=paths.processed_dataset)
    embeddings.add_argument("--output", type=_path, default=paths.vectorizer)
    embeddings.add_argument("--corpus", type=_path, default=paths.corpus)
    embeddings.add_argument("--dimension", type=int, default=130)
    embeddings.add_argument("--epochs", type=int, default=5)
    embeddings.add_argument("--workers", type=int)

    train = subparsers.add_parser("train", help="train and save the final classifier")
    train.add_argument("--dataset", type=_path, default=paths.processed_dataset)
    train.add_argument("--vectorizer", type=_path, default=paths.vectorizer)
    train.add_argument("--output", type=_path, default=paths.model)
    train.add_argument("--architecture", choices=("gru", "lstm"), default="gru")
    train.add_argument("--epochs", type=int, default=5)
    train.add_argument("--batch-size", type=int, default=128)
    train.add_argument("--no-class-weights", action="store_true")

    evaluate = subparsers.add_parser("evaluate", help="run stratified cross-validation")
    evaluate.add_argument("--dataset", type=_path, default=paths.processed_dataset)
    evaluate.add_argument("--output-dir", type=_path, default=paths.evaluation_dir)
    evaluate.add_argument(
        "--architectures", nargs="+", choices=("gru", "lstm"), default=("lstm", "gru")
    )
    evaluate.add_argument("--folds", type=int, default=5)
    evaluate.add_argument("--epochs", type=int, default=1)
    evaluate.add_argument("--batch-size", type=int, default=128)

    predict = subparsers.add_parser("predict", help="classify one review")
    predict.add_argument("--text", help="review text; prompts interactively when omitted")
    predict.add_argument("--vectorizer", type=_path, default=paths.vectorizer)
    predict.add_argument("--model", type=_path, default=paths.model)
    predict.add_argument("--json", action="store_true", dest="as_json")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    if args.command == "download":
        from sentiment_analyzer.data import download_dataset

        print(f"Downloaded dataset to {download_dataset(args.output)}")
    elif args.command == "prepare":
        from sentiment_analyzer.data import prepare_dataset

        dataset = prepare_dataset(args.input, args.output)
        print(f"Prepared {len(dataset):,} reviews at {args.output}")
    elif args.command == "eda":
        from sentiment_analyzer.eda import generate_eda

        print(json.dumps(generate_eda(args.input, args.output_dir), indent=2))
    elif args.command == "embeddings":
        from sentiment_analyzer.training import train_embeddings

        train_embeddings(
            args.dataset,
            args.output,
            corpus_path=args.corpus,
            vector_size=args.dimension,
            workers=args.workers,
            epochs=args.epochs,
        )
        print(f"Saved vectorizer to {args.output}")
    elif args.command == "train":
        from sentiment_analyzer.training import train_final_model

        _, metadata = train_final_model(
            args.dataset,
            args.vectorizer,
            args.output,
            config=ModelConfig(architecture=args.architecture),
            batch_size=args.batch_size,
            epochs=args.epochs,
            balance_classes=not args.no_class_weights,
        )
        print(json.dumps(metadata, indent=2))
    elif args.command == "evaluate":
        from sentiment_analyzer.evaluation import cross_validate

        results = cross_validate(
            args.dataset,
            args.output_dir,
            architectures=tuple(args.architectures),
            folds=args.folds,
            epochs=args.epochs,
            batch_size=args.batch_size,
        )
        print(json.dumps(results, indent=2))
    elif args.command == "predict":
        from sentiment_analyzer.inference import SentimentPredictor

        review = args.text or input("Review: ")
        result = SentimentPredictor.from_artifacts(args.vectorizer, args.model).predict(review)
        if args.as_json:
            print(json.dumps(result.to_dict(), indent=2))
        else:
            print(f"{result.rating}/5 — {result.sentiment} (confidence: {result.confidence:.1%})")
