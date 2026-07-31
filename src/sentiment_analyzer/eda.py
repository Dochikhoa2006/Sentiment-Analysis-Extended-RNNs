"""Exploratory plots derived from the raw Parquet dataset."""

from __future__ import annotations

from pathlib import Path


def generate_eda(source: Path, output_dir: Path) -> dict[str, int]:
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        import squarify
        from pyspark.sql import SparkSession
        from pyspark.sql import functions as sql
    except ImportError as exc:
        raise RuntimeError("Install data dependencies with: pip install -e '.[data]'") from exc

    if not source.exists():
        raise FileNotFoundError(f"raw dataset not found: {source}")
    output_dir.mkdir(parents=True, exist_ok=True)
    spark = SparkSession.builder.master("local[*]").appName("review-eda").getOrCreate()
    try:
        dataset = spark.read.parquet(str(source))
        row_count = dataset.count()
        application_count = dataset.select("package_name").distinct().count()

        ratings = dataset.select("star").dropna().toPandas()["star"]
        figure, axis = plt.subplots(figsize=(10, 5))
        sns.countplot(x=ratings, hue=ratings, palette="Blues", legend=False, ax=axis)
        axis.set(title="Rating distribution", xlabel="Star rating", ylabel="Reviews")
        figure.tight_layout()
        figure.savefig(output_dir / "rating_distribution.png", dpi=160)
        plt.close(figure)

        periods = (
            dataset.withColumn("date", sql.to_timestamp("date"))
            .dropna(subset=["date"])
            .groupBy(sql.window("date", "180 days"))
            .count()
            .orderBy("window.start")
            .select(sql.col("window.start").alias("period"), "count")
            .toPandas()
        )
        figure, axis = plt.subplots(figsize=(12, 5))
        axis.plot(periods["period"], periods["count"], marker="o", color="#2563eb")
        axis.set(title="Review volume by 180-day period", xlabel="Period", ylabel="Reviews")
        figure.autofmt_xdate()
        figure.tight_layout()
        figure.savefig(output_dir / "review_volume_by_period.png", dpi=160)
        plt.close(figure)

        applications = (
            dataset.groupBy("package_name").count().orderBy(sql.desc("count")).limit(30).toPandas()
        )
        figure, axis = plt.subplots(figsize=(14, 8))
        squarify.plot(
            sizes=applications["count"],
            label=applications["package_name"],
            alpha=0.85,
            ax=axis,
        )
        axis.set_title("Top 30 applications by review count")
        axis.axis("off")
        figure.tight_layout()
        figure.savefig(output_dir / "application_density.png", dpi=160)
        plt.close(figure)
    finally:
        spark.stop()

    return {"rows": row_count, "applications": application_count}
