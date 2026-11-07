"""Build the English Israel/Palestine corpus from the full post dataset.

Usage:
    uv run python -m bluesky_case_study.corpus

Writes data/ip_corpus.parquet with two flag columns:
    kw_core     matches an unambiguous I/P term (tier 1)
    kw_context  matches a term that is often, but not always, about I/P (tier 2)

Changing CORE or CONTEXT changes the corpus, so rebuild it and re-export the
annotation samples afterwards (posts keep their place in the sampling order,
see annotation.py, so most of the samples stay the same).
"""

import polars as pl

from bluesky_case_study.load import load_posts
from bluesky_case_study.paths import DATA_DIR
from bluesky_case_study.search_codes import CORE, CONTEXT


CORPUS_PATH = DATA_DIR / "ip_corpus.parquet"


def build_corpus() -> pl.DataFrame:
    df = (
        load_posts()
        .filter(pl.col("langs").list.contains("eng"))
        .with_columns(
            pl.col("text").str.contains(CORE).alias("kw_core"),
            pl.col("text").str.contains(CONTEXT).alias("kw_context"),
        )
        .filter(pl.col("kw_core") | pl.col("kw_context"))
        # The source files repeat some posts; dedupe after filtering (much cheaper).
        .unique("post_id")
        .sort("post_id")
        .collect(engine="streaming")
    )
    df.write_parquet(CORPUS_PATH, compression="zstd")
    return df


def load_corpus() -> pl.LazyFrame:
    return pl.scan_parquet(CORPUS_PATH)


def main() -> None:
    df = build_corpus()
    counts = df.select(
        pl.len().alias("posts"),
        pl.col("kw_core").sum().alias("core"),
        (pl.col("kw_context") & ~pl.col("kw_core")).sum().alias("context_only"),
        pl.col("user_id").n_unique().alias("authors"),
    ).row(0, named=True)
    print(f"Wrote {CORPUS_PATH}: {counts}")


if __name__ == "__main__":
    main()
