"""Build the English Israel/Palestine corpus from the full post dataset.

Usage:
    uv run python -m bluesky_case_study.corpus

Writes data/ip_corpus.parquet, original posts only, with two flag columns:
    kw_core     matches an unambiguous I/P term (tier 1)
    kw_context  matches a term that is often, but not always, about I/P (tier 2)

Reposts go to data/ip_reposts.parquet instead. In this dataset a repost is a
new row (the reposter's post_id and user_id) that copies the original's text,
counts and date, so it would count the same text many times. Each repost row
records who reposted which post (repost_from), which is reception data. Note
its date is the original's date, not when the repost happened.

Changing CORE or CONTEXT changes the corpus, so rebuild it and re-export the
annotation samples afterwards (posts keep their place in the sampling order,
see annotation.py, so most of the samples stay the same).
"""

import polars as pl

from bluesky_case_study.load import load_posts
from bluesky_case_study.paths import DATA_DIR
from bluesky_case_study.search_codes import CORE, CONTEXT


CORPUS_PATH = DATA_DIR / "ip_corpus.parquet"
REPOSTS_PATH = DATA_DIR / "ip_reposts.parquet"


def build_corpus() -> tuple[pl.DataFrame, pl.DataFrame]:
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
    is_repost = pl.col("repost_from").is_not_null()
    originals = df.filter(~is_repost)
    reposts = df.filter(is_repost).select(
        pl.col("post_id"),
        pl.col("user_id").alias("reposter_id"),
        pl.col("repost_from").alias("original_post_id"),
        pl.col("reposted_author").alias("original_author_id"),
        # True when the original post is in the corpus (its author is in the dataset)
        pl.col("repost_from").is_in(originals["post_id"].implode()).alias("original_in_corpus"),
    )
    originals.write_parquet(CORPUS_PATH, compression="zstd")
    reposts.write_parquet(REPOSTS_PATH, compression="zstd")
    return originals, reposts


def load_corpus() -> pl.LazyFrame:
    """Original I/P posts (no reposts)."""
    return pl.scan_parquet(CORPUS_PATH)


def load_reposts() -> pl.LazyFrame:
    """Who reposted which I/P post: reposter_id -> original_post_id."""
    return pl.scan_parquet(REPOSTS_PATH)


def main() -> None:
    originals, reposts = build_corpus()
    counts = originals.select(
        pl.len().alias("posts"),
        pl.col("kw_core").sum().alias("core"),
        (pl.col("kw_context") & ~pl.col("kw_core")).sum().alias("context_only"),
        pl.col("user_id").n_unique().alias("authors"),
    ).row(0, named=True)
    print(f"Wrote {CORPUS_PATH}: {counts}")
    r = reposts.select(
        pl.len().alias("reposts"),
        pl.col("original_in_corpus").sum().alias("of_posts_in_corpus"),
        pl.col("reposter_id").n_unique().alias("reposters"),
    ).row(0, named=True)
    print(f"Wrote {REPOSTS_PATH}: {r}")


if __name__ == "__main__":
    main()
