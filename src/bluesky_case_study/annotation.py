"""Draw the hand-annotation samples from the I/P corpus and export them to Excel.

Usage:
    uv run python -m bluesky_case_study.annotation
    uv run python -m bluesky_case_study.annotation --test-per-tier 150   # grow the test set

Writes to data/annotation/:
    dev.xlsx, test.xlsx          sheets to code (post text only, no author info)
    dev_key.parquet, ...         post_id -> author, tier, sampling weight, engagement
    manifest.json                seed, sizes and keyword patterns used for each split

How the dev and test sets stay separate and reproducible
--------------------------------------------------------
Each post gets a fingerprint: sha256(SEED + post_id). That depends only on the
post, not on row order, polars version or what else is in the corpus.
  * The first hex digits put every post in the dev pool or the test pool, based
    on whether the first 8 digits form an even or odd number, so the two sets
    can never overlap, whatever sizes you ask for.
  * The rest of the fingerprint orders the posts within each pool. A sample of
    size N is the first N posts of that pool (per tier), so asking for more
    later only adds posts and never changes the ones you've already coded.

Existing .xlsx files are never overwritten (they hold your labels) unless you
pass --force. To grow a set, rename the old file, export again, and copy your
labels across by post_id.
"""

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone

import polars as pl
import xlsxwriter

from bluesky_case_study.corpus import load_corpus
from bluesky_case_study.search_codes import CONTEXT, CORE
from bluesky_case_study.paths import DATA_DIR

OUT_DIR = DATA_DIR / "annotation"

# DO NOT TOUCH ONCE CODING STARTS
SEED = "ip-annotation-v1"

DEV_PER_TIER = 50  # 100 posts total
TEST_PER_TIER = 100  # 200 posts total

# Starting categories for the dropdown.
STANCES = [
    "pro_palestinian",
    "pro_israeli",
    "anti_war_humanitarian",
    "critical_of_both",
    "neutral_news",
    "unclear",
    "unrelated",
]

# Characters that aren't allowed in .xlsx files (the archive has a few).
_ILLEGAL_XLSX = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


def _fingerprint(post_id: int) -> str:
    return hashlib.sha256(f"{SEED}:{post_id}".encode()).hexdigest()


def assign_pools(corpus: pl.DataFrame) -> pl.DataFrame:
    """Add tier, pool (dev/test) and rank columns. Depends only on each post_id."""
    return corpus.with_columns(
        pl.when(pl.col("kw_core"))
        .then(pl.lit("core"))
        .otherwise(pl.lit("context"))
        .alias("tier"),
        pl.col("post_id")
        .map_elements(_fingerprint, return_dtype=pl.String)
        .alias("_fp"),
    ).with_columns(
        # Even first 8 hex digits -> dev pool, odd -> test pool (a 50/50 split).
        pl.when(pl.col("_fp").str.slice(0, 8).str.to_integer(base=16) % 2 == 0)
        .then(pl.lit("dev"))
        .otherwise(pl.lit("test"))
        .alias("pool"),
        pl.col("_fp").str.slice(8).alias("rank"),
    ).drop("_fp")


def draw_sample(pooled: pl.DataFrame, split: str, per_tier: int) -> pl.DataFrame:
    """The first per_tier posts of each tier in this split's pool, plus weights."""
    pool = pooled.filter(pl.col("pool") == split)
    tier_sizes = pooled.group_by("tier").len().rename({"len": "tier_size"})
    sample = (
        pool.sort("rank")
        .group_by("tier", maintain_order=True)
        .head(per_tier)
        .join(tier_sizes, on="tier")
    )
    short = sample.group_by("tier").len().filter(pl.col("len") < per_tier)
    if short.height:
        print(f"warning: {split} pool has fewer posts than asked for: {short.to_dicts()}")
    # Each sampled post stands for tier_size / n_sampled posts in the corpus.
    return sample.with_columns(
        (pl.col("tier_size") / pl.len().over("tier")).alias("sampling_weight"),
        pl.lit(split).alias("split"),
    ).sort("rank")  # mixes the tiers, so the sheet order gives no hint of the tier


def write_sheet(sample: pl.DataFrame, path) -> None:
    wb = xlsxwriter.Workbook(str(path))
    ws = wb.add_worksheet("posts")
    lists = wb.add_worksheet("stances")

    header = wb.add_format({"bold": True, "bg_color": "#DDDDDD", "border": 1})
    wrap = wb.add_format({"text_wrap": True, "valign": "top"})
    top = wb.add_format({"valign": "top"})
    entry = wb.add_format({"valign": "top", "bg_color": "#FFF8DC"})

    columns = ["row", "post_id", "text", "stance", "confidence", "notes"]
    for c, name in enumerate(columns):
        ws.write_string(0, c, name, header)
    ws.set_column("A:A", 5, top)
    ws.set_column("B:B", 12, top)
    ws.set_column("C:C", 90, wrap)
    ws.set_column("D:D", 24, entry)
    ws.set_column("E:E", 11, entry)
    ws.set_column("F:F", 40, entry)
    ws.freeze_panes(1, 0)

    for r, (post_id, text) in enumerate(
        sample.select("post_id", "text").iter_rows(), start=1
    ):
        ws.write_number(r, 0, r)
        # write_string so posts starting with "=" or containing URLs stay plain text
        ws.write_string(r, 1, str(post_id))
        ws.write_string(r, 2, _ILLEGAL_XLSX.sub("", text or ""))

    # Stance options live on their own tab (with spare rows) so you can add
    # categories in Excel and the dropdown picks them up.
    lists.write_string(0, 0, "stance options (edit freely)", header)
    for i, s in enumerate(STANCES, start=1):
        lists.write_string(i, 0, s)
    lists.set_column("A:A", 30)

    last = sample.height
    ws.data_validation(
        1, 3, last, 3,
        {"validate": "list", "source": "=stances!$A$2:$A$30", "ignore_blank": True},
    )
    ws.data_validation(
        1, 4, last, 4,
        {
            "validate": "list",
            "source": ["1", "2", "3"],
            "input_title": "Confidence",
            "input_message": "1 = unsure, 2 = fairly sure, 3 = certain",
        },
    )
    wb.close()


def _update_manifest(split: str, per_tier: int, sample: pl.DataFrame, corpus_size: int) -> None:
    path = OUT_DIR / "manifest.json"
    manifest = json.loads(path.read_text()) if path.exists() else {}
    manifest[split] = {
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seed": SEED,
        "per_tier": per_tier,
        "n_posts": sample.height,
        "corpus_size": corpus_size,
        "tier_sizes": dict(sample.select("tier", "tier_size").unique().iter_rows()),
        "core_pattern": CORE,
        "context_pattern": CONTEXT,
        "polars_version": pl.__version__,
    }
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))


KEY_COLUMNS = [
    "post_id", "split", "tier", "rank", "sampling_weight", "tier_size",
    "user_id", "date", "like_count", "reply_count", "repost_count",
    "reply_to", "repost_from", "quotes",
]


def export(dev_per_tier: int = DEV_PER_TIER, test_per_tier: int = TEST_PER_TIER, force: bool = False) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    corpus = load_corpus().collect()
    pooled = assign_pools(corpus)

    samples = {}
    for split, per_tier in (("dev", dev_per_tier), ("test", test_per_tier)):
        sample = draw_sample(pooled, split, per_tier)
        samples[split] = sample
        sheet = OUT_DIR / f"{split}.xlsx"
        if sheet.exists() and not force:
            print(f"skipped {sheet.name}: already exists (pass --force to overwrite your labels)")
            continue
        write_sheet(sample, sheet)
        sample.select(KEY_COLUMNS).write_parquet(OUT_DIR / f"{split}_key.parquet")
        _update_manifest(split, per_tier, sample, corpus.height)
        print(f"wrote {sheet.name} and {split}_key.parquet: {sample.height} posts")

    overlap = set(samples["dev"]["post_id"]) & set(samples["test"]["post_id"])
    assert not overlap, f"dev and test overlap on {len(overlap)} posts"


def load_labels(split: str) -> pl.DataFrame:
    """Read your coded sheet and join the hidden key back on by post_id."""
    labels = pl.read_excel(
        OUT_DIR / f"{split}.xlsx",
        sheet_name="posts",
        schema_overrides={
            "post_id": pl.String,
            "stance": pl.String,
            "confidence": pl.String,
            "notes": pl.String,
        },
    ).with_columns(pl.col("post_id").cast(pl.Int64))
    key = pl.read_parquet(OUT_DIR / f"{split}_key.parquet")
    return labels.join(key, on="post_id", how="left", validate="1:1")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dev-per-tier", type=int, default=DEV_PER_TIER)
    parser.add_argument("--test-per-tier", type=int, default=TEST_PER_TIER)
    parser.add_argument("--force", action="store_true", help="overwrite existing sheets")
    args = parser.parse_args()
    export(args.dev_per_tier, args.test_per_tier, args.force)


if __name__ == "__main__":
    main()
