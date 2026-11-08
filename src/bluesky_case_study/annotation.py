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
pass --force. To grow or redraw a set without losing work, move the old files
into a folder and pass it with --carry-over:

    uv run python -m bluesky_case_study.annotation --carry-over data/annotation/archive/v1_with_reposts

Labels from the old sheets are filled into the new ones by post_id (a label on
a repost moves to its original post). Labeled posts that aren't in the new
sample are written to {split}_extras.xlsx, so they can still serve as codebook
examples, and the export stops if any post you've already seen would land in
the test set.
"""

from pathlib import Path

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

DEV_PER_TIER = 100  # 200 posts total (was 50 until 2026-11-08; the first 50 per tier are unchanged)
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


LABEL_COLUMNS = ["stance", "confidence", "notes"]
SHEET_COLUMNS = ["row", "post_id", "text", *LABEL_COLUMNS]


def write_sheet(
    sample: pl.DataFrame,
    path,
    prefill: pl.DataFrame | None = None,
    stances: list[str] | None = None,
    extra_tabs: dict[str, list[list]] | None = None,
) -> None:
    """Write a coding sheet.

    prefill (post_id + LABEL_COLUMNS, plus any extra columns such as
    suggested_stance) fills in old labels; extra columns go after notes.
    stances overrides the dropdown options, and extra_tabs (name -> rows of
    cell values) are copied in as extra worksheets, so a regrown sheet keeps
    everything that was added to the old one in Excel.
    """
    if prefill is not None:
        sample = sample.join(prefill, on="post_id", how="left")
    else:
        sample = sample.with_columns(pl.lit(None, pl.String).alias(c) for c in LABEL_COLUMNS)
    extra_cols = [c for c in (prefill.columns if prefill is not None else [])
                  if c != "post_id" and c not in LABEL_COLUMNS]

    wb = xlsxwriter.Workbook(str(path))
    ws = wb.add_worksheet("posts")
    lists = wb.add_worksheet("stances")

    header = wb.add_format({"bold": True, "bg_color": "#DDDDDD", "border": 1})
    wrap = wb.add_format({"text_wrap": True, "valign": "top"})
    top = wb.add_format({"valign": "top"})
    entry = wb.add_format({"valign": "top", "bg_color": "#FFF8DC"})

    columns = [*SHEET_COLUMNS, *extra_cols]
    for c, name in enumerate(columns):
        ws.write_string(0, c, name, header)
    ws.set_column("A:A", 5, top)
    ws.set_column("B:B", 12, top)
    ws.set_column("C:C", 90, wrap)
    ws.set_column("D:D", 24, entry)
    ws.set_column("E:E", 11, entry)
    ws.set_column("F:F", 40, entry)
    for i in range(len(extra_cols)):
        ws.set_column(6 + i, 6 + i, 40, wrap)
    ws.freeze_panes(1, 0)

    for r, (post_id, text, *labels) in enumerate(
        sample.select("post_id", "text", *LABEL_COLUMNS, *extra_cols).iter_rows(), start=1
    ):
        ws.write_number(r, 0, r)
        # write_string so posts starting with "=" or containing URLs stay plain text
        ws.write_string(r, 1, str(post_id))
        ws.write_string(r, 2, _ILLEGAL_XLSX.sub("", text or ""))
        for c, value in enumerate(labels, start=3):
            if value is not None:
                ws.write_string(r, c, str(value), entry if c < 6 else wrap)

    # Stance options live on their own tab (with spare rows) so you can add
    # categories in Excel and the dropdown picks them up.
    lists.write_string(0, 0, "stance options (edit freely)", header)
    for i, s in enumerate(stances or STANCES, start=1):
        lists.write_string(i, 0, s)
    lists.set_column("A:A", 30)

    for name, rows in (extra_tabs or {}).items():
        tab = wb.add_worksheet(name)
        for r, row in enumerate(rows):
            for c, value in enumerate(row):
                if value is not None:
                    tab.write_string(r, c, str(value), header if r == 0 else wrap)
        tab.set_column(0, 0, 30)
        tab.set_column(1, 1, 100)

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


def _read_old_labels(folder: Path, split: str) -> dict | None:
    """Labels from an older export, moved from reposts onto their originals.

    Returns a dict with:
        labels      post_id + LABEL_COLUMNS + any columns added to the sheet in Excel
        seen        every post_id shown in that sheet, including the originals
                    behind reposts, whether or not it was labeled
        stances     the dropdown options on the old stances tab
        extra_tabs  any other worksheets (e.g. review notes), as rows of values
    """
    path = folder / f"{split}.xlsx"
    if not path.exists():
        return None
    old = load_labels(split, folder)
    old = old.with_columns(
        pl.coalesce("repost_from", "post_id").alias("target_id")
    )
    seen = set(old["post_id"]) | set(old["target_id"])
    added = [c for c in old.columns
             if c not in SHEET_COLUMNS and c not in KEY_COLUMNS and c != "target_id"]
    kept = [*LABEL_COLUMNS, *added]
    has_label = pl.any_horizontal(pl.col(c).is_not_null() for c in kept)
    labels = (
        old.filter(has_label)
        .select(pl.col("target_id").alias("post_id"), *[pl.col(c).cast(pl.String) for c in kept])
        .unique("post_id", keep="first", maintain_order=True)
    )

    tabs = _read_tabs(path)
    stances = None
    if len(tabs.get("stances", [])) > 1:
        stances = [row[0] for row in tabs["stances"][1:] if row and row[0]]
    extra_tabs = {name: rows for name, rows in tabs.items() if name not in ("posts", "stances")}
    return {"labels": labels, "seen": seen, "stances": stances, "extra_tabs": extra_tabs}


def _read_tabs(path: Path) -> dict[str, list[list[str | None]]]:
    """Every worksheet as rows of cell text, read straight from the .xlsx XML.

    Used for small side tabs only. (fastexcel skipped the last row of one of
    them, so this doesn't rely on it.)
    """
    import zipfile
    import xml.etree.ElementTree as ET

    ns = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
          "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
          "p": "http://schemas.openxmlformats.org/package/2006/relationships"}
    with zipfile.ZipFile(path) as z:
        names = z.namelist()
        shared = []
        if "xl/sharedStrings.xml" in names:
            for si in ET.fromstring(z.read("xl/sharedStrings.xml")).findall("m:si", ns):
                shared.append("".join(t.text or "" for t in si.iter(f"{{{ns['m']}}}t")))
        rels = {r.get("Id"): r.get("Target")
                for r in ET.fromstring(z.read("xl/_rels/workbook.xml.rels")).findall("p:Relationship", ns)}
        tabs = {}
        for sheet in ET.fromstring(z.read("xl/workbook.xml")).find("m:sheets", ns):
            target = rels[sheet.get(f"{{{ns['r']}}}id")].lstrip("/")
            target = target if target.startswith("xl/") else f"xl/{target}"
            rows = []
            for row in ET.fromstring(z.read(target)).iter(f"{{{ns['m']}}}row"):
                cells = {}
                for c in row.findall("m:c", ns):
                    col = re.match(r"[A-Z]+", c.get("r")).group()
                    idx = 0
                    for ch in col:
                        idx = idx * 26 + ord(ch) - 64
                    if c.get("t") == "inlineStr":
                        value = "".join(t.text or "" for t in c.iter(f"{{{ns['m']}}}t"))
                    else:
                        v = c.find("m:v", ns)
                        value = None if v is None else (shared[int(v.text)] if c.get("t") == "s" else v.text)
                    cells[idx - 1] = value
                rows.append([cells.get(i) for i in range(max(cells, default=-1) + 1)])
            tabs[sheet.get("name")] = rows
    return tabs


def export(
    dev_per_tier: int = DEV_PER_TIER,
    test_per_tier: int = TEST_PER_TIER,
    force: bool = False,
    carry_over: Path | None = None,
) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    corpus = load_corpus().collect()
    pooled = assign_pools(corpus)

    samples = {split: draw_sample(pooled, split, n)
               for split, n in (("dev", dev_per_tier), ("test", test_per_tier))}
    overlap = set(samples["dev"]["post_id"]) & set(samples["test"]["post_id"])
    assert not overlap, f"dev and test overlap on {len(overlap)} posts"

    old = {}
    if carry_over is not None:
        old = {s: r for s in ("dev", "test") if (r := _read_old_labels(carry_over, s))}
        # Posts you've read while building the codebook must never be test items.
        if "dev" in old:
            leaked = old["dev"]["seen"] & set(samples["test"]["post_id"])
            if leaked:
                raise SystemExit(
                    f"stopped: {len(leaked)} post(s) from the old dev sheet would land "
                    f"in the new test set: {sorted(leaked)}"
                )

    for split, sample in samples.items():
        per_tier = dev_per_tier if split == "dev" else test_per_tier
        sheet = OUT_DIR / f"{split}.xlsx"
        if sheet.exists() and not force:
            print(f"skipped {sheet.name}: already exists (pass --force to overwrite your labels)")
            continue
        prev = old.get(split)
        prefill = prev["labels"] if prev else None
        write_sheet(sample, sheet, prefill,
                    stances=prev["stances"] if prev else None,
                    extra_tabs=prev["extra_tabs"] if prev else None)
        sample.select(KEY_COLUMNS).write_parquet(OUT_DIR / f"{split}_key.parquet")
        _update_manifest(split, per_tier, sample, corpus.height)
        print(f"wrote {sheet.name} and {split}_key.parquet: {sample.height} posts")

        if prefill is not None:
            placed = prefill.filter(pl.col("post_id").is_in(sample["post_id"].implode()))
            extras = prefill.join(sample.select("post_id"), on="post_id", how="anti")
            print(f"  carried over {placed.height} label(s) into {sheet.name}")
            if extras.height:
                # Labeled posts outside the new sample: keep them, readable, with text.
                extra_rows = pooled.select("post_id", "text").join(
                    extras.select("post_id"), on="post_id", how="semi"
                )
                write_sheet(extra_rows, OUT_DIR / f"{split}_extras.xlsx",
                            extras.select("post_id", *LABEL_COLUMNS))
                missing = extras.height - extra_rows.height
                print(
                    f"  {extra_rows.height} labeled post(s) not in the new sample "
                    f"-> {split}_extras.xlsx"
                    + (f" ({missing} no longer in the corpus, dropped)" if missing else "")
                )


def load_labels(split: str, folder: Path = OUT_DIR) -> pl.DataFrame:
    """Read your coded sheet and join the hidden key back on by post_id."""
    labels = pl.read_excel(
        folder / f"{split}.xlsx",
        sheet_name="posts",
        schema_overrides={
            "post_id": pl.String,
            "stance": pl.String,
            "confidence": pl.String,
            "notes": pl.String,
        },
    ).with_columns(pl.col("post_id").cast(pl.Int64))
    key = pl.read_parquet(folder / f"{split}_key.parquet")
    return labels.join(key, on="post_id", how="left", validate="1:1")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dev-per-tier", type=int, default=DEV_PER_TIER)
    parser.add_argument("--test-per-tier", type=int, default=TEST_PER_TIER)
    parser.add_argument("--force", action="store_true", help="overwrite existing sheets")
    parser.add_argument(
        "--carry-over", type=Path, default=None,
        help="folder with older dev/test sheets whose labels should be filled in",
    )
    args = parser.parse_args()
    export(args.dev_per_tier, args.test_per_tier, args.force, args.carry_over)


if __name__ == "__main__":
    main()
