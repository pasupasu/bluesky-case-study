"""Hand-check of ISR-ISR pairs before any ISR-ISR result is reported (analysis plan, analysis 2).

    uv run python -m bluesky_case_study.isr_check export          # 50 pairs -> data/annotation/isr_pairs.xlsx
    uv run python -m bluesky_case_study.isr_check export --n 100  # grow it: the first 50 stay the same
    uv run python -m bluesky_case_study.isr_check score           # after coding

Every pair in the sheet was labelled pro_israeli on both sides by Haiku. For each post, code
its side as the codebook defines it: ISR (pro_israeli), PAL (pro_palestinian) or other (any
other label). The sheet shows post text only: no author, community or Haiku reasoning.

Pairs are drawn uniformly from all ISR-ISR pairs in data/pairs_stance.parquet. The order comes
from sha256(SEED + both post ids), so a larger --n only adds pairs. The key file records how
many ISR-ISR contrasts each pair contributes, so the score can also be weighted the way the
model weights pairs.
"""

import argparse
import hashlib
import math
import re
import sys

import polars as pl
import xlsxwriter

from bluesky_case_study.paths import DATA_DIR

SEED = "isr-pairs-v1"  # do not change once coding starts
N_PAIRS = 50
SIDES = ["ISR", "PAL", "other"]
OUT_DIR = DATA_DIR / "annotation"
SHEET = OUT_DIR / "isr_pairs.xlsx"
KEY = OUT_DIR / "isr_pairs_key.parquet"
_ILLEGAL_XLSX = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")

INSTRUCTIONS = [
    ["what", "how"],
    ["task", "Code each post's side on its own, as in the codebook: ISR = pro_israeli, "
             "PAL = pro_palestinian, other = any other label (unclear, neutral_news, "
             "anti_war_humanitarian, critical_of_both, unrelated)."],
    ["pairs", "Both posts in every row were labelled pro_israeli by the model. Judge each post "
              "independently; don't let one post decide the other."],
    ["confidence", "1 = unsure, 2 = fairly sure, 3 = certain (for the row as a whole)."],
    ["blind", "No author, community or model reasoning is shown on purpose. Don't look them up."],
]


def _rank(a: int, b: int) -> str:
    lo, hi = sorted((a, b))
    return hashlib.sha256(f"{SEED}:{lo}:{hi}".encode()).hexdigest()


def draw(n: int) -> pl.DataFrame:
    # Only the columns the key needs: pairs_stance also has the model's side_a / side_b,
    # which must not collide with the hand-coded columns of the same name.
    pairs = pl.read_parquet(DATA_DIR / "pairs_stance.parquet").filter(pl.col("side_pair") == "ISR-ISR")
    pairs = pairs.select("post_a", "post_b", "sim", "user_a", "user_b", "confidence_a", "confidence_b",
                         "uncertain", "short", "near_duplicate", "linked")
    pairs = pairs.with_columns(
        pl.struct("post_a", "post_b").map_elements(lambda s: _rank(s["post_a"], s["post_b"]),
                                                   return_dtype=pl.String).alias("rank")
    ).sort("rank").head(n)

    # How many ISR-ISR contrasts (pair x community) each drawn pair contributes to the model.
    c = pl.read_parquet(DATA_DIR / "reception" / "contrasts.parquet",
                        columns=["side_pair", "in_post", "out_post"]).filter(pl.col("side_pair") == "ISR-ISR")
    c = c.select(pl.min_horizontal("in_post", "out_post").alias("lo"),
                 pl.max_horizontal("in_post", "out_post").alias("hi")).group_by("lo", "hi").len("contrasts")
    pairs = (pairs.with_columns(pl.min_horizontal("post_a", "post_b").alias("lo"),
                                pl.max_horizontal("post_a", "post_b").alias("hi"))
             .join(c, on=["lo", "hi"], how="left").drop("lo", "hi")
             .with_columns(pl.col("contrasts").fill_null(0)))

    text = pl.read_parquet(DATA_DIR / "ip_corpus.parquet", columns=["post_id", "text"])
    return (pairs.join(text.rename({"post_id": "post_a", "text": "text_a"}), on="post_a", how="left")
                 .join(text.rename({"post_id": "post_b", "text": "text_b"}), on="post_b", how="left")
                 .with_columns(pl.int_range(1, pl.len() + 1).alias("pair"))
                 .sort("pair"))


def export(n: int, force: bool) -> None:
    sample = draw(n)
    if SHEET.exists():
        old = _read()
        coded = old.filter(pl.any_horizontal(pl.col("side_a", "side_b", "notes").is_not_null()))
        if coded.height and not force:
            sys.exit(f"{SHEET} already has {coded.height} coded rows: pass --force to rewrite it "
                     "(coded rows are carried over by pair).")
        if n < old.height:
            sys.exit(f"--n {n} is smaller than the existing sheet ({old.height} pairs)")
        sample = sample.join(old.select("pair", "side_a", "side_b", "confidence", "notes"),
                             on="pair", how="left")
    else:
        sample = sample.with_columns(pl.lit(None, pl.String).alias(c)
                                     for c in ("side_a", "side_b", "confidence", "notes"))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    wb = xlsxwriter.Workbook(str(SHEET))
    ws = wb.add_worksheet("pairs")
    header = wb.add_format({"bold": True, "bg_color": "#DDDDDD", "border": 1})
    wrap = wb.add_format({"text_wrap": True, "valign": "top"})
    top = wb.add_format({"valign": "top"})
    entry = wb.add_format({"valign": "top", "bg_color": "#FFF8DC"})
    cols = ["pair", "text_a", "text_b", "side_a", "side_b", "confidence", "notes"]
    for i, name in enumerate(cols):
        ws.write_string(0, i, name, header)
    ws.set_column("A:A", 5, top)
    ws.set_column("B:C", 60, wrap)
    ws.set_column("D:E", 9, entry)
    ws.set_column("F:F", 11, entry)
    ws.set_column("G:G", 40, entry)
    ws.freeze_panes(1, 0)
    for r, row in enumerate(sample.select(cols).iter_rows(), start=1):
        ws.write_number(r, 0, row[0])
        ws.write_string(r, 1, _ILLEGAL_XLSX.sub("", row[1] or ""))
        ws.write_string(r, 2, _ILLEGAL_XLSX.sub("", row[2] or ""))
        for c, v in enumerate(row[3:], start=3):
            if v is not None:
                ws.write_string(r, c, str(v), entry)
    last = sample.height
    ws.data_validation(1, 3, last, 4, {"validate": "list", "source": SIDES})
    ws.data_validation(1, 5, last, 5, {"validate": "list", "source": ["1", "2", "3"],
                                       "input_title": "Confidence",
                                       "input_message": "1 = unsure, 2 = fairly sure, 3 = certain"})
    info = wb.add_worksheet("instructions")
    for r, row in enumerate(INSTRUCTIONS):
        for c, v in enumerate(row):
            info.write_string(r, c, v, header if r == 0 else wrap)
    info.set_column(0, 0, 14)
    info.set_column(1, 1, 100)
    wb.close()

    sample.select("pair", "post_a", "post_b", "sim", "user_a", "user_b", "confidence_a", "confidence_b",
                  "uncertain", "short", "near_duplicate", "linked", "contrasts", "rank").write_parquet(KEY)
    print(f"Wrote {SHEET} ({sample.height} pairs, {sample['contrasts'].gt(0).sum()} of them in ISR-ISR "
          f"contrasts) and {KEY}")


def _read() -> pl.DataFrame:
    df = pl.read_excel(SHEET, sheet_name="pairs", schema_overrides={
        c: pl.String for c in ("side_a", "side_b", "confidence", "notes")})
    return df.with_columns(pl.col("pair").cast(pl.Int64),
                           *[pl.col(c).str.strip_chars() for c in ("side_a", "side_b")])


def _wilson(k: int, n: int) -> str:
    if n == 0:
        return "n/a"
    p, z = k / n, 1.96
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return f"{p:.0%} (95% CI {max(0, centre - half):.0%}-{min(1, centre + half):.0%}, n={n})"


def score() -> None:
    df = _read().join(pl.read_parquet(KEY), on="pair", how="left", validate="1:1")
    bad = df.filter(pl.col("side_a").is_not_null() & ~pl.col("side_a").is_in(SIDES)
                    | pl.col("side_b").is_not_null() & ~pl.col("side_b").is_in(SIDES))
    if bad.height:
        sys.exit(f"unknown side values in rows {bad['pair'].to_list()}: use {SIDES}")
    done = df.filter(pl.col("side_a").is_not_null() & pl.col("side_b").is_not_null())
    print(f"{done.height} of {df.height} pairs coded")
    if not done.height:
        return
    both = (pl.col("side_a") == "ISR") & (pl.col("side_b") == "ISR")
    flip = ((pl.col("side_a") == "PAL") | (pl.col("side_b") == "PAL"))
    posts = pl.concat([done["side_a"], done["side_b"]])
    print("Both posts truly ISR (pair is a true stance match):", _wilson(done.filter(both).height, done.height))
    print("At least one post truly PAL (opposite side):     ", _wilson(done.filter(flip).height, done.height))
    print("Posts truly ISR (per post):                      ", _wilson(int((posts == "ISR").sum()), posts.len()))
    print("Per-post sides:", dict(posts.value_counts().iter_rows()))
    w = done.filter(pl.col("contrasts") > 0)
    if w.height:
        share = w.filter(both)["contrasts"].sum() / w["contrasts"].sum()
        print(f"Weighted by ISR-ISR contrasts ({w.height} pairs): {share:.0%} both ISR (descriptive, no CI)")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export")
    e.add_argument("--n", type=int, default=N_PAIRS)
    e.add_argument("--force", action="store_true", help="rewrite a sheet that already has codes (keeps them)")
    sub.add_parser("score")
    a = p.parse_args()
    export(a.n, a.force) if a.cmd == "export" else score()


if __name__ == "__main__":
    main()
