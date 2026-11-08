"""Stance labelling of paired posts with Claude Haiku.

Every post is labelled once (not every pair). Two posts are a stance match when
their labels agree, so a pair is uncertain exactly when one of its posts is.
Labelling posts costs O(posts) instead of O(pairs), a post that sits in many
pairs is paid for once, and the dev/test gold labels are per post, so the
labels can be checked against them directly.

Only posts that appear in a content pair (data/pairs.parquet, from pairs.py)
are labelled, unless --all-posts is given.

Haiku gives each post a stance, a confidence of 1-3 and a runner-up label.
There is no second, stronger model: low-confidence labels are kept and marked,
so analyses can drop or down-weight them (see `uncertain` in pairs_stance).

Usage
    # 1. Check accuracy on the dev set (sync API, ~500 posts)
    uv run python -m bluesky_case_study.stance eval --split dev
    uv run python -m bluesky_case_study.stance eval --split dev --use-suggestions

    # 2. Label the paired posts with the Message Batches API (half price, async)
    #    (run `python -m bluesky_case_study.pairs embed` and `... pairs search` first)
    uv run python -m bluesky_case_study.stance submit [--limit N]
    uv run python -m bluesky_case_study.stance collect          # rerun until all batches end
    uv run python -m bluesky_case_study.stance finalize
    uv run python -m bluesky_case_study.stance pairs            # -> data/pairs_stance.parquet

    # 3. Score the frozen pipeline once on the test set (never tune on it)
    uv run python -m bluesky_case_study.stance eval --split test

The codebook is frozen (FROZEN_VERSION below): `eval --split test`, `submit` and
`finalize` stop if text/codebook.md has changed since. Check with `stance version`.

Every API answer is cached in data/stance/responses.jsonl, keyed by model,
prompt version and post, so reruns never pay twice. Editing the codebook
changes the prompt version, so posts are labelled again under the new rules.

Needs an API key: set ANTHROPIC_API_KEY (or run `ant auth login`).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import polars as pl

from bluesky_case_study.annotation import STANCES, load_labels
from bluesky_case_study.paths import DATA_DIR, PROJECT_ROOT

CODEBOOK_PATH = PROJECT_ROOT / "text" / "codebook.md"
OUT_DIR = DATA_DIR / "stance"
CACHE_PATH = OUT_DIR / "responses.jsonl"
BATCHES_PATH = OUT_DIR / "batches.json"
LABELS_PATH = OUT_DIR / "labels.parquet"
PAIR_STANCES_PATH = DATA_DIR / "pairs_stance.parquet"

MODEL = "claude-haiku-5-5"
# Thinking tokens count toward max_tokens, so leave room.
SETTINGS = {"max_tokens": 2048, "output_config": {"effort": "low"}}
# $ per million tokens: (input, output, cache read, cache write). Batches are half price.
PRICES = (0.10, 0.50, 0.01, 0.125)
BATCH_SIZE = 50_000  # requests per batch (the API allows up to 100k / 256 MB)

# The codebook was frozen on 2026-11-08 after tuning on the 200-post dev set
# (84% accuracy, kappa 0.79; per-post results in data/stance/eval_dev_frozen_<version>.parquet).
# `eval --split test` and `submit` refuse to run under any other prompt version, so the
# test score and the corpus labels always come from the codebook that was tuned on dev.
# The whole codebook is in the prompt, so ANY edit to text/codebook.md, even the
# changelog, changes the version. Pass --unfrozen to override and report the result as post hoc.
FROZEN_VERSION = "216304cab33e"


def check_frozen(allow_unfrozen: bool = False) -> None:
    version = prompt_version()
    if version == FROZEN_VERSION:
        return
    msg = (f"prompt version is {version}, but the codebook was frozen at {FROZEN_VERSION}. "
           "text/codebook.md (or the model settings) changed since the freeze.")
    if not allow_unfrozen:
        sys.exit(msg + " Revert the change, or pass --unfrozen and report results as post hoc.")
    print("WARNING: " + msg + " Running anyway (--unfrozen).")

# --------------------------------------------------------------------- prompt

INSTRUCTIONS = """You label the stance of Bluesky posts about the Israel-Palestine conflict.
Follow the codebook below exactly. It is the only authority on what each label means.

Each post arrives on its own, with no author, thread or link content. Code only what the text communicates, as the codebook says.

Answer with:
- reason: one or two short sentences naming the words or move in the post that decide the label (e.g. who is blamed, defended or mourned).
- stance: one label from the codebook.
- confidence: 3 = certain, 2 = fairly sure, 1 = unsure between labels (same scale as the codebook).
- runner_up: the next most plausible label, or "none" if no other label is plausible.

<codebook>
{codebook}
</codebook>"""

SCHEMA = {
    "type": "object",
    "properties": {
        "reason": {"type": "string"},
        "stance": {"type": "string", "enum": STANCES},
        "confidence": {"type": "integer", "enum": [1, 2, 3]},
        "runner_up": {"type": "string", "enum": [*STANCES, "none"]},
    },
    "required": ["reason", "stance", "confidence", "runner_up"],
    "additionalProperties": False,
}


def system_prompt() -> str:
    return INSTRUCTIONS.format(codebook=CODEBOOK_PATH.read_text(encoding="utf-8").strip())


def prompt_version() -> str:
    """Short hash of everything that affects answers except the post itself."""
    blob = json.dumps([system_prompt(), SCHEMA, MODEL, SETTINGS], sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


def request_params(text: str, system: str) -> dict:
    return {
        "model": MODEL,
        "max_tokens": SETTINGS["max_tokens"],
        # The codebook is identical for every post, so cache it.
        "system": [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
        "messages": [{"role": "user", "content": f"<post>\n{text}\n</post>"}],
        "output_config": {
            **SETTINGS["output_config"],
            "format": {"type": "json_schema", "schema": SCHEMA},
        },
    }


# ---------------------------------------------------------------- cache

_cache_lock = threading.Lock()


def _key(version: str, post_id: int) -> str:
    return f"{MODEL}|{version}|{post_id}|0"


def load_cache() -> dict[str, dict]:
    cache: dict[str, dict] = {}
    if CACHE_PATH.exists():
        with open(CACHE_PATH, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    cache[rec["key"]] = rec
    return cache


def _append_cache(records: list[dict]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with _cache_lock, open(CACHE_PATH, "a", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def parse_message(msg) -> dict:
    """Turn an API message into a cache record body (label or error)."""
    usage = msg.usage
    out = {
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "cache_read": usage.cache_read_input_tokens or 0,
        "cache_write": usage.cache_creation_input_tokens or 0,
    }
    if msg.stop_reason == "refusal":
        return {**out, "error": "refusal"}
    if msg.stop_reason == "max_tokens":
        return {**out, "error": "max_tokens"}
    text = next((b.text for b in msg.content if b.type == "text"), "")
    try:
        data = json.loads(text)
    except ValueError:
        return {**out, "error": "bad_json"}
    return {**out, **data}


def _record(version: str, pid: int, body: dict, batch: bool = False) -> dict:
    return {"key": _key(version, pid), "post_id": pid, "model": MODEL, "version": version,
            "batch": batch, **body}


# -------------------------------------------------------- synchronous runs

def run_sync(posts: pl.DataFrame, workers: int = 8, client=None) -> None:
    """Label posts (post_id, text) one request at a time. For small sets like dev/test."""
    import anthropic

    client = client or anthropic.Anthropic(max_retries=5)
    version = prompt_version()
    system = system_prompt()
    cache = load_cache()
    todo = [
        (pid, text) for pid, text in posts.select("post_id", "text").iter_rows()
        if _key(version, pid) not in cache
    ]
    if not todo:
        return
    print(f"Labelling {len(todo)} post(s) ({len(posts) - len(todo)} cached)")

    def one(pid: int, text: str) -> dict:
        msg = client.messages.create(**request_params(text, system))
        return _record(version, pid, parse_message(msg))

    with ThreadPoolExecutor(workers) as pool:
        futures = [pool.submit(one, pid, text) for pid, text in todo]
        for i, fut in enumerate(as_completed(futures), 1):
            _append_cache([fut.result()])
            if i % 25 == 0 or i == len(todo):
                print(f"  {i}/{len(todo)}", flush=True)


# ----------------------------------------------------------- batch runs

def _load_batches() -> list[dict]:
    return json.loads(BATCHES_PATH.read_text()) if BATCHES_PATH.exists() else []


def _save_batches(batches: list[dict]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    BATCHES_PATH.write_text(json.dumps(batches, indent=2))


def submit(posts: pl.DataFrame, client=None) -> None:
    """Send uncached posts to the Message Batches API. Run `collect` later."""
    import anthropic
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request

    client = client or anthropic.Anthropic()
    version = prompt_version()
    system = system_prompt()
    cache = load_cache()
    batches = _load_batches()
    pending = {
        pid for b in batches if b["version"] == version and not b.get("collected")
        for pid in b["post_ids"]
    }
    todo = [
        (pid, t) for pid, t in posts.select("post_id", "text").rows()
        if pid not in pending and _key(version, pid) not in cache
    ]
    print(f"{len(todo):,} post(s) to submit")

    for start in range(0, len(todo), BATCH_SIZE):
        chunk = todo[start : start + BATCH_SIZE]
        batch = client.messages.batches.create(requests=[
            Request(custom_id=f"p{pid}",
                    params=MessageCreateParamsNonStreaming(**request_params(text, system)))
            for pid, text in chunk
        ])
        batches.append({"id": batch.id, "model": MODEL, "version": version,
                        "post_ids": [pid for pid, _ in chunk]})
        _save_batches(batches)  # save after each one so a crash doesn't lose ids
        print(f"  submitted {batch.id} ({len(chunk):,} requests)")


def collect(client=None) -> None:
    """Download finished batches into the cache. Safe to rerun."""
    import anthropic

    client = client or anthropic.Anthropic()
    batches = _load_batches()
    for b in batches:
        if b.get("collected"):
            continue
        status = client.messages.batches.retrieve(b["id"])
        if status.processing_status != "ended":
            c = status.request_counts
            print(f"{b['id']}: {status.processing_status} ({c.processing:,} still processing)")
            continue
        records, failed = [], 0
        for res in client.messages.batches.results(b["id"]):
            pid = int(res.custom_id[1:])
            if res.result.type != "succeeded":
                failed += 1  # errored/expired/canceled: left uncached, resubmit picks it up
                continue
            records.append(_record(b["version"], pid, parse_message(res.result.message), batch=True))
        _append_cache(records)
        b["collected"] = True
        _save_batches(batches)
        print(f"{b['id']}: {len(records):,} saved, {failed:,} failed (resubmit to retry)")


# --------------------------------------------------------------- labels

def label_table(post_ids: list[int] | None = None) -> pl.DataFrame:
    """Cached labels under the current prompt, one row per post."""
    version = prompt_version()
    rows = [r for r in load_cache().values() if r["model"] == MODEL and r["version"] == version]
    cols = {
        "post_id": pl.Int64, "stance": pl.String, "confidence": pl.Int64,
        "runner_up": pl.String, "reason": pl.String, "error": pl.String,
        "input_tokens": pl.Int64, "output_tokens": pl.Int64,
        "cache_read": pl.Int64, "cache_write": pl.Int64, "batch": pl.Boolean,
    }
    df = pl.DataFrame([{c: r.get(c) for c in cols} for r in rows], schema=cols)
    if post_ids is not None:
        df = df.filter(pl.col("post_id").is_in(post_ids))
    return df.with_columns(
        _cost_expr().alias("cost"),
        # Refused or broken answers have no label: drop them or code them by hand.
        pl.col("stance").is_null().alias("needs_review"),
    ).select("post_id", "stance", "confidence", "runner_up", "reason", "error", "needs_review", "cost")


def _cost_expr() -> pl.Expr:
    p_in, p_out, p_read, p_write = PRICES
    full = (pl.col("input_tokens") * p_in + pl.col("output_tokens") * p_out
            + pl.col("cache_read") * p_read + pl.col("cache_write") * p_write) / 1e6
    return pl.when(pl.col("batch").fill_null(False)).then(full / 2).otherwise(full)


# ----------------------------------------------------------- evaluation

def gold_labels(split: str, use_suggestions: bool = False) -> pl.DataFrame:
    """Human labels with post text. Optionally take suggested_stance where present."""
    df = load_labels(split)
    if use_suggestions and "suggested_stance" in df.columns:
        df = df.with_columns(
            pl.coalesce(pl.col("suggested_stance").cast(pl.String), pl.col("stance")).alias("stance")
        )
    return df.filter(pl.col("stance").is_in(STANCES)).select("post_id", "text", pl.col("stance").alias("gold"))


def scores(pred: pl.Series, gold: pl.Series) -> dict:
    """Accuracy, macro-F1 over the labels present in gold, Cohen's kappa."""
    df = pl.DataFrame({"p": pred, "g": gold}).drop_nulls()
    n = len(df)
    if n == 0:
        return {"n": 0, "acc": None, "macro_f1": None, "kappa": None}
    acc = (df["p"] == df["g"]).mean()
    f1s = []
    for lab in df["g"].unique():
        tp = ((df["p"] == lab) & (df["g"] == lab)).sum()
        fp = ((df["p"] == lab) & (df["g"] != lab)).sum()
        fn = ((df["p"] != lab) & (df["g"] == lab)).sum()
        f1s.append(2 * tp / (2 * tp + fp + fn) if tp else 0.0)
    pe = sum((df["p"] == lab).sum() * (df["g"] == lab).sum() for lab in STANCES) / n**2
    kappa = (acc - pe) / (1 - pe) if pe < 1 else None
    return {"n": n, "acc": acc, "macro_f1": sum(f1s) / len(f1s), "kappa": kappa}


def evaluate(split: str, use_suggestions: bool = False, workers: int = 8) -> pl.DataFrame:
    gold = gold_labels(split, use_suggestions)
    if gold.is_empty():
        sys.exit(f"No labelled posts in data/annotation/{split}.xlsx yet: label some first.")
    run_sync(gold, workers)
    df = gold.join(label_table(gold["post_id"].to_list()), on="post_id")

    # Overall, then by confidence: shows how much is lost by keeping low-confidence labels,
    # and what dropping them would buy.
    rows = [{"subset": "all", "share": 1.0, **scores(df["stance"], df["gold"])}]
    for c in (3, 2, 1):
        sub = df.filter(pl.col("confidence") == c)
        rows.append({"subset": f"confidence = {c}", "share": len(sub) / len(df),
                     **scores(sub["stance"], sub["gold"])})
    keep = df.filter(pl.col("confidence") >= 2)
    rows.append({"subset": "confidence >= 2", "share": len(keep) / len(df),
                 **scores(keep["stance"], keep["gold"])})
    table = pl.DataFrame(rows)

    with pl.Config(tbl_rows=30, tbl_width_chars=140, float_precision=3):
        print(f"\n{split}: {len(gold)} gold-labelled posts"
              + (" (suggested_stance where present)" if use_suggestions else ""))
        print(table)
        print(f"\nNo label (refused/errored): {df['needs_review'].sum()}")
        # Cost per 1,000 posts at sync prices; batches halve it.
        print(f"Cost: ${df['cost'].sum():.3f} for this run, "
              f"${df['cost'].mean() * 1000:.3f} per 1k posts (sync), "
              f"${df['cost'].mean() * 500:.3f} per 1k posts (batch)")
        print("\nConfusion; rows = gold, columns = predicted")
        print(df.pivot("stance", index="gold", values="post_id", aggregate_function="len")
              .fill_null(0).sort("gold"))
    out = OUT_DIR / f"eval_{split}.parquet"
    df.write_parquet(out)
    print(f"\nPer-post predictions with reasons: {out}")
    return table


# ------------------------------------------------------------ corpus & pairs

def corpus_posts(limit: int | None = None, paired_only: bool = True) -> pl.DataFrame:
    """Posts to label. By default only posts in data/pairs.parquet (see pairs.py)."""
    from bluesky_case_study.corpus import load_corpus

    lf = load_corpus().select("post_id", "text").sort("post_id")
    if paired_only:
        from bluesky_case_study.pairs import pair_post_ids

        lf = lf.filter(pl.col("post_id").is_in(pair_post_ids()))
    return (lf.head(limit) if limit else lf).collect()


def finalize(paired_only: bool = True) -> pl.DataFrame:
    ids = corpus_posts(paired_only=paired_only)["post_id"].to_list()
    labels = label_table(ids)
    labels.write_parquet(LABELS_PATH)
    missing = len(ids) - len(labels)
    print(f"Wrote {LABELS_PATH}: {len(labels):,} posts, {labels['needs_review'].sum():,} need review"
          + (f", {missing:,} not labelled yet" if missing else "")
          + f", ${labels['cost'].sum():.2f} spent")
    print(labels["stance"].value_counts(sort=True))
    return labels


def write_pair_stances() -> pl.DataFrame:
    """data/pairs.parquet plus each post's stance and whether they match."""
    from bluesky_case_study.pairs import load_pairs

    out = match_pairs(load_pairs())
    out.write_parquet(PAIR_STANCES_PATH)
    print(f"Wrote {PAIR_STANCES_PATH}: {len(out):,} pairs")
    print(out.select(
        pl.col("same_stance").mean().alias("same_stance"),
        pl.col("uncertain").mean().alias("uncertain"),
        pl.col("stance_a").is_null().mean().alias("unlabelled"),
    ))
    return out


def match_pairs(pairs: pl.DataFrame, labels: pl.DataFrame | None = None) -> pl.DataFrame:
    """Add stance agreement to candidate pairs (columns post_a, post_b).

    same_stance is True when both posts got the same label; uncertain is True
    when either label is low-confidence (1) or missing.
    """
    labels = labels if labels is not None else pl.read_parquet(LABELS_PATH)
    lab = labels.select("post_id", "stance", "confidence", "needs_review")
    out = pairs.join(lab.rename(lambda c: f"{c}_a" if c != "post_id" else "post_a"), on="post_a", how="left")
    out = out.join(lab.rename(lambda c: f"{c}_b" if c != "post_id" else "post_b"), on="post_b", how="left")
    return out.with_columns(
        (pl.col("stance_a") == pl.col("stance_b")).alias("same_stance"),
        ((pl.min_horizontal("confidence_a", "confidence_b") < 2)
         | pl.col("needs_review_a") | pl.col("needs_review_b")).fill_null(True).alias("uncertain"),
    )


# ------------------------------------------------------------------- CLI

def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("eval", help="label a gold set and score the labels")
    e.add_argument("--split", choices=["dev", "test"], default="dev")
    e.add_argument("--use-suggestions", action="store_true",
                   help="score against suggested_stance where present")
    e.add_argument("--workers", type=int, default=8)
    e.add_argument("--unfrozen", action="store_true",
                   help="score the test set even though the codebook changed since the freeze")

    for name in ("submit", "finalize"):
        s = sub.add_parser(name)
        s.add_argument("--all-posts", action="store_true",
                       help="label the whole corpus, not just posts in data/pairs.parquet")
        s.add_argument("--unfrozen", action="store_true",
                       help="run even though the codebook changed since the freeze")
        if name == "submit":
            s.add_argument("--limit", type=int, default=None, help="first N corpus posts (trial run)")
    sub.add_parser("collect")
    sub.add_parser("pairs", help="attach stances to data/pairs.parquet")
    sub.add_parser("version", help="print the current prompt version and whether it is frozen")

    a = p.parse_args()
    if a.cmd == "eval":
        if a.split == "test":
            check_frozen(a.unfrozen)
        evaluate(a.split, a.use_suggestions, a.workers)
    elif a.cmd == "collect":
        collect()
    elif a.cmd == "version":
        v = prompt_version()
        print(v, "(frozen)" if v == FROZEN_VERSION else f"(NOT the frozen version {FROZEN_VERSION})")
    elif a.cmd == "pairs":
        write_pair_stances()
    elif a.cmd == "finalize":
        check_frozen(a.unfrozen)
        finalize(not a.all_posts)
    else:
        check_frozen(a.unfrozen)
        submit(corpus_posts(a.limit, not a.all_posts))


if __name__ == "__main__":
    main()
