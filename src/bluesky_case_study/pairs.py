"""Find pairs of corpus posts with similar content: embeddings + FAISS nearest neighbours.

Usage:
    uv run python -m bluesky_case_study.pairs embed                 # ~minutes on the GPU
    uv run python -m bluesky_case_study.pairs search --k 20 --min-sim 0.75
    uv run python -m bluesky_case_study.pairs inspect               # eyeball pairs by similarity band

Writes:
    data/embeddings/<model>.npy          one float16 row per corpus post, in post_id order
    data/embeddings/<model>_ids.parquet  the post_id of each row
    data/pairs.parquet                   one row per unordered pair (post_a < post_b)

A pair is two posts by different authors whose embeddings have cosine
similarity >= min_sim, where at least one is among the other's k nearest
neighbours. Same-author pairs are dropped because they can't separate in-group
from out-group reception.

Embeddings capture topic and wording, not stance: "Israel must stop the
genocide" and "Calling it genocide is a lie" can sit close together. That is
why stance is labelled separately (stance.py) and pairs are only stance
matches when both labels agree. Pick min_sim by reading pairs with `inspect`,
not by default.

The pairs have no community information yet. Join each author's Leiden
community on user_a / user_b to tell in-group pairs from out-group pairs.
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np
import polars as pl

from bluesky_case_study.corpus import load_corpus
from bluesky_case_study.paths import DATA_DIR

MODEL = "sentence-transformers/all-mpnet-base-v2"
EMB_DIR = DATA_DIR / "embeddings"
PAIRS_PATH = DATA_DIR / "pairs.parquet"


def _slug(model: str) -> str:
    return model.split("/")[-1]


def emb_paths(model: str = MODEL):
    return EMB_DIR / f"{_slug(model)}.npy", EMB_DIR / f"{_slug(model)}_ids.parquet"


# ------------------------------------------------------------------ embed

def embed(model: str = MODEL, batch_size: int = 256, limit: int | None = None) -> None:
    """Embed every corpus post (normalised, so inner product = cosine)."""
    import torch
    from sentence_transformers import SentenceTransformer

    posts = load_corpus().select("post_id", "text").sort("post_id")
    posts = (posts.head(limit) if limit else posts).collect()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Embedding {len(posts):,} posts with {model} on {device}")

    st = SentenceTransformer(model, device=device)
    if device == "cuda":
        st = st.half()
    start = time.monotonic()
    vecs = st.encode(
        posts["text"].to_list(),
        batch_size=batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    ).astype(np.float16)  # half the disk/RAM; similarity precision is plenty
    print(f"Done in {(time.monotonic() - start) / 60:.1f} min, shape {vecs.shape}")

    EMB_DIR.mkdir(parents=True, exist_ok=True)
    vec_path, ids_path = emb_paths(model)
    np.save(vec_path, vecs)
    posts.select("post_id").write_parquet(ids_path)
    print(f"Wrote {vec_path} and {ids_path}")


def load_embeddings(model: str = MODEL) -> tuple[np.ndarray, np.ndarray]:
    vec_path, ids_path = emb_paths(model)
    vecs = np.load(vec_path).astype(np.float32)
    ids = pl.read_parquet(ids_path)["post_id"].to_numpy()
    return vecs, ids


# ----------------------------------------------------------------- search

def build_index(vecs: np.ndarray, kind: str = "hnsw"):
    """Inner-product index. 'flat' is exact (slow on the full corpus); 'hnsw' is approximate."""
    import faiss

    d = vecs.shape[1]
    if kind == "flat":
        index = faiss.IndexFlatIP(d)
    else:
        index = faiss.IndexHNSWFlat(d, 32, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = 80
        index.hnsw.efSearch = 128
    start = time.monotonic()
    index.add(vecs)
    print(f"Built {kind} index of {index.ntotal:,} vectors in {(time.monotonic() - start) / 60:.1f} min")
    return index


def knn(index, vecs: np.ndarray, k: int, chunk: int = 50_000) -> tuple[np.ndarray, np.ndarray]:
    """k nearest neighbours of every vector (excluding itself), in chunks for progress."""
    sims, nbrs = [], []
    start = time.monotonic()
    for i in range(0, len(vecs), chunk):
        s, n = index.search(vecs[i : i + chunk], k + 1)  # +1 because each post finds itself
        sims.append(s)
        nbrs.append(n)
        done = min(i + chunk, len(vecs))
        print(f"  searched {done:,}/{len(vecs):,} ({(time.monotonic() - start) / 60:.1f} min)", flush=True)
    return np.vstack(sims), np.vstack(nbrs)


def find_pairs(
    k: int = 20, min_sim: float = 0.75, model: str = MODEL, kind: str = "hnsw"
) -> pl.DataFrame:
    vecs, ids = load_embeddings(model)
    index = build_index(vecs, kind)
    sims, nbrs = knn(index, vecs, k)

    rows = np.repeat(np.arange(len(ids)), sims.shape[1])
    cand = pl.DataFrame({
        "i": rows,
        "j": nbrs.ravel(),
        "sim": sims.ravel(),
    }).filter((pl.col("j") >= 0) & (pl.col("i") != pl.col("j")) & (pl.col("sim") >= min_sim))

    uniq = (
        cand.with_columns(
            pl.min_horizontal("i", "j").alias("lo"), pl.max_horizontal("i", "j").alias("hi")
        )
        # A pair found from both ends appears twice; keep one copy.
        .group_by("lo", "hi").agg(pl.col("sim").max())
    )
    # ids is sorted, so lo < hi also means post_a < post_b.
    pairs = pl.DataFrame({
        "post_a": ids[uniq["lo"].to_numpy()],
        "post_b": ids[uniq["hi"].to_numpy()],
        "sim": uniq["sim"].cast(pl.Float32),
    })

    meta = load_corpus().select(
        "post_id", "user_id", "date", "kw_core", "reply_to", "quotes",
        # Words left after removing links and @-mentions. Link-only posts (0 words)
        # match on URL text, e.g. two different haaretz.com links, so they're dropped.
        pl.col("text").str.replace_all(r"\S*(?:https?://|www\.|\.com/|\.org/)\S*|@\S+", " ")
        .str.count_matches(r"\w+").alias("words"),
    ).collect()
    pairs = (
        pairs.join(meta.rename(lambda c: "post_a" if c == "post_id" else f"{c}_a"), on="post_a")
        .join(meta.rename(lambda c: "post_b" if c == "post_id" else f"{c}_b"), on="post_b")
        .rename({"user_id_a": "user_a", "user_id_b": "user_b"})
        .filter(
            (pl.col("user_a") != pl.col("user_b")) & (pl.col("words_a") > 0) & (pl.col("words_b") > 0)
        )
        .with_columns(
            # Sl q ogans like "Free Palestine." match on almost nothing but the slogan
            (pl.min_horizontal("words_a", "words_b") < 4).alias("short"),
            ((pl.col("date_a") - pl.col("date_b")).abs().dt.total_hours() / 24)
            .cast(pl.Float32).alias("days_apart"),
            # Copies of the same text (copypasta, shared headlines) rather than paraphrases
            (pl.col("sim") >= 0.98).alias("near_duplicate"),
            # One post replies to or quotes the other: a conversation, not two independent posts
            ((pl.col("reply_to_a") == pl.col("post_b")) | (pl.col("reply_to_b") == pl.col("post_a"))
             | (pl.col("quotes_a") == pl.col("post_b")) | (pl.col("quotes_b") == pl.col("post_a")))
            .fill_null(False).alias("linked"),
        )
        .drop("reply_to_a", "reply_to_b", "quotes_a", "quotes_b")
        .sort("post_a", "post_b")
    )
    pairs.write_parquet(PAIRS_PATH)
    _summary(pairs, len(ids))
    return pairs


def _summary(pairs: pl.DataFrame, n_posts: int) -> None:
    posts_in_pairs = pl.concat([pairs["post_a"], pairs["post_b"]]).n_unique()
    print(f"Wrote {PAIRS_PATH}: {len(pairs):,} pairs covering {posts_in_pairs:,} of "
          f"{n_posts:,} posts ({posts_in_pairs / n_posts:.0%})")
    print(pairs.select(
        pl.col("sim").quantile(q).alias(f"sim_p{int(q * 100)}") for q in (0.1, 0.5, 0.9)
    ))
    print(pairs.select(
        pl.col("short").mean().alias("short"),
        pl.col("near_duplicate").mean().alias("near_duplicate"),
        pl.col("linked").mean().alias("linked"),
        (pl.col("days_apart") > 30).mean().alias("over_30_days_apart"),
    ))


def load_pairs() -> pl.DataFrame:
    return pl.read_parquet(PAIRS_PATH)


def pair_post_ids() -> list[int]:
    """Every post that appears in at least one pair (what stance.py needs to label)."""
    p = load_pairs()
    return pl.concat([p["post_a"], p["post_b"]]).unique().sort().to_list()


# ---------------------------------------------------------------- inspect

def inspect(per_band: int = 4, seed: int = 0) -> None:
    """Print random pairs from each similarity band, to choose min_sim by reading."""
    pairs = load_pairs()
    text = load_corpus().select("post_id", "text").collect()
    bands = [(0.75, 0.8), (0.8, 0.85), (0.85, 0.9), (0.9, 0.95), (0.95, 0.98), (0.98, 1.01)]
    for lo, hi in bands:
        band = pairs.filter(pl.col("sim").is_between(lo, hi, closed="left"))
        if band.is_empty():
            continue
        print(f"\n===== sim {lo:.2f}-{min(hi, 1):.2f}: {len(band):,} pairs =====")
        sample = band.sample(min(per_band, len(band)), seed=seed)
        sample = (sample.join(text.rename({"post_id": "post_a", "text": "text_a"}), on="post_a")
                  .join(text.rename({"post_id": "post_b", "text": "text_b"}), on="post_b"))
        for r in sample.iter_rows(named=True):
            print(f"\n[{r['sim']:.3f}]")
            print("  A:", r["text_a"].replace("\n", " ")[:280])
            print("  B:", r["text_b"].replace("\n", " ")[:280])


# -------------------------------------------------------------------- CLI

def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", default=MODEL)
    sub = p.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("embed")
    e.add_argument("--batch-size", type=int, default=256)
    e.add_argument("--limit", type=int, default=None, help="first N posts (trial run)")
    s = sub.add_parser("search")
    s.add_argument("--k", type=int, default=20, help="neighbours per post")
    s.add_argument("--min-sim", type=float, default=0.75)
    s.add_argument("--index", choices=["hnsw", "flat"], default="hnsw")
    i = sub.add_parser("inspect")
    i.add_argument("--per-band", type=int, default=4)
    a = p.parse_args()

    if a.cmd == "embed":
        embed(a.model, a.batch_size, a.limit)
    elif a.cmd == "search":
        find_pairs(a.k, a.min_sim, a.model, a.index)
    else:
        inspect(a.per_band)


if __name__ == "__main__":
    main()
