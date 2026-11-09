"""Reception of paired posts by community: who reposted, replied to and quoted each post.

Steps (each writes to data/reception/ and can be rerun on its own):

    uv run python -m bluesky_case_study.reception communities   # Leiden on the full active graph
    uv run python -m bluesky_case_study.reception engagement    # per post x responder community
    uv run python -m bluesky_case_study.reception exposure      # author followers per community
    uv run python -m bluesky_case_study.reception contrasts     # in-group vs out-group pair rows
    uv run python -m bluesky_case_study.reception summary       # descriptive first look
    uv run python -m bluesky_case_study.reception model         # pre-registered model (analysis plan)

Definitions (see text/analysis_plan.md)
    community C       Leiden community on the full active interaction graph (every user with
                      a reply, repost or quote edge), with the notebook's settings and seed.
                      `communities --k 100000` reproduces the notebook's top-100k sample.
    in-group author   author is a member of C.  out-group author: not a member of C
                      (in another community, or with no interaction edges at all).
    contrast          a content pair where exactly one author is in C: one row per (pair, C).
    positive          reposts by members of C.
    volume            replies and quotes by members of C (direction unknown: no classifier).
    negative tone     replies by members of C with the dataset's sent_label == 0 (negative).
                      This is tone, not disagreement: supportive replies are often negative
                      in tone on this topic. A weak proxy only.
    exposure          the author's followers who are members of C. Out-group authors usually
                      have far fewer, so compare rates per in-community follower, not raw counts.
    likes             counts only (no liker ids), so they can't be split by community.

Self-engagement (an author reposting, replying to or quoting their own post) is dropped.
Replies, quotes and reposts from accounts outside the dataset are missing.
"""

from __future__ import annotations

import argparse
import random
import sys
import time

import polars as pl

from bluesky_case_study.paths import DATA_DIR

OUT_DIR = DATA_DIR / "reception"
COMMUNITIES_PATH = DATA_DIR / "communities.parquet"
ENGAGEMENT_PATH = OUT_DIR / "engagement.parquet"  # post_id x community
POSTS_PATH = OUT_DIR / "posts.parquet"  # post_id: author, likes, totals
EXPOSURE_PATH = OUT_DIR / "exposure.parquet"  # author x community follower counts
CONTRASTS_PATH = OUT_DIR / "contrasts.parquet"  # (pair, community) rows

TOP_K = 100_000
SEED = 42


# ------------------------------------------------------------ communities

def detect_communities(k: int | None = None, seed: int = SEED) -> pl.DataFrame:
    """Leiden communities, with the same settings as notebooks/exploration.ipynb.

    k=None (default): the full active graph (every user with at least one reply, repost
    or quote edge, weight > 0), written to data/communities.parquet and used by the
    reception steps. k=100_000 reproduces the notebook's top-k sample and writes
    data/communities_top100k.parquet for comparison.
    """
    from bluesky_case_study import communities as cm
    from bluesky_case_study.load import load_top_k_graph, load_weighted_graph

    start = time.monotonic()
    if k is None:
        g = load_weighted_graph(active_only=True)  # vertex id = user_id; inactive ids are isolated
        g.vs["user_id"] = list(range(g.vcount()))
        path = COMMUNITIES_PATH
    else:
        g, _ = load_top_k_graph(k)
        path = DATA_DIR / f"communities_top{k // 1000}k.parquet"
    print(f"Graph built in {(time.monotonic() - start) / 60:.1f} min: "
          f"{g.vcount():,} vertices, {g.ecount():,} directed edges")
    g_und = g.as_undirected(mode="collapse", combine_edges="sum")
    del g
    start = time.monotonic()
    random.seed(seed)  # igraph uses Python's random module
    part = g_und.community_leiden(objective_function="modularity", weights="weight", n_iterations=-1)
    print(f"Leiden done in {(time.monotonic() - start) / 60:.1f} min")
    q = g_und.modularity(part.membership, weights="weight")

    strength = g_und.strength(weights="weight")
    df = pl.DataFrame({
        "user_id": pl.Series(g_und.vs["user_id"], dtype=pl.Int64),
        "raw": pl.Series(part.membership, dtype=pl.Int64),
        "strength": pl.Series(strength, dtype=pl.Float64),
        "degree": pl.Series(g_und.degree(), dtype=pl.Int64),
    }).filter(pl.col("degree") > 0)  # isolated ids (no interactions) aren't users of the graph
    out = df.with_columns(
        pl.Series("community", cm.relabel_by_size(df["raw"].to_list()), dtype=pl.Int32)  # C0 = largest
    ).select("user_id", "community", "strength")
    out.write_parquet(path)
    sizes = out["community"].value_counts()["count"]
    print(f"Wrote {path}: {out.height:,} users, {out['community'].n_unique():,} communities "
          f"({(sizes >= 100).sum():,} with 100+ members), modularity {q:.3f} (seed {seed})")
    return out


def load_communities() -> pl.DataFrame:
    if not COMMUNITIES_PATH.exists():
        sys.exit(f"{COMMUNITIES_PATH} is missing: run `reception communities` first")
    return pl.read_parquet(COMMUNITIES_PATH).select("user_id", "community")


# ----------------------------------------------------------- inputs

def _pairs() -> pl.DataFrame:
    """Pairs with stance if the stance step has run, else the bare content pairs."""
    from bluesky_case_study.stance import PAIR_STANCES_PATH

    if PAIR_STANCES_PATH.exists():
        return pl.read_parquet(PAIR_STANCES_PATH)
    from bluesky_case_study.pairs import load_pairs

    print(f"note: {PAIR_STANCES_PATH.name} not found, using pairs without stance")
    return load_pairs()


def _paired_posts(pairs: pl.DataFrame) -> pl.DataFrame:
    """post_id -> author for every post in a pair."""
    return pl.concat([
        pairs.select(pl.col("post_a").alias("post_id"), pl.col("user_a").alias("author")),
        pairs.select(pl.col("post_b").alias("post_id"), pl.col("user_b").alias("author")),
    ]).unique("post_id")


# ------------------------------------------------------------ engagement

def build_engagement() -> pl.DataFrame:
    """Reposts, replies and quotes of every paired post, counted by responder community."""
    from bluesky_case_study.corpus import load_corpus
    from bluesky_case_study.load import load_posts

    posts = _paired_posts(_pairs())
    ids = posts["post_id"].implode()
    comm = load_communities()
    start = time.monotonic()

    reposts = (
        pl.scan_parquet(DATA_DIR / "ip_reposts.parquet")
        .filter(pl.col("original_post_id").is_in(ids))
        .select(pl.col("original_post_id").alias("post_id"), pl.col("reposter_id").alias("responder"),
                pl.lit("repost").alias("kind"), pl.lit(None, pl.Int8).alias("sent_label"))
        .collect()
    )
    archive = load_posts().filter(pl.col("reply_to").is_in(ids) | pl.col("quotes").is_in(ids))
    responses = archive.select(
        pl.when(pl.col("reply_to").is_in(ids)).then(pl.col("reply_to")).otherwise(pl.col("quotes")).alias("post_id"),
        pl.col("user_id").alias("responder"),
        pl.when(pl.col("reply_to").is_in(ids)).then(pl.lit("reply")).otherwise(pl.lit("quote")).alias("kind"),
        "sent_label",
    ).collect()
    events = (
        pl.concat([reposts, responses])
        .join(posts, on="post_id")
        .filter(pl.col("responder") != pl.col("author"))  # drop self-engagement
        .join(comm.rename({"user_id": "responder"}), on="responder", how="left")
    )
    print(f"{events.height:,} engagement events in {(time.monotonic() - start) / 60:.1f} min "
          f"({events['community'].is_not_null().mean():.0%} from users with a community)")

    by_comm = (
        events.filter(pl.col("community").is_not_null())
        .group_by("post_id", "community")
        .agg(
            (pl.col("kind") == "repost").sum().alias("reposts"),
            (pl.col("kind") == "reply").sum().alias("replies"),
            (pl.col("kind") == "quote").sum().alias("quotes"),
            ((pl.col("kind") == "reply") & (pl.col("sent_label") == 0)).sum().alias("neg_replies"),
        )
    )
    totals = events.group_by("post_id").agg(
        (pl.col("kind") == "repost").sum().alias("reposts_all"),
        (pl.col("kind") == "reply").sum().alias("replies_all"),
        (pl.col("kind") == "quote").sum().alias("quotes_all"),
    )
    meta = load_corpus().select("post_id", "date", "like_count", "sent_label").collect()
    post_table = (
        posts.join(meta, on="post_id", how="left")
        .join(totals, on="post_id", how="left")
        .join(comm.rename({"user_id": "author", "community": "author_community"}), on="author", how="left")
        .with_columns(pl.col("reposts_all", "replies_all", "quotes_all").fill_null(0))
        .rename({"like_count": "likes", "sent_label": "post_sent_label"})
    )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    by_comm.write_parquet(ENGAGEMENT_PATH)
    post_table.write_parquet(POSTS_PATH)
    print(f"Wrote {ENGAGEMENT_PATH.name} ({by_comm.height:,} post x community rows) "
          f"and {POSTS_PATH.name} ({post_table.height:,} posts, "
          f"{post_table['author_community'].is_not_null().mean():.0%} by authors with a community)")
    return by_comm


# -------------------------------------------------------------- exposure

def build_exposure() -> pl.DataFrame:
    """For each paired author: followers in total, and followers who are members of each community."""
    from bluesky_case_study.load import load_followers

    authors = pl.read_parquet(POSTS_PATH, columns=["author"]).unique()["author"].implode()
    comm = load_communities()
    start = time.monotonic()
    edges = load_followers().filter(pl.col("dst").is_in(authors))  # src follows dst
    total = edges.group_by("dst").agg(pl.len().alias("followers_all")).collect()
    by_comm = (
        edges.join(comm.lazy().rename({"user_id": "src"}), on="src")
        .group_by("dst", "community").agg(pl.len().alias("followers_c")).collect()
    )
    out = by_comm.join(total, on="dst", how="full", coalesce=True).rename({"dst": "author"})
    out.write_parquet(EXPOSURE_PATH)
    print(f"Wrote {EXPOSURE_PATH.name}: {out['author'].n_unique():,} authors, "
          f"{out.height:,} author x community rows ({(time.monotonic() - start) / 60:.1f} min)")
    return out


# ------------------------------------------------------------- contrasts

OUTCOMES = ["reposts", "replies", "quotes", "neg_replies"]


def build_contrasts() -> pl.DataFrame:
    """One row per (pair, community C) where exactly one of the two authors is in C.

    in_* columns describe the in-group post (author in C), out_* the out-group post,
    with engagement counted only from members of C and exposure = author's followers in C.
    """
    pairs = _pairs()
    posts = pl.read_parquet(POSTS_PATH)
    eng = pl.read_parquet(ENGAGEMENT_PATH)
    exp = pl.read_parquet(EXPOSURE_PATH)
    comm_of = posts.select("post_id", "author_community")

    keep = [c for c in ["sim", "side_pair", "same_side", "same_stance", "uncertain", "short",
                        "near_duplicate", "linked", "days_apart"] if c in pairs.columns]
    p = (pairs.select("post_a", "post_b", *keep)
         .join(comm_of.rename({"post_id": "post_a", "author_community": "comm_a"}), on="post_a", how="left")
         .join(comm_of.rename({"post_id": "post_b", "author_community": "comm_b"}), on="post_b", how="left"))

    # a's community when b is not in it, and b's community when a is not in it
    a_in = (p.filter(pl.col("comm_a").is_not_null() & (pl.col("comm_b").is_null() | (pl.col("comm_b") != pl.col("comm_a"))))
            .select(pl.col("comm_a").alias("community"), pl.col("post_a").alias("in_post"),
                    pl.col("post_b").alias("out_post"), pl.col("comm_b").alias("out_community"), *keep))
    b_in = (p.filter(pl.col("comm_b").is_not_null() & (pl.col("comm_a").is_null() | (pl.col("comm_a") != pl.col("comm_b"))))
            .select(pl.col("comm_b").alias("community"), pl.col("post_b").alias("in_post"),
                    pl.col("post_a").alias("out_post"), pl.col("comm_a").alias("out_community"), *keep))
    rows = pl.concat([a_in, b_in])

    def side(prefix: str) -> pl.DataFrame:
        post_col = f"{prefix}_post"
        info = posts.select("post_id", "author", "date", "likes", "reposts_all", "replies_all", "quotes_all")
        out = (rows.select(post_col, "community").unique()
               .join(info.rename({"post_id": post_col}), on=post_col, how="left")
               .join(eng.rename({"post_id": post_col}), on=[post_col, "community"], how="left")
               .join(exp.rename({"author": "author"}), on=["author", "community"], how="left")
               .join(exp.select("author", "followers_all").unique("author"), on="author", how="left", suffix="_t")
               .with_columns(pl.coalesce("followers_all", "followers_all_t").alias("followers_all"))
               .drop("followers_all_t")
               .with_columns(pl.col(*OUTCOMES, "followers_c", "followers_all").fill_null(0)))
        return out.rename({c: f"{prefix}_{c}" for c in out.columns if c not in (post_col, "community")})

    out = (rows.join(side("in"), on=["in_post", "community"], how="left")
           .join(side("out"), on=["out_post", "community"], how="left")
           .with_columns(pl.col("out_community").is_not_null().alias("out_assigned")))
    out.write_parquet(CONTRASTS_PATH)
    print(f"Wrote {CONTRASTS_PATH.name}: {out.height:,} (pair, community) rows from "
          f"{pairs.height:,} pairs; {out['community'].n_unique()} communities; "
          f"out-group author in another community for {out['out_assigned'].mean():.0%}")
    if "same_side" in out.columns:
        print(out.group_by("side_pair").agg(pl.len().alias("rows")).sort("rows", descending=True))
    return out


# --------------------------------------------------------------- summary

def summary(subset: str = "PAL-PAL", min_followers_c: int = 10) -> pl.DataFrame:
    """Descriptive first look, not the final model: in-group minus out-group, per outcome.

    rate = count / followers in C. Rows where either author has fewer than
    min_followers_c followers in C are dropped (no exposure, rate undefined).
    """
    df = pl.read_parquet(CONTRASTS_PATH)
    if "side_pair" in df.columns and subset != "all":
        df = df.filter(pl.col("side_pair") == subset)
    df = df.filter((pl.col("in_followers_c") >= min_followers_c) & (pl.col("out_followers_c") >= min_followers_c))
    if df.is_empty():
        sys.exit("no rows left: check the subset name and that contrasts were built after the stance step")
    rows = []
    for o in OUTCOMES:
        rin = pl.col(f"in_{o}") / pl.col("in_followers_c")
        rout = pl.col(f"out_{o}") / pl.col("out_followers_c")
        s = df.select(
            pl.col(f"in_{o}").mean().alias("in_mean"),
            pl.col(f"out_{o}").mean().alias("out_mean"),
            (rin * 1000).mean().alias("in_per_1k_followers"),
            (rout * 1000).mean().alias("out_per_1k_followers"),
            ((rin.log1p() - rout.log1p()) * 1000).mean().alias("mean_diff_log1p_rate_x1000"),
            (rin > rout).mean().alias("share_in_higher"),
            (rin < rout).mean().alias("share_out_higher"),
        ).row(0, named=True)
        rows.append({"outcome": o, **s})
    likes = df.select(
        (pl.col("in_likes") / pl.col("in_followers_all").clip(1) * 1000).mean().alias("in"),
        (pl.col("out_likes") / pl.col("out_followers_all").clip(1) * 1000).mean().alias("out"),
    ).row(0)
    table = pl.DataFrame(rows)
    with pl.Config(tbl_width_chars=200, float_precision=3):
        print(f"\n{subset}: {df.height:,} (pair, community) rows, {df['community'].n_unique()} communities, "
              f"{pl.concat([df['in_post'], df['out_post']]).n_unique():,} posts")
        print(table)
        print(f"likes per 1k followers (all followers, community-blind): in {likes[0]:.2f}, out {likes[1]:.2f}")
        print("Descriptive only: rows share posts and communities, so no standard errors here.")
    return table


# ----------------------------------------------------------------- model
#
# Fixed-effect (conditional) Poisson per contrast; see text/analysis_plan.md, "Model".
# With one fixed effect per contrast and two posts per contrast, conditioning on the
# total gives a binomial model of the in-group share, whose intercept is beta:
#     y_in | y_in + y_out ~ Binomial(y_in + y_out, pi)
#     logit pi = beta + gamma * d_log_fc + delta * d_log_fall + lambda * d_age
# exp(beta) is the in-group / out-group rate ratio of engagement from members of C.

PRIMARY_OUTCOME = "reposts"
MIN_CONTRASTS = 100  # communities included in the pooled model
MIN_CONTRASTS_PER_COMMUNITY = 1_000  # communities reported on their own


def _design(df: pl.DataFrame) -> pl.DataFrame:
    """Within-contrast differences (in-group minus out-group)."""
    return df.with_columns(
        (pl.col("in_followers_c").log1p() - pl.col("out_followers_c").log1p()).alias("d_log_fc"),
        (pl.col("in_followers_all").log1p() - pl.col("out_followers_all").log1p()).alias("d_log_fall"),
        # age = collection end - date, so d_age = out_date - in_date (in days); the end date cancels
        ((pl.col("out_date") - pl.col("in_date")).dt.total_hours() / 24).fill_null(0.0).alias("d_age"),
    )


def fit_contrasts(df: pl.DataFrame, outcome: str, exposure: str = "linear") -> dict:
    """Fit the conditional model for one outcome. exposure: 'linear' or 'spline'."""
    import numpy as np
    import statsmodels.api as sm

    d = _design(df).filter((pl.col(f"in_{outcome}") + pl.col(f"out_{outcome}")) > 0)
    if d.height < 30:
        return {"outcome": outcome, "contrasts": d.height, "rate_ratio": None, "ci_low": None,
                "ci_high": None, "p": None, "note": "too few contrasts with any engagement"}
    if exposure == "spline":
        # Piecewise-linear spline in d_log_fc. Every basis term is 0 at d_log_fc = 0, so the
        # intercept stays the in-group effect at equal exposure. (Dummy bins don't do this:
        # their intercept is the effect at the reference bin's typical exposure gap.)
        x = d["d_log_fc"].to_numpy()
        knots = np.unique(np.quantile(x, [0.25, 0.5, 0.75]))
        hinge = [np.maximum(x - k, 0) - max(-k, 0) for k in knots]
        fc_cols = np.column_stack([x, *hinge])
    else:
        fc_cols = d["d_log_fc"].to_numpy()[:, None]
    covs = np.column_stack([fc_cols, d["d_log_fall"].to_numpy(), d["d_age"].to_numpy() / 30])  # age per 30 days
    # A covariate that doesn't vary (e.g. all posts on one day in a small subset) can't be
    # estimated; drop it rather than fail. The intercept (beta) is always kept.
    keep = covs.std(axis=0) > 0
    X = np.column_stack([np.ones(d.height), covs[:, keep]])
    y = np.column_stack([d[f"in_{outcome}"].to_numpy(), d[f"out_{outcome}"].to_numpy()])
    groups = np.column_stack([
        pl.Series(d["in_author"]).rank("dense").to_numpy(),
        pl.Series(d["out_author"]).rank("dense").to_numpy(),
    ]).astype(int)
    res = sm.GLM(y, X, family=sm.families.Binomial()).fit(
        cov_type="cluster", cov_kwds={"groups": groups})
    b, se = res.params[0], res.bse[0]
    return {
        "outcome": outcome,
        "contrasts": d.height,
        "communities": d["community"].n_unique(),
        "rate_ratio": float(np.exp(b)),
        "ci_low": float(np.exp(b - 1.96 * se)),
        "ci_high": float(np.exp(b + 1.96 * se)),
        "p": float(res.pvalues[0]),
        "exposure_coef": float(res.params[1]) if exposure == "linear" else None,
        "note": "",
    }


def model(subset: str = "PAL-PAL", outcomes: list[str] | None = None) -> pl.DataFrame:
    """Run the pre-registered set: primary, robustness checks and per-community estimates."""
    outcomes = outcomes or OUTCOMES
    all_rows = pl.read_parquet(CONTRASTS_PATH)
    if "side_pair" not in all_rows.columns:
        sys.exit("contrasts have no stance: run the stance step, then `reception contrasts` again")

    def pooled(df: pl.DataFrame) -> pl.DataFrame:
        big = df.group_by("community").len().filter(pl.col("len") >= MIN_CONTRASTS)["community"]
        return df.filter(pl.col("community").is_in(big.implode()))

    base = pooled(all_rows.filter(pl.col("side_pair") == subset))
    flags = [c for c in ("short", "near_duplicate", "linked") if c in base.columns]
    variants = {
        "primary": (base, "linear"),
        "exposure-matched (|d log followers_C| <= 1)": (
            _design(base).filter(pl.col("d_log_fc").abs() <= 1), "linear"),
        "exposure as a spline (knots at quartiles)": (base, "spline"),
        "both labels confidence >= 2": (base.filter(~pl.col("uncertain")), "linear"),
        "drop short / near-duplicate / linked": (
            base.filter(~pl.any_horizontal(*[pl.col(c) for c in flags])) if flags else base, "linear"),
    }
    if subset == "PAL-PAL":
        variants["ISR-ISR (exploratory)"] = (pooled(all_rows.filter(pl.col("side_pair") == "ISR-ISR")), "linear")
        variants["PAL-ISR (opposite sides, comparison)"] = (
            pooled(all_rows.filter(pl.col("side_pair") == "PAL-ISR")), "linear")

    rows = []
    for name, (df, exposure) in variants.items():
        for o in outcomes:
            rows.append({"analysis": name, **fit_contrasts(df, o, exposure)})
    for c, n in base.group_by("community").len().sort("len", descending=True).iter_rows():
        if n < MIN_CONTRASTS_PER_COMMUNITY:
            break
        rows.append({"analysis": f"community {c} only", **fit_contrasts(
            base.filter(pl.col("community") == c), PRIMARY_OUTCOME)})

    table = pl.DataFrame(rows, infer_schema_length=None)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"model_{subset}.parquet"
    table.write_parquet(out)
    with pl.Config(tbl_rows=100, tbl_width_chars=220, float_precision=3, fmt_str_lengths=50):
        print(f"\nIn-group / out-group rate ratio of engagement from community members ({subset}).")
        print(f"Headline: analysis 'primary', outcome '{PRIMARY_OUTCOME}'. Everything else is secondary.")
        print(table.select("analysis", "outcome", "contrasts", "communities", "rate_ratio",
                           "ci_low", "ci_high", "p", "note"))
    print(f"Wrote {out}")
    return table


# ------------------------------------------------------------------- CLI

def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("communities")
    c.add_argument("--k", type=int, default=None,
                   help="top-k sample instead of the full graph (the notebook used 100000)")
    c.add_argument("--seed", type=int, default=SEED)
    sub.add_parser("engagement")
    sub.add_parser("exposure")
    sub.add_parser("contrasts")
    s = sub.add_parser("summary")
    s.add_argument("--subset", default="PAL-PAL", help="side_pair to summarise, or 'all'")
    s.add_argument("--min-followers-c", type=int, default=10,
                   help="drop rows where either author has fewer followers in C (rates explode)")
    m = sub.add_parser("model", help="pre-registered model: primary, robustness, per community")
    m.add_argument("--subset", default="PAL-PAL", help="side_pair for the primary sample")
    a = p.parse_args()
    if a.cmd == "model":
        model(a.subset)
        return
    if a.cmd == "communities":
        detect_communities(a.k, a.seed)
    elif a.cmd == "engagement":
        build_engagement()
    elif a.cmd == "exposure":
        build_exposure()
    elif a.cmd == "contrasts":
        build_contrasts()
    else:
        summary(a.subset, a.min_followers_c)


if __name__ == "__main__":
    main()
