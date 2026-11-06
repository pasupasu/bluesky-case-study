import polars as pl

from bluesky_case_study.paths import DATA_DIR
from bluesky_case_study import load

WEIGHTS = {
    "replies": 1.0,
    "reposts": 1.0,
    "quotes": 1.0
}

# Load follower dataframe, excluding self-follows and any repeated follows
follows = load.load_followers().filter(pl.col("src") != pl.col("dst")).unique()

# Load interactions dataframe
inter = load.load_interactions()

# Turn a wide dataframe into a long one to get weights, each interaction kind gets a different weight row then collapsed
counts = (
    inter.unpivot(
        on = ["replied", "reposted", "quoted"],
        index = "src",
        variable_name = "kind",
        value_name = "dst"
    )
    .drop_nulls("dst")
    .filter(pl.col("src") != pl.col("dst"))
    .group_by("src", "dst")
    .agg(
        (pl.col("kind") == "replied").sum().alias("replies"),
        (pl.col("kind") == "reposted").sum().alias("reposts"),
        (pl.col("kind") == "quoted").sum().alias("quotes"),
    )
)

edges = (
    follows.join(counts, on = ["src", "dst"], how = "left")
    .with_columns(pl.col("replies", "reposts", "quotes").fill_null(0))
    .with_columns(weight = sum(pl.col(k) * w for k, w in WEIGHTS.items()))
)

edges.sink_parquet(DATA_DIR / "weighted_edges.parquet")
