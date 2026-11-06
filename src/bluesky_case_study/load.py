import polars as pl

from bluesky_case_study.paths import DATA_DIR

FOLLOWER_SCHEMA = {
    "src": pl.Int64,
    "dst": pl.Int64
}

INTERACTION_SCHEMA = {
    "src": pl.Int64,
    "replied": pl.Int64,
    "root": pl.Int64,
    "reposted": pl.Int64,
    "quoted": pl.Int64,
    "date": pl.Int64,
}


def load_followers() -> pl.LazyFrame:
    return pl.scan_csv(
    DATA_DIR / "followers.csv",
    has_header= False,
    schema = FOLLOWER_SCHEMA
    )

def load_interactions() -> pl.LazyFrame:
    return pl.scan_csv(
    DATA_DIR / "interactions.csv",
    has_header = False,
    schema = INTERACTION_SCHEMA,
    null_values = "None",
    )