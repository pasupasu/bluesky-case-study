import igraph as ig
import polars as pl

from bluesky_case_study.paths import DATA_DIR

FOLLOWER_SCHEMA = {"src": pl.Int64, "dst": pl.Int64}

INTERACTION_SCHEMA = {
    "src": pl.Int64,
    "replied": pl.Int64,
    "root": pl.Int64,
    "reposted": pl.Int64,
    "quoted": pl.Int64,
    "date": pl.String,  # read as text (YYYYMMDDHHMM), parsed in load_interactions
}


def load_followers() -> pl.LazyFrame:
    return pl.scan_csv(
        DATA_DIR / "followers.csv", has_header=False, schema=FOLLOWER_SCHEMA
    )


def load_interactions() -> pl.LazyFrame:
    return pl.scan_csv(
        DATA_DIR / "interactions.csv",
        has_header=False,
        schema=INTERACTION_SCHEMA,
        null_values="None",
    ).with_columns(pl.col("date").str.to_datetime("%Y%m%d%H%M", strict=True))


def load_weighted() -> pl.LazyFrame:
    return pl.scan_parquet(DATA_DIR / "weighted_edges.parquet")


def load_weighted_graph(active_only: bool = True) -> ig.Graph:
    lf = load_weighted().select("src", "dst", "weight")
    if active_only:
        lf = lf.filter(pl.col("weight") > 0)
    df = lf.collect()

    # Grab the max values of src and dst
    maxes = (
        load_weighted().select(pl.max("src", "dst")).collect()
    )  # 1 row: src_max, dst_max

    # Pick the larger of the two, increment by one, then use that for the number of vertices
    n = max(maxes.row(0)) + 1

    return ig.Graph(
        n=n,
        edges=df.select("src", "dst").to_numpy(),
        directed=True,
        edge_attrs={"weight": df["weight"].to_numpy()},
    )


def load_top_k_graph(k: int = 100_000) -> tuple[ig.Graph, pl.DataFrame]:
    """Graph of the k users with the highest strength (total weight sent + received),
    keeping only edges between them.

    Vertices are renumbered 0..k-1; the original ID is on g.vs["user_id"].
    Also returns a table of user_id, strength and vid (vertex index).
    """
    active = load_weighted().select("src", "dst", "weight").filter(pl.col("weight") > 0)

    top = (
        pl.concat(
            [
                active.select(pl.col("src").alias("user_id"), "weight"),
                active.select(pl.col("dst").alias("user_id"), "weight"),
            ]
        )
        .group_by("user_id")
        .agg(pl.sum("weight").alias("strength"))
        .top_k(k, by="strength")
        .with_row_index("vid")
        .collect()
    )

    ids = top["user_id"].implode()
    edges = (
        active.filter(pl.col("src").is_in(ids) & pl.col("dst").is_in(ids))
        .collect()
        .join(
            top.select(pl.col("user_id").alias("src"), pl.col("vid").alias("s")),
            on="src",
        )
        .join(
            top.select(pl.col("user_id").alias("dst"), pl.col("vid").alias("d")),
            on="dst",
        )
    )

    g = ig.Graph(
        n=top.height,
        edges=edges.select("s", "d").to_numpy(),
        directed=True,
        edge_attrs={"weight": edges["weight"].to_numpy()},
    )
    g.vs["user_id"] = top["user_id"].to_list()
    return g, top
