"""Summaries of a community partition (e.g. from leidenalg) on a weighted graph."""

import igraph as ig
import numpy as np
import polars as pl


def relabel_by_size(membership: list[int]) -> list[int]:
    """Renumber communities so 0 is the largest, 1 the next, and so on.

    Leiden's community numbers are arbitrary and change between runs.
    """
    m = pl.Series("c", membership)
    order = m.value_counts(sort=True)["c"].to_list()
    rank = {c: i for i, c in enumerate(order)}
    return [rank[c] for c in membership]


def community_edges(g: ig.Graph, membership: list[int]) -> pl.DataFrame:
    """One row per edge: source community, target community, weight."""
    m = np.asarray(membership)
    el = np.asarray(g.get_edgelist())
    return pl.DataFrame({"a": m[el[:, 0]], "b": m[el[:, 1]], "w": g.es["weight"]})


def community_summary(
    g: ig.Graph, membership: list[int], users: pl.DataFrame, n_top_users: int = 5
) -> pl.DataFrame:
    """One row per community, largest first.

    users: table with user_id and strength, one row per vertex in vertex order
    (the second value returned by load.load_top_k_graph).

    internal_share = weight on edges inside the community / all weight on edges
    touching it. High means a closed-off group; low means it mixes with others.
    """
    cedges = community_edges(g, membership)
    internal = (
        cedges.filter(pl.col("a") == pl.col("b"))
        .group_by("a")
        .agg(pl.sum("w").alias("internal_w"))
        .rename({"a": "community"})
    )
    total = (
        pl.concat(
            [
                cedges.select(pl.col("a").alias("community"), "w"),
                cedges.filter(pl.col("a") != pl.col("b")).select(
                    pl.col("b").alias("community"), "w"
                ),
            ]
        )
        .group_by("community")
        .agg(pl.sum("w").alias("total_w"))
    )
    return (
        users.with_columns(community=pl.Series(membership))
        .group_by("community")
        .agg(
            pl.len().alias("size"),
            pl.col("user_id")
            .sort_by("strength", descending=True)
            .head(n_top_users)
            .alias("top_users"),
        )
        .join(internal, on="community", how="left")
        .join(total, on="community", how="left")
        .with_columns(
            share=pl.col("size") / pl.col("size").sum(),
            internal_share=(pl.col("internal_w") / pl.col("total_w")).fill_null(0.0),
        )
        .sort("size", descending=True)
        .select("community", "size", "share", "internal_share", "top_users")
    )


def mixing_matrix(
    g: ig.Graph, membership: list[int], communities: list[int]
) -> np.ndarray:
    """Row i, column j = share of community i's outgoing weight that goes to community j.

    Only edges between the listed communities are counted, so each row sums to 1.
    """
    idx = {c: i for i, c in enumerate(communities)}
    M = np.zeros((len(communities), len(communities)))
    agg = (
        community_edges(g, membership)
        .filter(pl.col("a").is_in(communities) & pl.col("b").is_in(communities))
        .group_by("a", "b")
        .agg(pl.sum("w"))
    )
    for a, b, w in agg.iter_rows():
        M[idx[a], idx[b]] += w
    return M / M.sum(axis=1, keepdims=True)
