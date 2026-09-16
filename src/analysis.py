"""Graph metrics, centrality, and Louvain communities."""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import networkx as nx
import pandas as pd

try:
    import community as community_louvain
except ImportError:  # pragma: no cover
    community_louvain = None

from .preprocess import CATEGORY_LABELS


def global_metrics(G: nx.Graph) -> dict:
    n = G.number_of_nodes()
    m = G.number_of_edges()
    degrees = [d for _, d in G.degree()]
    avg_degree = float(np.mean(degrees)) if degrees else 0.0
    density = nx.density(G)
    clustering = nx.average_clustering(G, weight="weight") if n else 0.0

    if n == 0:
        return {
            "n_nodes": 0,
            "n_edges": 0,
            "density": 0.0,
            "average_degree": 0.0,
            "average_clustering": 0.0,
            "n_components": 0,
            "largest_component_size": 0,
            "avg_shortest_path": None,
            "diameter": None,
        }

    comps = list(nx.connected_components(G))
    giant_nodes = max(comps, key=len)
    giant = G.subgraph(giant_nodes).copy()
    if giant.number_of_nodes() > 1 and nx.is_connected(giant):
        avg_path = nx.average_shortest_path_length(giant, weight=None)
        diameter = nx.diameter(giant)
    else:
        avg_path = None
        diameter = None

    return {
        "n_nodes": n,
        "n_edges": m,
        "density": float(density),
        "average_degree": avg_degree,
        "average_clustering": float(clustering),
        "n_components": len(comps),
        "largest_component_size": len(giant_nodes),
        "avg_shortest_path": None if avg_path is None else float(avg_path),
        "diameter": None if diameter is None else int(diameter),
    }


def centrality_scores(G: nx.Graph) -> pd.DataFrame:
    if G.number_of_nodes() == 0:
        return pd.DataFrame(columns=["respondent_id", "degree", "betweenness", "eigenvector"])

    deg = dict(G.degree(weight="weight"))
    H = G.copy()
    for _u, _v, d in H.edges(data=True):
        w = float(d.get("weight", 0.0))
        d["distance"] = max(1e-6, 1.0 - w)
    btw = nx.betweenness_centrality(H, weight="distance", normalized=True)
    try:
        eig = nx.eigenvector_centrality(G, weight="weight", max_iter=1000)
    except nx.PowerIterationFailedConvergence:
        eig = {n: 0.0 for n in G.nodes()}

    rows = []
    for node in G.nodes():
        rows.append(
            {
                "respondent_id": node,
                "degree": float(deg.get(node, 0.0)),
                "betweenness": float(btw.get(node, 0.0)),
                "eigenvector": float(eig.get(node, 0.0)),
            }
        )
    return pd.DataFrame(rows).sort_values("eigenvector", ascending=False).reset_index(drop=True)


def louvain_communities(G: nx.Graph, resolution: float = 1.0, seed: int = 42) -> dict[str, int]:
    if community_louvain is None:
        raise ImportError("python-louvain is required")
    if G.number_of_edges() == 0:
        return {n: i for i, n in enumerate(G.nodes())}
    return community_louvain.best_partition(G, weight="weight", resolution=resolution, random_state=seed)


def modularity(G: nx.Graph, partition: dict[str, int]) -> float:
    if community_louvain is None or G.number_of_edges() == 0:
        return 0.0
    return float(community_louvain.modularity(partition, G, weight="weight"))


def community_profiles(
    encoded: pd.DataFrame,
    partition: dict[str, int],
    meta_df: pd.DataFrame,
) -> pd.DataFrame:
    """Mean Likert score per category for each community."""
    id_to_comm = partition
    scores = encoded.copy()
    scores["community"] = scores["respondent_id"].map(id_to_comm)

    rows = []
    for comm, group in scores.groupby("community"):
        row = {"community": int(comm), "size": int(len(group))}
        for cat, name in CATEGORY_LABELS.items():
            cols = meta_df.loc[meta_df["category"] == cat, "code"].tolist()
            row[name] = float(group[cols].mean().mean())
        row["overall"] = float(
            group[[c for c in group.columns if c not in ("respondent_id", "community")]].mean().mean()
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values("community").reset_index(drop=True)


def top_central(centrality: pd.DataFrame, n: int = 5) -> dict:
    out = {}
    for col in ("degree", "betweenness", "eigenvector"):
        top = centrality.nlargest(n, col)
        out[col] = [
            {"respondent_id": r.respondent_id, "score": float(getattr(r, col))}
            for r in top.itertuples(index=False)
        ]
    return out


def community_size_summary(partition: dict[str, int]) -> dict:
    counts: dict[int, int] = defaultdict(int)
    for cid in partition.values():
        counts[int(cid)] += 1
    return {str(k): v for k, v in sorted(counts.items())}
