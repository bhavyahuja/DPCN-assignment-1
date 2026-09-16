"""Build respondent-similarity networks from ordinal opinion vectors."""

from __future__ import annotations

import numpy as np
import networkx as nx


def pairwise_pearson_similarity(X: np.ndarray) -> np.ndarray:
    """Pearson correlation with pairwise-complete observations.

    Mean-centering is essential for Likert data: raw cosine is inflated because
    almost every answer sits on the Agree side of the scale.
    """
    n = X.shape[0]
    sim = np.eye(n, dtype=float)
    for i in range(n):
        xi = X[i]
        for j in range(i + 1, n):
            xj = X[j]
            mask = np.isfinite(xi) & np.isfinite(xj)
            if mask.sum() < 5:
                s = 0.0
            else:
                a = xi[mask]
                b = xj[mask]
                a = a - a.mean()
                b = b - b.mean()
                na = np.linalg.norm(a)
                nb = np.linalg.norm(b)
                if na == 0 or nb == 0:
                    s = 0.0
                else:
                    s = float(np.dot(a, b) / (na * nb))
            sim[i, j] = sim[j, i] = s
    return sim


# Backwards-compatible name used in early drafts
pairwise_cosine_similarity = pairwise_pearson_similarity


def mutual_knn_graph(
    similarity: np.ndarray,
    k: int = 8,
    respondent_ids: list[str] | None = None,
    mode: str = "union",
) -> nx.Graph:
    """Undirected k-nearest-neighbor graph, weighted by similarity.

    mode='union' (default): edge if i is among j's kNN or vice versa.
    mode='mutual': edge only if both list each other (sparser).
    """
    n = similarity.shape[0]
    if respondent_ids is None:
        respondent_ids = [str(i) for i in range(n)]
    k = min(k, n - 1)

    neighbors = []
    for i in range(n):
        row = similarity[i].copy()
        row[i] = -np.inf
        idx = np.argpartition(row, -k)[-k:]
        idx = idx[np.argsort(row[idx])[::-1]]
        neighbors.append(set(int(x) for x in idx.tolist()))

    G = nx.Graph()
    for i, rid in enumerate(respondent_ids):
        G.add_node(rid, index=i)

    for i in range(n):
        for j in neighbors[i]:
            if j <= i:
                continue
            if mode == "mutual" and i not in neighbors[j]:
                continue
            w = float(similarity[i, j])
            G.add_edge(respondent_ids[i], respondent_ids[j], weight=w)
    # Attach remaining isolates to their most similar peer so Louvain is not
    # dominated by singleton "communities".
    for i, rid in enumerate(respondent_ids):
        if G.degree(rid) > 0:
            continue
        row = similarity[i].copy()
        row[i] = -np.inf
        j = int(np.argmax(row))
        G.add_edge(rid, respondent_ids[j], weight=float(similarity[i, j]))
    return G


def connected_components_info(G: nx.Graph) -> dict:
    comps = list(nx.connected_components(G))
    sizes = sorted((len(c) for c in comps), reverse=True)
    return {
        "n_components": len(comps),
        "largest_component_size": sizes[0] if sizes else 0,
        "component_sizes": sizes,
    }
