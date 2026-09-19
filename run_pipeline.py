#!/usr/bin/env python3
"""End-to-end opinion-network pipeline: data -> graphs -> figures and metrics."""

from __future__ import annotations

import json
from pathlib import Path

from src.analysis import (
    centrality_scores,
    community_profiles,
    community_size_summary,
    global_metrics,
    louvain_communities,
    modularity,
    top_central,
)
from src.network import mutual_knn_graph, pairwise_pearson_similarity
from src.preprocess import (
    CATEGORY_LABELS,
    category_columns,
    drop_empty_respondents,
    encode_likert,
    load_survey,
    score_matrix,
    summarize_missing,
)
from src.visualize import (
    plot_category_modularity,
    plot_category_networks,
    plot_community_profiles,
    plot_degree_distribution,
    plot_network,
    plot_similarity_heatmap,
)

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "Survey_Results_UC.csv"
FIG = ROOT / "outputs" / "figures"
OUT_METRICS = ROOT / "outputs" / "metrics.json"
K = 8


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)

    df = load_survey(DATA)
    encoded, missing, meta = encode_likert(df)
    n_raw = len(encoded)
    encoded = drop_empty_respondents(encoded, min_answers=10)
    n_dropped = n_raw - len(encoded)
    ids = encoded["respondent_id"].tolist()
    X = score_matrix(encoded)
    sim = pairwise_pearson_similarity(X)
    G = mutual_knn_graph(sim, k=K, respondent_ids=ids)

    partition = louvain_communities(G)
    cent = centrality_scores(G)
    profiles = community_profiles(encoded, partition, meta)
    gmetrics = global_metrics(G)
    miss = summarize_missing(missing[missing["respondent_id"].isin(ids)].reset_index(drop=True))

    cat_graphs = {}
    cat_parts = {}
    cat_metrics = {}
    for cat in CATEGORY_LABELS:
        cols = category_columns(meta, cat)
        Xc = score_matrix(encoded, cols)
        simc = pairwise_pearson_similarity(Xc)
        Gc = mutual_knn_graph(simc, k=K, respondent_ids=ids)
        pc = louvain_communities(Gc)
        cat_graphs[cat] = Gc
        cat_parts[cat] = pc
        gm = global_metrics(Gc)
        gm["modularity"] = modularity(Gc, pc)
        gm["n_communities"] = len(set(pc.values()))
        cat_metrics[cat] = gm

    plot_network(G, partition, cent, FIG / "network_overall.png")
    plot_category_networks(cat_graphs, cat_parts, FIG / "category_networks.png")
    plot_similarity_heatmap(sim, ids, partition, FIG / "heatmap.png")
    plot_degree_distribution(G, FIG / "degree_distribution.png")
    plot_community_profiles(profiles, FIG / "community_profiles.png")
    plot_category_modularity({c: cat_metrics[c]["modularity"] for c in CATEGORY_LABELS}, FIG / "category_modularity.png")

    metrics = {
        "k": K,
        "missing": miss,
        "global": gmetrics,
        "modularity": modularity(G, partition),
        "community_sizes": community_size_summary(partition),
        "community_profiles": profiles.to_dict(orient="records"),
        "top_central": top_central(cent, 5),
        "category_metrics": cat_metrics,
        "n_respondents": len(ids),
        "n_dropped_empty": n_dropped,
    }

    OUT_METRICS.write_text(json.dumps(metrics, indent=2))
    cent.to_csv(ROOT / "outputs" / "centrality.csv", index=False)
    profiles.to_csv(ROOT / "outputs" / "community_profiles.csv", index=False)
    encoded.to_csv(ROOT / "outputs" / "encoded_scores.csv", index=False)

    print(f"Wrote {OUT_METRICS}")
    print(f"Communities: {metrics['community_sizes']}")
    print(f"Modularity: {metrics['modularity']:.3f}")
    print(f"Edges: {gmetrics['n_edges']}")


if __name__ == "__main__":
    main()
