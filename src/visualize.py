"""Figures for the opinion-network report."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns

from .preprocess import CATEGORY_LABELS

PALETTE = sns.color_palette("tab10", 10)


def _community_colors(partition: dict[str, int]) -> dict[str, tuple]:
    return {n: PALETTE[c % len(PALETTE)] for n, c in partition.items()}


def plot_network(
    G: nx.Graph,
    partition: dict[str, int],
    centrality: pd.DataFrame,
    out_path: Path,
    title: str = "Opinion similarity network",
) -> None:
    size_map = dict(zip(centrality["respondent_id"], centrality["eigenvector"]))
    sizes = [300 + 2500 * size_map.get(n, 0.0) for n in G.nodes()]
    colors = [_community_colors(partition).get(n, (0.5, 0.5, 0.5)) for n in G.nodes()]

    pos = nx.spring_layout(G, seed=42, k=1.2 / np.sqrt(max(G.number_of_nodes(), 1)), weight="weight")
    fig, ax = plt.subplots(figsize=(9.5, 7.2))
    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.25, width=0.8, edge_color="#555555")
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=sizes, node_color=colors, linewidths=0.4, edgecolors="white")
    ax.set_title(title)
    ax.axis("off")
    comm_ids = sorted(set(partition.values()))
    handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=PALETTE[c % len(PALETTE)], markersize=8, label=f"C{c}")
        for c in comm_ids
    ]
    ax.legend(handles=handles, title="Community", loc="upper right", frameon=True, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_category_networks(
    graphs: dict[str, nx.Graph],
    partitions: dict[str, dict[str, int]],
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10, 9))
    axes = axes.ravel()
    for ax, (cat, name) in zip(axes, CATEGORY_LABELS.items()):
        G = graphs[cat]
        part = partitions[cat]
        pos = nx.spring_layout(G, seed=7, k=1.1 / np.sqrt(max(G.number_of_nodes(), 1)), weight="weight")
        colors = [_community_colors(part).get(n, (0.5, 0.5, 0.5)) for n in G.nodes()]
        nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.2, width=0.6)
        nx.draw_networkx_nodes(G, pos, ax=ax, node_size=40, node_color=colors, linewidths=0.2, edgecolors="white")
        ax.set_title(f"{name} ({cat})")
        ax.axis("off")
    fig.suptitle("Category-specific similarity networks (8-NN)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_similarity_heatmap(
    similarity: np.ndarray,
    respondent_ids: list[str],
    partition: dict[str, int],
    out_path: Path,
) -> None:
    order = sorted(range(len(respondent_ids)), key=lambda i: (partition[respondent_ids[i]], respondent_ids[i]))
    S = similarity[np.ix_(order, order)]
    fig, ax = plt.subplots(figsize=(8.2, 7.0))
    sns.heatmap(
        S,
        ax=ax,
        cmap="vlag",
        center=0,
        vmin=-0.4,
        vmax=1.0,
        xticklabels=False,
        yticklabels=False,
        cbar_kws={"label": "Pearson similarity"},
    )
    ax.set_title("Respondent similarity, sorted by Louvain community")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_degree_distribution(G: nx.Graph, out_path: Path) -> None:
    degrees = [d for _, d in G.degree()]
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    sns.histplot(degrees, bins=range(0, max(degrees) + 2), ax=ax, discrete=True, color="#3b6ea5")
    ax.set_xlabel("Degree")
    ax.set_ylabel("Respondents")
    ax.set_title("Degree distribution of the opinion network")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_community_profiles(profiles: pd.DataFrame, out_path: Path) -> None:
    cats = list(CATEGORY_LABELS.values())
    data = profiles[profiles["size"] >= 3].copy() if "size" in profiles.columns else profiles
    if data.empty:
        data = profiles
    melted = data.melt(id_vars=["community", "size"], value_vars=cats, var_name="category", value_name="mean_score")
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    sns.barplot(data=melted, x="category", y="mean_score", hue="community", ax=ax, palette="tab10")
    ax.set_ylim(1, 5)
    ax.set_ylabel("Mean Likert score (1–5)")
    ax.set_xlabel("")
    ax.set_title("Community opinion profiles by survey category")
    ax.legend(title="Community", fontsize=8, loc="best")
    fig.subplots_adjust(bottom=0.15)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_category_modularity(modularity_by_cat: dict[str, float], out_path: Path) -> None:
    names = [CATEGORY_LABELS[c] for c in CATEGORY_LABELS]
    vals = [modularity_by_cat[c] for c in CATEGORY_LABELS]
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.bar(names, vals, color=["#3b6ea5", "#d97706", "#059669", "#7c3aed"])
    ax.set_ylabel("Louvain modularity")
    ax.set_title("Opinion clustering strength by category")
    ax.set_ylim(0, max(0.35, max(vals) * 1.15 if vals else 0.35))
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
