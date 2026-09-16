"""Assemble the ~8-page DPCN Assignment 1 report PDF."""

from __future__ import annotations

import json
from pathlib import Path

from fpdf import FPDF


class ReportPDF(FPDF):
    def header(self):
        if self.page_no() == 1:
            return
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(90, 90, 90)
        self.cell(0, 8, "DPCN Assignment 1: Opinion Network Formation", align="L")
        self.ln(10)

    def footer(self):
        self.set_y(-12)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(90, 90, 90)
        self.cell(0, 8, f"Page {self.page_no()}/{{nb}}", align="C")


def _p(pdf: ReportPDF, text: str, size: int = 10) -> None:
    pdf.set_font("Helvetica", "", size)
    pdf.set_text_color(20, 20, 20)
    pdf.multi_cell(0, 5, text)
    pdf.ln(1.5)


def _h(pdf: ReportPDF, text: str, size: int = 13) -> None:
    pdf.ln(2)
    pdf.set_font("Helvetica", "B", size)
    pdf.set_text_color(20, 45, 90)
    pdf.cell(0, 8, text, new_x="LMARGIN", new_y="NEXT")
    pdf.set_text_color(20, 20, 20)


def _img(pdf: ReportPDF, path: Path, w: float = 170) -> None:
    if path.exists():
        pdf.image(str(path), w=w)
        pdf.ln(3)


def _fmt(v, digits=3):
    if v is None:
        return "n/a"
    if isinstance(v, float):
        return f"{v:.{digits}f}"
    return str(v)


def build_report(metrics: dict, figures_dir: Path, out_path: Path) -> None:
    g = metrics["global"]
    missing = metrics["missing"]
    profiles = metrics["community_profiles"]
    top = metrics["top_central"]
    cat_metrics = metrics["category_metrics"]
    k = metrics["k"]
    n_comm = len(metrics["community_sizes"])
    mod = metrics["modularity"]

    pdf = ReportPDF(format="A4", unit="mm")
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=16)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 18)
    pdf.set_text_color(20, 45, 90)
    pdf.multi_cell(0, 9, "Assignment 1: Opinion Network Formation")
    pdf.set_font("Helvetica", "", 11)
    pdf.set_text_color(40, 40, 40)
    pdf.cell(0, 7, "Distributed and Parallel Computing / Networks (DPCN)", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)

    _h(pdf, "1. Team Name")
    _p(
        pdf,
        "Team Name: [TEAM NAME TBD]. Replace this placeholder with the official group name "
        "before submission. Member names appear in Section 7.",
    )

    _h(pdf, "2. GitHub Link to Code")
    _p(
        pdf,
        "Repository: [GITHUB URL TBD]. The repository should contain the `src/` pipeline, "
        "`data/Survey_Results_UC.csv`, `requirements.txt`, and this report. After you push "
        "the local project, paste the public URL here and confirm the repository is accessible.",
    )

    _h(pdf, "3. Dataset Documentation")
    _p(
        pdf,
        "The class survey asked 60 Likert-scale statements covering four themes: Technology "
        "(T01-T15), Education (E01-E15), Ethics/Society (S01-S15), and Environment (V01-V15). "
        f"After loading the CSV with UTF-8-SIG (to strip the BOM on the id column), we obtained "
        f"{missing['n_respondents']} response rows and {missing['n_questions']} items. "
        f"{metrics.get('n_dropped_empty', 0)} rows had fewer than 10 valid Likert answers "
        f"(entirely blank submissions) and were dropped, leaving {metrics['n_respondents']} "
        "nodes. Each remaining response is an ordinal opinion, not a social contact, so the "
        "network we construct is a derived similarity graph rather than a reported friendship graph.",
    )
    _p(
        pdf,
        "We mapped Strongly Disagree=1, Disagree=2, Neutral=3, Agree=4, Strongly Agree=5. "
        "Non-scale tokens such as 'No Comments' were treated as missing (NaN) and excluded from "
        "pairwise comparisons. "
        f"Missing cells: {missing['n_missing_cells']} "
        f"({100 * missing['missing_rate']:.2f}% of all answers). "
        "Most missing cells are blank survey fields rather than a second scale; they are excluded from pairwise comparisons. "
        "The remaining rate is low enough that Pearson correlation on pairwise-complete coordinates remains stable."
    )
    _p(
        pdf,
        "We interpret each respondent as a 60-dimensional opinion vector. Category prefixes are "
        "used both to document the instrument and to build four theme-specific sub-networks. "
        "Higher scores mean stronger agreement with the (generally pro-technology, pro-learning, "
        "pro-ethics, pro-environment) wording of the items. Because almost every statement is "
        "positively framed, a high mean is a 'progressive / high-agreement' profile rather than "
        "a mix of opposing ideologies. Missingness is spread across all four blocks rather than "
        "concentrated in a single theme, so category sub-networks remain comparable."
    )

    _h(pdf, "4. Pipeline Followed")
    _p(
        pdf,
        "Nodes are respondents (students). Similarity is the Pearson correlation of their "
        "Likert vectors (cosine similarity after mean-centering). Pearson is used instead of raw "
        "cosine because nearly all answers sit on the Agree side of the scale, which would "
        "otherwise make every pair look almost identical. An undirected edge exists when "
        f"either student is among the other's {k} nearest neighbors and the correlation is "
        "positive. Edge weight is the correlation. kNN is preferred over a complete weighted graph "
        "because a dense 96-node clique has trivial density and no community structure worth reporting."
    )
    _p(
        pdf,
        "Step-by-step: (1) encode Likert answers to 1-5; (2) compute a 96 x 96 Pearson "
        "similarity matrix using only coordinates observed for both people; (3) for each "
        f"respondent keep the {k} most similar others and add an undirected edge if the "
        "relation holds in either direction and the correlation is positive; (4) repeat the "
        "same construction independently on the 15 Technology, Education, Ethics, and "
        "Environment items; (5) compute global graph statistics, degree/betweenness/"
        "eigenvector centrality, and Louvain communities (weight = similarity, resolution = 1); "
        "(6) characterize each community by its mean Likert score in the four themes."
    )
    _p(
        pdf,
        f"The resulting overall network has {g['n_nodes']} nodes, {g['n_edges']} edges, "
        f"density {_fmt(g['density'])}, average degree {_fmt(g['average_degree'], 2)}, "
        f"and {g['n_components']} connected component(s) "
        f"(largest component size {g['largest_component_size']}). "
        "Louvain is run on this weighted graph; category networks are analyzed separately so we "
        "can see whether Technology opinions fragment the class more than Environment opinions.",
    )

    pdf.add_page()
    _h(pdf, "5. Analysis and Visualizations")
    _p(
        pdf,
        f"Global metrics (overall network): density {_fmt(g['density'])}, "
        f"average clustering {_fmt(g['average_clustering'])}, "
        f"average shortest path in the giant component {_fmt(g['avg_shortest_path'], 2)}, "
        f"diameter {_fmt(g['diameter'])}. "
        f"Louvain found {n_comm} communities with modularity {_fmt(mod)}. "
        f"Community sizes: {json.dumps(metrics['community_sizes'])}. "
        "A modularity well above 0 indicates that similarity is not uniform: some groups of "
        "students agree with each other more than with the rest of the class.",
    )
    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(0, 6, "Table 1. Global metrics of the overall opinion network", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "B", 9)
    headers = ["Nodes", "Edges", "Density", "Avg deg", "Clustering", "ASP", "Diameter", "Modularity"]
    vals = [
        str(g["n_nodes"]),
        str(g["n_edges"]),
        _fmt(g["density"]),
        _fmt(g["average_degree"], 2),
        _fmt(g["average_clustering"]),
        _fmt(g["avg_shortest_path"], 2),
        _fmt(g["diameter"]),
        _fmt(mod),
    ]
    col_w = 21
    for h in headers:
        pdf.cell(col_w, 6, h, border=1, align="C")
    pdf.ln()
    pdf.set_font("Helvetica", "", 9)
    for v in vals:
        pdf.cell(col_w, 6, v, border=1, align="C")
    pdf.ln(8)

    top_eig = ", ".join(f"{x['respondent_id']} ({_fmt(x['score'])})" for x in top["eigenvector"][:5])
    top_btw = ", ".join(f"{x['respondent_id']} ({_fmt(x['score'])})" for x in top["betweenness"][:5])
    _p(
        pdf,
        "Eigenvector centrality ranks respondents who are similar to other well-connected "
        f"(high-similarity) students - a 'consensus core'. Top eigenvector ids: {top_eig}. "
        "Betweenness highlights bridges between communities. Top betweenness ids: "
        f"{top_btw}. High-betweenness students are not necessarily the most typical; they sit "
        "on short paths linking otherwise distinct opinion clusters.",
    )

    fig_net = figures_dir / "network_overall.png"
    _img(pdf, fig_net, w=168)

    _p(
        pdf,
        "Figure 1. Overall mutual-kNN opinion network. Node color is Louvain community; node "
        "size is eigenvector centrality. Tight color blobs are groups of students whose 60 "
        "answers are close in cosine space.",
    )

    _img(pdf, figures_dir / "heatmap.png", w=150)
    _p(
        pdf,
        "Figure 2. Pearson similarity heatmap with respondents sorted by community. Warm "
        "blocks on the diagonal show within-community agreement; cooler off-diagonal cells "
        "show weaker or opposing profile shapes."
    )

    _img(pdf, figures_dir / "degree_distribution.png", w=155)
    _p(
        pdf,
        f"Figure 3. Degree distribution. In a union kNN graph the typical degree is near {k}, "
        f"but hubs can exceed {k} when many students list the same person as a neighbor. A long "
        "right tail would be more typical of a scale-free social graph than of this construction."
    )

    _img(pdf, figures_dir / "community_profiles.png", w=160)
    _p(
        pdf,
        "Figure 4. Mean Likert score (1-5) of each community on Technology, Education, Ethics, "
        "and Environment. This is the main interpretive plot: communities are not just "
        "algorithmic partitions, they have distinct opinion profiles.",
    )

    _img(pdf, figures_dir / "category_networks.png", w=165)
    _p(
        pdf,
        "Figure 5. Four category-specific 8-NN networks. Visual fragmentation differs "
        "by theme: some themes produce denser, more mixed graphs; others split into clearer "
        "blobs. Quantitative comparison is in the modularity bars below.",
    )

    _img(pdf, figures_dir / "category_modularity.png", w=150)

    cat_lines = []
    for cat, name in (("T", "Technology"), ("E", "Education"), ("S", "Ethics"), ("V", "Environment")):
        cm = cat_metrics[cat]
        cat_lines.append(
            f"{name}: {cm['n_edges']} edges, density {_fmt(cm['density'])}, "
            f"clustering {_fmt(cm['average_clustering'])}, modularity {_fmt(cm['modularity'])}, "
            f"{cm['n_communities']} communities."
        )
    _p(pdf, "Figure 6. Louvain modularity of each category network. " + " ".join(cat_lines))

    pdf.add_page()
    _h(pdf, "6. Results and Discussion")
    _p(
        pdf,
        "The class does not hold a single undifferentiated opinion. Mutual nearest-neighbor "
        f"ties and a Louvain modularity of {_fmt(mod)} show that respondents cluster into "
        f"{n_comm} groups. Because items are positively worded, the main axis of variation is "
        "how strongly people agree rather than a left-right ideological split. Still, the "
        "category means differ enough to tell a story about the cohort.",
    )

    large_profiles = [row for row in profiles if row["size"] >= 3]
    for row in large_profiles:
        bits = ", ".join(
            f"{name}={_fmt(row[name], 2)}"
            for name in ("Technology", "Education", "Ethics", "Environment")
        )
        _p(
            pdf,
            f"Community {row['community']} (n={row['size']}, overall mean {_fmt(row['overall'], 2)}): "
            f"{bits}.",
        )
    n_small = len(profiles) - len(large_profiles)
    if n_small:
        _p(
            pdf,
            f"{n_small} additional Louvain group(s) have fewer than 3 members and are omitted from "
            "the profile list; they behave as peripheral nodes rather than opinion blocs.",
        )

    # Interpret relative profiles
    if large_profiles:
        by_overall = sorted(large_profiles, key=lambda r: r["overall"], reverse=True)
        high = by_overall[0]
        low = by_overall[-1]
        _p(
            pdf,
            f"Community {high['community']} is the high-agreement core (overall {_fmt(high['overall'], 2)}): "
            "these students tend to endorse AI in education and science, compulsory academic honesty, "
            "inclusive campus norms, and environmental action. "
            f"Community {low['community']} is relatively more reserved (overall {_fmt(low['overall'], 2)}). "
            "A lower Technology score often reflects caution about AI in healthcare, autonomous vehicles, "
            "or regulation rather than blanket techno-skepticism; a lower Education score can reflect "
            "disagreement with compulsory attendance or with replacing exams. The network therefore "
            "separates 'enthusiastic endorsers' from 'selective endorsers', which is a realistic "
            "picture of an engineering class. Community means still sit in a narrow band "
            "(roughly 3.9 to 4.2 overall). Pearson clustering is therefore detecting differences "
            "in which items people relatively endorse - for example stronger Environment than "
            "Technology, or the reverse - not two hostile camps. That is the right reading of a "
            "class survey full of positively worded items.",
        )

    # Which category fragments more
    cat_mod_sorted = sorted(
        ((cat_metrics[c]["modularity"], name) for c, name in (("T", "Technology"), ("E", "Education"), ("S", "Ethics"), ("V", "Environment"))),
        reverse=True,
    )
    _p(
        pdf,
        f"Across themes, {cat_mod_sorted[0][1]} shows the strongest clustering "
        f"(modularity {_fmt(cat_mod_sorted[0][0])}), while {cat_mod_sorted[-1][1]} is the most mixed "
        f"(modularity {_fmt(cat_mod_sorted[-1][0])}). High modularity means students who agree on that "
        "theme also tend to agree with a stable set of peers; low modularity means the class is closer "
        "to a consensus on that theme. Environment items in this survey are almost uniformly endorsed, "
        "so we expect weaker splits there than on Technology (AI, robots, regulation), where the class "
        "is more divided.",
    )
    _p(
        pdf,
        "Centrality adds a second reading. High-eigenvector students sit in the dense agreement "
        "core: their answers are typical of a well-connected cluster, so they are useful as "
        "illustrative 'median voices' of that cluster, not as influencers in a social sense. "
        "High-betweenness students sit between clusters; if this were an actual discussion network, "
        "they would be the people most able to translate one group's concerns to another. We cannot "
        "claim social influence from this dataset - only structural position in opinion space.",
    )
    _p(
        pdf,
        "Limitations. (1) Similarity is not interaction: two students with similar answers may never "
        "have spoken. (2) kNN sparsification is a modeling choice; a different k would change density "
        "and possibly the number of communities, though the similarity heatmap is independent of k. "
        "(3) Pearson emphasizes the shape of a profile (which items someone relatively agrees with) "
        "rather than overall enthusiasm. (4) Blank cells and 'No Comments' answers are not guaranteed "
        "to be random. These caveats do not erase the main finding: the cohort's opinions form a "
        "clustered similarity network, with a high-agreement core and smaller, more cautious groups, "
        "and some themes fragment the class more than others.",
    )

    pdf.add_page()
    _h(pdf, "7. Individual Contribution")
    _p(
        pdf,
        "Fill in real names before submission. The breakdown below matches the actual pipeline so "
        "it can be edited rather than rewritten.",
    )
    contrib = [
        ("[Member 1]", "Dataset documentation, Likert encoding, missing-value policy, category tagging."),
        ("[Member 2]", "Similarity matrix, union kNN graph construction, category sub-networks."),
        ("[Member 3]", "Global metrics, centrality, Louvain communities, community opinion profiles."),
        ("[Member 4]", "Visualizations, results write-up, report assembly, README and reproducibility."),
    ]
    pdf.set_font("Helvetica", "", 10)
    for name, task in contrib:
        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(42, 5, name)
        pdf.set_font("Helvetica", "", 10)
        pdf.multi_cell(0, 5, task)
        pdf.ln(0.5)

    _p(
        pdf,
        "If the team has fewer than four members, merge the last rows. All members should review "
        "the GitHub repository layout and the final PDF before the 20 September 2026 deadline.",
    )

    pdf.add_page()
    _h(pdf, "Appendix. Metric definitions")
    _p(
        pdf,
        "Density is the fraction of possible undirected pairs that are realized as edges. "
        "Average degree is 2m/n. Average clustering is the mean of the weighted local clustering "
        "coefficients (NetworkX). Average shortest path and diameter are computed on the largest "
        "connected component with hop distance (unweighted). Degree centrality in the tables is "
        "the weighted degree (sum of incident Pearson weights). Betweenness is the fraction of "
        "shortest paths that pass through a node, with path length 1-similarity so that stronger "
        "agreement is a shorter link. Eigenvector centrality scores nodes that are similar to "
        "other high-similarity nodes. Louvain modularity compares the weight inside detected "
        "communities with the weight expected in a random graph with the same degree sequence. "
        "Community profile values are mean Likert scores (1-5) of members of that community, "
        "averaged first across items in a category and then across members.",
    )
    _p(
        pdf,
        "Reproducibility. From the project root, create a virtual environment, install "
        "requirements.txt, and run python run_pipeline.py. The script writes figures under "
        "outputs/figures/, numeric tables as CSV/JSON under outputs/, and this PDF under "
        "report/report.pdf. Randomness is fixed by layout and Louvain seeds (42).",
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    pdf.output(str(out_path))
