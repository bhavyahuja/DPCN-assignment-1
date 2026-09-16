# DPCN Assignment 1: Opinion Network Formation

Respondent-similarity network from the class survey on Technology, Education, Ethics, and Environment.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python run_pipeline.py
```

Outputs:

- `report/report.pdf` — assignment report
- `outputs/figures/` — network plots
- `outputs/metrics.json` — computed graph statistics

## Repository layout

- `data/Survey_Results_UC.csv` — survey responses
- `src/preprocess.py` — Likert encoding (1–5) and category tags
- `src/network.py` — Pearson similarity and k-NN graphs
- `src/analysis.py` — metrics, centrality, Louvain communities
- `src/visualize.py` — figures
- `src/report.py` — PDF assembly
- `run_pipeline.py` — end-to-end run

## Network definition

Nodes are respondents. Similarity is Pearson correlation of the 60-dimensional Likert vector. Two students are connected if either is among the other’s 8 nearest neighbors and the correlation is positive. The same construction is repeated on each 15-item category.

Replace `[TEAM NAME TBD]` and `[GITHUB URL TBD]` in the report (or this README) before submission.
