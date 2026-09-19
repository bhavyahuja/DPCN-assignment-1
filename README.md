# DPCN Assignment 1: Opinion Network Formation

Team **1885smashburgers**. Respondent-similarity network from the class survey on Technology, Education, Ethics, and Environment.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python run_pipeline.py
```

That rebuilds the graph, metrics, and figures. Then compile the report:

```bash
cd report
latexmk -xelatex report.tex
```

XeLaTeX needs the Libertinus fonts (bundled with TinyTeX as `libertinus-fonts`).

## Outputs

- `report/report.pdf` — assignment report
- `report/report.tex` — report source
- `outputs/figures/` — network plots
- `outputs/metrics.json` — computed graph statistics

## Repository layout

- `data/Survey_Results_UC.csv` — survey responses
- `src/preprocess.py` — Likert encoding (1–5) and category tags
- `src/network.py` — Pearson similarity and k-NN graphs
- `src/analysis.py` — metrics, centrality, Louvain communities
- `src/visualize.py` — figures
- `run_pipeline.py` — builds graphs, figures, and metrics
- `report/report.tex` — written report

## Network definition

Nodes are respondents. Similarity is Pearson correlation of the 60-dimensional Likert vector. Two students are connected if either is among the other’s 8 nearest neighbours. The same construction is repeated on each 15-item category.

Code: https://github.com/bhavyahuja/DPCN-assignment-1
