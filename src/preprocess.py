"""Load and encode the DPCN opinion survey."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

LIKERT_MAP = {
    "strongly disagree": 1,
    "disagree": 2,
    "neutral": 3,
    "agree": 4,
    "strongly agree": 5,
}

CATEGORY_LABELS = {
    "T": "Technology",
    "E": "Education",
    "S": "Ethics",
    "V": "Environment",
}


def _question_code(column: str) -> str:
    """Extract T01-style code from a survey column name."""
    return column.split(".", 1)[0].strip()


def _category_from_code(code: str) -> str:
    return code[0]


def load_survey(csv_path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    df.columns = [c.strip() for c in df.columns]
    id_col = df.columns[0]
    df = df.rename(columns={id_col: "respondent_id"})
    df["respondent_id"] = df["respondent_id"].astype(str)
    return df


def encode_likert(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Return ordinal scores (1-5), a missingness mask, and column metadata."""
    question_cols = [c for c in df.columns if c != "respondent_id"]
    meta = []
    encoded = pd.DataFrame({"respondent_id": df["respondent_id"]})
    missing = pd.DataFrame({"respondent_id": df["respondent_id"]})

    for col in question_cols:
        code = _question_code(col)
        cat = _category_from_code(code)
        meta.append(
            {
                "column": col,
                "code": code,
                "category": cat,
                "category_name": CATEGORY_LABELS.get(cat, cat),
                "text": col.split(".", 1)[1].strip() if "." in col else col,
            }
        )
        raw = df[col].astype(str).str.strip()
        scores = raw.str.lower().map(LIKERT_MAP)
        encoded[code] = scores
        missing[code] = scores.isna()

    meta_df = pd.DataFrame(meta)
    return encoded, missing, meta_df


def category_columns(meta_df: pd.DataFrame, category: str) -> list[str]:
    return meta_df.loc[meta_df["category"] == category, "code"].tolist()


def score_matrix(encoded: pd.DataFrame, columns: list[str] | None = None) -> np.ndarray:
    cols = columns if columns is not None else [c for c in encoded.columns if c != "respondent_id"]
    return encoded[cols].to_numpy(dtype=float)


def drop_empty_respondents(encoded: pd.DataFrame, min_answers: int = 10) -> pd.DataFrame:
    cols = [c for c in encoded.columns if c != "respondent_id"]
    keep = encoded[cols].notna().sum(axis=1) >= min_answers
    return encoded.loc[keep].reset_index(drop=True)


def summarize_missing(missing: pd.DataFrame) -> dict:
    cols = [c for c in missing.columns if c != "respondent_id"]
    n_cells = missing[cols].size
    n_missing = int(missing[cols].sum().sum())
    by_item = missing[cols].sum().to_dict()
    return {
        "n_respondents": int(len(missing)),
        "n_questions": int(len(cols)),
        "n_missing_cells": n_missing,
        "missing_rate": float(n_missing / n_cells) if n_cells else 0.0,
        "items_with_missing": {k: int(v) for k, v in by_item.items() if v},
    }
