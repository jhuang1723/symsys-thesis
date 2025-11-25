#!/usr/bin/env python3
# scripts/train_bible_match_classifier.py

import argparse
from typing import List, Tuple

import numpy as np
import pandas as pd
from bible_match_model import BibleMatchClassifier, normalize_text


SNIPPET_CANDIDATES = [
    "most_unusual_phrase",
    "snippet",
    "speech_snippet",
    "snippet_norm",
]
VERSE_CANDIDATES = [
    "verse",
    "verse_text",
]


def load_training_data(train_csv: str) -> pd.DataFrame:
    df = pd.read_csv(train_csv)
    if "match" not in df.columns:
        raise ValueError(f"Training CSV must have a 'match' column. ({train_csv})")

    snippet_col = next((c for c in SNIPPET_CANDIDATES if c in df.columns), None)
    if snippet_col is None:
        raise ValueError(f"Could not find snippet column in {train_csv}. Tried {SNIPPET_CANDIDATES}")

    verse_candidates = VERSE_CANDIDATES + [c for c in df.columns if c.lower() == "verse_ref"]
    verse_col = next((c for c in verse_candidates if c in df.columns), None)
    if verse_col is None:
        raise ValueError(f"Could not find verse column in {train_csv}. Tried {verse_candidates}")

    df["snippet_norm"] = df[snippet_col].apply(normalize_text)
    df["verse_norm"] = df[verse_col].apply(normalize_text)
    df["match"] = df["match"].fillna(0).astype(int)

    df = df[(df["snippet_norm"] != "") & (df["verse_norm"] != "")]
    df = df.reset_index(drop=True)
    return df[["snippet_norm", "verse_norm", "match"]]


def combine_datasets(dfs: List[pd.DataFrame]) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Concatenate datasets while assigning sample weights so that each
    source contributes equally overall.
    """
    if not dfs:
        raise ValueError("No training datasets provided.")

    n_sources = len(dfs)
    frames = []
    weights = []
    per_dataset_weight = 1.0 / n_sources

    for df in dfs:
        frames.append(df)
        row_weight = per_dataset_weight / max(len(df), 1)
        weights.append(np.full(len(df), row_weight, dtype=np.float32))

    merged = pd.concat(frames, ignore_index=True)
    sample_weights = np.concatenate(weights)
    return merged, sample_weights


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-csv", required=True, help="Primary training CSV (with match column).")
    ap.add_argument("--extra-csv", action="append", default=[], help="Additional labeled CSVs to include.")
    ap.add_argument("--model-out", required=True, help="Where to save model, e.g. models/apb_lgbm_sem.pkl")
    ap.add_argument("--sem-model-name", default="all-MiniLM-L12-v2")
    args = ap.parse_args()

    datasets = [load_training_data(args.train_csv)]
    for extra in args.extra_csv:
        datasets.append(load_training_data(extra))

    df_all, sample_weights = combine_datasets(datasets)
    print(f"[info] Combined rows: {len(df_all)}, positives: {df_all['match'].sum()}")

    clf = BibleMatchClassifier(sem_model_name=args.sem_model_name)
    clf.fit(df_all, sample_weights=sample_weights)
    clf.save(args.model_out)


if __name__ == "__main__":
    main()
