#!/usr/bin/env python3
# scripts/train_bible_match_classifier.py

import argparse
import pandas as pd
from bible_match_model import BibleMatchClassifier, normalize_text


def load_training_data(train_csv: str) -> pd.DataFrame:
    df = pd.read_csv(train_csv)
    if "match" not in df.columns:
        raise ValueError("Training CSV must have a 'match' column.")

    if "most_unusual_phrase" in df.columns:
        snippet_col = "most_unusual_phrase"
    elif "snippet" in df.columns:
        snippet_col = "snippet"
    else:
        raise ValueError("Could not find 'most_unusual_phrase' or 'snippet' in training CSV.")

    if "verse" in df.columns:
        verse_col = "verse"
    elif "verse_text" in df.columns:
        verse_col = "verse_text"
    else:
        raise ValueError("Could not find 'verse' or 'verse_text' in training CSV.")

    df["snippet_norm"] = df[snippet_col].apply(normalize_text)
    df["verse_norm"] = df[verse_col].apply(normalize_text)
    df["match"] = df["match"].fillna(0).astype(int)

    df = df[(df["snippet_norm"] != "") & (df["verse_norm"] != "")]
    df = df.reset_index(drop=True)
    return df[["snippet_norm", "verse_norm", "match"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-csv", required=True, help="APB training CSV (with match column).")
    ap.add_argument("--model-out", required=True, help="Where to save model, e.g. models/apb_lgbm_sem.pkl")
    ap.add_argument("--sem-model-name", default="all-MiniLM-L12-v2")
    args = ap.parse_args()

    df_train = load_training_data(args.train_csv)
    print(f"Training rows: {len(df_train)}, positives: {df_train['match'].sum()}")

    clf = BibleMatchClassifier(sem_model_name=args.sem_model_name)
    clf.fit(df_train)
    clf.save(args.model_out)


if __name__ == "__main__":
    main()
