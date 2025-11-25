#!/usr/bin/env python3
"""
Interactive CLI for reviewing pipeline matches.

By default, consumes results/app/<category>/matches_positive.csv, shows one
row at a time, and lets the reviewer press:
  y = confirmed match
  n = not a match
  s = skip (leave unlabeled for now)
  q = quit

Responses are appended to a CSV (default: data/training_data/reviewed_matches.csv)
that can be merged back into the training set later.
"""

from __future__ import annotations

import argparse
import csv
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


WRAP = textwrap.TextWrapper(width=100)


@dataclass(frozen=True)
class MatchKey:
    doc_id: str
    start_token: int
    verse_range: str

    @classmethod
    def from_row(cls, row: pd.Series) -> "MatchKey":
        return cls(
            doc_id=str(row.get("doc_id", "")),
            start_token=int(row.get("start_token", 0)),
            verse_range=str(row.get("verse_range", "")),
        )

    def serialize(self) -> str:
        return f"{self.doc_id}|{self.start_token}|{self.verse_range}"


def load_matches(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"Input matches file not found: {path}")
    if path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    if "match_proba" not in df.columns:
        df["match_proba"] = None
    if "verse_range" not in df.columns and "ref" in df.columns:
        df = df.rename(columns={"ref": "verse_range"})
    return df


def load_existing_labels(out_path: Path) -> set[str]:
    if not out_path.exists():
        return set()
    existing = set()
    with out_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = row.get("match_id")
            if key:
                existing.add(key)
    return existing


def append_label(out_path: Path, fieldnames: list[str], row: dict):
    new_file = not out_path.exists()
    with out_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if new_file:
            writer.writeheader()
        writer.writerow(row)


def format_block(title: str, text: str) -> str:
    wrapped = WRAP.fill(text.strip()) if isinstance(text, str) else ""
    return f"{title}:\n{wrapped}\n"


def prompt_user(row: pd.Series) -> str:
    print("=" * 100)
    print(f"Doc: {row.get('title','')} ({row.get('president','')}, {row.get('date','')})")
    print(f"Verse: {row.get('verse_range','')} (score={row.get('match_proba')})")
    snippet = row.get("snippet_norm") or row.get("snippet_raw") or ""
    verse_text = row.get("verse_text") or ""
    print(format_block("Snippet", snippet))
    print(format_block("Verse", verse_text))
    while True:
        resp = input("[y]es / [n]o / [s]kip / [q]uit > ").strip().lower()
        if resp in {"y", "n", "s", "q"}:
            return resp
        print("Please enter y, n, s, or q.")


def main():
    ap = argparse.ArgumentParser(description="Interactive reviewer for matches_positive.csv")
    ap.add_argument(
        "--input",
        default="results/app/eulogies/matches_positive.csv",
        help="CSV or Parquet file with matches to review.",
    )
    ap.add_argument(
        "--out",
        default="data/training_data/reviewed_matches.csv",
        help="Where to append reviewer labels.",
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max rows to review this session (default: all remaining).",
    )
    args = ap.parse_args()

    matches = load_matches(Path(args.input))
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    existing = load_existing_labels(out_path)
    print(f"[info] Loaded {len(matches)} rows ({len(existing)} already labeled).")

    fieldnames = [
        "match_id",
        "doc_id",
        "title",
        "president",
        "date",
        "snippet_norm",
        "verse_range",
        "verse_text",
        "match_proba",
        "label",
    ]

    reviewed = 0
    for row in matches.itertuples(index=False):
        row_dict = row._asdict()
        key = MatchKey.from_row(pd.Series(row_dict))
        key_str = key.serialize()
        if key_str in existing:
            continue
        resp = prompt_user(pd.Series(row_dict))
        if resp == "q":
            print("[info] exiting; progress saved.")
            break
        if resp == "s":
            continue
        label = 1 if resp == "y" else 0
        record = {
            "match_id": key_str,
            "doc_id": row_dict.get("doc_id"),
            "title": row_dict.get("title"),
            "president": row_dict.get("president"),
            "date": row_dict.get("date"),
            "snippet_norm": row_dict.get("snippet_norm"),
            "verse_range": row_dict.get("verse_range") or row_dict.get("ref", ""),
            "verse_text": row_dict.get("verse_text"),
            "match_proba": row_dict.get("match_proba"),
            "label": label,
        }
        append_label(out_path, fieldnames, record)
        existing.add(key_str)
        reviewed += 1
        if args.limit and reviewed >= args.limit:
            print(f"[info] Reached limit ({args.limit}); stopping.")
            break

    print(f"[done] Reviewed {reviewed} new rows. Labels stored at {out_path}")


if __name__ == "__main__":
    main()
