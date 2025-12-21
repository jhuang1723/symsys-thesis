#!/usr/bin/env python3
"""
Filter matches_positive.csv across all chunks by removing rows whose verse_text
contains a given phrase. Files are updated in place.
"""

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Filter matches_positive.csv by phrase in verse_text across all chunks "
            "(in place)."
        )
    )
    p.add_argument(
        "--phrase",
        required=True,
        help='Phrase to remove when found in verse_text (use quotes for spaces).',
    )
    p.add_argument(
        "--case-sensitive",
        action="store_true",
        help="Use case-sensitive matching (default: case-insensitive).",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    base_dir = Path(
        "results/app/spoken-addresses-and-remarks/chunks"
    )
    for part in range(1, 13):
        part_id = f"{part:02d}"
        path = base_dir / f"windows_part_{part_id}" / "matches_positive.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing file: {path}")

        df = pd.read_csv(path)
        if "verse_text" not in df.columns:
            raise ValueError(f"Expected column 'verse_text' in {path}")

        if args.case_sensitive:
            mask = df["verse_text"].astype(str).str.contains(args.phrase, na=False)
        else:
            mask = df["verse_text"].astype(str).str.contains(
                args.phrase, case=False, na=False
            )

        before = len(df)
        df = df[~mask]
        after = len(df)
        df.to_csv(path, index=False)
        removed = before - after
        print(f"[ok] {path} removed={removed} remaining={after}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
