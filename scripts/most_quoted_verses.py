#!/usr/bin/env python3
"""
Produce a table of the most quoted Bible verses from cleaned-results/full-results.csv.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path


def _read_rows(path: Path) -> tuple[list[str], list[list[str]], int]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)

        if "verse_range" not in header or "verse_text" not in header:
            raise ValueError("Expected columns: verse_range, verse_text")

        match_idx = header.index("match_proba") if "match_proba" in header else None
        expected_len = len(header)
        rows = []
        skipped = 0

        for row in reader:
            if len(row) == expected_len:
                rows.append(row)
                continue

            # Fix common malformed row with a stray empty field after verse_text.
            if match_idx is not None and len(row) == expected_len + 1 and row[match_idx] == "":
                row.pop(match_idx)
                if len(row) == expected_len:
                    rows.append(row)
                    continue

            skipped += 1

    return header, rows, skipped


def _write_csv(rows: list[dict[str, str]], out_path: Path | None) -> None:
    fieldnames = ["verse_range", "verse_text", "quote_count"]
    out_f = sys.stdout if out_path is None else out_path.open("w", encoding="utf-8", newline="")
    with out_f:
        writer = csv.DictWriter(out_f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(rows: list[dict[str, str]]) -> None:
    print("| verse_range | quote_count | verse_text |")
    print("| --- | ---: | --- |")
    for r in rows:
        verse_text = r["verse_text"].replace("|", "\\|")
        print(f'| {r["verse_range"]} | {r["quote_count"]} | {verse_text} |')


def main() -> int:
    p = argparse.ArgumentParser(
        description="Create a table of the most quoted Bible verses."
    )
    p.add_argument(
        "--input",
        default="cleaned-results/full-results.csv",
        help="Input CSV (default: cleaned-results/full-results.csv)",
    )
    p.add_argument(
        "--output",
        default="results/tables/most_quoted_verses.csv",
        help="Output CSV path (default: results/most_quoted_verses.csv). Use '-' for stdout.",
    )
    p.add_argument(
        "--top",
        type=int,
        default=20,
        help="Number of top verses to include (default: 20).",
    )
    p.add_argument(
        "--include-all",
        action="store_true",
        help="Include all judgements (default: only TRUE).",
    )
    p.add_argument(
        "--format",
        choices=["csv", "markdown"],
        default="csv",
        help="Output format (default: csv).",
    )
    args = p.parse_args()

    in_path = Path(args.input)
    header, rows, skipped = _read_rows(in_path)
    idx = {name: i for i, name in enumerate(header)}

    judgement_idx = idx.get("judgement")
    verse_range_idx = idx["verse_range"]
    verse_text_idx = idx["verse_text"]

    counts = Counter()
    text_counts: dict[str, Counter] = defaultdict(Counter)

    for row in rows:
        if judgement_idx is not None and not args.include_all:
            if row[judgement_idx].strip().upper() != "TRUE":
                continue

        verse_range = row[verse_range_idx].strip()
        verse_text = row[verse_text_idx].strip()
        if not verse_range:
            continue

        counts[verse_range] += 1
        if verse_text:
            text_counts[verse_range][verse_text] += 1

    top_items = counts.most_common(args.top)
    out_rows = []
    for verse_range, count in top_items:
        verse_text = ""
        if text_counts.get(verse_range):
            verse_text = text_counts[verse_range].most_common(1)[0][0]
        out_rows.append(
            {
                "verse_range": verse_range,
                "verse_text": verse_text,
                "quote_count": str(count),
            }
        )

    if args.format == "markdown":
        _write_markdown(out_rows)
    else:
        out_path = None if args.output == "-" else Path(args.output)
        if out_path is not None:
            out_path.parent.mkdir(parents=True, exist_ok=True)
        _write_csv(out_rows, out_path)

    if skipped:
        print(f"[warn] skipped {skipped} malformed rows", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
