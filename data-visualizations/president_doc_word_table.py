#!/usr/bin/env python3
"""Produce document and word-count table by president for two APP categories."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


CATEGORIES = {
    "spoken-addresses-and-remarks": "spoken",
    "proclamations": "proclamations",
}


def _resolve_rows_path(source: str, category: str, fmt: str | None) -> Path:
    if source == "cleaned":
        base = Path("cleaned_data/app")
        filename = "rows_norm"
    else:
        base = Path("data/app")
        filename = "rows"

    if fmt:
        candidate = base / category / f"{filename}.{fmt}"
        if candidate.exists():
            return candidate
        raise FileNotFoundError(f"Missing file: {candidate}")

    for ext in ("csv", "parquet"):
        candidate = base / category / f"{filename}.{ext}"
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        f"No rows file found for {category} in {base} with csv/parquet"
    )


def _load_rows(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported file format: {path}")


def _prepare_category(df: pd.DataFrame, short_label: str) -> pd.DataFrame:
    if "president" not in df.columns:
        raise KeyError("president column missing")
    if "date_iso" not in df.columns:
        raise KeyError("date_iso column missing")
    if "word_count" not in df.columns:
        raise KeyError("word_count column missing")

    out = df[["president", "date_iso", "word_count"]].copy()
    out["year"] = pd.to_datetime(out["date_iso"], errors="coerce").dt.year
    out["word_count"] = pd.to_numeric(out["word_count"], errors="coerce").fillna(0).astype(int)
    out["category"] = short_label
    return out.dropna(subset=["president", "year"])


def build_table(df: pd.DataFrame) -> pd.DataFrame:
    agg = (
        df.groupby(["president", "category"], dropna=False)
        .agg(documents=("category", "size"), words=("word_count", "sum"))
        .reset_index()
    )

    pivot_docs = (
        agg.pivot(index="president", columns="category", values="documents")
        .fillna(0)
        .reset_index()
    )
    pivot_words = (
        agg.pivot(index="president", columns="category", values="words")
        .fillna(0)
        .reset_index()
    )

    result = pivot_docs.merge(pivot_words, on="president", suffixes=("_documents", "_words"))
    result = result.rename(
        columns={
            "spoken_documents": "spoken_documents",
            "spoken_words": "spoken_words",
            "proclamations_documents": "proclamations_documents",
            "proclamations_words": "proclamations_words",
        }
    )
    for col in (
        "spoken_documents",
        "spoken_words",
        "proclamations_documents",
        "proclamations_words",
    ):
        if col not in result.columns:
            result[col] = 0
        result[col] = result[col].astype(int)

    result["total_documents"] = result["spoken_documents"] + result["proclamations_documents"]
    result["total_words"] = result["spoken_words"] + result["proclamations_words"]

    earliest_year = (
        df.groupby("president", as_index=False)["year"]
        .min()
        .rename(columns={"year": "first_year"})
    )
    result = result.merge(earliest_year, on="president", how="left")

    result = result.sort_values(["first_year", "president"], ascending=[True, True])
    result = result.drop(columns=["first_year"]).reset_index(drop=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Table of document and word counts by president."
    )
    parser.add_argument(
        "--source",
        choices=["raw", "cleaned"],
        default="raw",
        help="Use raw data in data/app or cleaned_data/app.",
    )
    parser.add_argument(
        "--format",
        choices=["csv", "parquet"],
        default=None,
        help="Force a specific input format if both exist.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/tables/president_doc_word_counts.csv"),
        help="Output CSV path.",
    )

    args = parser.parse_args()

    frames = []
    for category, short_label in CATEGORIES.items():
        path = _resolve_rows_path(args.source, category, args.format)
        df = _load_rows(path)
        frames.append(_prepare_category(df, short_label))

    all_df = pd.concat(frames, ignore_index=True)
    table = build_table(all_df)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
