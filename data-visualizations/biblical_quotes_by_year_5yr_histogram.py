#!/usr/bin/env python3
"""Histogram of biblical quote counts in 5-year buckets."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def _load_results(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, on_bad_lines="skip")
    except TypeError:
        return pd.read_csv(path, error_bad_lines=False)


def build_chart(df: pd.DataFrame, output_path: Path) -> None:
    sns.set_theme(
        style="whitegrid",
        context="paper",
        font="serif",
        font_scale=1.05,
    )
    plt.rcParams["font.serif"] = ["Times New Roman", "Times", "DejaVu Serif"]
    fig, ax = plt.subplots(figsize=(8.8, 3.2), dpi=300)

    sns.histplot(
        data=df,
        x="year",
        binwidth=5,
        color="#2ca02c",
        edgecolor="white",
        ax=ax,
    )

    ax.set_xlabel("Year")
    ax.set_ylabel("Number of biblical quotes")
    ax.set_title("Biblical Quotes by Year (5-Year Buckets)")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Histogram of biblical quote counts in 5-year buckets."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("cleaned-results/full-results.csv"),
        help="Path to full-results CSV.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/figures/biblical_quotes_by_year_5yr_histogram.png"),
        help="Output path for the figure.",
    )
    args = parser.parse_args()

    df = _load_results(args.input)
    if "date" not in df.columns:
        raise KeyError(f"date column not found in {args.input}")

    df = df.assign(
        year=pd.to_datetime(df["date"], errors="coerce").dt.year
    ).dropna(subset=["year"])
    df["year"] = df["year"].astype(int)

    if df.empty:
        raise ValueError("No valid dates found to plot.")

    build_chart(df, args.output)


if __name__ == "__main__":
    main()
