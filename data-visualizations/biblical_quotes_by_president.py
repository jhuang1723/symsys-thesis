#!/usr/bin/env python3
"""Bar chart of biblical quote counts by president (FDR and after)."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


PRESIDENT_ORDER = [
    "Franklin D. Roosevelt",
    "Harry S Truman",
    "Dwight D. Eisenhower",
    "John F. Kennedy",
    "Lyndon B. Johnson",
    "Richard Nixon",
    "Gerald R. Ford",
    "Jimmy Carter",
    "Ronald Reagan",
    "George Bush",
    "William J. Clinton",
    "George W. Bush",
    "Barack Obama",
    "Donald J. Trump (1st Term)",
    "Joseph R. Biden, Jr.",
]


def _load_results(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, on_bad_lines="skip")
    except TypeError:
        return pd.read_csv(path, error_bad_lines=False)


def _normalize_judgement(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.upper()


def build_chart(df: pd.DataFrame, output_path: Path) -> None:
    sns.set_theme(
        style="whitegrid",
        context="paper",
        font="serif",
        font_scale=1.0,
    )
    plt.rcParams["font.serif"] = ["Times New Roman", "Times", "DejaVu Serif"]
    fig, ax = plt.subplots(figsize=(9.5, 4.4), dpi=300)

    sns.barplot(
        data=df,
        x="president",
        y="quote_count",
        hue="president",
        palette="deep",
        legend=False,
        ax=ax,
    )

    ax.set_xlabel("President")
    ax.set_ylabel("Number of biblical quotes")
    ax.set_title("Biblical Quotes by President")
    ax.tick_params(axis="x", rotation=35, labelsize=8)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Bar chart of biblical quote counts by president (FDR and after)."
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
        default=Path("results/figures/biblical_quotes_by_president.png"),
        help="Output path for the figure.",
    )
    args = parser.parse_args()

    df = _load_results(args.input)
    if "president" not in df.columns:
        raise KeyError(f"president column not found in {args.input}")

    if "judgement" in df.columns:
        df = df[_normalize_judgement(df["judgement"]) == "TRUE"]

    df = df[df["president"].isin(PRESIDENT_ORDER)]
    if df.empty:
        raise ValueError("No rows found for FDR and later presidents.")

    counts = (
        df["president"]
        .value_counts()
        .reindex(PRESIDENT_ORDER)
        .dropna()
        .rename_axis("president")
        .reset_index(name="quote_count")
    )

    build_chart(counts, args.output)


if __name__ == "__main__":
    main()
