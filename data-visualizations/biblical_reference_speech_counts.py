#!/usr/bin/env python3
"""Stacked bar chart of total speeches vs biblical-reference speeches (FDR+)."""
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


def _load_full_results(path: Path) -> pd.DataFrame:
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
    fig, ax = plt.subplots(figsize=(9.8, 4.6), dpi=300)

    colors = {
        "Without biblical reference": "#9ecae1",
        "With biblical reference": "#3182bd",
    }

    ax.bar(
        df["president"],
        df["without_reference"],
        label="Without biblical reference",
        color=colors["Without biblical reference"],
    )
    ax.bar(
        df["president"],
        df["with_reference"],
        bottom=df["without_reference"],
        label="With biblical reference",
        color=colors["With biblical reference"],
    )

    ax.set_xlabel("President")
    ax.set_ylabel("Number of speeches")
    ax.set_title("Speeches With Biblical References (FDR and after)")
    ax.tick_params(axis="x", rotation=35, labelsize=8)
    ax.legend(frameon=True)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stacked bar chart of total speeches vs biblical-reference speeches."
    )
    parser.add_argument(
        "--full-results",
        type=Path,
        default=Path("cleaned-results/full-results.csv"),
        help="Path to full-results CSV with biblical matches.",
    )
    parser.add_argument(
        "--totals",
        type=Path,
        default=Path("results/tables/president_doc_word_counts.csv"),
        help="Path to president document totals table.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/figures/biblical_reference_speeches_by_president.png"),
        help="Output path for the figure.",
    )
    args = parser.parse_args()

    full_df = _load_full_results(args.full_results)
    if "president" not in full_df.columns or "doc_id" not in full_df.columns:
        raise KeyError("full-results must include president and doc_id columns.")

    if "judgement" in full_df.columns:
        full_df = full_df[_normalize_judgement(full_df["judgement"]) == "TRUE"]

    ref_counts = (
        full_df[full_df["president"].isin(PRESIDENT_ORDER)]
        .dropna(subset=["doc_id"])
        .drop_duplicates(subset=["president", "doc_id"])
        .groupby("president")["doc_id"]
        .count()
    )

    totals_df = pd.read_csv(args.totals)
    if "president" not in totals_df.columns or "spoken_documents" not in totals_df.columns:
        raise KeyError("totals table must include president and spoken_documents columns.")

    totals_df = totals_df[totals_df["president"].isin(PRESIDENT_ORDER)].copy()
    totals_df = totals_df.set_index("president").reindex(PRESIDENT_ORDER)

    merged = totals_df.assign(
        with_reference=ref_counts.reindex(PRESIDENT_ORDER).fillna(0).astype(int)
    )
    merged["with_reference"] = merged["with_reference"].astype(int)
    merged["total_speeches"] = merged["spoken_documents"].fillna(0).astype(int)
    merged["without_reference"] = merged["total_speeches"] - merged["with_reference"]

    if (merged["without_reference"] < 0).any():
        bad = merged[merged["without_reference"] < 0][
            ["total_speeches", "with_reference"]
        ]
        raise ValueError(f"Reference counts exceed totals for:\n{bad}")

    plot_df = (
        merged.reset_index()[["president", "with_reference", "without_reference"]]
        .dropna(subset=["president"])
    )

    if plot_df.empty:
        raise ValueError("No rows found for FDR and later presidents.")

    build_chart(plot_df, args.output)


if __name__ == "__main__":
    main()
