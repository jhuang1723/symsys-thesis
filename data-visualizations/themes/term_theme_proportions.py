#!/usr/bin/env python3
"""Stacked bar chart of theme proportions by presidential term."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


LABELS = {
    1: "Peace & Reconciliation",
    2: "Love & Moral Obligation",
    3: "Hope & Promise",
    5: "Justice & Righteousness",
    6: "Divine Authority & Sovereignty",
    7: "Joy & Praise",
    8: "Other",
}

LABEL_ORDER = [
    LABELS[1],
    LABELS[2],
    LABELS[3],
    LABELS[5],
    LABELS[6],
    LABELS[7],
]

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


def _load_csv(path: Path) -> pd.DataFrame:
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
    fig, ax = plt.subplots(figsize=(11, 5), dpi=300)

    palette = sns.color_palette("muted", n_colors=len(LABEL_ORDER))
    label_colors = dict(zip(LABEL_ORDER, palette))

    bottom = pd.Series([0.0] * len(df.index), index=df.index)
    for label in LABEL_ORDER:
        values = df[label]
        ax.bar(
            df.index,
            values,
            bottom=bottom,
            color=label_colors[label],
            label=label,
            width=0.7,
        )
        bottom = bottom + values

    ax.set_xlabel("Presidential term", fontsize=12, labelpad=34)
    ax.xaxis.set_label_coords(0.1, -0.36)
    ax.set_ylabel("Share of verse quotes", fontsize=12)
    ax.set_title("Theme Proportions by Presidential Term", fontsize=14)
    ax.tick_params(axis="x", rotation=30, labelsize=9)
    ax.tick_params(axis="y", labelsize=10)
    ax.set_ylim(0, 1.0)
    ax.yaxis.set_major_formatter(lambda x, pos: f"{x:.0%}")
    ax.legend(
        title="Theme",
        loc="upper center",
        bbox_to_anchor=(0.62, -0.22),
        frameon=False,
        fontsize=9,
        ncol=2,
    )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stacked bar chart of theme proportions by presidential term."
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("cleaned-results/full-results.csv"),
        help="Path to full-results CSV.",
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=Path("data-visualizations/themes/verse_labels_auto_unique_all.csv"),
        help="Path to verse labels CSV.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/figures/theme_proportions_by_term.png"),
        help="Output path for the figure.",
    )
    args = parser.parse_args()

    full_df = _load_csv(args.results)
    labels_df = _load_csv(args.labels)

    required_full = {"verse_range", "president"}
    missing_full = required_full - set(full_df.columns)
    if missing_full:
        raise KeyError(f"full-results missing columns: {sorted(missing_full)}")

    required_labels = {"verse_range", "primary_label"}
    missing_labels = required_labels - set(labels_df.columns)
    if missing_labels:
        raise KeyError(f"labels file missing columns: {sorted(missing_labels)}")

    if "judgement" in full_df.columns:
        full_df = full_df[_normalize_judgement(full_df["judgement"]) == "TRUE"]

    merged = full_df.merge(labels_df[["verse_range", "primary_label"]], on="verse_range", how="inner")
    if merged.empty:
        raise ValueError("No rows matched between full-results and labels.")

    merged = merged[merged["president"].isin(PRESIDENT_ORDER)]
    merged["primary_label"] = pd.to_numeric(merged["primary_label"], errors="coerce")
    merged = merged.dropna(subset=["primary_label", "president"])
    merged["primary_label"] = merged["primary_label"].astype(int)
    merged = merged[merged["primary_label"].isin(LABELS.keys())]
    merged["label_name"] = merged["primary_label"].map(LABELS)
    merged = merged[merged["label_name"].isin(LABEL_ORDER)]

    counts = (
        merged.groupby(["president", "label_name"], as_index=False)
        .size()
        .rename(columns={"size": "quote_count"})
    )

    pivot = (
        counts.pivot(index="president", columns="label_name", values="quote_count")
        .reindex(PRESIDENT_ORDER)
        .fillna(0)
    )
    proportions = pivot.div(pivot.sum(axis=1), axis=0).fillna(0)

    build_chart(proportions, args.output)


if __name__ == "__main__":
    main()
