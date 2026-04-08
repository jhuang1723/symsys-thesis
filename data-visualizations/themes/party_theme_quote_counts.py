#!/usr/bin/env python3
"""Bar chart of biblical quote counts by theme and party."""
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
    LABELS[8],
]

PRESIDENT_PARTY = {
    "Franklin D. Roosevelt": "Democratic",
    "Harry S Truman": "Democratic",
    "Dwight D. Eisenhower": "Republican",
    "John F. Kennedy": "Democratic",
    "Lyndon B. Johnson": "Democratic",
    "Richard Nixon": "Republican",
    "Gerald R. Ford": "Republican",
    "Jimmy Carter": "Democratic",
    "Ronald Reagan": "Republican",
    "George Bush": "Republican",
    "William J. Clinton": "Democratic",
    "George W. Bush": "Republican",
    "Barack Obama": "Democratic",
    "Donald J. Trump (1st Term)": "Republican",
    "Joseph R. Biden, Jr.": "Democratic",
    "Adlai Stevenson": "Democratic",
    "George McGovern": "Democratic",
    "Michael S. Dukakis": "Democratic",
    "Hillary Clinton": "Democratic",
    "Herbert Hoover": "Republican",
}

PARTY_COLORS = {
    "Democratic": "#1f77b4",
    "Republican": "#d62728",
}


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
    fig, ax = plt.subplots(figsize=(10.5, 4.8), dpi=300)

    sns.barplot(
        data=df,
        x="label_name",
        y="quote_count",
        hue="party",
        palette=PARTY_COLORS,
        order=LABEL_ORDER,
        hue_order=["Democratic", "Republican"],
        ax=ax,
    )

    ax.set_xlabel("Thematic category", fontsize=12)
    ax.set_ylabel("Number of verse quotes", fontsize=12)
    ax.set_title("Biblical Verse Quotes by Theme and Party", fontsize=14)
    ax.tick_params(axis="x", rotation=25, labelsize=10)
    ax.tick_params(axis="y", labelsize=10)
    ax.legend(title="Party", loc="upper right", frameon=False)
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.6)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Bar chart of biblical quote counts by theme and party."
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
        default=Path("results/figures/quotes_by_theme_party.png"),
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

    merged["primary_label"] = pd.to_numeric(merged["primary_label"], errors="coerce")
    merged = merged.dropna(subset=["primary_label", "president"])
    merged["primary_label"] = merged["primary_label"].astype(int)
    merged = merged[merged["primary_label"].isin(LABELS.keys())]
    merged["label_name"] = merged["primary_label"].map(LABELS)
    merged["party"] = merged["president"].map(PRESIDENT_PARTY)
    if merged["party"].isna().any():
        missing = sorted(merged[merged["party"].isna()]["president"].unique().tolist())
        raise ValueError(f"Missing party mapping for presidents: {missing}")

    counts = (
        merged.groupby(["label_name", "party"], as_index=False)
        .size()
        .rename(columns={"size": "quote_count"})
    )

    build_chart(counts, args.output)


if __name__ == "__main__":
    main()
