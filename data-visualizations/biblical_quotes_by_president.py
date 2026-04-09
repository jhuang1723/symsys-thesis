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
}
PARTY_COLORS = {
    "Democratic": "#1f77b4",
    "Republican": "#d62728",
}

def _load_results(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, on_bad_lines="skip")
    except TypeError:
        return pd.read_csv(path, error_bad_lines=False)


def _normalize_judgement(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.upper()


def build_chart(
    df: pd.DataFrame,
    output_path: Path,
    y_col: str,
    y_label: str,
    title: str,
) -> None:
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
        y=y_col,
        hue="party",
        palette=PARTY_COLORS,
        order=PRESIDENT_ORDER,
        dodge=False,
        ax=ax,
    )

    ax.set_xlabel("President", fontsize=13)
    ax.set_ylabel(y_label, fontsize=13)
    ax.set_title(title, fontsize=15)
    ax.tick_params(axis="x", rotation=35, labelsize=11)
    ax.tick_params(axis="y", labelsize=11)
    ax.legend(title="Party", loc="upper right", frameon=False)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _build_counts(df: pd.DataFrame) -> pd.DataFrame:
    counts = (
        df["president"]
        .value_counts()
        .reindex(PRESIDENT_ORDER)
        .dropna()
        .rename_axis("president")
        .reset_index(name="quote_count")
    )
    counts["party"] = counts["president"].map(PRESIDENT_PARTY)
    if counts["party"].isna().any():
        missing = counts[counts["party"].isna()]["president"].tolist()
        raise ValueError(f"Missing party mapping for presidents: {missing}")
    return counts


def _apply_normalization(counts: pd.DataFrame, totals_path: Path) -> pd.DataFrame:
    totals_df = pd.read_csv(totals_path)
    if "president" not in totals_df.columns or "spoken_documents" not in totals_df.columns:
        raise KeyError("totals table must include president and spoken_documents columns.")

    totals_df = totals_df[totals_df["president"].isin(PRESIDENT_ORDER)].copy()
    totals_df = totals_df.set_index("president").reindex(PRESIDENT_ORDER)
    totals_df["spoken_documents"] = totals_df["spoken_documents"].fillna(0).astype(int)

    merged = counts.set_index("president").join(totals_df[["spoken_documents"]])
    if merged["spoken_documents"].isna().any():
        missing = merged[merged["spoken_documents"].isna()].index.tolist()
        raise ValueError(f"Missing totals for presidents: {missing}")
    if (merged["spoken_documents"] <= 0).any():
        bad = merged[merged["spoken_documents"] <= 0].index.tolist()
        raise ValueError(f"Non-positive document totals for presidents: {bad}")

    merged["quotes_per_100_docs"] = (
        merged["quote_count"] / merged["spoken_documents"] * 100.0
    )
    return merged.reset_index()


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
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Normalize by spoken documents (quotes per 100 documents).",
    )
    parser.add_argument(
        "--totals",
        type=Path,
        default=Path("results/tables/president_doc_word_counts.csv"),
        help="Path to president document totals table.",
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

    counts = _build_counts(df)

    if args.normalize:
        plot_df = _apply_normalization(counts, args.totals)
        y_col = "quotes_per_100_docs"
        y_label = "Biblical quotes per 100 documents"
        title = "Biblical Quotes per 100 Documents by President"
    else:
        plot_df = counts
        y_col = "quote_count"
        y_label = "Number of biblical quotes"
        title = "Biblical Quotes by President"

    build_chart(plot_df, args.output, y_col, y_label, title)


if __name__ == "__main__":
    main()
