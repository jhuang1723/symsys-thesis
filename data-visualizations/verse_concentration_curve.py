#!/usr/bin/env python3
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot a cumulative concentration curve for quoted Bible verses."
    )
    parser.add_argument(
        "--input",
        default="cleaned-results/full-results.csv",
        help="Path to full-results.csv",
    )
    parser.add_argument(
        "--output",
        default="data-visualizations/verse_concentration_curve.png",
        help="Output path for the figure",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path, engine="python", on_bad_lines="warn")
    if "verse_range" not in df.columns:
        raise ValueError("Expected column 'verse_range' in input CSV.")

    verse_counts = (
        df.dropna(subset=["verse_range"])
        .groupby("verse_range")
        .size()
        .sort_values(ascending=False)
        .rename("quote_count")
    )
    if verse_counts.empty:
        raise ValueError("No verse counts could be computed from the input.")

    cumulative_share = verse_counts.cumsum() / verse_counts.sum() * 100
    rank = range(1, len(cumulative_share) + 1)

    sns.set_theme(
        style="whitegrid",
        rc={"font.family": "serif", "font.serif": ["DejaVu Serif"]},
    )

    fig, ax = plt.subplots(figsize=(8, 3.25))
    ax.plot(rank, cumulative_share, color="#4C72B0", linewidth=2.8)
    ax.fill_between(rank, cumulative_share, color="#9FC5E8", alpha=0.35)

    markers = [10, 20, 50, 100]
    label_offsets = {
        10: (10, 0),
        20: (10, 0),
        50: (16, 0),
        100: (18, 0),
    }
    for k in markers:
        if k <= len(cumulative_share):
            y = float(cumulative_share.iloc[k - 1])
            ax.scatter(k, y, color="#C44E52", s=22, zorder=3)
            offset_x, offset_y = label_offsets.get(k, (10, 0))
            ax.annotate(
                f"Top {k}: {y:.1f}%",
                xy=(k, y),
                xytext=(offset_x, offset_y),
                textcoords="offset points",
                va="center",
                fontsize=10,
                color="#333333",
            )

    ax.set_title("Cumulative concentration of Bible verse quotations")
    ax.set_xlabel("Top k verses")
    ax.set_ylabel("Cumulative share of all quotations")
    ax.set_xlim(1, len(cumulative_share))
    ax.set_ylim(0, 100)
    sns.despine(ax=ax)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    main()
