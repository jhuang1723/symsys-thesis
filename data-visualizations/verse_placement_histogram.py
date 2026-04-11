#!/usr/bin/env python3
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot the distribution of verse placement within speeches."
    )
    parser.add_argument(
        "--input",
        default="cleaned-results/full-results-with-positions.csv",
        help="Path to full-results-with-positions.csv",
    )
    parser.add_argument(
        "--output",
        default="results/figures/verse_placement_histogram.png",
        help="Output path for the plot image",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)
    if "mid_pct_through_speech" not in df.columns:
        raise ValueError("Expected column 'mid_pct_through_speech' in input CSV.")

    placement = pd.to_numeric(df["mid_pct_through_speech"], errors="coerce").dropna()
    placement = placement[(placement >= 0) & (placement <= 1)]
    if placement.empty:
        raise ValueError("No valid placement values found in the input.")

    mean_value = placement.mean()
    median_value = placement.median()

    sns.set_theme(
        style="whitegrid",
        rc={"font.family": "serif", "font.serif": ["DejaVu Serif"]},
    )

    fig, ax = plt.subplots(figsize=(8, 3.5))
    bins = [i / 20 for i in range(21)]
    sns.histplot(
        placement,
        bins=bins,
        stat="percent",
        color="#9FC5E8",
        edgecolor="white",
        linewidth=0.8,
        alpha=0.95,
        ax=ax,
    )
    sns.kdeplot(
        placement,
        color="#5B6C7D",
        linewidth=2.2,
        bw_adjust=0.9,
        clip=(0, 1),
        ax=ax,
    )

    ax.axvline(
        mean_value,
        color="#C44E52",
        linestyle="--",
        linewidth=2,
        label=f"Mean = {mean_value:.2f}",
    )
    ax.axvline(
        median_value,
        color="#4C72B0",
        linestyle=":",
        linewidth=2.4,
        label=f"Median = {median_value:.2f}",
    )

    ax.set_title("Where Bible verse quotations appear within speeches")
    ax.set_xlabel("Midpoint of verse window as share of speech length")
    ax.set_ylabel("Percent of verse quotations")
    ax.set_xlim(0, 1)
    ax.set_xticks([i / 10 for i in range(11)])
    ax.legend(frameon=True, facecolor="white", edgecolor="#CCCCCC")
    sns.despine(ax=ax)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    main()
