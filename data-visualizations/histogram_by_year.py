#!/usr/bin/env python3
"""Create a year-bucketed histogram for APP categories."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


CATEGORIES = {
    "spoken-addresses-and-remarks": "Spoken Addresses and Remarks",
    "proclamations": "Proclamations",
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


def build_histogram(df: pd.DataFrame, bins: int, output_path: Path) -> None:
    min_year = int(df["year"].min())
    max_year = int(df["year"].max())

    sns.set_theme(
        style="whitegrid",
        context="paper",
        font="serif",
        font_scale=1.05,
    )
    plt.rcParams["font.serif"] = ["Times New Roman", "Times", "DejaVu Serif"]
    fig, ax = plt.subplots(figsize=(8.0, 2.8), dpi=300)

    palette = {
        "Spoken Addresses and Remarks": "#1f77b4",
        "Proclamations": "#2ca02c",
    }

    sns.histplot(
        data=df,
        x="year",
        hue="category",
        bins=bins,
        stat="count",
        multiple="stack",
        alpha=1.0,
        edgecolor="white",
        palette=palette,
        ax=ax,
    )

    ax.set_xlabel("Year")
    ax.set_ylabel("Number of documents")
    ax.set_title("Documents by Year: Spoken Addresses and Remarks vs Proclamations")
    ax.set_xlim(min_year - 1, max_year + 1)
    legend_handles = [
        Patch(facecolor=palette["Spoken Addresses and Remarks"], label="Spoken Addresses and Remarks"),
        Patch(facecolor=palette["Proclamations"], label="Proclamations"),
    ]
    ax.legend(handles=legend_handles, title=None, frameon=True)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Histogram by year for APP categories (raw or cleaned)."
    )
    parser.add_argument(
        "--source",
        choices=["raw", "cleaned"],
        default="raw",
        help="Use raw data in data/app or cleaned_data/app.",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=9,
        help="Number of bins for year histogram (approx 8-10 recommended).",
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
        default=Path("results/figures/app_histogram_by_year.png"),
        help="Output path for the figure.",
    )

    args = parser.parse_args()

    frames = []
    for category, label in CATEGORIES.items():
        path = _resolve_rows_path(args.source, category, args.format)
        df = _load_rows(path)
        if "date_iso" not in df.columns:
            raise KeyError(f"date_iso not found in {path}")

        df = df.assign(
            category=label,
            year=pd.to_datetime(df["date_iso"], errors="coerce").dt.year,
        )
        frames.append(df[["category", "year"]])

    all_df = pd.concat(frames, ignore_index=True).dropna(subset=["year"])
    all_df["year"] = all_df["year"].astype(int)

    build_histogram(all_df, args.bins, args.output)


if __name__ == "__main__":
    main()
