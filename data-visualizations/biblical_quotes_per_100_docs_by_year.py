#!/usr/bin/env python3
"""Line chart of biblical verse counts per 100 documents by year."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


CATEGORIES = (
    "spoken-addresses-and-remarks",
    "proclamations",
)


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


def _load_results(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, on_bad_lines="skip")
    except TypeError:
        return pd.read_csv(path, error_bad_lines=False)


def build_chart(verses_df: pd.DataFrame, docs_df: pd.DataFrame, output_path: Path) -> None:
    sns.set_theme(
        style="whitegrid",
        context="paper",
        font="serif",
        font_scale=1.05,
    )
    plt.rcParams["font.serif"] = ["Times New Roman", "Times", "DejaVu Serif"]
    fig, ax = plt.subplots(figsize=(8.8, 3.2), dpi=300)

    verses_by_year = (
        verses_df.groupby("year")
        .size()
        .rename("verse_count")
        .reset_index()
    )
    docs_by_year = docs_df.groupby("year").size().rename("doc_count").reset_index()

    rates = verses_by_year.merge(docs_by_year, on="year", how="inner")
    rates = rates[rates["doc_count"] > 0].copy()
    rates["verses_per_100_docs"] = (
        rates["verse_count"] / rates["doc_count"] * 100.0
    )
    rates = rates.sort_values("year")

    def _local_maxima(df: pd.DataFrame) -> pd.DataFrame:
        """Return local maxima rows (strictly higher than neighbors)."""
        df = df.sort_values("year").reset_index(drop=True)
        if len(df) < 3:
            return df.iloc[0:0]
        prev_vals = df["verses_per_100_docs"].shift(1)
        next_vals = df["verses_per_100_docs"].shift(-1)
        mask = (df["verses_per_100_docs"] > prev_vals) & (
            df["verses_per_100_docs"] > next_vals
        )
        return df[mask].copy()

    def _select_spikes(
        peaks_df: pd.DataFrame,
        target_n: int = 7,
        min_gap_years: int = 6,
    ) -> pd.DataFrame:
        if peaks_df.empty:
            return peaks_df
        peaks_sorted = peaks_df.sort_values(
            "verses_per_100_docs", ascending=False
        ).reset_index(drop=True)
        selected = []
        gap = min_gap_years
        while len(selected) < target_n and gap >= 1:
            selected = []
            for _, row in peaks_sorted.iterrows():
                if all(abs(int(row["year"]) - int(s["year"])) >= gap for s in selected):
                    selected.append(row)
                    if len(selected) >= target_n:
                        break
            if len(selected) < target_n:
                gap -= 1
        return pd.DataFrame(selected)

    sns.lineplot(
        data=rates,
        x="year",
        y="verses_per_100_docs",
        color="#1f77b4",
        marker="o",
        linewidth=1.8,
        markersize=3.5,
        ax=ax,
    )

    ax.set_xlabel("Year")
    ax.set_ylabel("Biblical verses per 100 documents")
    ax.set_title("Biblical Verses per 100 Documents by Year")

    # Annotate local maxima spikes (6-8 labels, spread out).
    peaks = _local_maxima(rates)
    spikes = _select_spikes(peaks, target_n=7, min_gap_years=6)
    if spikes.empty:
        spikes = rates.sort_values(
            "verses_per_100_docs", ascending=False
        ).head(7)
    for _, row in spikes.iterrows():
        ax.annotate(
            f"{int(row['year'])}",
            (row["year"], row["verses_per_100_docs"]),
            textcoords="offset points",
            xytext=(0, 6),
            ha="center",
            fontsize=7.5,
            color="#1f77b4",
        )

    if not rates.empty:
        max_verses = verses_by_year.loc[verses_by_year["verse_count"].idxmax()]
        min_verses = verses_by_year.loc[verses_by_year["verse_count"].idxmin()]
        max_docs = docs_by_year.loc[docs_by_year["doc_count"].idxmax()]
        min_docs = docs_by_year.loc[docs_by_year["doc_count"].idxmin()]
        print(
            "Verses per year max:",
            f"{int(max_verses['year'])} -> {int(max_verses['verse_count'])}",
        )
        print(
            "Verses per year min:",
            f"{int(min_verses['year'])} -> {int(min_verses['verse_count'])}",
        )
        print(
            "Documents per year max:",
            f"{int(max_docs['year'])} -> {int(max_docs['doc_count'])}",
        )
        print(
            "Documents per year min:",
            f"{int(min_docs['year'])} -> {int(min_docs['doc_count'])}",
        )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Line chart of biblical verse counts per 100 documents by year."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("cleaned-results/full-results.csv"),
        help="Path to full-results CSV.",
    )
    parser.add_argument(
        "--source",
        choices=["raw", "cleaned"],
        default="raw",
        help="Use raw data in data/app or cleaned_data/app for document counts.",
    )
    parser.add_argument(
        "--format",
        choices=["csv", "parquet"],
        default=None,
        help="Force a specific input format if both exist for documents.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "results/figures/biblical_quotes_per_100_docs_by_year.png"
        ),
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

    frames = []
    for category in CATEGORIES:
        path = _resolve_rows_path(args.source, category, args.format)
        docs = _load_rows(path)
        if "date_iso" not in docs.columns:
            raise KeyError(f"date_iso not found in {path}")
        docs = docs.assign(
            year=pd.to_datetime(docs["date_iso"], errors="coerce").dt.year
        )
        frames.append(docs[["year"]])

    docs_by_year = pd.concat(frames, ignore_index=True).dropna(subset=["year"])
    docs_by_year["year"] = docs_by_year["year"].astype(int)

    if docs_by_year.empty:
        raise ValueError("No valid document dates found to plot.")

    build_chart(df, docs_by_year, args.output)


if __name__ == "__main__":
    main()
