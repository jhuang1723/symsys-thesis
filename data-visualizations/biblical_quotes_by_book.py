import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def build_counts(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df["verse_range"].notna()].copy()
    df["book"] = df["verse_range"].str.extract(r"^(.*?)(?=\s\d)", expand=False).str.strip()
    df = df[df["book"].notna()]

    counts = (
        df["book"]
        .value_counts()
        .head(10)
        .rename_axis("book")
        .reset_index(name="count")
    )
    return counts


def plot_counts(counts: pd.DataFrame, output_path: Path) -> None:
    sns.set_theme(style="whitegrid", font="serif", font_scale=1.0)
    plt.rcParams["font.serif"] = ["Times New Roman", "Times", "DejaVu Serif"]

    fig, ax = plt.subplots(figsize=(9, 6))
    cmap = plt.get_cmap("Blues_r")
    # Avoid the palest end of the palette so bars remain legible on white.
    palette = [cmap(0.15 + (0.65 * i / max(1, len(counts) - 1))) for i in range(len(counts))]

    sns.barplot(
        data=counts,
        y="book",
        x="count",
        hue="book",
        palette=palette,
        ax=ax,
        legend=False,
    )

    ax.set_title("Quoted Verses by Bible Book", pad=12, fontsize=14)
    ax.set_xlabel("Quoted Verses (count)")
    ax.set_ylabel("Bible Book")
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)

    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a bar chart of quoted verses by Bible book."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("cleaned-results/full-results.csv"),
        help="Path to full-results.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/figures/quoted_verses_by_book.png"),
        help="Path to save the chart PNG",
    )
    args = parser.parse_args()

    df = pd.read_csv(
        args.input,
        engine="python",
        on_bad_lines="skip",
    )
    counts = build_counts(df)
    plot_counts(counts, args.output)


if __name__ == "__main__":
    main()
