import argparse
import os
from pathlib import Path

import pandas as pd

# Avoid matplotlib cache issues in locked home directories.
os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path("/Users/jaynahuang/projects/thesis/tmp/matplotlib")),
)

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import PowerNorm

LABELS = {
    1: "Peace & Reconciliation",
    2: "Love & Moral Obligation",
    3: "Hope & Promise",
    5: "Justice & Righteousness",
    6: "Divine Authority & Sovereignty",
    7: "Joy & Praise",
    8: "Other",
}
LABEL_ORDER = [1, 2, 3, 5, 6, 7, 8]

def load_data(context_path: Path, theme_path: Path) -> pd.DataFrame:
    ctx = pd.read_csv(context_path)
    themes = pd.read_csv(theme_path, usecols=["verse_range", "primary_label"])
    merged = ctx.merge(themes, on="verse_range", how="inner")
    merged["theme_name"] = merged["primary_label"].map(LABELS).fillna("Other")
    return merged


def crosstab_matrix(df: pd.DataFrame) -> pd.DataFrame:
    theme_dtype = pd.CategoricalDtype(
        categories=[LABELS[i] for i in LABEL_ORDER],
        ordered=True,
    )
    df = df.copy()
    df["theme_name"] = df["theme_name"].astype(theme_dtype)
    counts = pd.crosstab(df["context_cluster"], df["theme_name"])
    order = counts.sum(axis=1).sort_values(ascending=False).index
    return counts.reindex(index=order)


def plot_heatmap(
    matrix: pd.DataFrame,
    title: str,
    out_path: Path,
) -> None:
    width = max(10, 0.7 * len(matrix.columns) + 6)
    height = max(6, 0.55 * len(matrix.index) + 4)
    sns.set_theme(style="white", font="serif", font_scale=1.0)
    plt.rcParams["font.serif"] = ["Times New Roman", "Times", "DejaVu Serif"]
    fig, ax = plt.subplots(figsize=(width, height))

    sns.heatmap(
        matrix,
        annot=True,
        fmt=".0%",
        cmap="Blues",
        vmin=0,
        vmax=1,
        norm=PowerNorm(gamma=0.7),
        linewidths=0.5,
        linecolor="#f2f2f2",
        cbar_kws={"label": "Share of verses"},
        ax=ax,
    )

    ax.set_title(title, pad=16, fontsize=14, fontweight="semibold")
    ax.set_xlabel("Verse themes", labelpad=10)
    ax.set_ylabel("Speech context", labelpad=10)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=25, ha="right")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--context-csv",
        default="/Users/jaynahuang/projects/thesis/context-clustering/speech_context_clusters.csv",
    )
    parser.add_argument(
        "--theme-csv",
        default="/Users/jaynahuang/projects/thesis/data-visualizations/themes/verse_labels_auto_unique_all.csv",
    )
    parser.add_argument(
        "--out-dir",
        default="/Users/jaynahuang/projects/thesis/data-visualizations/themes",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_data(Path(args.context_csv), Path(args.theme_csv))
    counts = crosstab_matrix(df)
    row_pct = counts.div(counts.sum(axis=1), axis=0)

    row_pct_out = out_dir / "context_theme_heatmap_row_pct.png"
    plot_heatmap(
        row_pct,
        "Verse themes by speech context",
        row_pct_out,
    )
    row_pct.to_csv(out_dir / "context_theme_row_pct.csv")

    print(f"Wrote: {row_pct_out}")


if __name__ == "__main__":
    main()
