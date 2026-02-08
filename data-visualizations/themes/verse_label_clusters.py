from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer


LABELS = {
    1: "Peace & Reconciliation",
    2: "Love & Moral Obligation",
    3: "Hope & Promise",
    4: "Sacrifice & Service",
    5: "Justice & Righteousness",
    6: "Divine Authority & Sovereignty",
    7: "Joy & Praise",
    8: "Other",
}


def main() -> None:
    csv_path = Path(__file__).with_name("verse_labels_auto_unique_all.csv")
    df = pd.read_csv(csv_path)

    texts = df["verse_text"].fillna("").astype(str)
    counts = df["occurrence_count"].fillna(1).astype(int).clip(lower=1)

    tfidf = TfidfVectorizer(
        stop_words="english",
        min_df=2,
        ngram_range=(1, 2),
        max_features=10000,
    )
    X = tfidf.fit_transform(texts)

    n_components = 50 if X.shape[1] > 50 else max(2, X.shape[1] - 1)
    svd = TruncatedSVD(n_components=n_components, random_state=0)
    X_reduced = svd.fit_transform(X)

    # Supervised layout that explicitly separates hand labels.
    # Exclude label 8 ("Other") from the LDA fit to sharpen separation.
    mask = df["primary_label"].fillna(8).astype(int) != 8
    y = df.loc[mask, "primary_label"].astype(int).to_numpy()
    X_fit = X_reduced[mask.to_numpy()]
    lda_components = min(2, len(np.unique(y)) - 1)
    if lda_components >= 1:
        lda = LinearDiscriminantAnalysis(n_components=lda_components)
        coords_fit = lda.fit_transform(X_fit, y)
        coords_sup = lda.transform(X_reduced)
        if lda_components == 1:
            # Pad with zeros to keep 2D plotting.
            coords_sup = np.column_stack([coords_sup, np.zeros_like(coords_sup)])
    else:
        coords_sup = X_reduced[:, :2]

    df["x_sup"] = coords_sup[:, 0]
    df["y_sup"] = coords_sup[:, 1]

    sizes = 10 + 20 * np.log1p(counts.to_numpy())

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    colors = {
        1: "#1f77b4",
        2: "#ff7f0e",
        3: "#2ca02c",
        4: "#b0b0b0",
        5: "#d62728",
        6: "#9467bd",
        7: "#8c564b",
        8: "#b0b0b0",
    }

    for label_id in sorted(LABELS):
        subset = df[df["primary_label"] == label_id]
        if subset.empty:
            continue
        alpha = 0.2 if label_id in {4, 8} else 0.8
        ax.scatter(
            subset["x_sup"],
            subset["y_sup"],
            s=sizes[subset.index],
            alpha=alpha,
            color=colors.get(label_id, "#333333"),
            label=LABELS[label_id],
        )

        # Label centroids (skip "Other" and "Sacrifice & Service").
        if label_id not in {4, 8}:
            weights = subset["occurrence_count"].fillna(1).astype(float)
            x_c_sup = np.average(subset["x_sup"], weights=weights)
            y_c_sup = np.average(subset["y_sup"], weights=weights)
            ax.text(
                x_c_sup,
                y_c_sup,
                LABELS[label_id],
                fontsize=8,
                weight="bold",
                ha="center",
                va="center",
                bbox={"boxstyle": "round,pad=0.2", "fc": "white", "alpha": 0.7},
            )

    ax.set_title("Supervised layout (LDA) by hand labels")
    ax.set_xlabel("LDA 1")
    ax.set_ylabel("LDA 2")
    handles, labels = ax.get_legend_handles_labels()
    filtered = [
        (h, l)
        for h, l in zip(handles, labels)
        if l not in {"Sacrifice & Service", "Other"}
    ]
    if filtered:
        handles, labels = zip(*filtered)
        ax.legend(handles, labels, markerscale=1.2, loc="upper left")
    fig.suptitle("Verse themes: size = frequency")
    fig.tight_layout()

    out_path = Path(__file__).with_name("verse_label_lda.png")
    fig.savefig(out_path, dpi=200)
    plt.show()


if __name__ == "__main__":
    main()
