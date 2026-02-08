import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE


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

    tfidf = TfidfVectorizer(stop_words="english", min_df=2)
    X = tfidf.fit_transform(texts)

    n_components = 50 if X.shape[1] > 50 else max(2, X.shape[1] - 1)
    svd = TruncatedSVD(n_components=n_components, random_state=0)
    X_reduced = svd.fit_transform(X)

    n_clusters = 8
    kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init=10)
    try:
        kmeans.fit(X_reduced, sample_weight=counts)
    except TypeError:
        kmeans.fit(X_reduced)
    cluster_id = kmeans.labels_

    perplexity = min(30, max(5, len(df) // 3))
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=0, init="pca")
    coords = tsne.fit_transform(X_reduced)
    df["x"] = coords[:, 0]
    df["y"] = coords[:, 1]
    df["cluster"] = cluster_id

    sizes = 10 + 20 * np.log1p(counts.to_numpy())

    plt.figure(figsize=(10, 8))
    for label_id in sorted(LABELS):
        subset = df[df["primary_label"] == label_id]
        if subset.empty:
            continue
        plt.scatter(
            subset["x"],
            subset["y"],
            s=sizes[subset.index],
            alpha=0.7,
            label=f"{label_id}. {LABELS[label_id]}",
        )

    # Annotate clusters by dominant primary label (skip "Other").
    for c in range(n_clusters):
        cluster_rows = df[df["cluster"] == c]
        if cluster_rows.empty:
            continue
        weights = cluster_rows["occurrence_count"].fillna(1).astype(float)
        x_center = np.average(cluster_rows["x"], weights=weights)
        y_center = np.average(cluster_rows["y"], weights=weights)
        label_counts = (
            cluster_rows.groupby("primary_label")["occurrence_count"]
            .sum()
            .sort_values(ascending=False)
        )
        dominant_label = int(label_counts.index[0])
        if dominant_label == 8:
            continue
        plt.text(
            x_center,
            y_center,
            LABELS[dominant_label],
            fontsize=9,
            weight="bold",
            ha="center",
            va="center",
            bbox={"boxstyle": "round,pad=0.2", "fc": "white", "alpha": 0.7},
        )

    plt.title("Verse themes: TF-IDF + SVD + t-SNE (size = frequency)")
    plt.legend(markerscale=1.5, bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()

    out_path = Path(__file__).with_name("verse_label_clusters.png")
    plt.savefig(out_path, dpi=200)
    plt.show()


if __name__ == "__main__":
    main()
