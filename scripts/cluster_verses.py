#!/usr/bin/env python3
import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cluster verse-centric text (verse + snippet) and plot 2D projection."
    )
    parser.add_argument(
        "--input",
        default="cleaned-results/full-results.csv",
        help="CSV with verse matches.",
    )
    parser.add_argument(
        "--output-dir",
        default="results/verse_clusters",
        help="Directory to write clustered CSV and plot.",
    )
    parser.add_argument(
        "--text-fields",
        default="verse_text,snippet_norm",
        help="Comma-separated fields to combine for embeddings.",
    )
    parser.add_argument(
        "--embedding",
        default="sbert",
        choices=["sbert", "tfidf"],
        help="Embedding type: sentence-transformers (sbert) or TF-IDF.",
    )
    parser.add_argument(
        "--model",
        default="all-MiniLM-L6-v2",
        help="Sentence-Transformers model name (if embedding=sbert).",
    )
    parser.add_argument(
        "--cluster",
        default="kmeans",
        choices=["kmeans", "agglomerative", "dbscan", "hdbscan"],
        help="Clustering algorithm.",
    )
    parser.add_argument(
        "--cluster-on",
        default="embeddings",
        choices=["embeddings", "reduced"],
        help="Cluster on original embeddings or 2D reduced coordinates.",
    )
    parser.add_argument(
        "--n-clusters",
        type=int,
        default=12,
        help="Number of clusters (kmeans/agglomerative).",
    )
    parser.add_argument(
        "--dbscan-eps",
        type=float,
        default=0.7,
        help="DBSCAN eps parameter.",
    )
    parser.add_argument(
        "--dbscan-min-samples",
        type=int,
        default=10,
        help="DBSCAN min_samples parameter.",
    )
    parser.add_argument(
        "--hdbscan-min-cluster-size",
        type=int,
        default=15,
        help="HDBSCAN min_cluster_size parameter.",
    )
    parser.add_argument(
        "--hdbscan-min-samples",
        type=int,
        default=None,
        help="HDBSCAN min_samples parameter (default: None).",
    )
    parser.add_argument(
        "--dim-reducer",
        default="pca",
        choices=["pca", "tsne", "umap"],
        help="2D projection method.",
    )
    parser.add_argument(
        "--umap-n-neighbors",
        type=int,
        default=15,
        help="UMAP n_neighbors parameter (if dim-reducer=umap).",
    )
    parser.add_argument(
        "--umap-min-dist",
        type=float,
        default=0.1,
        help="UMAP min_dist parameter (if dim-reducer=umap).",
    )
    parser.add_argument(
        "--max-plot-points",
        type=int,
        default=3000,
        help="Maximum points to plot (subsample if larger).",
    )
    parser.add_argument(
        "--filter-judgement-true",
        action="store_true",
        help="Keep only rows where judgement == TRUE.",
    )
    return parser.parse_args()


def load_data(path: str) -> pd.DataFrame:
    # The CSV has occasional malformed rows; skip them to proceed.
    return pd.read_csv(path, engine="python", on_bad_lines="skip")


def build_text(df: pd.DataFrame, fields: list[str]) -> list[str]:
    texts = []
    for _, row in df.iterrows():
        parts = []
        for field in fields:
            value = row.get(field, "")
            if pd.isna(value):
                value = ""
            value = str(value).strip()
            if value:
                parts.append(value)
        texts.append(" [SEP] ".join(parts))
    return texts


def embed_texts(texts: list[str], embedding: str, model_name: str) -> np.ndarray:
    if embedding == "sbert":
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise SystemExit(
                "sentence-transformers is required for --embedding sbert. "
                "Install it or use --embedding tfidf."
            ) from exc
        model = SentenceTransformer(model_name)
        return np.asarray(model.encode(texts, show_progress_bar=True))

    from sklearn.feature_extraction.text import TfidfVectorizer

    vectorizer = TfidfVectorizer(
        max_features=20000,
        ngram_range=(1, 2),
        stop_words="english",
    )
    return vectorizer.fit_transform(texts).toarray()


def cluster_embeddings(
    embeddings: np.ndarray,
    method: str,
    n_clusters: int,
    dbscan_eps: float,
    dbscan_min_samples: int,
    hdbscan_min_cluster_size: int,
    hdbscan_min_samples: int | None,
) -> np.ndarray:
    if method == "kmeans":
        from sklearn.cluster import KMeans

        model = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
        return model.fit_predict(embeddings)

    if method == "agglomerative":
        from sklearn.cluster import AgglomerativeClustering

        model = AgglomerativeClustering(n_clusters=n_clusters)
        return model.fit_predict(embeddings)

    if method == "hdbscan":
        try:
            import hdbscan
        except ImportError as exc:
            raise SystemExit(
                "hdbscan is required for --cluster hdbscan. "
                "Install it or use --cluster kmeans/agglomerative/dbscan."
            ) from exc
        model = hdbscan.HDBSCAN(
            min_cluster_size=hdbscan_min_cluster_size,
            min_samples=hdbscan_min_samples,
        )
        return model.fit_predict(embeddings)

    from sklearn.cluster import DBSCAN

    model = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples)
    return model.fit_predict(embeddings)


def reduce_to_2d(
    embeddings: np.ndarray,
    method: str,
    umap_n_neighbors: int,
    umap_min_dist: float,
) -> np.ndarray:
    if method == "tsne":
        from sklearn.manifold import TSNE

        n_samples = embeddings.shape[0]
        perplexity = min(30, max(5, (n_samples - 1) // 3))
        tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
        return tsne.fit_transform(embeddings)

    if method == "umap":
        try:
            import umap
        except ImportError as exc:
            raise SystemExit(
                "umap-learn is required for --dim-reducer umap. "
                "Install it or use --dim-reducer pca/tsne."
            ) from exc
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=umap_n_neighbors,
            min_dist=umap_min_dist,
            random_state=42,
        )
        return reducer.fit_transform(embeddings)

    from sklearn.decomposition import PCA

    return PCA(n_components=2, random_state=42).fit_transform(embeddings)


def plot_clusters(df: pd.DataFrame, out_path: Path, max_points: int) -> None:
    import matplotlib.pyplot as plt

    plot_df = df
    if len(df) > max_points:
        plot_df = df.sample(n=max_points, random_state=42)

    fig, ax = plt.subplots(figsize=(10, 7))
    scatter = ax.scatter(
        plot_df["x"],
        plot_df["y"],
        c=plot_df["cluster"],
        cmap="tab20",
        s=12,
        alpha=0.7,
    )
    ax.set_title("Verse-centric semantic clusters (2D projection)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True, linestyle="--", alpha=0.2)
    if plot_df["cluster"].nunique() <= 20:
        ax.legend(*scatter.legend_elements(), title="cluster", loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_data(str(input_path))
    if args.filter_judgement_true and "judgement" in df.columns:
        df = df[df["judgement"].astype(str).str.upper() == "TRUE"].copy()

    fields = [f.strip() for f in args.text_fields.split(",") if f.strip()]
    texts = build_text(df, fields)
    embeddings = embed_texts(texts, args.embedding, args.model)
    coords = reduce_to_2d(
        embeddings,
        args.dim_reducer,
        args.umap_n_neighbors,
        args.umap_min_dist,
    )
    cluster_input = coords if args.cluster_on == "reduced" else embeddings
    clusters = cluster_embeddings(
        cluster_input,
        args.cluster,
        args.n_clusters,
        args.dbscan_eps,
        args.dbscan_min_samples,
        args.hdbscan_min_cluster_size,
        args.hdbscan_min_samples,
    )

    df = df.copy()
    df["cluster"] = clusters
    df["x"] = coords[:, 0]
    df["y"] = coords[:, 1]

    out_csv = output_dir / "verse_clusters.csv"
    out_plot = output_dir / "verse_clusters.png"
    df.to_csv(out_csv, index=False)
    plot_clusters(df, out_plot, args.max_plot_points)

    print(f"Wrote {out_csv}")
    print(f"Wrote {out_plot}")


if __name__ == "__main__":
    main()
