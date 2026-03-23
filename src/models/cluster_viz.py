"""Visualise clustering results: UMAP scatter, feature heatmaps, radar charts.

Generates static plots (matplotlib) and an interactive HTML scatter (plotly).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import structlog

from src.models.cluster_analysis import build_cluster_profiles
from src.models.clustering import ClusteringPipeline

logger = structlog.get_logger(__name__)

CLUSTER_COLORS = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]
NOISE_COLOR = "#cccccc"


def plot_umap_scatter(
    embedding: np.ndarray,
    labels: np.ndarray,
    wallet_addresses: pd.Series | None = None,
    title: str = "UMAP Embedding — HDBSCAN Clusters",
) -> plt.Figure:
    """2D UMAP scatter plot colored by cluster assignment."""
    fig, ax = plt.subplots(figsize=(12, 8))

    unique_labels = sorted(set(labels))
    for label in unique_labels:
        mask = labels == label
        color = NOISE_COLOR if label == -1 else CLUSTER_COLORS[label % len(CLUSTER_COLORS)]
        name = "Noise" if label == -1 else f"Cluster {label}"
        alpha = 0.3 if label == -1 else 0.6
        size = 8 if label == -1 else 15

        ax.scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            c=color,
            label=f"{name} ({mask.sum()})",
            alpha=alpha,
            s=size,
            edgecolors="none",
        )

    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title(title)
    ax.legend(loc="best", framealpha=0.9)
    fig.tight_layout()
    return fig


def plot_umap_interactive(
    embedding: np.ndarray,
    labels: np.ndarray,
    features_df: pd.DataFrame,
    wallet_addresses: pd.Series | None = None,
) -> go.Figure:
    """Interactive plotly scatter of the UMAP embedding with hover info."""
    df = pd.DataFrame(
        {
            "UMAP_1": embedding[:, 0],
            "UMAP_2": embedding[:, 1],
            "cluster": [f"Cluster {cl}" if cl != -1 else "Noise" for cl in labels],
        }
    )

    if wallet_addresses is not None:
        df["wallet"] = wallet_addresses.values

    # Add top features to hover
    feature_cols = features_df.select_dtypes(include=[np.number]).columns.tolist()
    for col in feature_cols:
        df[col] = features_df[col].values

    hover_cols = ["wallet"] if wallet_addresses is not None else []
    hover_cols += feature_cols[:6]  # top 6 features in hover

    fig = px.scatter(
        df,
        x="UMAP_1",
        y="UMAP_2",
        color="cluster",
        hover_data=hover_cols,
        title="UMAP Embedding — HDBSCAN Clusters (interactive)",
        opacity=0.6,
    )
    fig.update_traces(marker={"size": 5})
    fig.update_layout(width=1000, height=700)
    return fig


def plot_feature_heatmap(
    profiles_df: pd.DataFrame,
    title: str = "Cluster Feature Profiles (ratio to global mean)",
) -> plt.Figure:
    """Heatmap showing each cluster's feature ratios vs global mean."""
    # Pivot to clusters x features
    pivot = profiles_df.pivot(index="cluster", columns="feature", values="ratio_to_global")
    pivot = pivot.sort_index()

    # Rename index for readability
    pivot.index = ["Noise" if i == -1 else f"Cluster {i}" for i in pivot.index]

    fig, ax = plt.subplots(figsize=(14, max(4, len(pivot) * 1.2)))

    im = ax.imshow(pivot.values, cmap="RdYlGn", aspect="auto", vmin=0, vmax=3)

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=10)

    # Add text annotations
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            color = "white" if val > 2.0 or val < 0.5 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8, color=color)

    fig.colorbar(im, ax=ax, label="Ratio to global mean", shrink=0.8)
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_radar_charts(
    profiles_df: pd.DataFrame,
) -> plt.Figure:
    """Radar (spider) chart for each cluster showing feature profile."""
    clusters = sorted(profiles_df["cluster"].unique())
    n_clusters = len(clusters)
    cols = min(3, n_clusters)
    rows = (n_clusters + cols - 1) // cols

    feature_names = sorted(profiles_df["feature"].unique())
    n_features = len(feature_names)
    angles = np.linspace(0, 2 * np.pi, n_features, endpoint=False).tolist()
    angles += angles[:1]  # close the polygon

    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows), subplot_kw={"polar": True})
    if n_clusters == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for idx, cluster_id in enumerate(clusters):
        ax = axes[idx]
        cluster_data = profiles_df[profiles_df["cluster"] == cluster_id]
        label = "Noise" if cluster_id == -1 else f"Cluster {cluster_id}"
        size = int(cluster_data.iloc[0]["size"])
        color = NOISE_COLOR if cluster_id == -1 else CLUSTER_COLORS[cluster_id % len(CLUSTER_COLORS)]

        # Get ratios in feature_names order
        ratio_map = dict(zip(cluster_data["feature"], cluster_data["ratio_to_global"], strict=True))
        values = [min(ratio_map.get(f, 1.0), 5.0) for f in feature_names]  # cap at 5x for readability
        values += values[:1]

        ax.plot(angles, values, "o-", color=color, linewidth=2, markersize=4)
        ax.fill(angles, values, alpha=0.2, color=color)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([f.replace("_", "\n") for f in feature_names], fontsize=7)
        ax.set_ylim(0, 5)
        ax.set_title(f"{label} ({size} wallets)", fontsize=11, fontweight="bold", pad=15)

        # Add reference circle at 1.0 (global mean)
        ref_circle = [1.0] * (n_features + 1)
        ax.plot(angles, ref_circle, "--", color="gray", linewidth=1, alpha=0.5)

    # Hide unused subplots
    for idx in range(n_clusters, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle("Cluster Feature Profiles (ratio to global mean, capped at 5x)", fontsize=13, y=1.02)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    structlog.configure(
        wrapper_class=structlog.make_filtering_bound_logger(0),
    )

    parser = argparse.ArgumentParser(description="Generate cluster visualizations.")
    parser.add_argument(
        "--features",
        type=str,
        default="data/features.parquet",
    )
    parser.add_argument(
        "--pipeline",
        type=str,
        default="models/artifacts/clustering_pipeline.joblib",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models/artifacts/plots",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load
    features_df = pd.read_parquet(args.features)
    pipeline = ClusteringPipeline.load(args.pipeline)

    if pipeline.labels_ is None or pipeline.embedding_ is None:
        print("ERROR: Pipeline has no fitted data. Run clustering first.")
        raise SystemExit(1)

    wallet_addresses = features_df["wallet_address"] if "wallet_address" in features_df.columns else None
    numeric_features = features_df.select_dtypes(include=[np.number])

    print("[1/4] UMAP scatter plot...")
    fig_scatter = plot_umap_scatter(pipeline.embedding_, pipeline.labels_, wallet_addresses)
    fig_scatter.savefig(output_dir / "umap_clusters.png", dpi=150, bbox_inches="tight")
    print(f"      Saved {output_dir / 'umap_clusters.png'}")

    print("[2/4] Interactive scatter (HTML)...")
    fig_interactive = plot_umap_interactive(pipeline.embedding_, pipeline.labels_, features_df, wallet_addresses)
    fig_interactive.write_html(str(output_dir / "umap_clusters_interactive.html"))
    print(f"      Saved {output_dir / 'umap_clusters_interactive.html'}")

    print("[3/4] Feature heatmap...")
    profiles = build_cluster_profiles(numeric_features, pipeline.labels_)
    fig_heatmap = plot_feature_heatmap(profiles)
    fig_heatmap.savefig(output_dir / "feature_heatmap.png", dpi=150, bbox_inches="tight")
    print(f"      Saved {output_dir / 'feature_heatmap.png'}")

    print("[4/4] Radar charts...")
    fig_radar = plot_radar_charts(profiles)
    fig_radar.savefig(output_dir / "radar_charts.png", dpi=150, bbox_inches="tight")
    print(f"      Saved {output_dir / 'radar_charts.png'}")

    print(f"\nAll plots saved to {output_dir}/")
    print("Open umap_clusters_interactive.html in a browser for hover details.")
