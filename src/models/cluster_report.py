"""Generate a comprehensive clustering report from fitted pipeline artifacts.

Usage:
    python -m src.models.cluster_report
    python -m src.models.cluster_report --features data/features.parquet \
        --pipeline models/artifacts/clustering_pipeline.joblib
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import structlog
from sklearn.metrics import silhouette_samples, silhouette_score

from src.models.cluster_analysis import build_cluster_profiles
from src.models.clustering import ClusteringPipeline

logger = structlog.get_logger(__name__)


def print_section(title: str) -> None:
    """Print a formatted section header."""
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")


def report_wallet_count(features_df: pd.DataFrame) -> None:
    """Report how many wallets survived the feature pipeline."""
    print_section("WALLET COUNT")
    n_wallets = len(features_df)
    feature_cols = [c for c in features_df.columns if c != "wallet_address"]
    print(f"  Wallets through filters: {n_wallets:,}")
    print(f"  Features ({len(feature_cols)}): {', '.join(feature_cols)}")


def report_clusters(labels: np.ndarray) -> None:
    """Report cluster breakdown and noise ratio."""
    print_section("CLUSTER BREAKDOWN")
    unique, counts = np.unique(labels, return_counts=True)
    n_clusters = len(unique[unique != -1])
    n_noise = int((labels == -1).sum())
    noise_ratio = n_noise / len(labels)

    print(f"  Clusters found: {n_clusters}")
    print(f"  Noise points:   {n_noise:,} ({noise_ratio:.1%})")
    print()
    for label, count in zip(unique, counts, strict=True):
        name = "Noise" if label == -1 else f"Cluster {label}"
        pct = count / len(labels) * 100
        bar = "#" * int(pct / 2)
        print(f"  {name:>12s}: {count:>5,} wallets ({pct:5.1f}%) {bar}")


def report_silhouette(
    embedding: np.ndarray,
    labels: np.ndarray,
) -> None:
    """Report silhouette scores overall and per-cluster."""
    print_section("SILHOUETTE ANALYSIS")
    valid = labels != -1
    n_valid_clusters = len(set(labels[valid]))

    if valid.sum() < 2 or n_valid_clusters < 2:
        print("  Insufficient non-noise clusters for silhouette analysis.")
        return

    overall = silhouette_score(embedding[valid], labels[valid])
    print(f"  Overall silhouette score: {overall:.4f}")

    sil = silhouette_samples(embedding[valid], labels[valid])
    valid_labels = labels[valid]

    print()
    print(f"  {'Cluster':>12s}  {'Mean':>8s}  {'Median':>8s}  {'Min':>8s}  {'Max':>8s}")
    print(f"  {'-' * 12}  {'-' * 8}  {'-' * 8}  {'-' * 8}  {'-' * 8}")
    for c in sorted(set(valid_labels)):
        cs = sil[valid_labels == c]
        print(
            f"  {'Cluster ' + str(c):>12s}"
            f"  {np.mean(cs):8.4f}"
            f"  {np.median(cs):8.4f}"
            f"  {np.min(cs):8.4f}"
            f"  {np.max(cs):8.4f}"
        )

    n_negative = int((sil < 0).sum())
    print(f"\n  Possibly misassigned (silhouette < 0): {n_negative} ({n_negative / len(sil):.1%})")


def report_label_validation(
    features_df: pd.DataFrame,
    labels: np.ndarray,
    ground_truth_path: Path,
) -> None:
    """Cross-tab clusters vs heuristic labels."""
    print_section("LABEL VALIDATION (heuristic overlay)")

    if not ground_truth_path.exists():
        print("  No ground_truth.parquet found. Skipping.")
        print("  Generate with: python -m src.data.ground_truth")
        return

    gt = pd.read_parquet(ground_truth_path)
    print(f"  Ground truth wallets: {len(gt):,}")
    print("  Label distribution:")
    for lbl, cnt in gt["label"].value_counts().items():
        print(f"    {lbl}: {cnt:,}")

    if "wallet_address" not in features_df.columns:
        print("  Cannot merge: no wallet_address column in features.")
        return

    merged = features_df[["wallet_address"]].copy()
    merged["cluster"] = labels
    merged = merged.merge(
        gt[["address", "label"]],
        left_on="wallet_address",
        right_on="address",
        how="left",
    )
    merged["label"] = merged["label"].fillna("unlabeled")

    n_labeled = (merged["label"] != "unlabeled").sum()
    print(f"\n  Overlap: {n_labeled:,} of {len(merged):,} wallets have heuristic labels")

    ct = pd.crosstab(merged["cluster"], merged["label"])
    print("\n  Cross-tab (cluster vs heuristic label):")
    # Rename index for readability
    ct.index = ["Noise" if i == -1 else f"Cluster {i}" for i in ct.index]
    print(ct.to_string(col_space=10).replace("\n", "\n  "))


def report_top_features(
    features_df: pd.DataFrame,
    labels: np.ndarray,
) -> None:
    """Show the most distinctive features per cluster."""
    print_section("TOP DISTINCTIVE FEATURES PER CLUSTER")
    numeric = features_df.select_dtypes(include=[np.number])
    profiles = build_cluster_profiles(numeric, labels)

    for cluster_id in sorted(profiles["cluster"].unique()):
        cluster = profiles[profiles["cluster"] == cluster_id]
        name = "Noise" if cluster_id == -1 else f"Cluster {cluster_id}"
        size = int(cluster.iloc[0]["size"])

        # Sort by absolute z-score
        top = cluster.sort_values("z_score", key=abs, ascending=False).head(5)

        print(f"\n  {name} ({size:,} wallets) — top 5 distinguishing features:")
        for _, row in top.iterrows():
            ratio = row["ratio_to_global"]
            direction = "HIGH" if ratio > 1.3 else ("LOW" if ratio < 0.7 else "~avg")
            print(f"    [{direction:>4s}] {row['feature']:40s} {ratio:.2f}x global mean (z={row['z_score']:+.2f})")


def main() -> None:
    """Run the full clustering report."""
    parser = argparse.ArgumentParser(description="Generate clustering report.")
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
        "--ground-truth",
        type=str,
        default="data/ground_truth.parquet",
    )
    args = parser.parse_args()

    features_df = pd.read_parquet(args.features)
    pipeline = ClusteringPipeline.load(args.pipeline)

    if pipeline.labels_ is None or pipeline.embedding_ is None:
        print("ERROR: Pipeline has no fitted data. Run clustering first.")
        raise SystemExit(1)

    print("\n" + "=" * 70)
    print("  SMART MONEY WALLET CLUSTERING REPORT")
    print("=" * 70)

    report_wallet_count(features_df)
    report_clusters(pipeline.labels_)
    report_silhouette(pipeline.embedding_, pipeline.labels_)
    report_top_features(features_df, pipeline.labels_)
    report_label_validation(features_df, pipeline.labels_, Path(args.ground_truth))

    print(f"\n{'=' * 70}")
    print("  END OF REPORT")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()
