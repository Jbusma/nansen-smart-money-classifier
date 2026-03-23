"""Analyse clustering results and generate human-readable cluster profiles.

Loads a fitted clustering pipeline and the feature matrix, then computes
per-cluster feature distributions to help a human label each cluster
with a behavioural archetype name.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import structlog

from src.models.clustering import ClusteringPipeline

logger = structlog.get_logger(__name__)


def build_cluster_profiles(
    features_df: pd.DataFrame,
    labels: np.ndarray,
) -> pd.DataFrame:
    """Compute per-cluster mean, std, and ratio-to-global for each feature.

    Parameters
    ----------
    features_df : pd.DataFrame
        Numeric feature matrix (no wallet_address column).
    labels : np.ndarray
        Cluster labels from HDBSCAN (including -1 for noise).

    Returns
    -------
    pd.DataFrame
        Multi-indexed by (cluster, feature) with columns:
        mean, std, global_mean, ratio_to_global, size, pct.
    """
    df = features_df.copy()
    df["cluster"] = labels
    feature_cols = [c for c in df.columns if c != "cluster"]

    global_means = df[feature_cols].mean()
    global_stds = df[feature_cols].std()

    records = []
    for cluster_id in sorted(df["cluster"].unique()):
        cluster_data = df[df["cluster"] == cluster_id]
        size = len(cluster_data)
        pct = size / len(df) * 100

        for col in feature_cols:
            val = cluster_data[col].mean()
            std = cluster_data[col].std()
            glob = global_means[col]
            ratio = val / glob if glob != 0 else 0.0
            z_score = (val - glob) / global_stds[col] if global_stds[col] != 0 else 0.0
            records.append(
                {
                    "cluster": cluster_id,
                    "feature": col,
                    "mean": val,
                    "std": std,
                    "global_mean": glob,
                    "ratio_to_global": ratio,
                    "z_score": z_score,
                    "size": size,
                    "pct": pct,
                }
            )

    return pd.DataFrame(records)


def print_cluster_profiles(profiles_df: pd.DataFrame) -> None:
    """Print a human-readable summary of each cluster's feature profile."""
    for cluster_id in sorted(profiles_df["cluster"].unique()):
        cluster = profiles_df[profiles_df["cluster"] == cluster_id]
        size = int(cluster.iloc[0]["size"])
        pct = cluster.iloc[0]["pct"]
        label = "NOISE" if cluster_id == -1 else f"Cluster {cluster_id}"

        print(f"\n{'=' * 70}")
        print(f"  {label}  —  {size} wallets ({pct:.1f}%)")
        print(f"{'=' * 70}")

        # Sort features by absolute z-score (most distinctive first)
        cluster_sorted = cluster.sort_values("z_score", key=abs, ascending=False)

        for _, row in cluster_sorted.iterrows():
            ratio = row["ratio_to_global"]
            if ratio > 2.0:
                indicator = "HIGH "
            elif ratio > 1.3:
                indicator = "high "
            elif ratio < 0.3:
                indicator = " LOW "
            elif ratio < 0.7:
                indicator = " low "
            else:
                indicator = "     "

            print(
                f"  [{indicator}] {row['feature']:40s}"
                f"  mean={row['mean']:10.4f}"
                f"  (global={row['global_mean']:.4f},"
                f"  {ratio:.2f}x)"
            )


def save_profiles(profiles_df: pd.DataFrame, path: str | Path) -> None:
    """Save cluster profiles to a JSON file for later reference."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    output: dict = {}
    for cluster_id in sorted(profiles_df["cluster"].unique()):
        cluster = profiles_df[profiles_df["cluster"] == cluster_id]
        size = int(cluster.iloc[0]["size"])
        pct = float(cluster.iloc[0]["pct"])

        features: dict = {}
        for _, row in cluster.iterrows():
            features[row["feature"]] = {
                "mean": round(float(row["mean"]), 6),
                "global_mean": round(float(row["global_mean"]), 6),
                "ratio": round(float(row["ratio_to_global"]), 3),
                "z_score": round(float(row["z_score"]), 3),
            }

        key = "noise" if cluster_id == -1 else f"cluster_{cluster_id}"
        output[key] = {
            "size": size,
            "pct": round(pct, 2),
            "features": features,
            "suggested_label": None,  # to be filled by human review
        }

    with open(path, "w") as f:
        json.dump(output, f, indent=2)
    logger.info("cluster_profiles.saved", path=str(path))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    structlog.configure(
        wrapper_class=structlog.make_filtering_bound_logger(0),
    )

    parser = argparse.ArgumentParser(
        description="Analyse clustering results and print feature profiles per cluster.",
    )
    parser.add_argument(
        "--features",
        type=str,
        default="data/features.parquet",
        help="Path to features parquet.",
    )
    parser.add_argument(
        "--pipeline",
        type=str,
        default="models/artifacts/clustering_pipeline.joblib",
        help="Path to fitted clustering pipeline.",
    )
    parser.add_argument(
        "--save-profiles",
        type=str,
        default="models/artifacts/cluster_profiles.json",
        help="Path to save JSON profiles for human review.",
    )
    args = parser.parse_args()

    # Load data
    features_df = pd.read_parquet(args.features)
    pipeline = ClusteringPipeline.load(args.pipeline)

    if pipeline.labels_ is None:
        print("ERROR: Pipeline has no labels. Run clustering first.")
        raise SystemExit(1)

    # Drop non-numeric columns
    numeric_features = features_df.select_dtypes(include=[np.number])

    # Build and display profiles
    profiles = build_cluster_profiles(numeric_features, pipeline.labels_)
    print_cluster_profiles(profiles)

    # Save for human review
    save_profiles(profiles, args.save_profiles)
    print(f"\nProfiles saved to {args.save_profiles}")
    print("Edit 'suggested_label' in that file to name each cluster.")
