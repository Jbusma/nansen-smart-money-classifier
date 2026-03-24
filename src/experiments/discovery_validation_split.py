"""Split wallets into discovery/validation sets and recluster for labeling.

Methodology:
1. Join features with ETH volume from ClickHouse
2. Sort by descending volume, interleave into two equal halves
   (odd indices → discovery, even indices → validation)
3. Recluster discovery set with UMAP + HDBSCAN
4. Export exemplars per cluster for AI-assisted labeling
5. Save splits + cluster assignments for downstream training

Usage:
    python -m src.experiments.discovery_validation_split
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import structlog
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from src.config import settings
from src.data.clickhouse_sync import get_client

logger = structlog.get_logger(__name__)

OUTPUT_DIR = Path("data/experiment")
RANDOM_SEED = settings.random_seed


def load_features_with_volume() -> pd.DataFrame:
    """Load feature vectors and join with raw ETH volume from ClickHouse."""
    logger.info("loading_features")
    df = pd.read_parquet("data/features.parquet")

    logger.info("querying_eth_volumes")
    client = get_client()
    result = client.query("""
        SELECT
            from_address AS wallet_address,
            sum(value_eth) AS total_eth_volume,
            count() AS tx_count
        FROM nansen.raw_transactions
        GROUP BY from_address
    """)

    vol_df = pd.DataFrame(
        result.result_rows,
        columns=["wallet_address", "total_eth_volume", "tx_count"],
    )

    merged = df.merge(vol_df, on="wallet_address", how="left")
    merged["total_eth_volume"] = merged["total_eth_volume"].fillna(0)
    merged["tx_count"] = merged["tx_count"].fillna(0).astype(int)

    logger.info(
        "features_loaded",
        wallets=len(merged),
        with_volume=int((merged["total_eth_volume"] > 0).sum()),
    )
    return merged


def stratified_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Sort by descending ETH volume, interleave into two equal halves."""
    sorted_df = df.sort_values("total_eth_volume", ascending=False).reset_index(drop=True)

    discovery = sorted_df.iloc[::2].copy().reset_index(drop=True)
    validation = sorted_df.iloc[1::2].copy().reset_index(drop=True)

    logger.info(
        "split_complete",
        discovery=len(discovery),
        validation=len(validation),
        discovery_vol=f"{discovery['total_eth_volume'].sum():,.0f}",
        validation_vol=f"{validation['total_eth_volume'].sum():,.0f}",
    )
    return discovery, validation


def recluster(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, dict]:
    """Run UMAP + HDBSCAN on the feature matrix."""
    import hdbscan

    feature_cols = [
        "tx_frequency_per_day",
        "activity_regularity",
        "hour_of_day_entropy",
        "weekend_vs_weekday_ratio",
        "avg_holding_duration_estimate",
        "gas_price_sensitivity",
        "is_contract",
        "dex_to_total_ratio",
        "lending_to_total_ratio",
        "counterparty_concentration",
        "value_velocity",
        "burst_score",
    ]

    features = df[feature_cols].values
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    logger.info("running_hdbscan_direct", n_samples=len(features_scaled), n_features=len(feature_cols))
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=30,
        min_samples=5,
        prediction_data=True,
    )
    labels = clusterer.fit_predict(features_scaled)
    embedding = features_scaled[:, :2]  # placeholder for exemplar selection

    unique, counts = np.unique(labels, return_counts=True)
    cluster_summary = {}
    for label, count in zip(unique, counts, strict=True):
        name = "noise" if label == -1 else f"cluster_{label}"
        pct = count / len(labels) * 100
        cluster_summary[name] = {"size": int(count), "pct": round(pct, 2)}
        logger.info("cluster_found", name=name, size=int(count), pct=f"{pct:.1f}%")

    pipeline = {
        "clusterer": clusterer,
        "scaler": scaler,
        "features_scaled_": features_scaled,
        "labels_": labels,
        "feature_names_": feature_cols,
    }

    return labels, embedding, pipeline


def get_exemplars(
    df: pd.DataFrame,
    labels: np.ndarray,
    embedding: np.ndarray,
    n_per_cluster: int = 15,
) -> pd.DataFrame:
    """Get the N wallets nearest each cluster centroid."""
    exemplars = []

    for cluster_id in sorted(set(labels)):
        if cluster_id == -1:
            continue

        mask = labels == cluster_id
        cluster_emb = embedding[mask]
        cluster_df = df.loc[mask].copy()

        centroid = cluster_emb.mean(axis=0)
        dists = np.linalg.norm(cluster_emb - centroid, axis=1)
        nearest_idx = np.argsort(dists)[:n_per_cluster]

        for idx in nearest_idx:
            exemplars.append(
                {
                    "wallet_address": cluster_df.iloc[idx]["wallet_address"],
                    "cluster_id": int(cluster_id),
                    "dist_to_centroid": float(dists[idx]),
                    "total_eth_volume": float(cluster_df.iloc[idx]["total_eth_volume"]),
                }
            )

    return pd.DataFrame(exemplars)


def profile_exemplars(exemplars_df: pd.DataFrame) -> pd.DataFrame:
    """Pull wallet context for each exemplar and summarize."""
    from src.data.wallet_context import get_wallet_context

    profiles = []
    for _, row in tqdm(
        exemplars_df.iterrows(),
        total=len(exemplars_df),
        desc="Profiling exemplars",
    ):
        ctx = get_wallet_context(row["wallet_address"])
        ts = ctx.get("transaction_summary") or {}
        contracts = ctx.get("top_contracts") or []
        timing = ctx.get("timing_patterns") or {}
        tokens = ctx.get("token_activity") or {}

        # Summarize top protocols
        labeled_contracts = [c for c in contracts if c.get("protocol_label")]
        top_protocols = (
            ", ".join(f"{c['protocol_label']} [{c['category']}]" for c in labeled_contracts[:5]) or "None identified"
        )

        unknown_pct = sum(1 for c in contracts if not c.get("protocol_label")) / max(len(contracts), 1)

        profiles.append(
            {
                "wallet_address": row["wallet_address"],
                "cluster_id": row["cluster_id"],
                "dist_to_centroid": row["dist_to_centroid"],
                "total_eth_volume": row["total_eth_volume"],
                "tx_count": ts.get("total_transactions", 0),
                "avg_tx_eth": ts.get("avg_tx_value_eth", 0),
                "first_seen": ts.get("first_seen"),
                "last_seen": ts.get("last_seen"),
                "unique_tokens": tokens.get("unique_tokens", 0),
                "weekday_ratio": timing.get("weekday_ratio", 0),
                "active_hours": str(timing.get("most_active_hours", [])),
                "top_protocols": top_protocols,
                "unknown_contract_pct": round(unknown_pct, 2),
            }
        )

    return pd.DataFrame(profiles)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load features + volume
    df = load_features_with_volume()

    # 2. Stratified split
    discovery, validation = stratified_split(df)

    discovery.to_parquet(OUTPUT_DIR / "discovery.parquet", index=False)
    validation.to_parquet(OUTPUT_DIR / "validation.parquet", index=False)
    logger.info("saved_splits", path=str(OUTPUT_DIR))

    # 3. Recluster discovery set
    labels, embedding, pipeline = recluster(discovery)

    discovery["cluster_id"] = labels
    discovery.to_parquet(OUTPUT_DIR / "discovery_clustered.parquet", index=False)
    joblib.dump(pipeline, OUTPUT_DIR / "discovery_pipeline.joblib")
    logger.info("saved_clustering", path=str(OUTPUT_DIR / "discovery_pipeline.joblib"))

    # 4. Get exemplars (use full scaled features for distance, not 2D placeholder)
    exemplars = get_exemplars(
        discovery,
        labels,
        pipeline["features_scaled_"],
        n_per_cluster=15,
    )
    logger.info("exemplars_selected", count=len(exemplars))

    # 5. Profile exemplars with wallet context
    profiles = profile_exemplars(exemplars)
    profiles.to_csv(OUTPUT_DIR / "exemplar_profiles.csv", index=False)
    logger.info("saved_profiles", path=str(OUTPUT_DIR / "exemplar_profiles.csv"))

    # Print summary for AI labeling
    print("\n" + "=" * 80)
    print("DISCOVERY SET CLUSTER PROFILES")
    print("=" * 80)

    for cluster_id in sorted(set(labels)):
        if cluster_id == -1:
            continue
        cluster_profiles = profiles[profiles["cluster_id"] == cluster_id]
        n = (labels == cluster_id).sum()
        print(f"\n--- Cluster {cluster_id} ({n} wallets) ---")
        print(f"  Avg ETH volume: {cluster_profiles['total_eth_volume'].mean():,.0f}")
        print(f"  Avg tx count: {cluster_profiles['tx_count'].mean():,.0f}")
        print(f"  Avg tx size: {cluster_profiles['avg_tx_eth'].mean():.2f} ETH")
        print(f"  Avg unique tokens: {cluster_profiles['unique_tokens'].mean():.0f}")
        print(f"  Avg weekday ratio: {cluster_profiles['weekday_ratio'].mean():.1%}")
        print(f"  Unknown contract %: {cluster_profiles['unknown_contract_pct'].mean():.0%}")
        print("  Top protocols seen:")
        all_protocols = ", ".join(cluster_profiles["top_protocols"].tolist())
        print(f"    {all_protocols[:200]}")

    noise_count = (labels == -1).sum()
    if noise_count > 0:
        print(f"\n--- Noise ({noise_count} wallets) ---")

    print("\nNext step: review exemplar_profiles.csv and assign labels")
    print("Then run: python -m src.experiments.train_from_labels")


if __name__ == "__main__":
    main()
