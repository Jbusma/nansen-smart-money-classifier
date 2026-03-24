"""Sample 20% of each cluster and pull wallet context for AI labeling.

Outputs a JSON file with full context per wallet, ready for analysis.

Usage:
    python -m src.experiments.sample_wallet_contexts
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import structlog
from tqdm import tqdm

from src.data.wallet_context import get_wallet_context

logger = structlog.get_logger(__name__)

OUTPUT_DIR = Path("data/experiment")
SAMPLE_RATIO = 0.20


def sample_from_cluster(
    df: pd.DataFrame,
    labels: np.ndarray,
    features_scaled: np.ndarray,
    cluster_id: int,
    ratio: float = SAMPLE_RATIO,
) -> list[str]:
    """Sample wallets from a cluster, stratified by distance to centroid.

    Takes evenly spaced wallets across the distance distribution
    (not just nearest) for better representativeness.
    """
    mask = labels == cluster_id
    cluster_features = features_scaled[mask]
    cluster_addrs = df.loc[mask, "wallet_address"].values

    centroid = cluster_features.mean(axis=0)
    dists = np.linalg.norm(cluster_features - centroid, axis=1)
    sorted_idx = np.argsort(dists)

    n_sample = max(1, int(len(cluster_addrs) * ratio))
    # Evenly spaced indices across the sorted distance array
    pick_idx = np.linspace(0, len(sorted_idx) - 1, n_sample, dtype=int)
    selected = sorted_idx[pick_idx]

    return [cluster_addrs[i] for i in selected]


def pull_contexts(addresses: list[str], cluster_id: int) -> list[dict]:
    """Pull wallet context for a list of addresses."""
    results = []
    for addr in tqdm(addresses, desc=f"Cluster {cluster_id}", unit="wallet"):
        ctx = get_wallet_context(addr)

        # Flatten for readability
        entry = {
            "wallet_address": addr,
            "cluster_id": cluster_id,
        }

        ts = ctx.get("transaction_summary") or {}
        entry["tx_count"] = ts.get("total_transactions", 0)
        entry["total_eth"] = round(ts.get("total_eth_volume", 0), 2)
        entry["avg_tx_eth"] = round(ts.get("avg_tx_value_eth", 0), 4)
        entry["first_seen"] = ts.get("first_seen")
        entry["last_seen"] = ts.get("last_seen")

        contracts = ctx.get("top_contracts") or []
        entry["top_contracts"] = [
            {
                "address": c["address"],
                "label": c.get("protocol_label") or "Unknown",
                "category": c.get("category", "unknown"),
                "interactions": c["interaction_count"],
                "eth": round(c["total_eth"], 2),
            }
            for c in contracts[:5]
        ]

        tokens = ctx.get("token_activity") or {}
        entry["unique_tokens"] = tokens.get("unique_tokens", 0)

        timing = ctx.get("timing_patterns") or {}
        entry["weekday_ratio"] = timing.get("weekday_ratio", 0)
        entry["active_hours"] = timing.get("most_active_hours", [])

        results.append(entry)

    return results


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    disc = pd.read_parquet(OUTPUT_DIR / "discovery_clustered.parquet")
    pipeline = joblib.load(OUTPUT_DIR / "discovery_pipeline.joblib")
    features_scaled = pipeline["features_scaled_"]
    labels = pipeline["labels_"]

    all_contexts: dict[str, list[dict]] = {}

    for cluster_id in sorted(set(labels)):
        if cluster_id == -1:
            continue

        n_total = (labels == cluster_id).sum()
        addresses = sample_from_cluster(disc, labels, features_scaled, cluster_id)
        logger.info(
            "sampling_cluster",
            cluster=cluster_id,
            total=n_total,
            sample=len(addresses),
        )

        contexts = pull_contexts(addresses, cluster_id)
        all_contexts[f"cluster_{cluster_id}"] = contexts

    # Save
    output_path = OUTPUT_DIR / "sampled_contexts.json"
    with open(output_path, "w") as f:
        json.dump(all_contexts, f, indent=2, default=str)

    logger.info("saved_contexts", path=str(output_path))

    # Print summary
    for key, contexts in all_contexts.items():
        print(f"\n{key}: {len(contexts)} wallets sampled")
        labeled = sum(1 for c in contexts if any(ct["label"] != "Unknown" for ct in c["top_contracts"]))
        print(f"  With identified protocols: {labeled}/{len(contexts)}")


if __name__ == "__main__":
    main()
