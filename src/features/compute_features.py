"""CLI script to compute features from raw parquet files in data/raw/.

Strategy: scan each large parquet file ONCE, keeping only rows that involve
active wallets.  This filtered data fits comfortably in memory on a 64GB
machine.  Then process wallets in batches for feature computation.

Usage:
    python -m src.features.compute_features [--batch-size 200] [--output data/features.parquet]
"""

from __future__ import annotations

import argparse
import pathlib
import time

import pandas as pd
import pyarrow.parquet as pq
import structlog
from tqdm import tqdm

from src.features.feature_engineering import (
    compute_all_features,
    impute_missing,
    preprocess_raw_data,
)

logger = structlog.get_logger(__name__)


def _load_filtered(
    parquet_path: pathlib.Path,
    address_columns: list[str],
    active_wallets: set[str],
    batch_size: int = 1_000_000,
) -> pd.DataFrame:
    """Scan a parquet file once, returning only rows matching active wallets.

    Reads in batches to avoid loading the full file into memory before
    filtering.

    Parameters
    ----------
    parquet_path : pathlib.Path
        Path to the parquet file.
    address_columns : list[str]
        Column names to check for wallet addresses.
    active_wallets : set[str]
        Lowercase wallet addresses to keep.
    batch_size : int
        Number of rows per batch during scanning.

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame with only active-wallet rows.
    """
    pf = pq.ParquetFile(parquet_path)
    total_batches = (pf.metadata.num_rows + batch_size - 1) // batch_size
    chunks: list[pd.DataFrame] = []

    for batch in tqdm(
        pf.iter_batches(batch_size=batch_size),
        desc=f"Loading {parquet_path.name}",
        total=total_batches,
        unit="batch",
    ):
        df = batch.to_pandas()
        mask = pd.Series(False, index=df.index)
        for col in address_columns:
            if col in df.columns:
                mask = mask | df[col].str.lower().isin(active_wallets)
        if mask.any():
            chunks.append(df[mask])

    if not chunks:
        return pd.DataFrame()
    result = pd.concat(chunks, ignore_index=True)
    logger.info(
        "loaded_filtered",
        file=parquet_path.name,
        original_rows=pf.metadata.num_rows,
        filtered_rows=len(result),
    )
    return result


def compute_features_from_raw(
    data_dir: pathlib.Path,
    output_path: pathlib.Path,
    batch_size: int = 200,
) -> pd.DataFrame:
    """Compute features for all active wallets from raw parquet files.

    Each large file is scanned ONCE to filter to active-wallet rows.
    Then wallets are processed in batches for feature computation.

    Parameters
    ----------
    data_dir : pathlib.Path
        Directory containing the raw parquet files.
    output_path : pathlib.Path
        Where to save the resulting features parquet.
    batch_size : int
        Number of wallets to process per batch during feature computation.

    Returns
    -------
    pd.DataFrame
        The computed features DataFrame.
    """
    t0 = time.time()

    # 1. Load the wallet list
    wallets_df = pd.read_parquet(data_dir / "active_wallets.parquet")
    all_wallets = wallets_df["wallet_address"].str.lower().unique().tolist()
    active_set = set(all_wallets)
    logger.info("wallets_loaded", count=len(all_wallets))

    # 2. Scan each file ONCE, filtering to active wallets
    txs_filtered = _load_filtered(
        data_dir / "transactions.parquet",
        ["from_address", "to_address"],
        active_set,
    )
    tt_filtered = _load_filtered(
        data_dir / "token_transfers.parquet",
        ["from_address", "to_address"],
        active_set,
    )
    ci_filtered = _load_filtered(
        data_dir / "contract_interactions.parquet",
        ["from_address", "to_address"],
        active_set,
    )

    logger.info(
        "all_files_loaded",
        txs=len(txs_filtered),
        tt=len(tt_filtered),
        ci=len(ci_filtered),
    )

    if txs_filtered.empty:
        logger.error("no_transactions_found_for_active_wallets")
        return pd.DataFrame()

    # 3. Preprocess once for all wallets (rename columns, add wallet_address)
    logger.info("preprocessing")
    txs_pp, tt_pp, ci_pp = preprocess_raw_data(all_wallets, txs_filtered, tt_filtered, ci_filtered)
    logger.info(
        "preprocessed",
        txs=len(txs_pp),
        tt=len(tt_pp),
        ci=len(ci_pp),
    )

    # Free raw data memory
    del txs_filtered, tt_filtered, ci_filtered

    if txs_pp.empty:
        logger.error("no_transactions_after_preprocess")
        return pd.DataFrame()

    # 4. Compute features in batches of wallets
    # Split wallet list into batches and compute features for each
    wallets_in_data = txs_pp["wallet_address"].unique().tolist()
    n_batches = (len(wallets_in_data) + batch_size - 1) // batch_size
    all_features: list[pd.DataFrame] = []

    for i in tqdm(range(n_batches), desc="Computing features", unit="batch"):
        batch_start = i * batch_size
        batch_end = min((i + 1) * batch_size, len(wallets_in_data))
        batch_wallets_set = set(wallets_in_data[batch_start:batch_end])

        txs_batch = txs_pp[txs_pp["wallet_address"].isin(batch_wallets_set)]
        tt_batch = tt_pp[tt_pp["wallet_address"].isin(batch_wallets_set)]
        ci_batch = ci_pp[ci_pp["wallet_address"].isin(batch_wallets_set)]

        batch_features = compute_all_features(txs_batch, tt_batch, ci_batch)
        all_features.append(batch_features)

    if not all_features:
        logger.error("no_features_computed")
        return pd.DataFrame()

    # 5. Combine and save
    features_df = pd.concat(all_features, ignore_index=True)
    features_df = impute_missing(features_df)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    features_df.to_parquet(output_path, index=False)

    elapsed = time.time() - t0
    logger.info(
        "features_computed",
        wallets=len(features_df),
        elapsed_minutes=round(elapsed / 60, 1),
        output=str(output_path),
    )

    return features_df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute behavioral features from raw parquet data.",
    )
    parser.add_argument(
        "--data-dir",
        type=pathlib.Path,
        default=pathlib.Path("data/raw"),
        help="Directory with raw parquet files (default: data/raw).",
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=pathlib.Path("data/features.parquet"),
        help="Output path for features parquet (default: data/features.parquet).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=200,
        help="Number of wallets per feature-computation batch (default: 200).",
    )
    args = parser.parse_args()

    structlog.configure(
        wrapper_class=structlog.make_filtering_bound_logger(0),
    )

    result = compute_features_from_raw(
        data_dir=args.data_dir,
        output_path=args.output,
        batch_size=args.batch_size,
    )
    print(f"\nComputed features for {len(result)} wallets")
    print(f"Saved to {args.output}")
