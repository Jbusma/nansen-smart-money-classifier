"""Sync feature and raw transaction data to ClickHouse.

Handles two pipelines:
1. Feature sync: wallet_features table for low-latency serving (<50ms).
2. Raw data sync: transactions, token_transfers, contract_interactions
   tables for on-chain context enrichment queries.

Usage:
    python -m src.data.clickhouse_sync               # create tables + health check
    python -m src.data.clickhouse_sync --sync-raw     # load raw parquets into CH
    python -m src.data.clickhouse_sync --sync-features # load features.parquet
"""

from __future__ import annotations

from pathlib import Path

import clickhouse_connect
import pandas as pd
import pyarrow.parquet as pq
import structlog
from tqdm import tqdm

from src.config import settings

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Schema definitions
# ---------------------------------------------------------------------------

WALLET_FEATURES_DDL = """
CREATE TABLE IF NOT EXISTS {database}.wallet_features (
    wallet_address                String,

    -- Behavioral features (12 derived columns)
    tx_frequency_per_day          Float64,
    activity_regularity           Float64,
    hour_of_day_entropy           Float64,
    weekend_vs_weekday_ratio      Float64,
    avg_holding_duration_estimate Float64,
    gas_price_sensitivity         Float64,
    is_contract                   Float64,
    dex_to_total_ratio            Float64,
    lending_to_total_ratio        Float64,
    counterparty_concentration    Float64,
    value_velocity                Float64,
    burst_score                   Float64,

    -- Metadata
    updated_at                    DateTime DEFAULT now()
)
ENGINE = MergeTree()
ORDER BY wallet_address
"""

GROUND_TRUTH_DDL = """
CREATE TABLE IF NOT EXISTS {database}.ground_truth (
    address         String,
    label           String,
    source          String,
    total_tx        Float64,
    dex_tx          Float64,
    dex_ratio       Float64,
    total_eth       Float64,
    tx_per_day      Float64,
    wallet_address  String
)
ENGINE = ReplacingMergeTree()
ORDER BY address
"""

PROTOCOL_REGISTRY_DDL = """
CREATE TABLE IF NOT EXISTS {database}.protocol_registry (
    address       String,
    label         String,
    category      String,
    source        String,
    updated_at    DateTime DEFAULT now()
)
ENGINE = ReplacingMergeTree(updated_at)
ORDER BY address
"""

LLM_NARRATIVE_CACHE_DDL = """
CREATE TABLE IF NOT EXISTS {database}.llm_narrative_cache (
    wallet_address  String,
    narrative       String,
    cluster_id      Int32,
    created_at      DateTime DEFAULT now()
)
ENGINE = MergeTree()
ORDER BY (wallet_address, created_at)
TTL created_at + INTERVAL 24 HOUR
"""

# ---------------------------------------------------------------------------
# Raw data schema definitions
# ---------------------------------------------------------------------------

RAW_TRANSACTIONS_DDL = """
CREATE TABLE IF NOT EXISTS {database}.raw_transactions (
    hash              String,
    block_number      Int64,
    block_timestamp   DateTime64(6, 'UTC'),
    from_address      String,
    to_address        String,
    value_eth         Float64,
    gas               Int64,
    gas_price         Int64,
    receipt_status    UInt8,
    method_id         String
)
ENGINE = MergeTree()
ORDER BY (from_address, block_timestamp)
"""

RAW_TOKEN_TRANSFERS_DDL = """
CREATE TABLE IF NOT EXISTS {database}.raw_token_transfers (
    transaction_hash  String,
    block_timestamp   DateTime64(6, 'UTC'),
    from_address      String,
    to_address        String,
    token_address     String,
    raw_value         String,
    is_erc20          UInt8,
    is_erc721         UInt8
)
ENGINE = MergeTree()
ORDER BY (from_address, block_timestamp)
"""

RAW_CONTRACT_INTERACTIONS_DDL = """
CREATE TABLE IF NOT EXISTS {database}.raw_contract_interactions (
    transaction_hash  String,
    block_timestamp   DateTime64(6, 'UTC'),
    from_address      String,
    to_address        String,
    trace_type        String,
    value_eth         Float64,
    gas_used          Float64,
    status            Float64,
    method_id         String,
    is_erc20          UInt8,
    is_erc721         UInt8
)
ENGINE = MergeTree()
ORDER BY (from_address, block_timestamp)
"""

RAW_TABLES = {
    "raw_transactions": RAW_TRANSACTIONS_DDL,
    "raw_token_transfers": RAW_TOKEN_TRANSFERS_DDL,
    "raw_contract_interactions": RAW_CONTRACT_INTERACTIONS_DDL,
}

RAW_PARQUET_MAP: dict[str, str] = {
    "raw_transactions": "data/raw/transactions.parquet",
    "raw_token_transfers": "data/raw/token_transfers.parquet",
    "raw_contract_interactions": "data/raw/contract_interactions.parquet",
}

# Columns expected in the wallet_features table (excluding updated_at which
# has a DEFAULT).
WALLET_FEATURE_COLUMNS = [
    "wallet_address",
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

# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


def get_client() -> clickhouse_connect.driver.Client:
    """Return a ClickHouse client configured from application settings."""
    return clickhouse_connect.get_client(
        host=settings.clickhouse_host,
        port=settings.clickhouse_port,
        username=settings.clickhouse_user,
        password=settings.clickhouse_password,
        database=settings.clickhouse_database,
    )


# ---------------------------------------------------------------------------
# DDL helpers
# ---------------------------------------------------------------------------


def create_tables(*, include_raw: bool = True) -> None:
    """Create all ClickHouse tables if they do not already exist.

    Parameters
    ----------
    include_raw : bool
        If True (default), also create the raw_transactions,
        raw_token_transfers, and raw_contract_interactions tables.
    """
    client = get_client()
    db = settings.clickhouse_database

    # Ensure the database itself exists.
    client.command(f"CREATE DATABASE IF NOT EXISTS {db}")

    client.command(WALLET_FEATURES_DDL.format(database=db))
    logger.info("ensured_table_exists", table="wallet_features", database=db)

    client.command(GROUND_TRUTH_DDL.format(database=db))
    logger.info("ensured_table_exists", table="ground_truth", database=db)

    client.command(PROTOCOL_REGISTRY_DDL.format(database=db))
    logger.info("ensured_table_exists", table="protocol_registry", database=db)

    client.command(LLM_NARRATIVE_CACHE_DDL.format(database=db))
    logger.info("ensured_table_exists", table="llm_narrative_cache", database=db)

    if include_raw:
        for table_name, ddl in RAW_TABLES.items():
            client.command(ddl.format(database=db))
            logger.info("ensured_table_exists", table=table_name, database=db)


# ---------------------------------------------------------------------------
# Write path
# ---------------------------------------------------------------------------


def sync_features(df: pd.DataFrame) -> None:
    """Insert or replace feature vectors into ClickHouse.

    Uses DELETE + INSERT to achieve upsert semantics on MergeTree.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain at least the columns listed in
        ``WALLET_FEATURE_COLUMNS``.
    """
    if df.empty:
        logger.warning("sync_features_called_with_empty_dataframe")
        return

    client = get_client()
    db = settings.clickhouse_database

    wallet_addresses = df["wallet_address"].tolist()

    # Delete existing rows for the wallets we are about to upsert so that
    # we don't accumulate duplicates in MergeTree.
    placeholders = ", ".join(f"'{w}'" for w in wallet_addresses)
    client.command(f"ALTER TABLE {db}.wallet_features DELETE WHERE wallet_address IN ({placeholders})")
    logger.info(
        "deleted_stale_rows",
        table="wallet_features",
        count=len(wallet_addresses),
    )

    # Ensure column order matches the DDL.
    insert_df = df[WALLET_FEATURE_COLUMNS].copy()

    client.insert_df(f"{db}.wallet_features", insert_df)
    logger.info(
        "inserted_features",
        table="wallet_features",
        rows=len(insert_df),
    )


def sync_ground_truth(parquet_path: str | Path = "data/ground_truth.parquet") -> int:
    """Load ground truth labels into ClickHouse (full replace).

    Returns the number of rows inserted.
    """
    path = Path(parquet_path)
    if not path.exists():
        raise FileNotFoundError(f"Ground truth not found: {path}")

    df = pd.read_parquet(path)
    if df.empty:
        logger.warning("sync_ground_truth_empty")
        return 0

    client = get_client()
    db = settings.clickhouse_database

    client.command(f"TRUNCATE TABLE IF EXISTS {db}.ground_truth")

    # Ensure column order and types match DDL
    expected_cols = [
        "address",
        "label",
        "source",
        "total_tx",
        "dex_tx",
        "dex_ratio",
        "total_eth",
        "tx_per_day",
        "wallet_address",
    ]
    for col in expected_cols:
        if col not in df.columns:
            df[col] = "" if col in ("address", "label", "source", "wallet_address") else 0.0

    insert_df = df[expected_cols].copy()
    str_cols = insert_df.select_dtypes(include=["object"]).columns
    insert_df[str_cols] = insert_df[str_cols].fillna("")

    client.insert_df(f"{db}.ground_truth", insert_df)
    logger.info("synced_ground_truth", rows=len(insert_df))
    return len(insert_df)


# ---------------------------------------------------------------------------
# Raw data sync (parquet -> ClickHouse in chunks)
# ---------------------------------------------------------------------------

# Column type coercions applied before inserting into ClickHouse.
_COLUMN_COERCIONS: dict[str, dict[str, str]] = {
    "raw_transactions": {
        "value_eth": "float64",
        "receipt_status": "uint8",
    },
    "raw_token_transfers": {
        "is_erc20": "uint8",
        "is_erc721": "uint8",
    },
    "raw_contract_interactions": {
        "is_erc20": "uint8",
        "is_erc721": "uint8",
    },
}


def _coerce_chunk(table_name: str, df: pd.DataFrame) -> pd.DataFrame:
    """Apply type coercions needed for ClickHouse insert."""
    # Convert boolean columns to int before numeric coercion (nullable
    # booleans reject fillna(0) directly).
    bool_cols = df.select_dtypes(include=["boolean", "bool"]).columns
    for col in bool_cols:
        df[col] = df[col].astype("object").fillna(0).astype("uint8")

    coercions = _COLUMN_COERCIONS.get(table_name, {})
    for col, dtype in coercions.items():
        if col in df.columns and df[col].dtype not in ("uint8", "int8"):
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(dtype)
    # Ensure string columns have no NaN (ClickHouse rejects them)
    str_cols = df.select_dtypes(include=["object"]).columns
    df[str_cols] = df[str_cols].fillna("")
    return df


def sync_raw_table(
    table_name: str,
    parquet_path: str | Path | None = None,
    *,
    batch_size: int = 100_000,
    truncate_first: bool = True,
) -> int:
    """Load a raw parquet file into ClickHouse in chunks.

    Parameters
    ----------
    table_name : str
        One of ``raw_transactions``, ``raw_token_transfers``,
        ``raw_contract_interactions``.
    parquet_path : str or Path, optional
        Override the default path from ``RAW_PARQUET_MAP``.
    batch_size : int
        Number of rows per INSERT batch.
    truncate_first : bool
        If True, TRUNCATE the target table before loading (full refresh).

    Returns
    -------
    int
        Total rows inserted.
    """
    if table_name not in RAW_TABLES:
        raise ValueError(f"Unknown table: {table_name}. Must be one of {list(RAW_TABLES)}")

    path = Path(parquet_path) if parquet_path else Path(RAW_PARQUET_MAP[table_name])
    if not path.exists():
        raise FileNotFoundError(f"Parquet file not found: {path}")

    client = get_client()
    db = settings.clickhouse_database
    full_table = f"{db}.{table_name}"

    if truncate_first:
        client.command(f"TRUNCATE TABLE IF EXISTS {full_table}")
        logger.info("truncated_table", table=table_name)

    pf = pq.ParquetFile(path)
    total_rows = pf.metadata.num_rows
    expected_columns = [
        col.split()[0]
        for col in RAW_TABLES[table_name].split("(", 1)[1].rsplit(")", 1)[0].strip().split("\n")
        if col.strip() and not col.strip().startswith("--")
    ]

    inserted = 0
    with tqdm(total=total_rows, desc=f"Loading {table_name}", unit="rows") as pbar:
        for batch in pf.iter_batches(batch_size=batch_size):
            chunk = batch.to_pandas()
            chunk = _coerce_chunk(table_name, chunk)

            # Keep only columns that match the DDL
            chunk = chunk[[c for c in expected_columns if c in chunk.columns]]

            client.insert_df(full_table, chunk)
            inserted += len(chunk)
            pbar.update(len(chunk))

    logger.info("sync_raw_complete", table=table_name, rows=inserted)
    return inserted


def sync_all_raw(*, batch_size: int = 100_000) -> dict[str, int]:
    """Load all three raw parquet files into ClickHouse.

    Returns a dict mapping table name -> rows inserted.
    """
    results: dict[str, int] = {}
    for table_name in RAW_TABLES:
        path = Path(RAW_PARQUET_MAP[table_name])
        if not path.exists():
            logger.warning("skipping_missing_parquet", table=table_name, path=str(path))
            continue
        results[table_name] = sync_raw_table(table_name, batch_size=batch_size)
    return results


# ---------------------------------------------------------------------------
# Read path
# ---------------------------------------------------------------------------


def get_wallet_features(wallet_address: str) -> dict:
    """Fetch the feature vector for a single wallet.

    Returns an empty dict if the wallet is not found.
    """
    client = get_client()
    db = settings.clickhouse_database

    result = client.query(
        f"SELECT * FROM {db}.wallet_features WHERE wallet_address = {{addr:String}}",
        parameters={"addr": wallet_address},
    )

    if not result.result_rows:
        logger.debug("wallet_not_found", wallet_address=wallet_address)
        return {}

    columns = result.column_names
    row = result.result_rows[0]
    return dict(zip(columns, row, strict=True))


def get_batch_features(wallet_addresses: list[str]) -> pd.DataFrame:
    """Fetch feature vectors for a batch of wallets.

    Returns a DataFrame with one row per found wallet.  Wallets that are not
    in ClickHouse are silently omitted.
    """
    if not wallet_addresses:
        return pd.DataFrame(columns=WALLET_FEATURE_COLUMNS)

    client = get_client()
    db = settings.clickhouse_database

    result = client.query(
        f"SELECT * FROM {db}.wallet_features WHERE wallet_address IN {{addrs:Array(String)}}",
        parameters={"addrs": wallet_addresses},
    )

    if not result.result_rows:
        return pd.DataFrame(columns=WALLET_FEATURE_COLUMNS)

    return pd.DataFrame(result.result_rows, columns=result.column_names)


# ---------------------------------------------------------------------------
# Ops
# ---------------------------------------------------------------------------


def health_check() -> bool:
    """Return True if ClickHouse is reachable and the wallet_features table
    exists."""
    try:
        client = get_client()
        db = settings.clickhouse_database
        result = client.query(
            "SELECT count() FROM system.tables WHERE database = {db:String} AND name = 'wallet_features'",
            parameters={"db": db},
        )
        table_exists = bool(result.result_rows[0][0] > 0)
        logger.info("health_check", ok=table_exists)
        return table_exists
    except Exception:
        logger.exception("health_check_failed")
        return False


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ClickHouse sync utilities.")
    parser.add_argument("--sync-raw", action="store_true", help="Load raw parquets into ClickHouse.")
    parser.add_argument("--sync-features", action="store_true", help="Load features.parquet into ClickHouse.")
    parser.add_argument("--sync-ground-truth", action="store_true", help="Load ground_truth.parquet into ClickHouse.")
    parser.add_argument("--batch-size", type=int, default=100_000, help="Rows per INSERT batch.")
    args = parser.parse_args()

    create_tables(include_raw=args.sync_raw or not args.sync_features)
    healthy = health_check()
    logger.info("clickhouse_ready", healthy=healthy)

    if args.sync_raw:
        results = sync_all_raw(batch_size=args.batch_size)
        for table, count in results.items():
            logger.info("loaded", table=table, rows=count)

    if args.sync_features:
        features_path = Path("data/features.parquet")
        if features_path.exists():
            features_df = pd.read_parquet(features_path)
            sync_features(features_df)
            logger.info("synced_features", rows=len(features_df))
        else:
            logger.warning("features_parquet_not_found", path=str(features_path))

    if args.sync_ground_truth:
        rows = sync_ground_truth()
        logger.info("synced_ground_truth", rows=rows)
