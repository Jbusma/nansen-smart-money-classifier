"""Sync feature data to ClickHouse for low-latency serving (<50ms queries).

Handles the daily batch pipeline: BigQuery/parquet -> ClickHouse.
Target table: wallet_features (MergeTree, ordered by wallet_address).
"""

from __future__ import annotations

import clickhouse_connect
import pandas as pd
import structlog

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


def create_tables() -> None:
    """Create the wallet_features and llm_narrative_cache tables if they
    do not already exist."""
    client = get_client()
    db = settings.clickhouse_database

    # Ensure the database itself exists.
    client.command(f"CREATE DATABASE IF NOT EXISTS {db}")

    client.command(WALLET_FEATURES_DDL.format(database=db))
    logger.info("ensured_table_exists", table="wallet_features", database=db)

    client.command(LLM_NARRATIVE_CACHE_DDL.format(database=db))
    logger.info("ensured_table_exists", table="llm_narrative_cache", database=db)


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
    client.command(
        f"ALTER TABLE {db}.wallet_features DELETE "
        f"WHERE wallet_address IN ({placeholders})"
    )
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
        f"SELECT * FROM {db}.wallet_features "
        f"WHERE wallet_address = {{addr:String}}",
        parameters={"addr": wallet_address},
    )

    if not result.result_rows:
        logger.debug("wallet_not_found", wallet_address=wallet_address)
        return {}

    columns = result.column_names
    row = result.result_rows[0]
    return dict(zip(columns, row))


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
        f"SELECT * FROM {db}.wallet_features "
        f"WHERE wallet_address IN {{addrs:Array(String)}}",
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
            "SELECT count() FROM system.tables "
            "WHERE database = {db:String} AND name = 'wallet_features'",
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
    create_tables()
    healthy = health_check()
    logger.info("clickhouse_ready", healthy=healthy)
