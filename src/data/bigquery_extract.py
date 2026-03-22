"""Extract Ethereum transaction data from BigQuery public dataset.

Pulls from ``bigquery-public-data.crypto_ethereum`` using a rolling window,
filters for wallets meeting minimum activity thresholds, and streams the
result set as local Parquet files (one per table category).

Results are streamed in chunks via ``to_dataframe_iterable()`` to avoid
loading entire result sets into memory.
"""

from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import structlog
from google.cloud import bigquery
from google.cloud.bigquery import QueryJobConfig, ScalarQueryParameter

from src.config import settings

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_OUTPUT_DIR: Path = Path("data/raw")

_COST_PER_TB: float = 6.25  # USD, BigQuery on-demand pricing

# ---------------------------------------------------------------------------
# Client helper
# ---------------------------------------------------------------------------


def _get_client() -> bigquery.Client:
    """Return a BigQuery client bound to the configured GCP project."""
    return bigquery.Client(project=settings.gcp_project_id)


def _cutoff_date() -> str:
    """Return the ISO date string for the start of the sampling window."""
    cutoff = dt.datetime.now(dt.UTC).date() - dt.timedelta(days=settings.sample_window_days)
    return cutoff.isoformat()


def _addresses_literal(addresses: Sequence[str]) -> str:
    """Build a SQL literal list like ``('0xabc', '0xdef', ...)``."""
    escaped = ", ".join(f"'{a}'" for a in addresses)
    return f"({escaped})"


# ---------------------------------------------------------------------------
# Query execution
# ---------------------------------------------------------------------------


def _run_query(
    client: bigquery.Client,
    sql: str,
    params: list | None = None,
    *,
    description: str = "",
) -> pd.DataFrame:
    """Execute *sql* and return a small result set as a DataFrame.

    Use only for queries that return a manageable number of rows (e.g.
    active wallets). For large result sets, use ``_stream_to_parquet``.
    """
    job_config = QueryJobConfig()
    if params:
        job_config.query_parameters = params

    log = logger.bind(description=description)
    log.info("bigquery.query.start")

    try:
        query_job = client.query(sql, job_config=job_config)
        df = query_job.to_dataframe()
    except Exception:
        log.exception("bigquery.query.failed")
        raise

    bytes_billed = getattr(query_job, "total_bytes_billed", None) or 0
    cost = (bytes_billed / 1e12) * _COST_PER_TB
    log.info(
        "bigquery.query.done",
        rows=len(df),
        gb_billed=round(bytes_billed / 1e9, 1),
        cost_usd=round(cost, 3),
    )
    return df


def _stream_to_parquet(
    client: bigquery.Client,
    sql: str,
    output_path: Path,
    *,
    description: str = "",
) -> int:
    """Execute *sql* and stream results to a Parquet file in chunks.

    Uses ``to_dataframe_iterable()`` so only one chunk (~10-50 MB) is in
    memory at a time. Returns the total number of rows written.
    """
    log = logger.bind(description=description)
    log.info("bigquery.stream.start")

    try:
        query_job = client.query(sql)
        row_iter = query_job.result()
    except Exception:
        log.exception("bigquery.stream.query_failed")
        raise

    total_rows = 0
    writer: pq.ParquetWriter | None = None

    try:
        for chunk_df in row_iter.to_dataframe_iterable():
            chunk_rows = len(chunk_df)
            if chunk_rows == 0:
                continue

            table = pa.Table.from_pandas(chunk_df, preserve_index=False)

            if writer is None:
                writer = pq.ParquetWriter(str(output_path), table.schema)

            writer.write_table(table)
            total_rows += chunk_rows
            print(f"\r      -> {total_rows:,} rows streamed...", end="", flush=True)
    finally:
        if writer is not None:
            writer.close()

    print()  # newline after progress

    bytes_billed = getattr(query_job, "total_bytes_billed", None) or 0
    cost = (bytes_billed / 1e12) * _COST_PER_TB
    log.info(
        "bigquery.stream.done",
        rows=total_rows,
        gb_billed=round(bytes_billed / 1e9, 1),
        cost_usd=round(cost, 3),
    )
    return total_rows


# ---------------------------------------------------------------------------
# Extraction functions
# ---------------------------------------------------------------------------


def extract_active_wallets(client: bigquery.Client) -> pd.DataFrame:
    """Identify wallets meeting the activity thresholds within the window.

    Samples up to ``settings.max_wallets`` wallets, stratified by ETH volume
    to get a mix of whales, mid-tier, and smaller wallets.
    """
    src = settings.bigquery_source_dataset
    cutoff = _cutoff_date()

    sql = f"""
    WITH outgoing AS (
        SELECT
            from_address AS wallet_address,
            COUNT(*)     AS tx_count,
            COALESCE(SUM(CAST(value AS BIGNUMERIC) / 1e18), 0) AS total_eth
        FROM `{src}.transactions`
        WHERE block_timestamp >= TIMESTAMP(@cutoff)
        GROUP BY from_address
    ),
    incoming AS (
        SELECT
            to_address AS wallet_address,
            COUNT(*)   AS tx_count,
            COALESCE(SUM(CAST(value AS BIGNUMERIC) / 1e18), 0) AS total_eth
        FROM `{src}.transactions`
        WHERE block_timestamp >= TIMESTAMP(@cutoff)
          AND to_address IS NOT NULL
        GROUP BY to_address
    ),
    combined AS (
        SELECT
            COALESCE(o.wallet_address, i.wallet_address)    AS wallet_address,
            COALESCE(o.tx_count, 0) + COALESCE(i.tx_count, 0) AS tx_count,
            COALESCE(o.total_eth, 0) + COALESCE(i.total_eth, 0) AS total_eth
        FROM outgoing o
        FULL OUTER JOIN incoming i
            ON o.wallet_address = i.wallet_address
    ),
    qualified AS (
        SELECT
            wallet_address,
            tx_count,
            CAST(total_eth AS FLOAT64) AS total_eth,
            NTILE(3) OVER (ORDER BY total_eth DESC) AS tier
        FROM combined
        WHERE tx_count  >= @min_tx
          AND total_eth >= @min_eth
    )
    SELECT wallet_address, tx_count, total_eth
    FROM (
        SELECT *, ROW_NUMBER() OVER (PARTITION BY tier ORDER BY FARM_FINGERPRINT(wallet_address)) AS rn
        FROM qualified
    )
    WHERE rn <= @max_per_tier
    ORDER BY total_eth DESC
    """

    max_per_tier = settings.max_wallets // 3

    params = [
        ScalarQueryParameter("cutoff", "STRING", cutoff),
        ScalarQueryParameter("min_tx", "INT64", settings.min_wallet_tx_count),
        ScalarQueryParameter("min_eth", "FLOAT64", settings.min_wallet_eth_transacted),
        ScalarQueryParameter("max_per_tier", "INT64", max_per_tier),
    ]

    df = _run_query(client, sql, params, description="extract_active_wallets")
    logger.info("active_wallets.extracted", count=len(df))
    return df


def _build_transactions_sql(addresses: Sequence[str]) -> str:
    src = settings.bigquery_source_dataset
    cutoff = _cutoff_date()
    addr_list = _addresses_literal(addresses)
    return f"""
    SELECT
        `hash`,
        block_number,
        block_timestamp,
        from_address,
        to_address,
        CAST(value AS BIGNUMERIC) / 1e18 AS value_eth,
        gas,
        gas_price,
        receipt_status,
        SUBSTR(input, 1, 10) AS method_id
    FROM `{src}.transactions`
    WHERE block_timestamp >= TIMESTAMP('{cutoff}')
      AND (from_address IN {addr_list} OR to_address IN {addr_list})
    """


def _build_token_transfers_sql(addresses: Sequence[str]) -> str:
    src = settings.bigquery_source_dataset
    cutoff = _cutoff_date()
    addr_list = _addresses_literal(addresses)
    return f"""
    SELECT
        tt.transaction_hash,
        tt.block_timestamp,
        tt.from_address,
        tt.to_address,
        tt.token_address,
        tt.value AS raw_value,
        c.is_erc20,
        c.is_erc721
    FROM `{src}.token_transfers` tt
    LEFT JOIN `{src}.contracts` c
        ON tt.token_address = c.address
    WHERE tt.block_timestamp >= TIMESTAMP('{cutoff}')
      AND (tt.from_address IN {addr_list} OR tt.to_address IN {addr_list})
    """


def _build_contract_interactions_sql(addresses: Sequence[str]) -> str:
    src = settings.bigquery_source_dataset
    cutoff = _cutoff_date()
    addr_list = _addresses_literal(addresses)
    return f"""
    SELECT
        t.transaction_hash,
        t.block_timestamp,
        t.from_address,
        t.to_address,
        t.trace_type,
        CAST(t.value AS BIGNUMERIC) / 1e18 AS value_eth,
        t.gas_used,
        t.status,
        SUBSTR(t.input, 1, 10) AS method_id,
        c.is_erc20,
        c.is_erc721
    FROM `{src}.traces` t
    LEFT JOIN `{src}.contracts` c
        ON t.to_address = c.address
    WHERE t.block_timestamp >= TIMESTAMP('{cutoff}')
      AND t.trace_type = 'call'
      AND (t.from_address IN {addr_list} OR t.to_address IN {addr_list})
    """


# ---------------------------------------------------------------------------
# Cost estimator
# ---------------------------------------------------------------------------


def estimate_cost(client: bigquery.Client) -> float:
    """Dry-run all queries and return estimated total cost in USD."""
    cutoff = _cutoff_date()
    src = settings.bigquery_source_dataset
    job_config = QueryJobConfig(dry_run=True, use_query_cache=False)

    ts_filter = f"block_timestamp >= TIMESTAMP('{cutoff}')"
    queries = [
        f"SELECT from_address FROM `{src}.transactions` WHERE {ts_filter}",
        f"SELECT `hash` FROM `{src}.transactions` WHERE {ts_filter}",
        f"SELECT transaction_hash FROM `{src}.token_transfers` WHERE {ts_filter}",
        f"SELECT transaction_hash FROM `{src}.traces` WHERE {ts_filter} AND trace_type = 'call'",
    ]

    total_bytes = 0
    for sql in queries:
        job = client.query(sql, job_config=job_config)
        total_bytes += job.total_bytes_processed or 0

    total_cost = (total_bytes / 1e12) * _COST_PER_TB
    logger.info(
        "cost_estimate",
        total_gb=round(total_bytes / 1e9, 1),
        total_usd=round(total_cost, 2),
    )
    return total_cost


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def run_full_extraction(
    output_dir: Path | None = None,
    skip_existing: bool = True,
) -> dict[str, Path]:
    """Run the complete extraction pipeline and stream to Parquet.

    All large queries use chunked streaming to avoid OOM. Each dataset is
    saved immediately so partial results survive crashes.

    If *skip_existing* is True, datasets that already exist on disk are
    skipped (useful for resuming after a failure).
    """
    output_dir = output_dir or _OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    client = _get_client()
    artifacts: dict[str, Path] = {}

    logger.info(
        "full_extraction.start",
        window_days=settings.sample_window_days,
        max_wallets=settings.max_wallets,
        min_tx=settings.min_wallet_tx_count,
        min_eth=settings.min_wallet_eth_transacted,
    )

    # Step 1 -- active wallets (small result, fits in memory)
    aw_path = output_dir / "active_wallets.parquet"
    if skip_existing and aw_path.exists():
        print("[1/4] Active wallets — loading from disk (already extracted)")
        wallets_df = pd.read_parquet(aw_path)
    else:
        print("[1/4] Extracting active wallets...")
        wallets_df = extract_active_wallets(client)
        if wallets_df.empty:
            logger.warning("full_extraction.no_wallets_found")
            return {}
        wallets_df.to_parquet(aw_path, index=False, engine="pyarrow")
        logger.info("saved", dataset="active_wallets", path=str(aw_path), rows=len(wallets_df))

    artifacts["active_wallets"] = aw_path
    print(f"      -> {len(wallets_df):,} wallets")

    addresses = wallets_df["wallet_address"].tolist()

    # Steps 2-4 -- large tables, streamed to parquet
    steps = [
        ("transactions", "[2/4]", _build_transactions_sql(addresses)),
        ("token_transfers", "[3/4]", _build_token_transfers_sql(addresses)),
        ("contract_interactions", "[4/4]", _build_contract_interactions_sql(addresses)),
    ]

    for name, step, sql in steps:
        path = output_dir / f"{name}.parquet"
        if skip_existing and path.exists():
            print(f"{step} {name} — skipping (already on disk)")
            artifacts[name] = path
            continue

        print(f"{step} Extracting {name}...")
        rows = _stream_to_parquet(client, sql, path, description=name)
        artifacts[name] = path
        print(f"      -> {rows:,} rows saved to {path}")

    logger.info("full_extraction.complete", datasets=list(artifacts.keys()))
    return artifacts


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(0),
    )

    parser = argparse.ArgumentParser(description="Extract Ethereum data from BigQuery")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_OUTPUT_DIR,
        help="Directory for output Parquet files (default: data/raw)",
    )
    parser.add_argument(
        "--max-wallets",
        type=int,
        default=None,
        help="Override max wallets to sample (default: from settings)",
    )
    parser.add_argument(
        "--estimate-cost",
        action="store_true",
        help="Dry-run to estimate cost without executing queries",
    )
    parser.add_argument(
        "--no-skip",
        action="store_true",
        help="Re-extract all datasets even if they exist on disk",
    )
    args = parser.parse_args()

    if args.max_wallets:
        settings.max_wallets = args.max_wallets

    bq_client = _get_client()

    if args.estimate_cost:
        cost = estimate_cost(bq_client)
        print(f"\nEstimated cost: ${cost:.2f}")
        sys.exit(0)

    try:
        cost = estimate_cost(bq_client)
        print(f"\n{'=' * 50}")
        print(f"  Estimated cost: ${cost:.2f}")
        print(f"  Wallets to sample: {settings.max_wallets:,}")
        print(f"  Window: {settings.sample_window_days} days")
        print(f"{'=' * 50}")

        confirm = input("\nProceed? [y/N] ").strip().lower()
        if confirm != "y":
            print("Aborted.")
            sys.exit(0)

        artifacts = run_full_extraction(
            output_dir=args.output_dir,
            skip_existing=not args.no_skip,
        )
        if artifacts:
            print(f"\nExtraction complete. {len(artifacts)} datasets written:")
            for name, path in artifacts.items():
                size_mb = path.stat().st_size / 1e6
                print(f"  {name}: {path} ({size_mb:.0f} MB)")
        else:
            print("\nNo data extracted (no qualifying wallets found).")
            sys.exit(1)
    except Exception as exc:
        logger.exception("extraction.fatal", error=str(exc))
        sys.exit(2)
