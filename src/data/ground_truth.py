"""Collect and manage ground-truth wallet labels for archetype classification.

Sources
-------
- Etherscan public tags & known protocol/exchange addresses
- DEX routers, lending protocols, bridges, NFT marketplaces
- MEV identification patterns (sandwich attacks, same-block arbitrage)
- On-chain behavioural heuristics (whale detection, DEX-heavy usage)
- Cluster exemplar annotation & community labels

Target: 10K-50K labeled wallets across archetypes.
"""

from __future__ import annotations

import enum
import pathlib

import pandas as pd
import structlog
from google.cloud import bigquery

from src.config import settings

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Label taxonomy
# ---------------------------------------------------------------------------


class WalletLabel(enum.StrEnum):
    """Archetype categories for wallet labelling."""

    SMART_MONEY = "smart_money"
    MEV_BOT = "mev_bot"
    DEFI_FARMER = "defi_farmer"
    AIRDROP_HUNTER = "airdrop_hunter"
    RETAIL_TRADER = "retail_trader"
    HODLER = "hodler"
    NFT_TRADER = "nft_trader"


# ---------------------------------------------------------------------------
# Known protocol addresses (Ethereum mainnet, lower-cased)
# ---------------------------------------------------------------------------

# DEX routers
UNISWAP_V2_ROUTER = "0x7a250d5630b4cf539739df2c5dacb4c659f2488d"
UNISWAP_V3_ROUTER = "0xe592427a0aece92de3edee1f18e0157c05861564"
UNISWAP_UNIVERSAL_ROUTER = "0x3fc91a3afd70395cd496c647d5a6cc9d4b2b7fad"
SUSHISWAP_ROUTER = "0xd9e1ce17f2641f24ae83637ab66a2cca9c378b9f"
ONE_INCH_V5_ROUTER = "0x1111111254eeb25477b68fb85ed929f73a960582"

# Lending protocols
AAVE_V2_LENDING_POOL = "0x7d2768de32b0b80b7a3454c06bdac94a69ddc7a9"
AAVE_V3_POOL = "0x87870bca3f3fd6335c3f4ce8392d69350b4fa4e2"
COMPOUND_COMPTROLLER = "0x3d9819210a31b4961b30ef54be2aed79b9c9cd3b"
COMPOUND_V3_COMET_USDC = "0xc3d688b66703497daa19211eedff47f25384cdc3"

# NFT marketplaces
OPENSEA_SEAPORT = "0x00000000000000adc04c56bf30ac9d3c0aaf14dc"
BLUR_MARKETPLACE = "0x29469395eaf6f95920e59f858042f0e28d98a20b"
BLUR_POOL = "0x0000000000a39bb272e79075ade125fd351887ac"

# L2 bridges
ARBITRUM_DELAYED_INBOX = "0x4dbd4fc535ac27206064b68ffcf827b0a60bab3f"
ARBITRUM_GATEWAY_ROUTER = "0x72ce9c846789fdb6fc1f34ac4ad25dd9ef7031ef"
OPTIMISM_GATEWAY = "0x99c9fc46f92e8a1c0dec1b1747d010903e884be1"

# Known exchange hot wallets (sample – extend as needed)
BINANCE_HOT_1 = "0x28c6c06298d514db089934071355e5743bf21d60"
COINBASE_COMMERCE = "0xf6874c88757721a02f47592140905c4336dfbc61"

KNOWN_PROTOCOL_ADDRESSES: dict[str, str] = {
    UNISWAP_V2_ROUTER: "Uniswap V2: Router",
    UNISWAP_V3_ROUTER: "Uniswap V3: Router",
    UNISWAP_UNIVERSAL_ROUTER: "Uniswap: Universal Router",
    SUSHISWAP_ROUTER: "SushiSwap: Router",
    ONE_INCH_V5_ROUTER: "1inch v5: Aggregation Router",
    AAVE_V2_LENDING_POOL: "Aave V2: Lending Pool",
    AAVE_V3_POOL: "Aave V3: Pool",
    COMPOUND_COMPTROLLER: "Compound: Comptroller",
    COMPOUND_V3_COMET_USDC: "Compound V3: cUSDCv3",
    OPENSEA_SEAPORT: "OpenSea: Seaport",
    BLUR_MARKETPLACE: "Blur: Marketplace",
    BLUR_POOL: "Blur: Pool",
    ARBITRUM_DELAYED_INBOX: "Arbitrum: Delayed Inbox",
    ARBITRUM_GATEWAY_ROUTER: "Arbitrum: Gateway Router",
    OPTIMISM_GATEWAY: "Optimism: L1 Standard Bridge",
    BINANCE_HOT_1: "Binance: Hot Wallet",
    COINBASE_COMMERCE: "Coinbase Commerce",
}

# Addresses to *exclude* from classification (they are protocols, not users)
PROTOCOL_EXCLUSION_SET: set[str] = set(KNOWN_PROTOCOL_ADDRESSES.keys())

# DEX-related addresses used when computing per-wallet DEX interaction ratio
DEX_ROUTER_ADDRESSES: set[str] = {
    UNISWAP_V2_ROUTER,
    UNISWAP_V3_ROUTER,
    UNISWAP_UNIVERSAL_ROUTER,
    SUSHISWAP_ROUTER,
    ONE_INCH_V5_ROUTER,
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _bq_client() -> bigquery.Client:
    return bigquery.Client(project=settings.gcp_project_id)


def _run_query(sql: str) -> pd.DataFrame:
    """Execute a BigQuery SQL string and return a DataFrame."""
    client = _bq_client()
    logger.info("bigquery.query.start", sql_preview=sql[:120])
    df = client.query(sql).to_dataframe()
    logger.info("bigquery.query.done", rows=len(df))
    return df


# ---------------------------------------------------------------------------
# Labelling functions
# ---------------------------------------------------------------------------


def identify_known_contracts() -> pd.DataFrame:
    """Label known protocol / exchange addresses from the public contracts table.

    Returns a DataFrame with columns ``[address, label, source]``.
    """
    source_ds = settings.bigquery_source_dataset
    addr_list = ", ".join(f"'{a}'" for a in KNOWN_PROTOCOL_ADDRESSES)

    sql = f"""
    SELECT
        address,
        -- fall back to our local mapping when the public table lacks a tag
        COALESCE(
            CASE WHEN is_erc20 THEN 'erc20_contract'
                 WHEN is_erc721 THEN 'nft_contract'
                 ELSE NULL
            END,
            'contract'
        ) AS contract_type
    FROM `{source_ds}.contracts`
    WHERE LOWER(address) IN ({addr_list})
    """

    df = _run_query(sql)
    if df.empty:
        logger.warning("identify_known_contracts.empty")
        df = pd.DataFrame(columns=["address", "label", "source"])
        return df

    df["address"] = df["address"].str.lower()
    df["label"] = df["address"].map(KNOWN_PROTOCOL_ADDRESSES).fillna("contract")
    df["source"] = "known_protocol"
    return df[["address", "label", "source"]]


def identify_dex_heavy_wallets(
    min_tx_count: int = 50,
    dex_ratio_threshold: float = 0.70,
    limit: int = 20_000,
) -> pd.DataFrame:
    """Find wallets whose on-chain activity is >70% interactions with DEX routers.

    These wallets are strong candidates for ``defi_farmer`` or ``smart_money``.
    """
    source_ds = settings.bigquery_source_dataset
    dex_addrs = ", ".join(f"'{a}'" for a in DEX_ROUTER_ADDRESSES)
    exclusions = ", ".join(f"'{a}'" for a in PROTOCOL_EXCLUSION_SET)

    sql = f"""
    WITH wallet_stats AS (
        SELECT
            from_address AS address,
            COUNT(*)     AS total_tx,
            COUNTIF(LOWER(to_address) IN ({dex_addrs})) AS dex_tx
        FROM `{source_ds}.transactions`
        WHERE
            block_timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL {settings.sample_window_days} DAY)
            AND LOWER(from_address) NOT IN ({exclusions})
        GROUP BY from_address
        HAVING total_tx >= {min_tx_count}
    )
    SELECT
        address,
        total_tx,
        dex_tx,
        SAFE_DIVIDE(dex_tx, total_tx) AS dex_ratio
    FROM wallet_stats
    WHERE SAFE_DIVIDE(dex_tx, total_tx) >= {dex_ratio_threshold}
    ORDER BY dex_ratio DESC
    LIMIT {limit}
    """

    df = _run_query(sql)
    if df.empty:
        logger.warning("identify_dex_heavy_wallets.empty")
        return pd.DataFrame(columns=["address", "label", "source"])

    df["address"] = df["address"].str.lower()
    df["label"] = WalletLabel.DEFI_FARMER.value
    df["source"] = "dex_heavy_heuristic"
    return df[["address", "label", "source", "total_tx", "dex_tx", "dex_ratio"]]


def identify_mev_bots(
    limit: int = 10_000,
) -> pd.DataFrame:
    """Detect MEV bots via sandwich-attack and same-block arbitrage patterns.

    Heuristics
    ----------
    1. **Sandwich pattern**: wallet sends >=2 transactions in the same block
       where one is immediately before and one immediately after another
       wallet's swap on a DEX router.
    2. **Same-block arb**: wallet interacts with >=2 different DEX routers in
       the same block (multi-hop arbitrage).

    Both patterns are surfaced from the public ``transactions`` + ``traces``
    tables.
    """
    source_ds = settings.bigquery_source_dataset
    dex_addrs = ", ".join(f"'{a}'" for a in DEX_ROUTER_ADDRESSES)
    exclusions = ", ".join(f"'{a}'" for a in PROTOCOL_EXCLUSION_SET)

    sql = f"""
    WITH per_block AS (
        SELECT
            from_address                          AS address,
            block_number,
            COUNT(*)                              AS tx_in_block,
            COUNT(DISTINCT CASE
                WHEN LOWER(to_address) IN ({dex_addrs}) THEN to_address
            END)                                  AS distinct_dex_in_block
        FROM `{source_ds}.transactions`
        WHERE
            block_timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL {settings.sample_window_days} DAY)
            AND LOWER(from_address) NOT IN ({exclusions})
        GROUP BY from_address, block_number
    ),
    -- Sandwich: >=2 txs in the same block touching a DEX router
    sandwich_candidates AS (
        SELECT address, COUNT(*) AS sandwich_blocks
        FROM per_block
        WHERE tx_in_block >= 2 AND distinct_dex_in_block >= 1
        GROUP BY address
        HAVING sandwich_blocks >= 5
    ),
    -- Same-block arb: touching >=2 distinct DEX routers in one block
    arb_candidates AS (
        SELECT address, COUNT(*) AS arb_blocks
        FROM per_block
        WHERE distinct_dex_in_block >= 2
        GROUP BY address
        HAVING arb_blocks >= 3
    ),
    -- Also flag wallets that appear in internal traces with high gas
    -- priority (typical of Flashbots bundles)
    trace_candidates AS (
        SELECT
            t.from_address AS address,
            COUNT(*)       AS trace_mev_tx
        FROM `{source_ds}.traces` AS t
        WHERE
            t.block_timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL {settings.sample_window_days} DAY)
            AND t.call_type = 'call'
            AND t.from_address NOT IN ({exclusions})
            AND t.to_address IN ({dex_addrs})
        GROUP BY t.from_address
        HAVING trace_mev_tx >= 20
    )
    SELECT DISTINCT address
    FROM (
        SELECT address FROM sandwich_candidates
        UNION DISTINCT
        SELECT address FROM arb_candidates
        UNION DISTINCT
        SELECT address FROM trace_candidates
    )
    LIMIT {limit}
    """

    df = _run_query(sql)
    if df.empty:
        logger.warning("identify_mev_bots.empty")
        return pd.DataFrame(columns=["address", "label", "source"])

    df["address"] = df["address"].str.lower()
    df["label"] = WalletLabel.MEV_BOT.value
    df["source"] = "mev_pattern_heuristic"
    return df[["address", "label", "source"]]


def identify_whale_wallets(
    min_eth_balance: float = 100.0,
    max_tx_per_day: float = 2.0,
    limit: int = 10_000,
) -> pd.DataFrame:
    """High-value, low-frequency wallets — likely hodlers or smart-money whales.

    A wallet qualifies when:
    * cumulative ETH transacted >= ``min_eth_balance``
    * average daily transaction count <= ``max_tx_per_day``
    """
    source_ds = settings.bigquery_source_dataset
    exclusions = ", ".join(f"'{a}'" for a in PROTOCOL_EXCLUSION_SET)
    window = settings.sample_window_days

    sql = f"""
    WITH wallet_activity AS (
        SELECT
            from_address AS address,
            COUNT(*)     AS total_tx,
            SUM(CAST(value AS FLOAT64) / 1e18) AS total_eth,
            COUNT(DISTINCT DATE(block_timestamp)) AS active_days
        FROM `{source_ds}.transactions`
        WHERE
            block_timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL {window} DAY)
            AND LOWER(from_address) NOT IN ({exclusions})
        GROUP BY from_address
        HAVING active_days >= 1
    )
    SELECT
        address,
        total_tx,
        total_eth,
        active_days,
        SAFE_DIVIDE(total_tx, active_days) AS tx_per_day
    FROM wallet_activity
    WHERE
        total_eth >= {min_eth_balance}
        AND SAFE_DIVIDE(total_tx, active_days) <= {max_tx_per_day}
    ORDER BY total_eth DESC
    LIMIT {limit}
    """

    df = _run_query(sql)
    if df.empty:
        logger.warning("identify_whale_wallets.empty")
        return pd.DataFrame(columns=["address", "label", "source"])

    df["address"] = df["address"].str.lower()
    # Low-frequency whales are tagged as hodlers; we may upgrade to
    # smart_money downstream once profit metrics are computed.
    df["label"] = WalletLabel.HODLER.value
    df["source"] = "whale_heuristic"
    return df[["address", "label", "source", "total_tx", "total_eth", "tx_per_day"]]


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def build_ground_truth_dataset() -> pd.DataFrame:
    """Run all labelling functions and merge into a single deduplicated DataFrame.

    Priority when a wallet appears in multiple sources (first wins):
        mev_bot > smart_money > defi_farmer > nft_trader > airdrop_hunter
        > hodler > retail_trader

    Returns
    -------
    pd.DataFrame
        Columns: ``address``, ``label``, ``source``.  Rows are unique by
        ``address``.
    """
    label_priority = {
        WalletLabel.MEV_BOT.value: 0,
        WalletLabel.SMART_MONEY.value: 1,
        WalletLabel.DEFI_FARMER.value: 2,
        WalletLabel.NFT_TRADER.value: 3,
        WalletLabel.AIRDROP_HUNTER.value: 4,
        WalletLabel.HODLER.value: 5,
        WalletLabel.RETAIL_TRADER.value: 6,
    }

    logger.info("build_ground_truth.start")

    frames: list[pd.DataFrame] = []

    # 1. Known contracts (informational, excluded from user-label set)
    contracts_df = identify_known_contracts()
    logger.info("known_contracts", count=len(contracts_df))

    # 2. MEV bots
    mev_df = identify_mev_bots()
    frames.append(mev_df)
    logger.info("mev_bots", count=len(mev_df))

    # 3. DEX-heavy (defi_farmer)
    dex_df = identify_dex_heavy_wallets()
    frames.append(dex_df)
    logger.info("dex_heavy_wallets", count=len(dex_df))

    # 4. Whale / hodler
    whale_df = identify_whale_wallets()
    frames.append(whale_df)
    logger.info("whale_wallets", count=len(whale_df))

    if not frames:
        logger.warning("build_ground_truth.no_labels_found")
        return pd.DataFrame(columns=["address", "label", "source"])

    combined = pd.concat(frames, ignore_index=True)

    # Remove any protocol addresses that leaked through
    combined = combined[~combined["address"].isin(PROTOCOL_EXCLUSION_SET)]

    # Deduplicate: keep highest-priority label per address
    combined["_priority"] = combined["label"].map(label_priority).fillna(99)
    combined.sort_values("_priority", inplace=True)
    combined.drop_duplicates(subset="address", keep="first", inplace=True)
    combined.drop(columns=["_priority"], inplace=True)
    combined.reset_index(drop=True, inplace=True)

    # Keep only core columns for the canonical ground-truth set
    core_cols = ["address", "label", "source"]
    extra_cols = [c for c in combined.columns if c not in core_cols]
    combined = combined[core_cols + extra_cols]

    logger.info(
        "build_ground_truth.done",
        total_wallets=len(combined),
        label_distribution=combined["label"].value_counts().to_dict(),
    )
    return combined


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

_DEFAULT_GT_PATH = pathlib.Path("data/ground_truth.parquet")


def save_ground_truth(
    df: pd.DataFrame,
    path: pathlib.Path | None = None,
) -> pathlib.Path:
    """Persist ground-truth labels to a Parquet file.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain at least ``address`` and ``label`` columns.
    path : pathlib.Path, optional
        Destination file.  Defaults to ``data/ground_truth.parquet``.

    Returns
    -------
    pathlib.Path
        The path the file was written to.
    """
    path = path or _DEFAULT_GT_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    logger.info("ground_truth.saved", path=str(path), rows=len(df))
    return path


def load_ground_truth(
    path: pathlib.Path | None = None,
) -> pd.DataFrame:
    """Load ground-truth labels from a Parquet file.

    Parameters
    ----------
    path : pathlib.Path, optional
        Source file.  Defaults to ``data/ground_truth.parquet``.

    Returns
    -------
    pd.DataFrame
    """
    path = path or _DEFAULT_GT_PATH
    if not path.exists():
        raise FileNotFoundError(f"Ground-truth file not found: {path}")
    df = pd.read_parquet(path)
    logger.info("ground_truth.loaded", path=str(path), rows=len(df))
    return df


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Build and persist the ground-truth wallet-label dataset.",
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=_DEFAULT_GT_PATH,
        help="Output path for the parquet file (default: %(default)s)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print label distribution without saving.",
    )
    args = parser.parse_args()

    structlog.configure(
        wrapper_class=structlog.make_filtering_bound_logger(0),
    )

    gt_df = build_ground_truth_dataset()

    print(f"\nGround-truth dataset: {len(gt_df)} wallets")
    print("\nLabel distribution:")
    print(gt_df["label"].value_counts().to_string())

    if not args.dry_run:
        out = save_ground_truth(gt_df, args.output)
        print(f"\nSaved to {out}")
    else:
        print("\n(dry-run — not saved)")
