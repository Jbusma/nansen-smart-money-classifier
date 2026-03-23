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
from typing import TYPE_CHECKING

import pandas as pd
import structlog
from tqdm import tqdm

if TYPE_CHECKING:
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

# NFT marketplace addresses
NFT_MARKETPLACE_ADDRESSES: set[str] = {
    OPENSEA_SEAPORT,
    BLUR_MARKETPLACE,
    BLUR_POOL,
}

# Lending protocol addresses
LENDING_PROTOCOL_ADDRESSES: set[str] = {
    AAVE_V2_LENDING_POOL,
    AAVE_V3_POOL,
    COMPOUND_COMPTROLLER,
    COMPOUND_V3_COMET_USDC,
}

# L2 bridge addresses
BRIDGE_ADDRESSES: set[str] = {
    ARBITRUM_DELAYED_INBOX,
    ARBITRUM_GATEWAY_ROUTER,
    OPTIMISM_GATEWAY,
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _bq_client() -> bigquery.Client:
    from google.cloud import bigquery as bq

    return bq.Client(project=settings.gcp_project_id)


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
# Local labelling functions (work from parquet files, $0 cost)
# ---------------------------------------------------------------------------


def identify_dex_heavy_wallets_local(
    transactions_df: pd.DataFrame,
    contract_interactions_df: pd.DataFrame,
    min_tx_count: int = 50,
    dex_ratio_threshold: float = 0.70,
) -> pd.DataFrame:
    """Find wallets whose on-chain activity is >70% interactions with DEX routers.

    Same heuristic as the BigQuery version but operates on local DataFrames.

    Parameters
    ----------
    transactions_df : pd.DataFrame
        Raw transactions from ``data/raw/transactions.parquet``.
    contract_interactions_df : pd.DataFrame
        Raw contract interactions from ``data/raw/contract_interactions.parquet``.
    min_tx_count : int
        Minimum number of transactions for a wallet to qualify.
    dex_ratio_threshold : float
        Minimum fraction of interactions targeting DEX routers.

    Returns
    -------
    pd.DataFrame
        Columns: ``address, label, source`` (plus ``total_tx, dex_tx, dex_ratio``).
    """
    logger.info("identify_dex_heavy_wallets_local.start")

    # Use contract_interactions which contain protocol-level interactions
    ci = contract_interactions_df.copy()
    ci["from_lower"] = ci["from_address"].str.lower()
    ci["to_lower"] = ci["to_address"].str.lower()

    # Exclude protocol addresses as senders
    ci = ci[~ci["from_lower"].isin(PROTOCOL_EXCLUSION_SET)]

    # Per-wallet stats
    stats = ci.groupby("from_lower").agg(
        total_tx=("to_lower", "size"),
        dex_tx=("to_lower", lambda s: s.isin(DEX_ROUTER_ADDRESSES).sum()),
    )
    stats = stats[stats["total_tx"] >= min_tx_count]
    stats["dex_ratio"] = stats["dex_tx"] / stats["total_tx"]
    stats = stats[stats["dex_ratio"] >= dex_ratio_threshold]

    if stats.empty:
        logger.warning("identify_dex_heavy_wallets_local.empty")
        return pd.DataFrame(columns=["address", "label", "source"])

    result = stats.reset_index().rename(columns={"from_lower": "address"})
    result["label"] = WalletLabel.DEFI_FARMER.value
    result["source"] = "dex_heavy_heuristic_local"

    logger.info("identify_dex_heavy_wallets_local.done", count=len(result))
    return result[["address", "label", "source", "total_tx", "dex_tx", "dex_ratio"]]


def identify_mev_bots_local(
    transactions_df: pd.DataFrame,
    min_sandwich_blocks: int = 5,
    min_arb_blocks: int = 3,
) -> pd.DataFrame:
    """Detect MEV bots via same-block multi-tx pattern detection from local data.

    Heuristics (mirrors the BigQuery version):
    1. Sandwich: wallet sends >=2 txs in the same block touching a DEX router
       (at least ``min_sandwich_blocks`` such blocks).
    2. Same-block arb: wallet interacts with >=2 distinct DEX routers in one
       block (at least ``min_arb_blocks`` such blocks).

    Parameters
    ----------
    transactions_df : pd.DataFrame
        Raw transactions from ``data/raw/transactions.parquet``.
    min_sandwich_blocks : int
        Minimum number of blocks with sandwich-like patterns.
    min_arb_blocks : int
        Minimum number of blocks with multi-DEX arb patterns.

    Returns
    -------
    pd.DataFrame
        Columns: ``address, label, source``.
    """
    logger.info("identify_mev_bots_local.start")

    txs = transactions_df.copy()
    txs["from_lower"] = txs["from_address"].str.lower()
    txs["to_lower"] = txs["to_address"].str.lower()

    # Exclude protocol addresses
    txs = txs[~txs["from_lower"].isin(PROTOCOL_EXCLUSION_SET)]

    # Per (wallet, block) stats
    per_block = txs.groupby(["from_lower", "block_number"]).agg(
        tx_in_block=("to_lower", "size"),
        distinct_dex_in_block=(
            "to_lower",
            lambda s: s[s.isin(DEX_ROUTER_ADDRESSES)].nunique(),
        ),
    )

    # Sandwich candidates: >=2 txs in block with at least 1 DEX interaction
    sandwich_mask = (per_block["tx_in_block"] >= 2) & (per_block["distinct_dex_in_block"] >= 1)
    sandwich_counts = per_block[sandwich_mask].groupby(level="from_lower").size().rename("sandwich_blocks")
    sandwich_wallets = set(sandwich_counts[sandwich_counts >= min_sandwich_blocks].index)

    # Arb candidates: >=2 distinct DEX routers in one block
    arb_mask = per_block["distinct_dex_in_block"] >= 2
    arb_counts = per_block[arb_mask].groupby(level="from_lower").size().rename("arb_blocks")
    arb_wallets = set(arb_counts[arb_counts >= min_arb_blocks].index)

    all_mev = sandwich_wallets | arb_wallets

    if not all_mev:
        logger.warning("identify_mev_bots_local.empty")
        return pd.DataFrame(columns=["address", "label", "source"])

    result = pd.DataFrame({"address": sorted(all_mev)})
    result["label"] = WalletLabel.MEV_BOT.value
    result["source"] = "mev_pattern_heuristic_local"

    logger.info("identify_mev_bots_local.done", count=len(result))
    return result[["address", "label", "source"]]


def identify_whale_wallets_local(
    transactions_df: pd.DataFrame,
    min_eth_balance: float = 100.0,
    max_tx_per_day: float = 2.0,
) -> pd.DataFrame:
    """High-value, low-frequency wallets from local data.

    A wallet qualifies when:
    * cumulative ETH transacted >= ``min_eth_balance``
    * average daily transaction count <= ``max_tx_per_day``

    Parameters
    ----------
    transactions_df : pd.DataFrame
        Raw transactions from ``data/raw/transactions.parquet``.
        Must have ``from_address``, ``value_eth``, ``block_timestamp``.
    min_eth_balance : float
        Minimum total ETH transacted.
    max_tx_per_day : float
        Maximum average daily transactions.

    Returns
    -------
    pd.DataFrame
        Columns: ``address, label, source`` (plus ``total_tx, total_eth, tx_per_day``).
    """
    logger.info("identify_whale_wallets_local.start")

    txs = transactions_df.copy()
    txs["from_lower"] = txs["from_address"].str.lower()

    # Exclude protocol addresses
    txs = txs[~txs["from_lower"].isin(PROTOCOL_EXCLUSION_SET)]

    # Parse value_eth to float (it may be string-typed from parquet)
    txs["value_float"] = pd.to_numeric(txs["value_eth"], errors="coerce").fillna(0.0)

    # Active days per wallet
    txs["tx_date"] = pd.to_datetime(txs["block_timestamp"]).dt.date

    stats = txs.groupby("from_lower").agg(
        total_tx=("from_lower", "size"),
        total_eth=("value_float", "sum"),
        active_days=("tx_date", "nunique"),
    )
    stats = stats[stats["active_days"] >= 1]
    stats["tx_per_day"] = stats["total_tx"] / stats["active_days"]

    qualified = stats[(stats["total_eth"] >= min_eth_balance) & (stats["tx_per_day"] <= max_tx_per_day)]

    if qualified.empty:
        logger.warning("identify_whale_wallets_local.empty")
        return pd.DataFrame(columns=["address", "label", "source"])

    result = qualified.reset_index().rename(columns={"from_lower": "address"})
    result["label"] = WalletLabel.HODLER.value
    result["source"] = "whale_heuristic_local"

    logger.info("identify_whale_wallets_local.done", count=len(result))
    return result[["address", "label", "source", "total_tx", "total_eth", "tx_per_day"]]


def identify_nft_traders_local(
    contract_interactions_df: pd.DataFrame,
    min_tx_count: int = 10,
    nft_ratio_threshold: float = 0.30,
) -> pd.DataFrame:
    """Wallets with >30% interactions targeting NFT marketplaces (OpenSea, Blur).

    Parameters
    ----------
    contract_interactions_df : pd.DataFrame
        Must have ``from_address``, ``to_address``.
    """
    logger.info("identify_nft_traders_local.start")

    ci = contract_interactions_df.copy()
    ci["from_lower"] = ci["from_address"].str.lower()
    ci["to_lower"] = ci["to_address"].str.lower()
    ci = ci[~ci["from_lower"].isin(PROTOCOL_EXCLUSION_SET)]

    stats = ci.groupby("from_lower").agg(
        total_tx=("to_lower", "size"),
        nft_tx=("to_lower", lambda s: s.isin(NFT_MARKETPLACE_ADDRESSES).sum()),
    )
    stats = stats[stats["total_tx"] >= min_tx_count]
    stats["nft_ratio"] = stats["nft_tx"] / stats["total_tx"]
    stats = stats[stats["nft_ratio"] >= nft_ratio_threshold]

    if stats.empty:
        logger.warning("identify_nft_traders_local.empty")
        return pd.DataFrame(columns=["address", "label", "source"])

    result = stats.reset_index().rename(columns={"from_lower": "address"})
    result["label"] = WalletLabel.NFT_TRADER.value
    result["source"] = "nft_marketplace_heuristic_local"

    logger.info("identify_nft_traders_local.done", count=len(result))
    return result[["address", "label", "source"]]


def identify_smart_money_local(
    transactions_df: pd.DataFrame,
    contract_interactions_df: pd.DataFrame,
    min_eth: float = 50.0,
    min_tx_per_day: float = 1.0,
    min_dex_ratio: float = 0.30,
) -> pd.DataFrame:
    """Sophisticated traders: high value + high frequency + significant DEX usage.

    These are diversified, active wallets that transact large volumes through
    DEX routers — not bots (too few same-block patterns) and not passive hodlers.

    Parameters
    ----------
    transactions_df : pd.DataFrame
        Must have ``from_address``, ``value_eth``, ``block_timestamp``.
    contract_interactions_df : pd.DataFrame
        Must have ``from_address``, ``to_address``.
    """
    logger.info("identify_smart_money_local.start")

    # Compute per-wallet value and frequency from transactions
    txs = transactions_df.copy()
    txs["from_lower"] = txs["from_address"].str.lower()
    txs = txs[~txs["from_lower"].isin(PROTOCOL_EXCLUSION_SET)]
    txs["value_float"] = pd.to_numeric(txs["value_eth"], errors="coerce").fillna(0.0)
    txs["tx_date"] = pd.to_datetime(txs["block_timestamp"]).dt.date

    tx_stats = txs.groupby("from_lower").agg(
        total_tx=("from_lower", "size"),
        total_eth=("value_float", "sum"),
        active_days=("tx_date", "nunique"),
    )
    tx_stats = tx_stats[tx_stats["active_days"] >= 1]
    tx_stats["tx_per_day"] = tx_stats["total_tx"] / tx_stats["active_days"]

    # Compute DEX ratio from contract interactions
    ci = contract_interactions_df.copy()
    ci["from_lower"] = ci["from_address"].str.lower()
    ci["to_lower"] = ci["to_address"].str.lower()
    ci = ci[~ci["from_lower"].isin(PROTOCOL_EXCLUSION_SET)]

    ci_stats = ci.groupby("from_lower").agg(
        ci_total=("to_lower", "size"),
        dex_tx=("to_lower", lambda s: s.isin(DEX_ROUTER_ADDRESSES).sum()),
    )
    ci_stats["dex_ratio"] = ci_stats["dex_tx"] / ci_stats["ci_total"]

    # Join and filter
    combined = tx_stats.join(ci_stats, how="inner")
    qualified = combined[
        (combined["total_eth"] >= min_eth)
        & (combined["tx_per_day"] >= min_tx_per_day)
        & (combined["dex_ratio"] >= min_dex_ratio)
    ]

    if qualified.empty:
        logger.warning("identify_smart_money_local.empty")
        return pd.DataFrame(columns=["address", "label", "source"])

    result = qualified.reset_index().rename(columns={"from_lower": "address"})
    result["label"] = WalletLabel.SMART_MONEY.value
    result["source"] = "smart_money_heuristic_local"

    logger.info("identify_smart_money_local.done", count=len(result))
    return result[["address", "label", "source"]]


def identify_airdrop_hunters_local(
    contract_interactions_df: pd.DataFrame,
    min_unique_contracts: int = 20,
    min_bridge_tx: int = 3,
) -> pd.DataFrame:
    """Wallets that interact with many unique contracts AND use L2 bridges.

    Airdrop hunters chase eligibility by interacting with many protocols and
    bridging to L2s early. They have high counterparty diversity.

    Parameters
    ----------
    contract_interactions_df : pd.DataFrame
        Must have ``from_address``, ``to_address``.
    """
    logger.info("identify_airdrop_hunters_local.start")

    ci = contract_interactions_df.copy()
    ci["from_lower"] = ci["from_address"].str.lower()
    ci["to_lower"] = ci["to_address"].str.lower()
    ci = ci[~ci["from_lower"].isin(PROTOCOL_EXCLUSION_SET)]

    stats = ci.groupby("from_lower").agg(
        unique_contracts=("to_lower", "nunique"),
        bridge_tx=("to_lower", lambda s: s.isin(BRIDGE_ADDRESSES).sum()),
    )

    qualified = stats[(stats["unique_contracts"] >= min_unique_contracts) & (stats["bridge_tx"] >= min_bridge_tx)]

    if qualified.empty:
        logger.warning("identify_airdrop_hunters_local.empty")
        return pd.DataFrame(columns=["address", "label", "source"])

    result = qualified.reset_index().rename(columns={"from_lower": "address"})
    result["label"] = WalletLabel.AIRDROP_HUNTER.value
    result["source"] = "airdrop_hunter_heuristic_local"

    logger.info("identify_airdrop_hunters_local.done", count=len(result))
    return result[["address", "label", "source"]]


def _deduplicate_labels(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Merge label DataFrames and deduplicate by priority.

    Priority (first wins):
        mev_bot > smart_money > defi_farmer > nft_trader > airdrop_hunter
        > hodler > retail_trader
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

    if not frames:
        return pd.DataFrame(columns=["address", "label", "source"])

    combined = pd.concat(frames, ignore_index=True)

    # Remove any protocol addresses that leaked through
    combined = combined[~combined["address"].isin(PROTOCOL_EXCLUSION_SET)]

    # Deduplicate: keep highest-priority label per address
    combined["_priority"] = combined["label"].map(label_priority).fillna(99)
    combined = combined.sort_values("_priority")
    combined = combined.drop_duplicates(subset="address", keep="first")
    combined = combined.drop(columns=["_priority"])
    combined = combined.reset_index(drop=True)

    # Keep only core columns for the canonical ground-truth set
    core_cols = ["address", "label", "source"]
    extra_cols = [c for c in combined.columns if c not in core_cols]
    combined = combined[core_cols + extra_cols]

    return combined


def _load_filtered_columns(
    parquet_path: pathlib.Path,
    columns: list[str],
    filter_col: str,
    filter_values: set[str],
    batch_size: int = 1_000_000,
    desc: str = "Scanning",
) -> pd.DataFrame:
    """Load specific columns from a parquet file, filtering to rows where
    ``filter_col`` (lowercased) is in ``filter_values``.

    Reads in batches to avoid full materialization of columns we don't need.
    """
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(parquet_path)
    total_batches = (pf.metadata.num_rows + batch_size - 1) // batch_size
    chunks: list[pd.DataFrame] = []

    for batch in tqdm(
        pf.iter_batches(batch_size=batch_size, columns=columns),
        desc=desc,
        total=total_batches,
        unit="batch",
    ):
        df = batch.to_pandas()
        mask = df[filter_col].str.lower().isin(filter_values)
        if mask.any():
            chunks.append(df[mask].copy())

    if not chunks:
        return pd.DataFrame(columns=columns)
    return pd.concat(chunks, ignore_index=True)


def build_ground_truth_local(
    data_dir: str | pathlib.Path = "data/raw",
) -> pd.DataFrame:
    """Build ground-truth labels from local parquet files ($0 cost).

    Loads only the columns needed for each heuristic, filtering to active
    wallets to keep memory usage reasonable.

    Parameters
    ----------
    data_dir : str | pathlib.Path
        Directory containing the raw parquet files.

    Returns
    -------
    pd.DataFrame
        Columns: ``address``, ``label``, ``source``, ``wallet_address``.
    """
    data_dir = pathlib.Path(data_dir)
    logger.info("build_ground_truth_local.start", data_dir=str(data_dir))

    # Load active wallet list for filtering
    wallets_df = pd.read_parquet(data_dir / "active_wallets.parquet")
    active_wallets = set(wallets_df["wallet_address"].str.lower().unique())
    logger.info("active_wallets_loaded", count=len(active_wallets))

    txs_path = data_dir / "transactions.parquet"
    ci_path = data_dir / "contract_interactions.parquet"

    frames: list[pd.DataFrame] = []

    # --- 1. MEV bots ---
    # Load only the columns needed: from_address, to_address, block_number
    logger.info("loading tx columns for MEV detection")
    txs_mev = _load_filtered_columns(
        txs_path,
        columns=["from_address", "to_address", "block_number"],
        filter_col="from_address",
        filter_values=active_wallets - PROTOCOL_EXCLUSION_SET,
        desc="  MEV: scanning transactions",
    )

    if not txs_mev.empty:
        mev_df = identify_mev_bots_local(txs_mev)
        frames.append(mev_df)
        logger.info("mev_bots_local", count=len(mev_df))
    else:
        logger.info("mev_bots_local", count=0)

    # --- 2. DEX-heavy ---
    logger.info("loading CI columns for DEX detection")
    ci_dex = _load_filtered_columns(
        ci_path,
        columns=["from_address", "to_address"],
        filter_col="from_address",
        filter_values=active_wallets - PROTOCOL_EXCLUSION_SET,
        desc="  DEX: scanning contract_interactions",
    )

    if not ci_dex.empty:
        dex_df = identify_dex_heavy_wallets_local(txs_mev, ci_dex)
        frames.append(dex_df)
        logger.info("dex_heavy_wallets_local", count=len(dex_df))
    else:
        logger.info("dex_heavy_wallets_local", count=0)

    # --- 3. NFT traders ---
    logger.info("detecting NFT traders from contract interactions")
    if not ci_dex.empty:
        nft_df = identify_nft_traders_local(ci_dex)
        frames.append(nft_df)
        logger.info("nft_traders_local", count=len(nft_df))
    else:
        logger.info("nft_traders_local", count=0)

    # --- 4. Smart money ---
    logger.info("loading tx columns for smart money detection")
    txs_sm = _load_filtered_columns(
        txs_path,
        columns=["from_address", "value_eth", "block_timestamp"],
        filter_col="from_address",
        filter_values=active_wallets - PROTOCOL_EXCLUSION_SET,
        desc="  Smart money: scanning transactions",
    )

    if not txs_sm.empty and not ci_dex.empty:
        sm_df = identify_smart_money_local(txs_sm, ci_dex)
        frames.append(sm_df)
        logger.info("smart_money_local", count=len(sm_df))
    else:
        logger.info("smart_money_local", count=0)

    # --- 5. Airdrop hunters ---
    logger.info("detecting airdrop hunters from contract interactions")
    if not ci_dex.empty:
        ah_df = identify_airdrop_hunters_local(ci_dex)
        frames.append(ah_df)
        logger.info("airdrop_hunters_local", count=len(ah_df))
    else:
        logger.info("airdrop_hunters_local", count=0)

    # --- 6. Whale / hodler ---
    logger.info("loading tx columns for whale detection")
    txs_whale = (
        txs_sm
        if not txs_sm.empty
        else _load_filtered_columns(
            txs_path,
            columns=["from_address", "value_eth", "block_timestamp"],
            filter_col="from_address",
            filter_values=active_wallets - PROTOCOL_EXCLUSION_SET,
            desc="  Whale: scanning transactions",
        )
    )

    if not txs_whale.empty:
        whale_df = identify_whale_wallets_local(txs_whale)
        frames.append(whale_df)
        logger.info("whale_wallets_local", count=len(whale_df))
    else:
        logger.info("whale_wallets_local", count=0)

    combined = _deduplicate_labels(frames)

    # --- 7. Label remaining wallets as retail_trader ---
    labeled_addresses = set(combined["address"].str.lower())
    unlabeled = active_wallets - labeled_addresses - PROTOCOL_EXCLUSION_SET
    if unlabeled:
        retail_df = pd.DataFrame({"address": sorted(unlabeled)})
        retail_df["label"] = WalletLabel.RETAIL_TRADER.value
        retail_df["source"] = "default_unlabeled"
        combined = pd.concat([combined, retail_df], ignore_index=True)
        logger.info("retail_trader_default", count=len(retail_df))

    # Add wallet_address column (alias for address) for train.py compatibility
    combined["wallet_address"] = combined["address"]

    logger.info(
        "build_ground_truth_local.done",
        total_wallets=len(combined),
        label_distribution=combined["label"].value_counts().to_dict(),
    )
    return combined


# ---------------------------------------------------------------------------
# Orchestration (BigQuery)
# ---------------------------------------------------------------------------


def build_ground_truth_dataset() -> pd.DataFrame:
    """Run all BigQuery labelling functions and merge into a deduplicated DataFrame.

    Priority when a wallet appears in multiple sources (first wins):
        mev_bot > smart_money > defi_farmer > nft_trader > airdrop_hunter
        > hodler > retail_trader

    Returns
    -------
    pd.DataFrame
        Columns: ``address``, ``label``, ``source``.  Rows are unique by
        ``address``.
    """
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

    combined = _deduplicate_labels(frames)

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
        "--mode",
        choices=["local", "bigquery"],
        default="local",
        help="Labeling mode: 'local' uses parquet files ($0), 'bigquery' queries BQ.",
    )
    parser.add_argument(
        "--data-dir",
        type=pathlib.Path,
        default=pathlib.Path("data/raw"),
        help="Directory with raw parquet files (for --mode local).",
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

    gt_df = build_ground_truth_local(args.data_dir) if args.mode == "local" else build_ground_truth_dataset()

    print(f"\nGround-truth dataset: {len(gt_df)} wallets")
    print("\nLabel distribution:")
    print(gt_df["label"].value_counts().to_string())

    if not args.dry_run:
        out = save_ground_truth(gt_df, args.output)
        print(f"\nSaved to {out}")
    else:
        print("\n(dry-run -- not saved)")
