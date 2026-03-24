"""Query ClickHouse for rich on-chain wallet context.

Provides transaction summaries, top contract interactions (with protocol
labels), token activity, and timing patterns for a given wallet address.
This context feeds into the MCP server so that Claude can generate
protocol-aware insights (e.g. "this wallet interacts heavily with Aave V3
leveraged staking vaults").

Usage:
    from src.data.wallet_context import get_wallet_context
    ctx = get_wallet_context("0xabc...")
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import structlog

from src.config import settings
from src.data.clickhouse_sync import get_client

logger = structlog.get_logger(__name__)


def _query_or_empty(
    client: Any,
    query: str,
    parameters: dict[str, Any],
) -> tuple[list[list[Any]], list[str]]:
    """Execute a query and return (rows, column_names), defaulting to empty."""
    result = client.query(query, parameters=parameters)
    return result.result_rows, result.column_names


def get_transaction_summary(wallet_address: str) -> dict[str, Any]:
    """Aggregate transaction stats for a wallet.

    Returns total count, ETH volume, average value, and first/last seen
    timestamps.
    """
    client = get_client()
    db = settings.clickhouse_database

    rows, _ = _query_or_empty(
        client,
        f"""
        SELECT
            count()                          AS total_transactions,
            coalesce(sum(value_eth), 0)      AS total_eth_volume,
            coalesce(avg(value_eth), 0)      AS avg_tx_value_eth,
            min(block_timestamp)             AS first_seen,
            max(block_timestamp)             AS last_seen
        FROM {db}.raw_transactions
        WHERE from_address = {{addr:String}}
           OR to_address   = {{addr:String}}
        """,
        {"addr": wallet_address},
    )

    if not rows or rows[0][0] == 0:
        return {
            "total_transactions": 0,
            "total_eth_volume": 0.0,
            "avg_tx_value_eth": 0.0,
            "first_seen": None,
            "last_seen": None,
        }

    row = rows[0]
    return {
        "total_transactions": int(row[0]),
        "total_eth_volume": float(row[1]),
        "avg_tx_value_eth": float(row[2]),
        "first_seen": row[3].isoformat() if isinstance(row[3], datetime) else str(row[3]),
        "last_seen": row[4].isoformat() if isinstance(row[4], datetime) else str(row[4]),
    }


def get_top_contracts(
    wallet_address: str,
    *,
    limit: int = 10,
) -> list[dict[str, Any]]:
    """Return the top contracts this wallet has interacted with.

    Each result includes the contract address, interaction count, and a
    human-readable protocol label looked up from the protocol_registry table.
    """
    client = get_client()
    db = settings.clickhouse_database

    rows, _ = _query_or_empty(
        client,
        f"""
        SELECT
            ci.to_address,
            count()                 AS interaction_count,
            sum(ci.value_eth)       AS total_eth,
            pr.label                AS protocol_label,
            pr.category             AS protocol_category
        FROM {db}.raw_contract_interactions ci
        LEFT JOIN (SELECT address, label, category FROM {db}.protocol_registry FINAL) AS pr
            ON ci.to_address = pr.address
        WHERE ci.from_address = {{addr:String}}
        GROUP BY ci.to_address, pr.label, pr.category
        ORDER BY interaction_count DESC
        LIMIT {{lim:UInt32}}
        """,
        {"addr": wallet_address, "lim": limit},
    )

    contracts: list[dict[str, Any]] = []
    for row in rows:
        contracts.append(
            {
                "address": row[0],
                "protocol_label": row[3] or None,
                "category": row[4] or "unknown",
                "interaction_count": int(row[1]),
                "total_eth": float(row[2]),
            }
        )

    return contracts


def get_token_activity(
    wallet_address: str,
    *,
    limit: int = 10,
) -> dict[str, Any]:
    """Summarise token transfer activity for a wallet.

    Returns unique token count and the most frequently transferred tokens.
    """
    client = get_client()
    db = settings.clickhouse_database

    # Unique token count
    count_rows, _ = _query_or_empty(
        client,
        f"""
        SELECT count(DISTINCT token_address) AS unique_tokens
        FROM {db}.raw_token_transfers
        WHERE from_address = {{addr:String}}
           OR to_address   = {{addr:String}}
        """,
        {"addr": wallet_address},
    )
    unique_tokens = int(count_rows[0][0]) if count_rows else 0

    # Top tokens by transfer count
    top_rows, _ = _query_or_empty(
        client,
        f"""
        SELECT
            token_address,
            count()                 AS transfer_count,
            countIf(is_erc20 = 1)   AS erc20_count,
            countIf(is_erc721 = 1)  AS erc721_count
        FROM {db}.raw_token_transfers
        WHERE from_address = {{addr:String}}
           OR to_address   = {{addr:String}}
        GROUP BY token_address
        ORDER BY transfer_count DESC
        LIMIT {{lim:UInt32}}
        """,
        {"addr": wallet_address, "lim": limit},
    )

    top_tokens = [
        {
            "token_address": row[0],
            "transfer_count": int(row[1]),
            "erc20_count": int(row[2]),
            "erc721_count": int(row[3]),
        }
        for row in top_rows
    ]

    return {
        "unique_tokens": unique_tokens,
        "top_tokens": top_tokens,
    }


def get_timing_patterns(wallet_address: str) -> dict[str, Any]:
    """Compute temporal activity patterns for a wallet.

    Returns hourly distribution (24 bins), most active hours, and
    weekday-vs-weekend ratio.
    """
    client = get_client()
    db = settings.clickhouse_database

    rows, _ = _query_or_empty(
        client,
        f"""
        SELECT
            toHour(block_timestamp)         AS hour,
            toDayOfWeek(block_timestamp)    AS dow,
            count()                         AS cnt
        FROM {db}.raw_transactions
        WHERE from_address = {{addr:String}}
        GROUP BY hour, dow
        ORDER BY hour, dow
        """,
        {"addr": wallet_address},
    )

    if not rows:
        return {
            "most_active_hours": [],
            "weekday_ratio": 0.0,
            "hourly_distribution": [0.0] * 24,
        }

    hourly = [0] * 24
    weekday_count = 0
    weekend_count = 0

    for row in rows:
        hour, dow, cnt = int(row[0]), int(row[1]), int(row[2])
        hourly[hour] += cnt
        if dow <= 5:  # Mon-Fri
            weekday_count += cnt
        else:
            weekend_count += cnt

    total = sum(hourly)
    hourly_dist = [h / total if total > 0 else 0.0 for h in hourly]

    # Top 3 most active hours
    sorted_hours = sorted(range(24), key=lambda h: hourly[h], reverse=True)
    most_active = sorted_hours[:3]

    weekday_ratio = weekday_count / total if total > 0 else 0.0

    return {
        "most_active_hours": most_active,
        "weekday_ratio": round(weekday_ratio, 4),
        "hourly_distribution": [round(h, 6) for h in hourly_dist],
    }


def get_wallet_context(wallet_address: str) -> dict[str, Any]:
    """Build a comprehensive on-chain context summary for a wallet.

    Combines transaction summary, top contracts (with protocol labels),
    token activity, and timing patterns into a single dict suitable for
    LLM consumption via the MCP server.
    """
    addr = wallet_address.lower()

    context: dict[str, Any] = {"wallet_address": addr}

    try:
        context["transaction_summary"] = get_transaction_summary(addr)
    except Exception:
        logger.warning("tx_summary_failed", wallet=addr, exc_info=True)
        context["transaction_summary"] = None

    try:
        context["top_contracts"] = get_top_contracts(addr)
    except Exception:
        logger.warning("top_contracts_failed", wallet=addr, exc_info=True)
        context["top_contracts"] = None

    try:
        context["token_activity"] = get_token_activity(addr)
    except Exception:
        logger.warning("token_activity_failed", wallet=addr, exc_info=True)
        context["token_activity"] = None

    try:
        context["timing_patterns"] = get_timing_patterns(addr)
    except Exception:
        logger.warning("timing_patterns_failed", wallet=addr, exc_info=True)
        context["timing_patterns"] = None

    return context
