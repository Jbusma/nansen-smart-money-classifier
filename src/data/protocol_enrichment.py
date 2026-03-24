"""Enrich the protocol_registry table with contract labels from free sources.

Sources (in order of priority):
1. Hardcoded seed — the 17 addresses already in ground_truth.py
2. CoinGecko/Uniswap token list — ~5K ERC-20 token addresses
3. DeFi Llama /protocols — protocol names + categories (token addresses)
4. Etherscan contract API — lazy lookup for top interacted contracts
   (requires ETHERSCAN_API_KEY, free tier: 5 req/sec)

Usage:
    python -m src.data.protocol_enrichment                # all free sources
    python -m src.data.protocol_enrichment --etherscan    # + top-N Etherscan lookups
    python -m src.data.protocol_enrichment --top-n 1000   # resolve top 1000 contracts
"""

from __future__ import annotations

import time

import requests
import structlog
from tqdm import tqdm

from src.config import settings
from src.data.clickhouse_sync import get_client

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

REGISTRY_COLUMNS = ["address", "label", "category", "source"]


def _upsert_registry(rows: list[dict[str, str]]) -> int:
    """Insert rows into protocol_registry (ReplacingMergeTree deduplicates)."""
    if not rows:
        return 0

    client = get_client()
    db = settings.clickhouse_database

    data = [[r["address"], r["label"], r["category"], r["source"]] for r in rows]
    client.insert(
        f"{db}.protocol_registry",
        data,
        column_names=REGISTRY_COLUMNS,
    )
    logger.info("upserted_registry", rows=len(data))
    return len(data)


def _existing_addresses() -> set[str]:
    """Return the set of addresses already in the registry."""
    client = get_client()
    db = settings.clickhouse_database
    result = client.query(f"SELECT DISTINCT address FROM {db}.protocol_registry")
    return {row[0] for row in result.result_rows}


# ---------------------------------------------------------------------------
# Source 1: Hardcoded seed from ground_truth.py
# ---------------------------------------------------------------------------


def seed_from_hardcoded() -> int:
    """Seed the registry with the 17 known addresses from ground_truth.py."""
    from src.data.ground_truth import (
        BRIDGE_ADDRESSES,
        DEX_ROUTER_ADDRESSES,
        KNOWN_PROTOCOL_ADDRESSES,
        LENDING_PROTOCOL_ADDRESSES,
        NFT_MARKETPLACE_ADDRESSES,
    )

    rows: list[dict[str, str]] = []
    for addr, label in KNOWN_PROTOCOL_ADDRESSES.items():
        if addr in DEX_ROUTER_ADDRESSES:
            cat = "dex"
        elif addr in LENDING_PROTOCOL_ADDRESSES:
            cat = "lending"
        elif addr in NFT_MARKETPLACE_ADDRESSES:
            cat = "nft_marketplace"
        elif addr in BRIDGE_ADDRESSES:
            cat = "bridge"
        else:
            cat = "exchange"

        rows.append(
            {
                "address": addr.lower(),
                "label": label,
                "category": cat,
                "source": "hardcoded",
            }
        )

    return _upsert_registry(rows)


# ---------------------------------------------------------------------------
# Source 2: CoinGecko/Uniswap token list
# ---------------------------------------------------------------------------

TOKEN_LIST_URL = "https://tokens.coingecko.com/uniswap/all.json"


def ingest_token_list() -> int:
    """Fetch the CoinGecko/Uniswap token list and insert ERC-20 addresses."""
    logger.info("fetching_token_list", url=TOKEN_LIST_URL)
    resp = requests.get(TOKEN_LIST_URL, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    tokens = data.get("tokens", [])
    existing = _existing_addresses()

    rows: list[dict[str, str]] = []
    for t in tokens:
        if t.get("chainId") != 1:
            continue
        addr = t["address"].lower()
        if addr in existing:
            continue
        rows.append(
            {
                "address": addr,
                "label": f"{t['name']} ({t['symbol']})",
                "category": "token",
                "source": "coingecko_token_list",
            }
        )

    inserted = _upsert_registry(rows)
    logger.info("token_list_done", total_tokens=len(tokens), inserted=inserted)
    return inserted


# ---------------------------------------------------------------------------
# Source 3: DeFi Llama protocols
# ---------------------------------------------------------------------------

DEFILLAMA_URL = "https://api.llama.fi/protocols"

# Map DeFi Llama categories to our simpler taxonomy
_LLAMA_CATEGORY_MAP: dict[str, str] = {
    "Dexs": "dex",
    "DEX Aggregator": "dex",
    "Lending": "lending",
    "CDP": "lending",
    "Liquid Staking": "staking",
    "Staking Pool": "staking",
    "Restaking": "staking",
    "Liquid Restaking": "staking",
    "Bridge": "bridge",
    "Canonical Bridge": "bridge",
    "Yield": "yield",
    "Yield Aggregator": "yield",
    "Farm": "yield",
    "Derivatives": "derivatives",
    "Prediction Market": "derivatives",
    "CEX": "exchange",
}


def ingest_defillama() -> int:
    """Fetch DeFi Llama protocol list and insert token addresses with metadata."""
    logger.info("fetching_defillama", url=DEFILLAMA_URL)
    resp = requests.get(DEFILLAMA_URL, timeout=30)
    resp.raise_for_status()
    protocols = resp.json()

    existing = _existing_addresses()

    rows: list[dict[str, str]] = []
    for p in protocols:
        addr = p.get("address")
        if not addr or not isinstance(addr, str) or not addr.startswith("0x"):
            continue

        # Only include protocols that are on Ethereum
        chains = p.get("chains", [])
        if "Ethereum" not in chains:
            continue

        addr = addr.lower()
        if addr in existing:
            continue

        llama_cat = p.get("category", "")
        category = _LLAMA_CATEGORY_MAP.get(llama_cat, llama_cat.lower().replace(" ", "_"))

        rows.append(
            {
                "address": addr,
                "label": p["name"],
                "category": category,
                "source": "defillama",
            }
        )

    inserted = _upsert_registry(rows)
    logger.info("defillama_done", total_protocols=len(protocols), inserted=inserted)
    return inserted


# ---------------------------------------------------------------------------
# Source 4: Etherscan contract lookup (requires API key)
# ---------------------------------------------------------------------------

ETHERSCAN_API = "https://api.etherscan.io/v2/api"
# Free tier: 5 calls/sec
ETHERSCAN_RATE_LIMIT = 0.21  # seconds between calls (slightly > 1/5)


def _etherscan_get_contract_name(address: str) -> str | None:
    """Look up a contract's source name via Etherscan's getsourcecode API."""
    api_key = settings.etherscan_api_key
    if not api_key:
        return None

    resp = requests.get(
        ETHERSCAN_API,
        params={
            "chainid": "1",
            "module": "contract",
            "action": "getsourcecode",
            "address": address,
            "apikey": api_key,
        },
        timeout=10,
    )
    resp.raise_for_status()
    data = resp.json()

    if data.get("status") != "1" or not data.get("result"):
        return None

    result = data["result"][0]
    name = result.get("ContractName", "").strip()
    return name if name else None


def _get_top_unknown_contracts(top_n: int) -> list[tuple[str, int]]:
    """Return top-N contracts by interaction count that aren't in the registry."""
    client = get_client()
    db = settings.clickhouse_database

    result = client.query(
        f"""
        SELECT ci.to_address, count() AS cnt
        FROM {db}.raw_contract_interactions ci
        LEFT ANTI JOIN {db}.protocol_registry pr
            ON ci.to_address = pr.address
        GROUP BY ci.to_address
        ORDER BY cnt DESC
        LIMIT {{top_n:UInt32}}
        """,
        parameters={"top_n": top_n},
    )
    return [(row[0], int(row[1])) for row in result.result_rows]


def ingest_etherscan(top_n: int = 500) -> int:
    """Resolve the top-N most interacted unknown contracts via Etherscan."""
    api_key = settings.etherscan_api_key
    if not api_key:
        logger.warning("etherscan_skipped", reason="no ETHERSCAN_API_KEY set")
        return 0

    logger.info("fetching_top_unknown_contracts", top_n=top_n)
    unknowns = _get_top_unknown_contracts(top_n)
    logger.info("unknown_contracts_to_resolve", count=len(unknowns))

    rows: list[dict[str, str]] = []
    resolved = 0
    failed = 0

    for addr, _cnt in tqdm(unknowns, desc="Etherscan lookups", unit="addr"):
        try:
            name = _etherscan_get_contract_name(addr)
            if name:
                rows.append(
                    {
                        "address": addr.lower(),
                        "label": name,
                        "category": _infer_category_from_name(name),
                        "source": "etherscan",
                    }
                )
                resolved += 1
            else:
                # Still insert as "unverified" so we don't re-query
                rows.append(
                    {
                        "address": addr.lower(),
                        "label": "Unverified Contract",
                        "category": "unknown",
                        "source": "etherscan",
                    }
                )
                failed += 1
        except Exception:
            logger.warning("etherscan_lookup_failed", address=addr, exc_info=True)
            failed += 1

        time.sleep(ETHERSCAN_RATE_LIMIT)

        # Batch insert every 100 to avoid losing progress
        if len(rows) >= 100:
            _upsert_registry(rows)
            rows = []

    # Final batch
    inserted = _upsert_registry(rows)
    total = resolved + failed
    logger.info(
        "etherscan_done",
        resolved=resolved,
        unverified=failed,
        total=total,
        inserted=inserted,
    )
    return resolved


def _infer_category_from_name(name: str) -> str:
    """Best-effort category from contract name string."""
    lower = name.lower()

    if any(k in lower for k in ("swap", "router", "pair", "pool", "amm", "dex")):
        return "dex"
    if any(k in lower for k in ("lend", "borrow", "aave", "compound", "morpho", "comet")):
        return "lending"
    if any(k in lower for k in ("nft", "erc721", "seaport", "blur", "opensea")):
        return "nft_marketplace"
    if any(k in lower for k in ("bridge", "gateway", "inbox", "l1", "l2")):
        return "bridge"
    if any(k in lower for k in ("stake", "staking", "lido", "reth", "sfrx")):
        return "staking"
    if any(k in lower for k in ("vault", "strategy", "yearn", "harvest", "yield")):
        return "yield"
    if any(k in lower for k in ("erc20", "token")):
        return "token"
    if any(k in lower for k in ("proxy", "upgradeable", "transparent")):
        return "proxy"

    return "other"


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def enrich_all(*, etherscan: bool = False, top_n: int = 500) -> dict[str, int]:
    """Run all enrichment sources and return counts per source."""
    from src.data.clickhouse_sync import PROTOCOL_REGISTRY_DDL, get_client

    # Ensure table exists
    client = get_client()
    db = settings.clickhouse_database
    client.command(f"CREATE DATABASE IF NOT EXISTS {db}")
    client.command(PROTOCOL_REGISTRY_DDL.format(database=db))

    results: dict[str, int] = {}

    results["hardcoded"] = seed_from_hardcoded()
    results["token_list"] = ingest_token_list()
    results["defillama"] = ingest_defillama()

    if etherscan:
        results["etherscan"] = ingest_etherscan(top_n=top_n)

    # Summary
    client = get_client()
    total = client.query(f"SELECT count() FROM {db}.protocol_registry FINAL")
    results["total_registry_size"] = int(total.result_rows[0][0])

    by_source = client.query(
        f"SELECT source, count() FROM {db}.protocol_registry FINAL GROUP BY source ORDER BY count() DESC"
    )
    for row in by_source.result_rows:
        logger.info("registry_by_source", source=row[0], count=int(row[1]))

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Enrich protocol registry from free sources.")
    parser.add_argument(
        "--etherscan",
        action="store_true",
        help="Also resolve top contracts via Etherscan API (requires ETHERSCAN_API_KEY).",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=500,
        help="Number of top unknown contracts to resolve via Etherscan (default: 500).",
    )
    args = parser.parse_args()

    results = enrich_all(etherscan=args.etherscan, top_n=args.top_n)
    for source, count in results.items():
        print(f"  {source}: {count}")
