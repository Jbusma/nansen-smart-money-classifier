"""ClickHouse-backed narrative cache with 24-hour TTL."""

from __future__ import annotations

import structlog

from src.data.clickhouse_sync import get_client

logger = structlog.get_logger(__name__)

_CACHE_TABLE_DDL = """
CREATE TABLE IF NOT EXISTS {database}.narrative_cache (
    wallet_address  String,
    narrative       String,
    cluster_id      Int32     DEFAULT -1,
    created_at      DateTime  DEFAULT now(),
    expires_at      DateTime  DEFAULT now() + INTERVAL 24 HOUR
) ENGINE = MergeTree()
ORDER BY (wallet_address, created_at)
TTL expires_at
SETTINGS index_granularity = 8192
"""


class NarrativeCache:
    """Read-through cache storing LLM-generated narratives in ClickHouse.

    Entries expire automatically after 24 hours via the ClickHouse TTL
    mechanism, but an explicit ``cleanup_expired`` helper is provided for
    on-demand purges.
    """

    def __init__(self, database: str = "nansen") -> None:
        self._database = database
        self._client = get_client()
        self._ensure_table()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_table(self) -> None:
        """Create the cache table if it does not already exist."""
        self._client.command(_CACHE_TABLE_DDL.format(database=self._database))
        logger.debug("narrative_cache table ensured", database=self._database)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self, wallet_address: str) -> str | None:
        """Return the cached narrative for *wallet_address*, or ``None``."""
        result = self._client.query(
            """
            SELECT narrative
            FROM {database:Identifier}.narrative_cache
            WHERE wallet_address = {addr:String}
              AND expires_at > now()
            ORDER BY created_at DESC
            LIMIT 1
            """,
            parameters={
                "database": self._database,
                "addr": wallet_address,
            },
        )
        if result.result_rows:
            logger.debug("cache_hit", wallet_address=wallet_address)
            return result.result_rows[0][0]
        logger.debug("cache_miss", wallet_address=wallet_address)
        return None

    def set(
        self,
        wallet_address: str,
        narrative: str,
        cluster_id: int = -1,
    ) -> None:
        """Store a narrative in the cache."""
        self._client.insert(
            f"{self._database}.narrative_cache",
            [[wallet_address, narrative, cluster_id]],
            column_names=["wallet_address", "narrative", "cluster_id"],
        )
        logger.debug(
            "cache_set",
            wallet_address=wallet_address,
            cluster_id=cluster_id,
        )

    def invalidate(self, wallet_address: str | None = None) -> None:
        """Invalidate cached entries.

        If *wallet_address* is given only that entry is removed; otherwise the
        entire cache is flushed.
        """
        if wallet_address is not None:
            self._client.command(
                f"ALTER TABLE {self._database}.narrative_cache DELETE WHERE wallet_address = %(addr)s",
                parameters={"addr": wallet_address},
            )
            logger.info("cache_invalidated", wallet_address=wallet_address)
        else:
            self._client.command(f"TRUNCATE TABLE {self._database}.narrative_cache")
            logger.info("cache_invalidated_all")

    def cleanup_expired(self) -> None:
        """Delete rows whose TTL has passed (supplement to ClickHouse TTL)."""
        self._client.command(f"ALTER TABLE {self._database}.narrative_cache DELETE WHERE expires_at <= now()")
        logger.info("cache_cleanup_expired")
