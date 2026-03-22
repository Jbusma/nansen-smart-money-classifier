"""ClickHouse client wrapper for the feature-store serving path."""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.config import settings

# The clickhouse_sync module is expected to expose ``get_client() -> Client``.
from src.data.clickhouse_sync import get_client

_FEATURE_TABLE = "wallet_features"

# Canonical ordered list of numeric feature columns persisted in ClickHouse.
_FEATURE_COLUMNS: list[str] = [
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


class FeatureStore:
    """Thin read/write layer over the ClickHouse ``wallet_features`` table.

    Parameters
    ----------
    database : str | None
        Override the default ClickHouse database from ``settings``.
    """

    def __init__(self, database: str | None = None) -> None:
        self._database = database or settings.clickhouse_database
        self._client = get_client()

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def get_features(self, wallet_address: str) -> dict[str, Any]:
        """Return the feature vector for a single wallet as a dict.

        Parameters
        ----------
        wallet_address : str
            The ``0x``-prefixed Ethereum address.

        Returns
        -------
        dict[str, Any]
            Feature name -> value mapping.  Empty dict if not found.
        """
        query = f"SELECT * FROM {self._database}.{_FEATURE_TABLE} WHERE wallet_address = %(addr)s LIMIT 1"
        result = self._client.query(query, parameters={"addr": wallet_address})
        if not result.result_rows:
            return {}
        columns = result.column_names
        row = result.result_rows[0]
        return dict(zip(columns, row, strict=False))

    def get_batch_features(self, addresses: list[str]) -> pd.DataFrame:
        """Return feature vectors for multiple wallets.

        Parameters
        ----------
        addresses : list[str]
            Wallet addresses to look up.

        Returns
        -------
        pd.DataFrame
            One row per found address; missing addresses are omitted.
        """
        if not addresses:
            return pd.DataFrame()
        query = f"SELECT * FROM {self._database}.{_FEATURE_TABLE} WHERE wallet_address IN %(addrs)s"
        result = self._client.query(query, parameters={"addrs": addresses})
        if not result.result_rows:
            return pd.DataFrame(columns=["wallet_address"] + _FEATURE_COLUMNS)
        return pd.DataFrame(result.result_rows, columns=result.column_names)

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def store_features(self, df: pd.DataFrame) -> None:
        """Upsert a DataFrame of wallet features into ClickHouse.

        Parameters
        ----------
        df : pd.DataFrame
            Must contain ``wallet_address`` and all feature columns listed
            in ``_FEATURE_COLUMNS``.
        """
        required = {"wallet_address"} | set(_FEATURE_COLUMNS)
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"DataFrame is missing columns: {missing}")

        cols = ["wallet_address"] + _FEATURE_COLUMNS
        subset = df[cols].copy()

        # Replace Python NaN / None with 0 for ClickHouse compatibility.
        subset = subset.fillna(0)

        self._client.insert(
            f"{self._database}.{_FEATURE_TABLE}",
            subset.values.tolist(),
            column_names=cols,
        )

    # ------------------------------------------------------------------
    # Metadata helpers
    # ------------------------------------------------------------------

    def get_feature_names(self) -> list[str]:
        """Return the canonical list of feature column names."""
        return list(_FEATURE_COLUMNS)

    def get_feature_stats(self) -> dict[str, dict[str, float]]:
        """Compute min, max, mean, std per feature across the whole store.

        Returns
        -------
        dict[str, dict[str, float]]
            ``{feature_name: {"min": ..., "max": ..., "mean": ..., "std": ...}}``.
        """
        agg_exprs = ", ".join(
            f"min({c}) AS {c}_min, max({c}) AS {c}_max, avg({c}) AS {c}_mean, stddevPop({c}) AS {c}_std"
            for c in _FEATURE_COLUMNS
        )
        query = f"SELECT {agg_exprs} FROM {self._database}.{_FEATURE_TABLE}"
        result = self._client.query(query)
        if not result.result_rows:
            return {}

        row = result.result_rows[0]
        col_names = result.column_names

        stats: dict[str, dict[str, float]] = {}
        for feat in _FEATURE_COLUMNS:
            stats[feat] = {
                "min": float(row[col_names.index(f"{feat}_min")]),
                "max": float(row[col_names.index(f"{feat}_max")]),
                "mean": float(row[col_names.index(f"{feat}_mean")]),
                "std": float(row[col_names.index(f"{feat}_std")]),
            }
        return stats
