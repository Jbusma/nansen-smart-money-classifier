"""Tests for ClickHouse sync schema correctness."""

from __future__ import annotations

from src.data.clickhouse_sync import (
    RAW_PARQUET_MAP,
    RAW_TABLES,
    WALLET_FEATURE_COLUMNS,
    WALLET_FEATURES_DDL,
)
from src.features.feature_engineering import FEATURE_COLUMNS


class TestSchemaCorrectness:
    def test_wallet_feature_columns_starts_with_wallet_address(self) -> None:
        assert WALLET_FEATURE_COLUMNS[0] == "wallet_address"

    def test_wallet_feature_columns_has_all_12_features(self) -> None:
        feature_cols = [c for c in WALLET_FEATURE_COLUMNS if c != "wallet_address"]
        assert len(feature_cols) == 12

    def test_wallet_feature_columns_match_feature_engineering(self) -> None:
        """ClickHouse columns (minus wallet_address) must match the
        canonical feature list from feature_engineering."""
        ch_feature_cols = [c for c in WALLET_FEATURE_COLUMNS if c != "wallet_address"]
        assert ch_feature_cols == FEATURE_COLUMNS

    def test_ddl_contains_all_columns(self) -> None:
        """The DDL string should reference every column."""
        for col in WALLET_FEATURE_COLUMNS:
            assert col in WALLET_FEATURES_DDL, f"DDL missing column: {col}"

    def test_ddl_contains_updated_at(self) -> None:
        assert "updated_at" in WALLET_FEATURES_DDL

    def test_no_old_schema_columns(self) -> None:
        """Ensure the old 26-column schema is gone."""
        old_columns = [
            "tx_count",
            "active_days",
            "first_seen",
            "last_seen",
            "account_age_days",
            "total_eth_sent",
            "total_eth_received",
            "net_eth_flow",
            "avg_tx_value_eth",
            "max_tx_value_eth",
        ]
        for col in old_columns:
            assert col not in WALLET_FEATURE_COLUMNS, f"Old column still present: {col}"


class TestRawTableSchemas:
    def test_raw_tables_count(self) -> None:
        assert len(RAW_TABLES) == 3

    def test_parquet_map_matches_raw_tables(self) -> None:
        assert set(RAW_PARQUET_MAP.keys()) == set(RAW_TABLES.keys())

    def test_all_raw_ddl_use_mergetree(self) -> None:
        for name, ddl in RAW_TABLES.items():
            assert "MergeTree()" in ddl, f"{name} does not use MergeTree"

    def test_all_raw_ddl_order_by_from_address(self) -> None:
        for name, ddl in RAW_TABLES.items():
            assert "ORDER BY (from_address" in ddl, f"{name} not ordered by from_address"
