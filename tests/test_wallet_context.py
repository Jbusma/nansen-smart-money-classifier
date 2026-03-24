"""Tests for wallet context queries and raw data sync infrastructure."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.data.clickhouse_sync import _COLUMN_COERCIONS, RAW_PARQUET_MAP, RAW_TABLES, _coerce_chunk

# ---------------------------------------------------------------------------
# Raw data DDL / schema tests
# ---------------------------------------------------------------------------


class TestRawDataSchema:
    def test_raw_tables_has_three_entries(self) -> None:
        assert len(RAW_TABLES) == 3

    def test_raw_table_names(self) -> None:
        expected = {"raw_transactions", "raw_token_transfers", "raw_contract_interactions"}
        assert set(RAW_TABLES.keys()) == expected

    def test_raw_parquet_map_matches_tables(self) -> None:
        assert set(RAW_PARQUET_MAP.keys()) == set(RAW_TABLES.keys())

    def test_ddl_has_from_address(self) -> None:
        """All raw tables should be queryable by from_address."""
        for name, ddl in RAW_TABLES.items():
            assert "from_address" in ddl, f"{name} DDL missing from_address"

    def test_ddl_has_block_timestamp(self) -> None:
        for name, ddl in RAW_TABLES.items():
            assert "block_timestamp" in ddl, f"{name} DDL missing block_timestamp"

    def test_ddl_ordered_by_from_address(self) -> None:
        """All raw tables should be ordered by from_address for efficient
        per-wallet queries."""
        for name, ddl in RAW_TABLES.items():
            assert "ORDER BY (from_address" in ddl, f"{name} not ordered by from_address"

    def test_transactions_ddl_has_value_eth(self) -> None:
        assert "value_eth" in RAW_TABLES["raw_transactions"]

    def test_token_transfers_ddl_has_token_address(self) -> None:
        assert "token_address" in RAW_TABLES["raw_token_transfers"]

    def test_contract_interactions_ddl_has_method_id(self) -> None:
        assert "method_id" in RAW_TABLES["raw_contract_interactions"]


# ---------------------------------------------------------------------------
# Coercion logic
# ---------------------------------------------------------------------------


class TestCoerceChunk:
    def test_coerces_float_column(self) -> None:
        df = pd.DataFrame({"value_eth": ["1.5", "2.0", None], "from_address": ["a", "b", "c"]})
        result = _coerce_chunk("raw_transactions", df)
        assert result["value_eth"].dtype == np.float64
        assert result["value_eth"].iloc[2] == 0.0  # NaN -> 0

    def test_coerces_bool_to_uint8(self) -> None:
        df = pd.DataFrame(
            {
                "is_erc20": [True, False, True],
                "is_erc721": [False, True, False],
                "from_address": ["a", "b", "c"],
            }
        )
        result = _coerce_chunk("raw_token_transfers", df)
        assert result["is_erc20"].dtype == np.uint8
        assert result["is_erc721"].dtype == np.uint8

    def test_fills_string_nan(self) -> None:
        df = pd.DataFrame({"from_address": ["a", None, "c"]})
        result = _coerce_chunk("raw_transactions", df)
        assert result["from_address"].iloc[1] == ""

    def test_unknown_table_passes_through(self) -> None:
        df = pd.DataFrame({"col": [1, 2, 3]})
        result = _coerce_chunk("nonexistent_table", df)
        assert len(result) == 3

    def test_coercions_dict_covers_all_raw_tables(self) -> None:
        for table in RAW_TABLES:
            assert table in _COLUMN_COERCIONS, f"No coercions for {table}"


# ---------------------------------------------------------------------------
# sync_raw_table (mocked ClickHouse)
# ---------------------------------------------------------------------------


class TestSyncRawTable:
    def test_rejects_unknown_table(self) -> None:
        from src.data.clickhouse_sync import sync_raw_table

        with pytest.raises(ValueError, match="Unknown table"):
            sync_raw_table("nonexistent_table")

    def test_rejects_missing_parquet(self) -> None:
        from src.data.clickhouse_sync import sync_raw_table

        with pytest.raises(FileNotFoundError):
            sync_raw_table("raw_transactions", parquet_path="/nonexistent.parquet")


# ---------------------------------------------------------------------------
# Wallet context query functions (mocked ClickHouse)
# ---------------------------------------------------------------------------


def _mock_query_result(rows: list[list[Any]], columns: list[str]) -> MagicMock:
    """Create a mock ClickHouse query result."""
    result = MagicMock()
    result.result_rows = rows
    result.column_names = columns
    return result


class TestGetTransactionSummary:
    @patch("src.data.wallet_context.get_client")
    def test_returns_zero_for_unknown_wallet(self, mock_get_client: MagicMock) -> None:
        from src.data.wallet_context import get_transaction_summary

        mock_client = MagicMock()
        mock_client.query.return_value = _mock_query_result(
            [[0, 0.0, 0.0, None, None]],
            ["total_transactions", "total_eth_volume", "avg_tx_value_eth", "first_seen", "last_seen"],
        )
        mock_get_client.return_value = mock_client

        result = get_transaction_summary("0x0000000000000000000000000000000000000000")
        assert result["total_transactions"] == 0
        assert result["total_eth_volume"] == 0.0

    @patch("src.data.wallet_context.get_client")
    def test_returns_summary_for_known_wallet(self, mock_get_client: MagicMock) -> None:
        from datetime import datetime

        from src.data.wallet_context import get_transaction_summary

        mock_client = MagicMock()
        first = datetime(2023, 1, 15, 10, 30)
        last = datetime(2024, 3, 20, 14, 22)
        mock_client.query.return_value = _mock_query_result(
            [[1234, 456.78, 0.37, first, last]],
            ["total_transactions", "total_eth_volume", "avg_tx_value_eth", "first_seen", "last_seen"],
        )
        mock_get_client.return_value = mock_client

        result = get_transaction_summary("0x" + "ab" * 20)
        assert result["total_transactions"] == 1234
        assert result["total_eth_volume"] == pytest.approx(456.78)
        assert "2023" in result["first_seen"]


class TestGetTopContracts:
    @patch("src.data.wallet_context.get_client")
    def test_empty_result(self, mock_get_client: MagicMock) -> None:
        from src.data.wallet_context import get_top_contracts

        mock_client = MagicMock()
        mock_client.query.return_value = _mock_query_result([], [])
        mock_get_client.return_value = mock_client

        result = get_top_contracts("0x" + "00" * 20)
        assert result == []

    @patch("src.data.wallet_context.get_client")
    def test_labels_known_protocol(self, mock_get_client: MagicMock) -> None:
        from src.data.ground_truth import UNISWAP_V3_ROUTER
        from src.data.wallet_context import get_top_contracts

        mock_client = MagicMock()
        mock_client.query.return_value = _mock_query_result(
            [[UNISWAP_V3_ROUTER, 150, 42.5, "Uniswap V3: Router", "dex"]],
            ["to_address", "interaction_count", "total_eth", "protocol_label", "protocol_category"],
        )
        mock_get_client.return_value = mock_client

        result = get_top_contracts("0x" + "ab" * 20)
        assert len(result) == 1
        assert result[0]["protocol_label"] == "Uniswap V3: Router"
        assert result[0]["category"] == "dex"
        assert result[0]["interaction_count"] == 150

    @patch("src.data.wallet_context.get_client")
    def test_unknown_contract_has_no_label(self, mock_get_client: MagicMock) -> None:
        from src.data.wallet_context import get_top_contracts

        unknown_addr = "0x" + "ff" * 20
        mock_client = MagicMock()
        mock_client.query.return_value = _mock_query_result(
            [[unknown_addr, 50, 10.0, None, None]],
            ["to_address", "interaction_count", "total_eth", "protocol_label", "protocol_category"],
        )
        mock_get_client.return_value = mock_client

        result = get_top_contracts("0x" + "ab" * 20)
        assert result[0]["protocol_label"] is None
        assert result[0]["category"] == "unknown"


class TestGetTokenActivity:
    @patch("src.data.wallet_context.get_client")
    def test_returns_unique_count_and_top_tokens(self, mock_get_client: MagicMock) -> None:
        from src.data.wallet_context import get_token_activity

        mock_client = MagicMock()
        # First call: unique tokens count
        # Second call: top tokens
        mock_client.query.side_effect = [
            _mock_query_result([[42]], ["unique_tokens"]),
            _mock_query_result(
                [["0x" + "aa" * 20, 200, 200, 0], ["0x" + "bb" * 20, 50, 0, 50]],
                ["token_address", "transfer_count", "erc20_count", "erc721_count"],
            ),
        ]
        mock_get_client.return_value = mock_client

        result = get_token_activity("0x" + "ab" * 20)
        assert result["unique_tokens"] == 42
        assert len(result["top_tokens"]) == 2
        assert result["top_tokens"][0]["transfer_count"] == 200


class TestGetTimingPatterns:
    @patch("src.data.wallet_context.get_client")
    def test_empty_wallet(self, mock_get_client: MagicMock) -> None:
        from src.data.wallet_context import get_timing_patterns

        mock_client = MagicMock()
        mock_client.query.return_value = _mock_query_result([], [])
        mock_get_client.return_value = mock_client

        result = get_timing_patterns("0x" + "00" * 20)
        assert result["most_active_hours"] == []
        assert result["weekday_ratio"] == 0.0
        assert len(result["hourly_distribution"]) == 24

    @patch("src.data.wallet_context.get_client")
    def test_weekday_heavy_wallet(self, mock_get_client: MagicMock) -> None:
        from src.data.wallet_context import get_timing_patterns

        mock_client = MagicMock()
        # 100 txs at 14:00 on Monday (dow=1), 20 at 3:00 on Saturday (dow=6)
        mock_client.query.return_value = _mock_query_result(
            [[14, 1, 100], [3, 6, 20]],
            ["hour", "dow", "cnt"],
        )
        mock_get_client.return_value = mock_client

        result = get_timing_patterns("0x" + "ab" * 20)
        assert result["most_active_hours"][0] == 14
        assert result["weekday_ratio"] == pytest.approx(100 / 120, abs=0.01)
        assert len(result["hourly_distribution"]) == 24
        assert sum(result["hourly_distribution"]) == pytest.approx(1.0, abs=0.01)


class TestGetWalletContext:
    @patch("src.data.wallet_context.get_client")
    def test_returns_all_sections(self, mock_get_client: MagicMock) -> None:
        from src.data.wallet_context import get_wallet_context

        mock_client = MagicMock()
        # Returns empty results for all queries — context should still have all keys
        mock_client.query.return_value = _mock_query_result(
            [[0, 0.0, 0.0, None, None]],
            ["total_transactions", "total_eth_volume", "avg_tx_value_eth", "first_seen", "last_seen"],
        )
        mock_get_client.return_value = mock_client

        ctx = get_wallet_context("0x" + "AB" * 20)
        assert "wallet_address" in ctx
        # Address should be lowercased
        assert ctx["wallet_address"] == "0x" + "ab" * 20
        assert "transaction_summary" in ctx
        assert "top_contracts" in ctx
        assert "token_activity" in ctx
        assert "timing_patterns" in ctx

    @patch("src.data.wallet_context.get_client")
    def test_handles_clickhouse_error_gracefully(self, mock_get_client: MagicMock) -> None:
        from src.data.wallet_context import get_wallet_context

        mock_client = MagicMock()
        mock_client.query.side_effect = ConnectionError("ClickHouse down")
        mock_get_client.return_value = mock_client

        ctx = get_wallet_context("0x" + "ab" * 20)
        # Should not raise, but sections will be None
        assert ctx["wallet_address"] == "0x" + "ab" * 20
        assert ctx["transaction_summary"] is None


# ---------------------------------------------------------------------------
# FastAPI endpoint (mocked context function)
# ---------------------------------------------------------------------------


class TestWalletContextEndpoint:
    @pytest.fixture
    def client(self) -> Any:
        from fastapi.testclient import TestClient

        from src.serving.api import app

        return TestClient(app)

    @patch("src.data.wallet_context.get_wallet_context")
    def test_endpoint_returns_200(self, mock_get_ctx: MagicMock, client: Any) -> None:
        addr = "0x" + "ab" * 20
        mock_get_ctx.return_value = {
            "wallet_address": addr,
            "transaction_summary": {
                "total_transactions": 0,
                "total_eth_volume": 0.0,
                "avg_tx_value_eth": 0.0,
                "first_seen": None,
                "last_seen": None,
            },
            "top_contracts": [],
            "token_activity": {"unique_tokens": 0, "top_tokens": []},
            "timing_patterns": {
                "most_active_hours": [],
                "weekday_ratio": 0.0,
                "hourly_distribution": [0.0] * 24,
            },
        }

        resp = client.get(f"/wallet/{addr}/context")
        assert resp.status_code == 200
        data = resp.json()
        assert data["wallet_address"] == addr

    @patch("src.data.wallet_context.get_wallet_context")
    def test_endpoint_returns_503_on_failure(self, mock_get_ctx: MagicMock, client: Any) -> None:
        mock_get_ctx.side_effect = ConnectionError("down")

        resp = client.get("/wallet/0x" + "ab" * 20 + "/context")
        assert resp.status_code == 503
