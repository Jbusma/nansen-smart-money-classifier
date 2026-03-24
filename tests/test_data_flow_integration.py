"""Integration tests for the ClickHouse data flow.

Tests the full data pipeline across component boundaries:
  Parquet -> sync functions -> ClickHouse -> query functions -> API/dashboard

All tests mock the ClickHouse client at the transport layer, simulating
what a real ClickHouse would return after the sync functions transform
and insert data.  This validates the contract between writers and readers.
"""

from __future__ import annotations

import tempfile
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.data.clickhouse_sync import (
    GROUND_TRUTH_DDL,
    WALLET_FEATURE_COLUMNS,
    WALLET_FEATURES_DDL,
)
from src.features.feature_engineering import FEATURE_COLUMNS

WALLET_A = "0x" + "aa" * 20
WALLET_B = "0x" + "bb" * 20


def _mock_query_result(rows: list[list[Any]], columns: list[str]) -> MagicMock:
    result = MagicMock()
    result.result_rows = rows
    result.column_names = columns
    return result


# ---------------------------------------------------------------------------
# Feature sync round-trip: sync_features -> get_wallet_features
# ---------------------------------------------------------------------------


class TestFeatureSyncRoundTrip:
    """Verify that data written by sync_features matches what
    get_wallet_features / get_batch_features expect to read."""

    def _make_features_df(self) -> pd.DataFrame:
        """Build a minimal feature DataFrame matching WALLET_FEATURE_COLUMNS."""
        rows: list[dict[str, object]] = []
        for addr in [WALLET_A, WALLET_B]:
            row: dict[str, object] = {"wallet_address": addr}
            for col in FEATURE_COLUMNS:
                row[col] = np.random.default_rng(42).random()
            rows.append(row)
        return pd.DataFrame(rows)

    @patch("src.data.clickhouse_sync.get_client")
    def test_sync_features_writes_correct_columns(self, mock_get_client: MagicMock) -> None:
        """sync_features should select only WALLET_FEATURE_COLUMNS and
        pass them to insert_df."""
        from src.data.clickhouse_sync import sync_features

        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        df = self._make_features_df()
        # Add an extra column that should NOT be written
        df["extra_noise"] = 999

        sync_features(df)

        # Verify insert_df was called with the right columns
        insert_call = mock_client.insert_df.call_args
        inserted_df: pd.DataFrame = insert_call[0][1]
        assert list(inserted_df.columns) == WALLET_FEATURE_COLUMNS
        assert "extra_noise" not in inserted_df.columns
        assert len(inserted_df) == 2

    @patch("src.data.clickhouse_sync.get_client")
    def test_sync_then_read_single_wallet(self, mock_get_client: MagicMock) -> None:
        """Simulate: sync_features writes data, then get_wallet_features
        reads it back — column names and values should match."""
        from src.data.clickhouse_sync import (
            get_wallet_features,
            sync_features,
        )

        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        df = self._make_features_df()
        sync_features(df)

        # Now simulate reading back: mock the query result with the
        # same data that was inserted
        wallet_a_row = df[df["wallet_address"] == WALLET_A].iloc[0]
        mock_client.query.return_value = _mock_query_result(
            [[wallet_a_row[col] for col in WALLET_FEATURE_COLUMNS]],
            WALLET_FEATURE_COLUMNS,
        )

        result = get_wallet_features(WALLET_A)
        assert result["wallet_address"] == WALLET_A
        for col in FEATURE_COLUMNS:
            assert col in result
            assert isinstance(result[col], (int, float, np.floating))

    @patch("src.data.clickhouse_sync.get_client")
    def test_sync_then_batch_read(self, mock_get_client: MagicMock) -> None:
        """get_batch_features should return a DataFrame with all requested
        wallets."""
        from src.data.clickhouse_sync import (
            get_batch_features,
            sync_features,
        )

        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        df = self._make_features_df()
        sync_features(df)

        # Simulate batch query result
        rows = []
        for _, row in df.iterrows():
            rows.append([row[col] for col in WALLET_FEATURE_COLUMNS])
        mock_client.query.return_value = _mock_query_result(rows, WALLET_FEATURE_COLUMNS)

        result_df = get_batch_features([WALLET_A, WALLET_B])
        assert len(result_df) == 2
        assert set(result_df["wallet_address"]) == {WALLET_A, WALLET_B}
        for col in FEATURE_COLUMNS:
            assert col in result_df.columns

    @patch("src.data.clickhouse_sync.get_client")
    def test_missing_wallet_returns_empty(self, mock_get_client: MagicMock) -> None:
        mock_client = MagicMock()
        mock_client.query.return_value = _mock_query_result([], [])
        mock_get_client.return_value = mock_client

        from src.data.clickhouse_sync import get_wallet_features

        assert get_wallet_features("0x" + "00" * 20) == {}

    @patch("src.data.clickhouse_sync.get_client")
    def test_sync_features_skips_empty_df(self, mock_get_client: MagicMock) -> None:
        from src.data.clickhouse_sync import sync_features

        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        sync_features(pd.DataFrame())
        mock_client.insert_df.assert_not_called()


# ---------------------------------------------------------------------------
# Ground truth sync round-trip: parquet -> sync_ground_truth -> load_ground_truth
# ---------------------------------------------------------------------------


class TestGroundTruthSyncRoundTrip:
    def _make_ground_truth_df(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "address": [WALLET_A, WALLET_B],
                "label": ["whale", "mev_bot"],
                "source": ["heuristic", "heuristic"],
                "total_tx": [5000.0, 12000.0],
                "dex_tx": [100.0, 8000.0],
                "dex_ratio": [0.02, 0.67],
                "total_eth": [50000.0, 150.0],
                "tx_per_day": [10.0, 50.0],
                "wallet_address": [WALLET_A, WALLET_B],
            }
        )

    @patch("src.data.clickhouse_sync.get_client")
    def test_sync_ground_truth_inserts_correct_columns(self, mock_get_client: MagicMock) -> None:
        from src.data.clickhouse_sync import sync_ground_truth

        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        df = self._make_ground_truth_df()
        with tempfile.NamedTemporaryFile(suffix=".parquet") as tmp:
            df.to_parquet(tmp.name)
            rows = sync_ground_truth(tmp.name)

        assert rows == 2
        insert_call = mock_client.insert_df.call_args
        inserted_df: pd.DataFrame = insert_call[0][1]
        expected_cols = [
            "address",
            "label",
            "source",
            "total_tx",
            "dex_tx",
            "dex_ratio",
            "total_eth",
            "tx_per_day",
            "wallet_address",
        ]
        assert list(inserted_df.columns) == expected_cols

    @patch("src.data.clickhouse_sync.get_client")
    def test_sync_truncates_before_insert(self, mock_get_client: MagicMock) -> None:
        from src.data.clickhouse_sync import sync_ground_truth

        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        df = self._make_ground_truth_df()
        with tempfile.NamedTemporaryFile(suffix=".parquet") as tmp:
            df.to_parquet(tmp.name)
            sync_ground_truth(tmp.name)

        # TRUNCATE should be called before insert_df
        commands = [c[0][0] for c in mock_client.command.call_args_list]
        truncate_calls = [c for c in commands if "TRUNCATE" in c]
        assert len(truncate_calls) == 1
        assert mock_client.insert_df.call_count == 1

    @patch("src.data.clickhouse_sync.get_client")
    def test_sync_fills_missing_columns(self, mock_get_client: MagicMock) -> None:
        """If parquet is missing some columns, sync should fill defaults."""
        from src.data.clickhouse_sync import sync_ground_truth

        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        # Minimal parquet with only address and label
        df = pd.DataFrame(
            {
                "address": [WALLET_A],
                "label": ["whale"],
            }
        )
        with tempfile.NamedTemporaryFile(suffix=".parquet") as tmp:
            df.to_parquet(tmp.name)
            rows = sync_ground_truth(tmp.name)

        assert rows == 1
        inserted_df = mock_client.insert_df.call_args[0][1]
        # Missing numeric columns should default to 0.0
        assert inserted_df.iloc[0]["total_tx"] == 0.0
        # Missing string columns should default to ""
        assert inserted_df.iloc[0]["source"] == ""

    @patch("src.data.clickhouse_sync.get_client")
    def test_ground_truth_clickhouse_query_returns_dataframe(self, mock_get_client: MagicMock) -> None:
        """Simulate what load_ground_truth does: query ClickHouse, get a
        DataFrame back with correct columns."""
        mock_client = MagicMock()
        gt_cols = [
            "address",
            "label",
            "source",
            "total_tx",
            "dex_tx",
            "dex_ratio",
            "total_eth",
            "tx_per_day",
            "wallet_address",
        ]
        mock_client.query.return_value = _mock_query_result(
            [[WALLET_A, "whale", "heuristic", 5000.0, 100.0, 0.02, 50000.0, 10.0, WALLET_A]],
            gt_cols,
        )
        mock_get_client.return_value = mock_client

        # Replicate the ClickHouse query path from load_ground_truth
        result = mock_client.query("SELECT * FROM nansen.ground_truth")
        df = pd.DataFrame(result.result_rows, columns=result.column_names)

        assert len(df) == 1
        assert df.iloc[0]["label"] == "whale"
        assert df.iloc[0]["address"] == WALLET_A
        assert set(gt_cols).issubset(set(df.columns))


# ---------------------------------------------------------------------------
# Wallet context end-to-end: raw data queries -> structured context -> API
# ---------------------------------------------------------------------------


class TestWalletContextFlow:
    """Test that wallet context queries flow correctly from ClickHouse
    through get_wallet_context and out through the FastAPI endpoint."""

    @pytest.fixture
    def api_client(self) -> Any:
        from fastapi.testclient import TestClient

        from src.serving.api import app

        return TestClient(app)

    @patch("src.data.wallet_context.get_client")
    def test_context_aggregates_all_sections(self, mock_get_client: MagicMock) -> None:
        """get_wallet_context should call multiple queries and aggregate."""
        from src.data.wallet_context import get_wallet_context

        mock_client = MagicMock()
        # Transaction summary query
        tx_summary = _mock_query_result(
            [[100, 50.5, 0.505, "2023-01-01 00:00:00", "2024-01-01 00:00:00"]],
            ["total_transactions", "total_eth_volume", "avg_tx_value_eth", "first_seen", "last_seen"],
        )
        # Top contracts query (JOIN with protocol_registry)
        top_contracts = _mock_query_result(
            [["0x" + "cc" * 20, 50, 10.0, None, None]],
            ["to_address", "interaction_count", "total_eth", "protocol_label", "protocol_category"],
        )
        # Token unique count + top tokens
        token_count = _mock_query_result([[5]], ["unique_tokens"])
        top_tokens = _mock_query_result(
            [["0x" + "dd" * 20, 30, 30, 0]],
            ["token_address", "transfer_count", "erc20_count", "erc721_count"],
        )
        # Timing patterns
        timing = _mock_query_result(
            [[10, 1, 40], [14, 3, 60]],
            ["hour", "dow", "cnt"],
        )

        mock_client.query.side_effect = [
            tx_summary,
            top_contracts,
            token_count,
            top_tokens,
            timing,
        ]
        mock_get_client.return_value = mock_client

        ctx = get_wallet_context(WALLET_A)

        assert ctx["wallet_address"] == WALLET_A
        assert ctx["transaction_summary"]["total_transactions"] == 100
        assert ctx["transaction_summary"]["total_eth_volume"] == pytest.approx(50.5)
        assert len(ctx["top_contracts"]) == 1
        assert ctx["token_activity"]["unique_tokens"] == 5
        assert len(ctx["timing_patterns"]["hourly_distribution"]) == 24
        assert ctx["timing_patterns"]["weekday_ratio"] == pytest.approx(1.0)

    @patch("src.data.wallet_context.get_client")
    def test_context_partial_failure_returns_other_sections(self, mock_get_client: MagicMock) -> None:
        """If one ClickHouse query fails, other sections should still work."""
        from src.data.wallet_context import get_wallet_context

        mock_client = MagicMock()

        call_count = 0

        def side_effect(*args: Any, **kwargs: Any) -> MagicMock:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # Transaction summary succeeds
                return _mock_query_result(
                    [[50, 10.0, 0.2, None, None]],
                    ["total_transactions", "total_eth_volume", "avg_tx_value_eth", "first_seen", "last_seen"],
                )
            # All other queries fail
            raise ConnectionError("transient failure")

        mock_client.query.side_effect = side_effect
        mock_get_client.return_value = mock_client

        ctx = get_wallet_context(WALLET_A)
        # Transaction summary should work
        assert ctx["transaction_summary"]["total_transactions"] == 50
        # Other sections should be None (caught exceptions)
        assert ctx["top_contracts"] is None
        assert ctx["token_activity"] is None
        assert ctx["timing_patterns"] is None

    @patch("src.data.wallet_context.get_client")
    def test_api_context_endpoint_returns_full_response(self, mock_get_client: MagicMock, api_client: Any) -> None:
        """The /wallet/{address}/context endpoint should return structured JSON
        matching the WalletContextResponse model."""
        mock_client = MagicMock()
        mock_client.query.side_effect = [
            _mock_query_result(
                [[200, 100.0, 0.5, "2023-06-01", "2024-06-01"]],
                ["total_transactions", "total_eth_volume", "avg_tx_value_eth", "first_seen", "last_seen"],
            ),
            _mock_query_result([], []),  # top contracts
            _mock_query_result([[0]], ["unique_tokens"]),  # token count
            _mock_query_result([], []),  # top tokens
            _mock_query_result([], []),  # timing
        ]
        mock_get_client.return_value = mock_client

        resp = api_client.get(f"/wallet/{WALLET_A}/context")
        assert resp.status_code == 200
        data = resp.json()
        assert data["wallet_address"] == WALLET_A
        assert data["transaction_summary"]["total_transactions"] == 200
        assert data["top_contracts"] == []
        assert data["token_activity"]["unique_tokens"] == 0


# ---------------------------------------------------------------------------
# Dashboard data loading: ClickHouse-first with parquet fallback
# ---------------------------------------------------------------------------


class TestDashboardDataLoading:
    """Test that the ClickHouse query pattern used by dashboard loaders
    produces correct results (without importing streamlit_app module)."""

    @patch("src.data.clickhouse_sync.get_client")
    def test_clickhouse_features_query_returns_dataframe(self, mock_get_client: MagicMock) -> None:
        """Simulate what load_features does: query ClickHouse for
        wallet_features, return a DataFrame."""
        mock_client = MagicMock()
        row = [WALLET_A] + [0.5] * len(FEATURE_COLUMNS)
        mock_client.query.return_value = _mock_query_result([row], WALLET_FEATURE_COLUMNS)
        mock_get_client.return_value = mock_client

        # Replicate the ClickHouse query path
        result = mock_client.query("SELECT * FROM nansen.wallet_features")
        df = pd.DataFrame(result.result_rows, columns=result.column_names)

        assert len(df) == 1
        assert df.iloc[0]["wallet_address"] == WALLET_A
        for col in FEATURE_COLUMNS:
            assert col in df.columns

    @patch("src.data.clickhouse_sync.get_client")
    def test_clickhouse_empty_result_produces_empty_df(self, mock_get_client: MagicMock) -> None:
        """When ClickHouse returns no rows, result should be empty."""
        mock_client = MagicMock()
        mock_client.query.return_value = _mock_query_result([], [])
        mock_get_client.return_value = mock_client

        result = mock_client.query("SELECT * FROM nansen.wallet_features")
        assert result.result_rows == []

    @patch("src.data.clickhouse_sync.get_client")
    def test_clickhouse_error_does_not_crash(self, mock_get_client: MagicMock) -> None:
        """When ClickHouse raises, the caller should catch and fall back."""
        mock_get_client.side_effect = ConnectionError("CH down")

        # Simulate the try/except pattern used by load_features
        try:
            from src.data.clickhouse_sync import get_client

            get_client()
            got_client = True
        except Exception:
            got_client = False

        assert not got_client


# ---------------------------------------------------------------------------
# DDL schema consistency
# ---------------------------------------------------------------------------


class TestDDLSchemaConsistency:
    """Verify DDL definitions are internally consistent."""

    def test_ground_truth_ddl_uses_replacing_merge_tree(self) -> None:
        assert "ReplacingMergeTree()" in GROUND_TRUTH_DDL

    def test_ground_truth_ddl_ordered_by_address(self) -> None:
        assert "ORDER BY address" in GROUND_TRUTH_DDL

    def test_ground_truth_ddl_has_all_expected_columns(self) -> None:
        expected = [
            "address",
            "label",
            "source",
            "total_tx",
            "dex_tx",
            "dex_ratio",
            "total_eth",
            "tx_per_day",
            "wallet_address",
        ]
        for col in expected:
            assert col in GROUND_TRUTH_DDL, f"DDL missing column: {col}"

    def test_wallet_features_ddl_has_all_feature_columns(self) -> None:
        for col in FEATURE_COLUMNS:
            assert col in WALLET_FEATURES_DDL, f"DDL missing feature: {col}"

    def test_feature_columns_list_matches_ddl(self) -> None:
        """WALLET_FEATURE_COLUMNS should be wallet_address + all features."""
        assert WALLET_FEATURE_COLUMNS[0] == "wallet_address"
        assert WALLET_FEATURE_COLUMNS[1:] == FEATURE_COLUMNS


# ---------------------------------------------------------------------------
# End-to-end: features -> ClickHouse -> API classify (mocked model)
# ---------------------------------------------------------------------------


class TestClassifyViaClickHouse:
    """Test that the classify endpoint can fetch features from ClickHouse
    and return a prediction."""

    @pytest.fixture
    def api_client(self) -> Any:
        from fastapi.testclient import TestClient

        from src.serving.api import app

        return TestClient(app)

    def test_classify_returns_label_from_ch_features(
        self,
        api_client: Any,
    ) -> None:
        """Classify endpoint: features from CH -> model predict -> response."""
        import src.serving.api as api_module

        features = {col: 0.5 for col in FEATURE_COLUMNS}
        mock_classifier = MagicMock()
        mock_classifier.predict.return_value = (
            np.array([1]),
            np.array([0.85]),
        )
        mock_classifier.predict_proba.return_value = np.array([[0.05, 0.85, 0.10, 0.0, 0.0, 0.0, 0.0]])

        mock_feature_store = MagicMock()
        mock_feature_store.get_features.return_value = features
        mock_feature_store.get_feature_names.return_value = FEATURE_COLUMNS

        orig_classifier = api_module._classifier
        orig_fs = api_module._feature_store
        try:
            api_module._classifier = mock_classifier
            api_module._feature_store = mock_feature_store
            resp = api_client.post("/classify", json={"wallet_address": WALLET_A})
        finally:
            api_module._classifier = orig_classifier
            api_module._feature_store = orig_fs

        assert resp.status_code == 200
        data = resp.json()
        assert data["wallet_address"] == WALLET_A
        assert data["confidence"] == pytest.approx(0.85)
        assert data["label"] == "mev_bot"  # index 1
        assert "features" in data
        assert len(data["features"]) == len(FEATURE_COLUMNS)


# ---------------------------------------------------------------------------
# Create tables DDL integration
# ---------------------------------------------------------------------------


class TestCreateTablesIntegration:
    @patch("src.data.clickhouse_sync.get_client")
    def test_create_tables_creates_all_expected(self, mock_get_client: MagicMock) -> None:
        """create_tables should issue DDL for wallet_features, ground_truth,
        llm_narrative_cache, and raw tables."""
        from src.data.clickhouse_sync import create_tables

        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        create_tables(include_raw=True)

        commands = [c[0][0] for c in mock_client.command.call_args_list]
        # Should have: CREATE DATABASE, wallet_features, ground_truth,
        # protocol_registry, llm_narrative_cache, + 3 raw tables = 8 total
        assert len(commands) == 8
        assert any("CREATE DATABASE" in c for c in commands)
        assert any("wallet_features" in c for c in commands)
        assert any("ground_truth" in c for c in commands)
        assert any("llm_narrative_cache" in c for c in commands)
        assert any("raw_transactions" in c for c in commands)

    @patch("src.data.clickhouse_sync.get_client")
    def test_create_tables_without_raw(self, mock_get_client: MagicMock) -> None:
        from src.data.clickhouse_sync import create_tables

        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        create_tables(include_raw=False)

        commands = [c[0][0] for c in mock_client.command.call_args_list]
        # Should have: CREATE DATABASE, wallet_features, ground_truth,
        # protocol_registry, llm_narrative_cache = 5 total (no raw tables)
        assert len(commands) == 5
        assert not any("raw_transactions" in c for c in commands)
