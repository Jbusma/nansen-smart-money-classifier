"""Tests for feature engineering pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.feature_engineering import (
    FEATURE_COLUMNS,
    activity_regularity,
    avg_holding_duration_estimate,
    burst_score,
    compute_all_features,
    counterparty_concentration,
    dex_to_total_ratio,
    gas_price_sensitivity,
    hour_of_day_entropy,
    impute_missing,
    is_contract,
    lending_to_total_ratio,
    normalize_features,
    preprocess_raw_data,
    tx_frequency_per_day,
    value_velocity,
    weekend_vs_weekday_ratio,
)

WALLET_A = "0x" + "a" * 40


def _make_txs(
    timestamps: list[str] | pd.DatetimeIndex,
    to_addresses: list[str] | None = None,
    values: list[float] | np.ndarray | None = None,
    gas_prices: list[float] | np.ndarray | None = None,
    input_data: list[str] | None = None,
    wallet_address: str = WALLET_A,
) -> pd.DataFrame:
    """Helper to build a transaction DataFrame matching raw BigQuery schema."""
    n = len(timestamps)
    data: dict[str, object] = {
        "block_timestamp": pd.to_datetime(timestamps),
        "from_address": [wallet_address] * n,
        "to_address": to_addresses if to_addresses is not None else ["0x" + f"{i % 16:x}" * 40 for i in range(n)],
        "value": values if values is not None else np.random.exponential(1.0, n),
        "gas_price": gas_prices if gas_prices is not None else np.random.uniform(10, 100, n),
        "wallet_address": [wallet_address] * n,
    }
    if input_data is not None:
        data["input"] = input_data
    return pd.DataFrame(data)


def _make_token_transfers(
    timestamps: list[str] | pd.DatetimeIndex,
    from_addresses: list[str] | None = None,
    to_addresses: list[str] | None = None,
    token_addresses: list[str] | None = None,
    wallet_address: str = WALLET_A,
) -> pd.DataFrame:
    """Helper to build a token_transfers DataFrame matching raw schema."""
    n = len(timestamps)
    return pd.DataFrame(
        {
            "block_timestamp": pd.to_datetime(timestamps),
            "from_address": from_addresses if from_addresses is not None else [wallet_address] * n,
            "to_address": to_addresses if to_addresses is not None else ["0x" + "b" * 40] * n,
            "token_address": token_addresses if token_addresses is not None else ["0x" + "c" * 40] * n,
            "value": np.ones(n),
            "transaction_hash": [f"0x{i:064x}" for i in range(n)],
            "block_number": list(range(n)),
            "wallet_address": [wallet_address] * n,
        }
    )


def _make_contract_interactions(
    timestamps: list[str] | pd.DatetimeIndex,
    to_addresses: list[str] | None = None,
    wallet_address: str = WALLET_A,
) -> pd.DataFrame:
    """Helper to build a contract_interactions DataFrame matching raw schema."""
    n = len(timestamps)
    return pd.DataFrame(
        {
            "block_timestamp": pd.to_datetime(timestamps),
            "from_address": [wallet_address] * n,
            "to_address": to_addresses if to_addresses is not None else ["0x" + "d" * 40] * n,
            "value_eth": np.ones(n),
            "gas_used": np.ones(n) * 21000,
            "input": ["0xabcdef12"] * n,
            "method_id": ["0xabcdef12"] * n,
            "transaction_hash": [f"0x{i:064x}" for i in range(n)],
            "block_number": list(range(n)),
            "wallet_address": [wallet_address] * n,
        }
    )


# ---------------------------------------------------------------------------
# tx_frequency_per_day
# ---------------------------------------------------------------------------


class TestTxFrequencyPerDay:
    def test_empty_returns_zero(self) -> None:
        txs = _make_txs([])
        assert tx_frequency_per_day(txs) == 0.0

    def test_single_day_returns_count(self) -> None:
        txs = _make_txs(["2026-01-01 12:00"] * 5)
        assert tx_frequency_per_day(txs) == 5.0

    def test_multi_day(self) -> None:
        timestamps = pd.date_range("2026-01-01", periods=10, freq="D")
        txs = _make_txs(timestamps)
        freq = tx_frequency_per_day(txs)
        assert freq == pytest.approx(10.0 / 9.0, rel=0.01)


# ---------------------------------------------------------------------------
# activity_regularity
# ---------------------------------------------------------------------------


class TestActivityRegularity:
    def test_empty_returns_zero(self) -> None:
        txs = _make_txs([])
        assert activity_regularity(txs) == 0.0

    def test_single_day_returns_zero(self) -> None:
        txs = _make_txs(["2026-01-01 12:00"] * 5)
        assert activity_regularity(txs) == 0.0

    def test_uneven_distribution_positive_std(self) -> None:
        ts = ["2026-01-01 12:00"] * 10 + ["2026-01-02 12:00"] * 1
        txs = _make_txs(ts)
        assert activity_regularity(txs) > 0.0


# ---------------------------------------------------------------------------
# hour_of_day_entropy
# ---------------------------------------------------------------------------


class TestHourOfDayEntropy:
    def test_uniform_distribution_high_entropy(self) -> None:
        timestamps = pd.date_range("2026-01-01", periods=240, freq="h")
        txs = _make_txs(timestamps)
        entropy = hour_of_day_entropy(txs)
        max_entropy = np.log2(24)
        assert entropy == pytest.approx(max_entropy, abs=0.1)

    def test_single_hour_zero_entropy(self) -> None:
        timestamps = ["2026-01-01 12:00"] * 50
        txs = _make_txs(timestamps)
        entropy = hour_of_day_entropy(txs)
        assert entropy == pytest.approx(0.0, abs=1e-10)

    def test_empty_returns_zero(self) -> None:
        txs = _make_txs([])
        assert hour_of_day_entropy(txs) == 0.0


# ---------------------------------------------------------------------------
# weekend_vs_weekday_ratio
# ---------------------------------------------------------------------------


class TestWeekendWeekdayRatio:
    def test_all_weekday(self) -> None:
        # 2026-01-05 is a Monday
        timestamps = pd.date_range("2026-01-05", periods=5, freq="D")
        txs = _make_txs(timestamps)
        ratio = weekend_vs_weekday_ratio(txs)
        assert ratio == pytest.approx(0.0)

    def test_all_weekend(self) -> None:
        # 2026-01-03 is Saturday, 2026-01-04 is Sunday
        timestamps = pd.to_datetime(["2026-01-03", "2026-01-04"])
        txs = _make_txs(timestamps)
        ratio = weekend_vs_weekday_ratio(txs)
        assert ratio > 0

    def test_empty_returns_zero(self) -> None:
        txs = _make_txs([])
        assert weekend_vs_weekday_ratio(txs) == 0.0


# ---------------------------------------------------------------------------
# avg_holding_duration_estimate
# ---------------------------------------------------------------------------


class TestAvgHoldingDurationEstimate:
    def test_empty_returns_zero(self) -> None:
        tt = _make_token_transfers([])
        assert avg_holding_duration_estimate(tt, WALLET_A) == 0.0

    def test_in_then_out_pattern(self) -> None:
        """Inbound then outbound 24h later should give ~24h duration."""
        wallet = WALLET_A
        other = "0x" + "b" * 40
        token = "0x" + "c" * 40
        tt = pd.DataFrame(
            {
                "block_timestamp": pd.to_datetime(["2026-01-01 00:00", "2026-01-02 00:00"]),
                "from_address": [other, wallet],
                "to_address": [wallet, other],
                "token_address": [token, token],
                "value": [100, 100],
                "transaction_hash": ["0x01", "0x02"],
                "block_number": [1, 2],
                "wallet_address": [wallet, wallet],
            }
        )
        dur = avg_holding_duration_estimate(tt, wallet)
        assert dur == pytest.approx(24.0, rel=0.01)

    def test_no_matching_pattern_returns_zero(self) -> None:
        """All outbound transfers should return 0."""
        wallet = WALLET_A
        other = "0x" + "b" * 40
        token = "0x" + "c" * 40
        tt = pd.DataFrame(
            {
                "block_timestamp": pd.to_datetime(["2026-01-01 00:00", "2026-01-02 00:00"]),
                "from_address": [wallet, wallet],
                "to_address": [other, other],
                "token_address": [token, token],
                "value": [100, 100],
                "transaction_hash": ["0x01", "0x02"],
                "block_number": [1, 2],
                "wallet_address": [wallet, wallet],
            }
        )
        dur = avg_holding_duration_estimate(tt, wallet)
        assert dur == 0.0


# ---------------------------------------------------------------------------
# gas_price_sensitivity
# ---------------------------------------------------------------------------


class TestGasPriceSensitivity:
    def test_empty_returns_zero(self) -> None:
        txs = _make_txs([])
        assert gas_price_sensitivity(txs) == 0.0

    def test_few_days_returns_zero(self) -> None:
        txs = _make_txs(["2026-01-01 12:00", "2026-01-02 12:00"])
        assert gas_price_sensitivity(txs) == 0.0

    def test_returns_finite(self) -> None:
        timestamps = pd.date_range("2026-01-01", periods=30, freq="D")
        txs = _make_txs(timestamps)
        result = gas_price_sensitivity(txs)
        assert np.isfinite(result)


# ---------------------------------------------------------------------------
# is_contract
# ---------------------------------------------------------------------------


class TestIsContract:
    def test_empty_returns_zero(self) -> None:
        txs = _make_txs([])
        assert is_contract(txs) == 0.0

    def test_with_input_data(self) -> None:
        txs = _make_txs(
            ["2026-01-01 12:00"],
            input_data=["0xabcdef1234567890"],
        )
        assert is_contract(txs) == 1.0

    def test_without_input_data(self) -> None:
        txs = _make_txs(
            ["2026-01-01 12:00"],
            input_data=["0x"],
        )
        assert is_contract(txs) == 0.0


# ---------------------------------------------------------------------------
# dex_to_total_ratio
# ---------------------------------------------------------------------------


class TestDexToTotalRatio:
    def test_empty_returns_zero(self) -> None:
        ci = _make_contract_interactions([])
        assert dex_to_total_ratio(ci) == 0.0

    def test_all_dex(self) -> None:
        from src.data.ground_truth import DEX_ROUTER_ADDRESSES

        dex_addr = list(DEX_ROUTER_ADDRESSES)[0]
        ci = _make_contract_interactions(
            ["2026-01-01 12:00"] * 10,
            to_addresses=[dex_addr] * 10,
        )
        assert dex_to_total_ratio(ci) == pytest.approx(1.0)

    def test_no_dex(self) -> None:
        ci = _make_contract_interactions(
            ["2026-01-01 12:00"] * 10,
            to_addresses=["0x" + "e" * 40] * 10,
        )
        assert dex_to_total_ratio(ci) == pytest.approx(0.0)

    def test_partial_dex(self) -> None:
        from src.data.ground_truth import DEX_ROUTER_ADDRESSES

        dex_addr = list(DEX_ROUTER_ADDRESSES)[0]
        non_dex = "0x" + "e" * 40
        ci = _make_contract_interactions(
            ["2026-01-01 12:00"] * 4,
            to_addresses=[dex_addr, non_dex, non_dex, non_dex],
        )
        assert dex_to_total_ratio(ci) == pytest.approx(0.25)


# ---------------------------------------------------------------------------
# lending_to_total_ratio
# ---------------------------------------------------------------------------


class TestLendingToTotalRatio:
    def test_empty_returns_zero(self) -> None:
        ci = _make_contract_interactions([])
        assert lending_to_total_ratio(ci) == 0.0

    def test_all_lending(self) -> None:
        from src.features.feature_engineering import LENDING_PROTOCOL_ADDRESSES

        lending_addr = list(LENDING_PROTOCOL_ADDRESSES)[0]
        ci = _make_contract_interactions(
            ["2026-01-01 12:00"] * 10,
            to_addresses=[lending_addr] * 10,
        )
        assert lending_to_total_ratio(ci) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# counterparty_concentration
# ---------------------------------------------------------------------------


class TestCounterpartyConcentration:
    def test_single_counterparty_max_hhi(self) -> None:
        timestamps = pd.date_range("2026-01-01", periods=100, freq="h")
        to_addrs = ["0x" + "b" * 40] * 100
        txs = _make_txs(timestamps, to_addresses=to_addrs)
        hhi = counterparty_concentration(txs)
        assert hhi == pytest.approx(1.0)

    def test_many_counterparties_low_hhi(self) -> None:
        timestamps = pd.date_range("2026-01-01", periods=100, freq="h")
        to_addrs = ["0x" + f"{i:040x}" for i in range(100)]
        txs = _make_txs(timestamps, to_addresses=to_addrs)
        hhi = counterparty_concentration(txs)
        assert hhi == pytest.approx(0.01, abs=0.001)

    def test_empty_returns_zero(self) -> None:
        txs = _make_txs([])
        assert counterparty_concentration(txs) == 0.0


# ---------------------------------------------------------------------------
# value_velocity
# ---------------------------------------------------------------------------


class TestValueVelocity:
    def test_empty_returns_zero(self) -> None:
        txs = _make_txs([])
        assert value_velocity(txs, WALLET_A) == 0.0

    def test_returns_positive(self) -> None:
        other = "0x" + "b" * 40
        txs = pd.DataFrame(
            {
                "block_timestamp": pd.to_datetime(["2026-01-01 00:00", "2026-01-02 00:00"]),
                "from_address": [other, WALLET_A],
                "to_address": [WALLET_A, other],
                "value": [100.0, 50.0],
                "gas_price": [10.0, 10.0],
                "wallet_address": [WALLET_A, WALLET_A],
            }
        )
        vel = value_velocity(txs, WALLET_A)
        assert vel > 0.0


# ---------------------------------------------------------------------------
# burst_score
# ---------------------------------------------------------------------------


class TestBurstScore:
    def test_uniform_activity_low_burst(self) -> None:
        timestamps = pd.date_range("2026-01-01", periods=100, freq="h")
        txs = _make_txs(timestamps)
        score = burst_score(txs)
        assert score == pytest.approx(1.0, abs=0.5)

    def test_clustered_activity_high_burst(self) -> None:
        clustered = pd.to_datetime(["2026-01-01 12:00:00"] * 50)
        spread = pd.date_range("2026-01-02", periods=50, freq="h")
        all_ts = list(clustered) + list(spread)
        txs = _make_txs(all_ts)
        score = burst_score(txs)
        assert score > 5.0

    def test_empty_returns_zero(self) -> None:
        txs = _make_txs([])
        assert burst_score(txs) == 0.0


# ---------------------------------------------------------------------------
# compute_all_features
# ---------------------------------------------------------------------------


class TestComputeAllFeatures:
    def test_output_has_correct_columns(self) -> None:
        timestamps = pd.date_range("2026-01-01", periods=10, freq="h")
        txs = _make_txs(timestamps)
        tt = _make_token_transfers(timestamps[:3])
        ci = _make_contract_interactions(timestamps[:3])

        result = compute_all_features(txs, tt, ci)

        assert "wallet_address" in result.columns
        for col in FEATURE_COLUMNS:
            assert col in result.columns, f"Missing column: {col}"

    def test_output_has_one_row_per_wallet(self) -> None:
        timestamps = pd.date_range("2026-01-01", periods=10, freq="h")
        txs = _make_txs(timestamps)
        tt = _make_token_transfers(timestamps[:3])
        ci = _make_contract_interactions(timestamps[:3])

        result = compute_all_features(txs, tt, ci)
        assert len(result) == 1

    def test_all_features_are_numeric(self) -> None:
        timestamps = pd.date_range("2026-01-01", periods=10, freq="h")
        txs = _make_txs(timestamps)
        tt = _make_token_transfers(timestamps[:3])
        ci = _make_contract_interactions(timestamps[:3])

        result = compute_all_features(txs, tt, ci)
        for col in FEATURE_COLUMNS:
            assert pd.api.types.is_numeric_dtype(result[col]), f"{col} is not numeric"

    def test_multiple_wallets(self) -> None:
        wallet_b = "0x" + "f" * 40
        timestamps = pd.date_range("2026-01-01", periods=5, freq="h")
        txs_a = _make_txs(timestamps, wallet_address=WALLET_A)
        txs_b = _make_txs(timestamps, wallet_address=wallet_b)
        txs = pd.concat([txs_a, txs_b], ignore_index=True)

        tt_a = _make_token_transfers(timestamps[:2], wallet_address=WALLET_A)
        tt_b = _make_token_transfers(timestamps[:2], wallet_address=wallet_b)
        tt = pd.concat([tt_a, tt_b], ignore_index=True)

        ci_a = _make_contract_interactions(timestamps[:2], wallet_address=WALLET_A)
        ci_b = _make_contract_interactions(timestamps[:2], wallet_address=wallet_b)
        ci = pd.concat([ci_a, ci_b], ignore_index=True)

        result = compute_all_features(txs, tt, ci)
        assert len(result) == 2


# ---------------------------------------------------------------------------
# Normalization & Imputation
# ---------------------------------------------------------------------------


class TestNormalization:
    def test_z_score_properties(self) -> None:
        np.random.seed(42)
        df = pd.DataFrame(
            {
                "feat_a": np.random.normal(100, 15, 1000),
                "feat_b": np.random.normal(0.5, 0.1, 1000),
            }
        )
        normalized = normalize_features(df)
        for col in normalized.columns:
            assert normalized[col].mean() == pytest.approx(0.0, abs=0.1)
            assert normalized[col].std() == pytest.approx(1.0, abs=0.1)


class TestImputation:
    def test_no_nans_after_imputation(self) -> None:
        df = pd.DataFrame(
            {
                "feat_a": [1.0, np.nan, 3.0, 4.0, np.nan],
                "feat_b": [np.nan, 2.0, np.nan, 4.0, 5.0],
            }
        )
        imputed = impute_missing(df)
        assert not imputed.isna().any().any()

    def test_median_imputation(self) -> None:
        df = pd.DataFrame({"feat_a": [1.0, np.nan, 3.0, 5.0, np.nan]})
        imputed = impute_missing(df)
        assert imputed["feat_a"].iloc[1] == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# Feature column canonical list
# ---------------------------------------------------------------------------


class TestFeatureColumns:
    def test_feature_columns_count(self) -> None:
        """Canonical feature list should have exactly 12 columns."""
        assert len(FEATURE_COLUMNS) == 12

    def test_feature_columns_match_spec(self) -> None:
        expected = [
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
        assert expected == FEATURE_COLUMNS


# ---------------------------------------------------------------------------
# preprocess_raw_data
# ---------------------------------------------------------------------------


def _make_raw_txs_bigquery_schema(
    n: int = 5,
    wallet: str = WALLET_A,
) -> pd.DataFrame:
    """Build a transactions DataFrame matching actual BigQuery extract output."""
    other = "0x" + "b" * 40
    return pd.DataFrame(
        {
            "hash": [f"0x{i:064x}" for i in range(n)],
            "block_number": list(range(1000, 1000 + n)),
            "block_timestamp": pd.date_range("2026-01-01", periods=n, freq="h"),
            "from_address": [wallet] * n,
            "to_address": [other] * n,
            "value_eth": [0.1 * (i + 1) for i in range(n)],
            "gas": [21000.0] * n,
            "gas_price": [20.0 + i for i in range(n)],
            "receipt_status": [1.0] * n,
            "method_id": ["0xabcdef12"] * n,
        }
    )


def _make_raw_tt_bigquery_schema(
    n: int = 5,
    wallet: str = WALLET_A,
) -> pd.DataFrame:
    """Build a token_transfers DataFrame matching actual BigQuery extract output."""
    other = "0x" + "c" * 40
    return pd.DataFrame(
        {
            "transaction_hash": [f"0x{i:064x}" for i in range(n)],
            "block_timestamp": pd.date_range("2026-01-01", periods=n, freq="h"),
            "from_address": [other if i % 2 == 0 else wallet for i in range(n)],
            "to_address": [wallet if i % 2 == 0 else other for i in range(n)],
            "token_address": ["0x" + "d" * 40] * n,
            "raw_value": [1000.0] * n,
            "is_erc20": [True] * n,
            "is_erc721": [False] * n,
        }
    )


def _make_raw_ci_bigquery_schema(
    n: int = 5,
    wallet: str = WALLET_A,
) -> pd.DataFrame:
    """Build a contract_interactions DataFrame matching actual BigQuery extract output."""
    return pd.DataFrame(
        {
            "transaction_hash": [f"0xci{i:062x}" for i in range(n)],
            "block_timestamp": pd.date_range("2026-01-01", periods=n, freq="h"),
            "from_address": [wallet] * n,
            "to_address": ["0x" + "e" * 40] * n,
            "trace_type": ["call"] * n,
            "value_eth": [0.1] * n,
            "gas_used": [50000.0] * n,
            "status": [1.0] * n,
            "method_id": ["0xabcdef12"] * n,
            "is_erc20": [False] * n,
            "is_erc721": [False] * n,
        }
    )


class TestPreprocessRawData:
    def test_renames_value_eth_to_value(self) -> None:
        """Transactions should have 'value' column after preprocessing."""
        txs_raw = _make_raw_txs_bigquery_schema()
        tt_raw = _make_raw_tt_bigquery_schema()
        ci_raw = _make_raw_ci_bigquery_schema()

        txs, _tt, _ci = preprocess_raw_data([WALLET_A], txs_raw, tt_raw, ci_raw)
        assert "value" in txs.columns
        assert "value_eth" not in txs.columns

    def test_renames_method_id_to_input(self) -> None:
        """Transactions should have 'input' column after preprocessing."""
        txs_raw = _make_raw_txs_bigquery_schema()
        tt_raw = _make_raw_tt_bigquery_schema()
        ci_raw = _make_raw_ci_bigquery_schema()

        txs, _tt, _ci = preprocess_raw_data([WALLET_A], txs_raw, tt_raw, ci_raw)
        assert "input" in txs.columns

    def test_renames_raw_value_to_value_in_token_transfers(self) -> None:
        """Token transfers should have 'value' column after preprocessing."""
        txs_raw = _make_raw_txs_bigquery_schema()
        tt_raw = _make_raw_tt_bigquery_schema()
        ci_raw = _make_raw_ci_bigquery_schema()

        _txs, tt, _ci = preprocess_raw_data([WALLET_A], txs_raw, tt_raw, ci_raw)
        assert "value" in tt.columns
        assert "raw_value" not in tt.columns

    def test_adds_wallet_address_column(self) -> None:
        """All three DataFrames should have a wallet_address column."""
        txs_raw = _make_raw_txs_bigquery_schema()
        tt_raw = _make_raw_tt_bigquery_schema()
        ci_raw = _make_raw_ci_bigquery_schema()

        txs, tt, ci = preprocess_raw_data([WALLET_A], txs_raw, tt_raw, ci_raw)
        assert "wallet_address" in txs.columns
        assert "wallet_address" in tt.columns
        assert "wallet_address" in ci.columns

    def test_wallet_address_values_are_lowercase(self) -> None:
        """Wallet addresses should be lowercased."""
        mixed_case = "0x" + "A" * 40
        txs_raw = _make_raw_txs_bigquery_schema(wallet=mixed_case)
        tt_raw = _make_raw_tt_bigquery_schema(wallet=mixed_case)
        ci_raw = _make_raw_ci_bigquery_schema(wallet=mixed_case)

        txs, tt, ci = preprocess_raw_data([mixed_case], txs_raw, tt_raw, ci_raw)
        assert all(addr == mixed_case.lower() for addr in txs["wallet_address"])

    def test_no_rename_if_columns_already_correct(self) -> None:
        """If the DataFrame already has 'value' and 'input', don't rename."""
        timestamps = pd.date_range("2026-01-01", periods=3, freq="h")
        txs_correct = pd.DataFrame(
            {
                "block_timestamp": timestamps,
                "from_address": [WALLET_A] * 3,
                "to_address": ["0x" + "b" * 40] * 3,
                "value": [1.0, 2.0, 3.0],
                "input": ["0x", "0xabcd", "0x"],
                "gas_price": [10.0, 20.0, 30.0],
            }
        )
        tt_raw = _make_raw_tt_bigquery_schema()
        ci_raw = _make_raw_ci_bigquery_schema()

        txs, _tt, _ci = preprocess_raw_data([WALLET_A], txs_correct, tt_raw, ci_raw)
        assert "value" in txs.columns
        assert "input" in txs.columns

    def test_end_to_end_with_bigquery_schema(self) -> None:
        """Full pipeline: BigQuery raw -> preprocess -> compute features."""
        txs_raw = _make_raw_txs_bigquery_schema(n=20)
        tt_raw = _make_raw_tt_bigquery_schema(n=10)
        ci_raw = _make_raw_ci_bigquery_schema(n=10)

        txs, tt, ci = preprocess_raw_data([WALLET_A], txs_raw, tt_raw, ci_raw)
        features_df = compute_all_features(txs, tt, ci)

        assert len(features_df) == 1
        assert features_df.iloc[0]["wallet_address"] == WALLET_A.lower()
        for col in FEATURE_COLUMNS:
            assert col in features_df.columns
            assert pd.api.types.is_numeric_dtype(features_df[col])

    def test_multiple_wallets_bigquery_schema(self) -> None:
        """Should handle two wallets appearing in the same raw data."""
        wallet_b = "0x" + "f" * 40
        other = "0x" + "b" * 40

        timestamps = pd.date_range("2026-01-01", periods=10, freq="h")
        txs_raw = pd.DataFrame(
            {
                "hash": [f"0x{i:064x}" for i in range(10)],
                "block_number": list(range(10)),
                "block_timestamp": timestamps,
                "from_address": [WALLET_A] * 5 + [wallet_b] * 5,
                "to_address": [other] * 10,
                "value_eth": [1.0] * 10,
                "gas": [21000.0] * 10,
                "gas_price": [20.0] * 10,
                "receipt_status": [1.0] * 10,
                "method_id": ["0x"] * 10,
            }
        )
        tt_raw = pd.DataFrame(
            {
                "transaction_hash": [f"0xtt{i:061x}" for i in range(4)],
                "block_timestamp": timestamps[:4],
                "from_address": [WALLET_A, other, wallet_b, other],
                "to_address": [other, WALLET_A, other, wallet_b],
                "token_address": ["0x" + "d" * 40] * 4,
                "raw_value": [100.0] * 4,
                "is_erc20": [True] * 4,
                "is_erc721": [False] * 4,
            }
        )
        ci_raw = pd.DataFrame(
            {
                "transaction_hash": [f"0xci{i:062x}" for i in range(6)],
                "block_timestamp": timestamps[:6],
                "from_address": [WALLET_A] * 3 + [wallet_b] * 3,
                "to_address": ["0x" + "e" * 40] * 6,
                "trace_type": ["call"] * 6,
                "value_eth": [0.1] * 6,
                "gas_used": [50000.0] * 6,
                "status": [1.0] * 6,
                "method_id": ["0x"] * 6,
                "is_erc20": [False] * 6,
                "is_erc721": [False] * 6,
            }
        )

        txs, tt, ci = preprocess_raw_data([WALLET_A, wallet_b], txs_raw, tt_raw, ci_raw)
        features_df = compute_all_features(txs, tt, ci)

        assert len(features_df) == 2
        wallets_in_result = set(features_df["wallet_address"])
        assert WALLET_A.lower() in wallets_in_result
        assert wallet_b.lower() in wallets_in_result
