"""Tests for feature engineering pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.feature_engineering import (
    burst_score,
    counterparty_concentration,
    hour_of_day_entropy,
    impute_missing,
    normalize_features,
    weekend_vs_weekday_ratio,
)


def _make_txs(timestamps, to_addresses=None, values=None, gas_prices=None):
    """Helper to build a transaction DataFrame."""
    n = len(timestamps)
    df = pd.DataFrame(
        {
            "block_timestamp": pd.to_datetime(timestamps),
            "from_address": ["0x" + "a" * 40] * n,
            "to_address": to_addresses if to_addresses is not None else ["0x" + f"{i % 16:x}" * 40 for i in range(n)],
            "value": values if values is not None else np.random.exponential(1.0, n),
            "gas_price": gas_prices if gas_prices is not None else np.random.uniform(10, 100, n),
        }
    )
    return df


class TestHourOfDayEntropy:
    def test_uniform_distribution_high_entropy(self):
        """Uniform distribution across hours should give high entropy."""
        # 10 txs per hour for 24 hours = 240 txs
        timestamps = pd.date_range("2026-01-01", periods=240, freq="h")
        txs = _make_txs(timestamps)
        entropy = hour_of_day_entropy(txs)
        # log2(24) ≈ 4.585 — max entropy in base 2
        max_entropy = np.log2(24)
        assert entropy == pytest.approx(max_entropy, abs=0.1)

    def test_single_hour_zero_entropy(self):
        """All transactions in same hour should give zero entropy."""
        timestamps = ["2026-01-01 12:00"] * 50
        txs = _make_txs(timestamps)
        entropy = hour_of_day_entropy(txs)
        assert entropy == pytest.approx(0.0, abs=1e-10)


class TestCounterpartyConcentration:
    def test_single_counterparty_max_hhi(self):
        """Single counterparty → HHI = 1.0."""
        timestamps = pd.date_range("2026-01-01", periods=100, freq="h")
        to_addrs = ["0x" + "b" * 40] * 100
        txs = _make_txs(timestamps, to_addresses=to_addrs)
        hhi = counterparty_concentration(txs)
        assert hhi == pytest.approx(1.0)

    def test_many_counterparties_low_hhi(self):
        """Many unique counterparties → low HHI."""
        timestamps = pd.date_range("2026-01-01", periods=100, freq="h")
        to_addrs = ["0x" + f"{i:040x}" for i in range(100)]
        txs = _make_txs(timestamps, to_addresses=to_addrs)
        hhi = counterparty_concentration(txs)
        assert hhi == pytest.approx(0.01, abs=0.001)


class TestBurstScore:
    def test_uniform_activity_low_burst(self):
        """Evenly spaced transactions should have burst score near 1."""
        timestamps = pd.date_range("2026-01-01", periods=100, freq="h")
        txs = _make_txs(timestamps)
        score = burst_score(txs)
        assert score == pytest.approx(1.0, abs=0.5)

    def test_clustered_activity_high_burst(self):
        """Transactions clustered in one hour should have high burst score."""
        clustered = pd.to_datetime(["2026-01-01 12:00:00"] * 50)
        spread = pd.date_range("2026-01-02", periods=50, freq="h")
        timestamps = clustered.append(spread)
        txs = _make_txs(timestamps)
        score = burst_score(txs)
        assert score > 5.0


class TestWeekendWeekdayRatio:
    def test_all_weekday(self):
        """Only weekday transactions → ratio = 0."""
        # 2026-01-05 is a Monday
        timestamps = pd.date_range("2026-01-05", periods=5, freq="D")
        txs = _make_txs(timestamps)
        ratio = weekend_vs_weekday_ratio(txs)
        assert ratio == pytest.approx(0.0)

    def test_all_weekend(self):
        """Only weekend transactions → ratio > 0."""
        # 2026-01-03 is Saturday, 2026-01-04 is Sunday
        timestamps = pd.to_datetime(["2026-01-03", "2026-01-04"])
        txs = _make_txs(timestamps)
        ratio = weekend_vs_weekday_ratio(txs)
        assert ratio > 0


class TestNormalization:
    def test_z_score_properties(self):
        """Z-score normalized features should have mean ≈ 0, std ≈ 1."""
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
    def test_no_nans_after_imputation(self):
        """Imputed DataFrame should have no NaN values."""
        df = pd.DataFrame(
            {
                "feat_a": [1.0, np.nan, 3.0, 4.0, np.nan],
                "feat_b": [np.nan, 2.0, np.nan, 4.0, 5.0],
            }
        )
        imputed = impute_missing(df)
        assert not imputed.isna().any().any()

    def test_median_imputation(self):
        """Missing values should be replaced with column median."""
        df = pd.DataFrame({"feat_a": [1.0, np.nan, 3.0, 5.0, np.nan]})
        imputed = impute_missing(df)
        assert imputed["feat_a"].iloc[1] == pytest.approx(3.0)
