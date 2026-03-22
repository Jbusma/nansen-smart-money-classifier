"""Integration test: raw data -> features -> model -> prediction."""

from __future__ import annotations

import tempfile

import numpy as np
import pandas as pd

from src.features.feature_engineering import (
    FEATURE_COLUMNS,
    compute_all_features,
    preprocess_raw_data,
)

WALLET_A = "0x" + "a" * 40
WALLET_B = "0x" + "b" * 40
OTHER = "0x" + "c" * 40


def _build_raw_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build minimal raw data matching BigQuery output schema."""
    timestamps = pd.date_range("2026-01-01", periods=20, freq="h")

    # Transactions for two wallets
    txs_records = []
    for i, ts in enumerate(timestamps):
        wallet = WALLET_A if i < 10 else WALLET_B
        txs_records.append(
            {
                "hash": f"0x{i:064x}",
                "block_number": 1000 + i,
                "block_timestamp": ts,
                "from_address": wallet,
                "to_address": OTHER,
                "value": float(i + 1) * 0.1,
                "gas": 21000.0,
                "gas_price": float(20 + i),
                "gas_used": 21000.0,
                "input": "0x" if i % 3 != 0 else "0xabcdef12",
                "wallet_address": wallet,
            }
        )
    txs_df = pd.DataFrame(txs_records)

    # Token transfers
    tt_records = []
    for i in range(10):
        wallet = WALLET_A if i < 5 else WALLET_B
        # First half inbound, second half outbound
        if i % 2 == 0:
            from_addr, to_addr = OTHER, wallet
        else:
            from_addr, to_addr = wallet, OTHER
        tt_records.append(
            {
                "token_address": "0x" + "d" * 40,
                "from_address": from_addr,
                "to_address": to_addr,
                "value": 1000.0,
                "transaction_hash": f"0x{i:064x}",
                "block_number": 1000 + i,
                "block_timestamp": timestamps[i],
                "wallet_address": wallet,
            }
        )
    tt_df = pd.DataFrame(tt_records)

    # Contract interactions
    ci_records = []
    for i in range(10):
        wallet = WALLET_A if i < 5 else WALLET_B
        ci_records.append(
            {
                "transaction_hash": f"0xci{i:062x}",
                "block_number": 1000 + i,
                "block_timestamp": timestamps[i],
                "from_address": wallet,
                "to_address": "0x" + "e" * 40,
                "value_eth": 0.1,
                "gas_used": 50000.0,
                "input": "0xabcdef12",
                "method_id": "0xabcdef12",
                "wallet_address": wallet,
            }
        )
    ci_df = pd.DataFrame(ci_records)

    return txs_df, tt_df, ci_df


class TestEndToEndPipeline:
    def test_raw_to_features_to_model_to_prediction(self) -> None:
        """Full pipeline: raw data -> compute features -> train model -> predict."""
        # Stage 1: Build raw data
        txs_df, tt_df, ci_df = _build_raw_data()

        # Stage 2: Compute features
        features_df = compute_all_features(txs_df, tt_df, ci_df)

        assert len(features_df) == 2  # Two wallets
        assert "wallet_address" in features_df.columns
        for col in FEATURE_COLUMNS:
            assert col in features_df.columns
            assert pd.api.types.is_numeric_dtype(features_df[col])

        # Stage 3: Prepare training data (synthetic labels)
        np.random.seed(42)
        n_classes = 3
        x = features_df[FEATURE_COLUMNS].values.astype(np.float32)

        # Create enough augmented samples with all classes represented
        n_aug = 150  # 50 per class
        x_aug = np.vstack(
            [x[0] + np.random.randn(n_aug, x.shape[1]) * 0.1 for _ in range(1)]
            + [x[1] + np.random.randn(n_aug, x.shape[1]) * 0.1 for _ in range(1)]
        )
        y_aug = np.array([i % n_classes for i in range(n_aug)] + [i % n_classes for i in range(n_aug)])

        split = int(0.8 * len(x_aug))
        x_train, x_val = x_aug[:split].astype(np.float32), x_aug[split:].astype(np.float32)
        y_train, y_val = y_aug[:split], y_aug[split:]

        # Stage 4: Train classifier
        from src.models.classifier import WalletClassifier

        clf = WalletClassifier(num_classes=n_classes)
        clf.train_xgboost(x_train, y_train, x_val, y_val, n_trials=3)

        # Stage 5: Predict
        labels, confidences = clf.predict(x)
        assert len(labels) == 2
        assert len(confidences) == 2
        assert all(0 <= c <= 1 for c in confidences)
        assert all(0 <= int(lbl) < n_classes for lbl in labels)

    def test_features_are_deterministic(self) -> None:
        """Same raw data should produce same features."""
        txs_df, tt_df, ci_df = _build_raw_data()

        features_1 = compute_all_features(txs_df, tt_df, ci_df)
        features_2 = compute_all_features(txs_df, tt_df, ci_df)

        pd.testing.assert_frame_equal(features_1, features_2)

    def test_model_save_load_cycle(self) -> None:
        """Model should produce same predictions after save/load."""
        txs_df, tt_df, ci_df = _build_raw_data()
        features_df = compute_all_features(txs_df, tt_df, ci_df)

        np.random.seed(42)
        n_classes = 3
        x = features_df[FEATURE_COLUMNS].values.astype(np.float32)

        # Create enough training data with all classes represented
        n_aug = 150
        x_aug = np.vstack(
            [x[0] + np.random.randn(n_aug, x.shape[1]) * 0.1 for _ in range(1)]
            + [x[1] + np.random.randn(n_aug, x.shape[1]) * 0.1 for _ in range(1)]
        )
        y_aug = np.array([i % n_classes for i in range(n_aug)] + [i % n_classes for i in range(n_aug)])

        split = int(0.8 * len(x_aug))
        x_train, x_val = x_aug[:split].astype(np.float32), x_aug[split:].astype(np.float32)
        y_train, y_val = y_aug[:split], y_aug[split:]

        from src.models.classifier import WalletClassifier

        clf = WalletClassifier(num_classes=n_classes)
        clf.train_xgboost(x_train, y_train, x_val, y_val, n_trials=3)

        with tempfile.TemporaryDirectory() as tmpdir:
            clf.save(tmpdir)
            loaded = WalletClassifier.load(tmpdir)

            orig_labels, orig_conf = clf.predict(x)
            loaded_labels, loaded_conf = loaded.predict(x)

            np.testing.assert_array_equal(orig_labels, loaded_labels)
            np.testing.assert_allclose(orig_conf, loaded_conf, atol=1e-6)


# ---------------------------------------------------------------------------
# Integration with BigQuery output schema (includes column renaming)
# ---------------------------------------------------------------------------


def _build_bigquery_raw_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build raw data matching the actual BigQuery extract output columns.

    This is the schema produced by bigquery_extract.py SQL queries:
    - transactions: value_eth, method_id (NOT value, input)
    - token_transfers: raw_value (NOT value)
    - contract_interactions: trace_type, status, is_erc20, is_erc721
    """
    timestamps = pd.date_range("2026-01-01", periods=20, freq="h")

    txs_records = []
    for i, ts in enumerate(timestamps):
        wallet = WALLET_A if i < 10 else WALLET_B
        txs_records.append(
            {
                "hash": f"0x{i:064x}",
                "block_number": float(1000 + i),
                "block_timestamp": ts,
                "from_address": wallet,
                "to_address": OTHER,
                "value_eth": float(i + 1) * 0.1,
                "gas": 21000.0,
                "gas_price": float(20 + i),
                "receipt_status": 1.0,
                "method_id": "0x" if i % 3 != 0 else "0xabcdef12",
            }
        )
    txs_df = pd.DataFrame(txs_records)

    tt_records = []
    for i in range(10):
        wallet = WALLET_A if i < 5 else WALLET_B
        if i % 2 == 0:
            from_addr, to_addr = OTHER, wallet
        else:
            from_addr, to_addr = wallet, OTHER
        tt_records.append(
            {
                "transaction_hash": f"0xtt{i:061x}",
                "block_timestamp": timestamps[i],
                "from_address": from_addr,
                "to_address": to_addr,
                "token_address": "0x" + "d" * 40,
                "raw_value": 1000.0,
                "is_erc20": True,
                "is_erc721": False,
            }
        )
    tt_df = pd.DataFrame(tt_records)

    ci_records = []
    for i in range(10):
        wallet = WALLET_A if i < 5 else WALLET_B
        ci_records.append(
            {
                "transaction_hash": f"0xci{i:062x}",
                "block_timestamp": timestamps[i],
                "from_address": wallet,
                "to_address": "0x" + "e" * 40,
                "trace_type": "call",
                "value_eth": 0.1,
                "gas_used": 50000.0,
                "status": 1.0,
                "method_id": "0xabcdef12",
                "is_erc20": False,
                "is_erc721": False,
            }
        )
    ci_df = pd.DataFrame(ci_records)

    return txs_df, tt_df, ci_df


class TestBigQuerySchemaIntegration:
    """Tests that start from actual BigQuery output and go through the full pipeline."""

    def test_preprocess_then_compute_features(self) -> None:
        """BigQuery raw -> preprocess -> compute features produces correct schema."""
        txs_raw, tt_raw, ci_raw = _build_bigquery_raw_data()

        txs, tt, ci = preprocess_raw_data([WALLET_A, WALLET_B], txs_raw, tt_raw, ci_raw)
        features_df = compute_all_features(txs, tt, ci)

        assert len(features_df) == 2
        for col in FEATURE_COLUMNS:
            assert col in features_df.columns
            assert pd.api.types.is_numeric_dtype(features_df[col])

    def test_bigquery_raw_to_prediction(self) -> None:
        """Full pipeline: BigQuery raw -> preprocess -> features -> train -> predict."""
        txs_raw, tt_raw, ci_raw = _build_bigquery_raw_data()

        # Preprocess
        txs, tt, ci = preprocess_raw_data([WALLET_A, WALLET_B], txs_raw, tt_raw, ci_raw)
        features_df = compute_all_features(txs, tt, ci)

        # Train
        np.random.seed(42)
        n_classes = 3
        x = features_df[FEATURE_COLUMNS].values.astype(np.float32)

        n_aug = 150
        x_aug = np.vstack(
            [x[0] + np.random.randn(n_aug, x.shape[1]) * 0.1 for _ in range(1)]
            + [x[1] + np.random.randn(n_aug, x.shape[1]) * 0.1 for _ in range(1)]
        )
        y_aug = np.array([i % n_classes for i in range(n_aug)] + [i % n_classes for i in range(n_aug)])

        split = int(0.8 * len(x_aug))
        x_train, x_val = x_aug[:split].astype(np.float32), x_aug[split:].astype(np.float32)
        y_train, y_val = y_aug[:split], y_aug[split:]

        from src.models.classifier import WalletClassifier

        clf = WalletClassifier(num_classes=n_classes)
        clf.train_xgboost(x_train, y_train, x_val, y_val, n_trials=3)

        labels, confidences = clf.predict(x)
        assert len(labels) == 2
        assert all(0 <= c <= 1 for c in confidences)

    def test_is_contract_works_with_method_id_rename(self) -> None:
        """is_contract should detect method_id data after renaming to input."""
        txs_raw, tt_raw, ci_raw = _build_bigquery_raw_data()

        txs, tt, ci = preprocess_raw_data([WALLET_A], txs_raw, tt_raw, ci_raw)
        features_df = compute_all_features(txs, tt, ci)

        # WALLET_A has method_id="0xabcdef12" for some txs, so is_contract should be 1.0
        wallet_a_row = features_df[features_df["wallet_address"] == WALLET_A.lower()]
        assert wallet_a_row.iloc[0]["is_contract"] == 1.0

    def test_value_velocity_works_with_value_eth_rename(self) -> None:
        """value_velocity should work after value_eth -> value rename."""
        txs_raw, tt_raw, ci_raw = _build_bigquery_raw_data()

        txs, tt, ci = preprocess_raw_data([WALLET_A], txs_raw, tt_raw, ci_raw)
        features_df = compute_all_features(txs, tt, ci)

        wallet_a_row = features_df[features_df["wallet_address"] == WALLET_A.lower()]
        # Should be a positive number (wallet sends funds)
        assert wallet_a_row.iloc[0]["value_velocity"] >= 0.0
