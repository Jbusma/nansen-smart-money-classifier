"""Tests for FeatureStore read/write consistency."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.feature_engineering import FEATURE_COLUMNS
from src.features.feature_store import _FEATURE_COLUMNS


class TestFeatureStoreSchema:
    def test_feature_store_columns_match_engineering(self) -> None:
        """Feature store columns must match the canonical engineering list."""
        assert _FEATURE_COLUMNS == FEATURE_COLUMNS

    def test_feature_store_has_12_columns(self) -> None:
        assert len(_FEATURE_COLUMNS) == 12

    def test_store_features_validates_missing_columns(self) -> None:
        """store_features should raise ValueError if columns are missing."""
        from unittest.mock import MagicMock, patch

        with patch("src.features.feature_store.get_client") as mock_get:
            mock_get.return_value = MagicMock()
            from src.features.feature_store import FeatureStore

            store = FeatureStore.__new__(FeatureStore)
            store._database = "test"
            store._client = MagicMock()

            # DataFrame missing required columns
            df = pd.DataFrame({"wallet_address": ["0x" + "a" * 40]})
            with pytest.raises(ValueError, match="missing columns"):
                store.store_features(df)

    def test_store_features_accepts_valid_df(self) -> None:
        """store_features should accept a DataFrame with all required columns."""
        from unittest.mock import MagicMock, patch

        with patch("src.features.feature_store.get_client") as mock_get:
            mock_get.return_value = MagicMock()
            from src.features.feature_store import FeatureStore

            store = FeatureStore.__new__(FeatureStore)
            store._database = "test"
            store._client = MagicMock()

            data: dict[str, list[object]] = {
                "wallet_address": ["0x" + "a" * 40],
            }
            for col in FEATURE_COLUMNS:
                data[col] = [np.random.random()]

            df = pd.DataFrame(data)
            # Should not raise
            store.store_features(df)
            store._client.insert.assert_called_once()

    def test_get_feature_names_returns_canonical_list(self) -> None:
        """get_feature_names should return the 12 canonical columns."""
        from unittest.mock import MagicMock, patch

        with patch("src.features.feature_store.get_client") as mock_get:
            mock_get.return_value = MagicMock()
            from src.features.feature_store import FeatureStore

            store = FeatureStore.__new__(FeatureStore)
            store._database = "test"
            store._client = MagicMock()

            names = store.get_feature_names()
            assert names == FEATURE_COLUMNS
