"""Python-side feature computation supplementing dbt SQL transformations.

Computes derived behavioral features for wallet classification from raw
transaction, token-transfer, and contract-interaction DataFrames.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

# ---------------------------------------------------------------------------
# Individual feature functions
# ---------------------------------------------------------------------------


def tx_frequency_per_day(txs: pd.DataFrame) -> float:
    if txs.empty:
        return 0.0
    dates = pd.to_datetime(txs["block_timestamp"])
    span = (dates.max() - dates.min()).total_seconds() / 86_400
    if span == 0:
        return float(len(txs))
    return len(txs) / span


def activity_regularity(txs: pd.DataFrame) -> float:
    if txs.empty:
        return 0.0
    dates = pd.to_datetime(txs["block_timestamp"]).dt.date
    daily_counts = dates.value_counts()
    return float(daily_counts.std()) if len(daily_counts) > 1 else 0.0


def hour_of_day_entropy(txs: pd.DataFrame) -> float:
    if txs.empty:
        return 0.0
    hours = pd.to_datetime(txs["block_timestamp"]).dt.hour
    counts = hours.value_counts(normalize=True)
    return float(stats.entropy(counts, base=2))


def weekend_vs_weekday_ratio(txs: pd.DataFrame) -> float:
    if txs.empty:
        return 0.0
    dow = pd.to_datetime(txs["block_timestamp"]).dt.dayofweek
    weekend = (dow >= 5).sum()
    weekday = (dow < 5).sum()
    if weekday == 0:
        return float(weekend) if weekend > 0 else 0.0
    return float(weekend / weekday)


def avg_holding_duration_estimate(token_transfers: pd.DataFrame) -> float:
    """Estimate average holding duration from token transfer in/out patterns.

    Uses the time between the first inbound and next outbound transfer for
    each token as a proxy for holding duration.
    """
    if token_transfers.empty:
        return 0.0
    durations: list[float] = []
    ts_col = "block_timestamp"
    for _, group in token_transfers.groupby("token_address"):
        sorted_g = group.sort_values(ts_col)
        timestamps = pd.to_datetime(sorted_g[ts_col])
        directions = sorted_g["direction"]  # "in" / "out"
        last_in: pd.Timestamp | None = None
        for t, d in zip(timestamps, directions, strict=False):
            if d == "in":
                last_in = t
            elif d == "out" and last_in is not None:
                delta = (t - last_in).total_seconds() / 3600.0
                durations.append(delta)
                last_in = None
    return float(np.mean(durations)) if durations else 0.0


def gas_price_sensitivity(txs: pd.DataFrame) -> float:
    if txs.empty or "gas_price" not in txs.columns:
        return 0.0
    dates = pd.to_datetime(txs["block_timestamp"]).dt.date
    daily_counts = txs.groupby(dates).size()
    daily_gas = txs.groupby(dates)["gas_price"].mean()
    merged = pd.DataFrame({"count": daily_counts, "gas": daily_gas}).dropna()
    if len(merged) < 3:
        return 0.0
    corr = merged["count"].corr(merged["gas"])
    return float(corr) if np.isfinite(corr) else 0.0


def is_contract(txs: pd.DataFrame) -> bool:
    if txs.empty:
        return False
    if "is_contract" in txs.columns:
        return bool(txs["is_contract"].iloc[0])
    if "input" in txs.columns:
        return bool((txs["input"].str.len() > 2).any())
    return False


def dex_to_total_ratio(contract_interactions: pd.DataFrame) -> float:
    if contract_interactions.empty:
        return 0.0
    total = len(contract_interactions)
    dex = (contract_interactions["protocol_type"] == "dex").sum()
    return float(dex / total)


def lending_to_total_ratio(contract_interactions: pd.DataFrame) -> float:
    if contract_interactions.empty:
        return 0.0
    total = len(contract_interactions)
    lending = (contract_interactions["protocol_type"] == "lending").sum()
    return float(lending / total)


def counterparty_concentration(txs: pd.DataFrame) -> float:
    if txs.empty:
        return 0.0
    col = "to_address" if "to_address" in txs.columns else None
    if col is None:
        return 0.0
    shares = txs[col].value_counts(normalize=True)
    hhi = float((shares**2).sum())
    return hhi


def value_velocity(txs: pd.DataFrame) -> float:
    """Turnover rate: total outbound volume / max running balance.

    Measures how quickly value flows through a wallet relative to what it
    holds.  A high velocity means the wallet moves funds rapidly rather
    than accumulating.  Falls back to 0 when balance data is unavailable.
    """
    if txs.empty or "value" not in txs.columns:
        return 0.0
    values = txs["value"].astype(float)
    # Estimate a running balance from cumulative in minus out
    if "direction" in txs.columns:
        signs = txs["direction"].map({"in": 1, "out": -1}).fillna(0)
        running = (values * signs).cumsum()
        peak_balance = running.abs().max()
    else:
        # Fallback: use cumulative received as a balance proxy
        peak_balance = values.cumsum().max()
    if peak_balance == 0:
        return 0.0
    total_outbound = values.sum()
    return float(total_outbound / peak_balance)


def burst_score(txs: pd.DataFrame) -> float:
    if txs.empty:
        return 0.0
    timestamps = pd.to_datetime(txs["block_timestamp"])
    hourly = timestamps.dt.floor("h")
    hourly_counts = hourly.value_counts()
    max_hourly = hourly_counts.max()
    avg_hourly = hourly_counts.mean()
    if avg_hourly == 0:
        return 0.0
    return float(max_hourly / avg_hourly)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def compute_all_features(
    wallet_txs_df: pd.DataFrame,
    token_transfers_df: pd.DataFrame,
    contract_interactions_df: pd.DataFrame,
) -> pd.DataFrame:
    """Compute the full derived-feature vector for every wallet.

    Parameters
    ----------
    wallet_txs_df : pd.DataFrame
        Raw transactions with at least ``wallet_address`` and
        ``block_timestamp`` columns.
    token_transfers_df : pd.DataFrame
        Token transfer records with ``wallet_address``, ``token_address``,
        ``block_timestamp``, and ``direction`` columns.
    contract_interactions_df : pd.DataFrame
        Contract interaction records with ``wallet_address`` and
        ``protocol_type`` columns.

    Returns
    -------
    pd.DataFrame
        One row per wallet with all derived features.
    """
    records: list[dict] = []

    wallets = wallet_txs_df["wallet_address"].unique()
    for addr in wallets:
        txs = wallet_txs_df[wallet_txs_df["wallet_address"] == addr]
        tt = token_transfers_df[token_transfers_df["wallet_address"] == addr]
        ci = contract_interactions_df[contract_interactions_df["wallet_address"] == addr]

        records.append(
            {
                "wallet_address": addr,
                "tx_frequency_per_day": tx_frequency_per_day(txs),
                "activity_regularity": activity_regularity(txs),
                "hour_of_day_entropy": hour_of_day_entropy(txs),
                "weekend_vs_weekday_ratio": weekend_vs_weekday_ratio(txs),
                "avg_holding_duration_estimate": avg_holding_duration_estimate(tt),
                "gas_price_sensitivity": gas_price_sensitivity(txs),
                "is_contract": is_contract(txs),
                "dex_to_total_ratio": dex_to_total_ratio(ci),
                "lending_to_total_ratio": lending_to_total_ratio(ci),
                "counterparty_concentration": counterparty_concentration(txs),
                "value_velocity": value_velocity(txs),
                "burst_score": burst_score(txs),
            }
        )

    return pd.DataFrame(records)


def normalize_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply z-score normalization to all numeric feature columns.

    Parameters
    ----------
    df : pd.DataFrame
        Feature DataFrame (output of ``compute_all_features``).

    Returns
    -------
    pd.DataFrame
        Copy of *df* with numeric columns z-score normalized.
    """
    result = df.copy()
    numeric_cols = result.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        std = result[col].std()
        if std and std > 0:
            result[col] = (result[col] - result[col].mean()) / std
        else:
            result[col] = 0.0
    return result


def impute_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Impute missing values: median for continuous, mode for categorical.

    Parameters
    ----------
    df : pd.DataFrame
        Feature DataFrame potentially containing NaN values.

    Returns
    -------
    pd.DataFrame
        Copy of *df* with missing values filled.
    """
    result = df.copy()
    for col in result.columns:
        if result[col].isna().sum() == 0:
            continue
        if pd.api.types.is_numeric_dtype(result[col]):
            result[col] = result[col].fillna(result[col].median())
        else:
            mode_vals = result[col].mode()
            fill = mode_vals.iloc[0] if not mode_vals.empty else "unknown"
            result[col] = result[col].fillna(fill)
    return result
