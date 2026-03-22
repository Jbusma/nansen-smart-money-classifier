"""Python-side feature computation supplementing dbt SQL transformations.

Computes derived behavioral features for wallet classification from raw
transaction, token-transfer, and contract-interaction DataFrames.

The 12 canonical behavioral features are:
    tx_frequency_per_day, activity_regularity, hour_of_day_entropy,
    weekend_vs_weekday_ratio, avg_holding_duration_estimate,
    gas_price_sensitivity, is_contract, dex_to_total_ratio,
    lending_to_total_ratio, counterparty_concentration, value_velocity,
    burst_score

Raw data columns available:
  - transactions: hash, block_number, block_timestamp, from_address,
    to_address, value, gas, gas_price, gas_used, input
  - token_transfers: token_address, from_address, to_address, value,
    transaction_hash, block_number, block_timestamp
  - contract_interactions: transaction_hash, block_number, block_timestamp,
    from_address, to_address, value_eth, gas_used, input, method_id

Note: ``direction`` and ``protocol_type`` do NOT exist in the raw data
and are derived here from address comparisons and known-address lookups.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm

from src.data.ground_truth import DEX_ROUTER_ADDRESSES

# Known lending protocol addresses (lowercase)
LENDING_PROTOCOL_ADDRESSES: set[str] = {
    "0x7d2768de32b0b80b7a3454c06bdac94a69ddc7a9",  # Aave V2
    "0x87870bca3f3fd6335c3f4ce8392d69350b4fa4e2",  # Aave V3
    "0x3d9819210a31b4961b30ef54be2aed79b9c9cd3b",  # Compound Comptroller
    "0xc3d688b66703497daa19211eedff47f25384cdc3",  # Compound V3
}

# Canonical feature column names (excluding wallet_address)
FEATURE_COLUMNS: list[str] = [
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


# ---------------------------------------------------------------------------
# Individual feature functions
# ---------------------------------------------------------------------------


def tx_frequency_per_day(txs: pd.DataFrame) -> float:
    """Average number of transactions per day over the activity span."""
    if txs.empty:
        return 0.0
    dates = pd.to_datetime(txs["block_timestamp"])
    span = (dates.max() - dates.min()).total_seconds() / 86_400
    if span == 0:
        return float(len(txs))
    return float(len(txs) / span)


def activity_regularity(txs: pd.DataFrame) -> float:
    """Standard deviation of daily transaction counts."""
    if txs.empty:
        return 0.0
    dates = pd.to_datetime(txs["block_timestamp"]).dt.date
    daily_counts = dates.value_counts()
    return float(daily_counts.std()) if len(daily_counts) > 1 else 0.0


def hour_of_day_entropy(txs: pd.DataFrame) -> float:
    """Shannon entropy (base 2) of the hour-of-day distribution."""
    if txs.empty:
        return 0.0
    hours = pd.to_datetime(txs["block_timestamp"]).dt.hour
    counts = hours.value_counts(normalize=True)
    return float(stats.entropy(counts, base=2))


def weekend_vs_weekday_ratio(txs: pd.DataFrame) -> float:
    """Ratio of weekend to weekday transaction counts."""
    if txs.empty:
        return 0.0
    dow = pd.to_datetime(txs["block_timestamp"]).dt.dayofweek
    weekend = int((dow >= 5).sum())
    weekday = int((dow < 5).sum())
    if weekday == 0:
        return float(weekend) if weekend > 0 else 0.0
    return float(weekend / weekday)


def avg_holding_duration_estimate(
    token_transfers: pd.DataFrame,
    wallet_address: str,
) -> float:
    """Estimate average holding duration from token transfer in/out patterns.

    Direction is inferred from the wallet address:
    - ``to_address == wallet_address`` -> inbound ("in")
    - ``from_address == wallet_address`` -> outbound ("out")

    Uses the time between the first inbound and next outbound transfer for
    each token as a proxy for holding duration (in hours).
    """
    if token_transfers.empty:
        return 0.0
    durations: list[float] = []
    ts_col = "block_timestamp"
    addr_lower = wallet_address.lower()

    for _, group in token_transfers.groupby("token_address"):
        sorted_g = group.sort_values(ts_col)
        timestamps = pd.to_datetime(sorted_g[ts_col])
        last_in: pd.Timestamp | None = None
        for idx, t in timestamps.items():
            row = sorted_g.loc[idx]
            to_addr = str(row.get("to_address", "")).lower()
            from_addr = str(row.get("from_address", "")).lower()
            if to_addr == addr_lower:
                # inbound
                last_in = t
            elif from_addr == addr_lower and last_in is not None:
                # outbound
                delta = (t - last_in).total_seconds() / 3600.0
                durations.append(delta)
                last_in = None
    return float(np.mean(durations)) if durations else 0.0


def gas_price_sensitivity(txs: pd.DataFrame) -> float:
    """Correlation between daily tx count and daily mean gas price."""
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


def is_contract(txs: pd.DataFrame) -> float:
    """Return 1.0 if the wallet appears to be a contract, else 0.0.

    Heuristic: checks if any transaction has input data longer than the
    empty ``0x`` prefix.
    """
    if txs.empty:
        return 0.0
    if "input" in txs.columns:
        has_input = txs["input"].dropna().astype(str).str.len() > 2
        return 1.0 if has_input.any() else 0.0
    return 0.0


def dex_to_total_ratio(
    contract_interactions: pd.DataFrame,
) -> float:
    """Fraction of contract interactions targeting known DEX router addresses.

    Uses ``to_address`` matched against the known DEX router set from
    ``ground_truth.py``.  No ``protocol_type`` column is needed.
    """
    if contract_interactions.empty:
        return 0.0
    if "to_address" not in contract_interactions.columns:
        return 0.0
    total = len(contract_interactions)
    to_addrs = contract_interactions["to_address"].str.lower()
    dex_count = to_addrs.isin(DEX_ROUTER_ADDRESSES).sum()
    return float(dex_count / total)


def lending_to_total_ratio(
    contract_interactions: pd.DataFrame,
) -> float:
    """Fraction of contract interactions targeting known lending protocols.

    Uses ``to_address`` matched against the known lending protocol set.
    No ``protocol_type`` column is needed.
    """
    if contract_interactions.empty:
        return 0.0
    if "to_address" not in contract_interactions.columns:
        return 0.0
    total = len(contract_interactions)
    to_addrs = contract_interactions["to_address"].str.lower()
    lending_count = to_addrs.isin(LENDING_PROTOCOL_ADDRESSES).sum()
    return float(lending_count / total)


def counterparty_concentration(txs: pd.DataFrame) -> float:
    """Herfindahl-Hirschman Index (HHI) of the to_address distribution."""
    if txs.empty:
        return 0.0
    if "to_address" not in txs.columns:
        return 0.0
    shares = txs["to_address"].value_counts(normalize=True)
    hhi = float((shares**2).sum())
    return hhi


def value_velocity(txs: pd.DataFrame, wallet_address: str) -> float:
    """Turnover rate: total outbound volume / max running balance.

    Direction is inferred from ``from_address`` / ``to_address`` relative
    to *wallet_address*.  A high velocity means the wallet moves funds
    rapidly rather than accumulating.
    """
    if txs.empty or "value" not in txs.columns:
        return 0.0
    values = txs["value"].astype(float)
    addr_lower = wallet_address.lower()

    # Derive direction signs: inbound = +1, outbound = -1
    from_addrs = txs["from_address"].str.lower()
    to_addrs = txs["to_address"].str.lower()
    signs = pd.Series(0.0, index=txs.index)
    signs[to_addrs == addr_lower] = 1.0
    signs[from_addrs == addr_lower] = -1.0

    running = (values * signs).cumsum()
    peak_balance = running.abs().max()
    if peak_balance == 0:
        return 0.0
    total_outbound = values[from_addrs == addr_lower].sum()
    if total_outbound == 0:
        total_outbound = values.sum()
    return float(total_outbound / peak_balance)


def burst_score(txs: pd.DataFrame) -> float:
    """Ratio of max hourly tx count to mean hourly tx count."""
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
# Preprocessing: bridge raw BigQuery output to feature computation
# ---------------------------------------------------------------------------


def preprocess_raw_data(
    wallet_addresses: list[str],
    txs_raw: pd.DataFrame,
    token_transfers_raw: pd.DataFrame,
    contract_interactions_raw: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Normalise raw BigQuery extract DataFrames for feature computation.

    The BigQuery extract uses column names that differ from what the feature
    functions expect (e.g. ``value_eth`` instead of ``value``, ``method_id``
    instead of ``input``).  This function:

    1. Renames columns to match feature-function expectations.
    2. Adds a ``wallet_address`` column so that ``compute_all_features``
       can group rows by wallet.

    Parameters
    ----------
    wallet_addresses : list[str]
        Active wallet addresses to include.
    txs_raw : pd.DataFrame
        Raw transactions from BigQuery extract.
    token_transfers_raw : pd.DataFrame
        Raw token transfers from BigQuery extract.
    contract_interactions_raw : pd.DataFrame
        Raw contract interactions (traces) from BigQuery extract.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        ``(transactions, token_transfers, contract_interactions)`` ready for
        ``compute_all_features``.
    """
    addr_set = set(a.lower() for a in wallet_addresses)

    # -- Transactions --
    txs = txs_raw.copy()
    # Rename value_eth -> value (feature functions expect 'value')
    if "value_eth" in txs.columns and "value" not in txs.columns:
        txs = txs.rename(columns={"value_eth": "value"})
    # Rename method_id -> input (is_contract checks 'input' column length)
    if "method_id" in txs.columns and "input" not in txs.columns:
        txs = txs.rename(columns={"method_id": "input"})
    # Assign wallet_address: each row appears for every matching wallet
    from_match = txs["from_address"].str.lower().isin(addr_set)
    to_match = txs["to_address"].str.lower().isin(addr_set)
    parts_txs: list[pd.DataFrame] = []
    if from_match.any():
        from_rows = txs[from_match].copy()
        from_rows["wallet_address"] = from_rows["from_address"].str.lower()
        parts_txs.append(from_rows)
    if to_match.any():
        to_rows = txs[to_match].copy()
        to_rows["wallet_address"] = to_rows["to_address"].str.lower()
        parts_txs.append(to_rows)
    txs_out = pd.concat(parts_txs, ignore_index=True) if parts_txs else txs.iloc[:0].copy()
    if "wallet_address" not in txs_out.columns:
        txs_out["wallet_address"] = pd.Series(dtype="str")

    # -- Token transfers --
    tt = token_transfers_raw.copy()
    if "raw_value" in tt.columns and "value" not in tt.columns:
        tt = tt.rename(columns={"raw_value": "value"})
    from_match_tt = tt["from_address"].str.lower().isin(addr_set)
    to_match_tt = tt["to_address"].str.lower().isin(addr_set)
    parts_tt: list[pd.DataFrame] = []
    if from_match_tt.any():
        from_rows_tt = tt[from_match_tt].copy()
        from_rows_tt["wallet_address"] = from_rows_tt["from_address"].str.lower()
        parts_tt.append(from_rows_tt)
    if to_match_tt.any():
        to_rows_tt = tt[to_match_tt].copy()
        to_rows_tt["wallet_address"] = to_rows_tt["to_address"].str.lower()
        parts_tt.append(to_rows_tt)
    tt_out = pd.concat(parts_tt, ignore_index=True) if parts_tt else tt.iloc[:0].copy()
    if "wallet_address" not in tt_out.columns:
        tt_out["wallet_address"] = pd.Series(dtype="str")
    # Deduplicate: a row where from==to for the same wallet should appear once
    tt_out = tt_out.drop_duplicates()

    # -- Contract interactions --
    ci = contract_interactions_raw.copy()
    from_match_ci = ci["from_address"].str.lower().isin(addr_set)
    to_match_ci = ci["to_address"].str.lower().isin(addr_set)
    parts_ci: list[pd.DataFrame] = []
    if from_match_ci.any():
        from_rows_ci = ci[from_match_ci].copy()
        from_rows_ci["wallet_address"] = from_rows_ci["from_address"].str.lower()
        parts_ci.append(from_rows_ci)
    if to_match_ci.any():
        to_rows_ci = ci[to_match_ci].copy()
        to_rows_ci["wallet_address"] = to_rows_ci["to_address"].str.lower()
        parts_ci.append(to_rows_ci)
    ci_out = pd.concat(parts_ci, ignore_index=True) if parts_ci else ci.iloc[:0].copy()
    if "wallet_address" not in ci_out.columns:
        ci_out["wallet_address"] = pd.Series(dtype="str")
    ci_out = ci_out.drop_duplicates()

    return txs_out, tt_out, ci_out


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
        ``block_timestamp`` columns.  The ``wallet_address`` column
        identifies which wallet each transaction belongs to.
    token_transfers_df : pd.DataFrame
        Token transfer records with ``wallet_address``, ``token_address``,
        ``from_address``, ``to_address``, and ``block_timestamp`` columns.
    contract_interactions_df : pd.DataFrame
        Contract interaction records with ``wallet_address``,
        ``from_address``, ``to_address``, and ``block_timestamp`` columns.

    Returns
    -------
    pd.DataFrame
        One row per wallet with ``wallet_address`` + 12 feature columns.
    """
    records: list[dict] = []

    wallets = wallet_txs_df["wallet_address"].unique()
    for addr in tqdm(wallets, desc="Computing features", unit="wallet"):
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
                "avg_holding_duration_estimate": avg_holding_duration_estimate(tt, addr),
                "gas_price_sensitivity": gas_price_sensitivity(txs),
                "is_contract": is_contract(txs),
                "dex_to_total_ratio": dex_to_total_ratio(ci),
                "lending_to_total_ratio": lending_to_total_ratio(ci),
                "counterparty_concentration": counterparty_concentration(txs),
                "value_velocity": value_velocity(txs, addr),
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
