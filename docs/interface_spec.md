# Pipeline Interface Specification

Canonical schema at each stage of the Smart Money Wallet Behavioral Classifier pipeline.

## Stage 1: BigQuery Raw Extract (data/raw/)

Source: `src/data/bigquery_extract.py`

### active_wallets.parquet
| Column         | Type    | Description                       |
|----------------|---------|-----------------------------------|
| wallet_address | string  | 0x-prefixed Ethereum address      |
| tx_count       | int64   | Total transaction count           |
| total_eth      | float64 | Total ETH transacted              |

### transactions.parquet
| Column          | Type      | Description                          |
|-----------------|-----------|--------------------------------------|
| hash            | string    | Transaction hash                     |
| block_number    | float64   | Block number                         |
| block_timestamp | timestamp | Block timestamp                      |
| from_address    | string    | Sender address                       |
| to_address      | string    | Recipient address                    |
| value_eth       | float64   | Transaction value in ETH             |
| gas             | float64   | Gas limit                            |
| gas_price       | float64   | Gas price (wei)                      |
| receipt_status  | float64   | 1 = success, 0 = fail               |
| method_id       | string    | First 4 bytes of input (selector)    |

### token_transfers.parquet
| Column           | Type      | Description                        |
|------------------|-----------|------------------------------------|
| transaction_hash | string    | Parent transaction hash            |
| block_timestamp  | timestamp | Block timestamp                    |
| from_address     | string    | Sender address                     |
| to_address       | string    | Recipient address                  |
| token_address    | string    | Token contract address             |
| raw_value        | float64   | Token transfer value (raw units)   |
| is_erc20         | bool      | Whether token is ERC-20            |
| is_erc721        | bool      | Whether token is ERC-721           |

### contract_interactions.parquet
| Column           | Type      | Description                        |
|------------------|-----------|------------------------------------|
| transaction_hash | string    | Transaction hash                   |
| block_timestamp  | timestamp | Block timestamp                    |
| from_address     | string    | Caller address                     |
| to_address       | string    | Contract address                   |
| trace_type       | string    | Trace type (e.g. "call")           |
| value_eth        | float64   | ETH value transferred              |
| gas_used         | float64   | Gas used                           |
| status           | float64   | Trace status                       |
| method_id        | string    | First 4 bytes of input (selector)  |
| is_erc20         | bool      | Whether target is ERC-20 contract  |
| is_erc721        | bool      | Whether target is ERC-721 contract |

**Note:** The raw data does NOT contain `direction` or `protocol_type` columns.
These must be derived during feature engineering. The raw columns also use
`value_eth` / `raw_value` / `method_id` rather than `value` / `input`, so
preprocessing renames columns before feature computation.

## Stage 1.5: Preprocessing (column normalization)

Source: `src/features/feature_engineering.py::preprocess_raw_data()`

Before feature computation, the raw DataFrames are normalized:
- **transactions**: `value_eth` -> `value`, `method_id` -> `input`. A `wallet_address` column is added for each wallet from the active_wallets list.
- **token_transfers**: `raw_value` -> `value`. A `wallet_address` column is added.
- **contract_interactions**: No column renames needed (`to_address` is the only required column). A `wallet_address` column is added.

After preprocessing, each DataFrame has a `wallet_address` column that identifies which wallet each row belongs to (a single row may appear in multiple wallets' groups since a transaction has both a sender and a receiver).

## Stage 2: Feature Engineering (12 behavioral features)

Source: `src/features/feature_engineering.py`

Entry point: `compute_all_features(wallet_txs_df, token_transfers_df, contract_interactions_df)`

Input DataFrames must include a `wallet_address` column identifying which wallet each row belongs to.
The function groups by wallet and computes all 12 features per wallet.

Output: One row per wallet with `wallet_address` + 12 derived features.

| Column                        | Type    | Derivation                                                                |
|-------------------------------|---------|---------------------------------------------------------------------------|
| wallet_address                | string  | The wallet being profiled                                                 |
| tx_frequency_per_day          | float64 | len(txs) / date_span_days from transactions                              |
| activity_regularity           | float64 | Std dev of daily tx counts from transactions                              |
| hour_of_day_entropy           | float64 | Shannon entropy (base 2) of tx hour distribution from transactions        |
| weekend_vs_weekday_ratio      | float64 | weekend_tx_count / weekday_tx_count from transactions                     |
| avg_holding_duration_estimate | float64 | Mean hours between first in-transfer and next out-transfer per token      |
| gas_price_sensitivity         | float64 | Correlation between daily tx count and daily mean gas price               |
| is_contract                   | float64 | 1.0 if wallet has contract-like input data, else 0.0 (cast bool->float)  |
| dex_to_total_ratio            | float64 | Fraction of contract interactions targeting known DEX routers             |
| lending_to_total_ratio        | float64 | Fraction of contract interactions targeting known lending protocols        |
| counterparty_concentration    | float64 | HHI of to_address distribution from transactions                         |
| value_velocity                | float64 | Total outbound value / peak cumulative balance from transactions          |
| burst_score                   | float64 | max(hourly_tx_count) / mean(hourly_tx_count) from transactions            |

Key derivation notes:
- `dex_to_total_ratio`: Uses known DEX router addresses (from ground_truth.py `DEX_ROUTER_ADDRESSES`) matched against `to_address` in contract_interactions. No `protocol_type` column exists.
- `lending_to_total_ratio`: Uses known lending protocol addresses (`LENDING_PROTOCOL_ADDRESSES` in feature_engineering.py) matched against `to_address` in contract_interactions.
- `avg_holding_duration_estimate`: Infers direction from whether wallet is `from_address` (out) or `to_address` (in) in token_transfers. No `direction` column exists.
- `value_velocity`: Uses `value` column with direction inferred from `from_address`/`to_address`. No `direction` column exists.
- `is_contract`: Uses `input` column (or `method_id` after renaming). Stored as float64 (0.0/1.0) so the feature matrix is all-numeric.

## Stage 3: ClickHouse Feature Store (wallet_features table)

Sources: `src/data/clickhouse_sync.py` (DDL, sync), `src/features/feature_store.py` (read/write wrapper)

Schema: `wallet_address` (String) + 12 behavioral feature columns (Float64) + `updated_at` (DateTime DEFAULT now()).

| Column                        | ClickHouse Type |
|-------------------------------|-----------------|
| wallet_address                | String          |
| tx_frequency_per_day          | Float64         |
| activity_regularity           | Float64         |
| hour_of_day_entropy           | Float64         |
| weekend_vs_weekday_ratio      | Float64         |
| avg_holding_duration_estimate | Float64         |
| gas_price_sensitivity         | Float64         |
| is_contract                   | Float64         |
| dex_to_total_ratio            | Float64         |
| lending_to_total_ratio        | Float64         |
| counterparty_concentration    | Float64         |
| value_velocity                | Float64         |
| burst_score                   | Float64         |
| updated_at                    | DateTime        |

The canonical column list is defined in three synchronized locations:
- `src/features/feature_engineering.py::FEATURE_COLUMNS` (source of truth)
- `src/data/clickhouse_sync.py::WALLET_FEATURE_COLUMNS` (with wallet_address prefix)
- `src/features/feature_store.py::_FEATURE_COLUMNS` (matches FEATURE_COLUMNS exactly)

## Stage 4: Clustering (UMAP + HDBSCAN)

Source: `src/models/clustering.py`

Input: The 12 numeric feature columns (excluding `wallet_address` and `updated_at`) as a pandas DataFrame.
Output: Cluster labels per wallet (int array, -1 for noise). Persisted via joblib.

Feature order consumed by `ClusteringPipeline.fit()`:
```
tx_frequency_per_day, activity_regularity, hour_of_day_entropy,
weekend_vs_weekday_ratio, avg_holding_duration_estimate, gas_price_sensitivity,
is_contract, dex_to_total_ratio, lending_to_total_ratio,
counterparty_concentration, value_velocity, burst_score
```

HDBSCAN is initialized with `prediction_data=True` to support `approximate_predict` for new wallets.

## Stage 5: Classification (XGBoost + MLP Ensemble)

Source: `src/models/classifier.py`

Input: Same 12 numeric features as clustering, as a numpy float32 array of shape `(n_samples, 12)`.
Output: `(predicted_labels, confidence_scores)` -- integer label indices and float confidences.

The classifier expects `input_dim = 12`.

Training is orchestrated by `src/models/train.py`, which:
1. Loads features from parquet or ClickHouse
2. Filters to the canonical 12 columns using `FEATURE_COLUMNS` from feature_engineering.py
3. Trains XGBoost with Optuna HPO + MLP with cosine annealing
4. Optimizes ensemble weights on validation set

## Stage 6: Serving (FastAPI)

Source: `src/serving/api.py`

The `/classify` endpoint:
1. Looks up features from ClickHouse via `FeatureStore.get_features(wallet_address)`.
2. Extracts the 12 feature values in canonical order via `FeatureStore.get_feature_names()`.
3. Passes the float array to `WalletClassifier.predict()`.
4. Returns label name, confidence, per-class probabilities, and raw features.

Response labels: `smart_money, mev_bot, defi_farmer, airdrop_hunter, retail_trader, hodler, nft_trader`

## Data Flow Summary

```
BigQuery Public Dataset
        |
        v
[bigquery_extract.py] --> data/raw/*.parquet (4 files)
        |
        v
[feature_engineering.py] preprocess_raw_data(wallets, txs, tt, ci)
        |                 --> renames columns, adds wallet_address
        v
[feature_engineering.py] compute_all_features(txs, token_transfers, contract_interactions)
        |                 (requires wallet_address column on each input DataFrame)
        v
DataFrame: wallet_address + 12 features
        |
        +---> [feature_store.py] store_features() --> ClickHouse wallet_features table
        |
        +---> [clickhouse_sync.py] sync_features() --> ClickHouse wallet_features table
        |
        +---> data/features.parquet (optional local cache)
        |
        v
[clustering.py] ClusteringPipeline.fit(features_df[12 cols])
        |
        v
[classifier.py] WalletClassifier.train_xgboost(X, y) + train_mlp(X, y)
        |
        v
[api.py] FeatureStore.get_features() --> classifier.predict() --> response
```

## Canonical Feature Column List

The single source of truth for feature names is `src/features/feature_engineering.py::FEATURE_COLUMNS`:

```python
FEATURE_COLUMNS = [
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
```

All downstream modules (clickhouse_sync, feature_store, train, clustering, classifier, api)
import from or match this list exactly.
