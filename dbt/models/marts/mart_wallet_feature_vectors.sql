{{
  config(
    materialized='table'
  )
}}

with features as (
  select * from {{ ref('int_wallet_behavioral_features') }}
),

z_scored as (
  select
    wallet_address,
    is_contract,
    first_tx_timestamp,
    last_tx_timestamp,

    -- Raw features preserved for reference
    tx_count,
    tx_count_sent,
    tx_count_received,

    -- Z-score normalized features
    (tx_count - avg(tx_count) over ()) / nullif(stddev(tx_count) over (), 0) as tx_count_z,
    (tx_count_sent - avg(tx_count_sent) over ()) / nullif(stddev(tx_count_sent) over (), 0) as tx_count_sent_z,
    (tx_count_received - avg(tx_count_received) over ()) / nullif(stddev(tx_count_received) over (), 0) as tx_count_received_z,
    (total_value_sent_eth - avg(total_value_sent_eth) over ()) / nullif(stddev(total_value_sent_eth) over (), 0) as total_value_sent_eth_z,
    (total_value_received_eth - avg(total_value_received_eth) over ()) / nullif(stddev(total_value_received_eth) over (), 0) as total_value_received_eth_z,
    (avg_value_per_tx - avg(avg_value_per_tx) over ()) / nullif(stddev(avg_value_per_tx) over (), 0) as avg_value_per_tx_z,
    (median_value_per_tx - avg(median_value_per_tx) over ()) / nullif(stddev(median_value_per_tx) over (), 0) as median_value_per_tx_z,
    (active_days - avg(active_days) over ()) / nullif(stddev(active_days) over (), 0) as active_days_z,
    (days_since_first_tx - avg(days_since_first_tx) over ()) / nullif(stddev(days_since_first_tx) over (), 0) as days_since_first_tx_z,
    (unique_counterparties - avg(unique_counterparties) over ()) / nullif(stddev(unique_counterparties) over (), 0) as unique_counterparties_z,
    (avg_gas_price - avg(avg_gas_price) over ()) / nullif(stddev(avg_gas_price) over (), 0) as avg_gas_price_z,
    (total_gas_spent_eth - avg(total_gas_spent_eth) over ()) / nullif(stddev(total_gas_spent_eth) over (), 0) as total_gas_spent_eth_z,
    (unique_tokens_interacted - avg(unique_tokens_interacted) over ()) / nullif(stddev(unique_tokens_interacted) over (), 0) as unique_tokens_interacted_z,
    (erc20_transfer_count - avg(erc20_transfer_count) over ()) / nullif(stddev(erc20_transfer_count) over (), 0) as erc20_transfer_count_z,
    (erc721_transfer_count - avg(erc721_transfer_count) over ()) / nullif(stddev(erc721_transfer_count) over (), 0) as erc721_transfer_count_z,
    (top_token_concentration - avg(top_token_concentration) over ()) / nullif(stddev(top_token_concentration) over (), 0) as top_token_concentration_z,
    (token_diversity_entropy - avg(token_diversity_entropy) over ()) / nullif(stddev(token_diversity_entropy) over (), 0) as token_diversity_entropy_z,
    (unique_contracts_called - avg(unique_contracts_called) over ()) / nullif(stddev(unique_contracts_called) over (), 0) as unique_contracts_called_z,
    (contract_call_count - avg(contract_call_count) over ()) / nullif(stddev(contract_call_count) over (), 0) as contract_call_count_z,
    (dex_interaction_count - avg(dex_interaction_count) over ()) / nullif(stddev(dex_interaction_count) over (), 0) as dex_interaction_count_z,
    (lending_interaction_count - avg(lending_interaction_count) over ()) / nullif(stddev(lending_interaction_count) over (), 0) as lending_interaction_count_z,
    (bridge_interaction_count - avg(bridge_interaction_count) over ()) / nullif(stddev(bridge_interaction_count) over (), 0) as bridge_interaction_count_z,
    (nft_marketplace_interaction_count - avg(nft_marketplace_interaction_count) over ()) / nullif(stddev(nft_marketplace_interaction_count) over (), 0) as nft_marketplace_interaction_count_z,
    (tx_frequency_per_day - avg(tx_frequency_per_day) over ()) / nullif(stddev(tx_frequency_per_day) over (), 0) as tx_frequency_per_day_z,
    (activity_regularity - avg(activity_regularity) over ()) / nullif(stddev(activity_regularity) over (), 0) as activity_regularity_z,
    (hour_of_day_entropy - avg(hour_of_day_entropy) over ()) / nullif(stddev(hour_of_day_entropy) over (), 0) as hour_of_day_entropy_z,
    (weekend_vs_weekday_ratio - avg(weekend_vs_weekday_ratio) over ()) / nullif(stddev(weekend_vs_weekday_ratio) over (), 0) as weekend_vs_weekday_ratio_z,
    (dex_to_total_ratio - avg(dex_to_total_ratio) over ()) / nullif(stddev(dex_to_total_ratio) over (), 0) as dex_to_total_ratio_z,
    (lending_to_total_ratio - avg(lending_to_total_ratio) over ()) / nullif(stddev(lending_to_total_ratio) over (), 0) as lending_to_total_ratio_z,
    (counterparty_concentration - avg(counterparty_concentration) over ()) / nullif(stddev(counterparty_concentration) over (), 0) as counterparty_concentration_z,
    (value_velocity - avg(value_velocity) over ()) / nullif(stddev(value_velocity) over (), 0) as value_velocity_z,
    (burst_score - avg(burst_score) over ()) / nullif(stddev(burst_score) over (), 0) as burst_score_z,

    -- Metadata
    current_timestamp() as feature_computed_at,
    {{ var('sample_window_days', 90) }} as sample_window_days

  from features
)

select * from z_scored
