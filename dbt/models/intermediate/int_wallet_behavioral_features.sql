{{
  config(
    materialized='table'
  )
}}

with tx_timestamps as (
  select
    wallet_address,
    block_timestamp
  from (
    select
      from_address as wallet_address,
      block_timestamp
    from {{ source('crypto_ethereum', 'transactions') }}
    where block_timestamp >= timestamp_sub(current_timestamp(), interval {{ var('sample_window_days', 90) }} day)
      and receipt_status = 1
      and from_address is not null

    union all

    select
      to_address as wallet_address,
      block_timestamp
    from {{ source('crypto_ethereum', 'transactions') }}
    where block_timestamp >= timestamp_sub(current_timestamp(), interval {{ var('sample_window_days', 90) }} day)
      and receipt_status = 1
      and to_address is not null
  )
),

-- Hour-of-day entropy per wallet
hour_distribution as (
  select
    wallet_address,
    extract(hour from block_timestamp) as tx_hour,
    count(*) as hour_count
  from tx_timestamps
  group by wallet_address, tx_hour
),

hour_entropy as (
  select
    wallet_address,
    -1.0 * sum(p * ln(p)) as hour_of_day_entropy
  from (
    select
      hd.wallet_address,
      hd.hour_count / sum(hd.hour_count) over (partition by hd.wallet_address) as p
    from hour_distribution hd
  )
  where p > 0
  group by wallet_address
),

-- Weekend vs weekday ratio
day_type_stats as (
  select
    wallet_address,
    countif(extract(dayofweek from block_timestamp) in (1, 7)) as weekend_tx_count,
    countif(extract(dayofweek from block_timestamp) not in (1, 7)) as weekday_tx_count
  from tx_timestamps
  group by wallet_address
),

-- Counterparty concentration (HHI) from sent transactions
counterparty_shares as (
  select
    from_address as wallet_address,
    to_address,
    count(*) as pair_count
  from {{ source('crypto_ethereum', 'transactions') }}
  where block_timestamp >= timestamp_sub(current_timestamp(), interval {{ var('sample_window_days', 90) }} day)
    and receipt_status = 1
    and from_address is not null
    and to_address is not null
  group by from_address, to_address
),

counterparty_hhi as (
  select
    wallet_address,
    sum(pow(pair_count / total_count, 2)) as counterparty_concentration
  from (
    select
      cs.wallet_address,
      cs.pair_count,
      sum(cs.pair_count) over (partition by cs.wallet_address) as total_count
    from counterparty_shares cs
  )
  group by wallet_address
),

-- Burst score: max hourly tx count / avg hourly tx count
hourly_tx_counts as (
  select
    wallet_address,
    timestamp_trunc(block_timestamp, hour) as tx_hour,
    count(*) as hourly_count
  from tx_timestamps
  group by wallet_address, tx_hour
),

burst_stats as (
  select
    wallet_address,
    max(hourly_count) / nullif(avg(hourly_count), 0) as burst_score
  from hourly_tx_counts
  group by wallet_address
),

-- Is contract flag
contracts as (
  select
    address as wallet_address,
    true as is_contract
  from {{ source('crypto_ethereum', 'contracts') }}
  where address is not null
),

-- Join all staging models and derived features
joined as (
  select
    wt.wallet_address,

    -- Transaction features
    wt.tx_count,
    wt.tx_count_sent,
    wt.tx_count_received,
    wt.total_value_sent_eth,
    wt.total_value_received_eth,
    wt.avg_value_per_tx,
    wt.median_value_per_tx,
    wt.first_tx_timestamp,
    wt.last_tx_timestamp,
    wt.active_days,
    wt.days_since_first_tx,
    wt.unique_counterparties,
    wt.avg_gas_price,
    wt.total_gas_spent_eth,

    -- Token activity features
    coalesce(ta.unique_tokens_interacted, 0) as unique_tokens_interacted,
    coalesce(ta.erc20_transfer_count, 0) as erc20_transfer_count,
    coalesce(ta.erc721_transfer_count, 0) as erc721_transfer_count,
    coalesce(ta.top_token_concentration, 0) as top_token_concentration,
    coalesce(ta.token_diversity_entropy, 0) as token_diversity_entropy,

    -- Contract interaction features
    coalesce(ci.unique_contracts_called, 0) as unique_contracts_called,
    coalesce(ci.contract_call_count, 0) as contract_call_count,
    coalesce(ci.dex_interaction_count, 0) as dex_interaction_count,
    coalesce(ci.lending_interaction_count, 0) as lending_interaction_count,
    coalesce(ci.bridge_interaction_count, 0) as bridge_interaction_count,
    coalesce(ci.nft_marketplace_interaction_count, 0) as nft_marketplace_interaction_count,

    -- Derived: tx frequency
    wt.tx_count / nullif(wt.days_since_first_tx, 0) as tx_frequency_per_day,

    -- Derived: activity regularity (active_days / days_since_first_tx)
    wt.active_days / nullif(wt.days_since_first_tx, 0) as activity_regularity,

    -- Derived: hour of day entropy
    coalesce(he.hour_of_day_entropy, 0) as hour_of_day_entropy,

    -- Derived: weekend vs weekday ratio
    safe_divide(ds.weekend_tx_count, ds.weekday_tx_count) as weekend_vs_weekday_ratio,

    -- Derived: is_contract
    coalesce(c.is_contract, false) as is_contract,

    -- Derived: protocol ratios
    safe_divide(ci.dex_interaction_count, ci.contract_call_count) as dex_to_total_ratio,
    safe_divide(ci.lending_interaction_count, ci.contract_call_count) as lending_to_total_ratio,

    -- Derived: counterparty concentration (HHI)
    coalesce(ch.counterparty_concentration, 0) as counterparty_concentration,

    -- Derived: value velocity (total value transacted / active days)
    (wt.total_value_sent_eth + wt.total_value_received_eth) / nullif(wt.active_days, 0) as value_velocity,

    -- Derived: burst score
    coalesce(bs.burst_score, 1) as burst_score

  from {{ ref('stg_wallet_transactions') }} wt
  left join {{ ref('stg_wallet_token_activity') }} ta
    on wt.wallet_address = ta.wallet_address
  left join {{ ref('stg_wallet_contract_interactions') }} ci
    on wt.wallet_address = ci.wallet_address
  left join hour_entropy he
    on wt.wallet_address = he.wallet_address
  left join day_type_stats ds
    on wt.wallet_address = ds.wallet_address
  left join contracts c
    on wt.wallet_address = c.wallet_address
  left join counterparty_hhi ch
    on wt.wallet_address = ch.wallet_address
  left join burst_stats bs
    on wt.wallet_address = bs.wallet_address
)

select * from joined
