{{
  config(
    materialized='view'
  )
}}

with txs as (
  select
    block_timestamp,
    from_address,
    to_address,
    value,
    gas,
    gas_price,
    receipt_gas_used
  from {{ source('crypto_ethereum', 'transactions') }}
  where block_timestamp >= timestamp_sub(current_timestamp(), interval {{ var('sample_window_days', 90) }} day)
    and receipt_status = 1
),

sent as (
  select
    from_address as wallet_address,
    count(*) as tx_count_sent,
    sum(cast(value as bignumeric) / 1e18) as total_value_sent_eth,
    min(block_timestamp) as first_sent_ts,
    max(block_timestamp) as last_sent_ts,
    count(distinct date(block_timestamp)) as active_days_sent,
    count(distinct to_address) as unique_counterparties_sent,
    sum(cast(receipt_gas_used as bignumeric) * cast(gas_price as bignumeric) / 1e18) as gas_spent_eth_sent,
    avg(cast(gas_price as float64)) as avg_gas_price_sent
  from txs
  where from_address is not null
  group by from_address
),

received as (
  select
    to_address as wallet_address,
    count(*) as tx_count_received,
    sum(cast(value as bignumeric) / 1e18) as total_value_received_eth,
    min(block_timestamp) as first_recv_ts,
    max(block_timestamp) as last_recv_ts,
    count(distinct date(block_timestamp)) as active_days_recv,
    count(distinct from_address) as unique_counterparties_recv
  from txs
  where to_address is not null
  group by to_address
),

median_values as (
  select
    wallet_address,
    approx_quantiles(tx_value_eth, 100)[offset(50)] as median_value_per_tx
  from (
    select
      from_address as wallet_address,
      cast(value as float64) / 1e18 as tx_value_eth
    from txs
    where from_address is not null
    union all
    select
      to_address as wallet_address,
      cast(value as float64) / 1e18 as tx_value_eth
    from txs
    where to_address is not null
  )
  group by wallet_address
),

combined as (
  select
    coalesce(s.wallet_address, r.wallet_address) as wallet_address,
    coalesce(s.tx_count_sent, 0) + coalesce(r.tx_count_received, 0) as tx_count,
    coalesce(s.tx_count_sent, 0) as tx_count_sent,
    coalesce(r.tx_count_received, 0) as tx_count_received,
    coalesce(s.total_value_sent_eth, 0) as total_value_sent_eth,
    coalesce(r.total_value_received_eth, 0) as total_value_received_eth,
    (coalesce(s.total_value_sent_eth, 0) + coalesce(r.total_value_received_eth, 0))
      / nullif(coalesce(s.tx_count_sent, 0) + coalesce(r.tx_count_received, 0), 0) as avg_value_per_tx,
    least(s.first_sent_ts, r.first_recv_ts) as first_tx_timestamp,
    greatest(s.last_sent_ts, r.last_recv_ts) as last_tx_timestamp,
    greatest(coalesce(s.active_days_sent, 0), coalesce(r.active_days_recv, 0)) as active_days,
    timestamp_diff(
      current_timestamp(),
      least(coalesce(s.first_sent_ts, r.first_recv_ts), coalesce(r.first_recv_ts, s.first_sent_ts)),
      day
    ) as days_since_first_tx,
    coalesce(s.unique_counterparties_sent, 0) + coalesce(r.unique_counterparties_recv, 0) as unique_counterparties,
    coalesce(s.avg_gas_price_sent, 0) as avg_gas_price,
    coalesce(s.gas_spent_eth_sent, 0) as total_gas_spent_eth
  from sent s
  full outer join received r
    on s.wallet_address = r.wallet_address
)

select
  c.*,
  m.median_value_per_tx
from combined c
left join median_values m
  on c.wallet_address = m.wallet_address
where c.tx_count >= {{ var('min_tx_count', 10) }}
  and (c.total_value_sent_eth + c.total_value_received_eth) >= {{ var('min_eth_transacted', 1) }}
