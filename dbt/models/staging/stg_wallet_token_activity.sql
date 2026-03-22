{{
  config(
    materialized='view'
  )
}}

with token_transfers as (
  select
    from_address,
    to_address,
    token_address,
    value,
    block_timestamp
  from {{ source('crypto_ethereum', 'token_transfers') }}
  where block_timestamp >= timestamp_sub(current_timestamp(), interval {{ var('sample_window_days', 90) }} day)
),

wallet_tokens as (
  select
    wallet_address,
    token_address,
    transfer_count,
    token_type
  from (
    select
      from_address as wallet_address,
      token_address,
      count(*) as transfer_count,
      case
        when safe_cast(value as bignumeric) is not null
             and safe_cast(value as bignumeric) > 1
          then 'erc20'
        else 'erc721'
      end as token_type
    from token_transfers
    where from_address is not null
    group by from_address, token_address, token_type

    union all

    select
      to_address as wallet_address,
      token_address,
      count(*) as transfer_count,
      case
        when safe_cast(value as bignumeric) is not null
             and safe_cast(value as bignumeric) > 1
          then 'erc20'
        else 'erc721'
      end as token_type
    from token_transfers
    where to_address is not null
    group by to_address, token_address, token_type
  )
),

wallet_summary as (
  select
    wallet_address,
    count(distinct token_address) as unique_tokens_interacted,
    sum(case when token_type = 'erc20' then transfer_count else 0 end) as erc20_transfer_count,
    sum(case when token_type = 'erc721' then transfer_count else 0 end) as erc721_transfer_count,
    sum(transfer_count) as total_transfer_count
  from wallet_tokens
  group by wallet_address
),

top_token as (
  select
    wallet_address,
    max(token_share) as top_token_concentration
  from (
    select
      wt.wallet_address,
      wt.transfer_count / ws.total_transfer_count as token_share
    from wallet_tokens wt
    inner join wallet_summary ws
      on wt.wallet_address = ws.wallet_address
  )
  group by wallet_address
),

entropy_calc as (
  select
    wallet_address,
    -1.0 * sum(p * ln(p)) as token_diversity_entropy
  from (
    select
      wt.wallet_address,
      wt.transfer_count / ws.total_transfer_count as p
    from wallet_tokens wt
    inner join wallet_summary ws
      on wt.wallet_address = ws.wallet_address
    where ws.total_transfer_count > 0
  )
  where p > 0
  group by wallet_address
)

select
  ws.wallet_address,
  ws.unique_tokens_interacted,
  ws.erc20_transfer_count,
  ws.erc721_transfer_count,
  tt.top_token_concentration,
  coalesce(ec.token_diversity_entropy, 0) as token_diversity_entropy
from wallet_summary ws
left join top_token tt
  on ws.wallet_address = tt.wallet_address
left join entropy_calc ec
  on ws.wallet_address = ec.wallet_address
