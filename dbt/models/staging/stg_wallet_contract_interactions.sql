{{
  config(
    materialized='view'
  )
}}

with traces as (
  select
    from_address,
    to_address,
    block_timestamp
  from {{ source('crypto_ethereum', 'traces') }}
  where block_timestamp >= timestamp_sub(current_timestamp(), interval {{ var('sample_window_days', 90) }} day)
    and trace_type = 'call'
    and status = 1
    and from_address is not null
    and to_address is not null
),

-- Known protocol router/contract addresses (lowercase)
dex_routers as (
  select address from unnest([
    -- Uniswap V2 Router
    '0x7a250d5630b4cf539739df2c5dacb4c659f2488d',
    -- Uniswap V3 Router
    '0xe592427a0aece92de3edee1f18e0157c05861564',
    -- Uniswap V3 Router 02
    '0x68b3465833fb72a70ecdf485e0e4c7bd8665fc45',
    -- Uniswap Universal Router
    '0x3fc91a3afd70395cd496c647d5a6cc9d4b2b7fad',
    -- SushiSwap Router
    '0xd9e1ce17f2641f24ae83637ab66a2cca9c378b9f',
    -- 1inch V5 Router
    '0x1111111254eeb25477b68fb85ed929f73a960582',
    -- 1inch V4 Router
    '0x1111111254fb6c44bac0bed2854e76f90643097d'
  ]) as address
),

lending_contracts as (
  select address from unnest([
    -- Aave V2 Lending Pool
    '0x7d2768de32b0b80b7a3454c06bdac94a69ddc7a9',
    -- Aave V3 Pool
    '0x87870bca3f3fd6335c3f4ce8392d69350b4fa4e2',
    -- Compound V2 Comptroller
    '0x3d9819210a31b4961b30ef54be2aed79b9c9cd3b',
    -- Compound V3 (cUSDCv3)
    '0xc3d688b66703497daa19211eedff47f25384cdc3'
  ]) as address
),

bridge_contracts as (
  select address from unnest([
    -- Arbitrum Gateway Router
    '0x72ce9c846789fdb6fc1f34ac4ad25dd9ef7031ef',
    -- Arbitrum Delayed Inbox
    '0x4dbd4fc535ac27206064b68ffcf827b0a60bab3f',
    -- Optimism L1 Standard Bridge
    '0x99c9fc46f92e8a1c0dec1b1747d010903e884be1',
    -- Optimism L1 Cross Domain Messenger
    '0x25ace71c97b33cc4729cf772ae268934f7ab5fa1'
  ]) as address
),

nft_marketplaces as (
  select address from unnest([
    -- OpenSea Seaport 1.1
    '0x00000000006c3852cbef3e08e8df289169ede581',
    -- OpenSea Seaport 1.4
    '0x00000000000001ad428e4906ae43d8f9852d0dd6',
    -- OpenSea Seaport 1.5
    '0x00000000000000adc04c56bf30ac9d3c0aaf14dc',
    -- Blur Marketplace
    '0x000000000000ad05ccc4f10045630fb830b95127',
    -- Blur Blend
    '0x29469395eaf6f95920e59f858042f0e28d98a20b'
  ]) as address
),

wallet_calls as (
  select
    t.from_address as wallet_address,
    count(distinct t.to_address) as unique_contracts_called,
    count(*) as contract_call_count,
    countif(d.address is not null) as dex_interaction_count,
    countif(l.address is not null) as lending_interaction_count,
    countif(b.address is not null) as bridge_interaction_count,
    countif(n.address is not null) as nft_marketplace_interaction_count
  from traces t
  left join dex_routers d
    on lower(t.to_address) = d.address
  left join lending_contracts l
    on lower(t.to_address) = l.address
  left join bridge_contracts b
    on lower(t.to_address) = b.address
  left join nft_marketplaces n
    on lower(t.to_address) = n.address
  group by t.from_address
)

select
  wallet_address,
  unique_contracts_called,
  contract_call_count,
  dex_interaction_count,
  lending_interaction_count,
  bridge_interaction_count,
  nft_marketplace_interaction_count
from wallet_calls
