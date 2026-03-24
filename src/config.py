"""Centralized configuration loaded from environment variables."""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings, loaded from .env or environment."""

    # GCP / BigQuery
    gcp_project_id: str = "your-gcp-project-id"
    bigquery_dataset: str = "nansen_smart_money"
    bigquery_source_dataset: str = "bigquery-public-data.crypto_ethereum"

    # Sampling
    sample_window_days: int = 365
    min_wallet_tx_count: int = 10
    min_wallet_eth_transacted: float = 1_000.0
    max_wallets: int = 5_000

    # Clickhouse
    clickhouse_host: str = "localhost"
    clickhouse_port: int = 8123
    clickhouse_user: str = "default"
    clickhouse_password: str = ""
    clickhouse_database: str = "nansen"

    # Anthropic
    anthropic_api_key: str = ""

    # Etherscan (free tier: 5 req/sec)
    etherscan_api_key: str = ""

    # Model
    model_artifacts_path: str = "models/artifacts"
    random_seed: int = 42

    # Serving
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # W&B
    wandb_project: str = "nansen-smart-money"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
