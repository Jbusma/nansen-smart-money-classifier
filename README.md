# Smart Money Wallet Behavioral Classifier

End-to-end ML pipeline that classifies Ethereum wallets into behavioral archetypes — **Smart Money, MEV Bots, DeFi Farmers, Airdrop Hunters, Retail Traders, HODLers, NFT Traders** — using onchain transaction data from BigQuery, and surfaces AI-generated intelligence via an MCP-compatible interface.

Built to mirror Nansen's core product loop: **ingest → label → surface signal**.

```
BigQuery (Ethereum) → dbt → Clickhouse → HDBSCAN/UMAP → XGBoost+MLP → Claude AI → MCP Server
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        DATA INGESTION LAYER                         │
│                                                                     │
│  BigQuery Public Dataset ──► dbt Transformations ──► Feature Store  │
│  (crypto_ethereum.*)          (behavioral features)   (Clickhouse)  │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       ML PIPELINE LAYER                             │
│                                                                     │
│  ┌──────────────┐    ┌──────────────────┐    ┌──────────────────┐   │
│  │ Unsupervised │    │    Supervised    │    │   LLM Insight    │   │
│  │  Clustering  │───►│   Classification │───►│   Generation     │   │
│  │ HDBSCAN+UMAP │    │  XGBoost + MLP   │    │   Claude API     │   │
│  └──────────────┘    └──────────────────┘    └──────────────────┘   │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       SERVING LAYER                                 │
│                                                                     │
│  ┌──────────────────────┐    ┌──────────────────────┐               │
│  │   MCP Server         │    │   Streamlit          │               │
│  │   (TypeScript)       │    │   Dashboard          │               │
│  │                      │    │                      │               │
│  │   Tools:             │    │   - Cluster viz      │               │
│  │   - classify_wallet  │    │   - Wallet lookup    │               │
│  │   - get_cluster_     │    │   - Feature explorer │               │
│  │     profile          │    │   - AI narratives    │               │
│  │   - find_similar     │    │                      │               │
│  │   - explain_wallet   │    │                      │               │
│  └──────────────────────┘    └──────────────────────┘               │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Quick Start

### Prerequisites

- Python 3.11+
- Node.js 22+
- Docker & Docker Compose
- GCP project with BigQuery access
- Anthropic API key

### 1. Clone & Install

```bash
git clone https://github.com/jesse-xxx/nansen-smart-money-classifier.git
cd nansen-smart-money-classifier

# Python dependencies
pip install -e ".[dev,tracking]"

# MCP server dependencies
cd mcp_server && npm install && cd ..
```

### 2. Configure

```bash
cp .env.example .env
# Edit .env with your GCP project ID, Anthropic API key, etc.
```

### 3. Run with Docker (easiest)

```bash
docker-compose up -d
# API:       http://localhost:8000
# Dashboard: http://localhost:8501
# Clickhouse: localhost:8123
```

### 4. Run the Pipeline

```bash
# Extract data from BigQuery
make extract

# Run dbt transformations
make dbt-run

# Sync features to Clickhouse
make sync-features

# Discover clusters
make cluster

# Train classifier
make train

# Start serving
make serve-api        # FastAPI on :8000
make serve-dashboard  # Streamlit on :8501
make serve-mcp        # MCP server on stdio
```

---

## MCP Server

Connect the classifier to Claude Desktop or any MCP-compatible client.

**Claude Desktop config** (`claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "smart-money": {
      "command": "node",
      "args": ["path/to/mcp_server/dist/index.js"],
      "env": {
        "API_URL": "http://localhost:8000"
      }
    }
  }
}
```

**Available tools:**

| Tool | Description |
|---|---|
| `classify_wallet` | Classify a wallet into a behavioral archetype with confidence score |
| `get_cluster_profile` | AI-generated intelligence profile for a behavioral cluster |
| `find_similar_wallets` | Find behaviorally similar wallets via cosine similarity |
| `explain_wallet` | Natural language intelligence briefing for a wallet |

---

## Data Pipeline

### Source

**BigQuery Public Ethereum Dataset** (`bigquery-public-data.crypto_ethereum`) — transactions, token transfers, traces, logs, contracts, balances.

### Sampling

- 90-day rolling window
- Wallets with ≥10 transactions AND ≥1 ETH transacted
- ~200K-500K wallets after filtering

### dbt Transformations

```
staging/
  stg_wallet_transactions.sql       — tx counts, values, gas, timing
  stg_wallet_token_activity.sql     — token diversity, ERC-20/721 activity
  stg_wallet_contract_interactions.sql — DEX, lending, bridge, NFT interactions
intermediate/
  int_wallet_behavioral_features.sql — derived behavioral signals
marts/
  mart_wallet_feature_vectors.sql   — z-score normalized feature matrix
```

### Behavioral Features (~25 dimensions)

| Category | Features |
|---|---|
| Activity | tx_frequency_per_day, activity_regularity, burst_score |
| Value | avg_value_per_tx, total_value_sent_eth, value_velocity |
| Temporal | hour_of_day_entropy, weekend_vs_weekday_ratio |
| Graph | unique_counterparties, counterparty_concentration (HHI) |
| DeFi | dex_to_total_ratio, lending_to_total_ratio, unique_contracts_called |
| Token | token_diversity_entropy, top_token_concentration |

---

## ML Pipeline

### Phase 1: Unsupervised Clustering

- **UMAP** for dimensionality reduction (cosine metric, 30 neighbors)
- **HDBSCAN** for density-based clustering (handles noise, variable-density clusters)
- Validation: silhouette score, Calinski-Harabasz, stability analysis

### Phase 2: Supervised Classification

- **XGBoost** primary model (Optuna hyperparameter tuning, 100 trials)
- **PyTorch MLP** secondary model (128→64→classes, BatchNorm, Dropout)
- **Ensemble**: weighted probability averaging, weights optimized on validation set
- **Evaluation**: stratified 5-fold CV, macro F1, SHAP feature importance
- **Target**: >0.85 macro F1

### Phase 3: LLM Intelligence

- **Claude API** generates natural language wallet narratives and cluster profiles
- Clickhouse-backed cache with 24h TTL
- Structured prompts for consistent, actionable intelligence output

---

## Stack

| Component | Technology |
|---|---|
| Data Warehouse | Google BigQuery |
| Transformations | dbt |
| Feature Store | Clickhouse |
| Clustering | UMAP + HDBSCAN |
| Classification | XGBoost + PyTorch |
| LLM Layer | Claude API (Anthropic) |
| API | FastAPI |
| MCP Server | TypeScript (MCP SDK) |
| Dashboard | Streamlit + Plotly |
| Orchestration | Cloud Composer (Airflow) |
| Deployment | Cloud Run, Docker |
| CI/CD | GitHub Actions, Cloud Build |
| Experiment Tracking | Weights & Biases |

---

## Project Structure

```
├── dbt/                          # dbt transformation project
│   ├── models/staging/           # Raw → aggregated per wallet
│   ├── models/intermediate/      # Derived behavioral features
│   └── models/marts/             # Normalized feature vectors
├── src/
│   ├── data/                     # BigQuery extraction, Clickhouse sync, ground truth
│   ├── features/                 # Feature engineering, feature store client
│   ├── models/                   # Clustering, classifier, evaluation, training
│   ├── llm/                      # Claude API integration, prompts, caching
│   └── serving/                  # FastAPI endpoint, Streamlit dashboard
├── mcp_server/                   # TypeScript MCP server
├── dags/                         # Airflow DAG for daily pipeline
├── tests/                        # Unit tests
├── notebooks/                    # EDA, clustering, training, LLM demo
├── Dockerfile                    # Multi-stage (API + Dashboard)
├── docker-compose.yml            # Full local stack
├── cloudbuild.yaml               # GCP Cloud Build deployment
└── .github/workflows/            # CI + model regression testing
```

---

## Development

```bash
make lint        # ruff check + format
make typecheck   # mypy
make test        # pytest
make format      # auto-fix lint issues
```

---

## License

MIT
