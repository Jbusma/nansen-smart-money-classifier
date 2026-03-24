# Smart Money Wallet Behavioral Classifier

End-to-end ML pipeline that classifies Ethereum wallets into behavioral archetypes using on-chain transaction data from BigQuery, and surfaces AI-generated intelligence via an MCP server that Claude can use directly.

Built to mirror Nansen's core product loop: **ingest → label → surface signal**.

```
BigQuery (Ethereum) → Python Features → ClickHouse → HDBSCAN Clustering → Claude AI Labeling → MCP Server
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        DATA INGESTION LAYER                         │
│                                                                     │
│  BigQuery Public Dataset ──► Python Feature Engineering ──► Store   │
│  (crypto_ethereum.*)          (12 behavioral features)  (ClickHouse)│
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       ML PIPELINE LAYER                             │
│                                                                     │
│  ┌──────────────┐    ┌──────────────────┐    ┌──────────────────┐   │
│  │ Unsupervised │    │  Protocol        │    │   LLM Insight    │   │
│  │  Clustering  │    │  Enrichment      │    │   Generation     │   │
│  │   HDBSCAN    │    │ 5,700+ addresses │    │   Claude API     │   │
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
│  │   8 Tools:           │    │   - Cluster viz      │               │
│  │   - classify_wallet  │    │   - Wallet lookup    │               │
│  │   - explain_wallet   │    │   - Feature explorer │               │
│  │   - get_wallet_      │    │   - AI narratives    │               │
│  │     context          │    │                      │               │
│  │   - enrich_registry  │    │                      │               │
│  │   - label_wallet     │    │                      │               │
│  │   - label_cluster    │    │                      │               │
│  │   + 2 more           │    │                      │               │
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
# Optional: ETHERSCAN_API_KEY for contract label enrichment
```

### 3. Run with Docker

```bash
docker-compose up -d
# ClickHouse:  localhost:8123 (HTTP) / localhost:9000 (native)
# API:         http://localhost:8000
# Dashboard:   http://localhost:8501
# MCP Server:  http://localhost:3001/mcp (Streamable HTTP)
```

### 4. Run the Pipeline

```bash
# Extract data from BigQuery
make extract

# Sync features to ClickHouse
make sync-features

# Discover clusters
make cluster

# Enrich protocol registry (labels unknown contracts)
python -m src.data.protocol_enrichment          # free sources only
python -m src.data.protocol_enrichment --etherscan --top-n 500  # + Etherscan

# Start serving
make serve-api        # FastAPI on :8000
make serve-dashboard  # Streamlit on :8501
make serve-mcp        # MCP server on stdio
```

---

## MCP Server

The MCP server lets Claude (or any MCP client) interact with the classifier, explore wallets, and label data — closing the human-in-the-loop.

**Transports:** Streamable HTTP (`POST /mcp` on port 3001) and stdio (`--stdio` flag).

**Claude Desktop config** (`claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "nansen-smart-money": {
      "command": "node",
      "args": ["path/to/mcp_server/dist/index.js", "--stdio"],
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
| `get_wallet_context` | Rich on-chain context: tx summary, top contracts with protocol labels, token activity, timing patterns |
| `enrich_registry` | Populate the protocol registry from free sources (CoinGecko, DeFi Llama, Etherscan) |
| `label_wallet` | Assign a behavioral label to a single wallet (persists to ClickHouse ground truth) |
| `label_cluster` | Bulk-label all wallets in a cluster with a single behavioral label |

---

## Data Pipeline

### Source

**BigQuery Public Ethereum Dataset** (`bigquery-public-data.crypto_ethereum`) — transactions, token transfers, traces, logs, contracts, balances.

### Sampling

Current config: wallets with ≥10 transactions AND ≥1,000 ETH transacted, capped at 5,000 wallets. This produces a dataset of ~4,998 high-value wallets.

### Behavioral Features (12 dimensions)

| Category | Features |
|---|---|
| Activity | `tx_frequency_per_day`, `activity_regularity`, `burst_score` |
| Temporal | `hour_of_day_entropy`, `weekend_vs_weekday_ratio` |
| Value | `avg_holding_duration_estimate`, `gas_price_sensitivity`, `value_velocity` |
| DeFi | `dex_to_total_ratio`, `lending_to_total_ratio` |
| Graph | `counterparty_concentration` |
| Type | `is_contract` |

### Protocol Registry

On-chain wallet context is only useful if we can identify *what* contracts a wallet interacts with. The protocol registry maps contract addresses → human-readable labels (e.g. "Uniswap V3 Router", "Aave V3 Pool").

**Sources (5,700+ addresses):**
- Hardcoded seed: 17 major DeFi protocols
- CoinGecko token list: ~4,950 ERC-20 tokens
- DeFi Llama protocols API: ~270 protocol addresses with categories
- Etherscan V2 contract API: top-N unknown contracts by interaction count (~510)

---

## ML Pipeline

### Unsupervised Clustering

- **HDBSCAN** for density-based clustering on standardized 12D feature space
- Consistently finds **3 clusters** in the full dataset:
  - **Cluster 0**: Institutional/OTC — high ETH volume, low tx frequency, concentrated counterparties
  - **Cluster 1**: DeFi Power Users — high dex/lending ratios, diverse token activity, many contracts
  - **Cluster 2**: Active Traders — high tx frequency, high burst scores, timing regularity

### LLM Intelligence

- **Claude API** generates natural language wallet narratives and cluster profiles
- ClickHouse-backed cache with 24h TTL
- Structured prompts for consistent, actionable intelligence output

### Ground Truth & Labeling

Ground truth labels are stored in ClickHouse (`ground_truth` table) and managed through the MCP tools. The `label_wallet` and `label_cluster` tools allow Claude to persist classification decisions during interactive analysis sessions.

Current state: ~4,985 wallets labeled via heuristic rules (99.5% `active_trader`), plus agent-assisted labels from cluster profiling. The heuristic imbalance means a supervised classifier trained on this data would trivially predict the majority class — see "Lessons Learned" below.

---

## The Journey: What We Learned

This project started as a clean ML pipeline design. Along the way, we ran into real data science problems that shaped the final system. Documenting them honestly here.

### 1. The "Unknown Contract" Problem

The wallet context tool was useless at first — every contract interaction showed "Unknown contract" because we only had 17 hardcoded addresses. We built a 4-source enrichment pipeline that grew the protocol registry to 5,700+ addresses, covering ~81% of all interactions by volume.

### 2. The Sampling Bias

With `min_wallet_eth_transacted >= 1000`, we're sampling DeFi **infrastructure**, not typical users. The dataset is dominated by:
- Protocol vaults (Aave, Compound, Lido)
- Multisig treasuries
- Token distributor contracts
- Bridge contracts

Cluster 1 turned out to be 100% smart contracts (protocol infrastructure), while Cluster 0 was 100% EOAs (externally owned accounts). The dominant clustering signal is simply `is_contract`.

### 3. The Classifier Experiment

We attempted a discovery/validation split to train a supervised classifier:
1. Split 4,998 wallets into two halves (interleaved by descending ETH volume)
2. Reclustered the discovery half with HDBSCAN
3. Sampled 20% from each cluster for AI-assisted labeling
4. Found only 2 clusters regardless of hyperparameters (UMAP on/off, various `min_cluster_size`)

The 2-cluster split maps cleanly to contract vs. EOA — not a meaningful behavioral taxonomy. A classifier trained on this would just be learning `is_contract`, which is a trivial on-chain lookup.

### 4. What Would Fix This

- **Lower the ETH threshold** (e.g., ≥1 ETH instead of ≥1,000) to capture real user behavior
- **Increase sample size** to 50K-200K wallets for more diverse behavioral patterns
- **Time-windowed features** instead of all-time aggregations
- **Exclude known infrastructure** (bridges, protocol contracts) from the training set

The pipeline, tooling, and infrastructure are all in place — the limiting factor is the sampling parameters, not the architecture.

---

## Stack

| Component | Technology |
|---|---|
| Data Warehouse | Google BigQuery |
| Feature Store | ClickHouse |
| Clustering | HDBSCAN |
| LLM Layer | Claude API (Anthropic) |
| API | FastAPI |
| MCP Server | TypeScript (MCP SDK) |
| Dashboard | Streamlit + Plotly |
| Protocol Enrichment | CoinGecko, DeFi Llama, Etherscan V2 |
| CI/CD | GitHub Actions |
| Containerization | Docker Compose |

---

## Project Structure

```
├── src/
│   ├── config.py                   # Pydantic settings (env vars)
│   ├── data/
│   │   ├── bigquery_extract.py     # BigQuery → parquet extraction
│   │   ├── clickhouse_sync.py      # DDL + feature/ground-truth sync
│   │   ├── ground_truth.py         # Heuristic labeling rules
│   │   ├── protocol_enrichment.py  # 4-source registry enrichment pipeline
│   │   └── wallet_context.py       # Rich on-chain context queries
│   ├── features/
│   │   ├── compute_features.py     # Raw → 12 behavioral features
│   │   ├── feature_engineering.py  # Feature transforms
│   │   └── feature_store.py        # ClickHouse feature store client
│   ├── models/
│   │   ├── clustering.py           # HDBSCAN clustering pipeline
│   │   ├── classifier.py           # Supervised classifier (XGBoost)
│   │   ├── train.py                # Training loop
│   │   └── evaluation.py           # Metrics and evaluation
│   ├── llm/
│   │   ├── insight_generator.py    # Claude API narrative generation
│   │   ├── prompts.py              # Structured prompts
│   │   └── cache.py                # ClickHouse narrative cache
│   ├── serving/
│   │   ├── api.py                  # FastAPI (classify, label, enrich, etc.)
│   │   └── streamlit_app.py        # Interactive dashboard
│   └── experiments/
│       ├── discovery_validation_split.py  # Split + recluster experiment
│       └── sample_wallet_contexts.py      # Stratified context sampling
├── mcp_server/
│   └── src/
│       ├── index.ts                # MCP server (8 tools, stdio + HTTP)
│       └── client.ts               # FastAPI HTTP client
├── tests/                          # pytest suite
├── dbt/                            # dbt project (BigQuery transforms)
├── dags/                           # Airflow DAG (daily pipeline)
├── Dockerfile                      # Multi-stage (API + Dashboard)
├── docker-compose.yml              # ClickHouse + API + Dashboard + MCP
├── Makefile                        # Common commands
└── .github/workflows/              # CI pipeline
```

---

## Docker Services

| Service | Port | Description |
|---|---|---|
| `clickhouse` | 8123, 9000 | Feature store, ground truth, protocol registry, narrative cache |
| `api` | 8000 | FastAPI backend for classification, labeling, enrichment |
| `dashboard` | 8501 | Streamlit interactive explorer |
| `mcp-server` | 3001 | MCP server (Streamable HTTP transport) |

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
