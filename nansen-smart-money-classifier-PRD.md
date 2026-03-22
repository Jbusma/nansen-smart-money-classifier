# PRD: Smart Money Wallet Behavioral Classifier

## Portfolio Project for Nansen — Senior AI/ML Engineer Application

**Author:** Jesse  
**Date:** March 2026  
**Status:** Draft  
**Repository:** `github.com/jesse-xxx/nansen-smart-money-classifier`

---

## 1. Executive Summary

This project builds an end-to-end ML pipeline that ingests raw Ethereum transaction data from Google BigQuery's public blockchain dataset, engineers behavioral features per wallet, clusters wallets into behavioral archetypes (whales, bots/MEV, DeFi farmers, airdrop hunters, retail traders, dormant holders), trains a supervised classifier to label unseen wallets, and surfaces AI-generated natural language insights via an LLM layer — all exposed through an MCP-compatible interface.

The project is designed to mirror Nansen's core product loop (**ingest → label → surface signal**) using their exact infrastructure stack, demonstrating production-grade AI/ML engineering on blockchain data at the intersection of data engineering, machine learning, and LLM integration.

---

## 2. Why This Project

### 2.1 Direct Alignment to Nansen's Core Moat

Nansen's CEO Alex Svanevik has stated that their primary AI use case is **automated wallet address labeling** using finely-tuned AI agents. Their database of 500M+ labeled wallets is the foundation of every product they ship: Smart Money tracking, Token God Mode, Profiler, and the new agentic trading interface. This project demonstrates the ability to build that core capability from scratch.

### 2.2 Stack Alignment

| Nansen's Stack | This Project |
|---|---|
| Google Cloud Platform | GCP (BigQuery, Cloud Run, Artifact Registry) |
| BigQuery (60TB+ blockchain data, 1PB/day processing) | BigQuery public Ethereum dataset (`bigquery-public-data.crypto_ethereum`) |
| BigQuery ML | BigQuery ML for initial feature extraction + baseline models |
| TensorFlow / PyTorch | PyTorch for the supervised classifier |
| Cloud Composer (Airflow) | Cloud Composer DAG for pipeline orchestration |
| Clickhouse | Clickhouse for fast analytical queries on feature store |
| dbt | dbt for data transformation layer |
| MCP (Model Context Protocol) | MCP server exposing classifier + insights to Claude/Cursor |
| LLMs (Claude, GPT-4) | Claude API for natural language insight generation |
| Python | Python throughout |

### 2.3 Skill Demonstration Matrix

| Job Requirement | How This Project Demonstrates It |
|---|---|
| Lead design/implementation of AI/ML models on blockchain data | Full pipeline: clustering → classification → LLM insight generation |
| High performance and scalability | BigQuery for petabyte-scale data, Clickhouse for fast feature queries, Cloud Run for serving |
| Integrate AI/ML models into platform | MCP server interface — the exact integration pattern Nansen is shipping |
| Familiarity with modern AI models, APIs, SDKs | Claude API for agentic insight layer, OpenAI as fallback |
| Full-stack with data / data engineering experience | BigQuery ingestion → dbt transformations → feature store → model training → serving |
| Passion for blockchain/crypto/Web3 | Domain-specific feature engineering requiring deep onchain knowledge |
| AI-first mindset | LLM-powered insight generation, MCP-native interface, AI throughout the workflow |

---

## 3. Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        DATA INGESTION LAYER                         │
│                                                                     │
│  BigQuery Public Dataset ──► dbt Transformations ──► Feature Store  │
│  (crypto_ethereum.*)          (behavioral features)   (Clickhouse)  │
│                                                                     │
│  Tables:                      Models:                               │
│  - transactions               - stg_wallet_transactions             │
│  - token_transfers            - stg_wallet_token_activity           │
│  - traces                     - int_wallet_behavioral_features      │
│  - logs                       - mart_wallet_feature_vectors         │
│  - contracts                                                        │
│  - balances                                                         │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       ML PIPELINE LAYER                             │
│                                                                     │
│  ┌──────────────┐    ┌──────────────────┐    ┌──────────────────┐  │
│  │  Unsupervised │    │    Supervised     │    │   LLM Insight    │  │
│  │  Clustering   │───►│   Classification  │───►│   Generation     │  │
│  │              │    │                  │    │                  │  │
│  │  HDBSCAN +   │    │  XGBoost primary │    │  Claude API      │  │
│  │  UMAP for    │    │  PyTorch MLP     │    │  Cluster profile │  │
│  │  discovery   │    │  secondary       │    │  summaries +     │  │
│  │              │    │                  │    │  wallet narratives│  │
│  └──────────────┘    └──────────────────┘    └──────────────────┘  │
│                                                                     │
│  Training: GCP Vertex AI or local GPU                               │
│  Experiment tracking: Weights & Biases                              │
│  Model registry: GCS bucket versioning                              │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       SERVING LAYER                                 │
│                                                                     │
│  ┌──────────────────┐    ┌──────────────────┐                      │
│  │   MCP Server      │    │   Streamlit       │                      │
│  │   (TypeScript)     │    │   Dashboard       │                      │
│  │                    │    │                    │                      │
│  │   Tools:           │    │   - Cluster viz    │                      │
│  │   - classify_wallet│    │   - Wallet lookup  │                      │
│  │   - get_cluster_   │    │   - Feature explorer│                     │
│  │     profile        │    │   - AI narratives  │                      │
│  │   - find_similar   │    │                    │                      │
│  │   - explain_wallet │    │                    │                      │
│  │                    │    │                    │                      │
│  │   Deployed on      │    │   Deployed on      │                      │
│  │   Cloud Run        │    │   Cloud Run        │                      │
│  └──────────────────┘    └──────────────────┘                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 4. Data Layer

### 4.1 Source: BigQuery Public Ethereum Dataset

**Dataset:** `bigquery-public-data.crypto_ethereum`  
**Update frequency:** Near real-time (~4 min delay for block finality)  
**Cost:** First 1TB of queries free; ~$5/TB after (GCP gives $300 free credit)

**Tables used:**

| Table | Purpose | Key Columns |
|---|---|---|
| `transactions` | Core wallet activity | `from_address`, `to_address`, `value`, `gas_price`, `receipt_gas_used`, `block_timestamp`, `input` |
| `token_transfers` | ERC-20/721 activity | `token_address`, `from_address`, `to_address`, `value` |
| `traces` | Internal transactions (contract calls) | `from_address`, `to_address`, `value`, `call_type`, `trace_type` |
| `logs` | Smart contract events | `address`, `topics`, `data`, `block_timestamp` |
| `contracts` | Identify contract vs. EOA | `address`, `bytecode` |
| `balances` | Current ETH balances (daily) | `address`, `eth_balance` |

### 4.2 Sampling Strategy

Full Ethereum history is massive. For a portfolio project, we scope to a meaningful but tractable slice:

- **Time window:** 90-day rolling window (most recent complete 90 days)
- **Wallet filter:** Active wallets with ≥10 transactions AND ≥1 ETH transacted in the window
- **Expected sample size:** ~200K-500K wallets after filtering
- **Rationale:** Enough diversity to capture all behavioral archetypes; small enough for tractable compute; recency ensures relevance

### 4.3 dbt Transformation Layer

The dbt project structures the raw BigQuery data into analytical models following a staging → intermediate → mart pattern.

**`stg_wallet_transactions`** — One row per wallet, aggregated transaction stats:
- `wallet_address`
- `tx_count`, `tx_count_sent`, `tx_count_received`
- `total_value_sent_eth`, `total_value_received_eth`
- `avg_value_per_tx`, `median_value_per_tx`
- `first_tx_timestamp`, `last_tx_timestamp`
- `active_days`, `days_since_first_tx`
- `unique_counterparties`
- `avg_gas_price`, `total_gas_spent_eth`

**`stg_wallet_token_activity`** — Token-level behavior:
- `wallet_address`
- `unique_tokens_interacted`
- `erc20_transfer_count`, `erc721_transfer_count`
- `top_token_concentration` (% of activity in top token)
- `token_diversity_entropy` (Shannon entropy across tokens)

**`stg_wallet_contract_interactions`** — Smart contract engagement:
- `wallet_address`
- `unique_contracts_called`
- `contract_call_count`
- `dex_interaction_count` (filtered to known DEX routers)
- `lending_interaction_count` (Aave, Compound, etc.)
- `bridge_interaction_count`
- `nft_marketplace_interaction_count`

**`int_wallet_behavioral_features`** — Derived behavioral features:
- `wallet_address`
- `tx_frequency_per_day`
- `activity_regularity` (std dev of daily tx counts)
- `hour_of_day_entropy` (temporal behavior diversity)
- `weekend_vs_weekday_ratio`
- `avg_holding_duration_estimate` (from token transfer patterns)
- `gas_price_sensitivity` (correlation of tx count with gas price)
- `is_contract` (boolean — filter out or label differently)
- `dex_to_total_ratio`, `lending_to_total_ratio`
- `counterparty_concentration` (HHI of counterparties)
- `value_velocity` (turnover rate: volume / avg balance)
- `burst_score` (max tx count in any 1-hour window / avg hourly rate)

**`mart_wallet_feature_vectors`** — Final feature matrix ready for ML:
- All features from intermediate layer
- Normalized (z-score within sample)
- Missing values imputed (median for continuous, mode for categorical)
- Feature vectors exported to Clickhouse for fast serving

### 4.4 Feature Store (Clickhouse)

Clickhouse handles the low-latency query layer for the serving path. When a user or MCP client queries a wallet, the feature store returns pre-computed features in <50ms rather than re-running BigQuery.

**Table:** `wallet_features`  
**Engine:** `MergeTree() ORDER BY wallet_address`  
**Update cadence:** Daily batch via Cloud Composer DAG

---

## 5. ML Pipeline

### 5.1 Phase 1: Unsupervised Discovery (Clustering)

**Goal:** Discover natural behavioral archetypes without labels.

**Approach:**

1. **Dimensionality Reduction:** UMAP (Uniform Manifold Approximation and Projection) to reduce the ~25-dimensional feature space to 2D/3D for visualization and to improve clustering performance.
   - `n_neighbors=30`, `min_dist=0.1`, `metric='cosine'`
   
2. **Clustering:** HDBSCAN on the UMAP embedding.
   - `min_cluster_size=100`, `min_samples=10`
   - HDBSCAN chosen over K-Means because: (a) doesn't require pre-specifying k, (b) handles noise points gracefully (important — many wallets won't fit clean archetypes), (c) finds clusters of varying density.

3. **Cluster Validation:**
   - Silhouette score on UMAP embedding
   - Calinski-Harabasz index
   - Manual inspection of cluster centroids against known wallet types
   - Stability analysis: re-run on 80% subsamples, measure cluster assignment consistency

**Expected Clusters (hypothesis, to be validated):**

| Cluster | Behavioral Signature |
|---|---|
| **Smart Money / Whale** | High value, low frequency, high holding duration, concentrated counterparties, early token entries |
| **MEV / Bot** | Extremely high frequency, burst patterns, high gas sensitivity, narrow contract interactions (DEX routers), near-zero holding duration |
| **DeFi Farmer** | High contract diversity, lending + DEX heavy, moderate frequency, yield-seeking patterns |
| **Airdrop Hunter** | High contract breadth, low depth per protocol, temporal bursts around known airdrop windows |
| **Retail Trader** | Low-moderate frequency, DEX heavy, high token diversity, erratic timing |
| **HODLer / Accumulator** | Very low frequency, high value per tx, minimal contract interaction, long holding duration |
| **NFT Trader** | ERC-721 heavy, marketplace contract interactions, burst patterns around collection launches |

### 5.2 Phase 2: Ground Truth Construction

This is the critical step that makes the supervised classifier possible.

**Sources for ground truth labels:**

1. **Etherscan public tags** — Known exchange wallets, protocol wallets, bridge wallets. Scraped from Etherscan's public label cloud. Free and extensive.

2. **Known protocol addresses** — DEX routers (Uniswap, SushiSwap, 1inch), lending (Aave, Compound), bridges (Arbitrum, Optimism), NFT marketplaces (OpenSea, Blur). These are publicly documented.

3. **MEV identification** — Wallets with high Flashbots bundle participation (from `traces` patterns: sandwich attacks, liquidation bots, arbitrage patterns identifiable by back-to-back swaps in same block).

4. **Cluster exemplar annotation** — For clusters discovered in Phase 1, manually inspect the 20 wallets nearest each cluster centroid. Assign labels based on on-chain activity review. This is the "human-in-the-loop" step.

5. **Community labels** — Cross-reference with publicly available labeled datasets (Blockchain-ETL community, academic datasets from papers on wallet classification).

**Target: 10K-50K labeled wallets** across all archetype categories. The rest of the 200K-500K sample serves as the unlabeled pool for evaluation.

### 5.3 Phase 3: Supervised Classification

**Primary Model: XGBoost**

- Chosen for: fast training, handles tabular data excellently, built-in feature importance, easy to deploy
- Hyperparameter tuning via Optuna (100 trials, TPE sampler)
- Key hyperparameters: `max_depth`, `learning_rate`, `n_estimators`, `subsample`, `colsample_bytree`, `min_child_weight`

**Secondary Model: PyTorch MLP**

- 3-layer feedforward network: `[input_dim → 128 → 64 → num_classes]`
- ReLU activation, BatchNorm, Dropout(0.3)
- Trained with AdamW optimizer, cosine annealing LR schedule
- Purpose: demonstrate PyTorch proficiency (job req), ensemble with XGBoost for production

**Ensemble Strategy:**
- Weighted average of XGBoost and MLP probabilities
- Weights optimized on validation set
- Final prediction = argmax of ensemble probabilities
- Confidence score = max ensemble probability (used for filtering low-confidence predictions)

**Evaluation:**
- Stratified 5-fold cross-validation
- Metrics: macro F1, per-class precision/recall, confusion matrix
- Particular attention to Smart Money recall (Nansen's core label)
- Calibration curve to ensure confidence scores are meaningful
- Target: >0.85 macro F1 on held-out test set

**Training Infrastructure:**
- Vertex AI custom training job (or local GPU if Vertex costs are prohibitive)
- W&B (Weights & Biases) for experiment tracking
- Model artifacts stored in GCS with version tagging

### 5.4 Phase 4: LLM Insight Generation

**Goal:** Transform raw classifications and feature vectors into natural language intelligence — the "Nansen AI agent" experience.

**Implementation:**

```python
def generate_wallet_narrative(wallet_address: str, features: dict, label: str, confidence: float) -> str:
    """Generate a natural language profile of a wallet using Claude API."""
    
    prompt = f"""You are an onchain intelligence analyst. Given the following wallet behavioral profile, 
    generate a concise 2-3 sentence intelligence briefing.

    Wallet: {wallet_address}
    Classification: {label} (confidence: {confidence:.1%})
    
    Key behavioral signals:
    - Transaction frequency: {features['tx_frequency_per_day']:.1f}/day
    - Avg transaction value: {features['avg_value_per_tx']:.2f} ETH
    - DEX interaction ratio: {features['dex_to_total_ratio']:.1%}
    - Token diversity (entropy): {features['token_diversity_entropy']:.2f}
    - Holding duration estimate: {features['avg_holding_duration_estimate']:.1f} days
    - Burst score: {features['burst_score']:.2f}
    - Unique contracts: {features['unique_contracts_called']}
    - Counterparty concentration (HHI): {features['counterparty_concentration']:.3f}
    
    Focus on what makes this wallet notable and what it signals about the entity behind it.
    Be specific and actionable — this is for a crypto investor."""
    
    response = anthropic.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=300,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.content[0].text
```

**Cluster-level profiles:**

```python
def generate_cluster_profile(cluster_id: int, cluster_stats: dict, exemplar_wallets: list) -> str:
    """Generate an intelligence profile for an entire behavioral cluster."""
    # Summarize cluster centroid features + notable exemplar wallets
    # Output: cluster name suggestion, behavioral description, trading implications
```

**Caching strategy:** LLM-generated narratives cached in Clickhouse with a 24-hour TTL. Regenerated on feature update or explicit request.

---

## 6. Serving Layer

### 6.1 MCP Server

**This is the key differentiator.** Nansen just launched their MCP integration and their CEO called it "a real game-changer." Building an MCP server for this classifier directly demonstrates the ability to contribute to their MCP product.

**Implementation:** TypeScript (Node.js), following the MCP specification from Anthropic.

**Tools exposed:**

```typescript
// Tool 1: Classify a wallet
{
  name: "classify_wallet",
  description: "Classify an Ethereum wallet address into a behavioral archetype (Smart Money, MEV Bot, DeFi Farmer, etc.) with confidence score and feature breakdown.",
  inputSchema: {
    type: "object",
    properties: {
      wallet_address: { type: "string", description: "Ethereum wallet address (0x...)" }
    },
    required: ["wallet_address"]
  }
}

// Tool 2: Get cluster profile
{
  name: "get_cluster_profile",
  description: "Get an AI-generated intelligence profile for a wallet behavioral cluster, including common traits, example wallets, and trading implications.",
  inputSchema: {
    type: "object",
    properties: {
      cluster_id: { type: "number", description: "Cluster ID from classifier output" }
    },
    required: ["cluster_id"]
  }
}

// Tool 3: Find similar wallets  
{
  name: "find_similar_wallets",
  description: "Find wallets with similar behavioral profiles to a given wallet, ranked by cosine similarity in feature space.",
  inputSchema: {
    type: "object",
    properties: {
      wallet_address: { type: "string" },
      top_k: { type: "number", default: 10 }
    },
    required: ["wallet_address"]
  }
}

// Tool 4: Explain wallet behavior
{
  name: "explain_wallet",
  description: "Generate a natural language intelligence briefing for a wallet, explaining its behavioral classification and what it signals.",
  inputSchema: {
    type: "object",
    properties: {
      wallet_address: { type: "string" }
    },
    required: ["wallet_address"]
  }
}
```

**Deployment:** Cloud Run (serverless, auto-scaling, matches Nansen's infrastructure).

### 6.2 Streamlit Dashboard

Visual interface for exploration and demo purposes.

**Pages:**

1. **Cluster Explorer** — UMAP scatter plot colored by cluster, interactive (hover to see wallet details). Cluster-level stats sidebar.

2. **Wallet Lookup** — Input an address, get: classification label, confidence, feature radar chart, AI-generated narrative, list of similar wallets.

3. **Feature Importance** — SHAP waterfall plots showing why a specific wallet was classified the way it was. Global feature importance bar chart.

4. **Model Performance** — Confusion matrix, per-class metrics, calibration curve, training history plots.

**Deployment:** Cloud Run, publicly accessible URL for portfolio.

---

## 7. Pipeline Orchestration

### 7.1 Cloud Composer (Airflow) DAG

```
daily_pipeline_dag:

  [1] extract_bigquery_transactions  (BigQuery → GCS staging)
       │
  [2] run_dbt_transformations        (dbt run — staging → intermediate → mart)
       │
  [3] export_features_to_clickhouse  (BigQuery → Clickhouse sync)
       │
  [4] check_model_drift              (compare current feature distributions to training)
       │
  [5a] retrain_model (if drift detected)  ──► [6] deploy_model_to_cloud_run
       │
  [5b] skip_retrain (if no drift)
       │
  [7] invalidate_llm_cache           (clear stale narratives)
       │
  [8] generate_fresh_cluster_profiles (re-run cluster-level LLM summaries)
```

### 7.2 CI/CD

- **GitHub Actions** for code quality: `ruff` linting, `mypy` type checking, `pytest` unit tests
- **Model CI:** On PR to `main`, run model evaluation on held-out test set. Fail if macro F1 drops >2% from baseline.
- **Deployment:** Merge to `main` triggers Cloud Build → push to Artifact Registry → deploy to Cloud Run

---

## 8. Repository Structure

```
nansen-smart-money-classifier/
├── README.md                          # Project overview, setup, architecture diagram
├── pyproject.toml                     # Python project config (uv/poetry)
├── Makefile                           # Common commands (setup, train, deploy, test)
│
├── dbt/                               # dbt transformation project
│   ├── dbt_project.yml
│   ├── models/
│   │   ├── staging/
│   │   │   ├── stg_wallet_transactions.sql
│   │   │   ├── stg_wallet_token_activity.sql
│   │   │   └── stg_wallet_contract_interactions.sql
│   │   ├── intermediate/
│   │   │   └── int_wallet_behavioral_features.sql
│   │   └── marts/
│   │       └── mart_wallet_feature_vectors.sql
│   └── tests/                         # dbt data tests
│
├── src/
│   ├── data/
│   │   ├── bigquery_extract.py        # BigQuery data extraction scripts
│   │   ├── clickhouse_sync.py         # Feature store sync
│   │   └── ground_truth.py            # Label collection and management
│   │
│   ├── features/
│   │   ├── feature_engineering.py     # Python-side feature computation
│   │   └── feature_store.py           # Clickhouse client wrapper
│   │
│   ├── models/
│   │   ├── clustering.py              # HDBSCAN + UMAP pipeline
│   │   ├── classifier.py              # XGBoost + PyTorch ensemble
│   │   ├── evaluation.py              # Metrics, confusion matrices, SHAP
│   │   └── train.py                   # Training entrypoint
│   │
│   ├── llm/
│   │   ├── insight_generator.py       # Claude API integration
│   │   ├── prompts.py                 # Prompt templates
│   │   └── cache.py                   # Clickhouse narrative cache
│   │
│   └── serving/
│       ├── api.py                     # FastAPI prediction endpoint (backing MCP)
│       └── streamlit_app.py           # Dashboard
│
├── mcp_server/                        # MCP server (TypeScript)
│   ├── package.json
│   ├── tsconfig.json
│   └── src/
│       ├── index.ts                   # MCP server entrypoint
│       ├── tools.ts                   # Tool definitions
│       └── client.ts                  # Calls to Python FastAPI backend
│
├── dags/
│   └── daily_pipeline.py             # Cloud Composer DAG
│
├── tests/
│   ├── test_features.py
│   ├── test_models.py
│   └── test_serving.py
│
├── notebooks/
│   ├── 01_data_exploration.ipynb     # EDA on BigQuery Ethereum data
│   ├── 02_clustering_analysis.ipynb  # UMAP + HDBSCAN experiments
│   ├── 03_classifier_training.ipynb  # Model training + eval
│   └── 04_llm_insights_demo.ipynb   # LLM narrative examples
│
├── Dockerfile                        # Multi-stage: Python API + Streamlit
├── docker-compose.yml                # Local dev: API + Clickhouse + MCP server
├── cloudbuild.yaml                   # GCP Cloud Build config
└── .github/
    └── workflows/
        ├── ci.yml                    # Lint, type check, test
        └── model_eval.yml            # Model regression testing on PR
```

---

## 9. Development Timeline

### Week 1: Data Foundation (Days 1-4)

| Day | Focus | Deliverables |
|---|---|---|
| 1 | GCP project setup, BigQuery access, initial EDA notebook | `01_data_exploration.ipynb`, GCP project configured |
| 2 | dbt project scaffolding, staging models | All `stg_*` models running, tested |
| 3 | Intermediate + mart models, feature engineering | `int_wallet_behavioral_features.sql`, `mart_wallet_feature_vectors.sql` |
| 4 | Clickhouse local setup, feature export pipeline | Docker Clickhouse running, features synced |

### Week 2: ML Pipeline (Days 5-8)

| Day | Focus | Deliverables |
|---|---|---|
| 5 | Clustering: UMAP + HDBSCAN experiments | `02_clustering_analysis.ipynb`, cluster assignments |
| 6 | Ground truth collection, label dataset assembly | `ground_truth.py`, labeled dataset (10K+ wallets) |
| 7 | Classifier training: XGBoost + PyTorch MLP | `03_classifier_training.ipynb`, trained models |
| 8 | Ensemble, evaluation, SHAP analysis | Evaluation report, model artifacts in GCS |

### Week 3: Serving + Polish (Days 9-12)

| Day | Focus | Deliverables |
|---|---|---|
| 9 | FastAPI serving endpoint, LLM insight integration | `api.py`, `insight_generator.py` working |
| 10 | MCP server implementation | `mcp_server/` complete, tested with Claude Desktop |
| 11 | Streamlit dashboard | All 4 pages functional |
| 12 | Cloud Run deployment, README, cleanup | Live URLs, polished repo |

### Stretch Goals (if time permits)

- Cloud Composer DAG for daily pipeline automation
- Extend to Solana (using Helius or public Solana data in BigQuery)
- Model drift monitoring dashboard
- SHAP-based explanation tool in MCP server
- Backtesting: does following "Smart Money" classified wallets generate alpha?

---

## 10. Key Technical Decisions + Rationale

### 10.1 HDBSCAN over K-Means

K-Means requires pre-specifying cluster count and assumes spherical clusters of equal variance. Wallet behavior is messy — some archetypes are rare (MEV bots), some are broad (retail). HDBSCAN handles variable-density clusters and identifies noise/outlier points, which is essential when many wallets don't fit clean categories.

### 10.2 XGBoost + PyTorch Ensemble over Pure Neural Net

For tabular data with <50 features and <500K rows, gradient-boosted trees consistently outperform deep learning (see Grinsztajn et al. 2022, "Why do tree-based models still outperform deep learning on tabular data?"). XGBoost handles the heavy lifting. The PyTorch MLP is included to: (a) demonstrate framework proficiency as the job requires, (b) provide a complementary learner for ensembling, (c) show awareness that pure tree models can miss interaction patterns in certain feature regimes.

### 10.3 MCP Server in TypeScript

Anthropic's MCP SDK reference implementation is in TypeScript. Nansen's MCP server runs on this stack. Writing the server in TypeScript (while the ML backend is Python/FastAPI) mirrors the real-world architecture of ML teams shipping to production through API boundaries.

### 10.4 dbt for Transformations (Not Raw SQL)

Nansen's BI team uses dbt + BigQuery as their core transformation layer. Using dbt demonstrates: (a) awareness of their stack, (b) production data engineering practices (version control, testing, documentation, lineage), (c) the transformation logic is reusable and auditable.

### 10.5 Claude API (Not OpenAI) for LLM Layer

Nansen provides unlimited Claude tokens to their team and explicitly lists Claude alongside GPT-4 as their AI tools. Using Claude for the insight generation layer signals alignment with their tooling preferences and makes the MCP integration more natural (Claude ↔ MCP is the primary supported workflow).

---

## 11. Application Strategy

### 11.1 Portfolio Link

The GitHub repository will be the primary portfolio piece linked in the application. The README should be exceptional — it's the first thing they'll see.

**README structure:**
1. One-line description + badge (live demo, CI status)
2. Architecture diagram (the ASCII art from this PRD, cleaned up)
3. 30-second demo GIF (Streamlit dashboard in action)
4. "Try it yourself" section (MCP server connection instructions for Claude Desktop)
5. Quick start (docker-compose up)
6. Technical deep dive sections with links to notebooks
7. Results summary (model metrics, example cluster profiles)
8. Future work (showing you think bigger than the demo)

### 11.2 "Why You" Answer

The application asks: "Why are you the right candidate for this role?"

The answer should hit:
- **Blockchain engineering depth:** Meta/Novi digital payments (stablecoin infrastructure at scale), Uproot DeFi lending protocol (smart contract development), OTC brokerage + arbitrage/market-making systems (the exact domain Nansen serves)
- **AI/ML in production:** This portfolio project demonstrates end-to-end ML pipeline on blockchain data using their exact stack
- **Data engineering:** Full-stack with data — BigQuery, dbt, Clickhouse, pipeline orchestration
- **AI-first mindset:** Uses Claude as primary development tool, built an MCP server, LLM-native insight generation
- **Location fit:** Based in Bangkok — one of Nansen's four tech hub cities
- **Domain fit:** Been in crypto since 2014 — Bitcoin ATMs, OTC, DeFi, Solana infrastructure. Not a tourist.

### 11.3 Cover Letter Tone

Direct. Technical. No fluff. Show don't tell — link to the live project. Their culture values "speed, ownership, curiosity, courage" and their job listing literally says "Safe return doubtful. Honour and recognition in case of success." (Nansen explorer reference.) Match that energy.

---

## 12. Risk Mitigation

| Risk | Mitigation |
|---|---|
| BigQuery costs spiral | Use `LIMIT` and date partitioning aggressively. 1TB free tier + $300 credit = ~70TB of queries. More than enough. |
| HDBSCAN produces poor clusters | Fallback: K-Means with elbow method / silhouette analysis. Less elegant but functional. |
| Ground truth labels too noisy | Focus on high-confidence labels (verified exchange/protocol addresses) first. Expand to softer labels only if base model performs well. |
| LLM hallucinations in narratives | Constrain prompts to only reference features present in the data. Include confidence scores. Add a disclaimer in output. |
| Clickhouse setup complexity | Docker Clickhouse is one line. For deployment, use Clickhouse Cloud free tier or fall back to BigQuery as the feature store (slower but simpler). |
| MCP server implementation blocky | Start with a minimal 1-tool MCP server, expand tools incrementally. The MCP spec is well-documented. |
| Time overrun | Prioritize: (1) working classifier with BigQuery + dbt, (2) Streamlit dashboard, (3) MCP server. The MCP is high-impact but the dashboard alone is a strong portfolio piece. |

---

## 13. Success Metrics

**For the project itself:**
- Macro F1 ≥ 0.85 on held-out test set
- ≥5 distinct behavioral clusters discovered with clean separation
- <100ms latency for wallet classification via API
- MCP server works with Claude Desktop end-to-end
- Streamlit dashboard loads in <3s with interactive visualizations

**For the application:**
- Interview callback within 2 weeks of submission
- Project cited as a differentiator in interview feedback
- Technical discussion in interview centers on the project (home court advantage)
