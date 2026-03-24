/**
 * HTTP client for the Python FastAPI backend.
 */

const API_BASE = process.env.API_URL || "http://localhost:8000";

interface ClassifyResult {
  wallet_address: string;
  label: string;
  confidence: number;
  probabilities: Record<string, number>;
  features: Record<string, number>;
  latency_ms: number;
}

interface ExplainResult {
  wallet_address: string;
  label: string;
  confidence: number;
  narrative: string;
  features: Record<string, number>;
}

interface SimilarWallet {
  wallet_address: string;
  similarity: number;
  label: string;
}

interface SimilarWalletsResult {
  query_wallet: string;
  similar_wallets: SimilarWallet[];
}

interface ClusterProfileResult {
  cluster_id: number;
  size: number;
  profile: string;
  top_features: Record<string, number>;
  exemplar_wallets: string[];
}

interface ContractInteraction {
  address: string;
  protocol_label: string | null;
  category: string;
  interaction_count: number;
  total_eth: number;
}

interface TokenSummary {
  token_address: string;
  transfer_count: number;
  erc20_count: number;
  erc721_count: number;
}

interface WalletContextResult {
  wallet_address: string;
  transaction_summary: {
    total_transactions: number;
    total_eth_volume: number;
    avg_tx_value_eth: number;
    first_seen: string | null;
    last_seen: string | null;
  } | null;
  top_contracts: ContractInteraction[] | null;
  token_activity: {
    unique_tokens: number;
    top_tokens: TokenSummary[];
  } | null;
  timing_patterns: {
    most_active_hours: number[];
    weekday_ratio: number;
    hourly_distribution: number[];
  } | null;
}

async function apiCall<T>(
  endpoint: string,
  method: "GET" | "POST" = "GET",
  body?: Record<string, unknown>
): Promise<T> {
  const url = `${API_BASE}${endpoint}`;
  const options: RequestInit = {
    method,
    headers: { "Content-Type": "application/json" },
  };
  if (body) {
    options.body = JSON.stringify(body);
  }

  const response = await fetch(url, options);
  if (!response.ok) {
    const text = await response.text();
    throw new Error(`API error (${response.status}): ${text}`);
  }
  return response.json() as Promise<T>;
}

export async function classifyWallet(
  walletAddress: string
): Promise<ClassifyResult> {
  return apiCall<ClassifyResult>("/classify", "POST", {
    wallet_address: walletAddress,
  });
}

export async function explainWallet(
  walletAddress: string
): Promise<ExplainResult> {
  return apiCall<ExplainResult>("/explain", "POST", {
    wallet_address: walletAddress,
  });
}

export async function findSimilarWallets(
  walletAddress: string,
  topK: number = 10
): Promise<SimilarWalletsResult> {
  return apiCall<SimilarWalletsResult>("/similar", "POST", {
    wallet_address: walletAddress,
    top_k: topK,
  });
}

export async function getClusterProfile(
  clusterId: number
): Promise<ClusterProfileResult> {
  return apiCall<ClusterProfileResult>(`/cluster/${clusterId}`);
}

export async function getWalletContext(
  walletAddress: string
): Promise<WalletContextResult> {
  return apiCall<WalletContextResult>(
    `/wallet/${walletAddress}/context`
  );
}

interface EnrichResult {
  hardcoded: number;
  token_list: number;
  defillama: number;
  etherscan: number;
  total_registry_size: number;
}

export async function enrichRegistry(
  etherscan: boolean = false,
  topN: number = 500
): Promise<EnrichResult> {
  return apiCall<EnrichResult>("/enrich", "POST", {
    etherscan,
    top_n: topN,
  });
}

interface LabelResult {
  labeled: number;
  label: string;
}

export async function labelWallet(
  walletAddress: string,
  label: string,
  confidence: number,
  evidence: string
): Promise<LabelResult> {
  return apiCall<LabelResult>("/label/wallet", "POST", {
    wallet_address: walletAddress,
    label,
    confidence,
    evidence,
    source: "agent_verified",
  });
}

export async function labelCluster(
  clusterId: number,
  label: string,
  confidence: number,
  evidence: string
): Promise<LabelResult> {
  return apiCall<LabelResult>("/label/cluster", "POST", {
    cluster_id: clusterId,
    label,
    confidence,
    evidence,
    source: "agent_cluster_label",
  });
}
