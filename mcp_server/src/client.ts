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
