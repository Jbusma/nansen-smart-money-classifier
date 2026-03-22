/**
 * MCP Server for the Nansen Smart Money Wallet Classifier.
 *
 * Exposes tools: classify_wallet, get_cluster_profile, find_similar_wallets, explain_wallet.
 * Calls the Python FastAPI backend for actual inference.
 */

import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { z } from "zod";

import {
  classifyWallet,
  explainWallet,
  findSimilarWallets,
  getClusterProfile,
} from "./client.js";

const server = new McpServer({
  name: "nansen-smart-money-classifier",
  version: "0.1.0",
});

// ── Tool: classify_wallet ─────────────────────────────────────────────

server.tool(
  "classify_wallet",
  "Classify an Ethereum wallet into a behavioral archetype (Smart Money, MEV Bot, DeFi Farmer, etc.) with confidence score and feature breakdown.",
  {
    wallet_address: z
      .string()
      .regex(/^0x[a-fA-F0-9]{40}$/)
      .describe("Ethereum wallet address (0x...)"),
  },
  async ({ wallet_address }) => {
    try {
      const result = await classifyWallet(wallet_address);

      const probaText = Object.entries(result.probabilities)
        .sort(([, a], [, b]) => b - a)
        .map(([label, prob]) => `  ${label}: ${(prob * 100).toFixed(1)}%`)
        .join("\n");

      const topFeatures = Object.entries(result.features)
        .slice(0, 8)
        .map(([k, v]) => `  ${k}: ${typeof v === "number" ? v.toFixed(4) : v}`)
        .join("\n");

      return {
        content: [
          {
            type: "text" as const,
            text: [
              `Wallet: ${result.wallet_address}`,
              `Classification: ${result.label.replace(/_/g, " ").toUpperCase()}`,
              `Confidence: ${(result.confidence * 100).toFixed(1)}%`,
              `Latency: ${result.latency_ms.toFixed(1)}ms`,
              "",
              "Class Probabilities:",
              probaText,
              "",
              "Key Features:",
              topFeatures,
            ].join("\n"),
          },
        ],
      };
    } catch (error) {
      return {
        content: [
          {
            type: "text" as const,
            text: `Error classifying wallet: ${error instanceof Error ? error.message : String(error)}`,
          },
        ],
        isError: true,
      };
    }
  }
);

// ── Tool: get_cluster_profile ─────────────────────────────────────────

server.tool(
  "get_cluster_profile",
  "Get an AI-generated intelligence profile for a wallet behavioral cluster.",
  {
    cluster_id: z.number().int().min(0).describe("Cluster ID from classifier output"),
  },
  async ({ cluster_id }) => {
    try {
      const result = await getClusterProfile(cluster_id);

      const topFeatures = Object.entries(result.top_features)
        .map(([k, v]) => `  ${k}: ${v.toFixed(4)}`)
        .join("\n");

      return {
        content: [
          {
            type: "text" as const,
            text: [
              `Cluster ${result.cluster_id} (${result.size} wallets)`,
              "",
              "Profile:",
              result.profile,
              "",
              "Top Features:",
              topFeatures,
              "",
              "Exemplar Wallets:",
              result.exemplar_wallets.map((w) => `  ${w}`).join("\n"),
            ].join("\n"),
          },
        ],
      };
    } catch (error) {
      return {
        content: [
          {
            type: "text" as const,
            text: `Error fetching cluster profile: ${error instanceof Error ? error.message : String(error)}`,
          },
        ],
        isError: true,
      };
    }
  }
);

// ── Tool: find_similar_wallets ────────────────────────────────────────

server.tool(
  "find_similar_wallets",
  "Find wallets with similar behavioral profiles, ranked by cosine similarity.",
  {
    wallet_address: z
      .string()
      .regex(/^0x[a-fA-F0-9]{40}$/)
      .describe("Ethereum wallet address (0x...)"),
    top_k: z
      .number()
      .int()
      .min(1)
      .max(100)
      .default(10)
      .describe("Number of similar wallets to return"),
  },
  async ({ wallet_address, top_k }) => {
    try {
      const result = await findSimilarWallets(wallet_address, top_k);

      const walletsText = result.similar_wallets
        .map(
          (w, i) =>
            `  ${i + 1}. ${w.wallet_address} — ${w.label} (similarity: ${(w.similarity * 100).toFixed(1)}%)`
        )
        .join("\n");

      return {
        content: [
          {
            type: "text" as const,
            text: [
              `Similar wallets to ${result.query_wallet}:`,
              "",
              walletsText,
            ].join("\n"),
          },
        ],
      };
    } catch (error) {
      return {
        content: [
          {
            type: "text" as const,
            text: `Error finding similar wallets: ${error instanceof Error ? error.message : String(error)}`,
          },
        ],
        isError: true,
      };
    }
  }
);

// ── Tool: explain_wallet ──────────────────────────────────────────────

server.tool(
  "explain_wallet",
  "Generate a natural language intelligence briefing for a wallet.",
  {
    wallet_address: z
      .string()
      .regex(/^0x[a-fA-F0-9]{40}$/)
      .describe("Ethereum wallet address (0x...)"),
  },
  async ({ wallet_address }) => {
    try {
      const result = await explainWallet(wallet_address);

      return {
        content: [
          {
            type: "text" as const,
            text: [
              `Wallet: ${result.wallet_address}`,
              `Classification: ${result.label.replace(/_/g, " ").toUpperCase()} (${(result.confidence * 100).toFixed(1)}% confidence)`,
              "",
              "Intelligence Briefing:",
              result.narrative,
            ].join("\n"),
          },
        ],
      };
    } catch (error) {
      return {
        content: [
          {
            type: "text" as const,
            text: `Error explaining wallet: ${error instanceof Error ? error.message : String(error)}`,
          },
        ],
        isError: true,
      };
    }
  }
);

// ── Start server ──────────────────────────────────────────────────────

async function main() {
  const transport = new StdioServerTransport();
  await server.connect(transport);
  console.error("Nansen Smart Money MCP server running on stdio");
}

main().catch((error) => {
  console.error("Fatal error:", error);
  process.exit(1);
});
