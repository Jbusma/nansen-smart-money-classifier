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
  getWalletContext,
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

// ── Tool: get_wallet_context ─────────────────────────────────────────

server.tool(
  "get_wallet_context",
  "Get rich on-chain context for a wallet: transaction summary, top contract interactions with protocol labels (Uniswap, Aave, etc.), token activity, and timing patterns. Use this to understand what a wallet actually does on-chain before generating insights.",
  {
    wallet_address: z
      .string()
      .regex(/^0x[a-fA-F0-9]{40}$/)
      .describe("Ethereum wallet address (0x...)"),
  },
  async ({ wallet_address }) => {
    try {
      const ctx = await getWalletContext(wallet_address);

      const sections: string[] = [
        `Wallet: ${ctx.wallet_address}`,
        "",
      ];

      // Transaction summary
      if (ctx.transaction_summary) {
        const ts = ctx.transaction_summary;
        sections.push(
          "── Transaction Summary ──",
          `  Total transactions: ${ts.total_transactions.toLocaleString()}`,
          `  Total ETH volume: ${ts.total_eth_volume.toFixed(4)} ETH`,
          `  Avg tx value: ${ts.avg_tx_value_eth.toFixed(4)} ETH`,
          `  First seen: ${ts.first_seen ?? "N/A"}`,
          `  Last seen: ${ts.last_seen ?? "N/A"}`,
          ""
        );
      }

      // Top contracts
      if (ctx.top_contracts && ctx.top_contracts.length > 0) {
        sections.push("── Top Contract Interactions ──");
        for (const c of ctx.top_contracts) {
          const label = c.protocol_label
            ? `${c.protocol_label} [${c.category}]`
            : `Unknown contract [${c.category}]`;
          sections.push(
            `  ${c.address.slice(0, 10)}... — ${label}`,
            `    ${c.interaction_count} interactions, ${c.total_eth.toFixed(4)} ETH`
          );
        }
        sections.push("");
      }

      // Token activity
      if (ctx.token_activity) {
        const ta = ctx.token_activity;
        sections.push(
          "── Token Activity ──",
          `  Unique tokens: ${ta.unique_tokens}`
        );
        if (ta.top_tokens.length > 0) {
          sections.push("  Top tokens:");
          for (const t of ta.top_tokens.slice(0, 5)) {
            const type = t.erc721_count > 0 ? "NFT" : "ERC-20";
            sections.push(
              `    ${t.token_address.slice(0, 10)}... — ${t.transfer_count} transfers (${type})`
            );
          }
        }
        sections.push("");
      }

      // Timing patterns
      if (ctx.timing_patterns) {
        const tp = ctx.timing_patterns;
        sections.push(
          "── Timing Patterns ──",
          `  Most active hours (UTC): ${tp.most_active_hours.map((h) => `${h}:00`).join(", ")}`,
          `  Weekday ratio: ${(tp.weekday_ratio * 100).toFixed(1)}%`,
          ""
        );
      }

      return {
        content: [
          {
            type: "text" as const,
            text: sections.join("\n"),
          },
        ],
      };
    } catch (error) {
      return {
        content: [
          {
            type: "text" as const,
            text: `Error fetching wallet context: ${error instanceof Error ? error.message : String(error)}`,
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
