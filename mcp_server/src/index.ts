/**
 * MCP Server for the Nansen Smart Money Wallet Classifier.
 *
 * Exposes tools: classify_wallet, get_cluster_profile, find_similar_wallets, explain_wallet.
 * Calls the Python FastAPI backend for actual inference.
 */

import { createServer } from "node:http";
import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { StreamableHTTPServerTransport } from "@modelcontextprotocol/sdk/server/streamableHttp.js";
import { z } from "zod";

import {
  classifyWallet,
  enrichRegistry,
  explainWallet,
  findSimilarWallets,
  getClusterProfile,
  getWalletContext,
  labelCluster,
  labelWallet,
} from "./client.js";

const SERVER_NAME = "nansen-smart-money-classifier";
const SERVER_VERSION = "0.1.0";

// ── Tool registration ─────────────────────────────────────────────────

function registerTools(server: McpServer): void {
  // ── classify_wallet ───────────────────────────────────────────────
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

  // ── get_cluster_profile ───────────────────────────────────────────
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

  // ── find_similar_wallets ──────────────────────────────────────────
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

  // ── explain_wallet ────────────────────────────────────────────────
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

  // ── get_wallet_context ────────────────────────────────────────────
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

  // ── enrich_registry ─────────────────────────────────────────────────
  server.tool(
    "enrich_registry",
    "Populate the protocol registry with contract labels from free sources (token lists, DeFi Llama, Etherscan). Run this when you see too many 'Unknown contract' labels in wallet context results. Optionally resolve the top-N most interacted unknown contracts via Etherscan.",
    {
      etherscan: z
        .boolean()
        .default(false)
        .describe("Also resolve unknown contracts via Etherscan API (slower, requires API key)"),
      top_n: z
        .number()
        .int()
        .min(1)
        .max(5000)
        .default(500)
        .describe("Number of top unknown contracts to resolve via Etherscan"),
    },
    async ({ etherscan, top_n }) => {
      try {
        const result = await enrichRegistry(etherscan, top_n);

        const lines = [
          "Protocol Registry Enrichment Complete",
          "",
          `  Hardcoded seed: ${result.hardcoded} addresses`,
          `  Token list (CoinGecko): ${result.token_list} addresses`,
          `  DeFi Llama protocols: ${result.defillama} addresses`,
        ];

        if (etherscan) {
          lines.push(`  Etherscan lookups: ${result.etherscan} resolved`);
        }

        lines.push("", `Total registry size: ${result.total_registry_size} addresses`);

        return {
          content: [
            {
              type: "text" as const,
              text: lines.join("\n"),
            },
          ],
        };
      } catch (error) {
        return {
          content: [
            {
              type: "text" as const,
              text: `Error enriching registry: ${error instanceof Error ? error.message : String(error)}`,
            },
          ],
          isError: true,
        };
      }
    }
  );

  // ── label_wallet ────────────────────────────────────────────────────
  server.tool(
    "label_wallet",
    "Assign a behavioral label to a single wallet address. Use this after analyzing a wallet's on-chain context to record your classification decision as ground truth.",
    {
      wallet_address: z
        .string()
        .regex(/^0x[a-fA-F0-9]{40}$/)
        .describe("Ethereum wallet address (0x...)"),
      label: z
        .string()
        .min(1)
        .describe("Behavioral label (e.g. 'defi_lender', 'dex_trader', 'institutional_otc')"),
      confidence: z
        .number()
        .min(0)
        .max(1)
        .describe("Confidence in the label (0.0 to 1.0)"),
      evidence: z
        .string()
        .default("")
        .describe("Brief explanation of why this label was chosen"),
    },
    async ({ wallet_address, label, confidence, evidence }) => {
      try {
        const result = await labelWallet(wallet_address, label, confidence, evidence);
        return {
          content: [
            {
              type: "text" as const,
              text: `Labeled ${wallet_address} as "${result.label}" (${result.labeled} wallet written to ground truth)`,
            },
          ],
        };
      } catch (error) {
        return {
          content: [
            {
              type: "text" as const,
              text: `Error labeling wallet: ${error instanceof Error ? error.message : String(error)}`,
            },
          ],
          isError: true,
        };
      }
    }
  );

  // ── label_cluster ───────────────────────────────────────────────────
  server.tool(
    "label_cluster",
    "Bulk-label ALL wallets in a cluster with a single behavioral label. Use this after profiling a cluster to apply labels to all its members at once. Cluster IDs: 0, 1, 2, or -1 for noise.",
    {
      cluster_id: z
        .number()
        .int()
        .min(-1)
        .describe("Cluster ID (0, 1, 2, or -1 for noise)"),
      label: z
        .string()
        .min(1)
        .describe("Behavioral label to apply to all wallets in the cluster"),
      confidence: z
        .number()
        .min(0)
        .max(1)
        .describe("Confidence in the label (0.0 to 1.0)"),
      evidence: z
        .string()
        .default("")
        .describe("Summary of cluster behavioral profile supporting this label"),
    },
    async ({ cluster_id, label, confidence, evidence }) => {
      try {
        const result = await labelCluster(cluster_id, label, confidence, evidence);
        return {
          content: [
            {
              type: "text" as const,
              text: `Labeled cluster ${cluster_id}: ${result.labeled} wallets assigned label "${result.label}"`,
            },
          ],
        };
      } catch (error) {
        return {
          content: [
            {
              type: "text" as const,
              text: `Error labeling cluster: ${error instanceof Error ? error.message : String(error)}`,
            },
          ],
          isError: true,
        };
      }
    }
  );
}

// ── Start server ──────────────────────────────────────────────────────

const MCP_PORT = parseInt(process.env.MCP_PORT || "3001", 10);
const USE_STDIO = process.argv.includes("--stdio");

async function startStdio() {
  const server = new McpServer({ name: SERVER_NAME, version: SERVER_VERSION });
  registerTools(server);
  const transport = new StdioServerTransport();
  await server.connect(transport);
  console.error("Nansen Smart Money MCP server running on stdio");
}

async function startHttp() {
  const httpServer = createServer(async (req, res) => {
    const url = new URL(req.url ?? "/", `http://localhost:${MCP_PORT}`);

    if (url.pathname !== "/mcp") {
      res.writeHead(404);
      res.end("Not Found");
      return;
    }

    try {
      // Stateless mode: fresh server + transport per request.
      // McpServer.connect() binds to a single transport, so each
      // concurrent request needs its own server instance.
      const reqServer = new McpServer({
        name: SERVER_NAME,
        version: SERVER_VERSION,
      });
      registerTools(reqServer);

      const transport = new StreamableHTTPServerTransport({
        sessionIdGenerator: undefined,
      });
      await reqServer.connect(transport);
      await transport.handleRequest(req, res);
      await reqServer.close();
    } catch (err) {
      console.error("MCP transport error:", err);
      if (!res.headersSent) {
        res.writeHead(500, { "Content-Type": "application/json" });
        res.end(JSON.stringify({ error: "Internal server error" }));
      }
    }
  });

  httpServer.listen(MCP_PORT, () => {
    console.error(`Nansen Smart Money MCP server listening on port ${MCP_PORT}`);
  });
}

(USE_STDIO ? startStdio() : startHttp()).catch((error) => {
  console.error("Fatal error:", error);
  process.exit(1);
});
