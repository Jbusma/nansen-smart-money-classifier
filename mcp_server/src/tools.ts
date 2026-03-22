/**
 * MCP tool definitions for the Smart Money Classifier.
 */

import { z } from "zod";

export const TOOL_DEFINITIONS = [
  {
    name: "classify_wallet",
    description:
      "Classify an Ethereum wallet address into a behavioral archetype " +
      "(Smart Money, MEV Bot, DeFi Farmer, Airdrop Hunter, Retail Trader, " +
      "HODLer, NFT Trader) with confidence score and feature breakdown.",
    inputSchema: {
      type: "object" as const,
      properties: {
        wallet_address: {
          type: "string",
          description: "Ethereum wallet address (0x...)",
        },
      },
      required: ["wallet_address"],
    },
  },
  {
    name: "get_cluster_profile",
    description:
      "Get an AI-generated intelligence profile for a wallet behavioral cluster, " +
      "including common traits, example wallets, and trading implications.",
    inputSchema: {
      type: "object" as const,
      properties: {
        cluster_id: {
          type: "number",
          description: "Cluster ID from classifier output",
        },
      },
      required: ["cluster_id"],
    },
  },
  {
    name: "find_similar_wallets",
    description:
      "Find wallets with similar behavioral profiles to a given wallet, " +
      "ranked by cosine similarity in feature space.",
    inputSchema: {
      type: "object" as const,
      properties: {
        wallet_address: {
          type: "string",
          description: "Ethereum wallet address (0x...)",
        },
        top_k: {
          type: "number",
          description: "Number of similar wallets to return (default 10)",
        },
      },
      required: ["wallet_address"],
    },
  },
  {
    name: "explain_wallet",
    description:
      "Generate a natural language intelligence briefing for a wallet, " +
      "explaining its behavioral classification and what it signals.",
    inputSchema: {
      type: "object" as const,
      properties: {
        wallet_address: {
          type: "string",
          description: "Ethereum wallet address (0x...)",
        },
      },
      required: ["wallet_address"],
    },
  },
];

// Zod schemas for input validation
export const ClassifyInputSchema = z.object({
  wallet_address: z.string().regex(/^0x[a-fA-F0-9]{40}$/),
});

export const ClusterProfileInputSchema = z.object({
  cluster_id: z.number().int().min(0),
});

export const FindSimilarInputSchema = z.object({
  wallet_address: z.string().regex(/^0x[a-fA-F0-9]{40}$/),
  top_k: z.number().int().min(1).max(100).default(10),
});

export const ExplainInputSchema = z.object({
  wallet_address: z.string().regex(/^0x[a-fA-F0-9]{40}$/),
});
