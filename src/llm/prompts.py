"""Prompt templates for LLM-powered wallet narratives and cluster profiles."""

WALLET_NARRATIVE_PROMPT = """\
You are a crypto intelligence analyst. Given the on-chain profile below, write a
2-3 sentence intelligence briefing suitable for a crypto investor.

Wallet: {wallet_address}
Assigned Label: {label} (confidence: {confidence:.1%})

On-chain features:
{features}

Focus on what this wallet's behavior reveals about the operator's strategy,
risk appetite, and likely role in the ecosystem. Be specific and actionable;
avoid generic statements.
"""

CLUSTER_PROFILE_PROMPT = """\
You are a crypto intelligence analyst specializing in behavioral clustering.

Cluster ID: {cluster_id}

Aggregate cluster statistics:
{cluster_stats}

Exemplar wallets (most representative members):
{exemplar_wallets}

Provide the following in valid JSON:
1. "cluster_name": A short, memorable name for this cluster (2-4 words).
2. "behavioral_description": A paragraph describing the shared behavioral
   pattern of wallets in this cluster.
3. "trading_implications": 2-3 bullet points (as a list of strings) explaining
   what it means for other market participants when this cluster is active.
"""

WALLET_COMPARISON_PROMPT = """\
You are a crypto intelligence analyst. Compare the two wallet profiles below
and produce a concise analysis.

Wallet A features:
{wallet_a_features}

Wallet B features:
{wallet_b_features}

Your analysis should cover:
1. Key similarities in on-chain behavior.
2. Key differences and what they imply about each operator's strategy.
3. Whether these wallets could plausibly be controlled by the same entity.
Keep the response under 200 words.
"""
