"""Streamlit dashboard for Smart Money Classifier exploration.

Run with: streamlit run src/serving/streamlit_app.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
from sklearn.metrics import silhouette_samples

from src.models.cluster_analysis import build_cluster_profiles
from src.models.clustering import ClusteringPipeline

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Nansen Smart Money Classifier",
    page_icon="🔍",
    layout="wide",
)

API_URL = "http://localhost:8000"

FEATURES_PATH = Path("data/features.parquet")
PIPELINE_PATH = Path("models/artifacts/clustering_pipeline.joblib")
LABELS_PATH = Path("models/artifacts/cluster_labels.json")
GROUND_TRUTH_PATH = Path("data/ground_truth.parquet")

CLUSTER_COLORS = px.colors.qualitative.Set2


# ---------------------------------------------------------------------------
# Data loading (cached)
# ---------------------------------------------------------------------------


@st.cache_data
def load_features() -> pd.DataFrame:
    return pd.read_parquet(FEATURES_PATH)


@st.cache_resource
def load_pipeline() -> ClusteringPipeline:
    return ClusteringPipeline.load(PIPELINE_PATH)


@st.cache_data
def load_cluster_labels() -> dict[str, str]:
    if LABELS_PATH.exists():
        with open(LABELS_PATH) as f:
            result: dict[str, str] = json.load(f)
            return result
    return {}


@st.cache_data
def load_ground_truth() -> pd.DataFrame | None:
    if GROUND_TRUTH_PATH.exists():
        return pd.read_parquet(GROUND_TRUTH_PATH)
    return None


def save_cluster_labels(labels: dict[str, str]) -> None:
    LABELS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LABELS_PATH, "w") as f:
        json.dump(labels, f, indent=2)


def api_call(
    endpoint: str,
    method: str = "GET",
    payload: dict | None = None,  # type: ignore[type-arg]
) -> dict | None:  # type: ignore[type-arg]
    """Helper to call the FastAPI backend."""
    try:
        if method == "GET":
            resp = requests.get(f"{API_URL}{endpoint}", timeout=30)
        else:
            resp = requests.post(f"{API_URL}{endpoint}", json=payload, timeout=30)
        resp.raise_for_status()
        result: dict = resp.json()  # type: ignore[assignment]
        return result
    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to API server. Start the API first.")
        return None
    except requests.exceptions.HTTPError as e:
        st.error(f"API error: {e.response.text}")
        return None


# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------

st.sidebar.title("Smart Money Classifier")
page = st.sidebar.radio(
    "Navigate",
    ["Cluster Explorer", "Wallet Lookup", "Feature Importance", "Model Performance"],
)


# =========================================================================
# Page: Cluster Explorer
# =========================================================================
if page == "Cluster Explorer":
    st.header("Cluster Explorer")

    if not FEATURES_PATH.exists() or not PIPELINE_PATH.exists():
        st.info(
            "Clustering artifacts not found. Run the clustering pipeline first:\n\n"
            "```bash\npython -m src.models.clustering\n```"
        )
    else:
        features_df = load_features()
        pipeline = load_pipeline()

        if pipeline.labels_ is None or pipeline.embedding_ is None:
            st.error("Clustering pipeline has no fitted data. Run clustering first.")
        else:
            labels = pipeline.labels_
            embedding = pipeline.embedding_
            wallet_addresses = features_df["wallet_address"] if "wallet_address" in features_df.columns else None
            numeric_features = features_df.select_dtypes(include=[np.number])
            feature_cols = list(numeric_features.columns)

            # Load any saved cluster names
            saved_labels = load_cluster_labels()
            unique_clusters = sorted(set(labels))

            # Build cluster name mapping
            cluster_names: dict[int, str] = {}
            for c in unique_clusters:
                key = str(c)
                if key in saved_labels:
                    cluster_names[c] = saved_labels[key]
                else:
                    cluster_names[c] = "Noise" if c == -1 else f"Cluster {c}"

            # Assign cluster info to dataframe
            viz_df = numeric_features.copy()
            viz_df["cluster_id"] = labels
            viz_df["cluster_name"] = [cluster_names[c] for c in labels]
            viz_df["UMAP_1"] = embedding[:, 0]
            viz_df["UMAP_2"] = embedding[:, 1]
            if wallet_addresses is not None:
                viz_df["wallet_address"] = wallet_addresses.values

            # ---- Sidebar: cluster labeling ----
            st.sidebar.markdown("---")
            st.sidebar.subheader("Label Clusters")
            st.sidebar.caption("Name each cluster based on the behavioral patterns you see.")

            new_labels: dict[str, str] = {}
            for c in unique_clusters:
                if c == -1:
                    new_labels[str(c)] = "Noise"
                    continue
                current = cluster_names.get(c, f"Cluster {c}")
                new_name = st.sidebar.text_input(
                    f"Cluster {c} ({(labels == c).sum()} wallets)",
                    value=current,
                    key=f"label_{c}",
                )
                new_labels[str(c)] = new_name

            if st.sidebar.button("Save Labels"):
                save_cluster_labels(new_labels)
                st.sidebar.success("Labels saved!")
                st.cache_data.clear()

            # ---- Cluster size metrics ----
            cluster_sizes = pd.Series(labels).value_counts().sort_index()
            metric_cols = st.columns(len(unique_clusters))
            for i, c in enumerate(unique_clusters):
                name = new_labels.get(str(c), cluster_names[c])
                metric_cols[i].metric(name, f"{cluster_sizes[c]:,} wallets")

            # ---- Tab layout ----
            (
                tab_scatter,
                tab_profiles,
                tab_compare,
                tab_parallel,
                tab_silhouette,
                tab_pairs,
                tab_labels,
                tab_wallets,
            ) = st.tabs(
                [
                    "UMAP Scatter",
                    "Cluster Profiles",
                    "Distributions",
                    "Parallel Coords",
                    "Silhouette",
                    "Feature Pairs",
                    "Label Validation",
                    "Explore Wallets",
                ]
            )

            # ---- Tab 1: UMAP Scatter ----
            with tab_scatter:
                color_by = st.selectbox(
                    "Color by",
                    ["Cluster"] + feature_cols,
                    index=0,
                )

                if color_by == "Cluster":
                    hover = ["wallet_address"] + feature_cols[:6] if wallet_addresses is not None else feature_cols[:6]
                    fig = px.scatter(
                        viz_df,
                        x="UMAP_1",
                        y="UMAP_2",
                        color="cluster_name",
                        hover_data=hover,
                        title="UMAP Embedding — HDBSCAN Clusters",
                        opacity=0.6,
                        color_discrete_sequence=CLUSTER_COLORS,
                    )
                else:
                    hover = ["wallet_address", "cluster_name"] if wallet_addresses is not None else ["cluster_name"]
                    fig = px.scatter(
                        viz_df,
                        x="UMAP_1",
                        y="UMAP_2",
                        color=color_by,
                        hover_data=hover,
                        title=f"UMAP Embedding — colored by {color_by}",
                        opacity=0.6,
                        color_continuous_scale="Viridis",
                    )

                fig.update_traces(marker={"size": 5})
                fig.update_layout(height=650)
                st.plotly_chart(fig, use_container_width=True)

            # ---- Tab 2: Cluster Profiles ----
            with tab_profiles:
                profiles = build_cluster_profiles(numeric_features, labels)

                # Heatmap
                st.subheader("Feature Heatmap (ratio to global mean)")
                pivot = profiles.pivot(index="cluster", columns="feature", values="ratio_to_global")
                pivot.index = [cluster_names.get(i, f"Cluster {i}") for i in pivot.index]

                fig_heat = go.Figure(
                    data=go.Heatmap(
                        z=pivot.values,
                        x=pivot.columns.tolist(),
                        y=pivot.index.tolist(),
                        colorscale="RdYlGn",
                        zmin=0,
                        zmax=3,
                        text=np.round(pivot.values, 2),
                        texttemplate="%{text}",
                        textfont={"size": 11},
                    )
                )
                fig_heat.update_layout(height=300, margin={"t": 30, "b": 0})
                st.plotly_chart(fig_heat, use_container_width=True)

                # Radar per cluster
                st.subheader("Radar Charts")
                selected_cluster = st.selectbox(
                    "Select cluster",
                    [cluster_names.get(c, f"Cluster {c}") for c in unique_clusters],
                    index=0,
                )
                selected_id = next(
                    c for c in unique_clusters if cluster_names.get(c, f"Cluster {c}") == selected_cluster
                )
                cluster_profile = profiles[profiles["cluster"] == selected_id].sort_values("feature")

                fig_radar = go.Figure()
                fig_radar.add_trace(
                    go.Scatterpolar(
                        r=[min(v, 5.0) for v in cluster_profile["ratio_to_global"]],
                        theta=cluster_profile["feature"].tolist(),
                        fill="toself",
                        name=selected_cluster,
                    )
                )
                fig_radar.add_trace(
                    go.Scatterpolar(
                        r=[1.0] * len(cluster_profile),
                        theta=cluster_profile["feature"].tolist(),
                        name="Global mean",
                        line={"dash": "dash", "color": "gray"},
                    )
                )
                fig_radar.update_layout(
                    polar={"radialaxis": {"range": [0, 5]}},
                    height=500,
                    title=f"{selected_cluster} — Feature Profile",
                )
                st.plotly_chart(fig_radar, use_container_width=True)

            # ---- Tab 3: Distributions (violin + box) ----
            with tab_compare:
                st.subheader("Feature Distributions by Cluster")
                col_feat, col_chart = st.columns([2, 1])
                with col_feat:
                    compare_feature = st.selectbox("Select feature", feature_cols, index=0)
                with col_chart:
                    chart_type = st.radio("Chart type", ["Violin", "Box"], horizontal=True)

                if chart_type == "Violin":
                    fig_dist = px.violin(
                        viz_df,
                        x="cluster_name",
                        y=compare_feature,
                        color="cluster_name",
                        box=True,
                        points="outliers",
                        color_discrete_sequence=CLUSTER_COLORS,
                        title=f"Distribution of {compare_feature} by cluster",
                    )
                else:
                    fig_dist = px.box(
                        viz_df,
                        x="cluster_name",
                        y=compare_feature,
                        color="cluster_name",
                        color_discrete_sequence=CLUSTER_COLORS,
                        title=f"Distribution of {compare_feature} by cluster",
                    )
                fig_dist.update_layout(height=500)
                st.plotly_chart(fig_dist, use_container_width=True)

                st.subheader("Summary Statistics")
                summary = viz_df.groupby("cluster_name")[feature_cols].agg(["mean", "median", "std"])
                summary.columns = [f"{col}_{stat}" for col, stat in summary.columns]
                st.dataframe(summary.T, use_container_width=True)

            # ---- Tab 4: Parallel Coordinates ----
            with tab_parallel:
                st.subheader("Parallel Coordinates — All Features at Once")
                st.caption("Each line is a wallet. Drag along axes to filter ranges.")

                norm_df = viz_df[feature_cols].copy()
                for col in feature_cols:
                    col_min = norm_df[col].min()
                    col_max = norm_df[col].max()
                    if col_max > col_min:
                        norm_df[col] = (norm_df[col] - col_min) / (col_max - col_min)
                    else:
                        norm_df[col] = 0.0
                norm_df["cluster_id"] = viz_df["cluster_id"]

                dimensions = [
                    {
                        "label": col.replace("_", " ").title(),
                        "values": norm_df[col],
                        "range": [0, 1],
                    }
                    for col in feature_cols
                ]

                non_noise = [c for c in unique_clusters if c != -1]
                color_map = {c: i for i, c in enumerate(non_noise)}
                color_map[-1] = len(non_noise)
                color_vals = [color_map[c] for c in norm_df["cluster_id"]]

                n_colors = max(len(unique_clusters), 1)
                fig_pc = go.Figure(
                    data=go.Parcoords(
                        line={
                            "color": color_vals,
                            "colorscale": [
                                [
                                    i / max(n_colors - 1, 1),
                                    CLUSTER_COLORS[i % len(CLUSTER_COLORS)],
                                ]
                                for i in range(n_colors)
                            ],
                            "showscale": False,
                        },
                        dimensions=dimensions,
                    )
                )
                fig_pc.update_layout(height=600, margin={"t": 40, "b": 30})
                st.plotly_chart(fig_pc, use_container_width=True)

                legend_cols = st.columns(len(unique_clusters))
                for i, c in enumerate(unique_clusters):
                    name = new_labels.get(str(c), cluster_names[c])
                    color = CLUSTER_COLORS[i % len(CLUSTER_COLORS)]
                    legend_cols[i].markdown(
                        f'<span style="color:{color}">&#9632;</span> {name}',
                        unsafe_allow_html=True,
                    )

            # ---- Tab 5: Silhouette Analysis ----
            with tab_silhouette:
                st.subheader("Silhouette Analysis — Per-Wallet Cluster Fit")
                st.caption(
                    "Values near 1 = well-matched to cluster. Near 0 = on boundary. Negative = possibly misassigned."
                )

                valid_mask = labels != -1
                n_valid_clusters = len(set(labels[valid_mask]))
                if valid_mask.sum() > 1 and n_valid_clusters >= 2:
                    sil_values = silhouette_samples(embedding[valid_mask], labels[valid_mask])
                    sil_df = pd.DataFrame(
                        {
                            "silhouette": sil_values,
                            "cluster_id": labels[valid_mask],
                            "cluster_name": [cluster_names[c] for c in labels[valid_mask]],
                        }
                    )

                    overall_sil = float(np.mean(sil_values))
                    st.metric("Overall Silhouette Score", f"{overall_sil:.3f}")

                    sil_df = sil_df.sort_values(["cluster_id", "silhouette"], ascending=[True, False])
                    sil_df["wallet_idx"] = range(len(sil_df))

                    fig_sil = go.Figure()
                    y_offset = 0
                    tick_positions = []
                    tick_labels_list = []

                    for c in sorted(set(labels[valid_mask])):
                        cluster_sil = sil_df[sil_df["cluster_id"] == c]["silhouette"].values
                        cluster_sil_sorted = np.sort(cluster_sil)[::-1]
                        color = CLUSTER_COLORS[c % len(CLUSTER_COLORS)]
                        name = cluster_names.get(c, f"Cluster {c}")
                        avg = float(np.mean(cluster_sil))

                        fig_sil.add_trace(
                            go.Bar(
                                y=list(
                                    range(
                                        y_offset,
                                        y_offset + len(cluster_sil_sorted),
                                    )
                                ),
                                x=cluster_sil_sorted,
                                orientation="h",
                                marker_color=color,
                                name=f"{name} (avg={avg:.3f})",
                                showlegend=True,
                            )
                        )
                        tick_positions.append(y_offset + len(cluster_sil_sorted) // 2)
                        tick_labels_list.append(name)
                        y_offset += len(cluster_sil_sorted) + 10

                    fig_sil.add_vline(
                        x=overall_sil,
                        line_dash="dash",
                        line_color="red",
                        annotation_text=f"Mean: {overall_sil:.3f}",
                    )
                    fig_sil.update_layout(
                        height=max(400, y_offset * 1.5),
                        yaxis={
                            "tickvals": tick_positions,
                            "ticktext": tick_labels_list,
                        },
                        xaxis_title="Silhouette Coefficient",
                        bargap=0,
                    )
                    st.plotly_chart(fig_sil, use_container_width=True)

                    st.subheader("Per-Cluster Silhouette Stats")
                    sil_stats = sil_df.groupby("cluster_name")["silhouette"].agg(
                        ["mean", "median", "std", "min", "max", "count"]
                    )
                    sil_stats.columns = [
                        "Mean",
                        "Median",
                        "Std",
                        "Min",
                        "Max",
                        "Count",
                    ]
                    fmt_cols = ["Mean", "Median", "Std", "Min", "Max"]
                    st.dataframe(
                        sil_stats.style.format("{:.3f}", subset=fmt_cols),
                        use_container_width=True,
                    )
                else:
                    st.warning("Need at least 2 non-noise clusters for silhouette analysis.")

            # ---- Tab 6: Feature Pair Scatter Matrix ----
            with tab_pairs:
                st.subheader("Feature Pair Scatter Matrix")
                st.caption("Select features to see pairwise relationships colored by cluster.")

                profiles = build_cluster_profiles(numeric_features, labels)
                feature_variance = profiles.groupby("feature")["ratio_to_global"].std().sort_values(ascending=False)
                top_discriminating = feature_variance.head(6).index.tolist()

                selected_features = st.multiselect(
                    "Features to plot (2-6 recommended)",
                    feature_cols,
                    default=top_discriminating[:4],
                )

                if len(selected_features) >= 2:
                    fig_matrix = px.scatter_matrix(
                        viz_df,
                        dimensions=selected_features,
                        color="cluster_name",
                        color_discrete_sequence=CLUSTER_COLORS,
                        opacity=0.5,
                        title="Feature Pair Scatter Matrix",
                    )
                    fig_matrix.update_traces(diagonal_visible=True, marker={"size": 3})
                    size = max(250 * len(selected_features), 600)
                    fig_matrix.update_layout(height=size, width=size)
                    st.plotly_chart(fig_matrix, use_container_width=True)
                else:
                    st.info("Select at least 2 features to render the scatter matrix.")

            # ---- Tab 7: Label Validation (Heuristic Overlay) ----
            with tab_labels:
                st.subheader("Heuristic Label Overlay on UMAP")
                st.caption(
                    "Overlays rule-based labels (MEV, whale, DeFi, etc.) onto the UMAP embedding to validate clusters."
                )

                gt_df = load_ground_truth()
                if gt_df is not None and wallet_addresses is not None:
                    gt_map = dict(zip(gt_df["address"], gt_df["label"], strict=False))
                    viz_df["heuristic_label"] = viz_df["wallet_address"].map(gt_map).fillna("unlabeled")

                    label_counts = viz_df["heuristic_label"].value_counts()
                    st.markdown("**Heuristic label distribution:**")
                    lbl_cols = st.columns(min(len(label_counts), 6))
                    for i, (lbl, cnt) in enumerate(label_counts.items()):
                        lbl_cols[i % len(lbl_cols)].metric(str(lbl), f"{cnt:,}")

                    color_mode = st.radio(
                        "Color by",
                        ["Heuristic Label", "Cluster (with label shape)"],
                        horizontal=True,
                        key="label_overlay_mode",
                    )

                    if color_mode == "Heuristic Label":
                        fig_overlay = px.scatter(
                            viz_df,
                            x="UMAP_1",
                            y="UMAP_2",
                            color="heuristic_label",
                            hover_data=["wallet_address", "cluster_name"],
                            title="UMAP — Colored by Heuristic Labels",
                            opacity=0.6,
                            color_discrete_sequence=px.colors.qualitative.D3,
                        )
                    else:
                        fig_overlay = px.scatter(
                            viz_df,
                            x="UMAP_1",
                            y="UMAP_2",
                            color="cluster_name",
                            symbol="heuristic_label",
                            hover_data=[
                                "wallet_address",
                                "heuristic_label",
                            ],
                            title="UMAP — Cluster color + Heuristic label shape",
                            opacity=0.6,
                            color_discrete_sequence=CLUSTER_COLORS,
                        )
                    fig_overlay.update_traces(marker={"size": 5})
                    fig_overlay.update_layout(height=650)
                    st.plotly_chart(fig_overlay, use_container_width=True)

                    st.subheader("Cluster vs Heuristic Label Cross-Tab")
                    crosstab = pd.crosstab(
                        viz_df["cluster_name"],
                        viz_df["heuristic_label"],
                        margins=True,
                    )
                    st.dataframe(crosstab, use_container_width=True)
                else:
                    st.info(
                        "No ground truth data found. Run "
                        "`python -m src.data.ground_truth` to generate "
                        "heuristic labels, then reload."
                    )

            # ---- Tab 8: Explore Wallets ----
            with tab_wallets:
                st.subheader("Individual Wallet Explorer")

                filter_cluster = st.selectbox(
                    "Filter by cluster",
                    ["All"] + [cluster_names.get(c, f"Cluster {c}") for c in unique_clusters if c != -1],
                    index=0,
                )

                display_df = viz_df.copy()
                if filter_cluster != "All":
                    display_df = display_df[display_df["cluster_name"] == filter_cluster]

                sort_col = st.selectbox("Sort by", feature_cols, index=0)
                sort_asc = st.checkbox("Ascending", value=False)
                display_df = display_df.sort_values(sort_col, ascending=sort_asc)

                if wallet_addresses is not None:
                    show_cols = ["wallet_address", "cluster_name"] + feature_cols
                else:
                    show_cols = ["cluster_name"] + feature_cols
                st.dataframe(
                    display_df[show_cols].head(100),
                    use_container_width=True,
                    height=500,
                )

                st.caption(f"Showing top 100 of {len(display_df):,} wallets. Sort by different features to explore.")


# =========================================================================
# Page: Wallet Lookup
# =========================================================================
elif page == "Wallet Lookup":
    st.header("Wallet Lookup")

    wallet = st.text_input("Ethereum Address", placeholder="0x...")

    if wallet and len(wallet) == 42 and wallet.startswith("0x"):
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Classification")
            result = api_call("/classify", method="POST", payload={"wallet_address": wallet})
            if result:
                st.metric("Label", result["label"].replace("_", " ").title())
                st.metric("Confidence", f"{result['confidence']:.1%}")
                st.caption(f"Latency: {result['latency_ms']:.1f}ms")

                proba_df = pd.DataFrame(
                    list(result["probabilities"].items()),
                    columns=["Label", "Probability"],
                )
                fig = px.bar(
                    proba_df,
                    x="Label",
                    y="Probability",
                    title="Class Probabilities",
                )
                st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("Feature Profile")
            if result:
                features = result["features"]
                radar_features = [
                    "tx_frequency_per_day",
                    "activity_regularity",
                    "dex_to_total_ratio",
                    "burst_score",
                    "counterparty_concentration",
                    "value_velocity",
                ]
                available = {k: features.get(k, 0) for k in radar_features if k in features}
                if available:
                    fig = go.Figure(
                        data=go.Scatterpolar(
                            r=list(available.values()),
                            theta=[k.replace("_", " ").title() for k in available],
                            fill="toself",
                        )
                    )
                    fig.update_layout(
                        title="Behavioral Radar",
                        polar={"radialaxis": {"visible": True}},
                    )
                    st.plotly_chart(fig, use_container_width=True)

        st.subheader("AI Intelligence Briefing")
        explain = api_call("/explain", method="POST", payload={"wallet_address": wallet})
        if explain:
            st.markdown(f"> {explain['narrative']}")

        st.subheader("Similar Wallets")
        similar = api_call(
            "/similar",
            method="POST",
            payload={"wallet_address": wallet, "top_k": 10},
        )
        if similar:
            sim_df = pd.DataFrame(similar["similar_wallets"])
            st.dataframe(sim_df, use_container_width=True)


# =========================================================================
# Page: Feature Importance
# =========================================================================
elif page == "Feature Importance":
    st.header("Feature Importance")
    st.markdown("SHAP-based feature importance analysis.")

    import joblib

    shap_path = Path("models/artifacts/shap_values.joblib")
    if shap_path.exists():
        shap_data = joblib.load(shap_path)
        feature_importance = shap_data.get("global_importance", {})

        if feature_importance:
            imp_df = pd.DataFrame(
                list(feature_importance.items()),
                columns=["Feature", "Importance"],
            ).sort_values("Importance", ascending=True)

            fig = px.bar(
                imp_df,
                x="Importance",
                y="Feature",
                orientation="h",
                title="Global Feature Importance (mean |SHAP|)",
                height=600,
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No SHAP data found. Run training first to generate feature importance analysis.")

    st.subheader("Per-Wallet Explanation")
    wallet = st.text_input("Enter wallet address for SHAP analysis", placeholder="0x...")
    if wallet:
        explain = api_call("/explain", method="POST", payload={"wallet_address": wallet})
        if explain:
            st.markdown(f"> {explain['narrative']}")


# =========================================================================
# Page: Model Performance
# =========================================================================
elif page == "Model Performance":
    st.header("Model Performance")

    metrics_path = Path("models/artifacts/evaluation_metrics.json")
    if metrics_path.exists():
        with open(metrics_path) as f:
            metrics = json.load(f)

        col1, col2, col3 = st.columns(3)
        col1.metric("Macro F1", f"{metrics.get('macro_f1', 0):.3f}")
        col2.metric("Accuracy", f"{metrics.get('accuracy', 0):.3f}")
        col3.metric("Weighted F1", f"{metrics.get('weighted_f1', 0):.3f}")

        if "per_class" in metrics:
            st.subheader("Per-Class Metrics")
            class_df = pd.DataFrame(metrics["per_class"]).T
            st.dataframe(class_df.style.format("{:.3f}"), use_container_width=True)

        if "confusion_matrix" in metrics:
            st.subheader("Confusion Matrix")
            cm = np.array(metrics["confusion_matrix"])
            cm_labels = metrics.get("label_names", [f"Class {i}" for i in range(len(cm))])
            fig = px.imshow(
                cm,
                labels=dict(x="Predicted", y="Actual", color="Count"),
                x=cm_labels,
                y=cm_labels,
                title="Confusion Matrix",
                color_continuous_scale="Blues",
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No evaluation metrics found. Run training first.")
