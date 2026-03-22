"""Streamlit dashboard for Smart Money Classifier exploration."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

API_URL = "http://localhost:8000"

st.set_page_config(
    page_title="Nansen Smart Money Classifier",
    page_icon="🔍",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.title("Smart Money Classifier")
page = st.sidebar.radio("Navigate", ["Cluster Explorer", "Wallet Lookup", "Feature Importance", "Model Performance"])


def api_call(endpoint: str, method: str = "GET", json: dict | None = None) -> dict | None:
    """Helper to call the FastAPI backend."""
    try:
        if method == "GET":
            resp = requests.get(f"{API_URL}{endpoint}", timeout=30)
        else:
            resp = requests.post(f"{API_URL}{endpoint}", json=json, timeout=30)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to API server. Run `make serve-api` first.")
        return None
    except requests.exceptions.HTTPError as e:
        st.error(f"API error: {e.response.text}")
        return None


# ---------------------------------------------------------------------------
# Page: Cluster Explorer
# ---------------------------------------------------------------------------
if page == "Cluster Explorer":
    st.header("Cluster Explorer")
    st.markdown("Interactive UMAP projection of wallet behavioral clusters.")

    # Load clustering data from local artifacts
    try:
        from pathlib import Path

        import joblib

        artifacts_path = Path("models/artifacts/clustering")
        if artifacts_path.exists():
            embedding = joblib.load(artifacts_path / "embedding.joblib")
            labels = joblib.load(artifacts_path / "labels.joblib")
            addresses = joblib.load(artifacts_path / "addresses.joblib")

            df_viz = pd.DataFrame(
                {
                    "UMAP_1": embedding[:, 0],
                    "UMAP_2": embedding[:, 1],
                    "Cluster": labels.astype(str),
                    "Address": addresses,
                }
            )

            # Noise points labeled as -1
            df_viz["Cluster"] = df_viz["Cluster"].replace("-1", "Noise")

            fig = px.scatter(
                df_viz,
                x="UMAP_1",
                y="UMAP_2",
                color="Cluster",
                hover_data=["Address"],
                title="Wallet Behavioral Clusters (UMAP Projection)",
                opacity=0.6,
                height=700,
            )
            fig.update_layout(legend_title_text="Cluster")
            st.plotly_chart(fig, use_container_width=True)

            # Cluster stats sidebar
            st.subheader("Cluster Statistics")
            cluster_counts = df_viz["Cluster"].value_counts()
            st.dataframe(cluster_counts.reset_index().rename(columns={"index": "Cluster", "Cluster": "Count"}))
        else:
            st.info("No clustering artifacts found. Run `make cluster` first.")
    except Exception as e:
        st.warning(f"Could not load clustering data: {e}")


# ---------------------------------------------------------------------------
# Page: Wallet Lookup
# ---------------------------------------------------------------------------
elif page == "Wallet Lookup":
    st.header("Wallet Lookup")

    wallet = st.text_input("Ethereum Address", placeholder="0x...")

    if wallet and len(wallet) == 42 and wallet.startswith("0x"):
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Classification")
            result = api_call("/classify", method="POST", json={"wallet_address": wallet})
            if result:
                st.metric("Label", result["label"].replace("_", " ").title())
                st.metric("Confidence", f"{result['confidence']:.1%}")
                st.caption(f"Latency: {result['latency_ms']:.1f}ms")

                # Probability distribution
                proba_df = pd.DataFrame(
                    list(result["probabilities"].items()),
                    columns=["Label", "Probability"],
                )
                fig = px.bar(proba_df, x="Label", y="Probability", title="Class Probabilities")
                st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("Feature Profile")
            if result:
                features = result["features"]
                # Radar chart of key features
                radar_features = [
                    "tx_frequency_per_day",
                    "avg_value_per_tx",
                    "dex_to_total_ratio",
                    "token_diversity_entropy",
                    "burst_score",
                    "counterparty_concentration",
                    "unique_contracts_called",
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
                    fig.update_layout(title="Behavioral Radar", polar=dict(radialaxis=dict(visible=True)))
                    st.plotly_chart(fig, use_container_width=True)

        # AI narrative
        st.subheader("AI Intelligence Briefing")
        explain = api_call("/explain", method="POST", json={"wallet_address": wallet})
        if explain:
            st.markdown(f"> {explain['narrative']}")

        # Similar wallets
        st.subheader("Similar Wallets")
        similar = api_call("/similar", method="POST", json={"wallet_address": wallet, "top_k": 10})
        if similar:
            sim_df = pd.DataFrame(similar["similar_wallets"])
            st.dataframe(sim_df, use_container_width=True)


# ---------------------------------------------------------------------------
# Page: Feature Importance
# ---------------------------------------------------------------------------
elif page == "Feature Importance":
    st.header("Feature Importance")
    st.markdown("SHAP-based feature importance analysis.")

    try:
        from pathlib import Path

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
            st.info("No SHAP data found. Run `make train` first to generate feature importance analysis.")
    except Exception as e:
        st.warning(f"Could not load SHAP data: {e}")

    # Per-wallet SHAP
    st.subheader("Per-Wallet Explanation")
    wallet = st.text_input("Enter wallet address for SHAP analysis", placeholder="0x...")
    if wallet:
        st.info(
            "Per-wallet SHAP waterfall plots are generated during model evaluation. "
            "See notebooks/03_classifier_training.ipynb."
        )


# ---------------------------------------------------------------------------
# Page: Model Performance
# ---------------------------------------------------------------------------
elif page == "Model Performance":
    st.header("Model Performance")

    try:
        import json
        from pathlib import Path

        metrics_path = Path("models/artifacts/evaluation_metrics.json")
        if metrics_path.exists():
            with open(metrics_path) as f:
                metrics = json.load(f)

            col1, col2, col3 = st.columns(3)
            col1.metric("Macro F1", f"{metrics.get('macro_f1', 0):.3f}")
            col2.metric("Accuracy", f"{metrics.get('accuracy', 0):.3f}")
            col3.metric("Weighted F1", f"{metrics.get('weighted_f1', 0):.3f}")

            # Per-class metrics
            if "per_class" in metrics:
                st.subheader("Per-Class Metrics")
                class_df = pd.DataFrame(metrics["per_class"]).T
                st.dataframe(class_df.style.format("{:.3f}"), use_container_width=True)

            # Confusion matrix
            if "confusion_matrix" in metrics:
                st.subheader("Confusion Matrix")
                cm = np.array(metrics["confusion_matrix"])
                labels = metrics.get("label_names", [f"Class {i}" for i in range(len(cm))])
                fig = px.imshow(
                    cm,
                    labels=dict(x="Predicted", y="Actual", color="Count"),
                    x=labels,
                    y=labels,
                    title="Confusion Matrix",
                    color_continuous_scale="Blues",
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No evaluation metrics found. Run `make train` first.")
    except Exception as e:
        st.warning(f"Could not load metrics: {e}")
