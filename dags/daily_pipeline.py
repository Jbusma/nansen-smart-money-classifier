"""Cloud Composer (Airflow) DAG for daily Smart Money pipeline."""

from __future__ import annotations

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import BranchPythonOperator, PythonOperator
from airflow.utils.trigger_rule import TriggerRule

default_args = {
    "owner": "smart-money",
    "depends_on_past": False,
    "email_on_failure": True,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="smart_money_daily_pipeline",
    default_args=default_args,
    description="Daily pipeline: extract → transform → sync → retrain (if drift) → serve",
    schedule_interval="@daily",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["smart-money", "ml-pipeline"],
) as dag:
    # ── Step 1: Extract from BigQuery ──────────────────────────────
    extract = BashOperator(
        task_id="extract_bigquery_transactions",
        bash_command="python -m src.data.bigquery_extract",
    )

    # ── Step 2: Run dbt transformations ────────────────────────────
    dbt_run = BashOperator(
        task_id="run_dbt_transformations",
        bash_command="cd dbt && dbt run --profiles-dir .",
    )

    # ── Step 3: Sync features to Clickhouse ────────────────────────
    sync_features = BashOperator(
        task_id="export_features_to_clickhouse",
        bash_command="python -m src.data.clickhouse_sync",
    )

    # ── Step 4: Check for model drift ──────────────────────────────
    def _check_drift(**context):
        """Compare current feature distributions to training baseline."""
        import json
        from pathlib import Path

        drift_report_path = Path("models/artifacts/drift_report.json")
        if not drift_report_path.exists():
            return "retrain_model"  # No baseline → must train

        with open(drift_report_path) as f:
            report = json.load(f)

        drift_detected = report.get("drift_detected", False)
        return "retrain_model" if drift_detected else "skip_retrain"

    check_drift = BranchPythonOperator(
        task_id="check_model_drift",
        python_callable=_check_drift,
    )

    # ── Step 5a: Retrain model ─────────────────────────────────────
    retrain = BashOperator(
        task_id="retrain_model",
        bash_command="python -m src.models.train",
    )

    # ── Step 5b: Skip retrain ──────────────────────────────────────
    skip_retrain = BashOperator(
        task_id="skip_retrain",
        bash_command="echo 'No drift detected, skipping retrain'",
    )

    # ── Step 6: Invalidate LLM cache ──────────────────────────────
    invalidate_cache = BashOperator(
        task_id="invalidate_llm_cache",
        bash_command="python -c \"from src.llm.cache import NarrativeCache; NarrativeCache().invalidate()\"",
        trigger_rule=TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS,
    )

    # ── Step 7: Regenerate cluster profiles ────────────────────────
    regen_profiles = BashOperator(
        task_id="generate_fresh_cluster_profiles",
        bash_command="python -c \""
        "from src.models.clustering import ClusteringPipeline; "
        "from src.llm.insight_generator import InsightGenerator; "
        "c = ClusteringPipeline.load('models/artifacts/clustering'); "
        "ig = InsightGenerator(); "
        "stats = c.get_cluster_stats(); "
        "[ig.generate_cluster_profile(cid, s, s.get('exemplar_addresses', [])) for cid, s in stats.items()]"
        "\"",
    )

    # ── DAG wiring ─────────────────────────────────────────────────
    extract >> dbt_run >> sync_features >> check_drift
    check_drift >> [retrain, skip_retrain]
    [retrain, skip_retrain] >> invalidate_cache >> regen_profiles
