"""
Model evaluation and version comparison.

Compares performance of models across different data versions
and reports KPI deltas. Generates comparison summary for evidence.
"""

import argparse
import json
import os

import mlflow
import numpy as np
import pandas as pd
import yaml


def load_params():
    with open("params.yaml", "r") as f:
        return yaml.safe_load(f)


def get_runs_by_data_version(experiment_name, tracking_uri):
    """Retrieve all MLflow runs grouped by data version."""
    mlflow.set_tracking_uri(tracking_uri)
    experiment = mlflow.get_experiment_by_name(experiment_name)

    if experiment is None:
        print(f"No experiment found: {experiment_name}")
        return {}

    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["start_time DESC"],
    )

    if runs.empty:
        print("No runs found.")
        return {}

    return runs


def compare_models():
    """Compare best models from each data version."""
    params = load_params()
    runs = get_runs_by_data_version(
        params["mlflow"]["experiment_name"],
        params["mlflow"]["tracking_uri"],
    )

    if isinstance(runs, dict) and not runs:
        return

    # Filter for relevant columns
    metric_cols = [c for c in runs.columns if c.startswith("metrics.test_")]
    tag_cols = ["tags.data_version", "tags.model_type", "tags.kpi_status"]
    info_cols = ["run_id", "start_time"]

    display_cols = info_cols + tag_cols + metric_cols
    available_cols = [c for c in display_cols if c in runs.columns]
    summary = runs[available_cols].copy()

    print("=" * 70)
    print("MODEL VERSION COMPARISON")
    print("=" * 70)

    # Group by data version — find best model per version (lowest RMSE)
    versions = {}
    for version in ["v1", "v2"]:
        mask = runs["tags.data_version"] == version
        version_runs = runs[mask]
        if version_runs.empty:
            continue

        best_idx = version_runs["metrics.test_rmse"].idxmin()
        best_run = version_runs.loc[best_idx]
        versions[version] = best_run

        print(f"\nBest model for data {version}:")
        print(f"  Run ID: {best_run['run_id'][:8]}...")
        print(f"  Model:  {best_run.get('tags.model_type', 'unknown')}")
        print(f"  RMSE:   {best_run['metrics.test_rmse']:.4f}")
        print(f"  MAE:    {best_run['metrics.test_mae']:.4f}")
        print(f"  R2:     {best_run['metrics.test_r2']:.4f}")
        print(f"  Status: {best_run.get('tags.kpi_status', 'unknown')}")

    # Compute deltas if both versions exist
    if "v1" in versions and "v2" in versions:
        print(f"\n{'='*70}")
        print("PERFORMANCE DELTA (v2 vs v1)")
        print(f"{'='*70}")

        v1 = versions["v1"]
        v2 = versions["v2"]

        rmse_delta = v2["metrics.test_rmse"] - v1["metrics.test_rmse"]
        mae_delta = v2["metrics.test_mae"] - v1["metrics.test_mae"]
        r2_delta = v2["metrics.test_r2"] - v1["metrics.test_r2"]

        # For RMSE and MAE: negative delta = improvement
        # For R2: positive delta = improvement
        print(f"  RMSE: {rmse_delta:+.4f} ({'improved' if rmse_delta < 0 else 'degraded'})")
        print(f"  MAE:  {mae_delta:+.4f} ({'improved' if mae_delta < 0 else 'degraded'})")
        print(f"  R2:   {r2_delta:+.4f} ({'improved' if r2_delta > 0 else 'degraded'})")

        # Decision logic
        thresholds = params["kpi"]["thresholds"]
        v2_passes = (
            v2["metrics.test_rmse"] < thresholds["rmse_acceptable"]
            and v2["metrics.test_r2"] > thresholds["r2_minimum"]
        )

        if v2_passes and rmse_delta <= 0:
            recommendation = "PROMOTE v2 model — improved on new data"
        elif v2_passes:
            recommendation = "PROMOTE v2 model — meets thresholds despite slight regression"
        else:
            recommendation = "KEEP v1 model — v2 does not meet KPI thresholds"

        print(f"\n  Recommendation: {recommendation}")

        # Save comparison to file
        comparison = {
            "v1_run_id": v1["run_id"],
            "v2_run_id": v2["run_id"],
            "v1_rmse": float(v1["metrics.test_rmse"]),
            "v2_rmse": float(v2["metrics.test_rmse"]),
            "v1_mae": float(v1["metrics.test_mae"]),
            "v2_mae": float(v2["metrics.test_mae"]),
            "v1_r2": float(v1["metrics.test_r2"]),
            "v2_r2": float(v2["metrics.test_r2"]),
            "rmse_delta": float(rmse_delta),
            "mae_delta": float(mae_delta),
            "r2_delta": float(r2_delta),
            "recommendation": recommendation,
        }

        os.makedirs("evidence", exist_ok=True)
        with open("evidence/model_comparison.json", "w") as f:
            json.dump(comparison, f, indent=2)
        print(f"\n  Comparison saved to evidence/model_comparison.json")

    # Print all runs summary
    print(f"\n{'='*70}")
    print("ALL EXPERIMENT RUNS")
    print(f"{'='*70}")
    for _, row in summary.iterrows():
        version = row.get("tags.data_version", "?")
        mtype = row.get("tags.model_type", "?")
        rmse = row.get("metrics.test_rmse", float("nan"))
        r2 = row.get("metrics.test_r2", float("nan"))
        status = row.get("tags.kpi_status", "?")
        print(f"  [{version}] {mtype:25s} RMSE={rmse:.4f}  R2={r2:.4f}  {status}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare model versions")
    parser.add_argument("--compare", action="store_true", default=True,
                        help="Compare all model versions")
    args = parser.parse_args()

    compare_models()
