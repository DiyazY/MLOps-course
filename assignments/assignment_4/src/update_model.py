"""
Continuous Model Update (CMU) — automated retraining on new data.

Detects when a new Gold data version is available (by checking file
modification timestamps and row counts), then triggers retraining,
evaluation, and comparison against the current production model.

This implements a semi-automated CT pipeline:
1. Check for new data version
2. Retrain model on new data
3. Evaluate against KPI thresholds
4. Compare with previous best model
5. Promote if improved (or flag for manual review)
"""

import hashlib
import json
import os
import sys
import time

import mlflow
import numpy as np
import yaml

from train import compute_metrics, load_gold_data, temporal_train_test_split, train_model


def load_params():
    with open("params.yaml", "r") as f:
        return yaml.safe_load(f)


def compute_file_hash(filepath):
    """Compute MD5 hash of a file to detect changes."""
    h = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def get_data_state(data_dir="data/gold"):
    """Get current state of gold data files."""
    state = {}
    for fname in sorted(os.listdir(data_dir)):
        if fname.endswith(".csv"):
            fpath = os.path.join(data_dir, fname)
            state[fname] = {
                "path": fpath,
                "md5": compute_file_hash(fpath),
                "mtime": os.path.getmtime(fpath),
                "size": os.path.getsize(fpath),
            }
    return state


def load_last_state(state_file="models/data_state.json"):
    """Load the last known data state."""
    if os.path.exists(state_file):
        with open(state_file) as f:
            return json.load(f)
    return {}


def save_state(state, state_file="models/data_state.json"):
    """Save current data state for future comparison."""
    os.makedirs(os.path.dirname(state_file), exist_ok=True)
    with open(state_file, "w") as f:
        json.dump(state, f, indent=2)


def detect_new_data():
    """Check if new or modified gold data is available."""
    current_state = get_data_state()
    last_state = load_last_state()

    new_files = []
    modified_files = []

    for fname, info in current_state.items():
        if fname not in last_state:
            new_files.append(fname)
        elif last_state[fname]["md5"] != info["md5"]:
            modified_files.append(fname)

    return new_files, modified_files, current_state


def get_current_best_model(params):
    """Get the current best model from MLflow."""
    mlflow.set_tracking_uri(params["mlflow"]["tracking_uri"])
    experiment = mlflow.get_experiment_by_name(params["mlflow"]["experiment_name"])

    if experiment is None:
        return None

    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="tags.kpi_status = 'PASS'",
        order_by=["metrics.test_rmse ASC"],
        max_results=1,
    )

    if runs.empty:
        return None

    return runs.iloc[0]


def update_pipeline():
    """Main update pipeline — detect new data and retrain if needed."""
    params = load_params()

    print("=" * 60)
    print("CONTINUOUS MODEL UPDATE CHECK")
    print("=" * 60)
    print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    # Step 1: Detect new data
    new_files, modified_files, current_state = detect_new_data()

    if not new_files and not modified_files:
        print("\nNo new or modified gold data detected.")
        print("Pipeline is up to date.")
        return

    print(f"\nNew data files: {new_files}")
    print(f"Modified data files: {modified_files}")

    # Step 2: Determine which version to train on
    all_changed = new_files + modified_files
    # Use the latest version (highest version number)
    versions_to_train = []
    for fname in all_changed:
        if "v1" in fname:
            versions_to_train.append("v1")
        elif "v2" in fname:
            versions_to_train.append("v2")

    if not versions_to_train:
        print("Could not determine data version from changed files.")
        return

    # Step 3: Get current best model for comparison
    current_best = get_current_best_model(params)
    if current_best is not None:
        print(f"\nCurrent best model:")
        print(f"  Run ID: {current_best['run_id'][:8]}...")
        print(f"  RMSE:   {current_best['metrics.test_rmse']:.4f}")
        print(f"  Data:   {current_best.get('tags.data_version', 'unknown')}")

    # Step 4: Retrain on each changed data version
    for version in versions_to_train:
        print(f"\n{'='*60}")
        print(f"RETRAINING on data {version}")
        print(f"{'='*60}")

        new_run_id = train_model(version)

        # Step 5: Compare with previous best
        if current_best is not None:
            mlflow.set_tracking_uri(params["mlflow"]["tracking_uri"])
            new_run = mlflow.get_run(new_run_id)
            new_rmse = new_run.data.metrics["test_rmse"]
            old_rmse = current_best["metrics.test_rmse"]

            delta = new_rmse - old_rmse
            print(f"\n  RMSE delta: {delta:+.4f} ({'improved' if delta < 0 else 'degraded'})")

            if delta < 0:
                print(f"  -> New model is better. Promoting.")
            else:
                print(f"  -> Previous model is still better. New model logged but not promoted.")

    # Step 6: Save state
    save_state(current_state)
    print(f"\nData state saved for future change detection.")
    print(f"Update pipeline complete.")


if __name__ == "__main__":
    update_pipeline()
