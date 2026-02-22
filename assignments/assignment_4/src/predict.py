"""
Production prediction script.

Loads the best model from MLflow and generates predictions on test.csv.
Saves predictions to evidence/ for the report.
"""

import json
import os

import mlflow
import numpy as np
import pandas as pd
import yaml


def load_params():
    with open("params.yaml", "r") as f:
        return yaml.safe_load(f)


def get_best_model_run(experiment_name, tracking_uri, data_version=None):
    """Find the best model run (lowest test RMSE) from MLflow."""
    mlflow.set_tracking_uri(tracking_uri)
    experiment = mlflow.get_experiment_by_name(experiment_name)

    if experiment is None:
        return None

    filter_str = ""
    if data_version:
        filter_str = f"tags.data_version = '{data_version}'"

    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=filter_str,
        order_by=["metrics.test_rmse ASC"],
        max_results=1,
    )

    if runs.empty:
        return None

    return runs.iloc[0]


def predict():
    params = load_params()

    # Get best model
    best_run = get_best_model_run(
        params["mlflow"]["experiment_name"],
        params["mlflow"]["tracking_uri"],
    )

    if best_run is None:
        print("No trained models found. Run train.py first.")
        return

    run_id = best_run["run_id"]
    print(f"Loading best model: {run_id[:8]}...")
    print(f"  Model type: {best_run.get('tags.model_type', 'unknown')}")
    print(f"  Data version: {best_run.get('tags.data_version', 'unknown')}")
    print(f"  Test RMSE: {best_run['metrics.test_rmse']:.4f}")

    # Load model
    model_uri = f"runs:/{run_id}/model"
    model = mlflow.sklearn.load_model(model_uri)

    # Load feature list used during training
    client = mlflow.tracking.MlflowClient(params["mlflow"]["tracking_uri"])
    artifact_path = client.download_artifacts(run_id, "feature_cols.json")
    with open(artifact_path) as f:
        feature_cols = json.load(f)["features"]

    # Load processed test data
    test_path = params["data"]["test_processed"]
    if not os.path.exists(test_path):
        print(f"\nProcessed test data not found at {test_path}")
        print("Run prepare_test.py first.")
        return

    test_df = pd.read_csv(test_path, parse_dates=["date"])
    print(f"\nTest data: {len(test_df)} rows")

    # Align features
    missing_cols = [c for c in feature_cols if c not in test_df.columns]
    if missing_cols:
        print(f"  Warning: missing features {missing_cols}, filling with 0")
        for col in missing_cols:
            test_df[col] = 0

    X_test = test_df[feature_cols]

    # Predict
    predictions = model.predict(X_test)

    # Build results dataframe
    results = pd.DataFrame({
        "date": test_df["date"],
        "predicted_meantemp_next_day": np.round(predictions, 2),
    })

    # Add actual meantemp if available (for reference — not the target)
    if "meantemp" in test_df.columns:
        results["actual_meantemp_today"] = test_df["meantemp"]

    # Save predictions
    os.makedirs("evidence", exist_ok=True)
    output_path = "evidence/test_predictions.csv"
    results.to_csv(output_path, index=False)

    print(f"\nPredictions saved to {output_path}")
    print(f"\nSample predictions (first 10):")
    print(results.head(10).to_string(index=False))

    # Summary statistics
    print(f"\nPrediction summary:")
    print(f"  Mean: {predictions.mean():.2f} C")
    print(f"  Std:  {predictions.std():.2f} C")
    print(f"  Min:  {predictions.min():.2f} C")
    print(f"  Max:  {predictions.max():.2f} C")


if __name__ == "__main__":
    predict()
