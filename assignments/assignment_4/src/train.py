"""
Model training pipeline with MLflow experiment tracking.

Trains a Gradient Boosting model for next-day temperature forecasting
on a specified Gold data version. Logs parameters, metrics, and model
artifacts to MLflow.
"""

import argparse
import json
import os
import sys

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit


def load_params():
    with open("params.yaml", "r") as f:
        return yaml.safe_load(f)


def load_gold_data(filepath):
    """Load gold dataset and split into features and target."""
    df = pd.read_csv(filepath, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)

    feature_cols = [c for c in df.columns if c not in ("date", "target")]
    X = df[feature_cols]
    y = df["target"]
    dates = df["date"]

    return X, y, dates, feature_cols


def temporal_train_test_split(X, y, dates, test_size=0.2):
    """
    Split data temporally — last `test_size` fraction is used for validation.
    This preserves the time ordering and avoids data leakage.
    """
    n = len(X)
    split_idx = int(n * (1 - test_size))

    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    dates_train = dates.iloc[:split_idx]
    dates_test = dates.iloc[split_idx:]

    return X_train, X_test, y_train, y_test, dates_train, dates_test


def compute_metrics(y_true, y_pred):
    """Compute all KPI metrics."""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {"rmse": rmse, "mae": mae, "r2": r2}


def cross_validate_timeseries(model, X, y, n_splits=5):
    """
    Perform time-series cross-validation to assess model stability.
    Returns mean and std of metrics across folds.
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    fold_metrics = []

    for train_idx, val_idx in tscv.split(X):
        X_fold_train, X_fold_val = X.iloc[train_idx], X.iloc[val_idx]
        y_fold_train, y_fold_val = y.iloc[train_idx], y.iloc[val_idx]

        model_clone = model.__class__(**model.get_params())
        model_clone.fit(X_fold_train, y_fold_train)
        y_pred = model_clone.predict(X_fold_val)

        metrics = compute_metrics(y_fold_val, y_pred)
        fold_metrics.append(metrics)

    results = {}
    for key in fold_metrics[0]:
        values = [m[key] for m in fold_metrics]
        results[f"cv_{key}_mean"] = np.mean(values)
        results[f"cv_{key}_std"] = np.std(values)

    return results


def train_model(data_version, model_type=None, custom_params=None):
    """
    Train a model on specified gold data version.

    Args:
        data_version: "v1" or "v2"
        model_type: override model type from params
        custom_params: override hyperparameters
    """
    params = load_params()

    # Resolve data path
    data_key = f"gold_{data_version}"
    data_path = params["data"][data_key]

    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found")
        sys.exit(1)

    # Load and split data
    X, y, dates, feature_cols = load_gold_data(data_path)
    X_train, X_test, y_train, y_test, dates_train, dates_test = \
        temporal_train_test_split(X, y, dates, params["model"]["test_size"])

    print(f"Data version: {data_version}")
    print(f"  Source: {data_path}")
    print(f"  Total samples: {len(X)}")
    print(f"  Training: {len(X_train)} ({dates_train.iloc[0].date()} to {dates_train.iloc[-1].date()})")
    print(f"  Validation: {len(X_test)} ({dates_test.iloc[0].date()} to {dates_test.iloc[-1].date()})")
    print(f"  Features: {len(feature_cols)}")

    # Select model type
    mtype = model_type or params["model"]["type"]
    model_params = params["model"]

    if mtype == "gradient_boosting":
        hp = custom_params or {
            "n_estimators": model_params["n_estimators"],
            "max_depth": model_params["max_depth"],
            "learning_rate": model_params["learning_rate"],
            "min_samples_split": model_params["min_samples_split"],
            "min_samples_leaf": model_params["min_samples_leaf"],
            "subsample": model_params["subsample"],
            "random_state": model_params["random_state"],
        }
        model = GradientBoostingRegressor(**hp)
    elif mtype == "random_forest":
        hp = custom_params or {
            "n_estimators": model_params.get("n_estimators", 100),
            "max_depth": model_params.get("max_depth", 5),
            "random_state": model_params["random_state"],
        }
        model = RandomForestRegressor(**hp)
    elif mtype == "linear_regression":
        hp = {}
        model = LinearRegression()
    else:
        print(f"Unknown model type: {mtype}")
        sys.exit(1)

    # Set up MLflow
    mlflow.set_tracking_uri(params["mlflow"]["tracking_uri"])
    mlflow.set_experiment(params["mlflow"]["experiment_name"])

    with mlflow.start_run(run_name=f"{mtype}_{data_version}") as run:
        # Log data version info
        mlflow.set_tag("data_version", data_version)
        mlflow.set_tag("data_path", data_path)
        mlflow.set_tag("model_type", mtype)
        mlflow.set_tag("n_features", len(feature_cols))
        mlflow.set_tag("n_train_samples", len(X_train))
        mlflow.set_tag("n_test_samples", len(X_test))
        mlflow.set_tag("train_date_range",
                        f"{dates_train.iloc[0].date()} to {dates_train.iloc[-1].date()}")
        mlflow.set_tag("test_date_range",
                        f"{dates_test.iloc[0].date()} to {dates_test.iloc[-1].date()}")

        # Log parameters
        mlflow.log_params(hp)
        mlflow.log_param("data_version", data_version)
        mlflow.log_param("model_type", mtype)
        mlflow.log_param("test_size", params["model"]["test_size"])

        # Train
        print(f"\nTraining {mtype}...")
        model.fit(X_train, y_train)

        # Predict
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        # Compute metrics
        train_metrics = compute_metrics(y_train, y_train_pred)
        test_metrics = compute_metrics(y_test, y_test_pred)

        # Log metrics
        for key, val in train_metrics.items():
            mlflow.log_metric(f"train_{key}", val)
        for key, val in test_metrics.items():
            mlflow.log_metric(f"test_{key}", val)

        # Cross-validation for stability assessment
        print("Running time-series cross-validation...")
        cv_metrics = cross_validate_timeseries(model, X_train, y_train)
        for key, val in cv_metrics.items():
            mlflow.log_metric(key, val)

        # Log model artifact
        mlflow.sklearn.log_model(model, "model")

        # Log feature list
        mlflow.log_dict({"features": feature_cols}, "feature_cols.json")

        # Print results
        print(f"\n{'='*50}")
        print(f"Run ID: {run.info.run_id}")
        print(f"\nTraining metrics:")
        for key, val in train_metrics.items():
            print(f"  {key}: {val:.4f}")
        print(f"\nValidation metrics:")
        for key, val in test_metrics.items():
            print(f"  {key}: {val:.4f}")
        print(f"\nCross-validation (5-fold):")
        for key, val in cv_metrics.items():
            print(f"  {key}: {val:.4f}")

        # Check against KPI thresholds
        thresholds = params["kpi"]["thresholds"]
        print(f"\nKPI Assessment:")
        rmse_ok = test_metrics["rmse"] < thresholds["rmse_acceptable"]
        mae_ok = test_metrics["mae"] < thresholds["mae_acceptable"]
        r2_ok = test_metrics["r2"] > thresholds["r2_minimum"]

        status = "PASS" if (rmse_ok and r2_ok) else "FAIL"
        print(f"  RMSE: {test_metrics['rmse']:.4f} (threshold: <{thresholds['rmse_acceptable']}) {'OK' if rmse_ok else 'FAIL'}")
        print(f"  MAE:  {test_metrics['mae']:.4f} (threshold: <{thresholds['mae_acceptable']}) {'OK' if mae_ok else 'FAIL'}")
        print(f"  R2:   {test_metrics['r2']:.4f} (threshold: >{thresholds['r2_minimum']}) {'OK' if r2_ok else 'FAIL'}")
        print(f"  Overall: {status}")

        mlflow.set_tag("kpi_status", status)

        return run.info.run_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train temperature forecasting model")
    parser.add_argument("--data-version", choices=["v1", "v2"], required=True,
                        help="Gold data version to train on")
    parser.add_argument("--model-type", choices=["gradient_boosting", "random_forest", "linear_regression"],
                        default=None, help="Override model type")
    args = parser.parse_args()

    run_id = train_model(args.data_version, args.model_type)
    print(f"\nMLflow run ID: {run_id}")
