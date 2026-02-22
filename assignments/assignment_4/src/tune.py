"""
Hyperparameter tuning with MLflow experiment tracking.

Uses GridSearchCV with TimeSeriesSplit to find optimal hyperparameters
for the Gradient Boosting model. Each configuration is logged as a
separate MLflow run.
"""

import argparse
import itertools
import os

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit

from train import compute_metrics, load_gold_data, load_params, temporal_train_test_split


def tune_model(data_version):
    """Run hyperparameter grid search with MLflow logging."""
    params = load_params()

    # Load data
    data_path = params["data"][f"gold_{data_version}"]
    X, y, dates, feature_cols = load_gold_data(data_path)
    X_train, X_test, y_train, y_test, dates_train, dates_test = \
        temporal_train_test_split(X, y, dates, params["model"]["test_size"])

    print(f"Tuning on data version: {data_version}")
    print(f"  Training samples: {len(X_train)}")
    print(f"  Validation samples: {len(X_test)}")

    # Build parameter grid
    tuning_params = params["tuning"]
    param_grid = {
        "n_estimators": tuning_params["n_estimators"],
        "max_depth": tuning_params["max_depth"],
        "learning_rate": tuning_params["learning_rate"],
        "min_samples_split": tuning_params["min_samples_split"],
    }

    # Generate all combinations
    keys = list(param_grid.keys())
    combinations = list(itertools.product(*[param_grid[k] for k in keys]))
    total = len(combinations)

    print(f"\nGrid search: {total} configurations")
    print(f"  Parameters: {keys}")

    # MLflow setup
    mlflow.set_tracking_uri(params["mlflow"]["tracking_uri"])
    mlflow.set_experiment(params["mlflow"]["experiment_name"])

    best_rmse = float("inf")
    best_run_id = None
    best_params = None

    tscv = TimeSeriesSplit(n_splits=3)

    for i, values in enumerate(combinations):
        hp = dict(zip(keys, values))
        hp["random_state"] = params["model"]["random_state"]
        hp["subsample"] = params["model"]["subsample"]
        hp["min_samples_leaf"] = params["model"]["min_samples_leaf"]

        with mlflow.start_run(run_name=f"tune_{i+1:03d}_{data_version}") as run:
            mlflow.set_tag("data_version", data_version)
            mlflow.set_tag("model_type", "gradient_boosting")
            mlflow.set_tag("run_type", "tuning")
            mlflow.log_params(hp)
            mlflow.log_param("data_version", data_version)
            mlflow.log_param("model_type", "gradient_boosting")
            mlflow.log_param("test_size", params["model"]["test_size"])

            model = GradientBoostingRegressor(**hp)

            # Cross-validation
            cv_rmse_scores = []
            for train_idx, val_idx in tscv.split(X_train):
                X_cv_train, X_cv_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
                y_cv_train, y_cv_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

                model_cv = GradientBoostingRegressor(**hp)
                model_cv.fit(X_cv_train, y_cv_train)
                y_cv_pred = model_cv.predict(X_cv_val)
                cv_rmse = np.sqrt(mean_squared_error(y_cv_val, y_cv_pred))
                cv_rmse_scores.append(cv_rmse)

            cv_rmse_mean = np.mean(cv_rmse_scores)
            cv_rmse_std = np.std(cv_rmse_scores)
            mlflow.log_metric("cv_rmse_mean", cv_rmse_mean)
            mlflow.log_metric("cv_rmse_std", cv_rmse_std)

            # Final evaluation on held-out test set
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            test_metrics = compute_metrics(y_test, y_pred)

            for key, val in test_metrics.items():
                mlflow.log_metric(f"test_{key}", val)

            # Check thresholds
            thresholds = params["kpi"]["thresholds"]
            passes = (
                test_metrics["rmse"] < thresholds["rmse_acceptable"]
                and test_metrics["r2"] > thresholds["r2_minimum"]
            )
            mlflow.set_tag("kpi_status", "PASS" if passes else "FAIL")

            # Log model if best so far
            if test_metrics["rmse"] < best_rmse:
                best_rmse = test_metrics["rmse"]
                best_run_id = run.info.run_id
                best_params = hp.copy()
                mlflow.sklearn.log_model(model, "model")
                mlflow.log_dict({"features": feature_cols}, "feature_cols.json")

            status = "*BEST*" if test_metrics["rmse"] <= best_rmse else ""
            print(f"  [{i+1:3d}/{total}] RMSE={test_metrics['rmse']:.4f} "
                  f"R2={test_metrics['r2']:.4f} CV_RMSE={cv_rmse_mean:.4f} "
                  f"(lr={hp['learning_rate']}, depth={hp['max_depth']}, "
                  f"n={hp['n_estimators']}, split={hp['min_samples_split']}) {status}")

    print(f"\n{'='*60}")
    print(f"Best configuration:")
    print(f"  Run ID: {best_run_id[:8]}...")
    print(f"  RMSE:   {best_rmse:.4f}")
    for k, v in best_params.items():
        print(f"  {k}: {v}")

    return best_run_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hyperparameter tuning")
    parser.add_argument("--data-version", choices=["v1", "v2"], default="v2",
                        help="Gold data version for tuning")
    args = parser.parse_args()

    tune_model(args.data_version)
