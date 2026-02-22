"""
Prepare test.csv with the same feature engineering as the training data.

The raw test.csv contains only base climate variables. This script
applies the same transformations used in the silver/gold pipeline:
rolling averages, lag features, season encoding, and feature selection.
"""

import os
import sys

import numpy as np
import pandas as pd
import yaml


def load_params():
    with open("params.yaml", "r") as f:
        return yaml.safe_load(f)


def add_temporal_features(df):
    """Add month, day_of_year, and season columns."""
    df["month"] = df["date"].dt.month
    df["day_of_year"] = df["date"].dt.dayofyear

    # Season mapping for Delhi's climate
    season_map = {12: "winter", 1: "winter", 2: "winter",
                  3: "spring", 4: "spring", 5: "spring",
                  6: "summer", 7: "summer", 8: "summer",
                  9: "autumn", 10: "autumn", 11: "autumn"}
    df["season"] = df["month"].map(season_map)
    return df


def add_rolling_features(df, windows=(7, 30)):
    """Add rolling mean features for meantemp and humidity."""
    for w in windows:
        df[f"meantemp_rolling_{w}d"] = df["meantemp"].rolling(w, min_periods=1).mean()
        df[f"humidity_rolling_{w}d"] = df["humidity"].rolling(w, min_periods=1).mean()
    return df


def add_lag_features(df, lags=(1, 7)):
    """Add lag features for all numeric climate variables."""
    for lag in lags:
        for col in ["meantemp", "humidity", "wind_speed", "meanpressure"]:
            df[f"{col}_lag_{lag}d"] = df[col].shift(lag)
    return df


def prepare_test_data(gold_data_path):
    """
    Prepare test data to match the feature set of the gold training data.
    Uses the gold data's last rows to bootstrap rolling/lag features.
    """
    params = load_params()
    raw_path = params["data"]["test_raw"]
    output_path = params["data"]["test_processed"]

    # Load test data
    test_df = pd.read_csv(raw_path, parse_dates=["date"])
    test_df = test_df.sort_values("date").reset_index(drop=True)
    print(f"Loaded raw test data: {len(test_df)} rows")
    print(f"  Date range: {test_df['date'].iloc[0].date()} to {test_df['date'].iloc[-1].date()}")

    # Clean obvious outliers (same logic as silver layer)
    test_df["meanpressure"] = test_df["meanpressure"].apply(
        lambda x: np.nan if x < 900 or x > 1100 else x
    )
    test_df["meantemp"] = test_df["meantemp"].clip(-10, 50)
    test_df["humidity"] = test_df["humidity"].clip(0, 100)
    test_df["wind_speed"] = test_df["wind_speed"].clip(lower=0)

    # Load gold training data to bootstrap rolling/lag features
    gold_df = pd.read_csv(gold_data_path, parse_dates=["date"])
    gold_df = gold_df.sort_values("date")

    # Get the last 30 rows of training data for rolling window context
    # We need base columns for feature engineering
    base_cols = ["date", "meantemp", "humidity", "wind_speed", "meanpressure"]
    gold_base = gold_df[["date"]].copy()

    # Reconstruct base columns from gold data (they're present in gold)
    for col in ["meantemp", "humidity", "wind_speed", "meanpressure"]:
        if col in gold_df.columns:
            gold_base[col] = gold_df[col]

    # Concatenate last 30 rows of training data with test data
    context_rows = gold_base.tail(30)[base_cols]
    combined = pd.concat([context_rows, test_df[base_cols]], ignore_index=True)

    # Apply feature engineering on combined data
    combined = add_temporal_features(combined)
    combined = add_rolling_features(combined)
    combined = add_lag_features(combined)

    # One-hot encode season
    combined = pd.get_dummies(combined, columns=["season"], dtype=int)

    # Impute any remaining NaN from edge effects
    combined = combined.ffill().bfill()

    # Keep only the test rows (last len(test_df) rows)
    test_processed = combined.tail(len(test_df)).reset_index(drop=True)

    # Select only the features used in training
    gold_features = [c for c in gold_df.columns if c not in ("date", "target")]

    # Align columns with training data
    for col in gold_features:
        if col not in test_processed.columns:
            test_processed[col] = 0  # Missing one-hot columns default to 0

    # Save processed test data
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    test_processed.to_csv(output_path, index=False)

    print(f"\nProcessed test data: {len(test_processed)} rows")
    print(f"  Columns: {len(test_processed.columns)}")
    print(f"  Saved to: {output_path}")

    return test_processed


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Prepare test data for prediction")
    parser.add_argument("--gold-data", default="data/gold/gold_v2.csv",
                        help="Gold data path (for feature alignment and context)")
    args = parser.parse_args()

    prepare_test_data(args.gold_data)
