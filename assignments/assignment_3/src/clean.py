"""
Silver Layer: Data cleaning, validation, and feature engineering.

Takes bronze data and produces a clean, schema-enforced dataset with
engineered features suitable for downstream ML tasks.
"""

import os

import yaml
import numpy as np
import pandas as pd


def load_params():
    """Load pipeline parameters from params.yaml."""
    with open("params.yaml", "r") as f:
        return yaml.safe_load(f)


def remove_duplicates(df):
    """Remove duplicate dates, keeping the first occurrence."""
    n_before = len(df)
    df = df.drop_duplicates(subset=["date"], keep="first").reset_index(drop=True)
    n_removed = n_before - len(df)
    print(f"  Removed {n_removed} duplicate date entries")
    return df


def fix_out_of_range(df, params):
    """Clip or nullify values that fall outside physically valid ranges."""
    clean_params = params["clean"]

    temp_lo, temp_hi = clean_params["meantemp_range"]
    hum_lo, hum_hi = clean_params["humidity_range"]
    ws_min = clean_params["wind_speed_min"]
    pres_lo, pres_hi = clean_params["meanpressure_range"]

    # Flag out-of-range meanpressure as NaN (known dataset issue)
    mask_pressure = (df["meanpressure"] < pres_lo) | (df["meanpressure"] > pres_hi)
    n_bad_pressure = mask_pressure.sum()
    df.loc[mask_pressure, "meanpressure"] = np.nan
    print(f"  Nullified {n_bad_pressure} out-of-range meanpressure values")

    # Clip temperature to valid range
    mask_temp = (df["meantemp"] < temp_lo) | (df["meantemp"] > temp_hi)
    n_bad_temp = mask_temp.sum()
    df["meantemp"] = df["meantemp"].clip(lower=temp_lo, upper=temp_hi)
    print(f"  Clipped {n_bad_temp} out-of-range meantemp values")

    # Clip humidity to valid range
    mask_hum = (df["humidity"] < hum_lo) | (df["humidity"] > hum_hi)
    n_bad_hum = mask_hum.sum()
    df["humidity"] = df["humidity"].clip(lower=hum_lo, upper=hum_hi)
    print(f"  Clipped {n_bad_hum} out-of-range humidity values")

    # Floor wind speed at 0
    mask_ws = df["wind_speed"] < ws_min
    n_bad_ws = mask_ws.sum()
    df["wind_speed"] = df["wind_speed"].clip(lower=ws_min)
    print(f"  Clipped {n_bad_ws} negative wind_speed values")

    return df


def handle_missing_values(df):
    """Impute missing values using forward-fill then backward-fill."""
    n_missing_before = df.isnull().sum().sum()

    # Forward fill (appropriate for time series), then backfill for leading NaNs
    numeric_cols = ["meantemp", "humidity", "wind_speed", "meanpressure"]
    df[numeric_cols] = df[numeric_cols].ffill().bfill()

    n_missing_after = df.isnull().sum().sum()
    print(f"  Imputed {n_missing_before - n_missing_after} missing values (ffill + bfill)")

    return df


def add_temporal_features(df):
    """Extract temporal features from the date column."""
    df["month"] = df["date"].dt.month
    df["day_of_year"] = df["date"].dt.dayofyear

    # Map months to seasons (Delhi climate)
    season_map = {
        12: "winter", 1: "winter", 2: "winter",
        3: "spring", 4: "spring", 5: "spring",
        6: "summer", 7: "summer", 8: "summer",
        9: "autumn", 10: "autumn", 11: "autumn",
    }
    df["season"] = df["month"].map(season_map)

    print(f"  Added temporal features: month, day_of_year, season")
    return df


def add_rolling_features(df, windows):
    """Compute rolling averages for key variables."""
    for w in windows:
        df[f"meantemp_rolling_{w}d"] = df["meantemp"].rolling(window=w, min_periods=1).mean()
        df[f"humidity_rolling_{w}d"] = df["humidity"].rolling(window=w, min_periods=1).mean()

    print(f"  Added rolling features: windows={windows}")
    return df


def add_lag_features(df, lags):
    """Create lag features for all numeric variables."""
    for lag in lags:
        for col in ["meantemp", "humidity", "wind_speed", "meanpressure"]:
            df[f"{col}_lag_{lag}d"] = df[col].shift(lag)

    print(f"  Added lag features: lags={lags}")
    return df


def main():
    params = load_params()
    clean_params = params["clean"]

    input_file = clean_params["input_file"]
    output_file = clean_params["output_file"]
    rolling_windows = clean_params["rolling_windows"]
    lag_days = clean_params["lag_days"]

    # Load bronze data
    df = pd.read_csv(input_file, parse_dates=["date"])
    print(f"Loaded bronze data: {len(df)} rows")

    # Sort by date for time-series operations
    df = df.sort_values("date").reset_index(drop=True)

    # Data cleaning
    print("\nCleaning:")
    df = remove_duplicates(df)
    df = fix_out_of_range(df, params)
    df = handle_missing_values(df)

    # Feature engineering
    print("\nFeature engineering:")
    df = add_temporal_features(df)
    df = add_rolling_features(df, rolling_windows)
    df = add_lag_features(df, lag_days)

    # Drop rows with NaN introduced by lagging (first few rows)
    n_before = len(df)
    df = df.dropna().reset_index(drop=True)
    n_dropped = n_before - n_before
    print(f"\nDropped {n_before - len(df)} rows with NaN from lag features")

    # Save silver data
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df.to_csv(output_file, index=False)
    print(f"\nSilver data: {len(df)} rows -> {output_file}")

    # Summary
    print(f"\nColumn list ({len(df.columns)}):")
    for col in df.columns:
        print(f"  - {col}")


if __name__ == "__main__":
    main()
