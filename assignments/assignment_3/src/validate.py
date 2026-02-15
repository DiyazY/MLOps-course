"""
Data Quality Validation for the Silver layer.

Runs automated checks on the cleaned dataset to ensure it meets
quality standards before being promoted to the Gold layer.
"""

import sys

import yaml
import pandas as pd


def load_params():
    """Load pipeline parameters from params.yaml."""
    with open("params.yaml", "r") as f:
        return yaml.safe_load(f)


class ValidationResult:
    """Tracks pass/fail results for validation checks."""

    def __init__(self):
        self.results = []

    def check(self, name, condition, detail=""):
        status = "PASS" if condition else "FAIL"
        self.results.append((name, status, detail))
        symbol = "✓" if condition else "✗"
        print(f"  [{symbol}] {name}" + (f" — {detail}" if detail else ""))

    @property
    def all_passed(self):
        return all(status == "PASS" for _, status, _ in self.results)

    def summary(self):
        passed = sum(1 for _, s, _ in self.results if s == "PASS")
        total = len(self.results)
        return f"{passed}/{total} checks passed"


def validate_silver(df, params):
    """Run all validation checks on the silver dataset."""
    clean_params = params["clean"]
    v = ValidationResult()

    print("Schema validation:")
    v.check("Date column is datetime", pd.api.types.is_datetime64_any_dtype(df["date"]))
    v.check("meantemp is numeric", pd.api.types.is_numeric_dtype(df["meantemp"]))
    v.check("humidity is numeric", pd.api.types.is_numeric_dtype(df["humidity"]))
    v.check("wind_speed is numeric", pd.api.types.is_numeric_dtype(df["wind_speed"]))
    v.check("meanpressure is numeric", pd.api.types.is_numeric_dtype(df["meanpressure"]))

    print("\nCompleteness:")
    null_count = df[["meantemp", "humidity", "wind_speed", "meanpressure"]].isnull().sum().sum()
    v.check("No missing values in core columns", null_count == 0, f"{null_count} nulls found")
    v.check("No duplicate dates", df["date"].duplicated().sum() == 0,
            f"{df['date'].duplicated().sum()} duplicates")

    print("\nValue range checks:")
    temp_lo, temp_hi = clean_params["meantemp_range"]
    v.check(
        f"meantemp in [{temp_lo}, {temp_hi}]",
        df["meantemp"].between(temp_lo, temp_hi).all(),
        f"min={df['meantemp'].min():.2f}, max={df['meantemp'].max():.2f}",
    )

    hum_lo, hum_hi = clean_params["humidity_range"]
    v.check(
        f"humidity in [{hum_lo}, {hum_hi}]",
        df["humidity"].between(hum_lo, hum_hi).all(),
        f"min={df['humidity'].min():.2f}, max={df['humidity'].max():.2f}",
    )

    ws_min = clean_params["wind_speed_min"]
    v.check(
        f"wind_speed >= {ws_min}",
        (df["wind_speed"] >= ws_min).all(),
        f"min={df['wind_speed'].min():.2f}",
    )

    pres_lo, pres_hi = clean_params["meanpressure_range"]
    v.check(
        f"meanpressure in [{pres_lo}, {pres_hi}]",
        df["meanpressure"].between(pres_lo, pres_hi).all(),
        f"min={df['meanpressure'].min():.2f}, max={df['meanpressure'].max():.2f}",
    )

    print("\nTemporal continuity:")
    date_diffs = df["date"].diff().dropna()
    max_gap = date_diffs.max()
    v.check(
        "No date gaps larger than 7 days",
        max_gap <= pd.Timedelta(days=7),
        f"max gap = {max_gap}",
    )
    v.check("Data is sorted by date", df["date"].is_monotonic_increasing)

    print("\nRow count consistency:")
    bronze_df = pd.read_csv(params["validate"]["bronze_file"])
    # Silver should have fewer rows (duplicates removed, but not drastically less)
    ratio = len(df) / len(bronze_df)
    v.check(
        "Silver row count is reasonable vs bronze",
        0.5 < ratio <= 1.0,
        f"silver={len(df)}, bronze={len(bronze_df)}, ratio={ratio:.2f}",
    )

    return v


def main():
    params = load_params()
    silver_file = params["validate"]["silver_file"]

    df = pd.read_csv(silver_file, parse_dates=["date"])
    print(f"Validating silver data: {len(df)} rows from {silver_file}\n")

    result = validate_silver(df, params)

    print(f"\n{'='*50}")
    print(f"Validation {result.summary()}")

    if not result.all_passed:
        print("VALIDATION FAILED — fix issues before proceeding to gold layer")
        sys.exit(1)
    else:
        print("All checks passed — data is ready for gold layer")


if __name__ == "__main__":
    main()
