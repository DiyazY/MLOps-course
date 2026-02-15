# Assignment 3: DataOps Pipeline

A DataOps pipeline for incremental data management using the Delhi Climate dataset.
The pipeline follows a Bronze-Silver-Gold medallion architecture with DVC for data versioning.

## Overview

This project implements a DataOps pipeline that:
- Manages data incrementally over time (5 temporal batches)
- Applies versioning, lineage tracking, validation, and reproducibility
- Transforms raw climate data into an ML-ready dataset for time-series forecasting

## Dataset

Delhi Daily Climate dataset from Kaggle, containing daily observations (Jan 2013 - Apr 2017):
- `meantemp` - mean daily temperature
- `humidity` - daily humidity
- `wind_speed` - daily wind speed
- `meanpressure` - daily atmospheric pressure

## Environment

**Local machine** with Python virtual environment and DVC for data versioning.

- **Storage**: Local filesystem with DVC cache
- **Compute**: Local Python execution
- **Trade-offs**: Full control and simplicity at the cost of scalability (acceptable for ~1500 rows)

## Project Structure

```
assignment_3/
├── README.md
├── requirements.txt
├── params.yaml
├── dvc.yaml
├── data/
│   ├── raw/          # Original train.csv (DVC tracked)
│   ├── batches/      # 5 time-ordered batches (DVC tracked)
│   ├── bronze/       # Raw ingested data with metadata
│   ├── silver/       # Cleaned and validated data
│   └── gold/         # ML-ready dataset
└── src/
    ├── split_batches.py
    ├── ingest.py
    ├── clean.py
    ├── validate.py
    └── prepare_gold.py
```

## Setup

```bash
cd assignments/assignment_3
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

Run the full pipeline:
```bash
dvc repro
```

## Pipeline Stages

1. **Split** - Divide train.csv into 5 temporal batches
2. **Ingest** - Bronze layer: incremental ingestion with metadata
3. **Clean** - Silver layer: data cleaning and feature engineering
4. **Validate** - Data quality tests
5. **Prepare Gold** - Gold layer: ML-ready dataset for forecasting

## Execution Steps

### Step 1: Batch Splitting

```bash
python src/split_batches.py
```

Loads `data/raw/train.csv`, sorts records by date, and splits them into 5 roughly
equal time-ordered batches (~293 records each). Each batch covers ~9-10 months of
climate observations, simulating how data would arrive incrementally over time.

Output: `data/batches/batch_{1..5}.csv`

### Step 2: Bronze Layer Ingestion

```bash
python src/ingest.py              # ingest all batches
python src/ingest.py --batch 3    # or ingest a single batch
```

Each batch goes through a simulated ingestion process that introduces realistic
data quality issues to mimic real-world data pipelines:
- **Dropped rows** (~5%) - simulates data loss during transfer
- **Missing values** (~3%) - simulates sensor failures or incomplete records
- **Duplicate rows** (~2%) - simulates duplicate event delivery

Metadata columns are added for lineage tracking: `batch_id`, `ingestion_timestamp`,
and `source_file`. Batches are appended incrementally to the bronze dataset.

Output: `data/bronze/bronze_data.csv`

### Step 3: Silver Layer Cleaning

```bash
python src/clean.py
```

Processes bronze data through several cleaning and transformation stages:
- **Duplicate removal** — removes repeated date entries (keeps first)
- **Out-of-range handling** — nullifies anomalous `meanpressure` values (e.g. -3, 7679), clips other variables to physically valid ranges
- **Missing value imputation** — forward-fill then backward-fill, appropriate for time-series continuity
- **Temporal features** — extracts `month`, `day_of_year`, and `season`
- **Rolling averages** — 7-day and 30-day windows for `meantemp` and `humidity`
- **Lag features** — 1-day and 7-day lags for all numeric variables

Output: `data/silver/silver_data.csv`

### Step 4: Data Quality Validation

```bash
python src/validate.py
```

Runs 14 automated checks on the silver dataset:
- Schema validation (column types)
- Completeness (no missing values, no duplicate dates)
- Value range checks for all climate variables
- Temporal continuity (no gaps > 7 days, sorted order)
- Row count consistency between bronze and silver layers

Exits with error code 1 if any check fails.

### Step 5: Gold Layer — ML-Ready Dataset

```bash
python src/prepare_gold.py
```

Transforms silver data into a supervised learning dataset for **next-day mean temperature forecasting**:
- Drops metadata columns (`batch_id`, `ingestion_timestamp`, `source_file`)
- Creates forecast target: `meantemp` shifted by 1 day
- One-hot encodes the `season` column
- Selects features based on correlation with the target (threshold=0.1), dropping low-signal features like `month`, `day_of_year`, and `season_autumn`

Final dataset: 1384 rows, 19 features + target. No model training is performed.

Output: `data/gold/gold_data.csv`

### Step 6: DVC Pipeline

All stages are defined in `dvc.yaml` with explicit dependencies, parameters, and outputs.
The full pipeline can be reproduced with:

```bash
dvc repro
```

Pipeline DAG:

```
train.csv.dvc → split → ingest → clean → prepare_gold
                                    ↘ validate
```

Each stage only re-runs when its dependencies or parameters change.
Use `dvc dag` to visualize the pipeline and `dvc repro --dry` to preview what would run.

### Step 7: Incremental Batch Ingestion

To simulate new data arriving over time, the `max_batch` parameter in `params.yaml`
controls how many batches are ingested. Incrementing it and re-running the pipeline
triggers a full re-processing of all downstream stages:

```bash
# Edit params.yaml: set max_batch to N (1-5)
dvc repro
git add params.yaml dvc.lock
git commit -m "ingested batch N into pipeline"
```

Each commit captures a snapshot of the data at that point in time. The git history
shows the data growing from 271 gold rows (batch 1) to 1384 rows (all 5 batches),
with DVC tracking the actual data files at each version.
