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
