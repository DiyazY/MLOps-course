# Assignment 4: ModelOps - Managing Models under Evolving Data

## Overview

ModelOps pipeline for next-day mean temperature forecasting using the Delhi Daily Climate dataset. This project manages the full model lifecycle: training, evaluation, versioning, and automated updates as new data becomes available.

Built on top of the Gold dataset produced in [Assignment 3](../assignment_3/).

## Project Structure

```
assignment_4/
├── params.yaml          # Pipeline configuration and hyperparameters
├── requirements.txt     # Python dependencies
├── src/
│   ├── train.py         # Model training with MLflow tracking
│   ├── evaluate.py      # Model evaluation and KPI computation
│   ├── predict.py       # Production predictions on test.csv
│   ├── tune.py          # Hyperparameter tuning
│   ├── update_model.py  # Continuous model update automation
│   └── prepare_test.py  # Feature engineering for test data
├── data/
│   ├── gold/            # Gold data versions (from Assignment 3)
│   └── test/            # Test dataset and processed features
├── models/              # Serialized model artifacts
├── mlruns/              # MLflow experiment tracking data
└── evidence/            # Screenshots and logs for report
```

## Setup

```bash
cd assignments/assignment_4
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Data Versions

The Gold dataset evolves as new batches are ingested in Assignment 3:

| Version | Source Commit | max_batch | Rows | Period |
|---------|--------------|-----------|------|--------|
| v1      | `49f6423`    | 3         | 828  | Jan 2013 - May 2015 |
| v2      | `5205f97`    | 5         | 1384 | Jan 2013 - Jan 2017 |

## Execution Flow

### 1. Train initial model on Gold v1
```bash
python src/train.py --data-version v1
```

### 2. Evaluate model
```bash
python src/evaluate.py --run-id <mlflow_run_id>
```

### 3. Retrain on Gold v2 (new data arrived)
```bash
python src/train.py --data-version v2
```

### 4. Compare model versions
```bash
python src/evaluate.py --compare
```

### 5. Hyperparameter tuning
```bash
python src/tune.py --data-version v2
```

### 6. Run production predictions on test.csv
```bash
python src/predict.py
```

### 7. Automated model update (bonus)
```bash
python src/update_model.py
```

## KPIs

| KPI | Type | Threshold | Rationale |
|-----|------|-----------|-----------|
| RMSE | Primary | < 3.0 C | Penalizes large errors in temperature forecasting |
| MAE | Secondary | < 2.0 C | Average prediction error in interpretable units |
| R2 | Secondary | > 0.85 | Model explains sufficient variance in temperature |

## MLflow Tracking

View experiments:
```bash
cd assignments/assignment_4
mlflow ui --backend-store-uri mlruns
```
Then open http://localhost:5000 in the browser.
