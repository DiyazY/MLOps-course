# Assignment 4: ModelOps Report

## 1. Time-Series Modeling Strategy

### Problem Definition

The task is next-day mean temperature forecasting for Delhi, using the Gold dataset produced in Assignment 3. The Gold dataset contains 19 engineered features (lag values, rolling averages, season indicators) and a target variable (`meantemp` at t+1). The data evolves incrementally as new time batches are ingested.

### Model Selection: Gradient Boosting Regressor

The primary model is scikit-learn's `GradientBoostingRegressor`. This choice is justified by:

- **Temporal structure**: The Gold dataset already encodes temporal dynamics through lag features (1-day, 7-day) and rolling averages (7-day, 30-day). These engineered features convert the time-series problem into a tabular regression task, which tree-based models handle naturally without requiring sequential architectures.
- **Feature availability**: With 19 numeric features including temperature lags, humidity trends, pressure readings, and seasonal indicators, gradient boosting can capture non-linear interactions between climate variables (e.g., how humidity's effect on temperature varies by season).
- **Data volume**: The dataset ranges from 828 rows (v1, 3 batches) to 1384 rows (v2, all 5 batches). This is sufficient for gradient boosting but too small for deep learning approaches, which typically require thousands of samples to generalize well.
- **Update frequency**: With batches arriving periodically, the model must be retrained efficiently. Gradient boosting trains in under 3 seconds, supporting rapid iteration cycles.

### Model Comparison

Three model types were compared on Gold data v1 (828 rows):

| Model             | Test RMSE | Test MAE | Test R2 | CV RMSE (mean) |
| ----------------- | --------- | -------- | ------- | -------------- |
| Gradient Boosting | 2.0166    | 1.6035   | 0.9334  | 2.8183         |
| Random Forest     | 1.9539    | 1.5475   | 0.9375  | 2.3688         |
| Linear Regression | 1.6994    | 1.3058   | 0.9527  | 3.9821         |

**Why not deep learning (LSTM, Transformer)?** The dataset is too small (~1000 rows) for neural networks, which risk severe overfitting. Additionally, the temporal structure is already encoded in the features — a sequential architecture would add complexity without benefit.

**Why not ARIMA/SARIMA?** Classical time-series models operate on univariate or low-dimensional data. With 19 multivariate features already engineered, a regression approach is more natural. ARIMA also requires stationarity transformations that are unnecessary with the lag-based feature design.

**Why not rule-based/heuristic?** While simpler, a heuristic approach (e.g., "tomorrow's temperature equals today's") would not leverage the rich feature set available. The baseline RMSE of a naive persistence forecast is approximately 2.5 C — all ML models improve on this significantly.

## 2. KPI Definition

### Primary KPI: RMSE (Root Mean Squared Error)

RMSE was selected as the primary metric because it penalizes large forecasting errors more heavily than small ones, which is appropriate for temperature forecasting where large deviations have outsized practical impact (e.g., extreme weather warnings). RMSE is measured in the same units as the target (degrees Celsius), making it directly interpretable.

**Acceptable threshold**: RMSE < 3.0 C. This threshold was chosen because a 3-degree error represents the boundary between a useful forecast and one that provides little value over climatological averages. Delhi's daily temperature range spans approximately 10-40 C, so 3 C represents roughly 10% of the range.

### Secondary KPIs

**MAE (Mean Absolute Error)**: Threshold < 2.0 C. MAE provides a linear measure of average prediction error without the squaring bias of RMSE. If RMSE is significantly higher than MAE, it indicates the presence of occasional large errors.

**R-squared (R2)**: Threshold > 0.85. R2 measures how much variance the model explains. A value above 0.85 indicates the model captures the dominant temperature patterns (seasonal cycles, trends). R2 also serves as a stability indicator — a significant drop in R2 when retraining on new data signals distribution shift.

### KPI Usage for Model Comparison

When a new data version triggers retraining, models are compared on:

1. **Primary criterion**: RMSE delta between old and new model. Negative delta (improvement) favors promotion.
2. **Threshold gate**: The new model must pass all KPI thresholds (RMSE < 3.0, R2 > 0.85) to be eligible for promotion. A model that degrades beyond thresholds is rejected regardless of relative improvement.
3. **Stability check**: Cross-validation RMSE standard deviation across time-series folds. High variance indicates the model is sensitive to the specific time period, suggesting potential overfitting.

## 3. Model Versioning and Update Logic

### Data-Model Linkage

Two Gold data versions are used, directly derived from Assignment 3's incremental pipeline:

| Data Version | Assignment 3 Commit | max_batch | Gold Rows | Period              |
| ------------ | ------------------- | --------- | --------- | ------------------- |
| v1           | `49f6423`           | 3         | 828       | Jan 2013 - May 2015 |
| v2           | `5205f97`           | 5         | 1384      | Jan 2013 - Jan 2017 |

Each MLflow run is tagged with `data_version`, `data_path`, `model_type`, `train_date_range`, and `test_date_range`, creating explicit traceability between data and model versions.

### Experiment Tracking with MLflow

All experiments are tracked using MLflow (local file-based tracking store). For each run, the following are logged:

- **Parameters**: All hyperparameters (n_estimators, max_depth, learning_rate, etc.), data version, model type, test split ratio
- **Metrics**: Training and validation RMSE/MAE/R2, 5-fold time-series cross-validation metrics (mean and std)
- **Tags**: Data version, date ranges, KPI pass/fail status, number of features and samples
- **Artifacts**: Serialized model (sklearn format), feature column list (JSON)

In total, 88 experiment runs were logged: 5 initial training runs (3 model types on v1, 2 on v2), 81 hyperparameter tuning configurations, and 2 automated update runs.

### Model Update Logic

When Gold data v2 became available (adding batches 4 and 5), all models were retrained:

| Model             | v1 RMSE | v2 RMSE | Delta   | Decision |
| ----------------- | ------- | ------- | ------- | -------- |
| Gradient Boosting | 2.0166  | 1.8588  | -0.1578 | Improved |
| Random Forest     | 1.9539  | 1.6718  | -0.2821 | Improved |

Both models improved with more data, confirming that the additional time batches added useful signal rather than noise. The Random Forest on v2 achieved the best overall RMSE (1.6718).

### Hyperparameter Tuning

A grid search over 81 configurations was performed on Gold v2:

- `n_estimators`: [50, 100, 200]
- `max_depth`: [3, 5, 7]
- `learning_rate`: [0.05, 0.1, 0.2]
- `min_samples_split`: [3, 5, 10]

Best configuration: `n_estimators=100, max_depth=3, learning_rate=0.05, min_samples_split=10` achieving RMSE=1.7147. Shallow trees (depth=3) with slower learning rates outperformed deeper trees, indicating that the feature space is well-structured and does not require complex decision boundaries.

## 4. Automation

### Continuous Model Update (CMU) Mechanism

The `update_model.py` script implements a semi-automated CT pipeline:

1. **Change detection**: Computes MD5 hashes of Gold data files and compares against the last known state (stored in `models/data_state.json`). Any new or modified file triggers the update pipeline.
2. **Automatic retraining**: Trains a new Gradient Boosting model on the changed data version using the current configuration from `params.yaml`.
3. **KPI evaluation**: Computes all metrics and checks against thresholds.
4. **Model comparison**: Compares the new model's RMSE against the current best model in MLflow. If improved, the new model is promoted.
5. **State persistence**: Saves the current data state to prevent redundant retraining on unchanged data.

The automation flow:

```
New gold data detected -> Retrain -> Evaluate KPIs -> Compare with best -> Promote/Reject -> Save state
```

Running `python src/update_model.py` after adding a new Gold data file to `data/gold/` triggers the full pipeline automatically. Manual intervention is only needed if the new model fails KPI thresholds and requires investigation.

## 5. Production Prediction on test.csv

The test dataset (DailyDelhiClimateTest.csv, 114 rows, Jan-Apr 2017) was processed through the same feature engineering pipeline as the training data:

1. Outlier handling (meanpressure values outside 900-1100 hPa)
2. Rolling averages and lag features computed using the last 30 training rows as context
3. Season one-hot encoding
4. Feature alignment with the training set

The best model (Random Forest on v2, RMSE=1.6718) was loaded from MLflow and applied to the processed test data. Predictions were saved to `evidence/test_predictions.csv`.

Prediction summary: Mean=21.65 C, Std=6.19 C, Min=11.40 C, Max=33.66 C. These statistics are consistent with Delhi's Jan-Apr temperature range.

## 6. Limitations and Assumptions

- **No remote MLflow server**: Tracking uses a local file store. Production deployment would require a centralized MLflow server for team collaboration.
- **Feature set changes between data versions**: Gold v1 has 21 features while v2 has 19 (correlation-based selection drops different features as data grows). This is handled by storing feature lists with each model artifact.
- **Test data context**: Rolling averages for the first few test samples rely on training data for context window bootstrapping. Edge effects at the boundary may reduce accuracy.
- **Single-machine execution**: The pipeline runs locally. Scaling would require container orchestration (connecting back to Assignments 1-2 Kubernetes infrastructure).
- **No concept drift detection**: The current pipeline retrains on schedule (when new data arrives) but does not actively monitor for distribution shift in incoming data.
