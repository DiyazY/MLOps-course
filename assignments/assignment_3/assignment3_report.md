# Assignment 3: DataOps Pipeline — Report

## 1. Environment and Infrastructure

### Choice: Local Machine

The pipeline runs entirely on a local machine using Python 3.13, a virtual environment, and DVC for data versioning.

**Storage solution**: Local filesystem with DVC cache. DVC stores data file hashes in `.dvc` files tracked by git, while the actual data resides in a local cache directory. This provides full version control over data without bloating the git repository.

**Compute solution**: Local Python execution. All scripts run sequentially via `dvc repro`.

**Trade-offs**:

- _Control_: Full control over the environment — no external dependencies, no cloud credentials, no network latency. Reproducibility is guaranteed through DVC lock files and pinned parameters.
- _Scalability_: Not scalable beyond a single machine. However, the Delhi Climate dataset (~1500 rows, ~78KB) is small enough that scalability is not a concern. Processing completes in seconds.
- _Operational complexity_: Minimal. No infrastructure to provision or maintain. A new collaborator only needs Python, `pip install -r requirements.txt`, and `dvc pull` (if remote storage were configured).

A cloud-based alternative (e.g., Databricks with Delta Lake) would offer scalability and collaboration features but would introduce unnecessary complexity for this dataset size. The local setup aligns with the principle of choosing the simplest tool that meets the requirements.

## 2. Tool Selection: DVC

DVC (Data Version Control) was chosen as the primary DataOps tool for several reasons:

- **Git integration**: DVC extends git's versioning model to data files. Each data state is captured alongside code changes, providing a unified version history.
- **Pipeline definition**: `dvc.yaml` declaratively defines stages with dependencies, parameters, and outputs. DVC automatically determines which stages need re-execution when inputs change.
- **Reproducibility**: `dvc repro` guarantees that the pipeline produces identical results given the same inputs and parameters. The `dvc.lock` file captures exact file hashes for verification.
- **Lightweight**: No server or database required. DVC operates as a command-line tool alongside git.

Alternatives considered:

- _Delta Lake_: Provides ACID transactions and schema enforcement for large-scale data lakes. Overkill for this dataset and requires a Spark runtime.
- _LakeFS_: Git-like branching for data lakes. More suited to enterprise data lake management than a single-dataset pipeline.

## 3. Batch Splitting Strategy

The raw dataset (1462 daily records, Jan 2013 — Jan 2017) is split into 5 time-ordered batches of roughly equal size (~292 records each):

| Batch | Period              | Records |
| ----- | ------------------- | ------- |
| 1     | Jan 2013 — Oct 2013 | 293     |
| 2     | Oct 2013 — Aug 2014 | 293     |
| 3     | Aug 2014 — May 2015 | 292     |
| 4     | May 2015 — Mar 2016 | 292     |
| 5     | Mar 2016 — Jan 2017 | 292     |

This temporal split simulates real-world incremental data arrivals. Each batch covers approximately 9-10 months of observations. The split is deterministic (sorted by date, then divided by index).

```
$ python src/split_batches.py
Loaded 1462 records from data/raw/train.csv
Date range: 2013-01-01 00:00:00 to 2017-01-01 00:00:00
Batch 1: 293 records (2013-01-01 to 2013-10-20) -> data/batches/batch_1.csv
Batch 2: 293 records (2013-10-21 to 2014-08-09) -> data/batches/batch_2.csv
Batch 3: 292 records (2014-08-10 to 2015-05-28) -> data/batches/batch_3.csv
Batch 4: 292 records (2015-05-29 to 2016-03-15) -> data/batches/batch_4.csv
Batch 5: 292 records (2016-03-16 to 2017-01-01) -> data/batches/batch_5.csv

Split complete: 5 batches created in data/batches/
```

## 4. Incremental Ingestion Design

The ingestion process is controlled by the `max_batch` parameter in `params.yaml`. Incrementing this value from 1 to 5 and running `dvc repro` after each change simulates new data arriving over time.

Each batch ingestion introduces simulated data quality issues:

- **5% row drops**: Simulates data loss during transfer or ingestion failures
- **3% missing values**: Simulates sensor failures or incomplete records (NaN injected into random numeric cells)
- **2% duplicate rows**: Simulates duplicate event delivery in streaming systems

Quality issue rates and the random seed are configurable in `params.yaml`, ensuring reproducibility. Each batch gets a deterministic seed derived from `random_seed + batch_number`.

Metadata columns are added during ingestion:

- `batch_id`: Identifies which batch the record came from
- `ingestion_timestamp`: When the batch was ingested
- `source_file`: Original batch file name

This metadata enables lineage tracking — any record in the bronze layer can be traced back to its source batch and ingestion time.

```
$ python src/ingest.py

Ingesting batch 1 from data/batches/batch_1.csv
  Original size: 293 rows
  Dropped 14 rows (5%)
  Injected 33 missing values (3%)
  Added 5 duplicate rows (2%)
  Bronze data now has 284 total rows -> data/bronze/bronze_data.csv

Ingesting batch 2 from data/batches/batch_2.csv
  Original size: 293 rows
  Dropped 14 rows (5%)
  Injected 33 missing values (3%)
  Added 5 duplicate rows (2%)
  Bronze data now has 568 total rows -> data/bronze/bronze_data.csv

Ingesting batch 3 from data/batches/batch_3.csv
  Original size: 292 rows
  Dropped 14 rows (5%)
  Injected 33 missing values (3%)
  Added 5 duplicate rows (2%)
  Bronze data now has 851 total rows -> data/bronze/bronze_data.csv

Ingesting batch 4 from data/batches/batch_4.csv
  Original size: 292 rows
  Dropped 14 rows (5%)
  Injected 33 missing values (3%)
  Added 5 duplicate rows (2%)
  Bronze data now has 1134 total rows -> data/bronze/bronze_data.csv

Ingesting batch 5 from data/batches/batch_5.csv
  Original size: 292 rows
  Dropped 14 rows (5%)
  Injected 33 missing values (3%)
  Added 5 duplicate rows (2%)
  Bronze data now has 1417 total rows -> data/bronze/bronze_data.csv
```

## 5. Medallion Architecture: Bronze → Silver → Gold

### Bronze Layer

Raw data with simulated quality issues and ingestion metadata. No cleaning or transformation is applied. The bronze layer preserves the data exactly as it was received, including errors and duplicates. This aligns with the medallion architecture principle of keeping raw data available for reprocessing.

Final state: 1417 rows (with duplicates and missing values).

### Silver Layer

Cleaned and enriched dataset. Transformations applied:

1. **Duplicate removal**: 25 duplicate date entries removed (kept first occurrence)
2. **Out-of-range handling**: 7 anomalous `meanpressure` values nullified (the Delhi dataset has known pressure outliers like -3 and 7679 hPa). Temperature, humidity, and wind speed clipped to physically valid ranges.
3. **Missing value imputation**: Forward-fill then backward-fill — appropriate for time-series data where adjacent days have similar conditions.
4. **Temporal features**: `month`, `day_of_year`, and `season` extracted from the date column. Season mapping reflects Delhi's climate (winter: Dec-Feb, spring: Mar-May, summer: Jun-Aug, autumn: Sep-Nov).
5. **Rolling averages**: 7-day and 30-day windows for `meantemp` and `humidity`, capturing short-term and monthly trends.
6. **Lag features**: 1-day and 7-day lags for all numeric variables, providing historical context for forecasting.

Final state: 1385 rows, 23 columns.

```
$ python src/clean.py
Loaded bronze data: 1417 rows

Cleaning:
  Removed 25 duplicate date entries
  Nullified 7 out-of-range meanpressure values
  Clipped 0 out-of-range meantemp values
  Clipped 0 out-of-range humidity values
  Clipped 0 negative wind_speed values
  Imputed 168 missing values (ffill + bfill)

Feature engineering:
  Added temporal features: month, day_of_year, season
  Added rolling features: windows=[7, 30]
  Added lag features: lags=[1, 7]

Dropped 7 rows with NaN from lag features

Silver data: 1385 rows -> data/silver/silver_data.csv

Column list (23):
  - date
  - meantemp
  - humidity
  - wind_speed
  - meanpressure
  - batch_id
  - ingestion_timestamp
  - source_file
  - month
  - day_of_year
  - season
  - meantemp_rolling_7d
  - humidity_rolling_7d
  - meantemp_rolling_30d
  - humidity_rolling_30d
  - meantemp_lag_1d
  - humidity_lag_1d
  - wind_speed_lag_1d
  - meanpressure_lag_1d
  - meantemp_lag_7d
  - humidity_lag_7d
  - wind_speed_lag_7d
  - meanpressure_lag_7d
```

### Gold Layer

ML-ready dataset for **next-day mean temperature forecasting**:

- Metadata columns (`batch_id`, `ingestion_timestamp`, `source_file`) dropped — not relevant for prediction
- Target variable created: `meantemp` shifted by 1 day (t+1)
- `season` column one-hot encoded into binary features
- Features selected based on Pearson correlation with the target (threshold > 0.1)
- 3 low-signal features dropped: `month`, `day_of_year`, `season_autumn`

No model training is performed — the gold layer is structured and ready for a downstream modelling step.

Final state: 1384 rows, 19 features + 1 target.

```
$ python src/prepare_gold.py
Loaded silver data: 1385 rows
Created target: meantemp at t+1 (1384 rows after dropping last 1)
Dropped metadata columns: ['batch_id', 'ingestion_timestamp', 'source_file']

Feature selection:
  Feature correlations with 'target' (threshold=0.1):
    meantemp: 0.973
    meantemp_rolling_7d: 0.958
    meantemp_lag_1d: 0.954
    meantemp_rolling_30d: 0.909
    meantemp_lag_7d: 0.908
    meanpressure: 0.836
    meanpressure_lag_1d: 0.823
    meanpressure_lag_7d: 0.796
    season_winter: 0.778
    humidity_rolling_30d: 0.619
    humidity_rolling_7d: 0.567
    humidity: 0.532
    humidity_lag_1d: 0.514
    humidity_lag_7d: 0.496
    season_summer: 0.493
    wind_speed_lag_7d: 0.325
    wind_speed_lag_1d: 0.281
    wind_speed: 0.277
    season_spring: 0.245
  Dropped 3 low-correlation features: ['month', 'day_of_year', 'season_autumn']

Gold dataset: 1384 rows, 21 columns -> data/gold/gold_data.csv
Features (19): ['meantemp', 'meantemp_rolling_7d', 'meantemp_lag_1d',
  'meantemp_rolling_30d', 'meantemp_lag_7d', 'meanpressure',
  'meanpressure_lag_1d', 'meanpressure_lag_7d', 'season_winter',
  'humidity_rolling_30d', 'humidity_rolling_7d', 'humidity',
  'humidity_lag_1d', 'humidity_lag_7d', 'season_summer',
  'wind_speed_lag_7d', 'wind_speed_lag_1d', 'wind_speed', 'season_spring']
Target: next-day meantemp
```

## 6. Data Validation and Testing

The validation script (`src/validate.py`) runs 14 automated checks on the silver dataset:

**Schema validation** (5 checks):

- Date column is datetime type
- All four climate variables are numeric types

**Completeness** (2 checks):

- No missing values in core columns after imputation
- No duplicate dates after deduplication

**Value range checks** (4 checks):

- `meantemp` in [-10, 50] °C
- `humidity` in [0, 100] %
- `wind_speed` >= 0
- `meanpressure` in [900, 1100] hPa

**Temporal continuity** (2 checks):

- No date gaps larger than 7 days
- Data is sorted chronologically

**Row count consistency** (1 check):

- Silver row count is between 50-100% of bronze count (ensures no catastrophic data loss during cleaning)

The validation script exits with error code 1 if any check fails, which halts the DVC pipeline and prevents invalid data from reaching the gold layer.

```
$ python src/validate.py
Validating silver data: 1385 rows from data/silver/silver_data.csv

Schema validation:
  [✓] Date column is datetime
  [✓] meantemp is numeric
  [✓] humidity is numeric
  [✓] wind_speed is numeric
  [✓] meanpressure is numeric

Completeness:
  [✓] No missing values in core columns — 0 nulls found
  [✓] No duplicate dates — 0 duplicates

Value range checks:
  [✓] meantemp in [-10, 50] — min=9.00, max=38.71
  [✓] humidity in [0, 100] — min=13.43, max=100.00
  [✓] wind_speed >= 0 — min=0.00
  [✓] meanpressure in [900, 1100] — min=938.07, max=1022.12

Temporal continuity:
  [✓] No date gaps larger than 7 days — max gap = 3 days 00:00:00
  [✓] Data is sorted by date

Row count consistency:
  [✓] Silver row count is reasonable vs bronze — silver=1385, bronze=1417, ratio=0.98

==================================================
Validation 14/14 checks passed
All checks passed — data is ready for gold layer
```

## 7. Versioning and Reproducibility

### Pipeline DAG

```
$ dvc dag
      +------------------------+
      | data/raw/train.csv.dvc |
      +------------------------+
                    *
                    *
                    *
               +-------+
               | split |
               +-------+
                    *
                    *
                    *
              +--------+
              | ingest |
              +--------+
             ***        ***
            *              *
          **                ***
    +-------+                  *
    | clean |*                 *
    +-------+ ****             *
        *         ****         *
        *             ****     *
        *                 **   *
+--------------+         +----------+
| prepare_gold |         | validate |
+--------------+         +----------+
```

### Full Pipeline Execution

```
$ dvc repro --force

Running stage 'split':
> python src/split_batches.py
Loaded 1462 records from data/raw/train.csv
Batch 1: 293 records (2013-01-01 to 2013-10-20) -> data/batches/batch_1.csv
Batch 2: 293 records (2013-10-21 to 2014-08-09) -> data/batches/batch_2.csv
Batch 3: 292 records (2014-08-10 to 2015-05-28) -> data/batches/batch_3.csv
Batch 4: 292 records (2015-05-29 to 2016-03-15) -> data/batches/batch_4.csv
Batch 5: 292 records (2016-03-16 to 2017-01-01) -> data/batches/batch_5.csv

Running stage 'ingest':
> python src/ingest.py
Bronze data now has 1417 total rows -> data/bronze/bronze_data.csv

Running stage 'clean':
> python src/clean.py
Silver data: 1385 rows -> data/silver/silver_data.csv

Running stage 'validate':
> python src/validate.py
Validation 14/14 checks passed

Running stage 'prepare_gold':
> python src/prepare_gold.py
Gold dataset: 1384 rows, 21 columns -> data/gold/gold_data.csv
```

### Git + DVC Version History

The incremental workflow produces a clear version history:

```
$ git log --format="%h %ad %s" --date=short -- assignments/assignment_3/
ba95ed7 2026-02-15 updated README with incremental ingestion workflow
5205f97 2026-02-15 ingested batch 5 into pipeline
47a8463 2026-02-15 ingested batch 4 into pipeline
49f6423 2026-02-15 ingested batch 3 into pipeline
bbf6d2b 2026-02-15 ingested batch 2 into pipeline
124f3ad 2026-02-15 ingested batch 1 into pipeline
ecfdbef 2026-02-15 defined DVC pipeline for reproducible execution
9fcd0c3 2026-02-15 Implemented gold layer ML-ready dataset preparation
1a185b3 2026-02-15 added silver layer data cleaning and validation
4e2e3f5 2026-02-15 Added bronze layer ingestion script and update README with execution steps
27beeba 2026-02-15 Added dataset and batch splitting logic
3b18714 2026-02-15 Add initial implementation of DataOps pipeline for assignment 3
```

Parameter evolution across incremental ingestion commits:

```
$ dvc params diff 124f3ad 5205f97
Path         Param             124f3ad    5205f97
params.yaml  ingest.max_batch  1          5
```

| Commit    | max_batch | Gold Rows | Description               |
| --------- | --------- | --------- | ------------------------- |
| `124f3ad` | 1         | 271       | First batch: Jan-Oct 2013 |
| `bbf6d2b` | 2         | 550       | Added Oct 2013-Aug 2014   |
| `49f6423` | 3         | 828       | Added Aug 2014-May 2015   |
| `47a8463` | 4         | 1106      | Added May 2015-Mar 2016   |
| `5205f97` | 5         | 1384      | Full dataset: all batches |

Each commit captures:

- The code state (git)
- The data state (DVC lock file hashes)
- The pipeline parameters (`params.yaml`)

### Reproducibility

Any previous data version can be restored:

```bash
git checkout <commit-hash>
dvc checkout
```

The full pipeline can be re-executed from scratch:

```bash
dvc repro
```

DVC ensures that only stages with changed dependencies are re-run, making incremental updates efficient.

## 8. Limitations and Assumptions

- **No remote DVC storage**: Data is only stored locally. In a production setting, a remote storage backend (S3, GCS, etc.) would be configured for team collaboration.
- **Simulated quality issues**: The data degradation is artificial. Real-world pipelines would encounter more complex and unpredictable issues.
- **Sequential ingestion**: Batches are re-ingested from scratch each time `max_batch` increases. A production system would only process new batches incrementally.
- **Small dataset**: At ~1500 rows, the dataset doesn't stress-test scalability. The architecture would need adaptation (e.g., chunked processing, Spark) for larger datasets.
- **Feature selection is static**: Correlation-based feature selection runs each time the gold layer is rebuilt. With more data, the selected features could change between versions.
