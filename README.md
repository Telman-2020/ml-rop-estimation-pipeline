<p align="center">
  <img src="assets/drilling-project-thumbnail.png" alt="Real-Time ROP Prediction ML thumbnail" width="900">
</p>

# Real-Time Prediction of Drilling Rate of Penetration

**Author:** Telman Maghrebi  
**Role:** Data Scientist  
**Project type:** End-to-end supervised machine learning regression pipeline  
**Domain:** Operational drilling engineering / directional drilling analytics

---

## Project overview

This project builds a complete machine learning pipeline for estimating **Rate of Penetration (ROP)** from operational drilling parameters. ROP is a key drilling performance target because it reflects how quickly the bit advances through the formation and is strongly influenced by operating conditions such as **weight on bit, rotary speed, torque, standpipe pressure, flow rate, mud properties, bit size, and well context**.

The repository demonstrates a practical, production-style workflow for drilling performance prediction:

- load and standardize raw drilling data
- detect and remove units rows when present
- correct known data-quality issues
- replace outliers with missing values before imputation
- impute missing values using a model-based approach
- smooth noisy ROP values by well
- apply leakage-aware train/test splitting for sequential depth data
- train and benchmark multiple regression models
- export metrics, logs, Excel summaries, and diagnostic plots
- generate EDA and model-performance visualizations

The goal is not only to train a model, but to create a traceable pipeline that can support operational drilling analytics and model comparison under realistic data-quality constraints.

---

## Why this project matters

In drilling operations, ROP prediction can support:

- drilling performance monitoring
- parameter optimization
- early identification of inefficient drilling intervals
- comparison of machine learning models for operational decision support
- reproducible experimentation with logs, plots, and Excel outputs

The project is structured to look like an applied engineering machine learning workflow rather than a notebook-only prototype.

---

## Repository structure

```text
.
├── assets/
│   └── drilling-project-thumbnail.png
├── data/
│   └── DrillingParameters.csv
├── outputs/
│   ├── logs/
│   │   └── rop_pipeline.log
│   ├── plots/
│   │   ├── EDA/
│   │   └── Metrics/
│   └── model_results.xlsx
├── rop_estimation_pipeline_log.py
├── visualizations.py
├── requirements.txt
├── README.md
└── REPORT.md
```

---

## Dataset schema

The pipeline expects a drilling dataset with the following columns:

| Column | Description |
|---|---|
| `wellname` | Well identifier |
| `depth` | Measured depth |
| `bitsize` | Bit size / hole section indicator |
| `rop` | Rate of Penetration target variable |
| `torque` | Surface torque |
| `spp` | Standpipe pressure |
| `rpm` | Rotary speed |
| `mw_in` | Mud weight in |
| `mw_out` | Mud weight out |
| `flowrate` | Mud flow rate |
| `wob` | Weight on bit |

---

## Main pipeline flow

### 1. Logging and output setup

The main script creates a logger that writes progress both to the console and to:

```text
outputs/logs/rop_pipeline.log
```

This gives the project traceability, stage-level timing, and a persistent record of the run.

### 2. Load and standardize data

The script loads:

```python
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "DrillingParameters.csv"
```

It then:

- detects whether the first row contains units
- extracts units into a dictionary
- removes the units row if detected
- standardizes column names to lowercase snake case
- validates required columns
- converts numeric columns
- casts `wellname` and `bitsize` as categorical variables
- applies a torque scaling correction for rows where `wellname == 2.0`

Logged data state from the detailed report:

| Stage | Value |
|---|---:|
| Raw loaded shape | `(9040, 11)` |
| Shape after units-row removal | `(9039, 11)` |
| Missing values after standardization | `398` |
| Extracted unit entries | `9` |
| Torque scaling corrected rows for `wellname=2.0` | `4598` |

### 3. Outlier cleaning before imputation

Numeric outliers are replaced with `NaN` using the IQR rule:

```text
lower bound = Q1 - 1.5 × IQR
upper bound = Q3 + 1.5 × IQR
```

This treats suspicious extreme values as missing values before imputation instead of allowing them to distort model training.

| Column | Additional NaNs from outlier removal |
|---|---:|
| `depth` | 0 |
| `wob` | 2 |
| `rpm` | 566 |
| `torque` | 180 |
| `flowrate` | 0 |
| `spp` | 562 |
| `rop` | 617 |
| `mw_in` | 14 |
| `mw_out` | 31 |

**Total outlier values replaced with `NaN`: 1,972**  
**Total missing values after cleaning: 2,370**

### 4. Model-based imputation

Missing values are imputed using `IterativeImputer` with a small `RandomForestRegressor` estimator.

Imputation workflow:

1. encode categorical variables with `OrdinalEncoder`
2. impute numeric values using iterative model-based imputation
3. inverse-transform categorical variables back to original labels
4. restore categorical dtypes

Logged imputation result:

| Item | Value |
|---|---:|
| Rows | `9039` |
| Numeric columns | `9` |
| Categorical columns | `2` |
| Missing values before imputation | `2370` |
| Missing values after imputation | `0` |
| Imputation time | `~7.24 sec` |

### 5. ROP smoothing

The target variable `rop` is smoothed within each well using a rolling mean:

| Configuration | Value |
|---|---|
| Target | `rop` |
| Group column | `wellname` |
| Rolling window | `10` |
| Minimum periods | `1` |

This reduces high-frequency noise while preserving well-level structure.

### 6. Feature preparation and leakage-aware split

The pipeline prepares machine learning inputs and avoids a naïve random split for sequential depth data.

The README version describes an **interval-block split within wells** to reduce leakage from neighbouring depth samples. The detailed report also records a stratified split by `bitsize`. In practice, the aim is the same: avoid overly optimistic results caused by inappropriate splitting of highly correlated depth samples.

Dropped columns:

```python
DROP_COLS = ["depth", "wellname", "mw_out"]
```

Target:

```python
y = rop
```

Preprocessing includes:

- categorical imputation
- one-hot encoding for `bitsize`
- numeric median imputation
- standard scaling

Logged split sizes:

| Split | Shape |
|---|---:|
| `X_train` | `(6327, 7)` |
| `X_test` | `(2712, 7)` |
| `y_train` | `(6327,)` |
| `y_test` | `(2712,)` |

---

## Models benchmarked

The pipeline trains and compares five regression models:

1. **Ridge Regression** — fast linear baseline
2. **Gradient Boosting Regressor** — tuned with Bayesian optimization
3. **HistGradientBoostingRegressor** — tuned histogram-based gradient boosting model
4. **Support Vector Regressor** — RBF-kernel nonlinear model
5. **Artificial Neural Network / MLP** — Keras-based neural regression model

---

## Model performance

The detailed report ranks models as follows based on the logged run.

### Leaderboard by test R²

| Rank | Model | Test R² | Test MAE | Test RMSE |
|---:|---|---:|---:|---:|
| 1 | Hist Gradient Boosting Regressor | 0.8436 | 1.0403 | 1.4372 |
| 2 | Artificial Neural Network | 0.8269 | 1.0757 | 1.5121 |
| 3 | Support Vector Regressor | 0.7973 | 1.1059 | 1.6361 |
| 4 | Gradient Boosting Regressor | 0.7359 | 1.3529 | 1.8676 |
| 5 | Ridge Regression | 0.4756 | 2.0756 | 2.6316 |

### Key result

**Best overall model in the detailed run:** `HistGradientBoostingRegressor`

It achieved the strongest test R² and the lowest error metrics among the benchmarked models.

> Note: the original README and REPORT files contained different logged leaderboard values. This merged README uses the more detailed REPORT values for the main leaderboard because it includes the full pipeline explanation, model configuration, and timing summary.

---

## Model-specific notes

### Ridge Regression

Ridge Regression acts as a simple linear baseline.

- very fast runtime
- useful reference point
- weaker predictive performance compared with nonlinear models

### Gradient Boosting Regressor

The standard Gradient Boosting Regressor is tuned with Bayesian optimization.

Outputs include:

- metrics table
- cross-validation results
- built-in feature importance
- permutation importance
- SHAP attempt log

The SHAP step failed because of a TreeExplainer additivity-check mismatch, but the pipeline handled this gracefully and saved the error instead of stopping the run.

### HistGradientBoostingRegressor

HGBR is the best classical tree-based model in the logged report.

Advantages:

- strong predictive performance
- efficient histogram-based splitting
- good balance of speed and generalization
- tuned with Bayesian optimization

### Support Vector Regressor

SVR performed strongly with an RBF kernel and fixed hyperparameters.

Configuration:

```text
kernel = rbf
C = 100
epsilon = 0.001
```

### Artificial Neural Network

The ANN uses a separate preprocessing path with one-hot encoding and scaling.

Architecture:

```text
Input layer
Dense(64, ReLU)
Dropout(0.2)
Dense(32, ReLU)
Dropout(0.1)
Dense(1, Linear)
```

Training setup:

- optimizer: Adam
- loss: Huber
- epochs: 200
- batch size: 32
- early stopping
- learning-rate scheduling

The ANN is highly competitive, but it is also the slowest model in the run.

---

## Visual outputs

The project generates EDA and model-performance plots under:

```text
outputs/plots/EDA/
outputs/plots/Metrics/
```

### EDA visuals

#### Data cleaning histograms

<p align="center">
  <img src="outputs/plots/EDA/01_before_cleaning_histograms.png" alt="Before cleaning histograms" width="700">
</p>

<p align="center">
  <img src="outputs/plots/EDA/02_after_outlier_removal_histograms.png" alt="After outlier removal histograms" width="700">
</p>

<p align="center">
  <img src="outputs/plots/EDA/03_after_imputation_histograms.png" alt="After imputation histograms" width="700">
</p>

#### Feature correlation

<p align="center">
  <img src="outputs/plots/EDA/04_correlation_heatmap.png" alt="Correlation heatmap" width="700">
</p>

#### Drilling parameter distributions by bit size

<p align="center">
  <img src="outputs/plots/EDA/05_boxplot_rop_by_bitsize.png" alt="ROP by bit size" width="700">
</p>

<p align="center">
  <img src="outputs/plots/EDA/06_boxplot_torque_by_bitsize.png" alt="Torque by bit size" width="700">
</p>

<p align="center">
  <img src="outputs/plots/EDA/07_boxplot_rpm_by_bitsize.png" alt="RPM by bit size" width="700">
</p>

#### Well log EDA views

<p align="center">
  <img src="outputs/plots/EDA/eda_well_1.0_logs.png" alt="EDA well 1 logs" width="700">
</p>

<p align="center">
  <img src="outputs/plots/EDA/eda_well_2.0_logs.png" alt="EDA well 2 logs" width="700">
</p>

---

### Model evaluation visuals

#### Actual vs predicted plots

<p align="center">
  <img src="outputs/plots/Metrics/hist_gradient_boosting_regressor_actual_vs_predicted.png" alt="HGBR actual vs predicted" width="700">
</p>

<p align="center">
  <img src="outputs/plots/Metrics/ann_model_actual_vs_predicted.png" alt="ANN actual vs predicted" width="700">
</p>

<p align="center">
  <img src="outputs/plots/Metrics/support_vector_regressor_actual_vs_predicted.png" alt="SVR actual vs predicted" width="700">
</p>

<p align="center">
  <img src="outputs/plots/Metrics/gradient_boosting_regressor_actual_vs_predicted.png" alt="GBR actual vs predicted" width="700">
</p>

<p align="center">
  <img src="outputs/plots/Metrics/ridge_regression_actual_vs_predicted.png" alt="Ridge actual vs predicted" width="700">
</p>

#### Well 2 prediction overlays

<p align="center">
  <img src="outputs/plots/Metrics/hist_gradient_boosting_regressor_well_2_overlay.png" alt="HGBR well 2 overlay" width="700">
</p>

<p align="center">
  <img src="outputs/plots/Metrics/ann_model_well_2_overlay.png" alt="ANN well 2 overlay" width="700">
</p>

<p align="center">
  <img src="outputs/plots/Metrics/support_vector_regressor_well_2_overlay.png" alt="SVR well 2 overlay" width="700">
</p>

#### Well logs with predictions

<p align="center">
  <img src="outputs/plots/Metrics/hist_gradient_boosting_regressor_well_2_logs_with_prediction.png" alt="HGBR well 2 logs with prediction" width="700">
</p>

<p align="center">
  <img src="outputs/plots/Metrics/ann_model_well_2_logs_with_prediction.png" alt="ANN well 2 logs with prediction" width="700">
</p>

#### Training and interpretation plots

<p align="center">
  <img src="outputs/plots/Metrics/ann_training_training_history.png" alt="ANN training history" width="700">
</p>

<p align="center">
  <img src="outputs/plots/Metrics/gbr_feature_importance.png" alt="GBR feature importance" width="700">
</p>

---

## Output artifacts

Running the pipeline produces:

| Artifact | Path |
|---|---|
| Excel workbook with metrics and interpretation tables | `outputs/model_results.xlsx` |
| Pipeline log file | `outputs/logs/rop_pipeline.log` |
| EDA visualizations | `outputs/plots/EDA/` |
| Model evaluation visualizations | `outputs/plots/Metrics/` |

The detailed report states that the full pipeline generated **26 plot files**.

---

## Excel workbook contents

The workbook `outputs/model_results.xlsx` includes model metrics and additional diagnostic sheets.

Expected sheets include:

- `ridge_metrics`
- `gbr_metrics`
- `gbr_cv_results`
- `gbr_built_in_feature_importance`
- `gbr_permutation_importance`
- `gbr_shap_importance_error`
- `hgbr_metrics`
- `hgbr_cv_results`
- `hgbr_permutation_importance`
- `hgbr_shap_importance`
- `svr_metrics`
- `ann_metrics`
- `summary`

---

## Timing summary

Approximate logged timings from the detailed run:

| Stage | Runtime |
|---|---:|
| Data build, cleaning, and imputation | 7.34 sec |
| Ridge Regression | 0.02 sec |
| Gradient Boosting Regressor | 12.57 sec |
| HistGradientBoostingRegressor | 32.37 sec |
| Support Vector Regressor | 10.31 sec |
| Artificial Neural Network | 44.39 sec |
| Plot generation | 15.27 sec |
| Full pipeline runtime | 122.79 sec |

---

## How to run

### 1. Create and activate a virtual environment

```bash
python -m venv .venv
```

On Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

On macOS/Linux:

```bash
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the pipeline

```bash
python rop_estimation_pipeline_log.py
```

### 4. Review outputs

After a successful run, check:

```text
outputs/logs/rop_pipeline.log
outputs/model_results.xlsx
outputs/plots/EDA/
outputs/plots/Metrics/
```

---

## Technical highlights

- **End-to-end supervised regression pipeline** for drilling ROP estimation
- **Operational data cleaning** with unit-row detection, type conversion, and known correction handling
- **Outlier-aware imputation strategy** using model-based iterative imputation
- **Well-level target smoothing** to reduce noise in ROP
- **Leakage-conscious splitting** for sequential depth-based drilling data
- **Multi-model benchmarking** across linear, kernel, ensemble, and neural models
- **Bayesian hyperparameter optimization** for gradient boosting models
- **Structured experiment outputs** through Excel, logs, and plots
- **Modular visualization layer** separated into `visualizations.py`

---

## Key takeaways

1. The preprocessing workflow is robust and traceable.
2. Outlier replacement substantially increases missing values, making imputation a critical stage.
3. HGBR achieved the strongest performance in the detailed logged run.
4. ANN was also highly competitive but slower to train.
5. Visualization and logging make the workflow easier to audit and communicate.
6. The project is suitable for portfolio presentation because it combines domain context, machine learning, diagnostics, and reproducible outputs.

---

## Suggested future improvements

- save the best trained model artifact to disk
- add automated model selection based on test R² and RMSE
- add grouped validation by well to evaluate generalization to unseen wells
- improve SHAP handling for the GBR transformed feature matrix
- add configuration files for experiment parameters
- add CI checks for formatting, import validation, and smoke testing
- add a Dockerfile for reproducible execution

---

## License

This project is released under the MIT License.
