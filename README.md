# ML ROP Estimation Pipeline

A machine learning pipeline for estimating **Rate of Penetration (ROP)** from drilling parameter data.

This repository focuses on a practical ML end-to-end workflow:
- load and standardize drilling data
- clean outliers and missing values
- impute missing values
- smooth the target feature per well_id
- build a 30 meter interval-based train/test split
- train multiple regression models
- save metrics, logs, and plots

## What the pipeline does

The main pipeline script performs the following stages:
1. load the CSV dataset and detect/remove a units row if present
2. standardize columns and fix known data issues such as `torque` scaling for `wellname=2.0`
3. replace outliers with `NaN`
4. impute missing values with `IterativeImputer` using a small `RandomForestRegressor`
5. smooth target variable `rop` by well using a low-pass filter (rolling mean) to omit noise
6. build an interval-block train/test split within wells to reduce leakage from neighbouring depth samples
7. train and evaluate five models:
   - Ridge Regression
   - Gradient Boosting Regressor
   - HistGradientBoostingRegressor
   - Support Vector Regressor (SVR)
   - Artificial Neural Network (ANN-MLP)
8. export Excel results, logs, and plots

## Repository structure

```text
.
├── data/
├── outputs/
│   ├── logs/
│   └── plots/
├── rop_estimation_pipeline_log.py
├── visualizations.py
├── requirements.txt
├── README.md
└── REPORT.md
```

## Quick start

### 1) Install dependencies

```bash
pip install -r requirements.txt
```

### 2) Run the pipeline

```bash
python rop_estimation_pipeline_log.py
```

## Expected input columns

The pipeline expects a drilling dataset containing at least these columns (schema):

- `wellname`
- `depth`
- `bitsize`
- `rop`
- `torque`
- `spp`
- `rpm`
- `mw_in`
- `mw_out`
- `flowrate`
- `wob`

## Output artifacts

Running the pipeline produces:
- `outputs/model_results.xlsx`
- `outputs/logs/rop_pipeline.log`
- plot files under `outputs/plots/`

The pipeline should save **26 plot files**, the Excel workbook (metrics performance) with estimated run time of **74.55 seconds**.

## Visualizations generated

`visualizations.py` is responsible for the plotting layer. It generates:
- data cleaning histograms before cleaning, after outlier removal, and after imputation
- correlation heatmap, showing any collinearity between input features and target variable
- ROP / torque / RPM boxplots by bit size
- actual vs predicted test samples scatter plots for each model
- well overlay plots for predicted vs actual ROP
- input well logs with predictions
- ANN training history
- feature importance plots (based on Gradient boosting regressor model)

## Latest model performance

The latest logged run on **2026-04-04** reported the following test metrics:

| Model | Test R² | Test MAE | Test RMSE |
|---|---:|---:|---:|
| SVR | 0.6387 | 1.6018 | 2.2594 |
| Hist Gradient Boosting | 0.5449 | 1.9575 | 2.5360 |
| ANN | 0.5119 | 1.9899 | 2.6263 |
| Gradient Boosting | 0.4539 | 2.1014 | 2.7780 |
| Ridge Regression | 0.4029 | 2.2741 | 2.9047 |

In this run, **SVR** was the strongest model on the held-out test set.

## Notes on implementation

- the train/test split is block-based within each well, which is better than a naïve random split for sequential depth data
- the pipeline logs timings for major steps
- Gradient Boosting and HistGradientBoosting also generate permutation importance and SHAP-based importance tables
- the ANN training includes learning-rate scheduling and early stopping helpers

## License

MIT
