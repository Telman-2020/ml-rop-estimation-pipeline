update this readme :# ROP Estimation Pipeline

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)]()
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange.svg)]()
[![TensorFlow](https://img.shields.io/badge/TensorFlow-ANN-red.svg)]()
[![License](https://img.shields.io/badge/License-MIT-green.svg)]()

A production-style machine learning pipeline for estimating **Rate of
Penetration (ROP)** from drilling parameters data.

------------------------------------------------------------------------

## 📌 Project Title & Description

### What is this project?

The **ROP Estimation Pipeline** is a modular machine learning workflow
designed to estimate ROP from drilling parameters log data using multiple
regression models. This prediction could be useful to optimize the drilling operation.

### What problem does it solve?

-   Noisy sensor data\
-   Missing values\
-   Complex feature relationships

This pipeline automates cleaning, imputation, modelling, and evaluation.

### Who is it for?

-   Data scientists\
-   Drilling Engineers working with rig data\
-   ML practitioners building pipelines

------------------------------------------------------------------------

## ▶️ Usage / Example

### Run the pipeline

``` bash
python rop_estimation_pipeline_log.py
```

### Python usage

``` python
from rop_estimation_pipeline_log import main

artifacts = main()
print(artifacts["excel_path"])
```

### Access predictions

``` python
results = artifacts["results"]
print(results["gbr"].y_pred[:5])
```

------------------------------------------------------------------------

## ⚙️ Configuration

``` python
main(
    csv_path="data/DrillingParameters.csv",
    plots_dir="outputs/plots",
    results_excel_path="outputs/model_results.xlsx",
    log_file="outputs/logs/rop_pipeline.log",
)
```

### Required inputs

-   CSV dataset
-   Required columns:
    - `wellname`  
    - `bitsize` (unit: *inches*)  
    - `torque` (unit: klbf·in or k.lbm)  
    - `rop` (rate of penetration — *target variable*)  
    - `spp` (standpipe pressure, *psi*)  
    - `rpm` (drill string rotations per minute)  
    - `mw_in` (mud weight in)  
    - `mw_out` (mud weight out — may be redundant)  
    - `flowrate` (mud flow rate, *gpm*)  
    - `wob` (weight on bit, unit: *klbf* or lbf) 

------------------------------------------------------------------------

## 📊 Outputs

-   Excel results → outputs/model_results.xlsx\
-   Logs → outputs/logs/rop_pipeline.log\
-   Plots → outputs/plots/

------------------------------------------------------------------------

## 📸 Visual Outputs

## 📊 Exploratory Data Analysis

### Distribution of Features Before Cleaning
![Feature Distributions Before Cleaning](outputs/plots/EDA/01_before_cleaning_histograms.png)

### Distribution of Features After Outlier Removal
![Feature Distributions After Outlier Removal](outputs/plots/EDA/02_after_outlier_removal_histograms.png)

### Correlation Heatmap
![Correlation Heatmap](outputs/plots/EDA/04_correlation_heatmap.png)

---

## 📈 Model Performance

### 📊 Actual vs Predicted ROP (All Models)

#### 📈 Ridge Regression
<img src="outputs/plots/Metrics/ridge_regression_actual_vs_predicted.png" width="500"/>

#### 🌳 Gradient Boosting Regressor
<img src="outputs/plots/Metrics/gradient_boosting_regressor_actual_vs_predicted.png" width="500"/>

#### 🌲 Histogram-Based Gradient Boosting
<img src="outputs/plots/Metrics/hist_gradient_boosting_regressor_actual_vs_predicted.png" width="500"/>

#### 📐 Support Vector Regressor
<img src="outputs/plots/Metrics/support_vector_regressor_actual_vs_predicted.png" width="500"/>

#### 🤖 Artificial Neural Network (MLP)
<img src="outputs/plots/Metrics/ann_model_actual_vs_predicted.png" width="500"/>

---

## 🛢️ Model Predictions on Well #2

### 📈 Ridge Regression – Well #2 Logs with Predicted ROP (Last Log)
![Ridge Regression Prediction](outputs/plots/Metrics/ridge_regression_well_2_logs_with_prediction.png)

### 🌳 Gradient Boosting Regressor – Well #2 Logs with Predicted ROP (Last Log)
![Gradient Boosting Prediction](outputs/plots/Metrics/gradient_boosting_regressor_well_2_logs_with_prediction.png)

### 🌲 Histogram-Based Gradient Boosting Regressor – Well #2 Logs with Predicted ROP (Last Log)
![Hist Gradient Boosting Prediction](outputs/plots/Metrics/hist_gradient_boosting_regressor_well_2_logs_with_prediction.png)

### 📐 Support Vector Regressor – Well #2 Logs with Predicted ROP (Last Log)
![SVR Prediction](outputs/plots/Metrics/support_vector_regressor_well_2_logs_with_prediction.png)

### 🤖 Artificial Neural Network (MLP) – Well #2 Logs with Predicted ROP (Last Log)
![ANN Prediction](outputs/plots/Metrics/ann_model_well_2_logs_with_prediction.png)

------------------------------------------------------------------------

## 🧠 How to Interpret Results

-   High R² → better model\
-   Low RMSE → fewer large errors\
-   Small train-test gap → good generalization

------------------------------------------------------------------------

## Installation

``` bash
pip install numpy pandas scikit-learn scikit-optimize shap openpyxl tensorflow keras
```

------------------------------------------------------------------------

## Project Structure

    .
    ├── rop_estimation_pipeline_log.py
    ├── visualizations.py
    ├── data/
    ├── outputs/
    └── README.md

------------------------------------------------------------------------

## License

MIT License