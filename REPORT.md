# ROP Estimation Pipeline

This project implements a refactored machine learning pipeline for **ROP (Rate of Penetration) estimation**. The main script is `rop_estimation_pipeline_log.py`, and the run summary in this README is based on the captured execution log in `rop_pipeline.log`.

The pipeline is structured to:

- load and standardize drilling data
- clean outliers before imputation
- impute missing values with a model-based approach
- smooth the target after imputation
- split data for training and testing
- train and evaluate multiple regression models
- export metrics and diagnostic tables to Excel
- generate EDA and model evaluation plots using functions defined in `visualizations.py`
- log every major stage with timing information

---

## Project structure

```text
project/
├── rop_estimation_pipeline_log.py
├── visualizations.py
├── data/
│   └── DrillingParameters.csv
└── outputs/
    ├── logs/
    │   └── rop_pipeline.log
    ├── model_results.xlsx
    └── plots/
        ├── EDA/
        └── Metrics/
```

---

## Main pipeline flow

The script runs through the following major stages.

### 1. Logging setup

The pipeline starts by creating a logger that writes both to the console and to a file. It also creates missing output directories automatically.

What this gives you:

- progress visibility in the terminal or VS Code console
- a persistent `rop_pipeline.log` file for traceability
- elapsed time for each major section

---

### 2. Load and standardize data

The script reads the CSV file from:

```python
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "DrillingParameters.csv"
```

Then it performs these actions:

- checks whether the first row contains units
- extracts those units into a dictionary
- removes the units row from the dataset if detected
- standardizes column names to lowercase with underscores
- validates that required columns exist
- casts `wellname` and `bitsize` to categorical types
- converts the remaining columns to numeric
- applies a torque scaling correction for rows where `wellname == 2.0`

### Data state after loading and standardization

- raw loaded shape: **(9040, 11)**
- after removing the units row: **(9039, 11)**
- total missing values after standardization: **398**
- unit entries extracted: **9**
- torque scaling applied to **4598** rows for `wellname=2.0`

Columns used by the standardized dataframe:

- `wellname`
- `depth`
- `wob`
- `rpm`
- `torque`
- `flowrate`
- `spp`
- `rop`
- `mw_in`
- `mw_out`
- `bitsize`

---

### 3. Cleaning before imputation

Before filling missing values, the pipeline replaces numeric outliers with `NaN` using the IQR rule:

- lower bound = `Q1 - 1.5 * IQR`
- upper bound = `Q3 + 1.5 * IQR`

This is done column by column so suspicious extreme values are treated as missing and later imputed instead of being left in the training data.

### Outlier replacement summary

Additional `NaN` values added per column:

- `depth`: 0
- `wob`: 2
- `rpm`: 566
- `torque`: 180
- `flowrate`: 0
- `spp`: 562
- `rop`: 617
- `mw_in`: 14
- `mw_out`: 31

Totals:

- outlier values replaced with `NaN`: **1972**
- cleaned dataframe shape: **(9039, 11)**
- total missing values after cleaning: **2370**

This is an important stage because it substantially increases the amount of missing data, but in a controlled way that prepares the data for model-based imputation.

---

### 4. Model-based imputation

The pipeline imputes missing values using a three-step process.

#### Step 1: Encode categorical features

- categorical columns considered: `wellname`, `bitsize`
- encoded using `OrdinalEncoder`

#### Step 2: Iterative imputation

- numeric columns considered: 9
- imputer: `IterativeImputer`
- estimator inside imputer: `RandomForestRegressor(n_estimators=5, n_jobs=-1, random_state=42)`

#### Step 3: Restore categorical variables

- inverse transform categorical columns back to original labels
- cast categorical columns back to category dtype

### Imputation result

- rows: **9039**
- numeric columns: **9**
- categorical columns: **2**
- missing values before imputation: **2370**
- missing values after imputation: **0**
- imputation time: about **7.24 sec**

---

### 5. Post-imputation smoothing

After imputation, the pipeline applies rolling-mean smoothing to the target variable `rop`.

Configuration:

- target: `rop`
- group column: `wellname`
- rolling window: **10**
- minimum periods: **1**

This means ROP is smoothed separately within each well using a 10-sample moving average.

---

### 6. Build ML features and train/test split

The prepared dataframe is then converted into machine learning inputs.

#### Target

- `rop`

#### Dropped columns

- `depth`
- `wellname`
- `mw_out`

#### Feature handling

- categorical feature: `bitsize`
- numeric features: remaining 6 columns
- split uses stratification by `bitsize`

#### Preprocessing pipeline

Categorical branch:

- `SimpleImputer(strategy="most_frequent")`
- `OneHotEncoder(handle_unknown="ignore", sparse_output=False)`

Numeric branch:

- `SimpleImputer(strategy="median")`
- `StandardScaler()`

#### Split sizes

- training set: `X_train` **(6327, 7)**
- test set: `X_test` **(2712, 7)**
- `y_train`: **(6327, )**
- `y_test`: **(2712, )**

---

## Models trained

The pipeline trains five regression models in sequence.

### 1. Ridge Regression

Configuration:

- `Ridge(alpha=10.0)`
- wrapped with the preprocessing pipeline

Results:

- train R²: **0.4535**
- test R²: **0.4756**
- test MAE: **2.0756**
- test MSE: **6.9251**
- test RMSE: **2.6316**
- runtime: about **0.02 sec**

Interpretation:

- this is the weakest model in the run
- it likely serves as a simple linear baseline

---

### 2. Gradient Boosting Regressor (GBR)

This model is tuned with Bayesian optimization.

Search setup:

- estimator: `GradientBoostingRegressor`
- `BayesSearchCV`
- iterations: **5**
- cross-validation folds: **2**
- scoring: **R²**

Best logged parameters:

- learning rate: **0.06168**
- loss: **huber**
- max depth: **3**
- max features: **log2**
- min samples leaf: **13**
- min samples split: **13**
- n estimators: **148**
- n iter no change: **14**
- subsample: **0.82798**
- tol: **5.94e-05**
- validation fraction: **0.13416**

Results:

- train R²: **0.7372**
- test R²: **0.7359**
- test MAE: **1.3529**
- test RMSE: **1.8676**
- Bayesian search runtime: about **9.20 sec**
- full model stage runtime: about **12.57 sec**

Interpretation artifacts generated:

- built-in feature importance table
- permutation importance table
- SHAP importance attempted

Important note:

- the GBR SHAP step **failed** due to a TreeExplainer additivity check mismatch
- instead of stopping the pipeline, the error message was saved into the Excel outputs as `gbr_shap_importance_error`

This is a good design choice because the model still trains successfully and the run remains reproducible.

---

### 3. Hist Gradient Boosting Regressor (HGBR)

This is also tuned with Bayesian optimization, but the candidate selection explicitly considers both validation performance and overfitting.

Search setup:

- estimator: `HistGradientBoostingRegressor`
- `BayesSearchCV`
- iterations: **20**
- cross-validation folds: **2**
- scoring: **R²**
- `refit=False`
- custom selection based on:
  - `mean_test_score`
  - `cv_gap = mean_train_score - mean_test_score`
  - `selection_score = mean_test_score - overfit_penalty * max(cv_gap, 0)`

Best logged parameters:

- l2 regularization: **1.0889**
- learning rate: **0.06966**
- loss: **squared_error**
- max depth: **4**
- max iter: **596**
- max leaf nodes: **22**
- min samples leaf: **59**
- n iter no change: **18**
- tol: **2.94e-04**
- validation fraction: **0.20246**

Results:

- train R²: **0.8689**
- test R²: **0.8436**
- test MAE: **1.0403**
- test RMSE: **1.4372**
- Bayesian search runtime: about **28.11 sec**
- full model stage runtime: about **32.37 sec**

Interpretation artifacts generated:

- permutation importance table
- SHAP importance table
- possibly built-in feature importance when available from the fitted regressor

Interpretation:

- this is the **best classical tree-based model** in the run
- it clearly outperforms Ridge, GBR, and SVR on test R² and error metrics

---

### 4. Support Vector Regressor (SVR)

Configuration:

- kernel: `rbf`
- `C=100`
- `epsilon=0.001`

Results:

- train R²: **0.8119**
- test R²: **0.7973**
- test MAE: **1.1059**
- test MSE: **2.6768**
- test RMSE: **1.6361**
- runtime: about **10.31 sec**

Interpretation:

- performs well and beats GBR
- still weaker than HGBR and ANN in this run

---

### 5. Artificial Neural Network (ANN)

The ANN uses a separate preprocessing path from the sklearn pipeline.

#### ANN preprocessing

- one-hot encodes categorical columns with `pd.get_dummies`
- aligns train and test columns
- scales inputs with `StandardScaler`
- scales target values with another `StandardScaler`

#### ANN architecture

- input layer with `input_dim=12`
- dense layer: 64 units, ReLU
- dropout: 0.2
- dense layer: 32 units, ReLU
- dropout: 0.1
- output layer: 1 unit, linear

#### Training configuration

- epochs: **200**
- batch size: **32**
- validation split: **0.20**
- optimizer: `Adam(learning_rate=0.001)`
- loss: `Huber()`
- callbacks:
  - `EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True)`
  - `ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6)`

Results:

- train R²: **0.8436**
- test R²: **0.8269**
- train MAE: **1.0275**
- test MAE: **1.0757**
- test MSE: **2.2863**
- test RMSE: **1.5121**
- training runtime: about **44.39 sec**

Interpretation:

- this is the **best overall model by test R²** among all logged models
- compared with HGBR, it has slightly lower test R² gap from training than the tree ensemble and very competitive absolute error
- it is also the slowest single model to train

---

## Model ranking from this run

Sorted by **test R²**:

1. **HGBR**: 0.8436
2. **ANN**: 0.8269
3. **SVR**: 0.7973
4. **GBR**: 0.7359
5. **Ridge**: 0.4756

Sorted by **lowest RMSE**:

1. **HGBR**: 1.4372
2. **ANN**: 1.5121
3. **SVR**: 1.6361
4. **GBR**: 1.8676
5. **Ridge**: 2.6316

### Result summary

- **Best overall test R²:** HGBR
- **Best overall RMSE / MAE among the strong models:** HGBR
- **Best neural model:** ANN
- **Fastest model:** Ridge
- **Slowest model:** ANN

Note: ANN is competitive, but based strictly on the logged metrics, **HGBR is the best-performing model in this run** because it has the highest test R² and the lowest RMSE / MAE among the stronger models.


---

## Model performance insights and design decisions

### Why gradient boosting models performed best

The Gradient Boosting Regressor (GBR) and Histogram Gradient Boosting Regressor (HGBR) were among the strongest performers in this pipeline, and a major reason is that both were tuned with Bayesian hyperparameter optimisation.

Unlike models such as Ridge and SVR, which were trained with fixed or manually chosen hyperparameters in this run, GBR and HGBR used `BayesSearchCV`, which:

- efficiently explores the hyperparameter space
- balances exploration and exploitation
- focuses on promising regions based on earlier evaluations
- optimises directly for the chosen evaluation metric (`R²`)

This generally supports:

- better generalisation
- reduced overfitting risk
- stronger test-set performance than untuned baselines

### GBR vs HGBR

#### Standard Gradient Boosting Regressor (GBR)

- uses exact split finding
- evaluates candidate split points more directly
- can be precise, but is slower and may overfit more easily depending on configuration

#### Histogram Gradient Boosting Regressor (HGBR)

- uses histogram-based binning of features
- computes splits on bins rather than raw continuous values

Typical advantages of HGBR:

- faster training
- lower memory usage
- built-in regularisation support
- better scalability on larger datasets

These properties help explain why HGBR emerged as the strongest classical tree-based model in this run.

### Why some features were dropped

The pipeline drops:

```python
DROP_COLS = ["depth", "wellname", "mw_out"]
```

This choice likely reflects two practical modeling concerns.

#### 1. Multicollinearity

Some features may be highly correlated with others. Keeping all of them can:

- reduce model interpretability
- make learned relationships less stable
- distort importance analysis

#### 2. Data leakage and weak generalisation

Some variables can act more like identifiers or target-adjacent signals than genuine predictive inputs. Examples include:

- `wellname` as an identifier
- derived or downstream measurements
- variables that may encode information too closely related to the target

Dropping these features helps reduce unrealistically optimistic performance and improves the chance that the model generalises better to unseen data.

### Additional takeaway

- Bayesian optimisation appears to be a major driver of the strong gradient boosting results.
- HGBR offers a strong balance of accuracy, speed, and robustness.
- Feature selection is part of the model quality strategy, not just preprocessing.

---

## Output files produced

### 1. Excel workbook

The pipeline writes all model results to:

```text
outputs/model_results.xlsx
```

The workbook contains:

- one metrics sheet per model
- a `summary` sheet combining all model metrics
- CV results sheets for tuned models
- extra interpretation tables when available

Logged Excel content:

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

### 2. Plot outputs

Plots are generated into:

```text
outputs/plots/
├── EDA/
└── Metrics/
```

The log reports **26 plot files** generated in total.

#### EDA plot generation

The script delegates EDA plotting to functions imported from `visualizations.py`.

Expected outputs include:

- histograms
- boxplots
- heatmaps
- missing-data style plots
- well log plots for each well

Logged EDA activity:

- EDA plots generated into `outputs/plots/EDA`
- EDA well logs generated for `well_name=1.0`
- EDA well logs generated for `well_name=2.0`

#### Metrics and model plots

Again, these are generated through `visualizations.py`.

Expected outputs include:

- actual vs predicted scatter plots
- residual-style plots
- well log plots with predictions
- ANN training history plots
- GBR feature importance plot
- HGBR feature importance plot when available

### Role of `visualizations.py`

The main training script does **not** keep plotting logic inline. Instead, all plotting is delegated to the following imported functions from `visualizations.py`:

- `generate_all_data_cleaning_visuals(...)`
- `generate_model_visuals(...)`
- `plot_feature_importance(...)`
- `plot_training_history(...)`
- `plot_well_logs(...)`

This separation is useful because it keeps the modeling pipeline easier to maintain, test, and extend.

---

## Timing summary

Approximate logged timings:

- data build including cleaning and imputation: **7.34 sec**
- Ridge: **0.02 sec**
- GBR: **12.57 sec**
- HGBR: **32.37 sec**
- SVR: **10.31 sec**
- ANN: **44.39 sec**
- plot generation: **15.27 sec**
- full pipeline runtime: **122.79 sec**

---

## Key takeaways

1. The preprocessing pipeline is robust and traceable.
   - It handles units rows, type standardization, outlier replacement, imputation, and target smoothing in a clean sequence.

2. The data quality workflow is substantial.
   - Missing values rise from **398** to **2370** after outlier removal, then drop to **0** after imputation.

3. HGBR is the strongest model in the logged run.
   - It achieved the best test R² and lowest RMSE among all trained models.

4. ANN is also very strong.
   - It is a close second and may still be attractive depending on future tuning or deployment preferences.

5. The logging design is good for production-style experimentation.
   - Every stage is timed.
   - Failures in interpretability steps do not crash the full pipeline.
   - Results are exported in a structured format.

6. Visualization code is intentionally modular.
   - All plot generation is handled in `visualizations.py`, which keeps the pipeline script focused on data and modeling.

---

## How to run

From the project root:

```bash
python rop_estimation_pipeline_log.py
```

Expected main outputs after a successful run:

- `outputs/logs/rop_pipeline.log`
- `outputs/model_results.xlsx`
- `outputs/plots/EDA/*`
- `outputs/plots/Metrics/*`

---

## Suggested next improvements

- fix the GBR SHAP additivity issue, possibly by checking the exact transformed feature matrix passed to the explainer or disabling additivity check only after validation
- add a final automated leaderboard table sorted by test R² and RMSE
- save the best trained model artifact to disk
- version the dataset and configuration values used for each experiment
- consider evaluating the pipeline with grouped validation by well to better reflect generalization across wells

