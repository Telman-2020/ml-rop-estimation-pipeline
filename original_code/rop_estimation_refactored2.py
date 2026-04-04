# -*- coding: utf-8 -*-
"""ROP_Estimation
"""

# !pip install scikit-optimize
# !pip -q install pyswarm

# import necessary modules and libraries
import pandas as pd
import numpy as np
from scipy import interpolate
import re
import random
import shap
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns
import time
from sklearn.model_selection import RandomizedSearchCV, train_test_split, KFold
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression, Ridge, Lasso
# from imblearn.pipeline import make_pipeline
from sklearn.metrics import max_error, r2_score, explained_variance_score, mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from skopt import BayesSearchCV
from pyswarm import pso
from sklearn.linear_model import 
from sklearn.inspection import permutation_importance
from sklearn.svm import SVR

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Activation, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau,ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import Huber


plt.rcParams['font.size'] = 20

path = r"C:\CodeProjects\test_python_codes\DrillingParameters.csv"
df = pd.read_csv (path)

def extract_and_remove_units(df, threshold=0.6):

    def is_unit(value):
        if pd.isna(value):
            return False
        value = str(value).strip()
        pattern = r"^\(?[a-zA-Z°/%]+(?:/[a-zA-Z]+)?\)?$"
        return re.match(pattern, value) is not None

    first_row = df.iloc[0]
    unit_flags = first_row.apply(is_unit)

    if unit_flags.mean() > threshold:
        units_dict = {
            col: str(first_row[col]).strip()
            for col in df.columns
            if is_unit(first_row[col])
        }
        df = df.iloc[1:].reset_index(drop=True)
        return df, units_dict

    return df, {}

df, units_dict =  extract_and_remove_units(df)

"""#  1) **Data Cleaning and Visualizations**"""

df.columns = [i.lower().replace(' ', '_') for i in df.columns]

exclude_cols = ['wellname', 'bitsize']

df[exclude_cols] = df[exclude_cols].astype('category')
num_cols = df.columns.difference(exclude_cols)
df[num_cols] = df[num_cols].apply(pd.to_numeric, errors='coerce').astype('float32')

df.loc[df['wellname'].isin([2.0]), 'torque'] *= 1000

features = ['wob', 'rpm', 'torque', 'flowrate', 'spp',
                      'rop', 'mw_in', 'mw_out', 'bitsize']

df[features].hist(bins=50, alpha=0.8, figsize=(13,10))
plt.show()

def replace_outliers_with_nan(df, factor=1.5):
    df_clean = df.copy()

    numeric_cols = df_clean.select_dtypes(include=['number']).columns

    for col in numeric_cols:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1

        lower = Q1 - factor * IQR
        upper = Q3 + factor * IQR

        mask = ~df_clean[col].between(lower, upper)
        df_clean.loc[mask, col] = np.nan

    return df_clean

df_clean = replace_outliers_with_nan(df)

df_clean[features].hist(bins=50, alpha=0.8, figsize=(13,10))
plt.show()

def ml_impute_with_progress(df):
    start_time = time.time()

    df_copy = df.copy()

    cat_cols = ['wellname', 'bitsize']
    num_cols = df_copy.select_dtypes(include=['float32', 'float64']).columns

    print(f"Starting imputation...")
    print(f"Rows: {len(df_copy)}, Numeric cols: {len(num_cols)}")

    # Encode categoricals
    print("\n[1/3] Encoding categorical variables...")
    t0 = time.time()

    encoder = OrdinalEncoder()
    df_copy[cat_cols] = encoder.fit_transform(df_copy[cat_cols])

    print(f"Done in {time.time() - t0:.2f}s")

    # Imputation
    print("\n[2/3] Running IterativeImputer...")
    t0 = time.time()

    imputer = IterativeImputer(
        estimator=RandomForestRegressor(n_estimators=20, n_jobs=-1)
    )

    df_copy[num_cols] = imputer.fit_transform(df_copy[num_cols])

    print(f"Imputation done in {time.time() - t0:.2f}s")

    # Decode categoricals
    print("\n[3/3] Restoring categorical variables...")
    t0 = time.time()

    df_copy[cat_cols] = encoder.inverse_transform(df_copy[cat_cols])
    df_copy[cat_cols] = df_copy[cat_cols].astype('category')

    print(f"Done in {time.time() - t0:.2f}s")

    total_time = time.time() - start_time
    print(f"\n✅ Total time: {total_time:.2f} seconds")

    return df_copy

df_ml = ml_impute_with_progress(df_clean)

df_ml[features].hist(bins=50, alpha=0.8, figsize=(13,10))
plt.show()

df_ml['rop'] = (df_ml.groupby('wellname')['rop'].transform(lambda x: x.rolling(window=10, min_periods=1).mean()))

rcParams['figure.figsize'] = 12,9
ax = sns.heatmap(df_ml.iloc[:,1:10].corr(),vmin=-1, vmax=1, linewidths=.05,annot=True)

ax = sns.boxplot(x ="bitsize", y="rop", data = df_ml)
plt.title('Box Plot')
plt.ylabel('rop')
plt.xlabel('bitsize')
plt.grid(axis='y')
plt.grid(axis='x')
plt.show()

ax = sns.boxplot(x ="bitsize", y="torque", data = df)
plt.title('Box Plot')
plt.ylabel('torque')
plt.xlabel('bitsize')
plt.grid(axis='y')
plt.grid(axis='x')
plt.show()

ax = sns.boxplot(x ="bitsize", y="rpm", data = df)
plt.title('Box Plot')
plt.ylabel('rpm')
plt.xlabel('bitsize')
plt.grid(axis='y')
plt.grid(axis='x')
plt.show()

def plot_well_logs(df, well_name, depth_col='depth', logs=None, colors=None, figsize=(22, 12)):
    """
    Plot multiple logs vs depth for a given well.

    Parameters:
    - df : pandas DataFrame
    - well_name : value in 'wellname' column to filter
    - depth_col : name of depth column
    - logs : list of log columns (default = all except first + well/depth)
    - colors : list of colors (optional)
    - figsize : tuple
    """

    # Filter once (performance improvement)
    df_well = df[df['wellname'] == well_name]

    if df_well.empty:
        raise ValueError(f"No data found for wellname={well_name}")

    # Default logs selection
    if logs is None:
        logs = [col for col in df.columns if col not in ['wellname', depth_col]]

    # Default colors
    if colors is None:
        colors = ['red', 'orange', 'green', 'cyan', 'blue',
                  'violet', 'purple', 'pink', 'gray', 'black']

    # Extend colors if needed
    if len(colors) < len(logs):
        colors = (colors * (len(logs) // len(colors) + 1))[:len(logs)]

    # Plot
    plt.figure(figsize=figsize)

    for i, log in enumerate(logs):
        plt.subplot(1, len(logs), i + 1)
        plt.plot(df_well[log], df_well[depth_col], color=colors[i])
        plt.title(log)
        plt.gca().invert_yaxis()
        plt.xlabel(log)

        # Only show depth label on first plot
        if i == 0:
            plt.ylabel(depth_col)
        else:
            plt.ylabel("")

    plt.tight_layout()
    plt.show()

well2_logs = plot_well_logs(df_ml, well_name= 2.0)

well1_logs = plot_well_logs(df_ml, well_name= 1.0)

sns.pairplot(df_ml[df_ml.columns.tolist()[1:]], corner=True, diag_kind= 'kde',)
plt.show()

"""## 1.1) Split data and feature scaling"""

def prepare_ml_data(
    df,
    target='rop',
    categorical_features=None,
    stratify_col='bitsize',
    drop_cols=None,
    test_size=0.3,
    random_state=42
):
    """
    Split dataframe into train/test sets and build a preprocessing pipeline,
    with optional column dropping.
    """

    df = df.copy()

    if target not in df.columns:
        raise ValueError(f"'{target}' column not found in dataframe.")

    # ✅ Drop unwanted columns first
    if drop_cols is not None:
        drop_cols = [col for col in drop_cols if col in df.columns]
        df = df.drop(columns=drop_cols)

    # Default categorical columns
    if categorical_features is None:
        categorical_features = [col for col in ['wellname', 'bitsize'] if col in df.columns]

    # Split X and y
    X = df.drop(columns=[target])
    y = df[target]

    # Stratification
    stratify_values = None
    if stratify_col is not None:
        if stratify_col not in df.columns:
            raise ValueError(f"stratify_col='{stratify_col}' not found in dataframe.")
        stratify_values = df[stratify_col]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=stratify_values,
        random_state=random_state
    )

    # Keep only categorical columns that exist
    categorical_features = [col for col in categorical_features if col in X.columns]

    # Numeric features = everything except categoricals
    numeric_features = [col for col in X.columns if col not in categorical_features]

    # Pipelines
    categorical_transformer = Pipeline([
        ('imputer_cat', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    numeric_transformer = Pipeline([
        ('imputer_num', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('categoricals', categorical_transformer, categorical_features),
            ('numericals', numeric_transformer, numeric_features)
        ],
        remainder='drop',
        sparse_threshold=0
    )

    return X_train, X_test, y_train, y_test, preprocessor

X_train, X_test, y_train, y_test, preprocessor = prepare_ml_data(
    df_ml,
    target='rop',
    drop_cols=['depth', 'wellname', 'mw_out']
    ,
    categorical_features=['bitsize'],
    stratify_col='bitsize'
)

preprocessor

"""# 2) **Traditinal Machine Learning Models**

## 2.1) Linear Regression Algorithm
"""

# Build pipeline
model_lr = make_pipeline(
    preprocessor,
    Ridge(alpha=10.0))

# Fit
model_lr.fit(X_train, y_train)

# Predict
y_pred_lr = model_lr.predict(X_test)

# Evaluate
train_score = r2_score(y_train, model_lr.predict(X_train))
test_score = r2_score(y_test, y_pred_lr)
print(f"Linear Regression - Train R2: {train_score:.4f}, Test R2: {test_score:.4f}")
rcParams['figure.figsize'] = 5,5

plt.scatter(y_test, y_pred_lr,s=10, c='blue', alpha=0.15,)
plt.xlabel('Actual_ROP')
plt.ylabel('Predicted_ROP')
plt.plot([0,100],[0,100],"r-", linewidth=1, label="Y = X")
plt.legend(loc="upper left", fontsize=14)
plt.xlim(-1,20)
plt.ylim(-1,20)
plt.show()

"""## 2.2) Gradient Boosting Regression Algorithm

### 2.2.1) *Baysian Search for GBR*
"""

def bayesian_search_gbr(
    X_train,
    X_test,
    y_train,
    y_test,
    preprocessor,
    n_iter=20,
    cv=3,
    random_state=42
):
    """
    Bayesian optimization for GradientBoostingRegressor with a search space
    designed to reduce overfitting.
    """

    pipe_gbr = Pipeline([
        ('preprocessor', preprocessor),
        ('model', GradientBoostingRegressor(
            random_state=random_state,
            validation_fraction=0.1,
            n_iter_no_change=10,
            tol=1e-4
        ))
    ])

    # More regularized / conservative search space
    search_spaces = {
        'model__n_estimators': Integer(50, 180),
        'model__learning_rate': Real(0.02, 0.08, prior='log-uniform'),
        'model__max_depth': Integer(2, 3),
        'model__min_samples_split': Integer(10, 40),
        'model__min_samples_leaf': Integer(5, 20),
        'model__subsample': Real(0.6, 0.85),
        'model__max_features': Categorical(['sqrt', 'log2']),
        'model__loss': Categorical(['huber', 'absolute_error']),
        'model__tol': Real(1e-5, 1e-3, prior='log-uniform'),
        'model__n_iter_no_change': Integer(5, 15),
        'model__validation_fraction': Real(0.1, 0.2)
    }

    bayes_search = BayesSearchCV(
        estimator=pipe_gbr,
        search_spaces=search_spaces,
        n_iter=n_iter,
        scoring='r2',
        cv=cv,
        n_jobs=-1,
        verbose=1,
        random_state=random_state,
        return_train_score=True
    )

    bayes_search.fit(X_train, y_train)

    best_model = bayes_search.best_estimator_
    y_train_pred = best_model.predict(X_train)
    y_test_pred = best_model.predict(X_test)

    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    results = {
        'best_params': bayes_search.best_params_,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'mae': mean_absolute_error(y_test, y_test_pred),
        'rmse': np.sqrt(mean_squared_error(y_test, y_test_pred)),
        'overfit_gap': train_r2 - test_r2
    }

    cv_results = pd.DataFrame(bayes_search.cv_results_).sort_values(
        ['rank_test_score', 'mean_train_score'],
        ascending=[True, True]
    )

    return best_model, results, cv_results

best_model_gbr, results_gbr, cv_results_gbr = bayesian_search_gbr(
    X_train, X_test, y_train, y_test, preprocessor,
    n_iter=20,
    cv=3,
    random_state=42
)


# Metrics
print(f"\nTrain R2: {results_gbr['train_r2']:.4f}")
print(f"Test R2 : {results_gbr['test_r2']:.4f}")
print(f"MAE     : {results_gbr['mae']:.4f}")
print(f"RMSE    : {results_gbr['rmse']:.4f}")

def bayesian_search_hgbr(
    X_train,
    X_test,
    y_train,
    y_test,
    preprocessor,
    n_iter=30,
    cv=None,
    random_state=42,
    overfit_penalty=0.5,
    max_allowed_gap=0.08
):
    """
    Bayesian optimization for HistGradientBoostingRegressor with
    overfitting-aware model selection.

    Main anti-overfitting improvements:
    - narrower / safer hyperparameter search space
    - stronger regularization options
    - larger min_samples_leaf
    - lower max_depth / max_leaf_nodes
    - custom selection criterion that penalizes train-validation gap

    Parameters
    ----------
    X_train, X_test, y_train, y_test : data
    preprocessor : sklearn transformer
    n_iter : int
        Number of Bayesian search iterations.
    cv : int or CV splitter or None
        If None, uses KFold(n_splits=3, shuffle=False).
        Better: pass a custom splitter that respects well/depth ordering.
    random_state : int
    overfit_penalty : float
        Penalty multiplier for (mean_train_score - mean_test_score).
        Higher values prefer more stable models.
    max_allowed_gap : float
        Preferred upper bound for CV train-test gap.

    Returns
    -------
    best_model, results, cv_results
    """

    if cv is None:
        cv = KFold(n_splits=3, shuffle=False)

    pipe_hgbr = Pipeline([
        ('preprocessor', preprocessor),
        ('model', HistGradientBoostingRegressor(
            random_state=random_state,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=20,
            tol=1e-4
        ))
    ])

    # Safer search space to reduce overfitting
    search_spaces = {
        'model__learning_rate': Real(0.02, 0.10, prior='log-uniform'),
        'model__max_iter': Integer(300, 800),
        'model__max_depth': Integer(2, 4),
        'model__min_samples_leaf': Integer(20, 80),
        'model__max_leaf_nodes': Integer(8, 31),
        'model__l2_regularization': Real(1e-2, 20.0, prior='log-uniform'),
        'model__loss': Categorical(['squared_error']),
        'model__tol': Real(1e-5, 1e-3, prior='log-uniform'),
        'model__n_iter_no_change': Integer(10, 30),
        'model__validation_fraction': Real(0.12, 0.25)
    }

    bayes_search = BayesSearchCV(
        estimator=pipe_hgbr,
        search_spaces=search_spaces,
        n_iter=n_iter,
        scoring='r2',
        cv=cv,
        n_jobs=-1,
        verbose=1,
        random_state=random_state,
        return_train_score=True,
        refit=False   # important: we will choose the best candidate ourselves
    )

    bayes_search.fit(X_train, y_train)

    cv_results = pd.DataFrame(bayes_search.cv_results_).copy()

    # Compute CV overfitting gap
    cv_results['cv_gap'] = cv_results['mean_train_score'] - cv_results['mean_test_score']

    # Penalized objective:
    # prefer high validation R2 but punish large train/validation gap
    cv_results['selection_score'] = (
        cv_results['mean_test_score']
        - overfit_penalty * np.maximum(cv_results['cv_gap'], 0)
    )

    # Prefer candidates whose gap is under threshold; otherwise use penalized score
    good_candidates = cv_results[cv_results['cv_gap'] <= max_allowed_gap].copy()

    if len(good_candidates) > 0:
        best_idx = good_candidates.sort_values(
            ['mean_test_score', 'cv_gap'],
            ascending=[False, True]
        ).index[0]
    else:
        best_idx = cv_results.sort_values(
            ['selection_score', 'cv_gap'],
            ascending=[False, True]
        ).index[0]

    best_params = cv_results.loc[best_idx, 'params']

    # Refit chosen model on full training set
    best_model = Pipeline([
        ('preprocessor', preprocessor),
        ('model', HistGradientBoostingRegressor(
            random_state=random_state,
            early_stopping=True,
            **{
                k.replace('model__', ''): v
                for k, v in best_params.items()
                if k.startswith('model__')
            }
        ))
    ])

    best_model.fit(X_train, y_train)

    y_train_pred = best_model.predict(X_train)
    y_test_pred = best_model.predict(X_test)

    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    results = {
        'best_params': best_params,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'mae': mean_absolute_error(y_test, y_test_pred),
        'rmse': np.sqrt(mean_squared_error(y_test, y_test_pred)),
        'overfit_gap': train_r2 - test_r2,
        'cv_mean_train_score': cv_results.loc[best_idx, 'mean_train_score'],
        'cv_mean_test_score': cv_results.loc[best_idx, 'mean_test_score'],
        'cv_gap': cv_results.loc[best_idx, 'cv_gap'],
        'selection_score': cv_results.loc[best_idx, 'selection_score']
    }

    cv_results = cv_results.sort_values(
        ['selection_score', 'mean_test_score', 'cv_gap'],
        ascending=[False, False, True]
    )

    return best_model, results, cv_results

best_model_hgbr, results_hgbr, cv_results_hgbr = bayesian_search_hgbr(
    X_train=X_train,
    X_test=X_test,
    y_train=y_train,
    y_test=y_test,
    preprocessor=preprocessor,
    n_iter=40,
    cv=3,
    random_state=42
)

# Metrics
print(f"\nTrain R2: {results_hgbr['train_r2']:.4f}")
print(f"Test R2 : {results_hgbr['test_r2']:.4f}")
print(f"MAE     : {results_hgbr['mae']:.4f}")
print(f"RMSE    : {results_hgbr['rmse']:.4f}")
print(f"Overfit Gap: {results_hgbr['overfit_gap']:.4f}")

"""### 2.2.2) Visuals for GBR Model"""

def plot_actual_vs_predicted_rop(df, y_test, y_pred, well_name, well_col='wellname', depth_col='depth', target_col='rop'):
    df_well = df[df[well_col] == well_name].copy()

    df_test_well = df.loc[y_test.index].copy()
    df_test_well = df_test_well[df_test_well[well_col] == well_name].copy()

    y_pred_series = pd.Series(y_pred, index=y_test.index)
    df_test_well['predicted_rop'] = y_pred_series.loc[df_test_well.index]

    plt.figure(figsize=(5, 32))

    plt.plot(df_well[target_col], df_well[depth_col], color='blue', label='Actual ROP')
    plt.plot(df_test_well[target_col], df_test_well[depth_col], 'o', color='black', markersize=2, label='Actual test')
    plt.plot(df_test_well['predicted_rop'], df_test_well[depth_col], 'o', color='red', markersize=2, label='Predicted test')

    plt.title(f'{target_col} - Well {well_name}')
    plt.xlabel(target_col)
    plt.ylabel(depth_col)
    plt.gca().invert_yaxis()
    plt.legend()
    plt.tight_layout()
    plt.grid()
    plt.show()

well2_plot_act_pred_hgbr = plot_actual_vs_predicted_rop(df_ml, y_test, best_model_hgbr.predict(X_test), well_name=2)

well2_plot_act_pred_gbr = plot_actual_vs_predicted_rop(df_ml, y_test, best_model_gbr.predict(X_test), well_name=2)

def plot_logs_with_prediction(
    df,
    well_name,
    y_test,
    y_pred,
    logs=None,   # ✅ NEW PARAMETER
    well_col='wellname',
    depth_col='depth',
    target_col='rop',
    exclude_cols=None,
    figsize=(12, 10)
):
    """
    Plot selected logs + actual vs predicted target for a given well.
    """

    df = df.copy()

    # Basic checks
    for col in [well_col, depth_col, target_col]:
        if col not in df.columns:
            raise ValueError(f"'{col}' not found in dataframe.")

    # Full well data
    df_well = df[df[well_col] == well_name].copy()

    if df_well.empty:
        raise ValueError(f"No data found for {well_col}={well_name}")

    # Test subset aligned by index
    df_test = df.loc[y_test.index].copy()
    df_test = df_test[df_test[well_col] == well_name].copy()

    # Align predictions
    y_pred_series = pd.Series(y_pred, index=y_test.index)
    df_test['Actual'] = y_test.loc[df_test.index]
    df_test['Predicted'] = y_pred_series.loc[df_test.index]

    # ✅ If logs not provided → auto select
    if logs is None:
        if exclude_cols is None:
            exclude_cols = [well_col, depth_col, target_col]
        logs = [col for col in df.columns if col not in exclude_cols]

    # Validate logs
    logs = [col for col in logs if col in df.columns]

    if len(logs) == 0:
        raise ValueError("No valid logs to plot.")

    # Colors
    base_colors = ['red', 'orange', 'green', 'cyan', 'blue',
                   'violet', 'purple', 'pink', 'gray', 'black']
    colors = (base_colors * (len(logs) // len(base_colors) + 1))[:len(logs)]

    n_plots = len(logs) + 1

    plt.figure(figsize=figsize)

    # Plot logs
    for i, log in enumerate(logs):
        plt.subplot(1, n_plots, i + 1)
        plt.plot(df_well[log], df_well[depth_col], color=colors[i])
        plt.title(log)
        plt.gca().invert_yaxis()

        if i == 0:
            plt.ylabel(depth_col)
        else:
            plt.ylabel("")

    # Plot actual vs prediction
    plt.subplot(1, n_plots, n_plots)

    plt.plot(df_well[target_col], df_well[depth_col],
             color='blue', label='Actual full')

    plt.plot(df_test['Actual'], df_test[depth_col],
             'o', color='black', markersize=0.5, label='Actual test')

    plt.plot(df_test['Predicted'], df_test[depth_col],
             'o', color='red', markersize=0.5, label='Predicted')

    plt.title('Actual vs Prediction')
    plt.gca().invert_yaxis()
    plt.legend()

    plt.tight_layout()
    plt.show()

well2_logs_hgbr = plot_logs_with_prediction(
                                            df= df_ml,
                                            well_name=2,
                                            y_test=y_test,
                                            y_pred= best_model_hgbr.predict(X_test),
                                            logs=['rpm', 'wob', 'spp','torque']
                                        )

rcParams['figure.figsize'] = 6,6

plt.scatter(y_test, best_model_gbr.predict(X_test), s=10, c='blue', alpha=0.15,)
plt.xlabel('Actual_ROP')
plt.ylabel('Predicted_ROP')
plt.title('Gradient Boosting Regressor')
plt.plot([0,100],[0,100],"r-", linewidth=1, label="Y = X")
plt.legend(loc="upper left", fontsize=14)
plt.xlim(-1,15)
plt.ylim(-1,15)
plt.show()

rcParams['figure.figsize'] = 6,6

plt.scatter(y_test, best_model_hgbr.predict(X_test), s=10, c='blue', alpha=0.15,)
plt.xlabel('Actual_ROP')
plt.ylabel('Predicted_ROP')
plt.title('Hist Gradient Boosting Regressor')
plt.plot([0,100],[0,100],"r-", linewidth=1, label="Y = X")
plt.legend(loc="upper left", fontsize=14)
plt.xlim(-1,15)
plt.ylim(-1,15)
plt.show()

"""### 2.2.3) Feature Importance"""

# =========================================================

# 1) GET TRANSFORMED FEATURE NAMES
# =========================================================
preprocessor = best_model_hgbr.named_steps["preprocessor"]
model = best_model_hgbr.named_steps["model"]
feature_names = preprocessor.get_feature_names_out()
print("Number of transformed features:", len(feature_names))
# =========================================================
# 2) PERMUTATION IMPORTANCE
# =========================================================
perm_result = permutation_importance(
    best_model_hgbr,
    X_test,
    y_test,
    n_repeats=10,
    random_state=42,
    scoring="r2",   # change if needed
    n_jobs=-1
)
original_feature_names = X_test.columns

perm_importance_df = pd.DataFrame({
    "feature": original_feature_names,
    "importance_mean": perm_result.importances_mean,
    "importance_std": perm_result.importances_std
}).sort_values("importance_mean", ascending=False)

top_n = 20
perm_top = perm_importance_df.head(top_n).iloc[::-1]

plt.figure(figsize=(10, 8))
plt.barh(perm_top["feature"], perm_top["importance_mean"], xerr=perm_top["importance_std"])
plt.xlabel("Permutation Importance (mean decrease in score)")
plt.ylabel("Feature")
plt.title(f"Top {top_n} Permutation Importances")
plt.tight_layout()
plt.show()

print(perm_importance_df.head(20))
# =========================================================
# 3) BUILT-IN FEATURE IMPORTANCE (ONLY IF SUPPORTED)
# =========================================================

if hasattr(model, "feature_importances_"):
    fi_df = pd.DataFrame({
        "feature": feature_names,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False)

    fi_top = fi_df.head(top_n).iloc[::-1]

    plt.figure(figsize=(10, 8))
    plt.barh(fi_top["feature"], fi_top["importance"])
    plt.xlabel("Model Feature Importance")
    plt.ylabel("Feature")
    plt.title(f"Top {top_n} Built-in Feature Importances")
    plt.tight_layout()
    plt.show()

    print(fi_df.head(20))
else:
    print("This model does not expose built-in feature_importances_.")
    print("Use permutation importance or SHAP importance instead.")

# =========================================================
# 4) SHAP GLOBAL FEATURE IMPORTANCE (BEST ALTERNATIVE)
# =========================================================
# Transform X_test using the preprocessor only
X_test_transformed = preprocessor.transform(X_test)

# Build DataFrame for readability
X_test_transformed_df = pd.DataFrame(
    X_test_transformed,
    columns=feature_names,
    index=X_test.index
)

# SHAP explainer for tree-based model
explainer = shap.Explainer(model)
shap_values = explainer(X_test_transformed_df)

# Beeswarm summary plot
shap.plots.beeswarm(shap_values, max_display=20)

# Bar plot of mean absolute SHAP values
shap.plots.bar(shap_values, max_display=20)

"""## 2.4) Support Vector Machine (SVR) Model"""

def train_svr_model(preprocessor, X_train, y_train, X_test, y_test,
                    C=100, epsilon=0.001, kernel='rbf'):
    """
    Train and evaluate an SVR model inside a pipeline.

    Parameters:
        preprocessor: sklearn transformer (e.g. ColumnTransformer)
        X_train, y_train: training data
        X_test, y_test: test data
        C, epsilon, kernel: SVR hyperparameters

    Returns:
        model: trained pipeline
        y_pred: predictions on test set
        train_score: R2 score on training set
        test_score: R2 score on test set
    """

    # Create pipeline (avoid naming conflict)
    model = make_pipeline(
        preprocessor,
        SVR(C=C, epsilon=epsilon, kernel=kernel)
    )

    # Fit model
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    # Scores
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)

    print(f'R2 on train set: {train_score:.2f}')
    print(f'R2 on test set: {test_score:.2f}')

    return model, y_pred, train_score, test_score

model_svr, y_pred_svr, train_r2, test_r2 = train_svr_model(
    preprocessor,
    X_train, y_train,
    X_test, y_test
)

rcParams['figure.figsize'] = 6,6

plt.scatter(y_test, y_pred_svr, s=10, c='blue', alpha=0.15,)
plt.xlabel('Actual_ROP')
plt.ylabel('Predicted_ROP')
plt.title('Support Vector Regressor')
plt.plot([0,100],[0,100],"r-", linewidth=1, label="Y = X")
plt.legend(loc="upper left", fontsize=14)
plt.xlim(-1,15)
plt.ylim(-1,15)
plt.show()

well2_logs_svr = plot_logs_with_prediction(
    df= df_ml,
    well_name=2,
    y_test=y_test,
    y_pred= y_pred_svr,
    logs=['rpm', 'wob', 'spp','torque']
)

"""# 3) **Neural Network Models**"""

def train_ann(X_train, y_train, X_test, y_test,
              epochs=200, batch_size=32, validation_split=0.2):

    # Encode categorical features
    X_train_proc = pd.get_dummies(X_train, drop_first=False)
    X_test_proc = pd.get_dummies(X_test, drop_first=False)

    # Align columns
    X_train_proc, X_test_proc = X_train_proc.align(
        X_test_proc, join='left', axis=1, fill_value=0
    )

    # Scale X
    x_scaler = StandardScaler()
    X_train_proc = x_scaler.fit_transform(X_train_proc)
    X_test_proc = x_scaler.transform(X_test_proc)

    # Convert y to numpy and scale it too
    y_train = np.asarray(y_train).reshape(-1, 1).astype("float32")
    y_test = np.asarray(y_test).reshape(-1, 1).astype("float32")

    y_scaler = StandardScaler()
    y_train_scaled = y_scaler.fit_transform(y_train)
    y_test_scaled = y_scaler.transform(y_test)

    # Build model
    model = Sequential([
        Input(shape=(X_train_proc.shape[1],)),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dropout(0.1),
        Dense(1, activation='linear')
    ])

    # Compile
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss=Huber(),
        metrics=['mae']
    )

    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        )
    ]

    # Fit
    history = model.fit(
        X_train_proc,
        y_train_scaled,
        validation_split=validation_split,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )

    # Predict in scaled space
    y_pred_scaled = model.predict(X_test_proc, verbose=0)
    train_pred_scaled = model.predict(X_train_proc, verbose=0)

    # Convert back to original scale
    y_pred = y_scaler.inverse_transform(y_pred_scaled).flatten()
    train_pred = y_scaler.inverse_transform(train_pred_scaled).flatten()

    y_train_orig = y_train.flatten()
    y_test_orig = y_test.flatten()

    metrics = {
        "train_mae": mean_absolute_error(y_train_orig, train_pred),
        "train_r2": r2_score(y_train_orig, train_pred),
        "test_mae": mean_absolute_error(y_test_orig, y_pred),
        "test_mse": mean_squared_error(y_test_orig, y_pred),
        "test_rmse": np.sqrt(mean_squared_error(y_test_orig, y_pred)),
        "test_r2": r2_score(y_test_orig, y_pred),
    }

    return model, history, y_pred, metrics, x_scaler, y_scaler

model_ann, history_ann, y_pred_ann, ann_metrics, x_scaler, y_scaler = train_ann(
    X_train, y_train,
    X_test, y_test,
    epochs=200,
    batch_size=32
)

rcParams['figure.figsize'] = (6, 4)

plt.figure()
plt.plot(history_ann.history['mae'])
plt.plot(history_ann.history['val_mae'])
plt.title('Model MAE')
plt.ylabel('MAE')
plt.xlabel('Epoch')
plt.legend(['train', 'validation'], loc='upper right')
plt.grid()
plt.show()

plt.figure()
plt.plot(history_ann.history['loss'])
plt.plot(history_ann.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['train', 'validation'], loc='upper right')
plt.grid()
plt.show()

rcParams['figure.figsize'] = 6,6

plt.scatter(y_test, y_pred_ann, s=10,c='blue', alpha=0.15)
plt.xlabel('Actual_ROP')
plt.title('ANN Model')
plt.ylabel('Predicted_ROP_ANN')
plt.plot([0,100],[0,100],"r-", linewidth=2, label="Y = X")
plt.legend(loc="upper left", fontsize=14)
plt.xlim(-1,20)
plt.ylim(-1,20)
plt.show()

