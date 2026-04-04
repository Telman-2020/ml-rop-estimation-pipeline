from __future__ import annotations

"""Refactored ROP estimation pipeline.

This script keeps the original modelling approach but organizes the workflow into
clear sections, validates the data flow between steps, moves all visualization
logic into a separate module, and exports model metrics to an Excel workbook.

Enhanced with:
- console logging for VS Code progress visibility
- file logging
- section checkpoints with timing
"""

# ============================================================
# 1. Imports
# ============================================================
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple
from contextlib import contextmanager

import logging
import re
import time

import numpy as np
import pandas as pd
import shap
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.svm import SVR
from skopt import BayesSearchCV
from skopt.space import Categorical, Integer, Real
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.layers import Dense, Dropout, Input
from keras.losses import Huber
from keras.models import Sequential
from keras.optimizers import Adam
from sklearn.linear_model import Ridge
from visualizations import (
    generate_all_data_cleaning_visuals,
    generate_model_visuals,
    plot_feature_importance,
    plot_training_history,
    plot_well_logs,
)

# ============================================================
# 2. Paths and configuration
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "DrillingParameters.csv"
DEFAULT_PLOTS_DIR = PROJECT_ROOT / "outputs" / "plots"
DEFAULT_EDA_DIR = DEFAULT_PLOTS_DIR / "EDA"
DEFAULT_METRICS_DIR = DEFAULT_PLOTS_DIR / "Metrics"
DEFAULT_RESULTS_XLSX = PROJECT_ROOT / "outputs" / "model_results.xlsx"
DEFAULT_LOG_DIR = PROJECT_ROOT / "outputs" / "logs"
DEFAULT_LOG_FILE = DEFAULT_LOG_DIR / "rop_pipeline.log"

TARGET_COL = "rop"
EXCLUDE_COLS = ["wellname", "bitsize"]
HISTOGRAM_FEATURES = ["wob", "rpm", "torque", "flowrate", "spp", "rop", "mw_in", "mw_out", "bitsize"]
DROP_COLS = ["depth", "wellname", "mw_out"]
CATEGORICAL_FEATURES = ["bitsize"]
STRATIFY_COL = "bitsize"

LOGGER_NAME = "rop_pipeline"


# ============================================================
# 3. Logging utilities
# ============================================================
def ensure_directory(path: Path | str) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def setup_logger(log_file: Path | str = DEFAULT_LOG_FILE) -> logging.Logger:
    log_file = Path(log_file)
    ensure_directory(log_file.parent)

    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(logging.INFO)

    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    logger.propagate = False

    logger.info("=" * 90)
    logger.info("Logger initialized")
    logger.info("Log file: %s", log_file)
    logger.info("=" * 90)
    return logger


logger = setup_logger()


@contextmanager
def log_section(section_name: str):
    start = time.time()
    logger.info("START | %s", section_name)
    try:
        yield
    except Exception as exc:
        elapsed = time.time() - start
        logger.exception("FAILED | %s | elapsed=%.2f sec | error=%s", section_name, elapsed, exc)
        raise
    else:
        elapsed = time.time() - start
        logger.info("END   | %s | elapsed=%.2f sec", section_name, elapsed)


def log_df_info(df: pd.DataFrame, df_name: str) -> None:
    missing_total = int(df.isna().sum().sum())
    logger.info(
        "%s | shape=%s | columns=%d | total_missing=%d",
        df_name,
        df.shape,
        df.shape[1],
        missing_total,
    )


# ============================================================
# 4. Data classes
# ============================================================
@dataclass
class DataArtifacts:
    raw_df: pd.DataFrame
    cleaned_df: pd.DataFrame
    imputed_df: pd.DataFrame
    units_dict: Dict[str, str]


@dataclass
class SplitArtifacts:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    preprocessor: ColumnTransformer


@dataclass
class ModelRunResult:
    model_name: str
    model: Any
    y_pred: np.ndarray
    metrics: Dict[str, Any]
    history: Any = None
    cv_results: Optional[pd.DataFrame] = None
    extra_tables: Optional[Dict[str, pd.DataFrame]] = None


# ============================================================
# 5. Validation helpers
# ============================================================
def validate_required_columns(df: pd.DataFrame, required_columns: Sequence[str], df_name: str) -> None:
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"{df_name} is missing required columns: {missing}")


def validate_prediction_alignment(y_true: pd.Series, y_pred: Sequence[float], model_name: str) -> None:
    if len(y_true) != len(y_pred):
        raise ValueError(
            f"Prediction length mismatch for {model_name}: "
            f"len(y_true)={len(y_true)} but len(y_pred)={len(y_pred)}"
        )


# ============================================================
# 6. Original helper functions retained with logging
# ============================================================
def extract_and_remove_units(df, threshold=0.6):
    logger.info("Checking first row for units...")

    def is_unit(value):
        if pd.isna(value):
            return False
        value = str(value).strip()
        pattern = r"^\(?[a-zA-Z°/%]+(?:/[a-zA-Z]+)?\)?$"
        return re.match(pattern, value) is not None

    first_row = df.iloc[0]
    unit_flags = first_row.apply(is_unit)

    if unit_flags.mean() > threshold:
        units_dict = {col: str(first_row[col]).strip() for col in df.columns if is_unit(first_row[col])}
        df = df.iloc[1:].reset_index(drop=True)
        logger.info("Units row detected and removed. Extracted %d unit entries.", len(units_dict))
        return df, units_dict

    logger.info("No units row detected.")
    return df, {}


def replace_outliers_with_nan(df, factor=1.5):
    with log_section("Outlier replacement with NaN"):
        df_clean = df.copy()
        numeric_cols = df_clean.select_dtypes(include=["number"]).columns

        total_replaced = 0
        for col in numeric_cols:
            before_missing = int(df_clean[col].isna().sum())

            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1

            lower = Q1 - factor * IQR
            upper = Q3 + factor * IQR

            mask = ~df_clean[col].between(lower, upper)
            df_clean.loc[mask, col] = np.nan

            after_missing = int(df_clean[col].isna().sum())
            replaced_here = after_missing - before_missing
            total_replaced += max(replaced_here, 0)

            logger.info(
                "Outlier cleaning | column=%s | added_nan=%d | lower=%.5f | upper=%.5f",
                col,
                max(replaced_here, 0),
                lower if pd.notna(lower) else np.nan,
                upper if pd.notna(upper) else np.nan,
            )

        logger.info("Total outlier values replaced with NaN: %d", total_replaced)
        log_df_info(df_clean, "cleaned_df_after_outlier_replacement")
        return df_clean


def ml_impute_with_progress(df):
    with log_section("ML imputation pipeline"):
        df_copy = df.copy()

        cat_cols = [col for col in ["wellname", "bitsize"] if col in df_copy.columns]
        num_cols = df_copy.select_dtypes(include=["float32", "float64", "int32", "int64"]).columns.tolist()

        logger.info("Imputation input | rows=%d | numeric_cols=%d | categorical_cols=%d", len(df_copy), len(num_cols), len(cat_cols))
        logger.info("Missing values before imputation: %d", int(df_copy.isna().sum().sum()))

        with log_section("Imputation step 1/3 - Encode categorical variables"):
            if cat_cols:
                encoder = OrdinalEncoder()
                df_copy[cat_cols] = encoder.fit_transform(df_copy[cat_cols])
            else:
                encoder = None
                logger.info("No categorical columns found for encoding.")

        with log_section("Imputation step 2/3 - IterativeImputer fit_transform"):
            imputer = IterativeImputer(
                estimator=RandomForestRegressor(n_estimators=5, n_jobs=-1, random_state=42),
                random_state=42
            )
            df_copy[num_cols] = imputer.fit_transform(df_copy[num_cols])

        with log_section("Imputation step 3/3 - Restore categorical variables"):
            if cat_cols and encoder is not None:
                df_copy[cat_cols] = encoder.inverse_transform(df_copy[cat_cols])
                for col in cat_cols:
                    df_copy[col] = df_copy[col].astype("category")

        logger.info("Missing values after imputation: %d", int(df_copy.isna().sum().sum()))
        log_df_info(df_copy, "imputed_df")
        return df_copy


def prepare_ml_data(
    df,
    target="rop",
    categorical_features=None,
    stratify_col=None,   # kept for compatibility
    drop_cols=None,
    test_size=0.3,
    random_state=42,
    well_col="wellname",
    depth_col="depth",
    block_size= 30, # as the consecutive depth intervals (one meter apart) within each 
                    # well are likely to be correlated, we use block-wise splitting 
                    # to avoid data leakage between train and test sets
    keep_split_cols_in_features= False
):
    with log_section("Prepare ML data and interval-block train/test split within wells"):
        df = df.copy()

        if target not in df.columns:
            raise ValueError(f"'{target}' column not found in dataframe.")
        if well_col not in df.columns:
            raise ValueError(f"'{well_col}' column not found in dataframe.")
        if depth_col not in df.columns:
            raise ValueError(f"'{depth_col}' column not found in dataframe.")

        logger.info("Initial ML dataframe shape: %s", df.shape)

        drop_cols = drop_cols or []

        # columns required for splitting must never be dropped before split
        protected_cols = {target, well_col, depth_col}
        requested_drop_cols = [col for col in drop_cols if col in df.columns]
        invalid_pre_split_drop = [col for col in requested_drop_cols if col in protected_cols]

        if invalid_pre_split_drop:
            logger.warning(
                "Ignoring protected columns in drop_cols before split: %s",
                invalid_pre_split_drop
            )

        # only drop columns that are not needed for split/target
        pre_split_drop_cols = [col for col in requested_drop_cols if col not in protected_cols]
        if pre_split_drop_cols:
            logger.info("Dropping columns before split: %s", pre_split_drop_cols)
            df = df.drop(columns=pre_split_drop_cols)

        rng = np.random.RandomState(random_state)

        df["_block_id"] = -1
        df["_is_test"] = False

        global_block_id = 0

        for well_name, well_df in df.groupby(well_col, sort=False):
            well_df_sorted = well_df.sort_values(depth_col).copy()
            n_rows = len(well_df_sorted)

            if n_rows == 0:
                continue

            n_blocks = int(np.ceil(n_rows / block_size))
            local_block_ids = np.repeat(np.arange(n_blocks), block_size)[:n_rows]

            well_df_sorted["_block_id_local"] = local_block_ids
            unique_blocks = well_df_sorted["_block_id_local"].unique()

            n_test_blocks = max(1, int(np.ceil(len(unique_blocks) * test_size)))
            test_blocks = rng.choice(unique_blocks, size=n_test_blocks, replace=False)

            well_df_sorted["_is_test"] = well_df_sorted["_block_id_local"].isin(test_blocks)
            well_df_sorted["_block_id"] = well_df_sorted["_block_id_local"] + global_block_id

            df.loc[well_df_sorted.index, "_block_id"] = well_df_sorted["_block_id"].values
            df.loc[well_df_sorted.index, "_is_test"] = well_df_sorted["_is_test"].values

            logger.info(
                "Well '%s': %d rows | %d blocks | %d test blocks",
                well_name, n_rows, n_blocks, n_test_blocks
            )

            global_block_id += n_blocks

        train_mask = ~df["_is_test"]
        test_mask = df["_is_test"]

        if train_mask.sum() == 0:
            raise ValueError("Train split is empty. Reduce test_size or block_size.")
        if test_mask.sum() == 0:
            raise ValueError("Test split is empty. Increase test_size or reduce block_size.")

        # build feature frame after split
        feature_drop_cols = [target, "_block_id", "_is_test"]

        if not keep_split_cols_in_features:
            feature_drop_cols.extend([well_col, depth_col])

        X = df.drop(columns=[col for col in feature_drop_cols if col in df.columns])
        y = df[target]

        X_train = X.loc[train_mask].copy()
        X_test = X.loc[test_mask].copy()
        y_train = y.loc[train_mask].copy()
        y_test = y.loc[test_mask].copy()

        if categorical_features is None:
            categorical_features = [col for col in ["wellname", "bitsize"] if col in X.columns]
        else:
            categorical_features = [col for col in categorical_features if col in X.columns]

        numeric_features = [col for col in X.columns if col not in categorical_features]

        logger.info("X_train shape: %s | X_test shape: %s", X_train.shape, X_test.shape)
        logger.info("y_train shape: %s | y_test shape: %s", y_train.shape, y_test.shape)
        logger.info("Categorical features: %s", categorical_features)
        logger.info("Numeric features count: %d", len(numeric_features))

        categorical_transformer = Pipeline([
            ("imputer_cat", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ])

        numeric_transformer = Pipeline([
            ("imputer_num", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ("categoricals", categorical_transformer, categorical_features),
                ("numericals", numeric_transformer, numeric_features)
            ],
            remainder="drop",
            sparse_threshold=0
        )

        return X_train, X_test, y_train, y_test, preprocessor
    

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
    with log_section("Train Gradient Boosting Regressor with Bayesian search"):
        pipe_gbr = Pipeline([
            ("preprocessor", preprocessor),
            ("model", GradientBoostingRegressor(
                random_state=random_state,
                validation_fraction=0.1,
                n_iter_no_change=10,
                tol=1e-4
            ))
        ])

        search_spaces = {
            "model__n_estimators": Integer(50, 180),
            "model__learning_rate": Real(0.02, 0.08, prior="log-uniform"),
            "model__max_depth": Integer(2, 3),
            "model__min_samples_split": Integer(10, 40),
            "model__min_samples_leaf": Integer(5, 20),
            "model__subsample": Real(0.6, 0.85),
            "model__max_features": Categorical(["sqrt", "log2"]),
            "model__loss": Categorical(["huber", "absolute_error"]),
            "model__tol": Real(1e-5, 1e-3, prior="log-uniform"),
            "model__n_iter_no_change": Integer(5, 15),
            "model__validation_fraction": Real(0.1, 0.2)
        }

        bayes_search = BayesSearchCV(
            estimator=pipe_gbr,
            search_spaces=search_spaces,
            n_iter=n_iter,
            scoring="r2",
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
            "best_params": bayes_search.best_params_,
            "train_r2": train_r2,
            "test_r2": test_r2,
            "mae": mean_absolute_error(y_test, y_test_pred),
            "rmse": np.sqrt(mean_squared_error(y_test, y_test_pred)),
            "overfit_gap": train_r2 - test_r2
        }

        logger.info("GBR best params: %s", bayes_search.best_params_)
        logger.info("GBR metrics | train_r2=%.4f | test_r2=%.4f | mae=%.4f | rmse=%.4f",
                    results["train_r2"], results["test_r2"], results["mae"], results["rmse"])

        cv_results = pd.DataFrame(bayes_search.cv_results_).sort_values(
            ["rank_test_score", "mean_train_score"],
            ascending=[True, True]
        )

        return best_model, results, cv_results


def bayesian_search_hgbr(
    X_train,
    X_test,
    y_train,
    y_test,
    preprocessor,
    n_iter=10,
    cv=None,
    random_state=42,
    overfit_penalty=0.5,
    max_allowed_gap=0.08
):
    with log_section("Train Hist Gradient Boosting Regressor with Bayesian search"):
        if cv is None:
            cv = KFold(n_splits=3, shuffle=False)

        pipe_hgbr = Pipeline([
            ("preprocessor", preprocessor),
            ("model", HistGradientBoostingRegressor(
                random_state=random_state,
                early_stopping=True,
                validation_fraction=0.15,
                n_iter_no_change=10,
                tol=1e-4
            ))
        ])

        search_spaces = {
            "model__learning_rate": Real(0.02, 0.10, prior="log-uniform"),
            "model__max_iter": Integer(300, 800),
            "model__max_depth": Integer(2, 4),
            "model__min_samples_leaf": Integer(20, 80),
            "model__max_leaf_nodes": Integer(8, 31),
            "model__l2_regularization": Real(1e-2, 20.0, prior="log-uniform"),
            "model__loss": Categorical(["squared_error"]),
            "model__tol": Real(1e-5, 1e-3, prior="log-uniform"),
            "model__n_iter_no_change": Integer(10, 30),
            "model__validation_fraction": Real(0.12, 0.25)
        }

        bayes_search = BayesSearchCV(
            estimator=pipe_hgbr,
            search_spaces=search_spaces,
            n_iter=n_iter,
            scoring="r2",
            cv=cv,
            n_jobs=-1,
            verbose=1,
            random_state=random_state,
            return_train_score=True,
            refit=False
        )

        bayes_search.fit(X_train, y_train)
        cv_results = pd.DataFrame(bayes_search.cv_results_).copy()
        cv_results["cv_gap"] = cv_results["mean_train_score"] - cv_results["mean_test_score"]
        cv_results["selection_score"] = cv_results["mean_test_score"] - overfit_penalty * np.maximum(cv_results["cv_gap"], 0)

        good_candidates = cv_results[cv_results["cv_gap"] <= max_allowed_gap].copy()
        if len(good_candidates) > 0:
            best_idx = good_candidates.sort_values(["mean_test_score", "cv_gap"], ascending=[False, True]).index[0]
        else:
            best_idx = cv_results.sort_values(["selection_score", "cv_gap"], ascending=[False, True]).index[0]

        best_params = cv_results.loc[best_idx, "params"]

        best_model = Pipeline([
            ("preprocessor", preprocessor),
            ("model", HistGradientBoostingRegressor(
                random_state=random_state,
                early_stopping=True,
                **{k.replace("model__", ""): v for k, v in best_params.items() if k.startswith("model__")}
            ))
        ])

        best_model.fit(X_train, y_train)
        y_train_pred = best_model.predict(X_train)
        y_test_pred = best_model.predict(X_test)

        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)

        results = {
            "best_params": best_params,
            "train_r2": train_r2,
            "test_r2": test_r2,
            "mae": mean_absolute_error(y_test, y_test_pred),
            "rmse": np.sqrt(mean_squared_error(y_test, y_test_pred)),
            "overfit_gap": train_r2 - test_r2,
            "cv_mean_train_score": cv_results.loc[best_idx, "mean_train_score"],
            "cv_mean_test_score": cv_results.loc[best_idx, "mean_test_score"],
            "cv_gap": cv_results.loc[best_idx, "cv_gap"],
            "selection_score": cv_results.loc[best_idx, "selection_score"]
        }

        logger.info("HGBR best params: %s", best_params)
        logger.info("HGBR metrics | train_r2=%.4f | test_r2=%.4f | mae=%.4f | rmse=%.4f",
                    results["train_r2"], results["test_r2"], results["mae"], results["rmse"])

        cv_results = cv_results.sort_values(["selection_score", "mean_test_score", "cv_gap"], ascending=[False, False, True])
        return best_model, results, cv_results


def train_svr_model(preprocessor, X_train, y_train, X_test, y_test, C=100, epsilon=0.001, kernel="rbf"):
    with log_section("Train SVR model"):
        model = make_pipeline(preprocessor, SVR(C=C, epsilon=epsilon, kernel=kernel))
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)

        logger.info("SVR params | C=%s | epsilon=%s | kernel=%s", C, epsilon, kernel)
        logger.info("SVR metrics | train_r2=%.4f | test_r2=%.4f", train_score, test_score)

        return model, y_pred, train_score, test_score


def train_ann(X_train, y_train, X_test, y_test, epochs=200, batch_size=32, validation_split=0.2):
    with log_section("Train ANN model"):
        X_train_proc = pd.get_dummies(X_train, drop_first=False)
        X_test_proc = pd.get_dummies(X_test, drop_first=False)
        X_train_proc, X_test_proc = X_train_proc.align(X_test_proc, join="left", axis=1, fill_value=0)

        x_scaler = StandardScaler()
        X_train_proc = x_scaler.fit_transform(X_train_proc)
        X_test_proc = x_scaler.transform(X_test_proc)

        y_train = np.asarray(y_train).reshape(-1, 1).astype("float32")
        y_test = np.asarray(y_test).reshape(-1, 1).astype("float32")

        y_scaler = StandardScaler()
        y_train_scaled = y_scaler.fit_transform(y_train)

        model = Sequential([
            Input(shape=(X_train_proc.shape[1],)),
            Dense(64, activation="relu"),
            Dropout(0.2),
            Dense(32, activation="relu"),
            Dropout(0.1),
            Dense(1, activation="linear")
        ])

        model.compile(optimizer=Adam(learning_rate=0.001), loss=Huber(), metrics=["mae"])

        callbacks = [
            EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6, verbose=1)
        ]

        logger.info(
            "ANN training started | epochs=%d | batch_size=%d | validation_split=%.2f | input_dim=%d",
            epochs, batch_size, validation_split, X_train_proc.shape[1]
        )

        history = model.fit(
            X_train_proc,
            y_train_scaled,
            validation_split=validation_split,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )

        y_pred_scaled = model.predict(X_test_proc, verbose=0)
        train_pred_scaled = model.predict(X_train_proc, verbose=0)

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

        logger.info("ANN metrics | train_r2=%.4f | test_r2=%.4f | test_mae=%.4f | test_rmse=%.4f",
                    metrics["train_r2"], metrics["test_r2"], metrics["test_mae"], metrics["test_rmse"])

        return model, history, y_pred, metrics, x_scaler, y_scaler


# ============================================================
# 7. Refactored orchestration helpers
# ============================================================
def load_data(csv_path: Path | str) -> tuple[pd.DataFrame, Dict[str, str]]:
    with log_section("Load CSV data"):
        csv_path = Path(csv_path)
        if not csv_path.exists():
            raise FileNotFoundError(f"Input CSV file not found: {csv_path}")

        logger.info("Reading CSV from: %s", csv_path)
        df = pd.read_csv(csv_path)
        log_df_info(df, "raw_loaded_df")

        df, units_dict = extract_and_remove_units(df)
        logger.info("Units extracted count: %d", len(units_dict))
        return df, units_dict


def standardize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    with log_section("Standardize dataframe"):
        df = df.copy()
        df.columns = [col.lower().replace(" ", "_") for col in df.columns]
        validate_required_columns(df, ["wellname", "bitsize", "torque", "rop"], "standardized dataframe")

        df[EXCLUDE_COLS] = df[EXCLUDE_COLS].astype("category")
        num_cols = df.columns.difference(EXCLUDE_COLS)
        df[num_cols] = df[num_cols].apply(pd.to_numeric, errors="coerce").astype("float32")

        affected_rows = int(df["wellname"].isin([2.0]).sum()) if "wellname" in df.columns else 0
        df.loc[df["wellname"].isin([2.0]), "torque"] *= 1000

        logger.info("Standardized columns: %s", list(df.columns))
        logger.info("Torque scaled for wellname=2.0 rows: %d", affected_rows)
        log_df_info(df, "standardized_df")
        return df


def smooth_target_by_well(df: pd.DataFrame, target_col: str = TARGET_COL, group_col: str = "wellname", window: int = 10) -> pd.DataFrame:
    with log_section("Smooth target by well"):
        df = df.copy()
        validate_required_columns(df, [group_col, target_col], "target smoothing input")
        df[target_col] = df.groupby(group_col)[target_col].transform(
            lambda x: x.rolling(window=window, min_periods=1).mean()
        )
        logger.info("Applied rolling mean smoothing | target=%s | group=%s | window=%d", target_col, group_col, window)
        return df


def build_data_artifacts(csv_path: Path | str) -> DataArtifacts:
    with log_section("Build data artifacts"):
        with log_section("Data stage 1 - Load and standardize"):
            raw_df, units_dict = load_data(csv_path)
            raw_df = standardize_dataframe(raw_df)

        with log_section("Data stage 2 - Cleaning before imputation"):
            logger.info("Cleaning BEFORE imputation started")
            cleaned_df = replace_outliers_with_nan(raw_df)
            logger.info("Cleaning BEFORE imputation finished")

        with log_section("Data stage 3 - Imputation"):
            logger.info("Imputation started")
            imputed_df = ml_impute_with_progress(cleaned_df)
            logger.info("Imputation finished")

        with log_section("Data stage 4 - Post-imputation smoothing / cleaning after imputation"):
            logger.info("Cleaning AFTER imputation started")
            imputed_df = smooth_target_by_well(imputed_df)
            logger.info("Cleaning AFTER imputation finished")

        log_df_info(raw_df, "final_raw_df")
        log_df_info(cleaned_df, "final_cleaned_df")
        log_df_info(imputed_df, "final_imputed_df")

        return DataArtifacts(raw_df=raw_df, cleaned_df=cleaned_df, imputed_df=imputed_df, units_dict=units_dict)


def build_split_artifacts(df_ml: pd.DataFrame) -> SplitArtifacts:
    with log_section("Build split artifacts"):
        X_train, X_test, y_train, y_test, preprocessor = prepare_ml_data(
            df_ml,
            target=TARGET_COL,
            drop_cols=DROP_COLS,
            categorical_features=CATEGORICAL_FEATURES,
            stratify_col=STRATIFY_COL,
        )
        return SplitArtifacts(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test, preprocessor=preprocessor)


def calculate_regression_metrics(y_true: pd.Series, y_pred: Sequence[float], prefix: str = "test") -> Dict[str, float]:
    validate_prediction_alignment(y_true, y_pred, prefix)
    return {
        f"{prefix}_mae": mean_absolute_error(y_true, y_pred),
        f"{prefix}_mse": mean_squared_error(y_true, y_pred),
        f"{prefix}_rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        f"{prefix}_r2": r2_score(y_true, y_pred),
    }


def run_ridge_model(split: SplitArtifacts) -> ModelRunResult:
    with log_section("Run Ridge Regression"):
        model = make_pipeline(split.preprocessor, Ridge(alpha=10.0))
        model.fit(split.X_train, split.y_train)
        y_pred = model.predict(split.X_test)
        validate_prediction_alignment(split.y_test, y_pred, "Ridge")

        metrics = {
            "train_r2": r2_score(split.y_train, model.predict(split.X_train)),
            **calculate_regression_metrics(split.y_test, y_pred, prefix="test"),
        }

        logger.info("Ridge metrics: %s", metrics)
        return ModelRunResult(model_name="Ridge Regression", model=model, y_pred=y_pred, metrics=metrics)


def run_gbr_model(split: SplitArtifacts) -> ModelRunResult:
    with log_section("Run Gradient Boosting Regressor"):
        model, search_metrics, cv_results = bayesian_search_gbr(
            split.X_train,
            split.X_test,
            split.y_train,
            split.y_test,
            split.preprocessor,
            n_iter=5,
            cv=2,
            random_state=42
        )

        y_pred = model.predict(split.X_test)
        validate_prediction_alignment(split.y_test, y_pred, "Gradient Boosting Regressor")

        extra_tables: Dict[str, pd.DataFrame] = {}

        preprocessor = model.named_steps["preprocessor"]
        regressor = model.named_steps["model"]
        feature_names = preprocessor.get_feature_names_out()

        if hasattr(regressor, "feature_importances_"):
            built_in_importance_df = pd.DataFrame({
                "feature": feature_names,
                "importance": regressor.feature_importances_,
            }).sort_values("importance", ascending=False)
            extra_tables["built_in_feature_importance"] = built_in_importance_df

        with log_section("GBR permutation importance"):
            perm_result = permutation_importance(
                model,
                split.X_test,
                split.y_test,
                n_repeats=10,
                random_state=42,
                scoring="r2",
                n_jobs=-1,
            )

            perm_importance_df = pd.DataFrame({
                "feature": split.X_test.columns,
                "importance_mean": perm_result.importances_mean,
                "importance_std": perm_result.importances_std,
            }).sort_values("importance_mean", ascending=False)

            extra_tables["permutation_importance"] = perm_importance_df

        with log_section("GBR SHAP importance"):
            try:
                X_test_transformed = preprocessor.transform(split.X_test)
                X_test_transformed_df = pd.DataFrame(
                    X_test_transformed,
                    columns=feature_names,
                    index=split.X_test.index
                )

                explainer = shap.Explainer(regressor, X_test_transformed_df)
                shap_values = explainer(X_test_transformed_df)

                shap_importance_df = pd.DataFrame({
                    "feature": feature_names,
                    "importance": np.abs(shap_values.values).mean(axis=0),
                }).sort_values("importance", ascending=False)

                extra_tables["shap_importance"] = shap_importance_df
            except Exception as exc:  # pragma: no cover
                logger.warning("GBR SHAP failed: %s", exc)
                extra_tables["shap_importance_error"] = pd.DataFrame({"message": [str(exc)]})

        return ModelRunResult(
            model_name="Gradient Boosting Regressor",
            model=model,
            y_pred=y_pred,
            metrics=search_metrics,
            cv_results=cv_results,
            extra_tables=extra_tables,
        )


def run_hgbr_model(split: SplitArtifacts) -> ModelRunResult:
    with log_section("Run Hist Gradient Boosting Regressor"):
        model, search_metrics, cv_results = bayesian_search_hgbr(
            X_train=split.X_train,
            X_test=split.X_test,
            y_train=split.y_train,
            y_test=split.y_test,
            preprocessor=split.preprocessor,
            n_iter=20,
            cv=2,
            random_state=42,
        )
        y_pred = model.predict(split.X_test)
        validate_prediction_alignment(split.y_test, y_pred, "Hist Gradient Boosting Regressor")

        extra_tables: Dict[str, pd.DataFrame] = {}

        preprocessor = model.named_steps["preprocessor"]
        regressor = model.named_steps["model"]
        feature_names = preprocessor.get_feature_names_out()

        with log_section("HGBR permutation importance"):
            perm_result = permutation_importance(
                model,
                split.X_test,
                split.y_test,
                n_repeats=10,
                random_state=42,
                scoring="r2",
                n_jobs=-1,
            )
            perm_importance_df = pd.DataFrame({
                "feature": split.X_test.columns,
                "importance_mean": perm_result.importances_mean,
                "importance_std": perm_result.importances_std,
            }).sort_values("importance_mean", ascending=False)
            extra_tables["permutation_importance"] = perm_importance_df

        if hasattr(regressor, "feature_importances_"):
            built_in_importance_df = pd.DataFrame({
                "feature": feature_names,
                "importance": regressor.feature_importances_,
            }).sort_values("importance", ascending=False)
            extra_tables["built_in_feature_importance"] = built_in_importance_df

        with log_section("HGBR SHAP importance"):
            try:
                X_test_transformed = preprocessor.transform(split.X_test)
                X_test_transformed_df = pd.DataFrame(X_test_transformed, columns=feature_names, index=split.X_test.index)
                explainer = shap.Explainer(regressor)
                shap_values = explainer(X_test_transformed_df)
                shap_importance_df = pd.DataFrame({
                    "feature": feature_names,
                    "importance": np.abs(shap_values.values).mean(axis=0),
                }).sort_values("importance", ascending=False)
                extra_tables["shap_importance"] = shap_importance_df
            except Exception as exc:  # pragma: no cover
                logger.warning("HGBR SHAP failed: %s", exc)
                extra_tables["shap_importance_error"] = pd.DataFrame({"message": [str(exc)]})

        return ModelRunResult(
            model_name="Hist Gradient Boosting Regressor",
            model=model,
            y_pred=y_pred,
            metrics=search_metrics,
            cv_results=cv_results,
            extra_tables=extra_tables,
        )


def run_svr_model(split: SplitArtifacts) -> ModelRunResult:
    with log_section("Run Support Vector Regressor"):
        model, y_pred, train_r2, test_r2 = train_svr_model(
            split.preprocessor,
            split.X_train,
            split.y_train,
            split.X_test,
            split.y_test,
        )
        validate_prediction_alignment(split.y_test, y_pred, "Support Vector Regressor")
        metrics = {
            "train_r2": train_r2,
            "test_r2": test_r2,
            **calculate_regression_metrics(split.y_test, y_pred, prefix="test"),
        }
        logger.info("SVR final metrics: %s", metrics)
        return ModelRunResult(model_name="Support Vector Regressor", model=model, y_pred=y_pred, metrics=metrics)


def run_ann_model(split: SplitArtifacts) -> ModelRunResult:
    with log_section("Run ANN Model"):
        model, history, y_pred, ann_metrics, _, _ = train_ann(
            split.X_train,
            split.y_train,
            split.X_test,
            split.y_test,
            epochs=200,
            batch_size=32,
        )
        validate_prediction_alignment(split.y_test, y_pred, "ANN Model")
        logger.info("ANN final metrics: %s", ann_metrics)
        return ModelRunResult(model_name="ANN Model", model=model, y_pred=y_pred, metrics=ann_metrics, history=history)


def run_all_models(split: SplitArtifacts) -> Dict[str, ModelRunResult]:
    results = {}

    model_runners = {
        'ridge': run_ridge_model,
        'gbr': run_gbr_model,
        'hgbr': run_hgbr_model,
        'svr': run_svr_model,
        'ann': run_ann_model,
    }

    for key, runner in model_runners.items():
        try:
            logger.info("Starting model: %s", key)
            results[key] = runner(split)
            logger.info("Finished model: %s", key)
        except Exception as exc:
            logger.exception("Model failed: %s | error=%s", key, exc)

    logger.info("Models completed successfully: %s", list(results.keys()))
    return results

def save_results_to_excel(results: Mapping[str, ModelRunResult], output_path: Path | str) -> Path:
    output_path = Path(output_path)
    ensure_directory(output_path.parent)

    logger.info("Preparing to save Excel. Models available: %s", list(results.keys()))

    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        summary_rows = []

        for key, result in results.items():
            try:
                logger.info("Writing Excel content for model: %s", key)

                metrics_df = pd.DataFrame([result.metrics])
                logger.info("Metrics shape for %s: %s", key, metrics_df.shape)
                metrics_df.to_excel(writer, sheet_name=f'{key}_metrics', index=False)

                summary_row = {'model_key': key, 'model_name': result.model_name}
                summary_row.update(result.metrics)
                summary_rows.append(summary_row)

                if result.cv_results is not None:
                    logger.info("Writing CV results for %s | shape=%s", key, result.cv_results.shape)
                    result.cv_results.to_excel(writer, sheet_name=f'{key}_cv_results'[:31], index=False)

                if result.extra_tables:
                    for table_name, table_df in result.extra_tables.items():
                        safe_sheet = f'{key}_{table_name}'[:31]
                        logger.info("Writing extra table for %s | sheet=%s | shape=%s", key, safe_sheet, table_df.shape)
                        table_df.to_excel(writer, sheet_name=safe_sheet, index=False)

            except Exception as exc:
                logger.exception("Failed while writing Excel sheets for model %s: %s", key, exc)

        pd.DataFrame(summary_rows).to_excel(writer, sheet_name='summary', index=False)

    logger.info("Excel saved successfully to: %s", output_path)
    return output_path


def generate_visual_outputs(
    data: DataArtifacts,
    split: SplitArtifacts,
    results: Mapping[str, ModelRunResult],
    plots_root_dir: Path | str,
) -> Dict[str, Path]:
    with log_section("Generate plot outputs"):
        plots_root_dir = ensure_directory(plots_root_dir)

        # Create folders on the fly
        eda_dir = ensure_directory(Path(plots_root_dir) / "EDA")
        metrics_dir = ensure_directory(Path(plots_root_dir) / "Metrics")

        logger.info("EDA plots directory: %s", eda_dir)
        logger.info("Metrics plots directory: %s", metrics_dir)

        outputs: Dict[str, Path] = {}

        # ============================================================
        # EDA plots
        # ============================================================
        with log_section("Generate EDA plots"):
            logger.info("Generating cleaning / preprocessing / imputation plots into EDA folder")

            # Includes histograms, boxplots, heatmaps, missing-data style visuals, etc.
            outputs.update(
                generate_all_data_cleaning_visuals(
                    raw_df=data.raw_df,
                    clean_df=data.cleaned_df,
                    imputed_df=data.imputed_df,
                    histogram_features=HISTOGRAM_FEATURES,
                    output_dir=eda_dir,
                )
            )

        with log_section("Generate EDA well logs"):
            if "wellname" in data.imputed_df.columns:
                for well_name in sorted(data.imputed_df["wellname"].dropna().unique()):
                    logger.info("Generating EDA well log plot for well_name=%s", well_name)
                    outputs[f"eda_well_logs_{well_name}"] = plot_well_logs(
                        data.imputed_df,
                        well_name=well_name,
                        output_dir=eda_dir,
                        filename=f"eda_well_{well_name}_logs.png",
                    )

        # ============================================================
        # Metrics / model evaluation plots
        # ============================================================
        with log_section("Generate model metric plots"):
            logger.info("Generating model evaluation plots into Metrics folder")

            model_predictions = {
                result.model_name: result.y_pred
                for result in results.values()
            }

            # Expected to include scatter plots: actual vs predicted,
            # well logs with predictions, residual-type plots, etc.
            outputs.update(
                generate_model_visuals(
                    data.imputed_df,
                    split.y_test,
                    model_predictions,
                    metrics_dir,
                    well_name=2
                )
            )

        ann_result = results.get("ann")
        if ann_result and ann_result.history is not None:
            with log_section("Generate ANN training history plots"):
                history_files = plot_training_history(
                    ann_result.history,
                    metrics_dir,
                    "ann_training"
                )
                for idx, path in enumerate(history_files, start=1):
                    outputs[f"metrics_ann_history_{idx}"] = path

        gbr_result = results.get("gbr")
        if gbr_result and gbr_result.extra_tables:
            built_in = gbr_result.extra_tables.get("built_in_feature_importance")
            if built_in is not None and not built_in.empty:
                with log_section("Generate GBR feature importance plot"):
                    outputs["metrics_gbr_feature_importance"] = plot_feature_importance(
                        built_in,
                        "Gradient Boosting Feature Importance",
                        metrics_dir,
                        "gbr_feature_importance.png",
                    )

        hgbr_result = results.get("hgbr")
        if hgbr_result and hgbr_result.extra_tables:
            built_in = hgbr_result.extra_tables.get("built_in_feature_importance")
            if built_in is not None and not built_in.empty:
                with log_section("Generate HGBR feature importance plot"):
                    outputs["metrics_hgbr_feature_importance"] = plot_feature_importance(
                        built_in,
                        "Hist Gradient Boosting Feature Importance",
                        metrics_dir,
                        "hgbr_feature_importance.png",
                    )

        logger.info("Generated %d plot files in total.", len(outputs))
        return outputs

# ============================================================
# 8. Main entry point
# ============================================================
def main(    
    csv_path: Path | str = DEFAULT_DATA_PATH,
    plots_dir: Path | str = DEFAULT_PLOTS_DIR,
    results_excel_path: Path | str = DEFAULT_RESULTS_XLSX,
    log_file: Path | str = DEFAULT_LOG_FILE,

        ) -> Dict[str, Any]:
    global logger
    logger = setup_logger(log_file)

    with log_section("MAIN PIPELINE"):
        csv_path = Path(csv_path)
        plots_dir = ensure_directory(plots_dir)
        results_excel_path = Path(results_excel_path)
        ensure_directory(results_excel_path.parent)

        data = build_data_artifacts(csv_path)
        split = build_split_artifacts(data.imputed_df)
        results = run_all_models(split)

        logger.info("Results collected for models: %s", list(results.keys()))

        excel_path = save_results_to_excel(results, results_excel_path)
        plot_paths = generate_visual_outputs(data, split, results, plots_dir)

        return {
            'data': data,
            'split': split,
            'results': results,
            'excel_path': excel_path,
            'visual_paths': plot_paths,
        }
    

if __name__ == "__main__":
    total_start = time.time()
    try:
        artifacts = main()
        total_elapsed = time.time() - total_start
        logger.info("Saved Excel results to: %s", artifacts["excel_path"])
        logger.info("Saved %d visual outputs to: %s", len(artifacts["visual_paths"]), DEFAULT_PLOTS_DIR)
        logger.info("TOTAL SCRIPT TIME: %.2f sec", total_elapsed)

        print(f"Saved Excel results to: {artifacts['excel_path']}")
        print(f"Saved {len(artifacts['visual_paths'])} visual outputs to: {DEFAULT_PLOTS_DIR}")
        print(f"Log file saved to: {DEFAULT_LOG_FILE}")
        print(f"Total script time: {total_elapsed:.2f} sec")
    except Exception as exc:
        logger.exception("Pipeline terminated with error: %s", exc)
        raise