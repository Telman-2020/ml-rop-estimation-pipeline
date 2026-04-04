from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams['font.size'] = 14


def _ensure_output_dir(output_dir: str | Path) -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path



def _save_current_figure(output_dir: str | Path, filename: str, dpi: int = 200) -> Path:
    output_path = _ensure_output_dir(output_dir) / filename
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    return output_path



def plot_histograms(df: pd.DataFrame, features: Sequence[str], title: str, output_dir: str | Path, filename: str) -> Path:
    existing = [col for col in features if col in df.columns]
    if not existing:
        raise ValueError('No valid features were provided for histogram plotting.')

    df[existing].hist(bins=50, alpha=0.8, figsize=(14, 10))
    plt.suptitle(title)
    return _save_current_figure(output_dir, filename)



def plot_boxplot(df: pd.DataFrame, x: str, y: str, title: str, output_dir: str | Path, filename: str) -> Path:
    if x not in df.columns or y not in df.columns:
        raise ValueError(f"Columns '{x}' and/or '{y}' were not found in the dataframe.")

    plt.figure(figsize=(8, 6))
    sns.boxplot(x=x, y=y, data=df)
    plt.title(title)
    plt.ylabel(y)
    plt.xlabel(x)
    plt.grid(axis='y')
    plt.grid(axis='x')
    return _save_current_figure(output_dir, filename)



def plot_correlation_heatmap(df: pd.DataFrame, columns: Sequence[str], title: str, output_dir: str | Path, filename: str) -> Path:
    existing = [col for col in columns if col in df.columns]
    if len(existing) < 2:
        raise ValueError('At least two valid columns are required for a correlation heatmap.')

    plt.figure(figsize=(12, 9))
    sns.heatmap(df[existing].corr(), vmin=-1, vmax=1, linewidths=0.05, annot=True)
    plt.title(title)
    return _save_current_figure(output_dir, filename)



def plot_well_logs(
    df: pd.DataFrame,
    well_name,
    output_dir: str | Path,
    filename: str,
    depth_col: str = 'depth',
    logs: Optional[Sequence[str]] = None,
    colors: Optional[Sequence[str]] = None,
    figsize: tuple[int, int] = (22, 12),
) -> Path:
    df_well = df[df['wellname'] == well_name]
    if df_well.empty:
        raise ValueError(f'No data found for wellname={well_name}')

    if depth_col not in df.columns:
        raise ValueError(f"'{depth_col}' column was not found in the dataframe.")

    if logs is None:
        logs = [col for col in df.columns if col not in ['wellname', depth_col]]
    logs = [col for col in logs if col in df.columns]
    if not logs:
        raise ValueError('No valid logs were available for plotting.')

    if colors is None:
        colors = ['red', 'orange', 'green', 'cyan', 'blue', 'violet', 'purple', 'pink', 'gray', 'black']
    if len(colors) < len(logs):
        colors = list(colors) * (len(logs) // len(colors) + 1)

    plt.figure(figsize=figsize)
    for i, log in enumerate(logs):
        plt.subplot(1, len(logs), i + 1)
        plt.plot(df_well[log], df_well[depth_col], color=colors[i])
        plt.title(log)
        plt.gca().invert_yaxis()
        plt.xlabel(log)
        plt.ylabel(depth_col if i == 0 else '')

    return _save_current_figure(output_dir, filename)

def plot_actual_vs_predicted_scatter(
    y_test,
    y_pred,
    output_dir,
    filename,
    model_name="Model",
    x_label="Actual ROP",
    y_label="Predicted ROP",
    xlim=None,
    ylim=None,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / filename

    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, y_pred, s=10, alpha=0.15)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(model_name)

    # --- Correlation coefficient ---
    r = np.corrcoef(y_test, y_pred)[0, 1]

    # Add annotation (top-right corner)
    plt.text(
        0.95, 0.95,
        f"r = {r:.3f}",
        transform=plt.gca().transAxes,
        ha="right",
        va="top",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7)
    )

    # --- Y = X reference line ---
    line_min = min(float(np.min(y_test)), float(np.min(y_pred)))
    line_max = max(float(np.max(y_test)), float(np.max(y_pred)))
    plt.plot([line_min, line_max], [line_min, line_max], "r-", linewidth=1, label="Y = X")
    plt.legend(loc="upper left", fontsize=10)

    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    return output_path

def plot_actual_vs_predicted_rop(
    df: pd.DataFrame,
    y_test: pd.Series,
    y_pred,
    well_name,
    output_dir: str | Path,
    filename: str,
    well_col: str = 'wellname',
    depth_col: str = 'depth',
    target_col: str = 'rop',
) -> Path:
    df_well = df[df[well_col] == well_name].copy()
    if df_well.empty:
        raise ValueError(f'No data found for {well_col}={well_name}')

    df_test_well = df.loc[y_test.index].copy()
    df_test_well = df_test_well[df_test_well[well_col] == well_name].copy()

    y_pred_series = pd.Series(y_pred, index=y_test.index)
    df_test_well['predicted_rop'] = y_pred_series.loc[df_test_well.index]

    plt.figure(figsize=(5, 24))
    plt.plot(df_well[target_col], df_well[depth_col], color='blue', label='Actual ROP')
    plt.plot(df_test_well[target_col], df_test_well[depth_col], 'o', color='black', markersize=2, label='Actual test')
    plt.plot(df_test_well['predicted_rop'], df_test_well[depth_col], 'o', color='red', markersize=2, label='Predicted test')
    plt.title(f'{target_col} - Well {well_name}')
    plt.xlabel(target_col)
    plt.ylabel(depth_col)
    plt.gca().invert_yaxis()
    plt.legend()
    plt.grid()
    return _save_current_figure(output_dir, filename)



def plot_logs_with_prediction(
    df: pd.DataFrame,
    well_name,
    y_test: pd.Series,
    y_pred,
    output_dir: str | Path,
    filename: str,
    logs: Optional[Sequence[str]] = None,
    well_col: str = 'wellname',
    depth_col: str = 'depth',
    target_col: str = 'rop',
    exclude_cols: Optional[Sequence[str]] = None,
    figsize: tuple[int, int] = (12, 10),
) -> Path:
    for col in [well_col, depth_col, target_col]:
        if col not in df.columns:
            raise ValueError(f"'{col}' not found in dataframe.")

    df_well = df[df[well_col] == well_name].copy()
    if df_well.empty:
        raise ValueError(f'No data found for {well_col}={well_name}')

    df_test = df.loc[y_test.index].copy()
    df_test = df_test[df_test[well_col] == well_name].copy()

    y_pred_series = pd.Series(y_pred, index=y_test.index)
    df_test['Actual'] = y_test.loc[df_test.index]
    df_test['Predicted'] = y_pred_series.loc[df_test.index]

    if logs is None:
        if exclude_cols is None:
            exclude_cols = [well_col, depth_col, target_col]
        logs = [col for col in df.columns if col not in exclude_cols]
    logs = [col for col in logs if col in df.columns]
    if not logs:
        raise ValueError('No valid logs to plot.')

    base_colors = ['red', 'orange', 'green', 'cyan', 'blue', 'violet', 'purple', 'pink', 'gray', 'black']
    colors = (base_colors * (len(logs) // len(base_colors) + 1))[:len(logs)]

    n_plots = len(logs) + 1
    plt.figure(figsize=figsize)

    for i, log in enumerate(logs):
        plt.subplot(1, n_plots, i + 1)
        plt.plot(df_well[log], df_well[depth_col], color=colors[i])
        plt.title(log)
        plt.gca().invert_yaxis()
        plt.ylabel(depth_col if i == 0 else '')

    plt.subplot(1, n_plots, n_plots)
    plt.plot(df_well[target_col], df_well[depth_col], color='blue', label='Actual full')
    plt.plot(df_test['Actual'], df_test[depth_col], 'o', color='black', markersize=0.5, label='Actual test')
    plt.plot(df_test['Predicted'], df_test[depth_col], 'o', color='red', markersize=0.5, label='Predicted')
    plt.title('Actual vs Prediction')
    plt.gca().invert_yaxis()
    plt.legend()

    return _save_current_figure(output_dir, filename)


def plot_training_history(history, output_dir: str | Path, filename_prefix: str) -> List[Path]:
    output_files: List[Path] = []
    history_dict = history.history

    has_mae = 'mae' in history_dict and 'val_mae' in history_dict
    has_loss = 'loss' in history_dict and 'val_loss' in history_dict

    if not (has_mae or has_loss):
        return output_files

    # Count how many subplots we need
    n_plots = int(has_mae) + int(has_loss)

    fig, axes = plt.subplots(n_plots, 1, figsize=(6, 4 * n_plots), sharex=True)

    # If only one subplot, make axes iterable
    if n_plots == 1:
        axes = [axes]

    idx = 0

    # MAE plot
    if has_mae:
        ax = axes[idx]
        ax.plot(history_dict['mae'])
        ax.plot(history_dict['val_mae'])
        ax.set_title('Training and Validation Learning Curve (MAE)')
        ax.set_ylabel('Mean Absolute Error')
        ax.legend(['train', 'validation'], loc='upper right')
        ax.grid()
        idx += 1

    # Loss plot
    if has_loss:
        ax = axes[idx]
        ax.plot(history_dict['loss'])
        ax.plot(history_dict['val_loss'])
        ax.set_title('Training and Validation Learning Curve (Loss)')
        ax.set_ylabel('Loss')
        ax.set_xlabel('Epoch')
        ax.legend(['train', 'validation'], loc='upper right')
        ax.grid()

    # Ensure bottom plot has x-label
    axes[-1].set_xlabel('Epoch')

    plt.tight_layout()

    # Save single combined figure
    output_files.append(_save_current_figure(output_dir, f'{filename_prefix}_training_history.png'))

    return output_files


def plot_feature_importance(importance_df: pd.DataFrame, title: str, output_dir: str | Path, filename: str, top_n: int = 20) -> Path:
    df = importance_df.copy()

    # Auto-fix common column names
    if 'importance_mean' in df.columns and 'importance' not in df.columns:
        df = df.rename(columns={'importance_mean': 'importance'})

    required = {'feature', 'importance'}
    if not required.issubset(df.columns):
        raise ValueError(f'importance_df must contain columns: {required}')

    top = df.sort_values('importance', ascending=False).head(top_n).iloc[::-1]

    plt.figure(figsize=(10, 8))
    plt.barh(top['feature'], top['importance'])
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.title(title)

    return _save_current_figure(output_dir, filename)


def generate_all_data_cleaning_visuals(
    raw_df: pd.DataFrame,
    clean_df: pd.DataFrame,
    imputed_df: pd.DataFrame,
    histogram_features: Sequence[str],
    output_dir: str | Path,
) -> Dict[str, Path]:
    outputs: Dict[str, Path] = {}
    outputs['raw_histograms'] = plot_histograms(raw_df, histogram_features, 'Before Cleaning', output_dir, '01_before_cleaning_histograms.png')
    outputs['clean_histograms'] = plot_histograms(clean_df, histogram_features, 'After Outlier Removal', output_dir, '02_after_outlier_removal_histograms.png')
    outputs['imputed_histograms'] = plot_histograms(imputed_df, histogram_features, 'After Imputation', output_dir, '03_after_imputation_histograms.png')
    outputs['corr_heatmap'] = plot_correlation_heatmap(imputed_df, imputed_df.select_dtypes(include=['number']).columns.tolist(), 'Correlation Heatmap', output_dir, '04_correlation_heatmap.png')
    outputs['rop_boxplot'] = plot_boxplot(imputed_df, 'bitsize', 'rop', 'ROP by Bit Size', output_dir, '05_boxplot_rop_by_bitsize.png')
    if 'torque' in raw_df.columns:
        outputs['torque_boxplot'] = plot_boxplot(raw_df, 'bitsize', 'torque', 'Torque by Bit Size', output_dir, '06_boxplot_torque_by_bitsize.png')
    if 'rpm' in raw_df.columns:
        outputs['rpm_boxplot'] = plot_boxplot(raw_df, 'bitsize', 'rpm', 'RPM by Bit Size', output_dir, '07_boxplot_rpm_by_bitsize.png')
    return outputs



def generate_model_visuals(
    df_ml: pd.DataFrame,
    y_test: pd.Series,
    predictions: Mapping[str, Sequence[float]],
    output_dir: str | Path,
    well_name=2,
) -> Dict[str, Path]:
    outputs: Dict[str, Path] = {}
    for model_name, y_pred in predictions.items():
        safe_name = model_name.lower().replace(' ', '_')
        outputs[f'{safe_name}_scatter'] = plot_actual_vs_predicted_scatter(
            y_test=y_test,
            y_pred=y_pred,
            model_name=model_name,
            output_dir=output_dir,
            filename=f'{safe_name}_actual_vs_predicted.png',
            xlim=(-1, 15),
            ylim=(-1, 15),
        )
        outputs[f'{safe_name}_well_overlay'] = plot_actual_vs_predicted_rop(
            df=df_ml,
            y_test=y_test,
            y_pred=y_pred,
            well_name=well_name,
            output_dir=output_dir,
            filename=f'{safe_name}_well_{well_name}_overlay.png',
        )
        outputs[f'{safe_name}_well_logs'] = plot_logs_with_prediction(
            df=df_ml,
            well_name=well_name,
            y_test=y_test,
            y_pred=y_pred,
            logs=['rpm', 'wob', 'spp', 'torque'],
            output_dir=output_dir,
            filename=f'{safe_name}_well_{well_name}_logs_with_prediction.png',
        )
    return outputs
