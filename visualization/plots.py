from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _import_matplotlib():
    """使用 Agg 后端生成静态图片。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    return plt, mdates


def _flatten_timestamps(target_timestamps: np.ndarray, max_points: int | None = None) -> pd.DatetimeIndex:
    timestamps = pd.to_datetime(np.asarray(target_timestamps).reshape(-1))
    if max_points is not None:
        timestamps = timestamps[:max_points]
    return pd.DatetimeIndex(timestamps)


def _format_time_axis(ax, mdates, rotation: int = 30) -> None:
    locator = mdates.AutoDateLocator(minticks=4, maxticks=8)
    formatter = mdates.ConciseDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    ax.tick_params(axis="x", rotation=rotation)


def plot_prediction_curve(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_timestamps: np.ndarray,
    output_dir: str | Path,
    window_name: str,
    model_name: str,
    output_window_hours: int,
    max_points: int = 500,
) -> Path:
    """使用真实 timestamp 绘制预测曲线图。"""
    plt, mdates = _import_matplotlib()
    output_path = Path(output_dir) / "prediction_curve.png"
    frame = pd.DataFrame(
        {
            "timestamp": _flatten_timestamps(target_timestamps),
            "y_true": np.asarray(y_true).reshape(-1),
            "y_pred": np.asarray(y_pred).reshape(-1),
        }
    )
    grouped = frame.groupby("timestamp", as_index=False)[["y_true", "y_pred"]].mean().sort_values("timestamp")
    grouped = grouped.head(max_points)
    time_flat = pd.DatetimeIndex(grouped["timestamp"])
    true_flat = grouped["y_true"].to_numpy()
    pred_flat = grouped["y_pred"].to_numpy()

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(time_flat, true_flat, label="y_true", linewidth=1.4)
    ax.plot(time_flat, pred_flat, label="y_pred", linewidth=1.2)
    ax.set_title(f"{window_name} | {model_name} | prediction target: {output_window_hours}h")
    ax.set_xlabel("timestamp")
    ax.set_ylabel("PM2.5")
    ax.legend()
    _format_time_axis(ax, mdates, rotation=30)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def plot_stage_errors(metrics: dict[str, Any], output_dir: str | Path) -> Path:
    """绘制三个预测阶段的误差对比。"""
    plt, _ = _import_matplotlib()
    output_path = Path(output_dir) / "stage_errors.png"
    stage_names = list(metrics["stages"].keys())
    rmse = [metrics["stages"][name]["RMSE"] for name in stage_names]
    mae = [metrics["stages"][name]["MAE"] for name in stage_names]
    x = np.arange(len(stage_names))
    width = 0.36

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width / 2, rmse, width, label="RMSE")
    ax.bar(x + width / 2, mae, width, label="MAE")
    ax.set_xticks(x, stage_names)
    ax.set_ylabel("error")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def plot_horizon_errors(metrics: dict[str, Any], output_dir: str | Path) -> Path:
    """绘制 1 到 72 小时逐 horizon 误差曲线。"""
    plt, _ = _import_matplotlib()
    output_path = Path(output_dir) / "horizon_errors.png"
    horizons = [row["horizon"] for row in metrics["horizon"]]
    rmse = [row["RMSE"] for row in metrics["horizon"]]
    mae = [row["MAE"] for row in metrics["horizon"]]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(horizons, rmse, label="RMSE", linewidth=1.4)
    ax.plot(horizons, mae, label="MAE", linewidth=1.4)
    ax.set_xlabel("forecast horizon")
    ax.set_ylabel("error")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def plot_attention_weights(attention_weights: np.ndarray, output_dir: str | Path) -> Path:
    """绘制 Attention-LSTM 平均注意力权重。"""
    plt, _ = _import_matplotlib()
    output_path = Path(output_dir) / "attention_weights.png"
    weights = np.asarray(attention_weights, dtype=float)
    if weights.ndim != 2:
        raise ValueError(f"Attention 权重必须为二维数组，实际为 {weights.shape}")

    mean_weights = weights.mean(axis=0)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(np.arange(1, len(mean_weights) + 1), mean_weights, linewidth=1.4)
    ax.set_xlabel("input time step")
    ax.set_ylabel("mean attention weight")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def plot_loss_curve(history: list[dict[str, Any]], output_dir: str | Path) -> Path:
    """绘制训练损失和验证损失曲线。"""
    plt, _ = _import_matplotlib()
    output_path = Path(output_dir) / "loss_curve.png"
    epochs = [row["epoch"] for row in history]
    train_loss = [row["train_loss"] for row in history]
    validation_loss = [row["validation_loss"] for row in history]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(epochs, train_loss, label="train_loss", linewidth=1.4)
    ax.plot(epochs, validation_loss, label="validation_loss", linewidth=1.4)
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def plot_peak_case(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_timestamps: np.ndarray,
    sample_id: int,
    output_dir: str | Path,
    window_name: str,
    model_name: str,
    output_window_hours: int,
) -> Path:
    """绘制真实峰值样本的 72 小时预测曲线，x 轴使用真实 timestamp。"""
    plt, mdates = _import_matplotlib()
    output_path = Path(output_dir) / "peak_case_top1.png"
    sample_timestamps = pd.to_datetime(target_timestamps[sample_id])

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(sample_timestamps, y_true[sample_id], label="y_true", linewidth=1.6)
    ax.plot(sample_timestamps, y_pred[sample_id], label="y_pred", linewidth=1.4)
    ax.set_title(f"{window_name} | {model_name} | peak case | prediction target: {output_window_hours}h")
    ax.set_xlabel("timestamp")
    ax.set_ylabel("PM2.5")
    ax.legend()
    _format_time_axis(ax, mdates, rotation=45)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def create_model_plots(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metrics: dict[str, Any],
    target_timestamps: np.ndarray,
    plots_dir: str | Path,
    window_name: str,
    model_name: str,
    output_window_hours: int,
    attention_weights: np.ndarray | None = None,
) -> None:
    """为单个模型生成标准 plots 目录内容。"""
    plots_path = Path(plots_dir)
    plots_path.mkdir(parents=True, exist_ok=True)
    plot_prediction_curve(y_true, y_pred, target_timestamps, plots_path, window_name, model_name, output_window_hours)
    plot_stage_errors(metrics, plots_path)
    plot_horizon_errors(metrics, plots_path)
    if attention_weights is not None:
        plot_attention_weights(attention_weights, plots_path)
