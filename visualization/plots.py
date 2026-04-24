from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def _import_matplotlib():
    """使用 Agg 后端生成静态图片，避免服务器或无 GUI 环境报错。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def plot_prediction_curve(y_true: np.ndarray, y_pred: np.ndarray, output_dir: str | Path, max_points: int = 500) -> Path:
    """绘制真实值与预测值曲线。

    只取前 max_points 个点是为了让论文和前端预览图保持可读性。
    """
    plt = _import_matplotlib()
    output_path = Path(output_dir) / "prediction_curve.png"
    true_flat = y_true.reshape(-1)[:max_points]
    pred_flat = y_pred.reshape(-1)[:max_points]

    plt.figure(figsize=(12, 5))
    plt.plot(true_flat, label="y_true", linewidth=1.4)
    plt.plot(pred_flat, label="y_pred", linewidth=1.2)
    plt.xlabel("time point")
    plt.ylabel("PM2.5")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()
    return output_path


def plot_stage_errors(metrics: dict[str, Any], output_dir: str | Path) -> Path:
    """绘制 1-24、25-48、49-72 三个阶段的误差对比。"""
    plt = _import_matplotlib()
    output_path = Path(output_dir) / "stage_errors.png"
    stage_names = list(metrics["stages"].keys())
    rmse = [metrics["stages"][name]["RMSE"] for name in stage_names]
    mae = [metrics["stages"][name]["MAE"] for name in stage_names]
    x = np.arange(len(stage_names))
    width = 0.36

    plt.figure(figsize=(8, 5))
    plt.bar(x - width / 2, rmse, width, label="RMSE")
    plt.bar(x + width / 2, mae, width, label="MAE")
    plt.xticks(x, stage_names)
    plt.ylabel("error")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()
    return output_path


def plot_horizon_errors(metrics: dict[str, Any], output_dir: str | Path) -> Path:
    """绘制 1 到 72 小时逐 horizon 误差曲线。"""
    plt = _import_matplotlib()
    output_path = Path(output_dir) / "horizon_errors.png"
    horizons = [row["horizon"] for row in metrics["horizon"]]
    rmse = [row["RMSE"] for row in metrics["horizon"]]
    mae = [row["MAE"] for row in metrics["horizon"]]

    plt.figure(figsize=(10, 5))
    plt.plot(horizons, rmse, label="RMSE", linewidth=1.4)
    plt.plot(horizons, mae, label="MAE", linewidth=1.4)
    plt.xlabel("forecast horizon")
    plt.ylabel("error")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()
    return output_path


def plot_attention_weights(attention_weights: np.ndarray, output_dir: str | Path) -> Path:
    """绘制 Attention-LSTM 在 168 个输入时间步上的平均注意力权重。"""
    plt = _import_matplotlib()
    output_path = Path(output_dir) / "attention_weights.png"
    weights = np.asarray(attention_weights, dtype=float)
    if weights.ndim != 2:
        raise ValueError(f"Attention 权重必须为二维数组，实际为 {weights.shape}")

    mean_weights = weights.mean(axis=0)
    plt.figure(figsize=(10, 4))
    plt.plot(np.arange(1, len(mean_weights) + 1), mean_weights, linewidth=1.4)
    plt.xlabel("input time step")
    plt.ylabel("mean attention weight")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()
    return output_path


def plot_loss_curve(history: list[dict[str, Any]], output_dir: str | Path) -> Path:
    """绘制训练损失和验证损失曲线，辅助判断是否早停过早或未充分收敛。"""
    plt = _import_matplotlib()
    output_path = Path(output_dir) / "loss_curve.png"
    epochs = [row["epoch"] for row in history]
    train_loss = [row["train_loss"] for row in history]
    validation_loss = [row["validation_loss"] for row in history]

    plt.figure(figsize=(9, 5))
    plt.plot(epochs, train_loss, label="train_loss", linewidth=1.4)
    plt.plot(epochs, validation_loss, label="validation_loss", linewidth=1.4)
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()
    return output_path


def plot_peak_case(y_true: np.ndarray, y_pred: np.ndarray, sample_id: int, output_dir: str | Path) -> Path:
    """绘制真实峰值最高样本的 72 小时预测曲线。"""
    plt = _import_matplotlib()
    output_path = Path(output_dir) / "peak_case_top1.png"
    horizons = np.arange(1, y_true.shape[1] + 1)

    plt.figure(figsize=(10, 5))
    plt.plot(horizons, y_true[sample_id], label="y_true", linewidth=1.6)
    plt.plot(horizons, y_pred[sample_id], label="y_pred", linewidth=1.4)
    plt.xlabel("forecast horizon")
    plt.ylabel("PM2.5")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()
    return output_path


def create_model_plots(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metrics: dict[str, Any],
    plots_dir: str | Path,
    attention_weights: np.ndarray | None = None,
) -> None:
    """为单个模型生成标准 plots 目录内容。"""
    plots_path = Path(plots_dir)
    plots_path.mkdir(parents=True, exist_ok=True)
    plot_prediction_curve(y_true, y_pred, plots_path)
    plot_stage_errors(metrics, plots_path)
    plot_horizon_errors(metrics, plots_path)
    if attention_weights is not None:
        plot_attention_weights(attention_weights, plots_path)
