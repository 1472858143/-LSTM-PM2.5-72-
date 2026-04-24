from __future__ import annotations

from typing import Any

import numpy as np


def _as_flat(y: np.ndarray) -> np.ndarray:
    return np.asarray(y, dtype=float).reshape(-1)


def _finite_or_none(value: float) -> float | None:
    value = float(value)
    if not np.isfinite(value):
        return None
    return value


def compute_metric_set(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    mape_denominator_min: float = 1.0,
) -> dict[str, float | None]:
    """计算项目规定的 8 个评价指标。

    传入值应为反归一化后的 PM2.5 浓度，这样 RMSE、MAE 等指标才具有真实物理单位。
    MAPE 的分母下限来自配置，避免真实值接近 0 时相对误差失真。
    """
    true = _as_flat(y_true)
    pred = _as_flat(y_pred)
    error = true - pred
    abs_error = np.abs(error)
    squared_error = error**2

    mse = np.mean(squared_error)
    rmse = np.sqrt(mse)
    mae = np.mean(abs_error)
    mape_denominator = np.maximum(np.abs(true), mape_denominator_min)
    mape = np.mean(abs_error / mape_denominator) * 100
    smape_denominator = np.maximum(np.abs(true) + np.abs(pred), 1e-12)
    smape = np.mean(2 * abs_error / smape_denominator) * 100

    # R2 和 Explained Variance 用于衡量趋势解释能力，常数真实序列时返回 0。
    ss_res = np.sum(squared_error)
    ss_tot = np.sum((true - np.mean(true)) ** 2)
    r2 = 0.0 if ss_tot <= 1e-12 else 1 - ss_res / ss_tot

    var_true = np.var(true)
    explained_variance = 0.0 if var_true <= 1e-12 else 1 - np.var(error) / var_true
    max_error = np.max(abs_error) if len(abs_error) else np.nan

    return {
        "RMSE": _finite_or_none(rmse),
        "MAE": _finite_or_none(mae),
        "MAPE": _finite_or_none(mape),
        "R2": _finite_or_none(r2),
        "SMAPE": _finite_or_none(smape),
        "MSE": _finite_or_none(mse),
        "Explained Variance": _finite_or_none(explained_variance),
        "Max Error": _finite_or_none(max_error),
    }


def compute_all_metrics(y_true: np.ndarray, y_pred: np.ndarray, config: dict[str, Any]) -> dict[str, Any]:
    """生成 metrics.json 的完整结构。

    overall 评估全部测试样本和 72 个 horizon；stages 对应 1-24、25-48、
    49-72 小时；horizon 用于分析误差随预测步长变化的稳定性。
    """
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    if y_true_arr.shape != y_pred_arr.shape:
        raise ValueError(f"y_true 和 y_pred shape 不一致: {y_true_arr.shape} != {y_pred_arr.shape}")
    if y_true_arr.ndim != 2:
        raise ValueError(f"指标计算要求二维数组 (samples, horizon)，实际为 {y_true_arr.shape}")

    mape_min = float(config["evaluation"]["mape_denominator_min"])
    stage_config = config["evaluation"]["multi_step_analysis"]["stages"]

    metrics: dict[str, Any] = {
        "overall": compute_metric_set(y_true_arr, y_pred_arr, mape_min),
        "stages": {},
        "horizon": [],
    }

    for name, (start, end) in stage_config.items():
        start_idx = int(start) - 1
        end_idx = int(end)
        metrics["stages"][name] = compute_metric_set(
            y_true_arr[:, start_idx:end_idx],
            y_pred_arr[:, start_idx:end_idx],
            mape_min,
        )

    for horizon_idx in range(y_true_arr.shape[1]):
        horizon_metrics = compute_metric_set(
            y_true_arr[:, horizon_idx : horizon_idx + 1],
            y_pred_arr[:, horizon_idx : horizon_idx + 1],
            mape_min,
        )
        metrics["horizon"].append({"horizon": horizon_idx + 1, **horizon_metrics})

    return metrics
