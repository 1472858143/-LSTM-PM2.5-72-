from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from utils.config import resolve_path


def model_output_dir(config: dict[str, Any], model_name: str) -> Path:
    """返回模型标准输出目录 outputs/{model}/。"""
    return resolve_path(config["outputs"]["model_dirs"][model_name])


def prepare_model_output_dir(config: dict[str, Any], model_name: str) -> Path:
    """创建模型输出目录和 plots 子目录。"""
    output_dir = model_output_dir(config, model_name)
    (output_dir / "plots").mkdir(parents=True, exist_ok=True)
    return output_dir


def save_predictions(
    config: dict[str, Any],
    model_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_timestamps: np.ndarray,
) -> Path:
    """保存前端和评估模块共同依赖的预测结果。

    每个测试样本有 72 行记录，timestamp 对应未来目标时间点，horizon 对应第几小时预测。
    """
    output_dir = prepare_model_output_dir(config, model_name)
    rows: list[dict[str, Any]] = []
    sample_count, horizon_count = y_true.shape

    for sample_id in range(sample_count):
        for horizon_idx in range(horizon_count):
            rows.append(
                {
                    "sample_id": sample_id,
                    "timestamp": str(target_timestamps[sample_id, horizon_idx]),
                    "horizon": horizon_idx + 1,
                    "y_true": float(y_true[sample_id, horizon_idx]),
                    "y_pred": float(y_pred[sample_id, horizon_idx]),
                }
            )

    columns = config["outputs"]["predictions_csv_columns"]
    path = output_dir / "predictions.csv"
    pd.DataFrame(rows, columns=columns).to_csv(path, index=False, encoding="utf-8")
    return path


def save_metrics(config: dict[str, Any], model_name: str, metrics: dict[str, Any]) -> Path:
    """保存统一指标文件 metrics.json。"""
    output_dir = prepare_model_output_dir(config, model_name)
    path = output_dir / "metrics.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    return path


def save_config_snapshot(config: dict[str, Any], model_name: str) -> Path:
    """保存本次运行配置快照，方便论文结果复现。"""
    output_dir = prepare_model_output_dir(config, model_name)
    path = output_dir / "config_snapshot.json"
    serializable = {k: v for k, v in config.items() if not k.startswith("_")}
    with path.open("w", encoding="utf-8") as f:
        json.dump(serializable, f, ensure_ascii=False, indent=2)
    return path


def copy_metrics_to_summary(config: dict[str, Any], model_name: str) -> None:
    """把各模型指标复制到汇总目录，便于后续前端或论文汇总使用。"""
    src = model_output_dir(config, model_name) / "metrics.json"
    dst_dir = resolve_path(config["paths"]["metrics_summary_dir"])
    dst_dir.mkdir(parents=True, exist_ok=True)
    if src.exists():
        shutil.copy2(src, dst_dir / f"{model_name}_metrics.json")


def save_metrics_tables(config: dict[str, Any], model_name: str, metrics: dict[str, Any]) -> None:
    """额外保存分阶段和逐 horizon 指标表。

    metrics.json 保持主接口不变；CSV 表方便论文直接引用和绘图。
    """
    output_dir = prepare_model_output_dir(config, model_name)
    stage_rows = [{"stage": stage, **values} for stage, values in metrics["stages"].items()]
    pd.DataFrame(stage_rows).to_csv(output_dir / "stage_metrics.csv", index=False, encoding="utf-8")
    pd.DataFrame(metrics["horizon"]).to_csv(output_dir / "horizon_metrics.csv", index=False, encoding="utf-8")


def save_training_history(config: dict[str, Any], model_name: str, history: list[dict[str, Any]]) -> None:
    """保存训练和验证损失曲线数据。

    training_log.json 提供摘要信息，training_history.csv 便于后续直接画图或写论文。
    """
    if not history:
        return
    output_dir = prepare_model_output_dir(config, model_name)
    best_row = min(history, key=lambda row: float(row["validation_loss"]))
    training_log = {
        "best_epoch": int(best_row["epoch"]),
        "best_validation_loss": float(best_row["validation_loss"]),
        "early_stopping_epoch": int(history[-1]["epoch"]),
        "epochs_completed": int(len(history)),
        "history": history,
    }
    with (output_dir / "training_log.json").open("w", encoding="utf-8") as f:
        json.dump(training_log, f, ensure_ascii=False, indent=2)
    with (output_dir / "training_history.json").open("w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
    pd.DataFrame(history).to_csv(output_dir / "training_history.csv", index=False, encoding="utf-8")


def save_attention_stats(config: dict[str, Any], model_name: str, attention_weights: np.ndarray) -> Path:
    """保存 Attention 权重统计，用于判断权重是否接近平均分布。"""
    output_dir = prepare_model_output_dir(config, model_name)
    weights = np.asarray(attention_weights, dtype=float)
    entropy = -(weights * np.log(np.clip(weights, 1e-12, None))).sum(axis=1)
    sorted_weights = np.sort(weights, axis=1)[:, ::-1]
    uniform_entropy = float(np.log(weights.shape[1]))
    mean_weights = weights.mean(axis=0)
    top_indices = np.argsort(mean_weights)[-5:][::-1]
    stats = {
        "shape": list(weights.shape),
        "uniform_weight": float(1.0 / weights.shape[1]),
        "mean": float(weights.mean()),
        "std": float(weights.std()),
        "min": float(weights.min()),
        "max": float(weights.max()),
        "top_1_weight": float(sorted_weights[:, 0].mean()),
        "top_5_weight_sum": float(sorted_weights[:, :5].sum(axis=1).mean()),
        "top_10_weight_sum": float(sorted_weights[:, :10].sum(axis=1).mean()),
        "max_per_sample_mean": float(weights.max(axis=1).mean()),
        "max_per_sample_p95": float(np.percentile(weights.max(axis=1), 95)),
        "entropy_mean": float(entropy.mean()),
        "entropy_std": float(entropy.std()),
        "uniform_entropy": uniform_entropy,
        "entropy_ratio_to_uniform": float(entropy.mean() / uniform_entropy),
        # 这里故意把阈值收紧到 0.999，用于捕捉“几乎完全均匀”的退化注意力。
        "near_uniform": bool(entropy.mean() / uniform_entropy > 0.999),
        "top_mean_weight_steps": [
            {"input_step": int(index + 1), "mean_weight": float(mean_weights[index])}
            for index in top_indices
        ],
    }
    path = output_dir / "attention_stats.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    return path


def save_peak_analysis(
    config: dict[str, Any],
    model_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_timestamps: np.ndarray,
) -> dict[str, Any]:
    """保存测试集中高 PM2.5 样本的逐 horizon 对比。

    选取真实峰值最高的若干测试样本，不改变原 predictions.csv，只额外输出论文分析文件。
    """
    output_dir = prepare_model_output_dir(config, model_name)
    analysis_cfg = config["models"][model_name].get("analysis", {})
    peak_quantile = float(analysis_cfg.get("peak_quantile", 0.9))
    top_k = int(analysis_cfg.get("peak_top_k", 5))

    sample_peak = np.max(y_true, axis=1)
    threshold = float(np.quantile(sample_peak, peak_quantile))
    selected = np.argsort(sample_peak)[-top_k:][::-1]

    rows: list[dict[str, Any]] = []
    for rank, sample_id in enumerate(selected, start=1):
        for horizon_idx in range(y_true.shape[1]):
            true_value = float(y_true[sample_id, horizon_idx])
            pred_value = float(y_pred[sample_id, horizon_idx])
            rows.append(
                {
                    "rank": rank,
                    "sample_id": int(sample_id),
                    "timestamp": str(target_timestamps[sample_id, horizon_idx]),
                    "horizon": horizon_idx + 1,
                    "y_true": true_value,
                    "y_pred": pred_value,
                    "error": true_value - pred_value,
                    "abs_error": abs(true_value - pred_value),
                }
            )

    pd.DataFrame(rows).to_csv(output_dir / "peak_analysis.csv", index=False, encoding="utf-8")
    summary = {
        "peak_quantile": peak_quantile,
        "peak_threshold": threshold,
        "top_k": top_k,
        "selected_sample_ids": [int(i) for i in selected],
        "selected_sample_true_peaks": [float(sample_peak[i]) for i in selected],
    }
    with (output_dir / "peak_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return summary
