from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from utils.config import apply_window_experiment, load_config, normalize_window_selection
from utils.metrics import compute_all_metrics
from utils.output import model_output_dir, save_metrics, save_metrics_tables
from visualization.plots import create_model_plots


def load_prediction_arrays(predictions_csv: str | Path) -> tuple[Any, Any, Any]:
    """从 predictions.csv 还原二维真实值、预测值和时间戳数组。"""
    df = pd.read_csv(predictions_csv)
    required = {"sample_id", "timestamp", "horizon", "y_true", "y_pred"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"predictions.csv 缺少字段: {sorted(missing)}")

    ordered = df.sort_values(["sample_id", "horizon"])
    true = ordered.pivot(index="sample_id", columns="horizon", values="y_true").sort_index(axis=1).to_numpy()
    pred = ordered.pivot(index="sample_id", columns="horizon", values="y_pred").sort_index(axis=1).to_numpy()
    timestamps = (
        ordered.pivot(index="sample_id", columns="horizon", values="timestamp").sort_index(axis=1).to_numpy()
    )
    return true, pred, timestamps


def evaluate_model_outputs(config: dict[str, Any], model_name: str) -> dict[str, Any]:
    """基于已有预测结果重新生成 metrics.json 和图表。"""
    output_dir = model_output_dir(config, model_name)
    predictions_csv = output_dir / "predictions.csv"
    y_true, y_pred, timestamps = load_prediction_arrays(predictions_csv)
    metrics = compute_all_metrics(y_true, y_pred, config)
    save_metrics(config, model_name, metrics)
    save_metrics_tables(config, model_name, metrics)
    create_model_plots(
        y_true,
        y_pred,
        metrics,
        timestamps,
        output_dir / "plots",
        str(config.get("_active_window_name", "default_window")),
        model_name,
        int(config["window"]["output_window_hours"]),
    )
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate saved model predictions.")
    parser.add_argument("--config", default="config/config.json", help="配置文件路径")
    parser.add_argument("--model", required=True, help="模型名称")
    parser.add_argument("--window", default=None, help="窗口实验名称")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.window is not None:
        window_experiment = normalize_window_selection(config, [args.window])[0]
        config = apply_window_experiment(config, window_experiment)
    metrics = evaluate_model_outputs(config, args.model)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
