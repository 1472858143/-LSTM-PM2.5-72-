from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from utils.config import load_config, resolve_path
from utils.metrics import compute_all_metrics
from utils.output import model_output_dir, save_metrics
from visualization.plots import create_model_plots


def load_prediction_arrays(predictions_csv: str | Path) -> tuple[Any, Any]:
    """从 predictions.csv 还原二维真实值和预测值数组。

    CSV 是前端友好的长表结构；评估时需要按 sample_id 和 horizon 还原为
    (测试样本数, 72)，才能计算整体、分阶段和逐 horizon 指标。
    """
    df = pd.read_csv(predictions_csv)
    required = {"sample_id", "timestamp", "horizon", "y_true", "y_pred"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"predictions.csv 缺少字段: {sorted(missing)}")

    true = df.pivot(index="sample_id", columns="horizon", values="y_true").sort_index(axis=1).to_numpy()
    pred = df.pivot(index="sample_id", columns="horizon", values="y_pred").sort_index(axis=1).to_numpy()
    return true, pred


def evaluate_model_outputs(config: dict[str, Any], model_name: str) -> dict[str, Any]:
    """基于已有预测结果重新生成 metrics.json 和图表。"""
    output_dir = model_output_dir(config, model_name)
    predictions_csv = output_dir / "predictions.csv"
    y_true, y_pred = load_prediction_arrays(predictions_csv)
    metrics = compute_all_metrics(y_true, y_pred, config)
    save_metrics(config, model_name, metrics)
    create_model_plots(y_true, y_pred, metrics, output_dir / "plots")
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate saved model predictions.")
    parser.add_argument("--config", default="config/config.json", help="配置文件路径")
    parser.add_argument("--model", required=True, help="模型名称")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    metrics = evaluate_model_outputs(config, args.model)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
