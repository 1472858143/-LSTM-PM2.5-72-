from __future__ import annotations

import argparse
import copy
import json
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from models.attention_lstm import AttentionLSTMForecastModel
from train.trainer import _validate_window_data
from utils.config import dump_json, ensure_project_dirs, load_config, resolve_path
from utils.console_utils import get_console, setup_console_encoding
from utils.data_loader import prepare_window_data
from utils.env import check_environment
from utils.metrics import compute_all_metrics
from utils.output import (
    prepare_model_output_dir,
    save_attention_stats,
    save_config_snapshot,
    save_metrics,
    save_metrics_tables,
    save_peak_analysis,
    save_predictions,
    save_training_history,
)
from utils.seed import set_global_seed
from visualization.plots import create_model_plots, plot_loss_curve, plot_peak_case


RESULT_COLUMNS = [
    "hidden_size",
    "num_layers",
    "dropout",
    "learning_rate",
    "RMSE",
    "MAE",
    "R2",
    "Max_Error",
    "amplitude_pred",
]


def parse_args() -> argparse.Namespace:
    """解析 Attention-LSTM 自动调参入口参数。"""
    parser = argparse.ArgumentParser(description="Auto tune Attention-LSTM for PM2.5 forecasting.")
    parser.add_argument("--config", default="config/config.json", help="配置文件路径")
    return parser.parse_args()


def _load_tuning_config(config: dict[str, Any]) -> dict[str, Any]:
    """读取 attention_lstm 的自动调参配置，并检查是否满足项目约束。"""
    tuning_cfg = config["tuning"]["attention_lstm"]
    selection_split = str(tuning_cfg["selection_split"]).lower()
    if selection_split == "test" and not config["global_constraints"].get("test_set_for_tuning_allowed", False):
        raise ValueError("当前项目禁止使用测试集进行调参，selection_split 必须为 train 或 validation。")
    if selection_split not in {"train", "validation"}:
        raise ValueError(f"不支持的 selection_split: {selection_split}")
    return tuning_cfg


def _iter_search_space(search_space: dict[str, list[Any]]) -> list[dict[str, Any]]:
    """生成有限网格搜索组合，仅覆盖题目要求的四个超参数。"""
    keys = ["hidden_size", "num_layers", "dropout", "learning_rate"]
    values = [search_space[key] for key in keys]
    return [dict(zip(keys, combo)) for combo in product(*values)]


def _build_trial_config(base_config: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
    """在不改动主配置对象的前提下，构造单次试验使用的 Attention-LSTM 配置。"""
    trial_config = copy.deepcopy(base_config)
    tuning_cfg = trial_config["tuning"]["attention_lstm"]
    training_cfg = tuning_cfg["training"]
    model_cfg = trial_config["models"]["attention_lstm"]

    model_cfg["hidden_size"] = int(params["hidden_size"])
    model_cfg["num_layers"] = int(params["num_layers"])
    model_cfg["dropout"] = float(params["dropout"])
    model_cfg["learning_rate"] = float(params["learning_rate"])
    model_cfg["epochs"] = int(training_cfg["epochs"])
    model_cfg["early_stopping_patience"] = int(training_cfg["early_stopping_patience"])
    model_cfg["batch_size"] = int(training_cfg["batch_size"])
    model_cfg["loss"] = str(training_cfg["loss"])
    model_cfg["huber_delta"] = float(training_cfg["huber_delta"])
    model_cfg["checkpoint_metric"] = str(training_cfg.get("checkpoint_metric", "q80_then_h1_then_rmse"))
    model_cfg["scheduler"] = str(training_cfg.get("scheduler", "ReduceLROnPlateau"))
    model_cfg["scheduler_factor"] = float(training_cfg.get("scheduler_factor", 0.5))
    model_cfg["scheduler_patience"] = int(training_cfg.get("scheduler_patience", 3))
    model_cfg["scheduler_min_lr"] = float(training_cfg.get("scheduler_min_lr", 1e-5))

    # 本轮调参只看基础损失与网络容量，不启用高值样本加权，避免把训练目标再引向新的变量。
    model_cfg.setdefault("high_value_weighting", {})
    model_cfg["high_value_weighting"]["enabled"] = not bool(training_cfg.get("disable_high_value_weighting", True))

    return trial_config


def _build_best_output_config(config: dict[str, Any]) -> dict[str, Any]:
    """把最优模型的正式输出目录切到 outputs/attention_lstm_tuning/。"""
    best_config = copy.deepcopy(config)
    tuning_cfg = best_config["tuning"]["attention_lstm"]
    results_dir = tuning_cfg["results_dir"]
    best_config["outputs"]["model_dirs"]["attention_lstm"] = results_dir
    best_config["models"]["attention_lstm"]["attention_weights_path"] = str(Path(results_dir) / "attention_weights.npy")
    return best_config


def _build_eval_data(data: dict[str, Any], split_name: str) -> dict[str, Any]:
    """复用模型现有 predict 接口，对指定 split 评估而不改模型定义。"""
    return {
        **data,
        "X_test": data[f"X_{split_name}"],
        "y_test": data[f"y_{split_name}"],
        "timestamps_test": data[f"timestamps_{split_name}"],
    }


def _evaluate_split(
    config: dict[str, Any],
    model: AttentionLSTMForecastModel,
    data: dict[str, Any],
    split_name: str,
) -> dict[str, Any]:
    """在指定 split 上生成反归一化预测并计算评价指标。"""
    eval_data = _build_eval_data(data, split_name)
    y_pred_scaled = model.predict(eval_data)
    y_true = data["scaler"].inverse_transform_target(data[f"y_{split_name}"])
    y_pred = data["scaler"].inverse_transform_target(y_pred_scaled)
    metrics = compute_all_metrics(y_true, y_pred, config)
    amplitude_pred = float(np.max(y_pred) - np.min(y_pred)) if y_pred.size else None
    return {
        "y_true": y_true,
        "y_pred": y_pred,
        "timestamps": data[f"timestamps_{split_name}"],
        "metrics": metrics,
        "amplitude_pred": amplitude_pred,
    }


def _make_result_row(
    params: dict[str, Any],
    overall_metrics: dict[str, float | None] | None,
    amplitude_pred: float | None,
) -> dict[str, Any]:
    """把单次试验结果整理成 results.csv 的一行。"""
    overall_metrics = overall_metrics or {}
    return {
        "hidden_size": int(params["hidden_size"]),
        "num_layers": int(params["num_layers"]),
        "dropout": float(params["dropout"]),
        "learning_rate": float(params["learning_rate"]),
        "RMSE": overall_metrics.get("RMSE"),
        "MAE": overall_metrics.get("MAE"),
        "R2": overall_metrics.get("R2"),
        "Max_Error": overall_metrics.get("Max Error"),
        "amplitude_pred": amplitude_pred,
    }


def _write_results_csv(path: str | Path, rows: list[dict[str, Any]]) -> Path:
    """每轮试验后刷新 results.csv，保证长时间网格搜索中途也能保留已完成结果。"""
    output_path = resolve_path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=RESULT_COLUMNS).to_csv(output_path, index=False, encoding="utf-8")
    return output_path


def _selection_key(row: dict[str, Any]) -> tuple[float, float, float]:
    """按题目要求排序：先 R2，再 amplitude_pred，最后 RMSE。"""
    r2 = float("-inf") if row["R2"] is None else float(row["R2"])
    amplitude = float("-inf") if row["amplitude_pred"] is None else float(row["amplitude_pred"])
    rmse = float("inf") if row["RMSE"] is None else float(row["RMSE"])
    return (r2, amplitude, -rmse)


def _save_best_outputs(
    config: dict[str, Any],
    model: AttentionLSTMForecastModel,
    data: dict[str, Any],
    evaluation: dict[str, Any],
) -> None:
    """把最优参数在测试集上的正式输出写入 attention_lstm_tuning 目录。"""
    model_name = "attention_lstm"
    output_dir = prepare_model_output_dir(config, model_name)

    save_predictions(config, model_name, evaluation["y_true"], evaluation["y_pred"], evaluation["timestamps"])
    save_metrics(config, model_name, evaluation["metrics"])
    save_config_snapshot(config, model_name)
    save_metrics_tables(config, model_name, evaluation["metrics"])
    model.save(output_dir / "model.pt")

    attention_weights = getattr(model, "attention_weights", None)
    attention_diagnostics = getattr(model, "attention_diagnostics", None)
    if attention_weights is not None:
        attention_path = resolve_path(config["models"][model_name]["attention_weights_path"])
        model.save_attention_weights(attention_path)
        save_attention_stats(config, model_name, attention_weights, attention_diagnostics)

    create_model_plots(
        evaluation["y_true"],
        evaluation["y_pred"],
        evaluation["metrics"],
        evaluation["timestamps"],
        output_dir / "plots",
        "tuning",
        model_name,
        int(config["window"]["output_window_hours"]),
        attention_weights,
    )

    history = getattr(model, "training_history", [])
    save_training_history(config, model_name, history)
    if history:
        plot_loss_curve(history, output_dir / "plots")

    peak_summary = save_peak_analysis(
        config,
        model_name,
        evaluation["y_true"],
        evaluation["y_pred"],
        evaluation["timestamps"],
    )
    if peak_summary["selected_sample_ids"]:
        plot_peak_case(
            evaluation["y_true"],
            evaluation["y_pred"],
            evaluation["timestamps"],
            peak_summary["selected_sample_ids"][0],
            output_dir / "plots",
            "tuning",
            model_name,
            int(config["window"]["output_window_hours"]),
        )


def main() -> None:
    """执行 Attention-LSTM 网格调参，并导出最优组合的正式测试集结果。"""
    setup_console_encoding()
    console = get_console()
    args = parse_args()
    base_config = load_config(args.config)
    tuning_cfg = _load_tuning_config(base_config)

    ensure_project_dirs(base_config)
    resolve_path(tuning_cfg["results_dir"]).mkdir(parents=True, exist_ok=True)
    check_environment(base_config, ["attention_lstm"])
    set_global_seed(int(base_config["environment"]["seed"]))

    data = prepare_window_data(base_config)
    _validate_window_data(data, base_config)

    selection_split = str(tuning_cfg["selection_split"]).lower()
    search_space = _iter_search_space(tuning_cfg["search_space"])
    results: list[dict[str, Any]] = []

    for index, params in enumerate(search_space, start=1):
        trial_config = _build_trial_config(base_config, params)
        set_global_seed(int(trial_config["environment"]["seed"]))
        model = AttentionLSTMForecastModel(trial_config)

        try:
            model.fit(data)
            evaluation = _evaluate_split(trial_config, model, data, selection_split)
            row = _make_result_row(params, evaluation["metrics"]["overall"], evaluation["amplitude_pred"])
        except Exception:
            # 单次组合失败不应让整轮网格搜索中断；失败组合在结果表中保留空指标，便于排查。
            row = _make_result_row(params, None, None)

        results.append(row)
        _write_results_csv(tuning_cfg["results_csv"], results)
        console.print(
            json.dumps(
                {
                    "trial": f"{index}/{len(search_space)}",
                    "params": params,
                    "metrics": {key: row[key] for key in ["RMSE", "MAE", "R2", "Max_Error", "amplitude_pred"]},
                },
                ensure_ascii=False,
            ),
            markup=False,
        )

    successful_rows = [row for row in results if row["R2"] is not None]
    if not successful_rows:
        raise RuntimeError("Attention-LSTM 自动调参未得到任何可用结果，请检查 CUDA 与依赖环境。")

    best_row = max(successful_rows, key=_selection_key)
    best_params = {
        "hidden_size": int(best_row["hidden_size"]),
        "num_layers": int(best_row["num_layers"]),
        "dropout": float(best_row["dropout"]),
        "learning_rate": float(best_row["learning_rate"]),
    }

    best_config = _build_best_output_config(_build_trial_config(base_config, best_params))
    set_global_seed(int(best_config["environment"]["seed"]))
    best_model = AttentionLSTMForecastModel(best_config)
    best_model.fit(data)
    test_evaluation = _evaluate_split(best_config, best_model, data, "test")
    _save_best_outputs(best_config, best_model, data, test_evaluation)

    summary = {
        "selection_split": selection_split,
        "selection_priority": tuning_cfg["selection_priority"],
        "best_params": best_params,
        "best_validation_metrics": {
            "RMSE": best_row["RMSE"],
            "MAE": best_row["MAE"],
            "R2": best_row["R2"],
            "Max_Error": best_row["Max_Error"],
            "amplitude_pred": best_row["amplitude_pred"],
        },
        "test_overall_metrics": test_evaluation["metrics"]["overall"],
        "test_amplitude_pred": test_evaluation["amplitude_pred"],
        "output_dir": str(resolve_path(tuning_cfg["results_dir"])),
        "results_csv": str(resolve_path(tuning_cfg["results_csv"])),
    }
    dump_json(summary, tuning_cfg["best_params_json"])
    console.print(json.dumps(summary, ensure_ascii=False, indent=2), markup=False)


if __name__ == "__main__":
    main()
