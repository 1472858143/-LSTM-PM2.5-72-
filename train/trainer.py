from __future__ import annotations

import copy
from datetime import datetime
from typing import Any, Iterable

import numpy as np
from tqdm import tqdm

from utils.config import apply_window_experiment, ensure_project_dirs, normalize_window_selection, resolve_path
from utils.data_loader import prepare_window_data
from utils.env import check_environment
from utils.metrics import compute_all_metrics
from utils.output import (
    copy_metrics_to_summary,
    prepare_model_output_dir,
    save_attention_stats,
    save_config_snapshot,
    save_execution_log,
    save_metrics,
    save_metrics_summary_tables,
    save_metrics_tables,
    save_peak_analysis,
    save_predictions,
    save_training_history,
)
from utils.runtime import runtime_label, runtime_write
from utils.seed import set_global_seed
from visualization.plots import create_model_plots, plot_loss_curve, plot_peak_case


def _instantiate_model(model_name: str, config: dict[str, Any]):
    """根据模型名称延迟导入并实例化模型。"""
    if model_name == "arima":
        from models.arima import ARIMAForecastModel

        return ARIMAForecastModel(config)
    if model_name == "sarima":
        from models.sarima import SARIMAForecastModel

        return SARIMAForecastModel(config)
    if model_name == "xgboost":
        from models.xgboost_model import XGBoostForecastModel

        return XGBoostForecastModel(config)
    if model_name == "random_forest":
        from models.random_forest import RandomForestForecastModel

        return RandomForestForecastModel(config)
    if model_name == "lstm":
        from models.lstm import LSTMForecastModel

        return LSTMForecastModel(config)
    if model_name == "attention_lstm":
        from models.attention_lstm import AttentionLSTMForecastModel

        return AttentionLSTMForecastModel(config)
    raise ValueError(f"不支持的模型名称: {model_name}")


def normalize_model_selection(config: dict[str, Any], models: Iterable[str] | str) -> list[str]:
    """校验命令行传入的模型名称。"""
    allowed = config["models"]["allowed_model_names"]
    if isinstance(models, str):
        if models == "all":
            selected = allowed
        else:
            selected = [models]
    else:
        selected = list(models)
        if selected == ["all"]:
            selected = allowed

    invalid = [model for model in selected if model not in allowed]
    if invalid:
        raise ValueError(f"模型名称不在允许列表中: {invalid}")
    return selected


def _validate_window_data(data: dict[str, Any], config: dict[str, Any]) -> None:
    """检查窗口数据 shape 是否符合当前窗口实验配置。"""
    input_window = int(config["window"]["input_window_hours"])
    output_window = int(config["window"]["output_window_hours"])
    feature_count = int(config["global_constraints"]["feature_count"])
    for split_name in ["train", "validation", "test"]:
        X = data[f"X_{split_name}"]
        y = data[f"y_{split_name}"]
        if X.ndim != 3 or X.shape[1:] != (input_window, feature_count):
            raise ValueError(f"{split_name} X shape 错误: {X.shape}")
        if y.ndim != 2 or y.shape[1] != output_window:
            raise ValueError(f"{split_name} y shape 错误: {y.shape}")
        if len(X) == 0:
            raise ValueError(f"{split_name} 没有可用滑动窗口样本。")


def _build_stage_counter(total_steps: int = 8):
    count = {"value": 0}

    def advance(config: dict[str, Any], executed_steps: list[str], message: str) -> None:
        count["value"] += 1
        executed_steps.append(message)
        runtime_write(config, f"{message} Progress: {count['value']}/{total_steps}")

    return advance


def _build_execution_log(
    config: dict[str, Any],
    model_name: str,
    executed_steps: list[str],
    start_time: datetime,
    end_time: datetime,
    status: str,
    error_message: str | None = None,
) -> dict[str, Any]:
    return {
        "window_name": str(config.get("_active_window_name", "default_window")),
        "model_name": model_name,
        "input_window_hours": int(config["window"]["input_window_hours"]),
        "output_window_hours": int(config["window"]["output_window_hours"]),
        "start_time": start_time.isoformat(),
        "end_time": end_time.isoformat(),
        "total_seconds": float((end_time - start_time).total_seconds()),
        "executed_steps": executed_steps,
        "status": status,
        "error_message": error_message,
    }


def _append_summary_rows(
    config: dict[str, Any],
    model_name: str,
    metrics: dict[str, Any],
    window_rows: list[dict[str, Any]],
    stage_rows: list[dict[str, Any]],
    horizon_rows: list[dict[str, Any]],
) -> None:
    window_name = str(config.get("_active_window_name", "default_window"))
    input_window_hours = int(config["window"]["input_window_hours"])

    window_rows.append(
        {
            "window_name": window_name,
            "input_window_hours": input_window_hours,
            "model": model_name,
            **metrics["overall"],
        }
    )
    for stage_name, values in metrics["stages"].items():
        stage_rows.append(
            {
                "window_name": window_name,
                "input_window_hours": input_window_hours,
                "model": model_name,
                "stage": stage_name,
                **values,
            }
        )
    for row in metrics["horizon"]:
        horizon_rows.append(
            {
                "window_name": window_name,
                "input_window_hours": input_window_hours,
                "model": model_name,
                **row,
            }
        )


def run_training_pipeline(
    config: dict[str, Any],
    selected_models: Iterable[str] | str = "all",
    selected_windows: Iterable[str] | str | None = None,
) -> dict[str, Any]:
    """统一训练主流程，支持单窗口与多窗口实验。"""
    models = normalize_model_selection(config, selected_models)
    windows = normalize_window_selection(config, selected_windows)
    ensure_project_dirs(config)
    check_environment(config, models)
    set_global_seed(int(config["environment"]["seed"]))

    deep_models = {"lstm", "attention_lstm"}
    results: dict[str, Any] = {}
    window_summary_rows: list[dict[str, Any]] = []
    stage_summary_rows: list[dict[str, Any]] = []
    horizon_summary_rows: list[dict[str, Any]] = []

    window_bar = tqdm(windows, desc="Window experiments", unit="window", dynamic_ncols=True)
    for window_index, window_experiment in enumerate(window_bar, start=1):
        window_config = apply_window_experiment(config, window_experiment)
        window_name = str(window_experiment["name"])
        window_bar.set_description(f"Window experiments: {window_index}/{len(windows)} {window_name}")

        data = prepare_window_data(window_config)
        _validate_window_data(data, window_config)

        results[window_name] = {}
        model_bar = tqdm(models, desc=f"Models in {window_name}", unit="model", leave=False, dynamic_ncols=True)
        for model_index, model_name in enumerate(model_bar, start=1):
            model_bar.set_description(f"Models in {window_name}: {model_index}/{len(models)} {model_name}")
            model_cfg = window_config["models"][model_name]
            if not model_cfg.get("enabled", True):
                continue

            runtime_config = copy.deepcopy(window_config)
            runtime_config["_active_model_name"] = model_name
            runtime_config["_runtime"] = {"writer": tqdm.write}
            output_dir = prepare_model_output_dir(runtime_config, model_name)
            executed_steps: list[str] = []
            advance_step = _build_stage_counter()
            start_time = datetime.now()
            history: list[dict[str, Any]] = []

            try:
                runtime_write(runtime_config, "Start")
                executed_steps.append("Start")
                advance_step(runtime_config, executed_steps, "Reading data...")
                advance_step(runtime_config, executed_steps, "Preprocessing data...")
                advance_step(runtime_config, executed_steps, "Building sliding windows...")

                model = _instantiate_model(model_name, runtime_config)

                advance_step(runtime_config, executed_steps, "Training model...")
                model.fit(data)
                history = getattr(model, "training_history", [])

                advance_step(runtime_config, executed_steps, "Predicting...")
                y_pred_scaled = model.predict(data)

                advance_step(runtime_config, executed_steps, "Calculating metrics...")
                y_true = data["scaler"].inverse_transform_target(data["y_test"])
                y_pred = data["scaler"].inverse_transform_target(y_pred_scaled)
                metrics = compute_all_metrics(y_true, y_pred, runtime_config)

                advance_step(runtime_config, executed_steps, "Saving outputs...")
                save_predictions(runtime_config, model_name, y_true, y_pred, data["timestamps_test"])
                save_metrics(runtime_config, model_name, metrics)
                save_config_snapshot(runtime_config, model_name)
                save_metrics_tables(runtime_config, model_name, metrics)
                model.save(output_dir / "model.pt")

                attention_weights = getattr(model, "attention_weights", None)
                if model_name == "attention_lstm" and attention_weights is not None:
                    attention_path = resolve_path(model_cfg["attention_weights_path"])
                    model.save_attention_weights(attention_path)
                    save_attention_stats(runtime_config, model_name, attention_weights)

                create_model_plots(
                    y_true,
                    y_pred,
                    metrics,
                    data["timestamps_test"],
                    output_dir / "plots",
                    window_name,
                    model_name,
                    int(runtime_config["window"]["output_window_hours"]),
                    attention_weights,
                )

                if model_name in deep_models:
                    save_training_history(runtime_config, model_name, history)
                    if history:
                        plot_loss_curve(history, output_dir / "plots")

                peak_summary = save_peak_analysis(runtime_config, model_name, y_true, y_pred, data["timestamps_test"])
                if peak_summary["selected_sample_ids"]:
                    plot_peak_case(
                        y_true,
                        y_pred,
                        data["timestamps_test"],
                        peak_summary["selected_sample_ids"][0],
                        output_dir / "plots",
                        window_name,
                        model_name,
                        int(runtime_config["window"]["output_window_hours"]),
                    )

                copy_metrics_to_summary(runtime_config, model_name)
                _append_summary_rows(
                    runtime_config,
                    model_name,
                    metrics,
                    window_summary_rows,
                    stage_summary_rows,
                    horizon_summary_rows,
                )

                end_time = datetime.now()
                save_execution_log(
                    runtime_config,
                    model_name,
                    _build_execution_log(
                        runtime_config,
                        model_name,
                        executed_steps + ["Finished"],
                        start_time,
                        end_time,
                        "success",
                    ),
                    history,
                )
                runtime_write(runtime_config, "Finished Progress: 8/8")

                results[window_name][model_name] = {
                    "output_dir": str(output_dir),
                    "predictions_shape": list(np.asarray(y_pred).shape),
                    "overall_metrics": metrics["overall"],
                    "status": "success",
                }
            except Exception as exc:
                end_time = datetime.now()
                save_execution_log(
                    runtime_config,
                    model_name,
                    _build_execution_log(
                        runtime_config,
                        model_name,
                        executed_steps,
                        start_time,
                        end_time,
                        "failed",
                        str(exc),
                    ),
                    history,
                )
                runtime_write(runtime_config, f"Failed: {exc}")
                results[window_name][model_name] = {
                    "output_dir": str(output_dir),
                    "status": "failed",
                    "error_message": str(exc),
                }

        model_bar.close()

    save_metrics_summary_tables(config, window_summary_rows, stage_summary_rows, horizon_summary_rows)
    window_bar.close()
    return results
