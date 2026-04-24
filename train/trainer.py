from __future__ import annotations

import copy
from datetime import datetime
from typing import Any, Iterable

import numpy as np

from utils.config import apply_window_experiment, ensure_project_dirs, normalize_window_selection, resolve_path
from utils.console_utils import create_progress, get_console, setup_console_encoding
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
from utils.runtime import runtime_label, runtime_remove_task, runtime_update_task, runtime_write
from utils.seed import set_global_seed
from visualization.plots import create_model_plots, plot_loss_curve, plot_peak_case


MODEL_STAGES = (
    "Reading data...",
    "Preprocessing data...",
    "Building sliding windows...",
    "Training model...",
    "Predicting...",
    "Calculating metrics...",
    "Saving outputs...",
    "Finished",
)


def _instantiate_model(model_name: str, config: dict[str, Any]):
    """Instantiate the selected model lazily."""
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
    raise ValueError(f"Unsupported model name: {model_name}")


def normalize_model_selection(config: dict[str, Any], models: Iterable[str] | str) -> list[str]:
    """Validate model names received from the CLI."""
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
        raise ValueError(f"Model names are not allowed: {invalid}")
    return selected


def _validate_window_data(data: dict[str, Any], config: dict[str, Any]) -> None:
    """Validate windowed split shapes for the active experiment."""
    input_window = int(config["window"]["input_window_hours"])
    output_window = int(config["window"]["output_window_hours"])
    feature_count = int(config["global_constraints"]["feature_count"])
    for split_name in ["train", "validation", "test"]:
        X = data[f"X_{split_name}"]
        y = data[f"y_{split_name}"]
        if X.ndim != 3 or X.shape[1:] != (input_window, feature_count):
            raise ValueError(f"{split_name} X shape mismatch: {X.shape}")
        if y.ndim != 2 or y.shape[1] != output_window:
            raise ValueError(f"{split_name} y shape mismatch: {y.shape}")
        if len(X) == 0:
            raise ValueError(f"{split_name} has no available sliding-window samples.")


def _enabled_models(window_config: dict[str, Any], models: Iterable[str]) -> list[str]:
    return [model_name for model_name in models if window_config["models"][model_name].get("enabled", True)]


def _advance_stage(config: dict[str, Any], executed_steps: list[str], message: str) -> None:
    executed_steps.append(message)
    runtime_write(config, message)
    runtime_update_task(config, config.get("_runtime", {}).get("step_task_id"), advance=1)


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
    """Run the full training pipeline across windows and models."""
    setup_console_encoding()
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

    console = get_console()
    with create_progress(console=console) as progress:
        window_task_id = progress.add_task("Window Experiments", total=max(len(windows), 1), stats="")

        for window_experiment in windows:
            window_config = apply_window_experiment(config, window_experiment)
            window_name = str(window_experiment["name"])

            data = prepare_window_data(window_config)
            _validate_window_data(data, window_config)

            results[window_name] = {}
            enabled_models = _enabled_models(window_config, models)
            model_task_id = None
            if enabled_models:
                model_task_id = progress.add_task(f"[{window_name}] Models", total=len(enabled_models), stats="")

            for model_name in enabled_models:
                model_cfg = window_config["models"][model_name]
                runtime_config = copy.deepcopy(window_config)
                runtime_config["_active_model_name"] = model_name

                step_task_id = progress.add_task(
                    f"{runtime_label(runtime_config)} Steps",
                    total=len(MODEL_STAGES),
                    stats="",
                )
                runtime_config["_runtime"] = {
                    "console": console,
                    "progress": progress,
                    "window_task_id": window_task_id,
                    "model_task_id": model_task_id,
                    "step_task_id": step_task_id,
                }

                output_dir = prepare_model_output_dir(runtime_config, model_name)
                executed_steps: list[str] = []
                start_time = datetime.now()
                history: list[dict[str, Any]] = []

                try:
                    runtime_write(runtime_config, "Start")
                    executed_steps.append("Start")

                    for message in MODEL_STAGES[:3]:
                        _advance_stage(runtime_config, executed_steps, message)

                    model = _instantiate_model(model_name, runtime_config)

                    _advance_stage(runtime_config, executed_steps, MODEL_STAGES[3])
                    model.fit(data)
                    history = getattr(model, "training_history", [])

                    _advance_stage(runtime_config, executed_steps, MODEL_STAGES[4])
                    y_pred_scaled = model.predict(data)

                    _advance_stage(runtime_config, executed_steps, MODEL_STAGES[5])
                    y_true = data["scaler"].inverse_transform_target(data["y_test"])
                    y_pred = data["scaler"].inverse_transform_target(y_pred_scaled)
                    metrics = compute_all_metrics(y_true, y_pred, runtime_config)

                    _advance_stage(runtime_config, executed_steps, MODEL_STAGES[6])
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

                    _advance_stage(runtime_config, executed_steps, MODEL_STAGES[7])
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
                            "success",
                        ),
                        history,
                    )

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
                    runtime_write(runtime_config, f"Failed: {exc}", message_style="bold red")
                    results[window_name][model_name] = {
                        "output_dir": str(output_dir),
                        "status": "failed",
                        "error_message": str(exc),
                    }
                finally:
                    if model_task_id is not None:
                        progress.advance(model_task_id, 1)
                    runtime_remove_task(runtime_config, step_task_id)

            if model_task_id is not None:
                progress.remove_task(model_task_id)
            progress.advance(window_task_id, 1)

    save_metrics_summary_tables(config, window_summary_rows, stage_summary_rows, horizon_summary_rows)
    return results
