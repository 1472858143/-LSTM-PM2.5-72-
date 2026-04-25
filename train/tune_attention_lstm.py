from __future__ import annotations

import argparse
import copy
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from train.trainer import _instantiate_model, _validate_window_data
from utils.config import (
    apply_model_window_profile,
    apply_window_experiment,
    dump_json,
    ensure_project_dirs,
    load_config,
    normalize_window_selection,
    resolve_path,
)
from utils.console_utils import get_console, setup_console_encoding
from utils.data_loader import prepare_window_data
from utils.env import check_environment
from utils.metrics import compute_all_metrics
from utils.seed import set_global_seed


RESULT_COLUMNS = [
    "window_name",
    "model_name",
    "trial_id",
    "dropout",
    "learning_rate",
    "weight_decay",
    "selection_score",
    "val_rmse",
    "val_stage1_rmse",
    "val_q90_mae",
    "RMSE",
    "MAE",
    "R2",
    "Max_Error",
    "amplitude_pred",
    "best_epoch",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune long-window LSTM or Attention-LSTM on the validation split.")
    parser.add_argument("--config", default="config/config.json", help="配置文件路径")
    parser.add_argument("--window", required=True, help="窗口实验名称，例如 input_720h 或 input_2160h")
    parser.add_argument("--model", required=True, choices=["lstm", "attention_lstm"], help="待调参模型")
    parser.add_argument("--max-trials", type=int, default=8, help="最多执行的 trial 数")
    return parser.parse_args()


def _load_tuning_entry(config: dict[str, Any], window_name: str, model_name: str) -> dict[str, Any]:
    tuning_root = config["tuning"]["long_window_deep_models"]
    selection_split = str(tuning_root.get("selection_split", "validation")).lower()
    if selection_split != "validation":
        raise ValueError("长窗口调参只允许使用 validation split。")

    try:
        return copy.deepcopy(tuning_root["windows"][window_name][model_name])
    except KeyError as exc:
        raise ValueError(f"未找到 {window_name}/{model_name} 的调参配置。") from exc


def _iter_search_space(search_space: dict[str, list[Any]], max_trials: int) -> list[dict[str, Any]]:
    keys = ["dropout", "learning_rate_scale", "weight_decay"]
    values = [list(search_space[key]) for key in keys]
    trials = [dict(zip(keys, combo)) for combo in product(*values)]
    return trials[: max(int(max_trials), 0)]


def _build_trial_config(
    runtime_config: dict[str, Any],
    model_name: str,
    params: dict[str, Any],
) -> dict[str, Any]:
    trial_config = copy.deepcopy(runtime_config)
    model_cfg = trial_config["models"][model_name]
    base_learning_rate = float(model_cfg["learning_rate"])
    model_cfg["dropout"] = float(params["dropout"])
    model_cfg["learning_rate"] = float(base_learning_rate * float(params["learning_rate_scale"]))
    model_cfg["weight_decay"] = float(params["weight_decay"])
    return trial_config


def _selection_score(
    val_rmse: float,
    val_stage1_rmse: float,
    val_q90_mae: float,
    weights: dict[str, float],
) -> float:
    if not all(np.isfinite(v) for v in [val_rmse, val_stage1_rmse, val_q90_mae]):
        return float("inf")
    return (
        float(weights.get("rmse", 0.5)) * float(val_rmse)
        + float(weights.get("stage1_rmse", 0.3)) * float(val_stage1_rmse)
        + float(weights.get("q90_mae", 0.2)) * float(val_q90_mae)
    )


def _build_eval_data(data: dict[str, Any], split_name: str) -> dict[str, Any]:
    return {
        **data,
        "X_test": data[f"X_{split_name}"],
        "y_test": data[f"y_{split_name}"],
        "timestamps_test": data[f"timestamps_{split_name}"],
    }


def _extract_best_history_row(history: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not history:
        return None

    best_rows = [row for row in history if bool(row.get("is_best_epoch", False))]
    if best_rows:
        return best_rows[-1]

    def row_key(row: dict[str, Any]) -> float:
        value = row.get("selection_score")
        if value is None:
            return float("inf")
        return float(value)

    return min(history, key=row_key)


def _evaluate_validation(
    model: Any,
    config: dict[str, Any],
    data: dict[str, Any],
) -> dict[str, Any]:
    eval_data = _build_eval_data(data, "validation")
    y_pred_scaled = model.predict(eval_data)
    y_true = data["scaler"].inverse_transform_target(data["y_validation"])
    y_pred = data["scaler"].inverse_transform_target(y_pred_scaled)
    metrics = compute_all_metrics(y_true, y_pred, config)
    q90_threshold = float(np.nanquantile(data["splits_raw"]["train"][data["target_column"]], 0.90))
    q90_mask = np.asarray(y_true, dtype=float) >= q90_threshold
    val_q90_mae = float(np.mean(np.abs(y_true[q90_mask] - y_pred[q90_mask]))) if np.any(q90_mask) else float(
        metrics["overall"]["MAE"]
    )
    val_stage1_rmse = float(metrics["stages"]["h1_24"]["RMSE"])
    val_rmse = float(metrics["overall"]["RMSE"])
    selection_weights = config["models"][model.name].get("selection_score_weights", {})
    selection_score = _selection_score(val_rmse, val_stage1_rmse, val_q90_mae, selection_weights)
    amplitude_pred = float(np.max(y_pred) - np.min(y_pred)) if y_pred.size else None
    return {
        "y_true": y_true,
        "y_pred": y_pred,
        "metrics": metrics,
        "val_rmse": val_rmse,
        "val_stage1_rmse": val_stage1_rmse,
        "val_q90_mae": val_q90_mae,
        "selection_score": selection_score,
        "amplitude_pred": amplitude_pred,
    }


def _make_result_row(
    window_name: str,
    model_name: str,
    trial_id: int,
    params: dict[str, Any],
    evaluation: dict[str, Any],
    best_history_row: dict[str, Any] | None,
) -> dict[str, Any]:
    overall = evaluation["metrics"]["overall"]
    return {
        "window_name": window_name,
        "model_name": model_name,
        "trial_id": trial_id,
        "dropout": float(params["dropout"]),
        "learning_rate": float(params["learning_rate"]),
        "weight_decay": float(params["weight_decay"]),
        "selection_score": float(evaluation["selection_score"]),
        "val_rmse": float(evaluation["val_rmse"]),
        "val_stage1_rmse": float(evaluation["val_stage1_rmse"]),
        "val_q90_mae": float(evaluation["val_q90_mae"]),
        "RMSE": overall.get("RMSE"),
        "MAE": overall.get("MAE"),
        "R2": overall.get("R2"),
        "Max_Error": overall.get("Max Error"),
        "amplitude_pred": evaluation["amplitude_pred"],
        "best_epoch": None if best_history_row is None else int(best_history_row["epoch"]),
    }


def _write_results_csv(path: str | Path, rows: list[dict[str, Any]]) -> Path:
    output_path = resolve_path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=RESULT_COLUMNS).to_csv(output_path, index=False, encoding="utf-8")
    return output_path


def _build_best_payload(
    window_name: str,
    model_name: str,
    best_row: dict[str, Any],
    best_history_row: dict[str, Any] | None,
) -> dict[str, Any]:
    payload = {
        "window_name": window_name,
        "model_name": model_name,
        "trial_id": int(best_row["trial_id"]),
        "params": {
            "dropout": float(best_row["dropout"]),
            "learning_rate": float(best_row["learning_rate"]),
            "weight_decay": float(best_row["weight_decay"]),
        },
        "validation_metrics": {
            "selection_score": float(best_row["selection_score"]),
            "val_rmse": float(best_row["val_rmse"]),
            "val_stage1_rmse": float(best_row["val_stage1_rmse"]),
            "val_q90_mae": float(best_row["val_q90_mae"]),
            "RMSE": None if pd.isna(best_row["RMSE"]) else float(best_row["RMSE"]),
            "MAE": None if pd.isna(best_row["MAE"]) else float(best_row["MAE"]),
            "R2": None if pd.isna(best_row["R2"]) else float(best_row["R2"]),
            "Max Error": None if pd.isna(best_row["Max_Error"]) else float(best_row["Max_Error"]),
            "amplitude_pred": None if pd.isna(best_row["amplitude_pred"]) else float(best_row["amplitude_pred"]),
        },
    }
    if best_history_row is not None:
        payload["best_epoch"] = int(best_history_row["epoch"])
    return payload


def main() -> None:
    setup_console_encoding()
    console = get_console()
    args = parse_args()

    base_config = load_config(args.config)
    window_experiment = normalize_window_selection(base_config, [args.window])[0]
    tuning_entry = _load_tuning_entry(base_config, str(window_experiment["name"]), args.model)

    ensure_project_dirs(base_config)
    check_environment(base_config, [args.model])

    window_config = apply_window_experiment(base_config, window_experiment)
    runtime_config = apply_model_window_profile(copy.deepcopy(window_config), args.model)
    resolve_path(tuning_entry["results_dir"]).mkdir(parents=True, exist_ok=True)

    set_global_seed(int(runtime_config["environment"]["seed"]))
    data = prepare_window_data(window_config)
    _validate_window_data(data, window_config)

    search_space = _iter_search_space(tuning_entry["search_space"], args.max_trials)
    if not search_space:
        raise ValueError("搜索空间为空，无法开始调参。")

    console.print(
        f"[bold cyan]Tuning[/bold cyan] "
        f"[{window_experiment['name']}][{args.model}] "
        f"trials={len(search_space)}"
    )

    results: list[dict[str, Any]] = []
    best_payload: dict[str, Any] | None = None
    best_score = float("inf")

    for trial_id, trial_params in enumerate(search_space, start=1):
        trial_config = _build_trial_config(runtime_config, args.model, trial_params)
        set_global_seed(int(trial_config["environment"]["seed"]))

        console.print(
            f"[{window_experiment['name']}][{args.model}] "
            f"Trial {trial_id}/{len(search_space)} "
            f"dropout={trial_config['models'][args.model]['dropout']:.3f} "
            f"lr={trial_config['models'][args.model]['learning_rate']:.6g} "
            f"wd={trial_config['models'][args.model].get('weight_decay', 0.0):.6g}"
        )

        model = _instantiate_model(args.model, trial_config)
        model.fit(data)
        history = getattr(model, "training_history", [])
        best_history_row = _extract_best_history_row(history)
        evaluation = _evaluate_validation(model, trial_config, data)

        result_row = _make_result_row(
            str(window_experiment["name"]),
            args.model,
            trial_id,
            {
                "dropout": trial_config["models"][args.model]["dropout"],
                "learning_rate": trial_config["models"][args.model]["learning_rate"],
                "weight_decay": trial_config["models"][args.model].get("weight_decay", 0.0),
            },
            evaluation,
            best_history_row,
        )
        results.append(result_row)
        _write_results_csv(tuning_entry["results_csv"], results)

        console.print(
            f"[{window_experiment['name']}][{args.model}] "
            f"selection={evaluation['selection_score']:.6f} "
            f"rmse={evaluation['val_rmse']:.6f} "
            f"stage1={evaluation['val_stage1_rmse']:.6f} "
            f"q90_mae={evaluation['val_q90_mae']:.6f}"
        )

        if float(evaluation["selection_score"]) < best_score:
            best_score = float(evaluation["selection_score"])
            best_payload = _build_best_payload(
                str(window_experiment["name"]),
                args.model,
                result_row,
                best_history_row,
            )
            dump_json(best_payload, tuning_entry["best_params_json"])

    if best_payload is None:
        raise RuntimeError("调参结束但没有得到有效结果。")

    console.print(
        f"[bold green]Best[/bold green] "
        f"[{best_payload['window_name']}][{best_payload['model_name']}] "
        f"selection={best_payload['validation_metrics']['selection_score']:.6f} "
        f"lr={best_payload['params']['learning_rate']:.6g} "
        f"dropout={best_payload['params']['dropout']:.3f} "
        f"wd={best_payload['params']['weight_decay']:.6g}"
    )


if __name__ == "__main__":
    main()
