from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from utils.config import resolve_path
from utils.runtime import active_window_name


def model_output_dir(config: dict[str, Any], model_name: str) -> Path:
    """Return the model output directory for the active window."""
    return resolve_path(config["outputs"]["model_dirs"][model_name])


def prepare_model_output_dir(config: dict[str, Any], model_name: str) -> Path:
    """Ensure the model output directory and plots subdirectory exist."""
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
    """Save the unified predictions.csv artifact."""
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
    """Save metrics.json."""
    output_dir = prepare_model_output_dir(config, model_name)
    path = output_dir / "metrics.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    return path


def save_config_snapshot(config: dict[str, Any], model_name: str) -> Path:
    """Save a snapshot of the current runtime config."""
    output_dir = prepare_model_output_dir(config, model_name)
    path = output_dir / "config_snapshot.json"
    serializable = {k: v for k, v in config.items() if not k.startswith("_")}
    with path.open("w", encoding="utf-8") as f:
        json.dump(serializable, f, ensure_ascii=False, indent=2)
    return path


def copy_metrics_to_summary(config: dict[str, Any], model_name: str) -> None:
    """Copy per-model metrics.json into the summary directory."""
    src = model_output_dir(config, model_name) / "metrics.json"
    dst_dir = resolve_path(config["paths"]["metrics_summary_dir"])
    dst_dir.mkdir(parents=True, exist_ok=True)
    if src.exists():
        window_name = active_window_name(config)
        shutil.copy2(src, dst_dir / f"{window_name}_{model_name}_metrics.json")


def save_metrics_tables(config: dict[str, Any], model_name: str, metrics: dict[str, Any]) -> None:
    """Save stage and horizon CSV tables."""
    output_dir = prepare_model_output_dir(config, model_name)
    stage_rows = [{"stage": stage, **values} for stage, values in metrics["stages"].items()]
    pd.DataFrame(stage_rows).to_csv(output_dir / "stage_metrics.csv", index=False, encoding="utf-8")
    pd.DataFrame(metrics["horizon"]).to_csv(output_dir / "horizon_metrics.csv", index=False, encoding="utf-8")


def save_training_history(config: dict[str, Any], model_name: str, history: list[dict[str, Any]]) -> None:
    """Save training history JSON/CSV artifacts."""
    if not history:
        return
    output_dir = prepare_model_output_dir(config, model_name)
    with (output_dir / "training_history.json").open("w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
    pd.DataFrame(history).to_csv(output_dir / "training_history.csv", index=False, encoding="utf-8")


def _attention_segment_stats(weights: np.ndarray) -> dict[str, Any]:
    array = np.asarray(weights, dtype=float)
    entropy = -(array * np.log(np.clip(array, 1e-12, None))).sum(axis=1)
    sorted_weights = np.sort(array, axis=1)[:, ::-1]
    uniform_entropy = float(np.log(array.shape[1]))
    mean_weights = array.mean(axis=0)
    top_indices = np.argsort(mean_weights)[-5:][::-1]
    segment_length = max(array.shape[1] // 10, 1)
    first_10pct_mass = float(array[:, :segment_length].sum(axis=1).mean())
    last_10pct_mass = float(array[:, -segment_length:].sum(axis=1).mean())
    index_axis = np.arange(array.shape[1], dtype=float)
    index_std = float(index_axis.std())
    centered_index = index_axis - index_axis.mean()
    correlations: list[float] = []
    for row in array:
        row_std = float(row.std())
        if row_std <= 1e-12 or index_std <= 1e-12:
            correlations.append(0.0)
            continue
        centered_row = row - row.mean()
        corr = float(np.mean(centered_index * centered_row) / (index_std * row_std))
        correlations.append(corr)
    return {
        "shape": list(array.shape),
        "uniform_weight": float(1.0 / array.shape[1]),
        "mean": float(array.mean()),
        "std": float(array.std()),
        "min": float(array.min()),
        "max": float(array.max()),
        "top_1_weight": float(sorted_weights[:, 0].mean()),
        "top_5_weight_sum": float(sorted_weights[:, :5].sum(axis=1).mean()),
        "top_10_weight_sum": float(sorted_weights[:, :10].sum(axis=1).mean()),
        "max_per_sample_mean": float(array.max(axis=1).mean()),
        "max_per_sample_p95": float(np.percentile(array.max(axis=1), 95)),
        "entropy_mean": float(entropy.mean()),
        "entropy_std": float(entropy.std()),
        "uniform_entropy": uniform_entropy,
        "entropy_ratio_to_uniform": float(entropy.mean() / uniform_entropy),
        "near_uniform": bool(entropy.mean() / uniform_entropy > 0.999),
        "first_10pct_mass": first_10pct_mass,
        "last_10pct_mass": last_10pct_mass,
        "index_correlation": float(np.mean(correlations)) if correlations else 0.0,
        "top_mean_weight_steps": [
            {"input_step": int(index + 1), "mean_weight": float(mean_weights[index])}
            for index in top_indices
        ],
    }


def save_execution_log(
    config: dict[str, Any],
    model_name: str,
    execution_log: dict[str, Any],
    history: list[dict[str, Any]] | None = None,
) -> Path:
    """Save the runtime training_log.json artifact."""
    output_dir = prepare_model_output_dir(config, model_name)
    log_payload = dict(execution_log)

    if history:
        best_rows = [row for row in history if bool(row.get("is_best_epoch", False))]
        best_row = best_rows[-1] if best_rows else min(history, key=lambda row: float(row["validation_loss"]))
        log_payload.update(
            {
                "best_epoch": int(best_row["epoch"]),
                "best_validation_loss": float(best_row["validation_loss"]),
                "early_stopping_epoch": int(history[-1]["epoch"]),
                "epochs_completed": int(len(history)),
            }
        )
        for key in [
            "val_rmse",
            "val_stage1_rmse",
            "val_q80_mae",
            "val_q90_mae",
            "val_h1_rmse",
            "best_val_rmse",
            "best_val_q80_mae",
            "best_val_q90_mae",
            "best_val_h1_rmse",
            "selection_score",
            "best_selection_score",
            "checkpoint_metric",
        ]:
            if key not in best_row:
                continue
            value = best_row[key]
            log_payload[key] = float(value) if isinstance(value, (int, float)) else value

    path = output_dir / "training_log.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(log_payload, f, ensure_ascii=False, indent=2)
    return path


def save_attention_stats(
    config: dict[str, Any],
    model_name: str,
    attention_weights: np.ndarray,
    diagnostics: dict[str, np.ndarray] | None = None,
) -> Path:
    """Save attention summary statistics."""
    output_dir = prepare_model_output_dir(config, model_name)
    stats = _attention_segment_stats(np.asarray(attention_weights, dtype=float))
    stats["combined_first_10pct_mass"] = stats["first_10pct_mass"]
    stats["combined_last_10pct_mass"] = stats["last_10pct_mass"]

    if diagnostics:
        branch_profiles = diagnostics.get("branch_profiles")
        branch_gates = diagnostics.get("branch_gates")
        gate = diagnostics.get("gate")
        raw_gate = diagnostics.get("raw_gate")
        global_profile = diagnostics.get("global_profile")
        recent_profile = diagnostics.get("recent_profile")
        combined_profile = diagnostics.get("combined_profile")

        if gate is not None:
            gate_arr = np.asarray(gate, dtype=float).reshape(-1)
            stats["gate_mean"] = float(gate_arr.mean())
            stats["gate_std"] = float(gate_arr.std())
        if raw_gate is not None:
            raw_gate_arr = np.asarray(raw_gate, dtype=float).reshape(-1)
            stats["raw_gate_mean"] = float(raw_gate_arr.mean())
            stats["raw_gate_std"] = float(raw_gate_arr.std())
        if global_profile is not None:
            global_stats = _attention_segment_stats(np.asarray(global_profile, dtype=float))
            stats["global_first_10pct_mass"] = float(global_stats["first_10pct_mass"])
            stats["global_last_10pct_mass"] = float(global_stats["last_10pct_mass"])
        if recent_profile is not None:
            recent_stats = _attention_segment_stats(np.asarray(recent_profile, dtype=float))
            stats["recent_first_10pct_mass"] = float(recent_stats["first_10pct_mass"])
            stats["recent_last_10pct_mass"] = float(recent_stats["last_10pct_mass"])
        if combined_profile is not None:
            combined_stats = _attention_segment_stats(np.asarray(combined_profile, dtype=float))
            stats["combined_first_10pct_mass"] = float(combined_stats["first_10pct_mass"])
            stats["combined_last_10pct_mass"] = float(combined_stats["last_10pct_mass"])
        if isinstance(branch_gates, dict):
            stats["branch_gate_mean"] = {}
            stats["branch_gate_std"] = {}
            for name, values in branch_gates.items():
                gate_arr = np.asarray(values, dtype=float).reshape(-1)
                stats["branch_gate_mean"][name] = float(gate_arr.mean())
                stats["branch_gate_std"][name] = float(gate_arr.std())
        if isinstance(branch_profiles, dict):
            for name, values in branch_profiles.items():
                branch_stats = _attention_segment_stats(np.asarray(values, dtype=float))
                stats[f"{name}_first_10pct_mass"] = float(branch_stats["first_10pct_mass"])
                stats[f"{name}_last_10pct_mass"] = float(branch_stats["last_10pct_mass"])

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
    """Save top-peak prediction cases for qualitative inspection."""
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


def save_metrics_summary_tables(
    config: dict[str, Any],
    window_rows: list[dict[str, Any]],
    stage_rows: list[dict[str, Any]],
    horizon_rows: list[dict[str, Any]],
) -> None:
    """Generate cross-window summary CSV tables."""
    summary_dir = resolve_path(config["paths"]["metrics_summary_dir"])
    summary_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(window_rows).to_csv(summary_dir / "window_model_metrics.csv", index=False, encoding="utf-8")
    pd.DataFrame(stage_rows).to_csv(summary_dir / "stage_metrics_summary.csv", index=False, encoding="utf-8")
    pd.DataFrame(horizon_rows).to_csv(summary_dir / "horizon_metrics_summary.csv", index=False, encoding="utf-8")
