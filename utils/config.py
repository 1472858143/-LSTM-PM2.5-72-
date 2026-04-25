from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Iterable


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_path(path_value: str | Path, root: Path | None = None) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (root or project_root()) / path


def load_config(config_path: str | Path = "config/config.json") -> dict[str, Any]:
    path = resolve_path(config_path)
    with path.open("r", encoding="utf-8") as f:
        config = json.load(f)
    config["_config_path"] = str(path)
    config["_project_root"] = str(project_root())
    return config


def dump_json(data: Any, path: str | Path, indent: int = 2) -> None:
    output_path = resolve_path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)


def _deep_merge_dict(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge_dict(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def get_window_experiments(config: dict[str, Any]) -> list[dict[str, Any]]:
    experiments = config.get("window_experiments")
    if experiments:
        enabled = [item for item in experiments if item.get("enabled", True)]
        if enabled:
            return enabled

    return [
        {
            "name": "default_window",
            "input_window_hours": int(config["window"]["input_window_hours"]),
            "output_window_hours": int(config["window"]["output_window_hours"]),
            "enabled": True,
            "legacy_mode": True,
        }
    ]


def normalize_window_selection(
    config: dict[str, Any],
    windows: Iterable[str] | str | None,
) -> list[dict[str, Any]]:
    experiments = get_window_experiments(config)
    name_to_experiment = {item["name"]: item for item in experiments}

    if windows is None:
        selected_names = list(name_to_experiment.keys())
    elif isinstance(windows, str):
        selected_names = list(name_to_experiment.keys()) if windows == "all" else [windows]
    else:
        selected_names = list(windows)
        if selected_names == ["all"]:
            selected_names = list(name_to_experiment.keys())

    invalid = [name for name in selected_names if name not in name_to_experiment]
    if invalid:
        raise ValueError(f"窗口实验名称不在允许列表中: {invalid}")
    return [name_to_experiment[name] for name in selected_names]


def apply_window_experiment(config: dict[str, Any], experiment: dict[str, Any]) -> dict[str, Any]:
    window_name = str(experiment["name"])
    input_window = int(experiment["input_window_hours"])
    output_window = int(experiment["output_window_hours"])
    legacy_mode = bool(experiment.get("legacy_mode", False))

    config_copy = copy.deepcopy(config)
    config_copy["window"]["input_window_hours"] = input_window
    config_copy["window"]["output_window_hours"] = output_window
    config_copy["window"]["input_shape"] = [input_window, int(config_copy["global_constraints"]["feature_count"])]
    config_copy["window"]["output_shape"] = [output_window]
    config_copy["global_constraints"]["input_window_hours"] = input_window
    config_copy["global_constraints"]["output_window_hours"] = output_window
    config_copy["_active_window_name"] = window_name
    config_copy["_legacy_single_window_mode"] = legacy_mode

    processed_window_dir = Path(config_copy["paths"]["processed_dir"]) / window_name
    config_copy["paths"]["windows_npz"] = str(processed_window_dir / "windows.npz")
    config_copy["paths"]["window_log_json"] = str(processed_window_dir / "window_log.json")

    if not legacy_mode:
        outputs_root = Path(config_copy["outputs"]["root"])
        config_copy["outputs"]["model_dirs"] = {
            model_name: str(outputs_root / window_name / model_name)
            for model_name in config_copy["models"]["allowed_model_names"]
        }
        config_copy["models"]["attention_lstm"]["attention_weights_path"] = str(
            outputs_root / window_name / "attention_lstm" / "attention_weights.npy"
        )

    return config_copy


def apply_model_window_profile(config: dict[str, Any], model_name: str) -> dict[str, Any]:
    active_window = str(config.get("_active_window_name", ""))
    model_cfg = config["models"][model_name]
    window_profiles = model_cfg.get("window_profiles", {})
    if not active_window or active_window not in window_profiles:
        return config

    config["models"][model_name] = _deep_merge_dict(model_cfg, window_profiles[active_window])
    return config


def ensure_project_dirs(config: dict[str, Any]) -> None:
    paths = config["paths"]
    for key in ["processed_dir", "outputs_root", "metrics_summary_dir"]:
        resolve_path(paths[key]).mkdir(parents=True, exist_ok=True)

    for experiment in get_window_experiments(config):
        experiment_config = apply_window_experiment(config, experiment)
        resolve_path(experiment_config["paths"]["windows_npz"]).parent.mkdir(parents=True, exist_ok=True)
        for model_dir in experiment_config["outputs"]["model_dirs"].values():
            (resolve_path(model_dir) / "plots").mkdir(parents=True, exist_ok=True)
