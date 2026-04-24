from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd

from utils.config import resolve_path


def create_sliding_windows(
    df: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    timestamp_column: str,
    input_window: int,
    output_window: int,
    step: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, int]]:
    """将连续时间序列切分为统一监督学习样本。

    单个输入样本形状为 (168, 6)，单个标签形状为 (72,)。
    函数同时返回 72 个目标时间戳，用于生成前端可读的 predictions.csv。
    """
    features = df[feature_columns].to_numpy(dtype=float)
    target = df[target_column].to_numpy(dtype=float)
    timestamps = pd.to_datetime(df[timestamp_column])
    timestamp_strings = timestamps.astype(str).to_numpy()

    X: list[np.ndarray] = []
    y: list[np.ndarray] = []
    y_timestamps: list[np.ndarray] = []
    skipped_nan = 0
    skipped_non_hourly = 0

    # 最大起点保证输入窗口和未来 72 小时标签都不会越界。
    max_start = len(df) - input_window - output_window + 1
    if max_start <= 0:
        return (
            np.empty((0, input_window, len(feature_columns)), dtype=np.float32),
            np.empty((0, output_window), dtype=np.float32),
            np.empty((0, output_window), dtype="<U32"),
            {"created": 0, "skipped_nan": 0, "skipped_non_hourly": 0},
        )

    # 168 小时输入 + 72 小时输出必须覆盖连续小时；时间断点窗口直接跳过。
    expected_span_hours = input_window + output_window - 1
    for start in range(0, max_start, step):
        input_end = start + input_window
        output_end = input_end + output_window

        window_times = timestamps.iloc[start:output_end]
        span = window_times.iloc[-1] - window_times.iloc[0]
        if span != pd.Timedelta(hours=expected_span_hours):
            skipped_non_hourly += 1
            continue

        X_window = features[start:input_end]
        y_window = target[input_end:output_end]
        # 输入或未来标签仍存在缺失时剔除该窗口，避免训练模型学习插值不充分的数据。
        if np.isnan(X_window).any() or np.isnan(y_window).any():
            skipped_nan += 1
            continue

        X.append(X_window.astype(np.float32))
        y.append(y_window.astype(np.float32))
        y_timestamps.append(timestamp_strings[input_end:output_end])

    created = len(X)
    return (
        np.asarray(X, dtype=np.float32),
        np.asarray(y, dtype=np.float32),
        np.asarray(y_timestamps),
        {
            "created": int(created),
            "skipped_nan": int(skipped_nan),
            "skipped_non_hourly": int(skipped_non_hourly),
        },
    )


def build_window_splits(
    scaled_splits: dict[str, pd.DataFrame],
    config: dict[str, Any],
) -> dict[str, Any]:
    """在 train/validation/test 内部分别构造窗口。

    不跨 split 构造窗口，确保训练窗口的输出不会进入验证集或测试集，
    验证窗口的输出也不会进入测试集。
    """
    feature_columns = config["data"]["model_input_features"]
    target_column = config["data"]["target"]
    timestamp_column = config["data"]["timestamp_column"]
    input_window = int(config["window"]["input_window_hours"])
    output_window = int(config["window"]["output_window_hours"])
    step = int(config["window"]["step_hours"])

    arrays: dict[str, Any] = {}
    log: dict[str, Any] = {
        "input_window": input_window,
        "output_window": output_window,
        "feature_columns": feature_columns,
        "target_column": target_column,
        "splits": {},
    }

    for split_name, split_df in scaled_splits.items():
        X, y, target_timestamps, stats = create_sliding_windows(
            split_df,
            feature_columns,
            target_column,
            timestamp_column,
            input_window,
            output_window,
            step,
        )
        arrays[f"X_{split_name}"] = X
        arrays[f"y_{split_name}"] = y
        arrays[f"timestamps_{split_name}"] = target_timestamps
        log["splits"][split_name] = {
            **stats,
            "rows": int(len(split_df)),
            "X_shape": list(X.shape),
            "y_shape": list(y.shape),
        }

    npz_path = resolve_path(config["paths"]["windows_npz"])
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(npz_path, **arrays)

    log_path = resolve_path(config["paths"]["window_log_json"])
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, indent=2)

    arrays["window_log"] = log
    return arrays
