from __future__ import annotations

from typing import Any

import pandas as pd

from utils.config import resolve_path
from utils.preprocess import FeatureMinMaxScaler, prepare_canonical_dataset, scale_splits, split_by_time
from utils.window import build_window_splits


def prepare_window_data(config: dict[str, Any]) -> dict[str, Any]:
    """执行训练前的完整数据准备流程。

    流程顺序固定为：字段规范化 -> 时间顺序划分 -> 仅训练集拟合归一化 ->
    在各 split 内独立构造 168->72 滑动窗口。这样可以避免验证集和测试集
    信息泄露到训练阶段。
    """
    canonical = prepare_canonical_dataset(config)
    splits_raw = split_by_time(canonical, config)
    splits_scaled, scaler = scale_splits(splits_raw, config)
    windows = build_window_splits(splits_scaled, config)

    data = {
        "canonical_df": canonical,
        "splits_raw": splits_raw,
        "splits_scaled": splits_scaled,
        "scaler": scaler,
        "feature_columns": config["data"]["model_input_features"],
        "target_column": config["data"]["target"],
        "timestamp_column": config["data"]["timestamp_column"],
    }
    data.update(windows)
    return data


def load_canonical_csv(config: dict[str, Any]) -> pd.DataFrame:
    """读取已经规范化后的小时级 CSV，主要供排查和后续分析使用。"""
    return pd.read_csv(resolve_path(config["paths"]["canonical_csv"]))


def load_scaler(config: dict[str, Any]) -> FeatureMinMaxScaler:
    """加载训练集拟合得到的 MinMaxScaler 参数。"""
    return FeatureMinMaxScaler.load(config["paths"]["scaler_json"])
