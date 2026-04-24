from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from utils.config import resolve_path


@dataclass
class FeatureMinMaxScaler:
    """项目内部 MinMaxScaler。

    只在训练集上 fit，再应用到验证集和测试集，避免测试集统计量泄露。
    额外提供目标列反归一化方法，确保最终指标在真实 PM2.5 单位上计算。
    """

    feature_columns: list[str]
    target_column: str
    data_min_: dict[str, float]
    data_max_: dict[str, float]

    @classmethod
    def fit(
        cls,
        df: pd.DataFrame,
        feature_columns: list[str],
        target_column: str,
    ) -> "FeatureMinMaxScaler":
        data_min = df[feature_columns].min(skipna=True).to_dict()
        data_max = df[feature_columns].max(skipna=True).to_dict()
        return cls(feature_columns, target_column, data_min, data_max)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()
        for column in self.feature_columns:
            min_value = self.data_min_[column]
            max_value = self.data_max_[column]
            # 常数列使用极小分母保护，避免除零导致训练样本出现 inf。
            denominator = max(max_value - min_value, 1e-12)
            result[column] = (result[column] - min_value) / denominator
        return result

    def inverse_transform_target(self, values: np.ndarray) -> np.ndarray:
        min_value = self.data_min_[self.target_column]
        max_value = self.data_max_[self.target_column]
        denominator = max(max_value - min_value, 1e-12)
        return np.asarray(values, dtype=float) * denominator + min_value

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": "MinMaxScaler",
            "feature_columns": self.feature_columns,
            "target_column": self.target_column,
            "data_min": self.data_min_,
            "data_max": self.data_max_,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FeatureMinMaxScaler":
        return cls(
            list(data["feature_columns"]),
            data["target_column"],
            {k: float(v) for k, v in data["data_min"].items()},
            {k: float(v) for k, v in data["data_max"].items()},
        )

    def save(self, path: str | Path) -> None:
        output_path = resolve_path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "FeatureMinMaxScaler":
        with resolve_path(path).open("r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))


def _standardize_columns(columns: list[str]) -> dict[str, str]:
    return {column: column.strip() for column in columns}


def _rename_to_canonical(df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """把原始 CSV 字段映射为项目书规定的规范字段。

    这里仅做字段重命名和列筛选，不新增任何派生特征，保证输入特征仍为 6 个。
    """
    mapping = config["data"]["raw_to_canonical_field_mapping"]
    lowered_mapping = {key.strip().lower(): value for key, value in mapping.items()}
    rename_map: dict[str, str] = {}
    for column in df.columns:
        key = column.strip().lower()
        if key in lowered_mapping:
            rename_map[column] = lowered_mapping[key]

    renamed = df.rename(columns=rename_map)
    canonical_columns = config["data"]["canonical_columns"]
    missing = [column for column in canonical_columns if column not in renamed.columns]
    if missing:
        raise ValueError(f"输入 CSV 缺少必需字段: {missing}")
    return renamed[canonical_columns].copy()


def _mark_outliers_as_missing(df: pd.DataFrame, config: dict[str, Any]) -> dict[str, int]:
    """按物理约束标记明显异常值。

    极端但连续的 PM2.5 高值可能是真实污染过程，因此这里只处理违反物理意义的值，
    后续再由缺失值规则决定是否填补或在窗口阶段剔除。
    """
    rules = config["preprocessing"]["outliers"]
    counts: dict[str, int] = {}

    checks = {
        "pm2_5": df["pm2_5"] < rules["pm2_5_min"],
        "humidity": (df["humidity"] < rules["humidity_min"]) | (df["humidity"] > rules["humidity_max"]),
        "wind_speed_10m": df["wind_speed_10m"] < rules["wind_speed_10m_min"],
        "precipitation": df["precipitation"] < rules["precipitation_min"],
        "temperature_2m": (df["temperature_2m"] < rules["temperature_2m_min"])
        | (df["temperature_2m"] > rules["temperature_2m_max"]),
        "surface_pressure": (df["surface_pressure"] < rules["surface_pressure_min"])
        | (df["surface_pressure"] > rules["surface_pressure_max"]),
    }

    for column, mask in checks.items():
        counts[column] = int(mask.sum())
        df.loc[mask, column] = np.nan
    return counts


def _fill_gaps_with_limit(series: pd.Series, max_gap: int) -> pd.Series:
    """只填补不超过配置上限的连续缺失。

    长时间缺失不强行插值，保留 NaN 给窗口构造阶段剔除，避免模型学习不可靠标签。
    """
    missing = series.isna()
    if not missing.any():
        return series

    interpolated = series.interpolate(method="linear", limit_direction="both")
    result = series.copy()
    group_id = (missing != missing.shift(fill_value=False)).cumsum()
    for _, group_mask in missing.groupby(group_id):
        if not bool(group_mask.iloc[0]):
            continue
        indices = group_mask.index[group_mask.to_numpy()]
        if len(indices) <= max_gap:
            result.loc[indices] = interpolated.loc[indices]
    return result


def prepare_canonical_dataset(config: dict[str, Any]) -> pd.DataFrame:
    """生成规范小时级数据文件。

    输出列固定为 timestamp、pm2_5 和 5 个气象变量；时间轴重建为小时频率，
    用处理日志记录缺失、异常和插入时间戳数量，方便论文复现实验过程。
    """
    paths = config["paths"]
    raw_path = resolve_path(paths["raw_input_csv"])
    output_path = resolve_path(paths["canonical_csv"])
    log_path = resolve_path(paths["preprocess_log_json"])

    # 原始文件可能沿用旧字段名，例如 pm25/temp/pressure，先统一映射到规范字段。
    raw = pd.read_csv(raw_path)
    raw = raw.rename(columns=_standardize_columns(list(raw.columns)))
    df = _rename_to_canonical(raw, config)
    timestamp_column = config["data"]["timestamp_column"]
    numeric_columns = [column for column in config["data"]["canonical_columns"] if column != timestamp_column]

    # 时间戳是时间顺序划分和滑动窗口的依据，无效时间戳不能参与训练。
    df[timestamp_column] = pd.to_datetime(df[timestamp_column], errors="coerce")
    invalid_timestamps = int(df[timestamp_column].isna().sum())
    df = df.dropna(subset=[timestamp_column])

    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    before_duplicate_count = len(df)
    df = df.sort_values(timestamp_column)
    # 同一小时出现重复记录时取均值，保持每个 timestamp 唯一。
    df = df.groupby(timestamp_column, as_index=False)[numeric_columns].mean()
    duplicate_rows_removed = before_duplicate_count - len(df)

    outlier_counts = _mark_outliers_as_missing(df, config)
    missing_before_fill = df[numeric_columns].isna().sum().to_dict()

    df = df.set_index(timestamp_column).sort_index()
    full_index = pd.date_range(
        start=df.index.min(),
        end=df.index.max(),
        freq=config["preprocessing"]["hourly_frequency"],
    )
    inserted_missing_timestamps = len(full_index) - len(df.index)
    # 重建完整小时索引后，原始数据中的时间断点会显式变为 NaN。
    df = df.reindex(full_index)
    df.index.name = timestamp_column

    max_gap = int(config["preprocessing"]["max_fill_gap_hours"])
    for column in numeric_columns:
        df[column] = _fill_gaps_with_limit(df[column], max_gap=max_gap)

    missing_after_fill = df[numeric_columns].isna().sum().to_dict()
    df = df.reset_index()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8")

    log = {
        "raw_path": str(raw_path),
        "canonical_path": str(output_path),
        "raw_rows": int(len(raw)),
        "canonical_rows": int(len(df)),
        "invalid_timestamps_dropped": invalid_timestamps,
        "duplicate_rows_removed": int(duplicate_rows_removed),
        "inserted_missing_timestamps": int(inserted_missing_timestamps),
        "outliers_marked_as_missing": outlier_counts,
        "missing_before_fill": {k: int(v) for k, v in missing_before_fill.items()},
        "missing_after_fill": {k: int(v) for k, v in missing_after_fill.items()},
        "canonical_columns": config["data"]["canonical_columns"],
    }
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, indent=2)

    return df


def split_by_time(df: pd.DataFrame, config: dict[str, Any]) -> dict[str, pd.DataFrame]:
    """按时间顺序划分训练、验证和测试集。

    时间序列预测禁止随机划分和 shuffle，否则未来信息可能泄露到训练集。
    """
    split_cfg = config["split"]
    if split_cfg.get("shuffle", False):
        raise ValueError("时间序列划分禁止 shuffle。")

    n_rows = len(df)
    train_end = int(n_rows * float(split_cfg["train_ratio"]))
    val_end = train_end + int(n_rows * float(split_cfg["validation_ratio"]))

    return {
        "train": df.iloc[:train_end].reset_index(drop=True),
        "validation": df.iloc[train_end:val_end].reset_index(drop=True),
        "test": df.iloc[val_end:].reset_index(drop=True),
    }


def scale_splits(
    splits: dict[str, pd.DataFrame],
    config: dict[str, Any],
) -> tuple[dict[str, pd.DataFrame], FeatureMinMaxScaler]:
    """对三个 split 做 MinMax 归一化。

    scaler 只使用训练集统计量 fit，验证集和测试集只 transform，符合实验公平性要求。
    """
    feature_columns = config["data"]["model_input_features"]
    target_column = config["data"]["target"]
    scaler = FeatureMinMaxScaler.fit(splits["train"], feature_columns, target_column)
    scaled = {name: scaler.transform(split) for name, split in splits.items()}

    if config["preprocessing"]["scaler"].get("save_scaler", True):
        scaler.save(config["paths"]["scaler_json"])

    return scaled, scaler
