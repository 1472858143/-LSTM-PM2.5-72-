from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


# =========================
# 配置区：按需修改
# =========================
INPUT_FILES = [
    "23-01-01 to 23-12-31.csv",
    "24-01-01 to 24-12-31.csv",
    "25-01-01 to 25-08-24.csv",
]

OUTPUT_DIR = "processed_noaa"
RAW_MERGED_FILE = "noaa_beijing_merged_raw.csv"
FINAL_FILE = "noaa_beijing_weather_final.csv"

# 北京时区：NOAA DATE 通常为 UTC
TIMEZONE_OFFSET_HOURS = 8

# 是否只保留整点数据
KEEP_HOURLY_ONLY = True

# 是否按小时重采样
RESAMPLE_TO_HOURLY = True

# 缺失值标记规则
NOAA_MISSING_INT = {
    9999,
    99999,
    999999,
    9999999,
    -9999,
    -99999,
    -999999,
    -9999999,
}


# =========================
# 基础解析函数
# =========================
def ensure_output_dir(path: str | Path) -> Path:
    output_dir = Path(path)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def safe_split(value: object, sep: str = ",") -> list[str]:
    if pd.isna(value):
        return []
    return str(value).split(sep)


def parse_signed_tenths(field: object) -> Optional[float]:
    """
    NOAA TMP / DEW 示例：
    -0059,1  -> -5.9
    +0130,1  -> 13.0
    """
    parts = safe_split(field)
    if not parts:
        return np.nan

    raw = parts[0].strip()
    if raw in {"+9999", "-9999", "9999"}:
        return np.nan

    try:
        value = int(raw)
    except ValueError:
        return np.nan

    if value in NOAA_MISSING_INT:
        return np.nan

    return value / 10.0


def parse_wind_speed(field: object) -> Optional[float]:
    """
    NOAA WND 示例：
    060,1,N,0010,1
    第4段是风速，单位通常为 0.1 m/s
    """
    parts = safe_split(field)
    if len(parts) < 4:
        return np.nan

    raw = parts[3].strip()
    if raw == "9999":
        return np.nan

    try:
        value = int(raw)
    except ValueError:
        return np.nan

    if value in NOAA_MISSING_INT:
        return np.nan

    return value / 10.0


def parse_sea_level_pressure(field: object) -> Optional[float]:
    """
    NOAA SLP 示例：
    10365,1 -> 1036.5 hPa
    """
    parts = safe_split(field)
    if not parts:
        return np.nan

    raw = parts[0].strip()
    if raw == "99999":
        return np.nan

    try:
        value = int(raw)
    except ValueError:
        return np.nan

    if value in NOAA_MISSING_INT:
        return np.nan

    return value / 10.0


def parse_surface_pressure_from_ma1(field: object) -> Optional[float]:
    """
    NOAA MA1 示例：
    99999,9,10320,1

    这里按经验使用第3段作为站点/地面气压候选值，单位 0.1 hPa。
    若缺失则返回 NaN。
    """
    parts = safe_split(field)
    if len(parts) < 3:
        return np.nan

    raw = parts[2].strip()
    if raw == "99999":
        return np.nan

    try:
        value = int(raw)
    except ValueError:
        return np.nan

    if value in NOAA_MISSING_INT:
        return np.nan

    return value / 10.0


def parse_precip_equivalent_mm_per_hour(*fields: object) -> Optional[float]:
    """
    NOAA 降水常见在 AA1 / AA2 / AA3 中。
    示例：
    AA1: 06,0000,9,1
    AA2: 12,0000,9,1
    AA3: 24,0000,9,1

    规则：
    1. 从多个字段中选择“有效且 period 最短”的累计降水
    2. amount 通常是 0.1 mm
    3. 转成等效每小时降水量： (amount_mm / period_hours)

    注意：
    这不是严格的逐小时观测降水，而是“等效小时降水率”
    """
    candidates: list[tuple[int, float]] = []

    for field in fields:
        parts = safe_split(field)
        if len(parts) < 2:
            continue

        period_raw = parts[0].strip()
        amount_raw = parts[1].strip()

        if period_raw in {"99", "98"} or amount_raw in {"9999", "9998"}:
            continue

        try:
            period_hours = int(period_raw)
            amount_tenths_mm = int(amount_raw)
        except ValueError:
            continue

        if period_hours <= 0:
            continue
        if amount_tenths_mm in NOAA_MISSING_INT:
            continue

        amount_mm = amount_tenths_mm / 10.0
        hourly_equivalent = amount_mm / period_hours
        candidates.append((period_hours, hourly_equivalent))

    if not candidates:
        return np.nan

    # 选 period 最短的，保留时间分辨率相对更高的记录
    candidates.sort(key=lambda x: x[0])
    return candidates[0][1]


def calc_relative_humidity(temp_c: pd.Series, dew_c: pd.Series) -> pd.Series:
    """
    由温度和露点计算相对湿度 RH(%)
    Magnus 公式
    """
    a = 17.625
    b = 243.04

    es = np.exp((a * temp_c) / (b + temp_c))
    ed = np.exp((a * dew_c) / (b + dew_c))
    rh = 100.0 * (ed / es)

    # 限制在合理范围
    rh = rh.clip(lower=0, upper=100)
    return rh


# =========================
# 主处理流程
# =========================
def load_and_merge_csv(files: list[str]) -> pd.DataFrame:
    frames = []
    for file in files:
        path = Path(file)
        if not path.exists():
            raise FileNotFoundError(f"未找到文件: {file}")

        df = pd.read_csv(path, dtype=str)
        df["source_file"] = path.name
        frames.append(df)

    merged = pd.concat(frames, ignore_index=True)
    return merged


def preprocess_datetime(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["timestamp"] = pd.to_datetime(df["DATE"], errors="coerce", utc=True).dt.tz_convert("Asia/Shanghai")
    df = df.dropna(subset=["timestamp"])

    

    if KEEP_HOURLY_ONLY:
        df = df[df["timestamp"].dt.minute.eq(0) & df["timestamp"].dt.second.eq(0)]

    return df


def parse_core_fields(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["temperature_2m"] = df["TMP"].apply(parse_signed_tenths)
    df["dew_point_2m"] = df["DEW"].apply(parse_signed_tenths)
    df["wind_speed_10m"] = df["WND"].apply(parse_wind_speed)

    df["sea_level_pressure"] = df["SLP"].apply(parse_sea_level_pressure)
    df["surface_pressure"] = df["MA1"].apply(parse_surface_pressure_from_ma1)
    # 若 surface_pressure 缺失，则退回 sea_level_pressure
    df["surface_pressure"] = df["surface_pressure"].fillna(df["sea_level_pressure"])
    df["surface_pressure"] = df["surface_pressure"].interpolate()
    df["precipitation"] = df.apply(
        lambda row: parse_precip_equivalent_mm_per_hour(
            row.get("AA1"), row.get("AA2"), row.get("AA3")
        ),
        axis=1,
    )
    df["precipitation"] = df["precipitation"].fillna(0)

    df["relative_humidity_2m"] = calc_relative_humidity(
        df["temperature_2m"], df["dew_point_2m"]
    )

    return df


def select_and_clean(df: pd.DataFrame) -> pd.DataFrame:
    keep_cols = [
        "timestamp",
        "temperature_2m",
        "dew_point_2m",
        "relative_humidity_2m",
        "wind_speed_10m",
        "precipitation",
        "surface_pressure",
        "sea_level_pressure",
        "STATION",
        "NAME",
        "LATITUDE",
        "LONGITUDE",
        "ELEVATION",
        "source_file",
    ]

    result = df[keep_cols].copy()

    # 类型统一
    numeric_cols = [
        "temperature_2m",
        "dew_point_2m",
        "relative_humidity_2m",
        "wind_speed_10m",
        "precipitation",
        "surface_pressure",
        "sea_level_pressure",
        "LATITUDE",
        "LONGITUDE",
        "ELEVATION",
    ]
    for col in numeric_cols:
        result[col] = pd.to_numeric(result[col], errors="coerce")

    # 基础物理范围过滤
    result.loc[(result["relative_humidity_2m"] < 0) | (result["relative_humidity_2m"] > 100), "relative_humidity_2m"] = np.nan
    result.loc[result["wind_speed_10m"] < 0, "wind_speed_10m"] = np.nan
    result.loc[result["precipitation"] < 0, "precipitation"] = np.nan
    result.loc[result["surface_pressure"] < 800, "surface_pressure"] = np.nan
    result.loc[result["surface_pressure"] > 1100, "surface_pressure"] = np.nan
    result.loc[result["temperature_2m"] < -80, "temperature_2m"] = np.nan
    result.loc[result["temperature_2m"] > 60, "temperature_2m"] = np.nan

    # 去重：同一时间保留第一条
    result = result.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="first")

    return result


def resample_hourly(df: pd.DataFrame) -> pd.DataFrame:
    if not RESAMPLE_TO_HOURLY:
        return df

    df = df.copy().set_index("timestamp").sort_index()

    agg_map = {
        "temperature_2m": "mean",
        "dew_point_2m": "mean",
        "relative_humidity_2m": "mean",
        "wind_speed_10m": "mean",
        "precipitation": "mean",
        "surface_pressure": "mean",
        "sea_level_pressure": "mean",
        "STATION": "first",
        "NAME": "first",
        "LATITUDE": "first",
        "LONGITUDE": "first",
        "ELEVATION": "first",
        "source_file": "first",
    }

    hourly = df.resample("1h").agg(agg_map).reset_index()
    meta_cols = ["STATION", "NAME", "LATITUDE", "LONGITUDE", "ELEVATION", "source_file"]
    hourly[meta_cols] = hourly[meta_cols].ffill().bfill()
    return hourly


def add_project_alias_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    增加更贴近项目文档命名的别名列
    """
    df = df.copy()
    df["humidity"] = df["relative_humidity_2m"]
    return df


def save_outputs(raw_df: pd.DataFrame, final_df: pd.DataFrame, output_dir: Path) -> None:
    raw_path = output_dir / RAW_MERGED_FILE
    final_path = output_dir / FINAL_FILE

    raw_df.to_csv(raw_path, index=False, encoding="utf-8-sig")
    final_df.to_csv(final_path, index=False, encoding="utf-8-sig")

    print(f"已保存原始合并文件: {raw_path}")
    print(f"已保存最终结果文件: {final_path}")


def print_summary(df: pd.DataFrame) -> None:
    print("\n===== 数据摘要 =====")
    print(f"总行数: {len(df):,}")
    if not df.empty:
        print(f"时间范围: {df['timestamp'].min()}  ->  {df['timestamp'].max()}")
    print("\n缺失率：")
    cols = [
        "temperature_2m",
        "dew_point_2m",
        "relative_humidity_2m",
        "wind_speed_10m",
        "precipitation",
        "surface_pressure",
    ]
    for col in cols:
        missing_rate = df[col].isna().mean() * 100
        print(f"  {col}: {missing_rate:.2f}%")

    print("\n最终建议建模字段：")
    print([
        "timestamp",
        "temperature_2m",
        "humidity",
        "wind_speed_10m",
        "precipitation",
        "surface_pressure",
    ])


def main() -> None:
    output_dir = ensure_output_dir(OUTPUT_DIR)

    # 1. 读取并合并
    raw_df = load_and_merge_csv(INPUT_FILES)

    # 2. 处理时间
    dt_df = preprocess_datetime(raw_df)

    # 3. 解析字段
    parsed_df = parse_core_fields(dt_df)

    # 4. 选择列并清洗
    clean_df = select_and_clean(parsed_df)

    # 5. 按小时重采样
    hourly_df = resample_hourly(clean_df)

    # 6. 添加项目别名列
    final_df = add_project_alias_columns(hourly_df)

    # 7. 保存
    save_outputs(raw_df, final_df, output_dir)

    # 8. 打印摘要
    print_summary(final_df)


if __name__ == "__main__":
    main()