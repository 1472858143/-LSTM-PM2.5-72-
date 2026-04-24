#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
功能：
读取 pm25_raw 文件夹中所有 .csv 文件，直接合并为一个文件
输出文件名：merge_pm25.csv
（不做任何字段处理、不筛选、不改时间）
"""

from pathlib import Path
import pandas as pd


def main():
    input_dir = Path("pm25_raw")
    output_file = Path("merge_pm25.csv")

    if not input_dir.exists():
        raise FileNotFoundError("pm25_raw 文件夹不存在")

    csv_files = list(input_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError("pm25_raw 中没有 .csv 文件")

    all_data = []

    for file in csv_files:
        try:
            df = pd.read_csv(file)
            all_data.append(df)
            print(f"[OK] 已读取: {file.name}，行数: {len(df)}")
        except Exception as e:
            print(f"[ERROR] 读取失败: {file.name} -> {e}")

    merged_df = pd.concat(all_data, ignore_index=True)

    merged_df.to_csv(output_file, index=False, encoding="utf-8-sig")

    print(f"\n合并完成，总行数: {len(merged_df)}")
    print(f"输出文件: {output_file.resolve()}")


if __name__ == "__main__":
    main()
