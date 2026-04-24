import pandas as pd

input_file = "Beijing_dataset_final.csv"
output_file = "Beijing_dataset_selected.csv"

df = pd.read_csv(input_file)

# =========================
# 1. 统一列名（防止匹配失败）
# =========================
df.columns = df.columns.str.strip().str.lower()

# =========================
# 2. 列名映射（你的数据可能不统一）
# =========================
rename_map = {
    'pm2.5': 'pm25',
    'pm_2.5': 'pm25',
    'temperature_2m': 'temp',
    'temperature': 'temp',
    'relative_humidity_2m': 'humidity',
    'humidity': 'humidity',
    'wind_speed_10m': 'wind_speed',
    'wind_speed': 'wind_speed',
    'surface_pressure': 'pressure',
    'pressure': 'pressure',
    'precipitation': 'precipitation',
    'rain': 'precipitation'
}

df = df.rename(columns=rename_map)

# =========================
# 3. 只保留需要的列
# =========================
target_cols = [
    'timestamp',
    'pm25',
    'temp',
    'humidity',
    'wind_speed',
    'precipitation',
    'pressure'
]

# 保留存在的列
df = df[[col for col in target_cols if col in df.columns]]

# =========================
# 4. 时间格式统一
# =========================
df['timestamp'] = pd.to_datetime(df['timestamp'])

# =========================
# 5. 排序
# =========================
df = df.sort_values('timestamp').reset_index(drop=True)

# =========================
# 6. 保存
# =========================
df.to_csv(output_file, index=False, encoding='utf-8-sig')

print("处理完成！")
print(f"输出文件: {output_file}")
print(df.head())