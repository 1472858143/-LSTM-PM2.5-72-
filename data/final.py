import pandas as pd

# 输入文件
pm25_file = "pm25/Beijing_PM25_final.csv"
weather_file = "weather/processed_noaa/noaa_beijing_weather_final.csv"

# 输出文件
output_file = "Beijing_dataset_final.csv"

# =========================
# 1. 读取数据
# =========================
pm25_df = pd.read_csv(pm25_file)
weather_df = pd.read_csv(weather_file)

# =========================
# 2. 转 timestamp（防止字符串不一致）
# =========================
pm25_df['timestamp'] = pd.to_datetime(pm25_df['timestamp'])
weather_df['timestamp'] = pd.to_datetime(weather_df['timestamp'])

# =========================
# 3. 按时间严格合并（核心）
# =========================
df_merged = pd.merge(
    pm25_df,
    weather_df,
    on='timestamp',
    how='inner'   # 只保留完全匹配时间
)

# =========================
# 4. 排序
# =========================
df_merged = df_merged.sort_values('timestamp').reset_index(drop=True)

# =========================
# 5. 保存
# =========================
df_merged.to_csv(output_file, index=False, encoding='utf-8-sig')

print("合并完成！")
print(f"合并后数据量: {len(df_merged)}")
print(f"输出文件: {output_file}")
print(df_merged.head())