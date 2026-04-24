import pandas as pd

input_file = "merge_pm25_timestamp.csv"
output_file = "Beijing_PM25_final.csv"

# 读取数据
df = pd.read_csv(input_file)

# 只保留 PM2.5
if 'type' in df.columns:
    df = df[df['type'] == 'PM2.5'].copy()

# 删除 hour 列
if 'hour' in df.columns:
    df = df.drop(columns=['hour'])

# 必要列检查
if 'timestamp' not in df.columns:
    raise KeyError("缺少 timestamp 列")

# 不参与平均的非站点列
exclude_cols = {'timestamp', 'date', 'type'}

# 找出可转为数值的列
candidate_cols = [col for col in df.columns if col not in exclude_cols]

# 全部转成数值，无法转换的变成 NaN
for col in candidate_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# 只保留至少有一个数值的列
value_cols = [col for col in candidate_cols if df[col].notna().any()]

if not value_cols:
    raise ValueError("没有找到可用于计算平均值的站点数值列")

print("参与平均的列数：", len(value_cols))
print("前几个参与平均的列：", value_cols[:10])

# 按行计算平均值：每一行所有站点加起来再除以有效站点数
df['PM2.5'] = df[value_cols].mean(axis=1, skipna=True)

# 输出最终结果
result = df[['timestamp', 'PM2.5']].copy()
result = result.sort_values('timestamp').reset_index(drop=True)

result.to_csv(output_file, index=False, encoding='utf-8-sig')

print("完成！")
print(f"输出文件: {output_file}")
print(result.head())