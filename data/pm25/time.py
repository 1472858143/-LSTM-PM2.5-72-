import pandas as pd

# 输入输出文件
input_file = "merge_pm25_filtered.csv"
output_file = "merge_pm25_timestamp.csv"

# 读取数据
df = pd.read_csv(input_file)

# 检查必要字段
if 'date' not in df.columns:
    raise KeyError("缺少 date 列")
if 'hour' not in df.columns:
    raise KeyError("缺少 hour 列")

# 转换 date 为 datetime
df['date'] = pd.to_datetime(df['date'], format='%Y%m%d', errors='coerce')

# 处理 hour（兼容 0-23 或 1-24）
def fix_hour(h):
    h = int(h)
    if h == 24:
        return 0
    return h

df['hour'] = df['hour'].apply(fix_hour)

# 生成 timestamp
df['timestamp'] = df['date'] + pd.to_timedelta(df['hour'], unit='h')

# 设置为北京时间（核心步骤）
df['timestamp'] = df['timestamp'].dt.tz_localize('Asia/Shanghai')

# 删除原 date 列（按你的要求）
df = df.drop(columns=['date'])

# 调整列顺序（把 timestamp 放第一列）
cols = ['timestamp'] + [c for c in df.columns if c != 'timestamp']
df = df[cols]

# 保存
df.to_csv(output_file, index=False, encoding='utf-8-sig')

print("处理完成！")
print(f"输出文件: {output_file}")