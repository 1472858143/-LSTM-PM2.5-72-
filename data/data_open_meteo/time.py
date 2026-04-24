import pandas as pd

# 1. 读取数据
df = pd.read_csv("data.csv")

# 2. 转换 time 列为 datetime
df["time"] = pd.to_datetime(df["time"], errors="coerce")

# 3. 判断是否有时区（关键逻辑）
if df["time"].dt.tz is None:
    # 默认当作 UTC（Open-Meteo / 大部分气象数据都是UTC）
    df["time"] = df["time"].dt.tz_localize("UTC")

# 4. 转换为北京时间（UTC+8）
df["timestamp"] = df["time"].dt.tz_convert("Asia/Shanghai")

# 5. 去掉时区（模型训练必须）
df["timestamp"] = df["timestamp"].dt.tz_localize(None)

# 6. 删除原 time 列
df = df.drop(columns=["time"])

# 7. 保存新文件
df.to_csv("processed_beijing.csv", index=False, encoding="utf-8-sig")

print("转换完成，已生成 processed_beijing.csv")