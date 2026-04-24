import pandas as pd

# 输入输出文件
input_file = "merge_pm25.csv"
output_file = "merge_pm25_filtered.csv"

# 读取数据
df = pd.read_csv(input_file)

# 确保 type 列存在
if 'type' not in df.columns:
    raise KeyError("文件中不存在 'type' 列")

# 筛选数据
df_filtered = df[df['type'].isin(['PM2.5'])]

# 保存结果
df_filtered.to_csv(output_file, index=False, encoding='utf-8-sig')

print("处理完成！")
print(f"原始数据行数: {len(df)}")
print(f"筛选后数据行数: {len(df_filtered)}")
print(f"输出文件: {output_file}")