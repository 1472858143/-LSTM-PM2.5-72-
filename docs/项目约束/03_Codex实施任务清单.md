# 03_Codex实施任务清单

## 1. 项目结构

项目结构必须按照项目书确定的工程化设计执行，并增加正式前端展示系统模块。推荐结构如下：

```text
project/
│
├── data/
├── processed/
│
├── models/
│   ├── arima.py
│   ├── sarima.py
│   ├── xgboost_model.py
│   ├── random_forest.py
│   ├── lstm.py
│   └── attention_lstm.py
│
├── utils/
│   ├── preprocess.py
│   ├── window.py
│   ├── metrics.py
│
├── train/
├── evaluate/
├── visualization/
├── config/
├── outputs/
│   ├── arima/
│   ├── sarima/
│   ├── xgboost/
│   ├── random_forest/
│   ├── lstm/
│   ├── attention_lstm/
│   └── metrics_summary/
│
└── frontend/
```

## 2. 模块职责

### 2.1 `data/`

存放原始数据。该目录只保存原始文件，不允许在原始文件上直接修改。

~~待补充：

1. 数据集名称：NOAA + 北京PM2.5数据
2. 下载地址：NOAA(www.ncei.noaa.gov)+北京市生态环境监测中心(www.bjmemc.com.cn)
3. 文件路径：项目根目录/data/Beijing_dataset_selected.csv

### 2.2 `processed/`

存放处理后的数据文件，包括清洗后数据、归一化数据、滑动窗口样本、训练集、验证集和测试集。

### 2.3 `models/`

存放模型定义和模型训练所需的核心类或函数。

模块职责如下：

| 文件 | 职责 |
| --- | --- |
| `arima.py` | 实现 ARIMA 单变量 72 小时多步预测 |
| `sarima.py` | 实现 SARIMA 单变量 72 小时多步预测 |
| `xgboost_model.py` | 实现 XGBoost 展平输入与 72 小时多输出预测 |
| `random_forest.py` | 实现 Random Forest 展平输入与 72 小时多输出预测 |
| `lstm.py` | 实现 LSTM 三维输入与 72 小时多输出预测 |
| `attention_lstm.py` | 实现 Attention-LSTM、Attention 权重计算与 72 小时多输出预测 |

### 2.4 `utils/`

存放通用工具模块。

| 文件 | 职责 |
| --- | --- |
| `preprocess.py` | 数据清洗、缺失值处理、异常值处理、归一化 |
| `window.py` | 固定 168 小时输入、72 小时输出的滑动窗口构造 |
| `metrics.py` | 统一计算全部评价指标、分阶段误差和 horizon 误差 |

### 2.5 `train/`

存放训练入口脚本。每类模型应有独立训练入口，训练入口必须读取统一配置，不允许硬编码窗口长度、数据划分比例或输出路径。

### 2.6 `evaluate/`

存放统一评估逻辑。评估模块读取各模型的 `predictions.csv`，计算整体误差、分阶段误差和 horizon 曲线数据，并保存到 `metrics.json` 和 `metrics_summary/`。

### 2.7 `visualization/`

存放 Python 可视化脚本。必须生成真实值与预测值折线图、多模型对比图、指标图、峰值图和 Attention 权重图。

### 2.8 `config/`

存放配置文件。至少应包含：

1. 数据路径
2. 输出路径
3. 输入窗口 168
4. 输出窗口 72
5. 数据划分 70 / 15 / 15
6. seed = 42
7. device = GPU
8. 各模型参数

未确定参数必须留空或使用显式占位，不允许代码自行推断。

### 2.9 `outputs/`

存放所有模型输出结果。目录结构固定为：

```text
outputs/
├── arima/
├── sarima/
├── xgboost/
├── random_forest/
├── lstm/
├── attention_lstm/
├── metrics_summary/
```

每个模型目录必须包含：

1. `predictions.csv`
2. `metrics.json`
3. `plots/`
4. `model.pt`

`predictions.csv` 格式固定为：

```text
sample_id,horizon,y_true,y_pred
```

### 2.10 `frontend/`

前端是正式系统模块，不是附加项。技术栈固定为：

1. Vue 3
2. ECharts

前端职责：

1. 读取 `outputs/` 中的预测结果和指标结果。
2. 展示模型选择控件。
3. 展示真实值与预测值曲线。
4. 展示模型指标对比。
5. 展示 1-24 / 25-48 / 49-72 分阶段误差。
6. 展示 horizon 曲线。
7. 展示 Attention-LSTM 的 Attention 权重图。

前端禁止重新计算核心指标，禁止修改预测结果，禁止改变模型集合或实验结论。

## 3. 开发顺序

开发顺序必须固定如下：

1. 项目结构
2. 数据处理
3. 滑动窗口
4. 指标模块
5. LSTM
6. Attention-LSTM
7. XGBoost / RF
8. ARIMA / SARIMA
9. 可视化
10. 前端

不得跳过数据处理、滑动窗口和指标模块直接开发模型。

## 4. 每阶段产出

### 4.1 阶段一：项目结构

任务：

1. 创建标准目录结构。
2. 创建配置文件。
3. 创建输出目录。
4. 固定随机种子配置。
5. 固定 GPU / PyTorch 配置。

产出：

1. `data/`
2. `processed/`
3. `models/`
4. `utils/`
5. `train/`
6. `evaluate/`
7. `visualization/`
8. `config/`
9. `outputs/`
10. `frontend/`

### 4.2 阶段二：数据处理

任务：

1. 读取原始数据。
2. 执行字段映射。
3. 处理时间戳。
4. 执行缺失值处理。
5. 执行异常值处理。
6. 使用 MinMaxScaler 归一化。
7. 仅在训练集 fit scaler。

产出：

1. 清洗后的数据文件。
2. 归一化后的数据文件。
3. scaler 文件。
4. 数据处理日志。

### 4.3 阶段三：滑动窗口

任务：

1. 构造输入窗口 168 小时。
2. 构造输出窗口 72 小时。
3. 生成输入 shape `(168, 6)`。
4. 生成输出 shape `(72,)`。
5. 按时间顺序划分 70 / 15 / 15。
6. 禁止 shuffle。

产出：

1. 训练集窗口样本。
2. 验证集窗口样本。
3. 测试集窗口样本。
4. 窗口样本统计信息。

### 4.4 阶段四：指标模块

任务：

1. 实现 RMSE。
2. 实现 MAE。
3. 实现 MAPE，分母必须大于等于 1。
4. 实现 R²。
5. 实现 SMAPE。
6. 实现 MSE。
7. 实现 Explained Variance。
8. 实现 Max Error。
9. 实现整体误差。
10. 实现分阶段误差。
11. 实现 horizon 曲线指标。

产出：

1. `utils/metrics.py`
2. 指标单元测试或验证样例。
3. 统一指标输出结构。

### 4.5 阶段五：LSTM

任务：

1. 使用 PyTorch 实现 LSTM。
2. 输入 shape 固定为 `(batch_size, 168, 6)`。
3. 输出 shape 固定为 `(batch_size, 72)`。
4. 使用 MSE 损失。
5. 使用 Adam 优化器。
6. 使用 GPU。
7. seed 固定为 42。

产出：

1. `models/lstm.py`
2. LSTM 训练入口。
3. LSTM `predictions.csv`
4. LSTM `metrics.json`
5. LSTM `model.pt`
6. LSTM 图表文件。

### 4.6 阶段六：Attention-LSTM

任务：

1. 使用 PyTorch 实现 Attention-LSTM。
2. 输入 shape 固定为 `(batch_size, 168, 6)`。
3. 输出 shape 固定为 `(batch_size, 72)`。
4. 保存 Attention 权重。
5. 保证 Attention 权重对应 168 小时输入窗口。
6. 使用 MSE 损失。
7. 使用 Adam 优化器。
8. 使用 GPU。
9. seed 固定为 42。

产出：

1. `models/attention_lstm.py`
2. Attention-LSTM 训练入口。
3. Attention-LSTM `predictions.csv`
4. Attention-LSTM `metrics.json`
5. Attention-LSTM `model.pt`
6. Attention 权重文件。
7. Attention 权重图。

### 4.7 阶段七：XGBoost / RF

任务：

1. 将 `(168, 6)` 输入展平为 `(1008,)`。
2. 实现 XGBoost 多输出预测。
3. 实现 Random Forest 多输出预测。
4. 输出未来 72 小时 PM2.5。
5. 保存统一格式预测结果。

产出：

1. `models/xgboost_model.py`
2. `models/random_forest.py`
3. XGBoost `predictions.csv`
4. Random Forest `predictions.csv`
5. 对应 `metrics.json`
6. 对应 `model.pt`
7. 对应图表文件。

### 4.8 阶段八：ARIMA / SARIMA

任务：

1. 使用 PM2.5 单变量序列建模。
2. 实现 ARIMA 72 小时多步预测。
3. 实现 SARIMA 72 小时多步预测。
4. SARIMA 优先考虑 24 小时季节周期。
5. 保存统一格式预测结果。

产出：

1. `models/arima.py`
2. `models/sarima.py`
3. ARIMA `predictions.csv`
4. SARIMA `predictions.csv`
5. 对应 `metrics.json`
6. 对应 `model.pt`
7. 对应图表文件。

### 4.9 阶段九：可视化

任务：

1. 绘制真实值 vs 预测值折线图。
2. 绘制多模型对比图。
3. 绘制指标对比图。
4. 绘制峰值图。
5. 绘制分阶段误差图。
6. 绘制 horizon 曲线。
7. 绘制 Attention 权重图。

产出：

1. 每个模型 `plots/` 目录。
2. `outputs/metrics_summary/` 汇总图。
3. 论文可用图表。

### 4.10 阶段十：前端

任务：

1. 使用 Vue 3 创建前端项目。
2. 使用 ECharts 绘制图表。
3. 实现模型选择。
4. 展示预测曲线。
5. 展示指标对比。
6. 展示分阶段误差。
7. 展示 horizon 曲线。
8. 展示 Attention 可视化。
9. 从 `outputs/` 读取标准结果文件。

产出：

1. `frontend/` 项目。
2. 模型选择页面。
3. 预测曲线图表。
4. 指标对比图表。
5. 分阶段误差图表。
6. Attention 权重图表。

## 5. 代码规范

### 5.1 命名规范

1. 文件名使用小写字母和下划线。
2. 模型文件名必须与项目结构一致。
3. 输出目录名称必须与模型名称一致。
4. 字段名必须使用标准字段映射。

### 5.2 配置规范

以下内容必须配置化：

1. 数据路径
2. 输出路径
3. 输入窗口 168
4. 输出窗口 72
5. 数据划分 70 / 15 / 15
6. seed = 42
7. device = GPU
8. 模型参数

### 5.3 结果规范

每个模型必须输出：

1. `predictions.csv`
2. `metrics.json`
3. `plots/`
4. `model.pt`

CSV 格式必须为：

```text
sample_id,horizon,y_true,y_pred
```

### 5.4 实验规范

1. 所有模型使用同一测试集。
2. 所有指标基于反归一化结果计算。
3. 所有模型输出未来 72 小时预测结果。
4. 所有模型必须进行整体误差、分阶段误差和 horizon 曲线分析。
5. 训练日志必须记录关键参数和输出路径。

### 5.5 前端规范

1. 前端只能读取标准输出文件。
2. 前端不得重新计算核心评价指标。
3. 前端不得修改预测结果。
4. 前端不得新增模型。
5. 前端不得新增输入特征。
6. 前端展示结果必须与后端 `outputs/` 保持一致。

## 6. 禁止事项

以下事项严格禁止：

1. 不允许改输入特征。
2. 不允许改窗口。
3. 不允许改模型集合。
4. 不允许随机划分。
5. 不允许 shuffle。
6. 不允许使用未定义数据。
7. 不允许缺少指标。
8. 不允许输出格式不一致。
9. 不允许使用递归预测。
10. 不允许测试集参与 scaler fit。
11. 不允许测试集参与调参。
12. 不允许前端重新计算核心指标。
13. 不允许前端展示与后端输出不一致的结果。
14. 不允许将 Attention-LSTM 简化为普通 LSTM。
15. 不允许只报告整体误差而缺少分阶段误差和 horizon 曲线。

## 7. 最终交付要求

最终工程系统必须能够支持以下毕业论文内容：

1. 数据预处理方法说明。
2. 滑动窗口构造说明。
3. 六类模型对比实验。
4. 八个评价指标结果。
5. 1-24 / 25-48 / 49-72 分阶段误差分析。
6. horizon 曲线分析。
7. 峰值预测分析。
8. 趋势拟合分析。
9. Attention 权重可视化分析。
10. Vue 3 + ECharts 前端展示系统说明。
