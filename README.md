# 基于 Attention-LSTM 的北京市 PM2.5 浓度 72 小时多步预测系统

## 1. 项目简介

本项目用于完成“基于注意力机制 LSTM 的北京市 PM2.5 浓度 72 小时多步预测研究”的核心实验系统，实现统一的数据处理、模型训练、评估分析与结果输出。

研究目标固定为：

- 输入特征：6 个
  - `temperature_2m`
  - `humidity`
  - `wind_speed_10m`
  - `precipitation`
  - `surface_pressure`
  - `pm2_5`
- 预测目标：未来连续 `72` 小时 `pm2_5`
- 多步预测方式：直接多输出预测
- 核心模型：`Attention-LSTM`
- 对比模型：`ARIMA`、`SARIMA`、`XGBoost`、`Random Forest`、`LSTM`、`Attention-LSTM`

当前版本在固定 72 小时预测目标不变的前提下，新增了不同历史输入窗口长度对比实验：

- `168h -> 72h`
- `720h -> 72h`
- `2160h -> 72h`

该设计用于分析：更长的历史序列是否能为 `LSTM` 与 `Attention-LSTM` 提供额外有效信息，以及过长窗口是否会引入冗余和训练成本问题。

## 2. 项目目录结构

```text
project/
├── config/                    # 全局配置文件
├── data/                      # 原始或中间数据
├── docs/                      # 项目书与说明文档
├── evaluate/                  # 基于 predictions.csv 的独立评估入口
├── models/                    # 六类模型实现
├── outputs/                   # 训练结果输出目录
├── processed/                 # 规范化数据、滑动窗口数组和日志
├── train/                     # 训练入口与调度逻辑
├── utils/                     # 配置、数据处理、指标、输出等公共模块
├── visualization/             # 静态图表生成
├── requirements.txt           # Python 依赖
└── requirements-cuda.txt      # PyTorch CUDA 安装说明
```

重点目录说明：

- `config/`
  - `config/config.json` 是唯一配置源。
  - 所有路径、模型参数、窗口实验、输出规则都从这里读取。
- `models/`
  - 包含 `ARIMA`、`SARIMA`、`XGBoost`、`Random Forest`、`LSTM`、`Attention-LSTM` 六类模型。
- `train/`
  - `train/run_all.py` 是统一训练入口。
  - 支持按模型、按窗口实验、按组合批量运行。
- `evaluate/`
  - `evaluate/evaluator.py` 可在已有 `predictions.csv` 基础上重建 `metrics.json` 和图表。
- `outputs/`
  - 新版输出目录结构为 `outputs/{window_name}/{model}/`
  - 用于隔离不同输入窗口实验结果。

## 3. 运行环境要求

- Python：`3.10`
- PyTorch：建议安装 CUDA 版本
- CUDA：`cu126`
- GPU：`LSTM` 和 `Attention-LSTM` 强制使用 CUDA GPU
- Node.js：`24.14.0`（如果后续启用前端展示模块）

推荐使用独立环境：

```powershell
conda create -n pm25 python=3.10
conda activate pm25
```

## 4. 依赖安装

安装通用依赖：

```powershell
pip install -r requirements.txt
```

安装 PyTorch CUDA cu126：

```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

检查 CUDA 是否可用：

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.version.cuda)"
```

## 5. 数据说明

输入数据为单一小时级 CSV，当前配置默认读取：

```text
data/data_open_meteo/processed_beijing.csv
```

系统会将原始字段映射为规范字段，并生成：

```text
processed/pm25_hourly_canonical.csv
```

规范字段必须固定为：

| 字段 | 含义 |
| --- | --- |
| `timestamp` | 小时级时间戳 |
| `pm2_5` | PM2.5 浓度，既是目标也是历史输入 |
| `temperature_2m` | 2 米气温 |
| `humidity` | 湿度 |
| `wind_speed_10m` | 10 米风速 |
| `precipitation` | 降水 |
| `surface_pressure` | 地面气压 |

数据约束：

- 时间间隔必须为 `1` 小时
- 输入特征数量固定为 `6`
- 不允许增加额外衍生特征

## 6. 配置文件说明

主配置文件：

```text
config/config.json
```

重要配置块：

- `window`
  - 保留原有默认单窗口配置，兼容旧训练方式
- `window_experiments`
  - 新增多窗口实验配置
  - 当前提供：
    - `input_168h`
    - `input_720h`
    - `input_2160h`
- `models`
  - 六类模型参数
- `paths`
  - 数据、处理中间文件、输出目录路径
- `evaluation`
  - 固定评价指标和阶段分析配置
- `outputs`
  - 输出文件结构与预测 CSV 列定义

重要原则：

- 不要在代码中硬编码模型参数或路径
- 需要调整窗口、训练参数、模型开关时，优先修改 `config/config.json`

## 7. 多窗口实验设计

当前项目支持三组固定输入窗口实验，输出窗口始终固定为 `72`：

| 窗口实验名 | 输入窗口 | 输出窗口 | 含义 |
| --- | --- | --- | --- |
| `input_168h` | 168h | 72h | 过去 7 天预测未来 72 小时 |
| `input_720h` | 720h | 72h | 过去 30 天预测未来 72 小时 |
| `input_2160h` | 2160h | 72h | 过去 90 天预测未来 72 小时 |

设计目的：

- `168h`：主实验基线
- `720h`：验证更长历史是否提升中长期趋势学习能力
- `2160h`：验证超长历史是否进一步提升 Attention-LSTM，或引入冗余信息

不建议默认扩展到 `1 year` 或 `2 year` 窗口，因为：

- 输入维度增长过快
- 训练成本显著上升
- 冗余历史信息可能削弱有效学习

## 8. 训练入口与命令

### 8.1 训练全部窗口、全部模型

```powershell
python -m train.run_all --config config/config.json --models all --windows all
```

### 8.2 只训练某个窗口

```powershell
python -m train.run_all --config config/config.json --models all --windows input_168h
```

### 8.3 只训练某些模型和某些窗口

```powershell
python -m train.run_all --config config/config.json --models lstm attention_lstm --windows input_720h input_2160h
```

### 8.4 不传 `--windows`

如果不传 `--windows`，训练入口默认运行 `window_experiments` 中 `enabled=true` 的所有窗口实验。

### 8.5 兼容旧单窗口逻辑

如果未来移除 `window_experiments` 配置，系统仍会退回旧版单窗口训练逻辑，继续使用 `config.window` 中的默认窗口。

## 9. 推荐训练流程

### 第一阶段：快速验证

```powershell
python -m train.run_all --config config/config.json --models random_forest xgboost --windows input_168h
```

### 第二阶段：核心深度模型对比

```powershell
python -m train.run_all --config config/config.json --models lstm attention_lstm --windows input_168h input_720h input_2160h
```

### 第三阶段：完整六模型实验

```powershell
python -m train.run_all --config config/config.json --models all --windows all
```

推荐解释：

- `168h` 是基线实验
- `720h` 与 `2160h` 用于验证长历史窗口是否能提升 `LSTM` / `Attention-LSTM`
- 长窗口实验重点比较精度、趋势稳定性和训练代价，不只比较单一 RMSE

## 10. 输出目录结构

新版输出目录结构为：

```text
outputs/
├── input_168h/
│   ├── arima/
│   ├── sarima/
│   ├── xgboost/
│   ├── random_forest/
│   ├── lstm/
│   └── attention_lstm/
├── input_720h/
│   └── ...
├── input_2160h/
│   └── ...
└── metrics_summary/
```

每个模型目录中保留统一结构：

- `predictions.csv`
- `metrics.json`
- `stage_metrics.csv`
- `horizon_metrics.csv`
- `config_snapshot.json`
- `model.pt`
- `plots/`
- `training_log.json`

深度学习模型额外输出：

- `training_history.json`
- `training_history.csv`
- `plots/loss_curve.png`

Attention-LSTM 额外输出：

- `attention_weights.npy`
- `attention_stats.json`
- `plots/attention_weights.png`

`predictions.csv` 列结构固定为：

```text
sample_id,timestamp,horizon,y_true,y_pred
```

## 11. 汇总结果

训练全部窗口或多个窗口后，会在：

```text
outputs/metrics_summary/
```

生成以下汇总文件：

- `window_model_metrics.csv`
- `stage_metrics_summary.csv`
- `horizon_metrics_summary.csv`

其中：

- `window_model_metrics.csv`
  - 用于比较同一模型在不同输入窗口下的整体指标变化
- `stage_metrics_summary.csv`
  - 用于比较 1-24h、25-48h、49-72h 三阶段误差
- `horizon_metrics_summary.csv`
  - 用于分析误差随预测步长增长的稳定性

## 12. 图表说明

所有模型的 `prediction_curve.png` 和 `peak_case_top1.png` 均已改为使用真实 `timestamp` 作为 x 轴，而不是 `time point`。

图像特性：

- x 轴自动使用时间刻度格式化
- 标签自动旋转，避免重叠
- 标题包含：
  - `window_name`
  - `model_name`
  - `prediction target: 72h`

这使得不同窗口实验的结果图可以直接用于论文和答辩展示。

## 13. 控制台日志与进度条

项目运行时提供三层可读性增强：

1. 窗口实验总进度
2. 当前窗口下模型总进度
3. 深度学习模型 epoch 进度

每个模型至少会输出以下阶段日志：

1. `Start`
2. `Reading data...`
3. `Preprocessing data...`
4. `Building sliding windows...`
5. `Training model...`
6. `Predicting...`
7. `Calculating metrics...`
8. `Saving outputs...`
9. `Finished`

日志和进度条统一使用 `tqdm`，兼容 Windows PowerShell 与 PyCharm Terminal。

## 14. 论文分析方向

引入不同输入窗口长度后，推荐围绕以下问题展开论文分析：

1. 输入窗口长度对整体误差的影响
2. 输入窗口长度对 `1-24h`、`25-48h`、`49-72h` 三阶段误差的影响
3. `Attention-LSTM` 是否随窗口变长获得更明显收益
4. 过长输入窗口是否引入冗余信息、训练不稳定和额外时间成本
5. `Attention-LSTM` 的注意力权重是否在长窗口下仍能保持有效区分度

## 15. 常见问题

### 15.1 缺少依赖

```powershell
pip install -r requirements.txt
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

### 15.2 CUDA 不可用

```powershell
python -c "import torch; print(torch.cuda.is_available()); print(torch.version.cuda)"
```

### 15.3 长窗口样本数不足

当输入窗口扩大到 `2160h` 时，可用样本数会明显减少。如果日志中出现窗口样本不足，需要先检查：

- 数据总时长是否足够
- 数据缺失是否过多
- 时间序列是否存在大段断点

### 15.4 输出目录为空

检查：

- 是否缺少训练依赖
- 是否有模型运行失败
- 对应模型目录下 `training_log.json` 的 `status` 是否为 `failed`

### 15.5 树模型维度异常

当前 `XGBoost` 与 `Random Forest` 已按当前窗口动态计算展平维度：

- `168h -> 1008`
- `720h -> 4320`
- `2160h -> 12960`

如果仍出现维度错误，优先检查输入窗口配置和滑动窗口输出 shape。

## 16. 项目核心约束

以下内容不得随意修改：

- 预测目标固定为未来 `72` 小时 `pm2_5`
- 输入特征固定为 6 个
- 模型集合固定为 6 类
- 数据划分必须按时间顺序，禁止 shuffle
- 评价指标固定为：
  - `RMSE`
  - `MAE`
  - `MAPE`
  - `R2`
  - `SMAPE`
  - `MSE`
  - `Explained Variance`
  - `Max Error`
- `predictions.csv` 列结构固定为：
  - `sample_id,timestamp,horizon,y_true,y_pred`

## 17. 后续扩展建议

当前毕业设计强制范围之外，可后续扩展：

- 前端展示系统读取 `outputs/{window_name}/{model}/`
- 跨窗口可视化对比面板
- 更细粒度的训练资源统计
- 更系统的自动调参脚本
- 更丰富的峰值事件对比报告
