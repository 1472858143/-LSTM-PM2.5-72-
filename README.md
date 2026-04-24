# 基于 Attention-LSTM 的北京市 PM2.5 浓度 72 小时多步预测系统

## 1. 项目简介

本项目是毕业设计“基于注意力机制 LSTM 的北京市 PM2.5 浓度 72 小时多步预测研究”的核心训练与预测系统。系统面向北京市小时级空气质量预测任务，使用过去 168 小时的历史 PM2.5 浓度与气象数据，直接预测未来连续 72 小时 PM2.5 浓度。

项目目标是形成一套可复现、可评估、可供前端展示读取的实验工程系统，用于支撑毕业论文中的模型对比、误差分析、趋势拟合分析和 Attention 权重解释。

核心任务：

- 输入：过去 168 小时、6 个固定特征。
- 输出：未来 72 小时 PM2.5 浓度。
- 核心模型：Attention-LSTM。
- 对比模型：ARIMA、SARIMA、XGBoost、Random Forest、LSTM、Attention-LSTM。

本项目严格遵守项目书和 `docs/` 中的规范文件。若项目书与其他文档冲突，以项目书为准；若项目书未明确，以 `docs/01_项目开发总说明.md`、`docs/02_数据与实验规范.md`、`docs/03_Codex实施任务清单.md` 为准。

## 2. 项目目录结构说明

```text
project/
├── config/               # 全局配置文件，集中管理路径、字段、窗口、模型参数、指标和输出目录
├── data/                 # 原始或中间数据文件，不在原始文件上直接修改
├── docs/                 # 项目书、开发规范、实施计划等文档
├── evaluate/             # 统一评估入口，可基于 predictions.csv 重新计算 metrics.json
├── models/               # 六类模型实现与模型统一基类
├── outputs/              # 各模型预测结果、指标文件、图表和模型文件
├── processed/            # 规范化 CSV、滑动窗口样本、归一化参数和处理日志
├── train/                # 统一训练入口与训练调度流程
├── utils/                # 配置、数据预处理、窗口构造、指标、输出、环境检查等公共模块
├── visualization/        # Python 静态图表生成模块
├── requirements.txt      # Python 传统模型和通用依赖
└── requirements-cuda.txt # PyTorch CUDA cu126 安装说明
```

重点目录说明：

- `config/`：核心配置目录。`config/config.json` 是所有路径、字段、模型参数、训练参数和输出规则的唯一配置来源。后续开发不要在代码中硬编码这些参数。
- `models/`：模型实现目录。包含 ARIMA、SARIMA、XGBoost、Random Forest、LSTM、Attention-LSTM 六类模型，以及统一接口 `BaseForecastModel`。
- `train/`：训练入口目录。`train/run_all.py` 是当前后端主入口，负责按配置完成数据处理、窗口构造、训练、预测、评估和输出。
- `evaluate/`：评估目录。`evaluate/evaluator.py` 可读取某个模型的 `predictions.csv` 并重新生成 `metrics.json` 和图表。
- `outputs/`：标准输出目录。前端展示系统应从这里读取预测结果、指标和 Attention 权重。
- `processed/`：处理后数据目录。保存规范 CSV、窗口样本、MinMaxScaler 参数和日志，用于复现实验过程。

## 3. 运行环境要求

推荐环境：

- Python：3.10
- PyTorch：CUDA 12.6 对应版本
- CUDA：cu126
- GPU：必须具备可用 CUDA GPU，LSTM 和 Attention-LSTM 强制使用 GPU
- Node.js：24.14.0，预留给后续 Vue 3 + ECharts 前端模块

当前后端训练系统会在启动时检查依赖。若运行 LSTM 或 Attention-LSTM，系统会检查 `torch.cuda.is_available()`；如果 CUDA 不可用，深度学习模型会停止运行。

建议使用conda环境：

```powershell
# 安装anaconda或者miniconda
conda create -n pm25 python=3.10
```

## 4. 依赖安装说明

安装通用 Python 依赖：

```powershell
pip install -r requirements.txt
```

安装 PyTorch CUDA cu126：

```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```
安装速度太慢可以配置pip源

```powershell
# 配置清华大学镜像站
python -m pip config --user set global.index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
python -m pip config --user set global.timeout 120

# 检查配置
python -m pip config list
# 看到类似输出则配置成功:
# global.index-url='https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple'
# global.timeout='120'
```

安装完成后可检查核心环境：

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.version.cuda)"
```

如果只验证数据处理和窗口构造，至少需要 `pandas` 和 `numpy`。如果要完整训练六个模型，需要安装 `scikit-learn`、`statsmodels`、`xgboost`、`matplotlib`、`joblib` 和 PyTorch CUDA 版本。

## 5. 数据说明

输入数据为单一 CSV 文件。当前配置默认读取：

```text
data/Beijing_dataset_selected.csv
```

系统会将旧字段名映射为规范字段，并生成：

```text
processed/pm25_hourly_canonical.csv
```

规范 CSV 字段必须固定为：

| 字段 | 含义 |
| --- | --- |
| `timestamp` | 小时级时间戳 |
| `pm2_5` | PM2.5 浓度，预测目标，同时作为历史输入特征 |
| `temperature_2m` | 2 米气温 |
| `humidity` | 相对湿度 |
| `wind_speed_10m` | 10 米风速 |
| `precipitation` | 降水量 |
| `surface_pressure` | 地面气压 |

模型实际输入特征顺序固定为：

```text
temperature_2m, humidity, wind_speed_10m, precipitation, surface_pressure, pm2_5
```

数据要求：

- 时间粒度必须为 1 小时。
- 输入特征数量必须为 6。
- 不允许新增时间编码、节假日、站点编号、其他污染物等额外特征。
- 缺失值和异常值处理在滑动窗口构造前完成。

## 6. 配置文件说明

主配置文件：

```text
config/config.json
```

重要配置项：

- `paths`：原始 CSV、规范 CSV、窗口文件、归一化参数、输出目录等路径。
- `data`：规范字段、旧字段映射、输入特征顺序和预测目标。
- `preprocessing`：时间频率、重复时间戳处理、异常值阈值、MinMaxScaler 设置。
- `window`：输入窗口 168 小时、输出窗口 72 小时、步长 1 小时。
- `split`：按时间顺序划分训练集、验证集、测试集，比例为 70% / 15% / 15%，禁止 shuffle。
- `models`：六类模型的参数。所有模型参数必须从这里读取。
- `evaluation`：8 个评价指标、MAPE 分母下限、分阶段指标和 horizon 指标设置。
- `outputs`：各模型标准输出目录和 `predictions.csv` 列结构。

重要原则：

- 不要在代码中硬编码模型参数、窗口长度、字段名或输出路径。
- 如果需要调整训练轮数、树模型参数、ARIMA 搜索范围等，应修改 `config/config.json`。
- 不要修改输入特征、窗口大小、模型集合、评价指标和输出目录结构。

## 7. 运行步骤

以下命令均在项目根目录执行。

### 7.1 准备数据

将输入 CSV 放到配置文件指定位置，默认：

```text
data/Beijing_dataset_selected.csv
```

如果使用其他文件名，修改 `config/config.json` 中：

```json
"raw_input_csv": "data/Beijing_dataset_selected.csv"
```

### 7.2 安装依赖

```powershell
pip install -r requirements.txt
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

### 7.3 检查配置

确认 `config/config.json` 中以下内容符合本次实验：

- `input_window_hours` 为 `168`
- `output_window_hours` 为 `72`
- `feature_count` 为 `6`
- `seed` 为 `42`
- `allowed_model_names` 包含六个固定模型

### 7.4 运行完整训练流程

训练全部模型：

```powershell
python -m train.run_all --config config/config.json --models all
```

只训练部分模型：

```powershell
python -m train.run_all --config config/config.json --models random_forest xgboost
```

只训练 Attention-LSTM：

```powershell
python -m train.run_all --config config/config.json --models attention_lstm
```

训练流程会依次执行：

1. 读取配置。
2. 检查依赖和 CUDA。
3. 生成规范 CSV。
4. 按时间顺序划分 70% / 15% / 15%。
5. 仅使用训练集拟合 MinMaxScaler。
6. 构造 168 -> 72 滑动窗口。
7. 训练指定模型。
8. 对测试集预测。
9. 将预测值反归一化后计算指标。
10. 写入 `outputs/{model}/`。

### 7.5 单独执行评估

如果某个模型已经生成 `predictions.csv`，可重新计算指标：

```powershell
python -m evaluate.evaluator --config config/config.json --model random_forest
```

### 7.6 查看输出结果

训练完成后查看：

```text
outputs/{model}/predictions.csv
outputs/{model}/metrics.json
outputs/{model}/plots/
```

Attention-LSTM 额外查看：

```text
outputs/attention_lstm/attention_weights.npy
```

### 7.7 前端展示

当前阶段未实现前端代码，但后端已经提供标准输出接口。后续 Vue 3 + ECharts 前端应只读取 `outputs/` 下的标准文件，不重新计算核心指标，不修改预测结果。

## 8. 输出结果说明

输出目录结构：

```text
outputs/
├── arima/
├── sarima/
├── xgboost/
├── random_forest/
├── lstm/
├── attention_lstm/
└── metrics_summary/
```

每个模型目录包含：

- `predictions.csv`：测试集逐样本、逐 horizon 的真实值和预测值。
- `metrics.json`：整体指标、分阶段指标和 1-72 小时逐 horizon 指标。
- `plots/`：预测曲线图、分阶段误差图、horizon 误差曲线图。
- `model.pt`：模型文件或模型配置快照，传统机器学习模型也使用该统一文件名保存。
- `config_snapshot.json`：本次运行使用的配置快照。

`predictions.csv` 固定列：

```text
sample_id,timestamp,horizon,y_true,y_pred
```

`metrics.json` 包含：

- `overall`：整体 72 小时指标。
- `stages`：`h1_24`、`h25_48`、`h49_72` 三个阶段指标。
- `horizon`：第 1 小时到第 72 小时逐步长指标。

评价指标固定为：

```text
RMSE, MAE, MAPE, R2, SMAPE, MSE, Explained Variance, Max Error
```

前端展示模块应读取：

- 预测曲线：`outputs/{model}/predictions.csv`
- 指标表：`outputs/{model}/metrics.json`
- 图表资源：`outputs/{model}/plots/`
- Attention 可视化：`outputs/attention_lstm/attention_weights.npy`

## 9. 常见问题与排查

### 9.1 依赖安装失败

现象：运行训练入口时报缺少 `sklearn`、`statsmodels`、`xgboost`、`matplotlib`、`joblib` 或 `torch`。

处理：

```powershell
pip install -r requirements.txt
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

如果下载慢，可配置可信镜像，但 PyTorch CUDA 版本建议优先使用官方 wheel 源。

### 9.2 CUDA / GPU 不可用

现象：LSTM 或 Attention-LSTM 报错 `torch.cuda.is_available() 为 False`。

处理：

```powershell
python -c "import torch; print(torch.cuda.is_available()); print(torch.version.cuda)"
```

确认显卡驱动、CUDA 版本和 PyTorch CUDA wheel 匹配。项目要求深度学习模型使用 GPU，不建议改成 CPU 训练。

### 9.3 数据字段不匹配

现象：报错 `输入 CSV 缺少必需字段`。

处理：

- 检查原始 CSV 是否包含可映射字段。
- 当前配置允许将 `pm25` 映射为 `pm2_5`，`temp` 映射为 `temperature_2m`，`pressure` 映射为 `surface_pressure`。
- 如果字段名不同，应只修改 `config/config.json` 中的 `raw_to_canonical_field_mapping`，不要改模型输入特征集合。

### 9.4 输出目录为空

现象：`outputs/{model}/` 没有结果文件。

处理：

- 确认训练命令执行成功。
- 检查是否缺少依赖导致训练入口提前退出。
- 检查所选模型是否在 `config/config.json` 中 `enabled: true`。

### 9.5 模型训练报错

处理顺序：

1. 查看报错模型名称。
2. 检查该模型依赖是否安装。
3. 检查 `processed/window_log.json` 中训练、验证、测试样本数量是否大于 0。
4. 检查 `config/config.json` 中模型参数是否为有效数值。

### 9.6 前端读取不到结果文件

处理：

- 确认后端已经生成 `outputs/{model}/predictions.csv` 和 `metrics.json`。
- 确认前端读取路径与 `config/config.json` 中输出目录一致。
- 前端不得重新计算核心指标，应直接读取后端输出。

## 10. 项目约束说明

以下内容是毕业设计核心约束，后续开发者不得随意修改：

- 输入特征必须固定为 6 个。
- 输入字段必须为 `temperature_2m`、`humidity`、`wind_speed_10m`、`precipitation`、`surface_pressure`、`pm2_5`。
- 预测目标必须为未来 72 小时 `pm2_5`。
- 滑动窗口固定为 168 小时输入、72 小时输出。
- 数据必须按时间顺序划分 70% / 15% / 15%，禁止 shuffle。
- 模型集合固定为 ARIMA、SARIMA、XGBoost、Random Forest、LSTM、Attention-LSTM。
- 多步预测必须是直接多输出预测，禁止递归预测。
- 评价指标固定为 8 项。
- 输出目录和文件结构必须保持 `outputs/{model}/predictions.csv`、`metrics.json`、`plots/`。
- Attention-LSTM 必须输出与 168 个历史时间步对应的 Attention 权重。

## 11. 后续扩展建议

后续可以在不影响当前毕业设计实验结论的前提下扩展：

- 实现 Vue 3 + ECharts 前端展示系统。
- 增加多模型指标汇总图和论文图表导出。
- 在配置中扩大 ARIMA/SARIMA 搜索范围。
- 增加日志系统，记录每轮训练损失和验证损失。
- 增加自动化测试，覆盖字段校验、窗口构造、指标计算和输出格式。

这些扩展不属于当前核心实验规则，不能替代或改变现有六模型对比体系。
