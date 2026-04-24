# 配置项说明

## 1. `project`

用于记录项目名称、任务类型和配置文件作用。后续代码不应从该字段推断模型逻辑，仅用于说明和日志记录。

## 2. `environment`

用于固定运行环境，包括 Python、PyTorch、GPU、CUDA、Node.js 和随机种子。后续训练脚本、前端启动脚本和环境检查脚本均应以该字段为准。

## 3. `global_constraints`

用于集中声明全局硬约束，包括统一配置、禁止硬编码路径、输出根目录、输入窗口、输出窗口、直接多输出预测、禁止递归预测和禁止 shuffle。

## 4. `data`

用于管理数据源、时间字段、固定输入特征、模型内部字段顺序、字段映射和预测目标。数据源和原始字段映射当前留空，等待数据确定后补充。

## 5. `preprocessing`

用于管理数据预处理规则，包括时间排序、缺失值处理、异常值处理和 MinMaxScaler 归一化规则。归一化必须只在训练集上拟合。

## 6. `window`

用于管理滑动窗口规则。输入窗口固定为 168 小时，输出窗口固定为 72 小时，输入 shape 固定为 `(168, 6)`，输出 shape 固定为 `(72,)`。

## 7. `split`

用于管理数据集划分规则。划分方式固定为时间顺序划分，比例固定为 70 / 15 / 15，禁止 shuffle，并要求避免窗口边界泄露。

## 8. `models`

用于管理所有模型的默认参数。六个模型必须共享该配置文件，禁止各模型单独硬编码参数。

### 8.1 `models.allowed_model_names`

用于限定模型名称，只允许 `arima`、`sarima`、`xgboost`、`random_forest`、`lstm`、`attention_lstm`。

### 8.2 `models.arima`

用于管理 ARIMA 参数。参数选择方式为 auto search，基于 AIC/BIC，`p,d,q` 均为 auto。

### 8.3 `models.sarima`

用于管理 SARIMA 参数。参数选择方式为 auto search，基于 AIC/BIC，`seasonal_period` 固定为 24，非季节项 `p,d,q` 和季节项 `P,D,Q` 均为 auto。

### 8.4 `models.xgboost`

用于管理 XGBoost 默认参数，包括树数量、深度、学习率、采样比例、目标函数和随机种子。输入为展平后的 1008 维向量，输出为 72 维预测。

### 8.5 `models.random_forest`

用于管理 Random Forest 默认参数，包括树数量、最大深度、节点划分参数、随机种子和并行设置。输入为展平后的 1008 维向量，输出为 72 维预测。

### 8.6 `models.lstm`

用于管理 LSTM 默认参数，包括隐藏层维度、层数、dropout、batch size、学习率、训练轮数、早停、优化器、损失函数、设备和随机种子。

### 8.7 `models.attention_lstm`

用于管理 Attention-LSTM 默认参数。参数与 LSTM 保持一致，并额外要求输出 Attention 权重，权重保存路径固定为 `outputs/attention_lstm/attention_weights.npy`。

## 9. `evaluation`

用于管理统一评价指标和多步分析规则。指标包括 RMSE、MAE、MAPE、R2、SMAPE、MSE、Explained Variance 和 Max Error。MAPE 分母下限固定为 1。所有指标必须在反归一化后的 PM2.5 值上计算。

## 10. `outputs`

用于管理输出目录和统一产物格式。所有模型输出必须写入 `outputs/` 下对应模型目录，并生成 `predictions.csv`、`metrics.json`、`plots/` 和 `model.pt`。

## 11. `frontend`

用于管理前端展示系统配置。前端技术栈固定为 Vue 3 和 ECharts，模型名称、预测文件路径、指标文件路径、图表目录和 Attention 权重路径均从该字段读取，前端禁止自行计算指标或修改路径。
