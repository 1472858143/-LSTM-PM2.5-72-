from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import numpy as np

from utils.config import ensure_project_dirs, resolve_path
from utils.data_loader import prepare_window_data
from utils.env import check_environment
from utils.metrics import compute_all_metrics
from utils.output import (
    copy_metrics_to_summary,
    prepare_model_output_dir,
    save_attention_stats,
    save_config_snapshot,
    save_metrics,
    save_metrics_tables,
    save_peak_analysis,
    save_predictions,
    save_training_history,
)
from utils.seed import set_global_seed
from visualization.plots import create_model_plots, plot_loss_curve, plot_peak_case


def _instantiate_model(model_name: str, config: dict[str, Any]):
    """根据配置中的模型名称创建模型实例。

    使用延迟导入可以避免只运行数据处理时强制加载所有模型依赖。
    """
    if model_name == "arima":
        from models.arima import ARIMAForecastModel

        return ARIMAForecastModel(config)
    if model_name == "sarima":
        from models.sarima import SARIMAForecastModel

        return SARIMAForecastModel(config)
    if model_name == "xgboost":
        from models.xgboost_model import XGBoostForecastModel

        return XGBoostForecastModel(config)
    if model_name == "random_forest":
        from models.random_forest import RandomForestForecastModel

        return RandomForestForecastModel(config)
    if model_name == "lstm":
        from models.lstm import LSTMForecastModel

        return LSTMForecastModel(config)
    if model_name == "attention_lstm":
        from models.attention_lstm import AttentionLSTMForecastModel

        return AttentionLSTMForecastModel(config)
    raise ValueError(f"不支持的模型名称: {model_name}")


def normalize_model_selection(config: dict[str, Any], models: Iterable[str] | str) -> list[str]:
    """校验命令行传入的模型名称，防止运行未定义模型。"""
    allowed = config["models"]["allowed_model_names"]
    if isinstance(models, str):
        if models == "all":
            selected = allowed
        else:
            selected = [models]
    else:
        selected = list(models)
        if selected == ["all"]:
            selected = allowed

    invalid = [model for model in selected if model not in allowed]
    if invalid:
        raise ValueError(f"模型名称不在允许列表中: {invalid}")
    return selected


def _validate_window_data(data: dict[str, Any], config: dict[str, Any]) -> None:
    """检查滑动窗口 shape 是否符合项目书约束。

    这里是训练前的最后一道防线，确保任何模型拿到的数据都是 (N,168,6)->(N,72)。
    """
    input_window = int(config["window"]["input_window_hours"])
    output_window = int(config["window"]["output_window_hours"])
    feature_count = int(config["global_constraints"]["feature_count"])
    for split_name in ["train", "validation", "test"]:
        X = data[f"X_{split_name}"]
        y = data[f"y_{split_name}"]
        if X.ndim != 3 or X.shape[1:] != (input_window, feature_count):
            raise ValueError(f"{split_name} X shape 错误: {X.shape}")
        if y.ndim != 2 or y.shape[1] != output_window:
            raise ValueError(f"{split_name} y shape 错误: {y.shape}")
        if len(X) == 0:
            raise ValueError(f"{split_name} 没有可用滑动窗口样本。")


def run_training_pipeline(config: dict[str, Any], selected_models: Iterable[str] | str = "all") -> dict[str, Any]:
    """统一训练主流程。

    该函数串联环境检查、数据预处理、窗口构造、模型训练、预测反归一化、
    指标计算、结果保存和图表生成，保证六类模型共享同一实验流程。
    """
    models = normalize_model_selection(config, selected_models)
    ensure_project_dirs(config)
    check_environment(config, models)
    set_global_seed(int(config["environment"]["seed"]))

    # 数据准备阶段会生成规范 CSV、scaler、windows.npz 和处理日志。
    data = prepare_window_data(config)
    _validate_window_data(data, config)
    deep_models = {"lstm", "attention_lstm"}

    results: dict[str, Any] = {}
    for model_name in models:
        model_cfg = config["models"][model_name]
        if not model_cfg.get("enabled", True):
            continue

        model = _instantiate_model(model_name, config)
        output_dir = prepare_model_output_dir(config, model_name)
        # 每个模型只实现自己的 fit/predict，公共输出逻辑集中在这里。
        model.fit(data)
        y_pred_scaled = model.predict(data)

        # 评价指标必须基于真实 PM2.5 单位计算，因此预测和标签都要反归一化。
        y_true = data["scaler"].inverse_transform_target(data["y_test"])
        y_pred = data["scaler"].inverse_transform_target(y_pred_scaled)
        metrics = compute_all_metrics(y_true, y_pred, config)

        save_predictions(config, model_name, y_true, y_pred, data["timestamps_test"])
        save_metrics(config, model_name, metrics)
        save_config_snapshot(config, model_name)
        save_metrics_tables(config, model_name, metrics)

        # 统一模型文件名为 model.pt，便于前端或后续脚本按固定路径查找。
        model_artifact = output_dir / "model.pt"
        model.save(model_artifact)

        # Attention-LSTM 额外保存权重，其他模型不会生成该文件。
        attention_weights = getattr(model, "attention_weights", None)
        if model_name == "attention_lstm" and attention_weights is not None:
            attention_path = resolve_path(model_cfg["attention_weights_path"])
            model.save_attention_weights(attention_path)
            save_attention_stats(config, model_name, attention_weights)

        create_model_plots(y_true, y_pred, metrics, output_dir / "plots", attention_weights)
        if model_name in deep_models:
            history = getattr(model, "training_history", [])
            save_training_history(config, model_name, history)
            if history:
                plot_loss_curve(history, output_dir / "plots")
            peak_summary = save_peak_analysis(config, model_name, y_true, y_pred, data["timestamps_test"])
            if peak_summary["selected_sample_ids"]:
                plot_peak_case(y_true, y_pred, peak_summary["selected_sample_ids"][0], output_dir / "plots")
        copy_metrics_to_summary(config, model_name)

        results[model_name] = {
            "output_dir": str(output_dir),
            "predictions_shape": list(np.asarray(y_pred).shape),
            "overall_metrics": metrics["overall"],
        }

    return results
