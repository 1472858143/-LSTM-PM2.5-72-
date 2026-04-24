from __future__ import annotations

import importlib.util
from typing import Iterable


DEEP_LEARNING_MODELS = {"lstm", "attention_lstm"}
MODEL_PACKAGES = {
    "arima": ["statsmodels", "matplotlib"],
    "sarima": ["statsmodels", "matplotlib"],
    "xgboost": ["sklearn", "xgboost", "joblib", "matplotlib"],
    "random_forest": ["sklearn", "joblib", "matplotlib"],
    "lstm": ["torch", "matplotlib"],
    "attention_lstm": ["torch", "matplotlib"],
}


class EnvironmentError(RuntimeError):
    """运行环境不满足项目要求时抛出的明确错误。"""
    pass


def _missing_packages(packages: Iterable[str]) -> list[str]:
    return [package for package in packages if importlib.util.find_spec(package) is None]


def check_environment(config: dict, selected_models: Iterable[str] | None = None) -> None:
    """按所选模型检查依赖和 CUDA。

    传统模型、机器学习模型和深度学习模型依赖不同，只检查当前运行所需依赖；
    但 LSTM/Attention-LSTM 必须使用 CUDA GPU，这是项目书确定的运行约束。
    """
    selected = set(selected_models or [])
    env = config["environment"]
    required = list(env.get("required_core_packages", []))

    for model_name in selected:
        required.extend(MODEL_PACKAGES.get(model_name, []))

    needs_torch = bool(selected & DEEP_LEARNING_MODELS)

    missing = sorted(set(_missing_packages(required)))
    if missing:
        raise EnvironmentError(
            "缺少必要依赖: "
            + ", ".join(missing)
            + "。请先安装 requirements.txt；PyTorch CUDA 版本请参考 requirements-cuda.txt。"
        )

    if needs_torch and env.get("require_cuda_for_deep_learning", True):
        import torch

        if not torch.cuda.is_available():
            raise EnvironmentError(
                "深度学习模型要求 CUDA GPU，但当前 torch.cuda.is_available() 为 False。"
            )
