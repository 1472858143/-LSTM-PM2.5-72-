from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np


class BaseForecastModel(ABC):
    """六类模型的统一接口。

    训练流程只依赖 fit/predict/save 三个方法，从而保证传统时间序列模型、
    机器学习模型和深度学习模型可以共用同一套训练与输出逻辑。
    """

    name: str

    def __init__(self, config: dict[str, Any], model_name: str) -> None:
        self.config = config
        self.model_name = model_name
        self.model_config = config["models"][model_name]

    @abstractmethod
    def fit(self, data: dict[str, Any]) -> None:
        """使用训练集和验证集拟合模型。"""
        raise NotImplementedError

    @abstractmethod
    def predict(self, data: dict[str, Any]) -> np.ndarray:
        """对测试窗口直接输出未来 72 小时预测，shape 为 (N, 72)。"""
        raise NotImplementedError

    @abstractmethod
    def save(self, path: str | Path) -> None:
        """保存模型产物到 outputs/{model}/model.pt。"""
        raise NotImplementedError

    def load(self, path: str | Path) -> None:
        raise NotImplementedError(f"{self.__class__.__name__} 暂未实现 load。")
