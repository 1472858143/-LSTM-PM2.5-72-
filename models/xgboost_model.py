from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from models.base import BaseForecastModel


class XGBoostForecastModel(BaseForecastModel):
    """XGBoost 多变量多输出回归模型。

    机器学习模型不能直接处理三维序列，因此将 (168, 6) 窗口展平为 1008 维，
    再通过 MultiOutputRegressor 一次性输出 72 个预测步长。
    """

    name = "xgboost"

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config, self.name)
        self.model: Any = None

    def fit(self, data: dict[str, Any]) -> None:
        print("XGBOOST START")
        from sklearn.multioutput import MultiOutputRegressor
        from xgboost import XGBRegressor

        params = {
            key: value
            for key, value in self.model_config.items()
            if key
            in {
                "n_estimators",
                "max_depth",
                "learning_rate",
                "subsample",
                "colsample_bytree",
                "objective",
                "random_state",
                "n_jobs",
            }
        }
        # XGBoost 原生回归器负责单输出，外层 MultiOutputRegressor 保持直接多输出策略。
        base_model = XGBRegressor(**params)
        self.model = MultiOutputRegressor(base_model)
        # 展平只改变模型输入形态，不新增任何特征。
        X_train = data["X_train"].reshape(data["X_train"].shape[0], -1)
        self.model.fit(X_train, data["y_train"])

    def predict(self, data: dict[str, Any]) -> np.ndarray:
        print("XGBOOST PREDICT START")
        if self.model is None:
            raise RuntimeError("XGBoost 尚未 fit。")
        X_test = data["X_test"].reshape(data["X_test"].shape[0], -1)
        return np.asarray(self.model.predict(X_test), dtype=np.float32)

    def save(self, path: str | Path) -> None:
        import joblib

        joblib.dump(self.model, Path(path))
