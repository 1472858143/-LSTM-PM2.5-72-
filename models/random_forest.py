from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from models.base import BaseForecastModel


class RandomForestForecastModel(BaseForecastModel):
    """Random Forest 多变量多输出基线模型。

    与 XGBoost 相同，Random Forest 使用展平后的 1008 维历史窗口，
    直接预测未来 72 小时 PM2.5。
    """

    name = "random_forest"

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config, self.name)
        self.model: Any = None

    def fit(self, data: dict[str, Any]) -> None:
        print("RANDON_FOREST START")
        from sklearn.ensemble import RandomForestRegressor

        params = {
            key: value
            for key, value in self.model_config.items()
            if key
            in {
                "n_estimators",
                "max_depth",
                "min_samples_split",
                "min_samples_leaf",
                "random_state",
                "n_jobs",
            }
        }
        self.model = RandomForestRegressor(**params)
        # 输入从 (N, 168, 6) 展平到 (N, 1008)，输出仍保持 (N, 72)。
        X_train = data["X_train"].reshape(data["X_train"].shape[0], -1)
        self.model.fit(X_train, data["y_train"])

    def predict(self, data: dict[str, Any]) -> np.ndarray:
        print("RANDON_FOREST PREDICT START")
        if self.model is None:
            raise RuntimeError("Random Forest 尚未 fit。")
        X_test = data["X_test"].reshape(data["X_test"].shape[0], -1)
        return np.asarray(self.model.predict(X_test), dtype=np.float32)

    def save(self, path: str | Path) -> None:
        import joblib

        joblib.dump(self.model, Path(path))
