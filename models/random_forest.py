from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from models.base import BaseForecastModel
from utils.runtime import runtime_write


class RandomForestForecastModel(BaseForecastModel):
    """Random Forest 多变量多输出基线模型。"""

    name = "random_forest"

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config, self.name)
        self.model: Any = None

    def fit(self, data: dict[str, Any]) -> None:
        from sklearn.ensemble import RandomForestRegressor

        flatten_dim = int(data["X_train"].shape[1] * data["X_train"].shape[2])
        runtime_write(self.config, f"Flatten input dimension: {flatten_dim}")
        runtime_write(self.config, "Training model internals...")

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
        X_train = data["X_train"].reshape(data["X_train"].shape[0], -1)
        self.model.fit(X_train, data["y_train"])

    def predict(self, data: dict[str, Any]) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Random Forest 尚未 fit。")
        runtime_write(self.config, "Predicting with flattened windows...")
        X_test = data["X_test"].reshape(data["X_test"].shape[0], -1)
        return np.asarray(self.model.predict(X_test), dtype=np.float32)

    def save(self, path: str | Path) -> None:
        import joblib

        joblib.dump(self.model, Path(path))
