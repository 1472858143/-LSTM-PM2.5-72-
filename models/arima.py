from __future__ import annotations

import itertools
import pickle
import warnings
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from models.base import BaseForecastModel
from utils.runtime import runtime_label, runtime_write


class ARIMAForecastModel(BaseForecastModel):
    """ARIMA 单变量基线模型。"""

    name = "arima"

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config, self.name)
        self.order: tuple[int, int, int] | None = None
        self.selection_score: float | None = None

    def fit(self, data: dict[str, Any]) -> None:
        from statsmodels.tsa.arima.model import ARIMA

        runtime_write(self.config, "Parameter selection start...")
        train_series = data["splits_scaled"]["train"][self.config["data"]["target"]].dropna().to_numpy(dtype=float)
        max_points = int(self.model_config["selection_train_points"])
        if len(train_series) > max_points:
            train_series = train_series[-max_points:]

        best_score = np.inf
        best_order: tuple[int, int, int] | None = None
        criterion = self.model_config.get("selection_criterion", "aic").lower()
        orders = [order for order in itertools.product(
            self.model_config["p_values"],
            self.model_config["d_values"],
            self.model_config["q_values"],
        ) if order != (0, 0, 0)]

        progress = tqdm(orders, desc=f"{runtime_label(self.config)} ARIMA search", leave=False, dynamic_ncols=True)
        for order in progress:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    result = ARIMA(train_series, order=order).fit()
                score = float(getattr(result, criterion))
                progress.set_postfix({"best": best_score if np.isfinite(best_score) else None, "order": order})
                if score < best_score:
                    best_score = score
                    best_order = tuple(int(v) for v in order)
            except Exception:
                continue
        progress.close()

        if best_order is None:
            best_order = (1, 0, 0)
            best_score = float("nan")

        self.order = best_order
        self.selection_score = best_score
        runtime_write(self.config, f"Selected order={self.order}, {criterion}={self.selection_score}")

    def predict(self, data: dict[str, Any]) -> np.ndarray:
        if self.order is None:
            raise RuntimeError("ARIMA 尚未 fit。")

        from statsmodels.tsa.arima.model import ARIMA

        runtime_write(self.config, "Rolling forecast start...")
        target_index = data["feature_columns"].index(self.config["data"]["target"])
        X_test = data["X_test"]
        horizon = int(self.model_config["forecast_horizon"])
        predictions: list[np.ndarray] = []

        progress = tqdm(X_test, desc=f"{runtime_label(self.config)} ARIMA predict", leave=False, dynamic_ncols=True)
        for sample in progress:
            series = sample[:, target_index].astype(float)
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    result = ARIMA(series, order=self.order).fit()
                forecast = np.asarray(result.forecast(steps=horizon), dtype=float)
            except Exception:
                forecast = np.repeat(series[-1], horizon).astype(float)
            predictions.append(forecast)
        progress.close()

        return np.asarray(predictions, dtype=np.float32)

    def save(self, path: str | Path) -> None:
        with Path(path).open("wb") as f:
            pickle.dump(
                {
                    "model_name": self.name,
                    "order": self.order,
                    "selection_score": self.selection_score,
                    "config": self.model_config,
                },
                f,
            )
