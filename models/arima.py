from __future__ import annotations

import itertools
import pickle
import warnings
from pathlib import Path
from typing import Any

import numpy as np

from models.base import BaseForecastModel
from utils.runtime import runtime_add_task, runtime_label, runtime_remove_task, runtime_update_task, runtime_write


class ARIMAForecastModel(BaseForecastModel):
    """ARIMA single-variable baseline model."""

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
        orders = [
            order
            for order in itertools.product(
                self.model_config["p_values"],
                self.model_config["d_values"],
                self.model_config["q_values"],
            )
            if order != (0, 0, 0)
        ]

        search_task_id = runtime_add_task(
            self.config,
            f"{runtime_label(self.config)} Search",
            total=len(orders),
            stats="best=- order=-",
        )
        try:
            for order in orders:
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        result = ARIMA(train_series, order=order).fit()
                    score = float(getattr(result, criterion))
                    if score < best_score:
                        best_score = score
                        best_order = tuple(int(v) for v in order)
                except Exception:
                    pass

                runtime_update_task(
                    self.config,
                    search_task_id,
                    advance=1,
                    stats=(
                        f"best={best_score:.6f} order={best_order or order}"
                        if np.isfinite(best_score)
                        else f"best=- order={order}"
                    ),
                )
        finally:
            runtime_remove_task(self.config, search_task_id)

        if best_order is None:
            best_order = (1, 0, 0)
            best_score = float("nan")

        self.order = best_order
        self.selection_score = best_score
        runtime_write(self.config, f"Selected order={self.order}, {criterion}={self.selection_score}")

    def predict(self, data: dict[str, Any]) -> np.ndarray:
        if self.order is None:
            raise RuntimeError("ARIMA has not been fit yet.")

        from statsmodels.tsa.arima.model import ARIMA

        runtime_write(self.config, "Rolling forecast start...")
        target_index = data["feature_columns"].index(self.config["data"]["target"])
        X_test = data["X_test"]
        horizon = int(self.model_config["forecast_horizon"])
        predictions: list[np.ndarray] = []

        predict_task_id = runtime_add_task(
            self.config,
            f"{runtime_label(self.config)} Predict",
            total=len(X_test),
            stats="",
        )
        try:
            for sample in X_test:
                series = sample[:, target_index].astype(float)
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        result = ARIMA(series, order=self.order).fit()
                    forecast = np.asarray(result.forecast(steps=horizon), dtype=float)
                except Exception:
                    forecast = np.repeat(series[-1], horizon).astype(float)
                predictions.append(forecast)
                runtime_update_task(self.config, predict_task_id, advance=1)
        finally:
            runtime_remove_task(self.config, predict_task_id)

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
