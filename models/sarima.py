from __future__ import annotations

import itertools
import pickle
import warnings
from pathlib import Path
from typing import Any

import numpy as np

from models.base import BaseForecastModel
from utils.runtime import runtime_add_task, runtime_label, runtime_remove_task, runtime_update_task, runtime_write


class SARIMAForecastModel(BaseForecastModel):
    """SARIMA seasonal single-variable baseline model."""

    name = "sarima"

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config, self.name)
        self.order: tuple[int, int, int] | None = None
        self.seasonal_order: tuple[int, int, int, int] | None = None
        self.selection_score: float | None = None

    def fit(self, data: dict[str, Any]) -> None:
        from statsmodels.tsa.statespace.sarimax import SARIMAX

        runtime_write(self.config, "Parameter selection start...")
        train_series = data["splits_scaled"]["train"][self.config["data"]["target"]].dropna().to_numpy(dtype=float)
        max_points = int(self.model_config["selection_train_points"])
        if len(train_series) > max_points:
            train_series = train_series[-max_points:]

        period = int(self.model_config["seasonal_period"])
        criterion = self.model_config.get("selection_criterion", "aic").lower()
        best_score = np.inf
        best_order: tuple[int, int, int] | None = None
        best_seasonal: tuple[int, int, int, int] | None = None

        orders = list(
            itertools.product(
                self.model_config["p_values"],
                self.model_config["d_values"],
                self.model_config["q_values"],
            )
        )
        seasonal_orders = list(
            itertools.product(
                self.model_config["P_values"],
                self.model_config["D_values"],
                self.model_config["Q_values"],
            )
        )

        combos: list[tuple[tuple[int, int, int], tuple[int, int, int, int]]] = []
        for order in orders:
            for seasonal in seasonal_orders:
                if order == (0, 0, 0) and seasonal == (0, 0, 0):
                    continue
                combos.append((tuple(int(v) for v in order), tuple(int(v) for v in seasonal) + (period,)))

        search_task_id = runtime_add_task(
            self.config,
            f"{runtime_label(self.config)} Search",
            total=len(combos),
            stats="best=- order=-",
        )
        try:
            for order, seasonal_order in combos:
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        result = SARIMAX(
                            train_series,
                            order=order,
                            seasonal_order=seasonal_order,
                            enforce_stationarity=False,
                            enforce_invertibility=False,
                        ).fit(disp=False)
                    score = float(getattr(result, criterion))
                    if score < best_score:
                        best_score = score
                        best_order = order
                        best_seasonal = seasonal_order
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

        if best_order is None or best_seasonal is None:
            best_order = (1, 0, 0)
            best_seasonal = (0, 0, 0, period)
            best_score = float("nan")

        self.order = best_order
        self.seasonal_order = best_seasonal
        self.selection_score = best_score
        runtime_write(
            self.config,
            f"Selected order={self.order}, seasonal_order={self.seasonal_order}, {criterion}={self.selection_score}",
        )

    def predict(self, data: dict[str, Any]) -> np.ndarray:
        if self.order is None or self.seasonal_order is None:
            raise RuntimeError("SARIMA has not been fit yet.")

        from statsmodels.tsa.statespace.sarimax import SARIMAX

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
                        result = SARIMAX(
                            series,
                            order=self.order,
                            seasonal_order=self.seasonal_order,
                            enforce_stationarity=False,
                            enforce_invertibility=False,
                        ).fit(disp=False)
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
                    "seasonal_order": self.seasonal_order,
                    "selection_score": self.selection_score,
                    "config": self.model_config,
                },
                f,
            )
