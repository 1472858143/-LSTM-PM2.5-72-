from __future__ import annotations

import itertools
import pickle
from pathlib import Path
from typing import Any
import warnings

import numpy as np

from models.base import BaseForecastModel


class SARIMAForecastModel(BaseForecastModel):
    """SARIMA 单变量季节性基线模型。

    SARIMA 与 ARIMA 一样只使用 pm2_5，但额外建模 24 小时日周期，
    用于判断显式季节性结构对 72 小时预测的贡献。
    """

    name = "sarima"

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config, self.name)
        self.order: tuple[int, int, int] | None = None
        self.seasonal_order: tuple[int, int, int, int] | None = None
        self.selection_score: float | None = None

    def fit(self, data: dict[str, Any]) -> None:
        from statsmodels.tsa.statespace.sarimax import SARIMAX

        # 只在训练集上做阶数搜索，保持测试集独立。
        train_series = data["splits_scaled"]["train"][self.config["data"]["target"]].dropna().to_numpy(dtype=float)
        max_points = int(self.model_config["selection_train_points"])
        if len(train_series) > max_points:
            train_series = train_series[-max_points:]

        period = int(self.model_config["seasonal_period"])
        criterion = self.model_config.get("selection_criterion", "aic").lower()
        best_score = np.inf
        best_order: tuple[int, int, int] | None = None
        best_seasonal: tuple[int, int, int, int] | None = None

        orders = itertools.product(
            self.model_config["p_values"],
            self.model_config["d_values"],
            self.model_config["q_values"],
        )
        seasonal_orders = list(
            itertools.product(
                self.model_config["P_values"],
                self.model_config["D_values"],
                self.model_config["Q_values"],
            )
        )

        # 非季节项和季节项搜索范围均来自 config，季节周期固定为 24 小时。
        for order in orders:
            for seasonal in seasonal_orders:
                if order == (0, 0, 0) and seasonal == (0, 0, 0):
                    continue
                seasonal_order = tuple(int(v) for v in seasonal) + (period,)
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
                    score = getattr(result, criterion)
                    if score < best_score:
                        best_score = float(score)
                        best_order = tuple(int(v) for v in order)
                        best_seasonal = seasonal_order
                except Exception:
                    continue

        if best_order is None or best_seasonal is None:
            best_order = (1, 0, 0)
            best_seasonal = (0, 0, 0, period)
            best_score = float("nan")

        self.order = best_order
        self.seasonal_order = best_seasonal
        self.selection_score = best_score

    def predict(self, data: dict[str, Any]) -> np.ndarray:
        if self.order is None or self.seasonal_order is None:
            raise RuntimeError("SARIMA 尚未 fit。")

        from statsmodels.tsa.statespace.sarimax import SARIMAX

        target_index = data["feature_columns"].index(self.config["data"]["target"])
        X_test = data["X_test"]
        horizon = int(self.model_config["forecast_horizon"])
        predictions: list[np.ndarray] = []

        for sample in X_test:
            # 从统一窗口中抽取 pm2_5 列，保证 SARIMA 不使用任何气象特征。
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
                # 个别窗口拟合失败时回退到持久性预测，避免影响结果文件生成。
                forecast = np.repeat(series[-1], horizon).astype(float)
            predictions.append(forecast)

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
