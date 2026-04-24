from __future__ import annotations

import itertools
import pickle
from pathlib import Path
from typing import Any
import warnings

import numpy as np

from models.base import BaseForecastModel


class ARIMAForecastModel(BaseForecastModel):
    """ARIMA 单变量基线模型。

    按项目书要求，ARIMA 只使用历史 pm2_5 序列，不引入气象变量；
    每个测试窗口都直接 forecast 72 步，用于和多变量模型公平对比。
    """

    name = "arima"

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config, self.name)
        self.order: tuple[int, int, int] | None = None
        self.selection_score: float | None = None

    def fit(self, data: dict[str, Any]) -> None:
        print("ARIMA START")
        from statsmodels.tsa.arima.model import ARIMA

        # 选阶只使用训练集中的 pm2_5，避免验证集和测试集信息参与调参。
        train_series = data["splits_scaled"]["train"][self.config["data"]["target"]].dropna().to_numpy(dtype=float)
        max_points = int(self.model_config["selection_train_points"])
        if len(train_series) > max_points:
            train_series = train_series[-max_points:]

        best_score = np.inf
        best_order: tuple[int, int, int] | None = None
        criterion = self.model_config.get("selection_criterion", "aic").lower()

        # p/d/q 搜索范围来自配置文件，不在代码中硬编码实验参数。
        for order in itertools.product(
            self.model_config["p_values"],
            self.model_config["d_values"],
            self.model_config["q_values"],
        ):
            if order == (0, 0, 0):
                continue
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    result = ARIMA(train_series, order=order).fit()
                score = getattr(result, criterion)
                if score < best_score:
                    best_score = float(score)
                    best_order = tuple(int(v) for v in order)
            except Exception:
                continue

        if best_order is None:
            best_order = (1, 0, 0)
            best_score = float("nan")

        self.order = best_order
        self.selection_score = best_score

    def predict(self, data: dict[str, Any]) -> np.ndarray:
        print("ARIMA PREDICT START")
        if self.order is None:
            raise RuntimeError("ARIMA 尚未 fit。")

        from statsmodels.tsa.arima.model import ARIMA

        target_index = data["feature_columns"].index(self.config["data"]["target"])
        X_test = data["X_test"]
        horizon = int(self.model_config["forecast_horizon"])
        predictions: list[np.ndarray] = []

        for sample in X_test:
            # ARIMA 是单变量模型，从 168x6 窗口中取出 pm2_5 这一列进行预测。
            series = sample[:, target_index].astype(float)
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    result = ARIMA(series, order=self.order).fit()
                forecast = np.asarray(result.forecast(steps=horizon), dtype=float)
            except Exception:
                # 个别窗口拟合失败时使用最后观测值作为保底，保证统一输出结构不断裂。
                forecast = np.repeat(series[-1], horizon).astype(float)
            predictions.append(forecast)

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
