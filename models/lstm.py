from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from models.base import BaseForecastModel


class LSTMForecastModel(BaseForecastModel):
    """基础 LSTM 多变量序列模型。

    相比只使用最后隐藏状态的最简版本，这里保留 LSTM 主体不变，只在输出头上
    增加一个轻量序列表达汇聚：将最后隐藏状态与时间维最大池化结果拼接，
    以减轻长窗口信息压缩过强带来的振幅压缩问题。
    """

    name = "lstm"

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config, self.name)
        self.network: Any = None
        self.device: Any = None
        self.training_history: list[dict[str, float | int]] = []
        self.high_value_thresholds: dict[str, float] | None = None

    def _build_network(self) -> Any:
        import torch
        from torch import nn

        class LSTMRegressor(nn.Module):
            """PyTorch LSTM 回归网络，输出维度固定为 72。"""

            def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float, output_size: int):
                super().__init__()
                lstm_dropout = dropout if num_layers > 1 else 0.0
                self.lstm = nn.LSTM(
                    input_size=input_size,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    dropout=lstm_dropout,
                    batch_first=True,
                )
                self.head = nn.Sequential(
                    nn.Linear(hidden_size * 2, hidden_size),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_size, output_size),
                )

            def forward(self, x):
                # outputs 保留 168 个历史时间步的表示，hidden[-1] 是最后时间步摘要。
                outputs, (hidden, _) = self.lstm(x)
                last_hidden = hidden[-1]
                # 时间维最大池化能够更容易保留历史强响应，对峰值低估更友好。
                max_pool = outputs.max(dim=1).values
                features = torch.cat([last_hidden, max_pool], dim=1)
                # 输出层仍然直接给出未来 72 小时预测，不改变任务定义。
                return self.head(features)

        return LSTMRegressor(
            input_size=int(self.config["global_constraints"]["feature_count"]),
            hidden_size=int(self.model_config["hidden_size"]),
            num_layers=int(self.model_config["num_layers"]),
            dropout=float(self.model_config["dropout"]),
            output_size=int(self.config["global_constraints"]["output_window_hours"]),
        )

    def fit(self, data: dict[str, Any]) -> None:
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        # 项目书要求深度学习模型使用 GPU，避免不同设备导致实验条件不一致。
        if not torch.cuda.is_available():
            raise RuntimeError("LSTM 训练要求 CUDA GPU，但当前 CUDA 不可用。")

        self.device = torch.device("cuda")
        self.network = self._build_network().to(self.device)
        criterion = self._build_loss()
        self.high_value_thresholds = self._prepare_high_value_thresholds(data)
        optimizer = torch.optim.Adam(self.network.parameters(), lr=float(self.model_config["learning_rate"]))

        X_train = torch.tensor(data["X_train"], dtype=torch.float32)
        y_train = torch.tensor(data["y_train"], dtype=torch.float32)
        X_val = torch.tensor(data["X_validation"], dtype=torch.float32)
        y_val = torch.tensor(data["y_validation"], dtype=torch.float32)

        train_loader = DataLoader(
            TensorDataset(X_train, y_train),
            batch_size=int(self.model_config["batch_size"]),
            shuffle=False,
        )
        val_loader = DataLoader(
            TensorDataset(X_val, y_val),
            batch_size=int(self.model_config["batch_size"]),
            shuffle=False,
        )

        best_state = None
        best_val_loss = float("inf")
        patience = int(self.model_config["early_stopping_patience"])
        bad_epochs = 0

        for _ in range(int(self.model_config["epochs"])):
            self.network.train()
            train_losses: list[float] = []
            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                optimizer.zero_grad()
                predictions = self.network(batch_X)
                loss = self._compute_loss(predictions, batch_y, criterion)
                loss.backward()
                optimizer.step()
                train_losses.append(float(loss.item()))

            # 验证集只用于 early stopping，不参与参数更新。
            val_loss = self._evaluate_loss(val_loader, criterion)
            epoch = len(self.training_history) + 1
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.detach().cpu().clone() for k, v in self.network.state_dict().items()}
                bad_epochs = 0
            else:
                bad_epochs += 1
                if bad_epochs >= patience:
                    self.training_history.append(
                        {
                            "epoch": epoch,
                            "train_loss": float(np.mean(train_losses)) if train_losses else float("nan"),
                            "validation_loss": float(val_loss),
                            "best_validation_loss": float(best_val_loss),
                            "bad_epochs": bad_epochs,
                        }
                    )
                    break

            self.training_history.append(
                {
                    "epoch": epoch,
                    "train_loss": float(np.mean(train_losses)) if train_losses else float("nan"),
                    "validation_loss": float(val_loss),
                    "best_validation_loss": float(best_val_loss),
                    "bad_epochs": bad_epochs,
                }
            )

        if best_state is not None:
            self.network.load_state_dict(best_state)

    def _build_loss(self) -> Any:
        """构造逐元素 Huber/SmoothL1 损失。"""
        from torch import nn

        loss_name = str(self.model_config.get("loss", "HuberLoss")).lower()
        delta = float(self.model_config.get("huber_delta", 1.0))
        if loss_name == "smoothl1loss":
            return nn.SmoothL1Loss(reduction="none")
        try:
            return nn.HuberLoss(delta=delta, reduction="none")
        except AttributeError:
            return nn.SmoothL1Loss(reduction="none")
        except TypeError:
            return nn.SmoothL1Loss(reduction="none")

    def _prepare_high_value_thresholds(self, data: dict[str, Any]) -> dict[str, float] | None:
        """基于训练集标签分位数构造高值样本阈值。"""
        weighting_cfg = self.model_config.get("high_value_weighting", {})
        if not weighting_cfg.get("enabled", False):
            return None
        mid_quantile = float(weighting_cfg.get("mid_quantile", 0.75))
        high_quantile = float(weighting_cfg.get("high_quantile", 0.9))
        y_train = data["y_train"]
        return {
            "mid": float(np.nanquantile(y_train, mid_quantile)),
            "high": float(np.nanquantile(y_train, high_quantile)),
        }

    def _compute_loss(self, predictions: Any, targets: Any, criterion: Any) -> Any:
        """在基础回归损失上对高值 PM2.5 区间增加权重。"""
        raw_loss = criterion(predictions, targets)
        weighting_cfg = self.model_config.get("high_value_weighting", {})
        if self.high_value_thresholds is None or not weighting_cfg.get("enabled", False):
            return raw_loss.mean()

        import torch

        mid_weight = float(weighting_cfg.get("mid_weight", 2.0))
        high_weight = float(weighting_cfg.get("high_weight", 3.0))
        weights = torch.ones_like(targets)
        weights = torch.where(
            targets > float(self.high_value_thresholds["mid"]),
            torch.full_like(targets, mid_weight),
            weights,
        )
        weights = torch.where(
            targets > float(self.high_value_thresholds["high"]),
            torch.full_like(targets, high_weight),
            weights,
        )
        return (raw_loss * weights).sum() / torch.clamp(weights.sum(), min=1e-6)

    def _evaluate_loss(self, loader: Any, criterion: Any) -> float:
        import torch

        self.network.eval()
        losses: list[float] = []
        with torch.no_grad():
            for batch_X, batch_y in loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                losses.append(float(self._compute_loss(self.network(batch_X), batch_y, criterion).item()))
        return float(np.mean(losses)) if losses else float("inf")

    def predict(self, data: dict[str, Any]) -> np.ndarray:
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        if self.network is None:
            raise RuntimeError("LSTM 尚未 fit。")
        self.network.eval()
        X_test = torch.tensor(data["X_test"], dtype=torch.float32)
        loader = DataLoader(
            TensorDataset(X_test),
            batch_size=int(self.model_config["batch_size"]),
            shuffle=False,
        )
        predictions: list[np.ndarray] = []
        # 测试阶段不计算梯度，只生成统一的 (N, 72) 预测数组。
        with torch.no_grad():
            for (batch_X,) in loader:
                batch_pred = self.network(batch_X.to(self.device)).detach().cpu().numpy()
                predictions.append(batch_pred)
        return np.vstack(predictions).astype(np.float32)

    def save(self, path: str | Path) -> None:
        import torch

        if self.network is None:
            raise RuntimeError("LSTM 尚未 fit，无法保存。")
        # 保存 state_dict 和关键结构参数，便于后续复现实验或加载推理。
        torch.save(
            {
                "model_name": self.name,
                "model_state_dict": self.network.state_dict(),
                "model_config": self.model_config,
                "feature_count": self.config["global_constraints"]["feature_count"],
                "output_window": self.config["global_constraints"]["output_window_hours"],
                "training_history": self.training_history,
                "high_value_thresholds": self.high_value_thresholds,
            },
            Path(path),
        )
