from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from models.base import BaseForecastModel


class LSTMForecastModel(BaseForecastModel):
    """基础 LSTM 多变量序列模型。

    输入保持为 (batch, 168, 6)，LSTM 使用最后时间步隐藏状态表示过去一周信息，
    全连接层直接输出未来 72 小时预测。
    """

    name = "lstm"

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config, self.name)
        self.network: Any = None
        self.device: Any = None

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
                self.fc = nn.Linear(hidden_size, output_size)

            def forward(self, x):
                # hidden[-1] 是最后一层 LSTM 的最终隐藏状态，作为 168 小时历史信息编码。
                _, (hidden, _) = self.lstm(x)
                # 输出层为 72 维，对应未来第 1 到第 72 小时的直接多输出预测。
                return self.fc(hidden[-1])

        return LSTMRegressor(
            input_size=int(self.config["global_constraints"]["feature_count"]),
            hidden_size=int(self.model_config["hidden_size"]),
            num_layers=int(self.model_config["num_layers"]),
            dropout=float(self.model_config["dropout"]),
            output_size=int(self.config["global_constraints"]["output_window_hours"]),
        )

    def fit(self, data: dict[str, Any]) -> None:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset

        # 项目书要求深度学习模型使用 GPU，避免不同设备导致实验条件不一致。
        if not torch.cuda.is_available():
            raise RuntimeError("LSTM 训练要求 CUDA GPU，但当前 CUDA 不可用。")

        self.device = torch.device("cuda")
        self.network = self._build_network().to(self.device)
        criterion = nn.MSELoss()
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
            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                optimizer.zero_grad()
                loss = criterion(self.network(batch_X), batch_y)
                loss.backward()
                optimizer.step()

            # 验证集只用于 early stopping，不参与参数更新。
            val_loss = self._evaluate_loss(val_loader, criterion)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                # 保存验证集最优权重，训练结束后恢复，降低过拟合风险。
                best_state = {k: v.detach().cpu().clone() for k, v in self.network.state_dict().items()}
                bad_epochs = 0
            else:
                bad_epochs += 1
                if bad_epochs >= patience:
                    break

        if best_state is not None:
            self.network.load_state_dict(best_state)

    def _evaluate_loss(self, loader: Any, criterion: Any) -> float:
        import torch

        self.network.eval()
        losses: list[float] = []
        with torch.no_grad():
            for batch_X, batch_y in loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                losses.append(float(criterion(self.network(batch_X), batch_y).item()))
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
            },
            Path(path),
        )
