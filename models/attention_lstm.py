from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from models.base import BaseForecastModel


class AttentionLSTMForecastModel(BaseForecastModel):
    """Attention-LSTM 核心模型。

    LSTM 输出 168 个历史时间步的隐藏状态，Attention 层为每个时间步分配权重，
    权重与输入窗口一一对应，可用于后续可视化和论文解释。
    """

    name = "attention_lstm"

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config, self.name)
        self.network: Any = None
        self.device: Any = None
        self.attention_weights: np.ndarray | None = None
        self.training_history: list[dict[str, float | int]] = []
        self.high_value_threshold: float | None = None

    def _build_network(self) -> Any:
        from torch import nn

        class AttentionLSTMRegressor(nn.Module):
            """带时间步注意力机制的 LSTM 回归网络。"""

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
                self.attention = nn.Linear(hidden_size, 1)
                self.fc = nn.Linear(hidden_size, output_size)

            def forward(self, x):
                # outputs shape: (batch, 168, hidden)，保留每个历史小时的隐藏表示。
                outputs, _ = self.lstm(x)
                # attention 输出每个历史时间步的重要性分数，再通过 softmax 归一化为权重。
                scores = self.attention(outputs).squeeze(-1)
                weights = scores.softmax(dim=1)
                # 加权求和得到上下文向量，突出对未来 72 小时预测更重要的历史片段。
                context = (outputs * weights.unsqueeze(-1)).sum(dim=1)
                # 输出层固定为 72 维，保持直接多输出预测策略。
                predictions = self.fc(context)
                return predictions, weights

        return AttentionLSTMRegressor(
            input_size=int(self.config["global_constraints"]["feature_count"]),
            hidden_size=int(self.model_config["hidden_size"]),
            num_layers=int(self.model_config["num_layers"]),
            dropout=float(self.model_config["dropout"]),
            output_size=int(self.config["global_constraints"]["output_window_hours"]),
        )

    def fit(self, data: dict[str, Any]) -> None:
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        # Attention-LSTM 是核心深度学习模型，按项目约束必须使用 CUDA GPU。
        if not torch.cuda.is_available():
            raise RuntimeError("Attention-LSTM 训练要求 CUDA GPU，但当前 CUDA 不可用。")

        self.device = torch.device("cuda")
        self.network = self._build_network().to(self.device)
        criterion = self._build_loss()
        self.high_value_threshold = self._prepare_high_value_threshold(data)
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
                predictions, _ = self.network(batch_X)
                loss = self._compute_loss(predictions, batch_y, criterion)
                loss.backward()
                optimizer.step()
                train_losses.append(float(loss.item()))

            # 验证集用于选择最优 epoch，不参与训练更新。
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
        """构造逐元素 Huber/SmoothL1 损失。

        reduction='none' 是为了后续对高 PM2.5 标签施加更大权重，缓解峰值被均值化。
        """
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

    def _prepare_high_value_threshold(self, data: dict[str, Any]) -> float | None:
        """用训练集标签分位数确定高值样本阈值。

        阈值来自训练集 y_train，不使用验证/测试标签统计量，避免信息泄露。
        """
        weighting_cfg = self.model_config.get("high_value_weighting", {})
        if not weighting_cfg.get("enabled", False):
            return None
        quantile = float(weighting_cfg.get("quantile", 0.75))
        return float(np.nanquantile(data["y_train"], quantile))

    def _compute_loss(self, predictions: Any, targets: Any, criterion: Any) -> Any:
        """计算带高值权重的训练损失。

        普通 HuberLoss 容易让多步预测向均值收缩；这里仅对高 PM2.5 真实值提高
        损失权重，使模型在峰值区间犯错时付出更大代价。
        """
        raw_loss = criterion(predictions, targets)
        weighting_cfg = self.model_config.get("high_value_weighting", {})
        if self.high_value_threshold is None or not weighting_cfg.get("enabled", False):
            return raw_loss.mean()

        import torch

        max_weight = float(weighting_cfg.get("max_weight", 3.0))
        denominator = max(1.0 - float(self.high_value_threshold), 1e-6)
        high_ratio = torch.clamp((targets - float(self.high_value_threshold)) / denominator, min=0.0, max=1.0)
        weights = 1.0 + (max_weight - 1.0) * high_ratio
        return (raw_loss * weights).sum() / torch.clamp(weights.sum(), min=1e-6)

    def _evaluate_loss(self, loader: Any, criterion: Any) -> float:
        import torch

        self.network.eval()
        losses: list[float] = []
        # 验证损失使用同一加权准则，使 early stopping 与峰值优化目标一致。
        with torch.no_grad():
            for batch_X, batch_y in loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                predictions, _ = self.network(batch_X)
                losses.append(float(self._compute_loss(predictions, batch_y, criterion).item()))
        return float(np.mean(losses)) if losses else float("inf")

    def predict(self, data: dict[str, Any]) -> np.ndarray:
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        if self.network is None:
            raise RuntimeError("Attention-LSTM 尚未 fit。")
        self.network.eval()
        X_test = torch.tensor(data["X_test"], dtype=torch.float32)
        loader = DataLoader(
            TensorDataset(X_test),
            batch_size=int(self.model_config["batch_size"]),
            shuffle=False,
        )
        predictions: list[np.ndarray] = []
        weights: list[np.ndarray] = []
        with torch.no_grad():
            for (batch_X,) in loader:
                batch_pred, batch_weights = self.network(batch_X.to(self.device))
                predictions.append(batch_pred.detach().cpu().numpy())
                weights.append(batch_weights.detach().cpu().numpy())
        self.attention_weights = np.vstack(weights).astype(np.float32)
        return np.vstack(predictions).astype(np.float32)

    def save_attention_weights(self, path: str | Path) -> Path:
        """保存 Attention 权重，供前端和论文可视化使用。"""
        if self.attention_weights is None:
            raise RuntimeError("Attention 权重尚未生成。")
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(output_path, self.attention_weights)
        return output_path

    def save(self, path: str | Path) -> None:
        import torch

        if self.network is None:
            raise RuntimeError("Attention-LSTM 尚未 fit，无法保存。")
        # 模型文件保存网络权重和结构参数；Attention 权重单独保存为 npy。
        torch.save(
            {
                "model_name": self.name,
                "model_state_dict": self.network.state_dict(),
                "model_config": self.model_config,
                "feature_count": self.config["global_constraints"]["feature_count"],
                "output_window": self.config["global_constraints"]["output_window_hours"],
                "training_history": self.training_history,
                "high_value_threshold": self.high_value_threshold,
            },
            Path(path),
        )
