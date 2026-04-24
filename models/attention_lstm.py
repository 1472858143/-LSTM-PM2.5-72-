from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from models.base import BaseForecastModel


def weighted_mse_loss(y_pred: Any, y_true: Any, q75: float, q90: float):
    """Apply weighted MSE on multi-step targets with shape (B, 72)."""
    import torch

    weights = torch.ones_like(y_true)
    weights = torch.where(y_true >= q75, torch.full_like(y_true, 3.0), weights)
    weights = torch.where(y_true >= q90, torch.full_like(y_true, 8.0), weights)
    loss = torch.mean(weights * torch.square(y_pred - y_true))
    return loss, weights


class AttentionLSTMForecastModel(BaseForecastModel):
    """Attention-LSTM forecaster with query-based additive attention."""

    name = "attention_lstm"

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config, self.name)
        self.network: Any = None
        self.device: Any = None
        self.attention_weights: np.ndarray | None = None
        self.training_history: list[dict[str, float | int]] = []
        self.weighted_loss_thresholds: dict[str, float] | None = None
        self.high_value_threshold: dict[str, float] | None = None

    def _build_network(self) -> Any:
        import torch
        from torch import nn

        class AttentionLSTMRegressor(nn.Module):
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
                self.key_proj = nn.Linear(hidden_size, hidden_size)
                self.query_proj = nn.Linear(hidden_size, hidden_size, bias=False)
                self.attention_score = nn.Linear(hidden_size, 1, bias=False)
                self.head = nn.Sequential(
                    nn.Linear(hidden_size * 2, hidden_size),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_size, output_size),
                )

            def forward(self, x):
                outputs, (hidden, _) = self.lstm(x)
                last_hidden = hidden[-1]
                query = self.query_proj(last_hidden).unsqueeze(1)
                scores = self.attention_score(torch.tanh(self.key_proj(outputs) + query)).squeeze(-1)
                weights = scores.softmax(dim=1)
                context = (outputs * weights.unsqueeze(-1)).sum(dim=1)
                combined = torch.cat([context, last_hidden], dim=1)
                predictions = self.head(combined)
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

        if not torch.cuda.is_available():
            raise RuntimeError("Attention-LSTM 训练要求 CUDA GPU，但当前 CUDA 不可用。")

        self.device = torch.device("cuda")
        self.network = self._build_network().to(self.device)
        self.training_history = []
        self.weighted_loss_thresholds = self._prepare_weighted_loss_thresholds(data)
        self.high_value_threshold = self.weighted_loss_thresholds
        print(
            f"Attention-LSTM weighted loss thresholds: "
            f"q75={self.weighted_loss_thresholds['q75']:.6f}, "
            f"q90={self.weighted_loss_thresholds['q90']:.6f}"
        )
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
            train_mean_weights: list[float] = []
            train_peak_ratios: list[float] = []

            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                optimizer.zero_grad()
                predictions, _ = self.network(batch_X)
                loss, weight_stats = self._compute_loss(predictions, batch_y)
                loss.backward()
                optimizer.step()

                train_losses.append(float(loss.item()))
                train_mean_weights.append(weight_stats["mean_weight"])
                train_peak_ratios.append(weight_stats["peak_ratio"])

            val_loss, val_mean_weight, val_peak_ratio = self._evaluate_loss(val_loader)
            epoch = len(self.training_history) + 1
            history_row = {
                "epoch": epoch,
                "train_loss": float(np.mean(train_losses)) if train_losses else float("nan"),
                "validation_loss": float(val_loss),
                "best_validation_loss": float(min(best_val_loss, val_loss)),
                "bad_epochs": bad_epochs,
                "q75": float(self.weighted_loss_thresholds["q75"]),
                "q90": float(self.weighted_loss_thresholds["q90"]),
                "train_mean_weight": float(np.mean(train_mean_weights)) if train_mean_weights else float("nan"),
                "validation_mean_weight": float(val_mean_weight),
                "train_peak_ratio": float(np.mean(train_peak_ratios)) if train_peak_ratios else float("nan"),
                "validation_peak_ratio": float(val_peak_ratio),
            }

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                history_row["best_validation_loss"] = float(best_val_loss)
                best_state = {k: v.detach().cpu().clone() for k, v in self.network.state_dict().items()}
                bad_epochs = 0
                history_row["bad_epochs"] = bad_epochs
            else:
                bad_epochs += 1
                history_row["bad_epochs"] = bad_epochs

            self.training_history.append(history_row)
            if bad_epochs >= patience:
                break

        if best_state is not None:
            self.network.load_state_dict(best_state)

    def _prepare_weighted_loss_thresholds(self, data: dict[str, Any]) -> dict[str, float]:
        """Estimate q75/q90 from raw training PM2.5 only, without leakage."""
        target_column = data["target_column"]
        train_target = np.asarray(data["splits_raw"]["train"][target_column], dtype=float)
        scaler = data["scaler"]
        return {
            "q75": float(np.nanquantile(train_target, 0.75)),
            "q90": float(np.nanquantile(train_target, 0.90)),
            "target_min": float(scaler.data_min_[target_column]),
            "target_max": float(scaler.data_max_[target_column]),
        }

    def _inverse_transform_target_tensor(self, values: Any) -> Any:
        if self.weighted_loss_thresholds is None:
            raise RuntimeError("加权 MSE 阈值尚未初始化。")
        target_min = float(self.weighted_loss_thresholds["target_min"])
        target_range = max(
            float(self.weighted_loss_thresholds["target_max"]) - target_min,
            1e-12,
        )
        return values * target_range + target_min

    def _compute_loss(self, predictions: Any, targets: Any) -> tuple[Any, dict[str, float]]:
        if self.weighted_loss_thresholds is None:
            raise RuntimeError("加权 MSE 阈值尚未初始化。")

        predictions_raw = self._inverse_transform_target_tensor(predictions)
        targets_raw = self._inverse_transform_target_tensor(targets)
        loss, weights = weighted_mse_loss(
            predictions_raw,
            targets_raw,
            float(self.weighted_loss_thresholds["q75"]),
            float(self.weighted_loss_thresholds["q90"]),
        )
        peak_ratio = (weights == 8.0).to(dtype=predictions_raw.dtype).mean()
        return loss, {
            "mean_weight": float(weights.mean().item()),
            "peak_ratio": float(peak_ratio.item()),
        }

    def _evaluate_loss(self, loader: Any) -> tuple[float, float, float]:
        import torch

        self.network.eval()
        losses: list[float] = []
        mean_weights: list[float] = []
        peak_ratios: list[float] = []
        with torch.no_grad():
            for batch_X, batch_y in loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                predictions, _ = self.network(batch_X)
                loss, weight_stats = self._compute_loss(predictions, batch_y)
                losses.append(float(loss.item()))
                mean_weights.append(weight_stats["mean_weight"])
                peak_ratios.append(weight_stats["peak_ratio"])
        return (
            float(np.mean(losses)) if losses else float("inf"),
            float(np.mean(mean_weights)) if mean_weights else float("nan"),
            float(np.mean(peak_ratios)) if peak_ratios else float("nan"),
        )

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
        torch.save(
            {
                "model_name": self.name,
                "model_state_dict": self.network.state_dict(),
                "model_config": self.model_config,
                "feature_count": self.config["global_constraints"]["feature_count"],
                "output_window": self.config["global_constraints"]["output_window_hours"],
                "training_history": self.training_history,
                "weighted_loss_thresholds": self.weighted_loss_thresholds,
                "high_value_threshold": self.high_value_threshold,
            },
            Path(path),
        )
