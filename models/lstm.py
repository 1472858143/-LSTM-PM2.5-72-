from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from models.base import BaseForecastModel
from utils.runtime import runtime_add_task, runtime_label, runtime_remove_task, runtime_update_task, runtime_write


def weighted_mse_loss(y_pred: Any, y_true: Any, q75: float, q90: float):
    """Apply weighted MSE to multi-step targets with shape (B, 72)."""
    import torch

    weights = torch.ones_like(y_true)
    weights = torch.where(y_true >= q75, torch.full_like(y_true, 3.0), weights)
    weights = torch.where(y_true >= q90, torch.full_like(y_true, 8.0), weights)
    loss = torch.mean(weights * torch.square(y_pred - y_true))
    return loss, weights


def _metric_text(value: float) -> str:
    return "-" if not np.isfinite(value) else f"{value:.6f}"


def _format_epoch_stats(
    epoch: int,
    total_epochs: int,
    train_loss: float,
    val_loss: float,
    best_val_loss: float,
    bad_epochs: int,
    patience: int,
    current_lr: float,
) -> str:
    return (
        f"epoch={epoch}/{total_epochs} "
        f"train={_metric_text(train_loss)} "
        f"val={_metric_text(val_loss)} "
        f"best={_metric_text(best_val_loss)} "
        f"patience={bad_epochs}/{patience} "
        f"lr={current_lr:.6g}"
    )


class LSTMForecastModel(BaseForecastModel):
    """Core LSTM multi-variate multi-step forecasting model."""

    name = "lstm"

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config, self.name)
        self.network: Any = None
        self.device: Any = None
        self.training_history: list[dict[str, float | int]] = []
        self.weighted_loss_thresholds: dict[str, float] | None = None
        self.high_value_thresholds: dict[str, float] | None = None

    def _build_network(self) -> Any:
        import torch
        from torch import nn

        class LSTMRegressor(nn.Module):
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
                outputs, (hidden, _) = self.lstm(x)
                last_hidden = hidden[-1]
                max_pool = outputs.max(dim=1).values
                features = torch.cat([last_hidden, max_pool], dim=1)
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

        if not torch.cuda.is_available():
            raise RuntimeError("LSTM training requires a CUDA GPU, but CUDA is unavailable.")

        self.device = torch.device("cuda")
        self.network = self._build_network().to(self.device)
        self.training_history = []
        self.weighted_loss_thresholds = self._prepare_weighted_loss_thresholds(data)
        self.high_value_thresholds = self.weighted_loss_thresholds
        runtime_write(
            self.config,
            f"q75={self.weighted_loss_thresholds['q75']:.6f}, q90={self.weighted_loss_thresholds['q90']:.6f}",
        )

        optimizer = torch.optim.Adam(self.network.parameters(), lr=float(self.model_config["learning_rate"]))
        current_lr = float(self.model_config["learning_rate"])

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
        total_epochs = int(self.model_config["epochs"])
        patience = int(self.model_config["early_stopping_patience"])
        bad_epochs = 0
        epoch_task_id = runtime_add_task(
            self.config,
            f"{runtime_label(self.config)} Epochs",
            total=total_epochs,
            stats=_format_epoch_stats(0, total_epochs, float("nan"), float("nan"), float("nan"), 0, patience, current_lr),
        )

        try:
            for epoch_idx in range(total_epochs):
                self.network.train()
                train_losses: list[float] = []
                train_mean_weights: list[float] = []
                train_peak_ratios: list[float] = []

                for batch_X, batch_y in train_loader:
                    batch_X = batch_X.to(self.device)
                    batch_y = batch_y.to(self.device)
                    optimizer.zero_grad()
                    predictions = self.network(batch_X)
                    loss, weight_stats = self._compute_loss(predictions, batch_y)
                    loss.backward()
                    optimizer.step()

                    train_losses.append(float(loss.item()))
                    train_mean_weights.append(weight_stats["mean_weight"])
                    train_peak_ratios.append(weight_stats["peak_ratio"])

                val_loss, val_mean_weight, val_peak_ratio = self._evaluate_loss(val_loader)
                epoch = epoch_idx + 1
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
                runtime_update_task(
                    self.config,
                    epoch_task_id,
                    advance=1,
                    stats=_format_epoch_stats(
                        epoch,
                        total_epochs,
                        float(history_row["train_loss"]),
                        float(history_row["validation_loss"]),
                        float(best_val_loss),
                        bad_epochs,
                        patience,
                        current_lr,
                    ),
                )

                if bad_epochs >= patience:
                    break
        finally:
            runtime_remove_task(self.config, epoch_task_id)

        if best_state is not None:
            self.network.load_state_dict(best_state)

    def _prepare_weighted_loss_thresholds(self, data: dict[str, Any]) -> dict[str, float]:
        """Compute q75/q90 only from the raw training PM2.5 series."""
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
            raise RuntimeError("Weighted loss thresholds have not been initialized.")
        target_min = float(self.weighted_loss_thresholds["target_min"])
        target_range = max(float(self.weighted_loss_thresholds["target_max"]) - target_min, 1e-12)
        return values * target_range + target_min

    def _compute_loss(self, predictions: Any, targets: Any) -> tuple[Any, dict[str, float]]:
        if self.weighted_loss_thresholds is None:
            raise RuntimeError("Weighted loss thresholds have not been initialized.")

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
                predictions = self.network(batch_X)
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
            raise RuntimeError("LSTM has not been fit yet.")
        self.network.eval()
        X_test = torch.tensor(data["X_test"], dtype=torch.float32)
        loader = DataLoader(
            TensorDataset(X_test),
            batch_size=int(self.model_config["batch_size"]),
            shuffle=False,
        )
        predictions: list[np.ndarray] = []
        with torch.no_grad():
            for (batch_X,) in loader:
                batch_pred = self.network(batch_X.to(self.device)).detach().cpu().numpy()
                predictions.append(batch_pred)
        return np.vstack(predictions).astype(np.float32)

    def save(self, path: str | Path) -> None:
        import torch

        if self.network is None:
            raise RuntimeError("LSTM has not been fit yet, so it cannot be saved.")
        torch.save(
            {
                "model_name": self.name,
                "model_state_dict": self.network.state_dict(),
                "model_config": self.model_config,
                "feature_count": self.config["global_constraints"]["feature_count"],
                "output_window": self.config["global_constraints"]["output_window_hours"],
                "training_history": self.training_history,
                "weighted_loss_thresholds": self.weighted_loss_thresholds,
                "high_value_thresholds": self.high_value_thresholds,
            },
            Path(path),
        )
