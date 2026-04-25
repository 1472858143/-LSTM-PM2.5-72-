from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from models.base import BaseForecastModel
from utils.runtime import runtime_add_task, runtime_label, runtime_remove_task, runtime_update_task, runtime_write


def weighted_mse_loss(y_pred: Any, y_true: Any, q75: float, q90: float):
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


def _safe_mean(values: list[float], default: float = float("nan")) -> float:
    return float(np.mean(values)) if values else default


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean(np.square(y_true - y_pred))))


def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.size == 0:
        return float("nan")
    return float(np.mean(np.abs(y_true - y_pred)))


def _compute_selection_score(
    val_rmse: float,
    val_stage1_rmse: float,
    val_q90_mae: float,
    weights: dict[str, float],
) -> float:
    if not all(np.isfinite(v) for v in [val_rmse, val_stage1_rmse, val_q90_mae]):
        return float("inf")
    return (
        float(weights["rmse"]) * float(val_rmse)
        + float(weights["stage1_rmse"]) * float(val_stage1_rmse)
        + float(weights["q90_mae"]) * float(val_q90_mae)
    )


class LSTMForecastModel(BaseForecastModel):
    name = "lstm"

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config, self.name)
        self.network: Any = None
        self.device: Any = None
        self.training_history: list[dict[str, Any]] = []
        self.weighted_loss_thresholds: dict[str, float] | None = None
        self.high_value_thresholds: dict[str, float] | None = None
        self.pm25_index = int(self.config["data"]["model_input_features"].index(self.config["data"]["target"]))
        self.selection_score_weights = {
            "rmse": 0.50,
            "stage1_rmse": 0.30,
            "q90_mae": 0.20,
            **self.model_config.get("selection_score_weights", {}),
        }

    def _use_long_window_strategy(self) -> bool:
        return bool(self.model_config.get("branch_layout"))

    def _branch_specs(self) -> list[dict[str, Any]]:
        input_window = int(self.config["window"]["input_window_hours"])
        layout = list(self.model_config.get("branch_layout", []))
        if not layout:
            return []

        spans: dict[str, int] = {}
        if "recent" in layout:
            spans["recent"] = 168
        if "mid" in layout:
            spans["mid"] = 552
        if "context" in layout:
            spans["context"] = input_window - 168
        if "far" in layout:
            spans["far"] = input_window - spans.get("mid", 0) - spans.get("recent", 0)

        hidden_sizes = self.model_config["branch_hidden_sizes"]
        pool_sizes = self.model_config["pool_sizes_hours"]
        specs: list[dict[str, Any]] = []
        cursor = 0
        for name in layout:
            hours = int(spans[name])
            if hours <= 0:
                raise ValueError(f"Invalid branch hours for {name}: {hours}")
            specs.append(
                {
                    "name": name,
                    "start": cursor,
                    "end": cursor + hours,
                    "pool_hours": int(pool_sizes.get(name, 1)),
                    "hidden_size": int(hidden_sizes[name]),
                }
            )
            cursor += hours
        if cursor != input_window:
            raise ValueError(f"Branch layout does not cover the active input window: {cursor} != {input_window}")
        return specs

    def _build_network(self) -> Any:
        import torch
        from torch import nn

        if not self._use_long_window_strategy():

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

                def forward(self, x: Any) -> Any:
                    outputs, (hidden, _) = self.lstm(x)
                    last_hidden = hidden[-1]
                    max_pool = outputs.max(dim=1).values
                    return self.head(torch.cat([last_hidden, max_pool], dim=1))

            return LSTMRegressor(
                input_size=int(self.config["global_constraints"]["feature_count"]),
                hidden_size=int(self.model_config["hidden_size"]),
                num_layers=int(self.model_config["num_layers"]),
                dropout=float(self.model_config["dropout"]),
                output_size=int(self.config["global_constraints"]["output_window_hours"]),
            )

        branch_specs = self._branch_specs()
        fusion_hidden = int(sum(spec["hidden_size"] for spec in branch_specs))
        input_size = int(self.config["global_constraints"]["feature_count"])
        num_layers = int(self.model_config.get("num_layers", 1))
        dropout = float(self.model_config["dropout"])
        output_size = int(self.config["global_constraints"]["output_window_hours"])
        target_index = self.pm25_index

        class LongWindowLSTMRegressor(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.branch_specs = branch_specs
                self.target_index = target_index
                self.encoders = nn.ModuleDict()
                for spec in self.branch_specs:
                    lstm_dropout = dropout if num_layers > 1 else 0.0
                    self.encoders[spec["name"]] = nn.LSTM(
                        input_size=input_size,
                        hidden_size=int(spec["hidden_size"]),
                        num_layers=num_layers,
                        dropout=lstm_dropout,
                        batch_first=True,
                    )
                total_summary_dim = int(sum(spec["hidden_size"] * 2 for spec in self.branch_specs))
                self.head = nn.Sequential(
                    nn.Linear(total_summary_dim, fusion_hidden),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(fusion_hidden, output_size),
                )

            @staticmethod
            def _pool_branch(x: Any, pool_hours: int) -> Any:
                if pool_hours <= 1:
                    return x
                if x.size(1) % pool_hours != 0:
                    raise ValueError(f"Branch length {x.size(1)} is not divisible by pool_hours={pool_hours}")
                batch, steps, features = x.shape
                return x.reshape(batch, steps // pool_hours, pool_hours, features).mean(dim=2)

            def forward(self, x: Any) -> Any:
                summaries: list[Any] = []
                for spec in self.branch_specs:
                    segment = x[:, spec["start"] : spec["end"], :]
                    pooled = self._pool_branch(segment, int(spec["pool_hours"]))
                    outputs, (hidden, _) = self.encoders[spec["name"]](pooled)
                    last_hidden = hidden[-1]
                    mean_pool = outputs.mean(dim=1)
                    summaries.append(torch.cat([last_hidden, mean_pool], dim=1))

                delta_logits = self.head(torch.cat(summaries, dim=1))
                anchor = x[:, -1, self.target_index].unsqueeze(1).expand(-1, output_size)
                anchor = torch.clamp(anchor, min=1e-4, max=1 - 1e-4)
                anchor_logit = torch.log(anchor / (1 - anchor))
                return torch.sigmoid(anchor_logit + delta_logits)

        return LongWindowLSTMRegressor()

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

        threshold_keys = ["q75", "q90"] if not self._use_long_window_strategy() else ["q75", "q90"]
        runtime_write(
            self.config,
            ", ".join(f"{key}={self.weighted_loss_thresholds[key]:.6f}" for key in threshold_keys),
        )

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

        if self._use_long_window_strategy():
            self._fit_long_window(train_loader, val_loader)
        else:
            self._fit_default(train_loader, val_loader)

    def _fit_default(self, train_loader: Any, val_loader: Any) -> None:
        import torch

        optimizer = torch.optim.Adam(self.network.parameters(), lr=float(self.model_config["learning_rate"]))
        current_lr = float(optimizer.param_groups[0]["lr"])
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
                    loss, weight_stats = self._compute_default_loss(predictions, batch_y)
                    loss.backward()
                    optimizer.step()

                    train_losses.append(float(loss.item()))
                    train_mean_weights.append(weight_stats["mean_weight"])
                    train_peak_ratios.append(weight_stats["peak_ratio"])

                val_loss, val_mean_weight, val_peak_ratio = self._evaluate_default_loss(val_loader)
                epoch = epoch_idx + 1
                history_row = {
                    "epoch": epoch,
                    "train_loss": _safe_mean(train_losses),
                    "validation_loss": float(val_loss),
                    "best_validation_loss": float(min(best_val_loss, val_loss)),
                    "bad_epochs": bad_epochs,
                    "q75": float(self.weighted_loss_thresholds["q75"]),
                    "q90": float(self.weighted_loss_thresholds["q90"]),
                    "train_mean_weight": _safe_mean(train_mean_weights),
                    "validation_mean_weight": float(val_mean_weight),
                    "train_peak_ratio": _safe_mean(train_peak_ratios),
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

    def _fit_long_window(self, train_loader: Any, val_loader: Any) -> None:
        import torch

        optimizer = torch.optim.AdamW(
            self.network.parameters(),
            lr=float(self.model_config["learning_rate"]),
            weight_decay=float(self.model_config.get("weight_decay", 0.0)),
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=float(self.model_config.get("scheduler_factor", 0.5)),
            patience=int(self.model_config.get("scheduler_patience", 3)),
            min_lr=float(self.model_config.get("scheduler_min_lr", 1e-5)),
        )
        current_lr = float(optimizer.param_groups[0]["lr"])
        best_state = None
        best_val_loss = float("inf")
        best_selection_score = float("inf")
        best_history_index: int | None = None
        total_epochs = int(self.model_config["epochs"])
        patience = int(self.model_config["early_stopping_patience"])
        bad_epochs = 0
        grad_clip = float(self.model_config.get("grad_clip", 1.0))
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
                    loss, weight_stats = self._compute_long_window_loss(predictions, batch_y)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.network.parameters(), grad_clip)
                    optimizer.step()

                    train_losses.append(float(loss.item()))
                    train_mean_weights.append(weight_stats["mean_weight"])
                    train_peak_ratios.append(weight_stats["peak_ratio"])

                validation_stats = self._evaluate_long_window_epoch(val_loader)
                epoch = epoch_idx + 1
                best_val_loss = min(best_val_loss, float(validation_stats["validation_loss"]))
                is_best_epoch = float(validation_stats["selection_score"]) < best_selection_score

                if is_best_epoch:
                    best_selection_score = float(validation_stats["selection_score"])
                    best_state = {k: v.detach().cpu().clone() for k, v in self.network.state_dict().items()}
                    bad_epochs = 0
                    if best_history_index is not None:
                        self.training_history[best_history_index]["is_best_epoch"] = False
                    best_history_index = len(self.training_history)
                else:
                    bad_epochs += 1

                scheduler.step(float(validation_stats["selection_score"]))
                current_lr = float(optimizer.param_groups[0]["lr"])

                history_row = {
                    "epoch": epoch,
                    "train_loss": _safe_mean(train_losses),
                    "validation_loss": float(validation_stats["validation_loss"]),
                    "best_validation_loss": float(best_val_loss),
                    "bad_epochs": bad_epochs,
                    "q75": float(self.weighted_loss_thresholds["q75"]),
                    "q90": float(self.weighted_loss_thresholds["q90"]),
                    "train_mean_weight": _safe_mean(train_mean_weights),
                    "validation_mean_weight": float(validation_stats["validation_mean_weight"]),
                    "train_peak_ratio": _safe_mean(train_peak_ratios),
                    "validation_peak_ratio": float(validation_stats["validation_peak_ratio"]),
                    "val_rmse": float(validation_stats["val_rmse"]),
                    "val_stage1_rmse": float(validation_stats["val_stage1_rmse"]),
                    "val_q90_mae": float(validation_stats["val_q90_mae"]),
                    "selection_score": float(validation_stats["selection_score"]),
                    "best_selection_score": float(best_selection_score),
                    "checkpoint_metric": "selection_score",
                    "learning_rate": current_lr,
                    "is_best_epoch": bool(is_best_epoch),
                }
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

    def _compute_default_loss(self, predictions: Any, targets: Any) -> tuple[Any, dict[str, float]]:
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

    def _compute_long_window_loss(self, predictions: Any, targets: Any) -> tuple[Any, dict[str, float]]:
        import torch
        import torch.nn.functional as F

        predictions_raw = self._inverse_transform_target_tensor(predictions)
        targets_raw = self._inverse_transform_target_tensor(targets)
        beta = float(self.model_config.get("loss_beta", 12.0))

        level_weights = torch.ones_like(targets_raw)
        q75 = float(self.weighted_loss_thresholds["q75"])
        q90 = float(self.weighted_loss_thresholds["q90"])
        level_weights = torch.where(targets_raw >= q75, torch.full_like(level_weights, 1.8), level_weights)
        level_weights = torch.where(targets_raw >= q90, torch.full_like(level_weights, 3.0), level_weights)

        horizon_weights = torch.ones((1, targets_raw.size(1)), device=targets_raw.device, dtype=targets_raw.dtype)
        horizon_weights[:, :24] = 1.2
        horizon_weights[:, 24:48] = 1.0
        horizon_weights[:, 48:] = 0.9
        total_weights = level_weights * horizon_weights

        loss_matrix = F.smooth_l1_loss(predictions_raw, targets_raw, beta=beta, reduction="none")
        weighted_loss = (loss_matrix * total_weights).sum() / total_weights.sum().clamp_min(1e-12)
        peak_ratio = (targets_raw >= q90).to(dtype=targets_raw.dtype).mean()
        return weighted_loss, {
            "mean_weight": float(total_weights.mean().item()),
            "peak_ratio": float(peak_ratio.item()),
        }

    def _evaluate_default_loss(self, loader: Any) -> tuple[float, float, float]:
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
                loss, weight_stats = self._compute_default_loss(predictions, batch_y)
                losses.append(float(loss.item()))
                mean_weights.append(weight_stats["mean_weight"])
                peak_ratios.append(weight_stats["peak_ratio"])
        return (
            _safe_mean(losses, default=float("inf")),
            _safe_mean(mean_weights),
            _safe_mean(peak_ratios),
        )

    def _evaluate_long_window_epoch(self, loader: Any) -> dict[str, float]:
        import torch

        self.network.eval()
        losses: list[float] = []
        mean_weights: list[float] = []
        peak_ratios: list[float] = []
        y_true_batches: list[np.ndarray] = []
        y_pred_batches: list[np.ndarray] = []

        with torch.no_grad():
            for batch_X, batch_y in loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                predictions = self.network(batch_X)
                loss, weight_stats = self._compute_long_window_loss(predictions, batch_y)
                losses.append(float(loss.item()))
                mean_weights.append(weight_stats["mean_weight"])
                peak_ratios.append(weight_stats["peak_ratio"])
                y_true_batches.append(self._inverse_transform_target_tensor(batch_y).detach().cpu().numpy())
                y_pred_batches.append(self._inverse_transform_target_tensor(predictions).detach().cpu().numpy())

        if y_true_batches:
            y_true = np.vstack(y_true_batches).astype(np.float64)
            y_pred = np.vstack(y_pred_batches).astype(np.float64)
        else:
            horizon = int(self.config["global_constraints"]["output_window_hours"])
            y_true = np.empty((0, horizon), dtype=np.float64)
            y_pred = np.empty((0, horizon), dtype=np.float64)

        val_rmse = _rmse(y_true, y_pred)
        val_stage1_rmse = _rmse(y_true[:, :24], y_pred[:, :24]) if y_true.size else float("nan")
        q90_mask = y_true >= float(self.weighted_loss_thresholds["q90"])
        val_q90_mae = _mae(y_true[q90_mask], y_pred[q90_mask]) if np.any(q90_mask) else _mae(y_true, y_pred)
        selection_score = _compute_selection_score(
            val_rmse,
            val_stage1_rmse,
            val_q90_mae,
            self.selection_score_weights,
        )

        return {
            "validation_loss": _safe_mean(losses, default=float("inf")),
            "validation_mean_weight": _safe_mean(mean_weights),
            "validation_peak_ratio": _safe_mean(peak_ratios),
            "val_rmse": val_rmse,
            "val_stage1_rmse": val_stage1_rmse,
            "val_q90_mae": val_q90_mae,
            "selection_score": selection_score,
        }

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
