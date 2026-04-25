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


def peak_mae_loss(y_pred: Any, y_true: Any, q80: float):
    import torch

    mask = y_true >= q80
    abs_error = torch.abs(y_pred - y_true)
    if torch.any(mask):
        loss = abs_error[mask].mean()
    else:
        loss = abs_error.mean()
    return loss, mask.to(dtype=y_true.dtype).mean()


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


def _is_better_epoch(
    val_q80_mae: float,
    val_h1_rmse: float,
    val_rmse: float,
    best_val_q80_mae: float,
    best_val_h1_rmse: float,
    best_val_rmse: float,
    *,
    q80_atol: float = 1.25,
    h1_atol: float = 0.25,
    rmse_atol: float = 0.25,
) -> bool:
    current = (float(val_q80_mae), float(val_h1_rmse), float(val_rmse))
    best = (float(best_val_q80_mae), float(best_val_h1_rmse), float(best_val_rmse))
    if not all(np.isfinite(value) for value in current):
        return False
    if not all(np.isfinite(value) for value in best):
        return True

    if current[0] < best[0] - q80_atol:
        return True
    if abs(current[0] - best[0]) <= q80_atol:
        if current[1] < best[1] - h1_atol:
            return True
        if abs(current[1] - best[1]) <= h1_atol:
            return current[2] < best[2] - rmse_atol
    return False


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


class AttentionLSTMForecastModel(BaseForecastModel):
    name = "attention_lstm"

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config, self.name)
        self.network: Any = None
        self.device: Any = None
        self.attention_weights: np.ndarray | None = None
        self.attention_diagnostics: dict[str, Any] | None = None
        self.training_history: list[dict[str, Any]] = []
        self.weighted_loss_thresholds: dict[str, float] | None = None
        self.high_value_threshold: dict[str, float] | None = None
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
        import torch.nn.functional as F
        from torch import nn

        class AttentionContext(nn.Module):
            def __init__(self, hidden_size: int) -> None:
                super().__init__()
                self.key_proj = nn.Linear(hidden_size, hidden_size)
                self.query_proj = nn.Linear(hidden_size, hidden_size, bias=False)
                self.attention_score = nn.Linear(hidden_size, 1, bias=False)

            def forward(self, sequence: Any, query: Any) -> tuple[Any, Any]:
                query_proj = self.query_proj(query).unsqueeze(1)
                scores = self.attention_score(torch.tanh(self.key_proj(sequence) + query_proj)).squeeze(-1)
                weights = scores.softmax(dim=1)
                context = torch.bmm(weights.unsqueeze(1), sequence).squeeze(1)
                return context, weights

        if not self._use_long_window_strategy():
            recent_steps = int(self.model_config.get("recent_attention_steps", 24))
            global_pool_steps = int(self.model_config.get("global_attention_pool_steps", 168))
            recent_gate_cap = float(self.model_config.get("recent_gate_cap", 0.12))
            input_size = int(self.config["global_constraints"]["feature_count"])
            hidden_size = int(self.model_config["hidden_size"])
            num_layers = int(self.model_config["num_layers"])
            dropout = float(self.model_config["dropout"])
            output_size = int(self.config["global_constraints"]["output_window_hours"])
            bounded_output = int(self.config["window"]["input_window_hours"]) > global_pool_steps

            class DefaultAttentionLSTMRegressor(nn.Module):
                def __init__(self) -> None:
                    super().__init__()
                    lstm_dropout = dropout if num_layers > 1 else 0.0
                    self.lstm = nn.LSTM(
                        input_size=input_size,
                        hidden_size=hidden_size,
                        num_layers=num_layers,
                        dropout=lstm_dropout,
                        batch_first=True,
                    )
                    self.recent_steps = max(recent_steps, 1)
                    self.global_pool_steps = max(global_pool_steps, 1)
                    self.recent_gate_cap = float(min(max(recent_gate_cap, 0.0), 1.0))
                    self.recent_attention = AttentionContext(hidden_size)
                    self.global_attention = AttentionContext(hidden_size)
                    self.context_gate = nn.Linear(hidden_size, 1)
                    head_layers: list[nn.Module] = [
                        nn.Linear(hidden_size * 3, hidden_size),
                        nn.ReLU(),
                        nn.Dropout(dropout),
                        nn.Linear(hidden_size, output_size),
                    ]
                    if bounded_output:
                        head_layers.append(nn.Sigmoid())
                    self.head = nn.Sequential(*head_layers)

                @staticmethod
                def _build_recent_profile(recent_weights: Any, original_steps: int) -> Any:
                    if recent_weights.size(1) == original_steps:
                        return recent_weights
                    expanded = recent_weights.new_zeros((recent_weights.size(0), original_steps))
                    expanded[:, -recent_weights.size(1) :] = recent_weights
                    return expanded

                @staticmethod
                def _build_global_profile(global_weights: Any, original_steps: int) -> Any:
                    pooled_steps = global_weights.size(1)
                    if pooled_steps == original_steps:
                        return global_weights

                    indices = torch.floor(
                        torch.arange(original_steps, device=global_weights.device, dtype=global_weights.dtype)
                        * pooled_steps
                        / max(original_steps, 1)
                    ).long()
                    indices = torch.clamp(indices, max=pooled_steps - 1)
                    counts = torch.bincount(indices, minlength=pooled_steps).to(
                        device=global_weights.device,
                        dtype=global_weights.dtype,
                    )
                    expanded = global_weights[:, indices] / counts[indices].unsqueeze(0).clamp_min(1.0)
                    return expanded / expanded.sum(dim=1, keepdim=True).clamp_min(1e-12)

                def forward(self, x: Any) -> tuple[Any, dict[str, Any]]:
                    outputs, (hidden, _) = self.lstm(x)
                    last_hidden = hidden[-1]
                    original_steps = outputs.size(1)

                    recent_outputs = outputs[:, -min(self.recent_steps, original_steps) :, :]
                    recent_context, recent_weights = self.recent_attention(recent_outputs, last_hidden)

                    if original_steps > self.global_pool_steps:
                        pooled_outputs = F.adaptive_avg_pool1d(
                            outputs.transpose(1, 2),
                            self.global_pool_steps,
                        ).transpose(1, 2)
                    else:
                        pooled_outputs = outputs
                    global_context, global_weights = self.global_attention(pooled_outputs, last_hidden)

                    raw_gate = torch.sigmoid(self.context_gate(last_hidden))
                    recent_mix = raw_gate * self.recent_gate_cap
                    global_mix = 1.0 - recent_mix
                    gated_recent_context = recent_mix * recent_context
                    gated_global_context = global_mix * global_context
                    delta_logits = self.head(torch.cat([last_hidden, gated_recent_context, gated_global_context], dim=1))

                    global_profile = self._build_global_profile(global_weights, original_steps)
                    recent_profile = self._build_recent_profile(recent_weights, original_steps)
                    mix_profile = recent_mix.expand(-1, original_steps)
                    combined_profile = mix_profile * recent_profile + global_mix.expand(-1, original_steps) * global_profile
                    combined_profile = combined_profile / combined_profile.sum(dim=1, keepdim=True).clamp_min(1e-12)

                    return delta_logits, {
                        "combined_weights": combined_profile,
                        "global_profile": global_profile,
                        "recent_profile": recent_profile,
                        "gate": recent_mix.squeeze(-1),
                        "raw_gate": raw_gate.squeeze(-1),
                    }

            return DefaultAttentionLSTMRegressor()

        branch_specs = self._branch_specs()
        input_size = int(self.config["global_constraints"]["feature_count"])
        num_layers = int(self.model_config.get("num_layers", 1))
        dropout = float(self.model_config["dropout"])
        output_size = int(self.config["global_constraints"]["output_window_hours"])
        target_index = self.pm25_index
        fusion_hidden = int(sum(spec["hidden_size"] for spec in branch_specs))
        branch_rep_dim = int(sum(spec["hidden_size"] * 2 for spec in branch_specs))

        class LongWindowAttentionLSTMRegressor(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.branch_specs = branch_specs
                self.target_index = target_index
                self.encoders = nn.ModuleDict()
                self.attentions = nn.ModuleDict()
                for spec in self.branch_specs:
                    lstm_dropout = dropout if num_layers > 1 else 0.0
                    self.encoders[spec["name"]] = nn.LSTM(
                        input_size=input_size,
                        hidden_size=int(spec["hidden_size"]),
                        num_layers=num_layers,
                        dropout=lstm_dropout,
                        batch_first=True,
                    )
                    self.attentions[spec["name"]] = AttentionContext(int(spec["hidden_size"]))
                self.branch_gate = nn.Linear(branch_rep_dim, len(self.branch_specs))
                self.head = nn.Sequential(
                    nn.Linear(branch_rep_dim, fusion_hidden),
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

            @staticmethod
            def _expand_profile(weights: Any, branch_hours: int, pool_hours: int) -> Any:
                if pool_hours <= 1:
                    return weights
                expanded = torch.repeat_interleave(weights / float(pool_hours), repeats=pool_hours, dim=1)
                if expanded.size(1) != branch_hours:
                    raise ValueError(f"Expanded attention profile mismatch: {expanded.size(1)} != {branch_hours}")
                return expanded

            def forward(self, x: Any) -> tuple[Any, dict[str, Any]]:
                total_steps = x.size(1)
                branch_reps: list[Any] = []
                branch_profiles: dict[str, Any] = {}
                branch_weights: dict[str, Any] = {}

                for spec in self.branch_specs:
                    segment = x[:, spec["start"] : spec["end"], :]
                    pooled = self._pool_branch(segment, int(spec["pool_hours"]))
                    outputs, (hidden, _) = self.encoders[spec["name"]](pooled)
                    last_hidden = hidden[-1]
                    context, attn_weights = self.attentions[spec["name"]](outputs, last_hidden)
                    branch_reps.append(torch.cat([context, last_hidden], dim=1))

                    expanded_profile = self._expand_profile(attn_weights, spec["end"] - spec["start"], int(spec["pool_hours"]))
                    full_profile = x.new_zeros((x.size(0), total_steps))
                    full_profile[:, spec["start"] : spec["end"]] = expanded_profile
                    branch_profiles[spec["name"]] = full_profile
                    branch_weights[spec["name"]] = attn_weights

                gate_logits = self.branch_gate(torch.cat(branch_reps, dim=1))
                gate_values = gate_logits.softmax(dim=1)
                gated_reps = [gate_values[:, idx : idx + 1] * rep for idx, rep in enumerate(branch_reps)]
                delta_logits = self.head(torch.cat(gated_reps, dim=1))

                anchor = x[:, -1, self.target_index].unsqueeze(1).expand(-1, output_size)
                anchor = torch.clamp(anchor, min=1e-4, max=1 - 1e-4)
                anchor_logit = torch.log(anchor / (1 - anchor))
                predictions = torch.sigmoid(anchor_logit + delta_logits)

                combined_profile = x.new_zeros((x.size(0), total_steps))
                gate_dict: dict[str, Any] = {}
                for idx, spec in enumerate(self.branch_specs):
                    gate_slice = gate_values[:, idx : idx + 1]
                    combined_profile = combined_profile + gate_slice * branch_profiles[spec["name"]]
                    gate_dict[spec["name"]] = gate_values[:, idx]
                combined_profile = combined_profile / combined_profile.sum(dim=1, keepdim=True).clamp_min(1e-12)

                return predictions, {
                    "combined_weights": combined_profile,
                    "branch_profiles": branch_profiles,
                    "branch_gates": gate_dict,
                    "branch_weights": branch_weights,
                }

        return LongWindowAttentionLSTMRegressor()

    def fit(self, data: dict[str, Any]) -> None:
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        if not torch.cuda.is_available():
            raise RuntimeError("Attention-LSTM training requires a CUDA GPU, but CUDA is unavailable.")

        self.device = torch.device("cuda")
        self.network = self._build_network().to(self.device)
        self.training_history = []
        self.attention_diagnostics = None
        self.weighted_loss_thresholds = self._prepare_weighted_loss_thresholds(data)
        self.high_value_threshold = self.weighted_loss_thresholds
        runtime_write(
            self.config,
            (
                f"q75={self.weighted_loss_thresholds['q75']:.6f}, "
                f"q80={self.weighted_loss_thresholds['q80']:.6f}, "
                f"q90={self.weighted_loss_thresholds['q90']:.6f}"
            ),
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
        scheduler = None
        if str(self.model_config.get("scheduler", "")).lower() == "reducelronplateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=float(self.model_config.get("scheduler_factor", 0.5)),
                patience=int(self.model_config.get("scheduler_patience", 3)),
                min_lr=float(self.model_config.get("scheduler_min_lr", 1e-5)),
            )
        current_lr = float(optimizer.param_groups[0]["lr"])
        checkpoint_metric = str(self.model_config.get("checkpoint_metric", "q80_then_h1_then_rmse"))
        best_state = None
        best_val_loss = float("inf")
        best_val_rmse = float("inf")
        best_val_q80_mae = float("inf")
        best_val_q90_mae = float("inf")
        best_val_h1_rmse = float("inf")
        best_history_index: int | None = None
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
                    predictions, _ = self.network(batch_X)
                    loss, weight_stats = self._compute_default_loss(predictions, batch_y)
                    loss.backward()
                    optimizer.step()

                    train_losses.append(float(loss.item()))
                    train_mean_weights.append(weight_stats["mean_weight"])
                    train_peak_ratios.append(weight_stats["peak_ratio"])

                validation_stats = self._evaluate_default_epoch(val_loader)
                epoch = epoch_idx + 1
                best_val_loss = min(best_val_loss, float(validation_stats["validation_loss"]))
                is_best_epoch = _is_better_epoch(
                    float(validation_stats["val_q80_mae"]),
                    float(validation_stats["val_h1_rmse"]),
                    float(validation_stats["val_rmse"]),
                    best_val_q80_mae,
                    best_val_h1_rmse,
                    best_val_rmse,
                )

                if is_best_epoch:
                    best_val_rmse = float(validation_stats["val_rmse"])
                    best_val_q80_mae = float(validation_stats["val_q80_mae"])
                    best_val_q90_mae = float(validation_stats["val_q90_mae"])
                    best_val_h1_rmse = float(validation_stats["val_h1_rmse"])
                    best_state = {k: v.detach().cpu().clone() for k, v in self.network.state_dict().items()}
                    bad_epochs = 0
                    if best_history_index is not None:
                        self.training_history[best_history_index]["is_best_epoch"] = False
                    best_history_index = len(self.training_history)
                else:
                    bad_epochs += 1

                if scheduler is not None and np.isfinite(float(validation_stats["val_q80_mae"])):
                    scheduler.step(float(validation_stats["val_q80_mae"]))
                current_lr = float(optimizer.param_groups[0]["lr"])

                history_row = {
                    "epoch": epoch,
                    "train_loss": _safe_mean(train_losses),
                    "validation_loss": float(validation_stats["validation_loss"]),
                    "best_validation_loss": float(best_val_loss),
                    "bad_epochs": bad_epochs,
                    "q75": float(self.weighted_loss_thresholds["q75"]),
                    "q80": float(self.weighted_loss_thresholds["q80"]),
                    "q90": float(self.weighted_loss_thresholds["q90"]),
                    "train_mean_weight": _safe_mean(train_mean_weights),
                    "validation_mean_weight": float(validation_stats["validation_mean_weight"]),
                    "train_peak_ratio": _safe_mean(train_peak_ratios),
                    "validation_peak_ratio": float(validation_stats["validation_peak_ratio"]),
                    "val_rmse": float(validation_stats["val_rmse"]),
                    "best_val_rmse": float(best_val_rmse),
                    "val_q80_mae": float(validation_stats["val_q80_mae"]),
                    "best_val_q80_mae": float(best_val_q80_mae),
                    "val_q90_mae": float(validation_stats["val_q90_mae"]),
                    "best_val_q90_mae": float(best_val_q90_mae),
                    "val_h1_rmse": float(validation_stats["val_h1_rmse"]),
                    "best_val_h1_rmse": float(best_val_h1_rmse),
                    "checkpoint_metric": checkpoint_metric,
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
                    predictions, _ = self.network(batch_X)
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
            "q80": float(np.nanquantile(train_target, 0.80)),
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
        import torch.nn.functional as F

        predictions_raw = self._inverse_transform_target_tensor(predictions)
        targets_raw = self._inverse_transform_target_tensor(targets)
        weighted_mse, weights = weighted_mse_loss(
            predictions_raw,
            targets_raw,
            float(self.weighted_loss_thresholds["q75"]),
            float(self.weighted_loss_thresholds["q90"]),
        )
        huber = F.huber_loss(
            predictions_raw,
            targets_raw,
            delta=float(self.model_config.get("huber_delta", 10.0)),
            reduction="mean",
        )
        peak_mae, peak_ratio_q80 = peak_mae_loss(
            predictions_raw,
            targets_raw,
            float(self.weighted_loss_thresholds["q80"]),
        )
        loss = 0.5 * huber + 0.3 * weighted_mse + 0.2 * peak_mae
        peak_ratio = (weights == 8.0).to(dtype=predictions_raw.dtype).mean()
        return loss, {
            "mean_weight": float(weights.mean().item()),
            "peak_ratio": float(peak_ratio.item()),
            "q80_ratio": float(peak_ratio_q80.item()),
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

    def _evaluate_default_epoch(self, loader: Any) -> dict[str, float]:
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
                predictions, _ = self.network(batch_X)
                loss, weight_stats = self._compute_default_loss(predictions, batch_y)
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

        q80_mask = y_true >= float(self.weighted_loss_thresholds["q80"])
        q90_mask = y_true >= float(self.weighted_loss_thresholds["q90"])
        q80_mae = _mae(y_true[q80_mask], y_pred[q80_mask]) if np.any(q80_mask) else _mae(y_true, y_pred)
        q90_mae = _mae(y_true[q90_mask], y_pred[q90_mask]) if np.any(q90_mask) else _mae(y_true, y_pred)
        h1_rmse = _rmse(y_true[:, :1], y_pred[:, :1]) if y_true.size else float("nan")
        return {
            "validation_loss": _safe_mean(losses, default=float("inf")),
            "validation_mean_weight": _safe_mean(mean_weights),
            "validation_peak_ratio": _safe_mean(peak_ratios),
            "val_rmse": _rmse(y_true, y_pred),
            "val_q80_mae": q80_mae,
            "val_q90_mae": q90_mae,
            "val_h1_rmse": h1_rmse,
        }

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
                predictions, _ = self.network(batch_X)
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
            raise RuntimeError("Attention-LSTM has not been fit yet.")
        self.network.eval()
        X_test = torch.tensor(data["X_test"], dtype=torch.float32)
        loader = DataLoader(
            TensorDataset(X_test),
            batch_size=int(self.model_config["batch_size"]),
            shuffle=False,
        )

        predictions: list[np.ndarray] = []
        combined_weights: list[np.ndarray] = []
        diagnostics: dict[str, Any] = {}

        with torch.no_grad():
            for (batch_X,) in loader:
                batch_pred, attention_info = self.network(batch_X.to(self.device))
                predictions.append(batch_pred.detach().cpu().numpy())
                combined_weights.append(attention_info["combined_weights"].detach().cpu().numpy())

                if self._use_long_window_strategy():
                    diagnostics.setdefault("branch_profiles", {})
                    diagnostics.setdefault("branch_gates", {})
                    for name, profile in attention_info["branch_profiles"].items():
                        diagnostics["branch_profiles"].setdefault(name, []).append(profile.detach().cpu().numpy())
                    for name, gate in attention_info["branch_gates"].items():
                        diagnostics["branch_gates"].setdefault(name, []).append(gate.detach().cpu().numpy())
                else:
                    diagnostics.setdefault("global_profile", []).append(attention_info["global_profile"].detach().cpu().numpy())
                    diagnostics.setdefault("recent_profile", []).append(attention_info["recent_profile"].detach().cpu().numpy())
                    diagnostics.setdefault("gate", []).append(attention_info["gate"].detach().cpu().numpy())
                    diagnostics.setdefault("raw_gate", []).append(attention_info["raw_gate"].detach().cpu().numpy())

        self.attention_weights = np.vstack(combined_weights).astype(np.float32)
        if self._use_long_window_strategy():
            self.attention_diagnostics = {
                "combined_profile": self.attention_weights,
                "branch_profiles": {
                    name: np.vstack(chunks).astype(np.float32)
                    for name, chunks in diagnostics.get("branch_profiles", {}).items()
                },
                "branch_gates": {
                    name: np.concatenate(chunks).astype(np.float32)
                    for name, chunks in diagnostics.get("branch_gates", {}).items()
                },
            }
        else:
            self.attention_diagnostics = {
                "combined_profile": self.attention_weights,
                "global_profile": np.vstack(diagnostics.get("global_profile", [])).astype(np.float32),
                "recent_profile": np.vstack(diagnostics.get("recent_profile", [])).astype(np.float32),
                "gate": np.concatenate(diagnostics.get("gate", [])).astype(np.float32),
                "raw_gate": np.concatenate(diagnostics.get("raw_gate", [])).astype(np.float32),
            }
        return np.vstack(predictions).astype(np.float32)

    def save_attention_weights(self, path: str | Path) -> Path:
        if self.attention_weights is None:
            raise RuntimeError("Attention weights have not been generated yet.")
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(output_path, self.attention_weights)
        return output_path

    def save(self, path: str | Path) -> None:
        import torch

        if self.network is None:
            raise RuntimeError("Attention-LSTM has not been fit yet, so it cannot be saved.")
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
