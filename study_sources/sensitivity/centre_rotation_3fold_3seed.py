#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CKD LSTM-v2: complete fixed-input, post-hoc centre-rotation training.

Self-contained. Run all cells of the supplied Notebook, or:
%run "__CKD_WORKDIR__/CKD_crosscenter_TRAIN.py"

This runner trains actual models, estimates training-OOF calibration, and evaluates
held-out centres. No old primary weights/scalers/calibrators are loaded.
The archived internal test cohort is EXCLUDED by default, without changing it.
Inputs were previously imputed: historical imputation provenance cannot be undone
by this program. Results remain conditional on that input processing and on the
already selected architecture/hyperparameters, not fully nested validation.
"""
from __future__ import annotations
import gc
import hashlib
import json
import math
import os
import random
import shutil
import sys
import time
import traceback
import warnings
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from scipy.optimize import minimize
from scipy.special import expit

# ===================== 本次完整训练的配置 =====================
ROOT = Path(r"__CKD_WORKDIR__")
OUTPUT_NAME = "crosscenter_LSTM_3fold_3seed_TRAIN_v1"
# 不重划原7:3：默认仅用深圳/南宁原development + 重庆。
PRESERVE_LOCKED_TEST = True
SEEDS = [0, 1, 2]
N_INNER_FOLDS = 3
PARTITION_SEED = 20260727
EARLYSTOP_FRACTION = 0.10
MAX_EPOCHS = 60
MIN_EPOCHS = 5
PATIENCE = 10
MIN_DELTA = 1e-5
TOP_SNAPSHOTS = 3
CPU_THREADS = 4
DEVICE = "cpu"
# 默认运行全部训练、评价及bootstrap。bootstrap不重新训练。
BOOTSTRAP_REPS = 1000
BOOTSTRAP_SEED = 20260909
RESUME = True
# 保留旧分析评价人群：早于首个半年区间删失者不进入有效origin。
EVALUATION_ORIGIN_RULE = "archived_valid_origin"



# ===================== 已恢复的原模型计算核心 =====================
EXPECTED_HISTORY_STEP_N = 11


EXPECTED_STATIC_N = 16


EXPECTED_BASE_DYNAMIC_N = 40


EXPECTED_LAB_N = 15


EXPECTED_EXTRA_DYNAMIC_N = EXPECTED_LAB_N * 3


EXPECTED_ENHANCED_DYNAMIC_N = (
    EXPECTED_BASE_DYNAMIC_N + EXPECTED_EXTRA_DYNAMIC_N
)


EXPECTED_LANDMARK_N = 6


EXPECTED_FUTURE_INTERVAL_N = 10


EPS = 1e-7


NUM_WORKERS = 0


GRADIENT_CLIP_NORM = 1.0


def autocast_context(device: torch.device, enabled: bool):
    try:
        return torch.amp.autocast(
            device_type=device.type,
            enabled=enabled,
        )
    except (AttributeError, TypeError):
        return torch.cuda.amp.autocast(enabled=enabled)


def make_grad_scaler(enabled: bool):
    try:
        return torch.amp.GradScaler(
            "cuda",
            enabled=enabled,
        )
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


class HybridAttentionLSTMSurvival(nn.Module):
    def __init__(
        self,
        dynamic_n: int,
        static_n: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        projection_size: int,
        bidirectional: bool,
        pooling_mode: str,
        static_hidden: int,
        summary_hidden: int,
        horizon_embed_dim: int,
        future_n: int,
    ):
        super().__init__()

        self.dynamic_n = int(dynamic_n)
        self.static_n = int(static_n)
        self.hidden_size = int(hidden_size)
        self.num_layers = int(num_layers)
        self.bidirectional = bool(bidirectional)
        self.pooling_mode = str(pooling_mode)
        self.future_n = int(future_n)
        self.direction_n = 2 if self.bidirectional else 1
        self.representation_n = self.hidden_size * self.direction_n

        if self.pooling_mode not in {
            "last_attention",
            "last_attention_mean",
        }:
            raise ValueError("pooling_mode无效。")

        self.input_encoder = nn.Sequential(
            nn.LayerNorm(self.dynamic_n + 2),
            nn.Linear(self.dynamic_n + 2, int(projection_size)),
            nn.SiLU(),
            nn.Dropout(float(dropout)),
        )

        self.forward_cells = nn.ModuleList(
            [
                nn.LSTMCell(
                    input_size=(
                        int(projection_size)
                        if layer_index == 0
                        else self.hidden_size
                    ),
                    hidden_size=self.hidden_size,
                )
                for layer_index in range(self.num_layers)
            ]
        )
        if self.bidirectional:
            self.backward_cells = nn.ModuleList(
                [
                    nn.LSTMCell(
                        input_size=(
                            int(projection_size)
                            if layer_index == 0
                            else self.hidden_size
                        ),
                        hidden_size=self.hidden_size,
                    )
                    for layer_index in range(self.num_layers)
                ]
            )
        else:
            self.backward_cells = None

        self.recurrent_dropout = nn.Dropout(float(dropout))

        self.attention_hidden = nn.Linear(
            self.representation_n,
            self.representation_n,
            bias=False,
        )
        self.attention_query = nn.Linear(
            self.representation_n,
            self.representation_n,
            bias=False,
        )
        self.attention_score = nn.Linear(
            self.representation_n,
            1,
            bias=False,
        )

        self.static_encoder = nn.Sequential(
            nn.LayerNorm(self.static_n),
            nn.Linear(self.static_n, int(static_hidden)),
            nn.SiLU(),
            nn.Dropout(float(dropout)),
        )

        self.dynamic_summary_encoder = nn.Sequential(
            nn.LayerNorm(self.dynamic_n * 2),
            nn.Linear(self.dynamic_n * 2, int(summary_hidden)),
            nn.SiLU(),
            nn.Dropout(float(dropout)),
        )

        recurrent_context_n = self.representation_n * 2
        if self.pooling_mode == "last_attention_mean":
            recurrent_context_n += self.representation_n

        context_n = (
            recurrent_context_n
            + int(static_hidden)
            + int(summary_hidden)
            + 2
        )
        self.context_encoder = nn.Sequential(
            nn.LayerNorm(context_n),
            nn.Linear(context_n, self.hidden_size),
            nn.SiLU(),
            nn.Dropout(float(dropout)),
        )

        self.horizon_embedding = nn.Embedding(
            self.future_n,
            int(horizon_embed_dim),
        )
        head_input_n = self.hidden_size + int(horizon_embed_dim)
        head_hidden_n = max(32, self.hidden_size // 2)
        self.hazard_head = nn.Sequential(
            nn.LayerNorm(head_input_n),
            nn.Linear(head_input_n, head_hidden_n),
            nn.SiLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(head_hidden_n, 1),
        )
        self.interval_bias = nn.Parameter(
            torch.zeros(self.future_n, dtype=torch.float32)
        )

    def _run_direction(
        self,
        encoded_sequence: torch.Tensor,
        active_mask: torch.Tensor,
        cells: nn.ModuleList,
        reverse: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_n, time_n, _ = encoded_sequence.shape
        hidden = [
            torch.zeros(
                batch_n,
                self.hidden_size,
                dtype=encoded_sequence.dtype,
                device=encoded_sequence.device,
            )
            for _ in range(self.num_layers)
        ]
        cell = [torch.zeros_like(hidden[0]) for _ in range(self.num_layers)]
        history = [None] * time_n

        indices = range(time_n - 1, -1, -1) if reverse else range(time_n)
        for step_index in indices:
            step_active = active_mask[:, step_index].unsqueeze(1)
            layer_input = encoded_sequence[:, step_index, :]

            for layer_index, recurrent_cell in enumerate(cells):
                candidate_hidden, candidate_cell = recurrent_cell(
                    layer_input,
                    (hidden[layer_index], cell[layer_index]),
                )
                hidden[layer_index] = torch.where(
                    step_active,
                    candidate_hidden,
                    hidden[layer_index],
                )
                cell[layer_index] = torch.where(
                    step_active,
                    candidate_cell,
                    cell[layer_index],
                )
                layer_input = hidden[layer_index]
                if layer_index < self.num_layers - 1:
                    layer_input = self.recurrent_dropout(layer_input)

            history[step_index] = hidden[-1]

        return torch.stack(history, dim=1), hidden[-1]

    def forward(
        self,
        dynamic_sequence: torch.Tensor,
        row_mask: torch.Tensor,
        static_baseline: torch.Tensor,
        age_at_landmark: torch.Tensor,
        landmark_normalized: torch.Tensor,
        landmark_bin: torch.Tensor,
    ) -> torch.Tensor:
        batch_n, time_n, dynamic_n = dynamic_sequence.shape
        if dynamic_n != self.dynamic_n:
            raise ValueError("模型收到的动态特征数不正确。")

        step_indices = torch.arange(
            time_n,
            device=dynamic_sequence.device,
        )
        active_mask = (
            row_mask
            & (
                step_indices[None, :]
                <= landmark_bin[:, None]
            )
        )
        if (~active_mask.any(dim=1)).any():
            raise ValueError("部分样本在Landmark前没有有效历史行。")

        time_normalized = (
            step_indices.float()
            / float(max(time_n - 1, 1))
        ).expand(batch_n, time_n)

        previous_active = torch.full(
            (batch_n,),
            -1,
            dtype=torch.long,
            device=dynamic_sequence.device,
        )
        gap_values = []
        for step_index in range(time_n):
            current_active = active_mask[:, step_index]
            gap = torch.where(
                current_active & (previous_active >= 0),
                (
                    step_index - previous_active
                ).float()
                / float(max(time_n - 1, 1)),
                torch.zeros(
                    batch_n,
                    dtype=dynamic_sequence.dtype,
                    device=dynamic_sequence.device,
                ),
            )
            gap_values.append(gap)
            previous_active = torch.where(
                current_active,
                torch.full_like(previous_active, step_index),
                previous_active,
            )
        gap_normalized = torch.stack(gap_values, dim=1)

        encoded_sequence = self.input_encoder(
            torch.cat(
                [
                    dynamic_sequence,
                    time_normalized.unsqueeze(2),
                    gap_normalized.unsqueeze(2),
                ],
                dim=2,
            )
        )

        forward_history, forward_final = self._run_direction(
            encoded_sequence,
            active_mask,
            self.forward_cells,
            reverse=False,
        )

        if self.bidirectional:
            backward_history, backward_final = self._run_direction(
                encoded_sequence,
                active_mask,
                self.backward_cells,
                reverse=True,
            )
            recurrent_history = torch.cat(
                [forward_history, backward_history],
                dim=2,
            )
            final_hidden = torch.cat(
                [forward_final, backward_final],
                dim=1,
            )
        else:
            recurrent_history = forward_history
            final_hidden = forward_final

        attention_logits = self.attention_score(
            torch.tanh(
                self.attention_hidden(recurrent_history)
                + self.attention_query(final_hidden).unsqueeze(1)
            )
        ).squeeze(2)
        attention_logits = attention_logits.masked_fill(
            ~active_mask,
            -1e4,
        )
        attention_weight = torch.softmax(attention_logits, dim=1)
        attention_pool = torch.sum(
            recurrent_history * attention_weight.unsqueeze(2),
            dim=1,
        )

        active_float = active_mask.unsqueeze(2).to(dynamic_sequence.dtype)
        active_count = active_float.sum(dim=1).clamp_min(1.0)
        recurrent_mean = (
            recurrent_history * active_float
        ).sum(dim=1) / active_count

        dynamic_mean = (
            dynamic_sequence * active_float
        ).sum(dim=1) / active_count

        last_position = (
            active_mask.long()
            * (step_indices[None, :] + 1)
        ).argmax(dim=1)
        batch_index = torch.arange(
            batch_n,
            device=dynamic_sequence.device,
        )
        dynamic_last = dynamic_sequence[
            batch_index,
            last_position,
            :,
        ]

        static_encoded = self.static_encoder(static_baseline)
        summary_encoded = self.dynamic_summary_encoder(
            torch.cat([dynamic_last, dynamic_mean], dim=1)
        )

        recurrent_parts = [final_hidden, attention_pool]
        if self.pooling_mode == "last_attention_mean":
            recurrent_parts.append(recurrent_mean)

        context = self.context_encoder(
            torch.cat(
                [
                    *recurrent_parts,
                    static_encoded,
                    summary_encoded,
                    age_at_landmark.unsqueeze(1),
                    landmark_normalized.unsqueeze(1),
                ],
                dim=1,
            )
        )

        horizon_index = torch.arange(
            self.future_n,
            device=dynamic_sequence.device,
        )
        horizon_embedding = self.horizon_embedding(
            horizon_index
        ).unsqueeze(0).expand(batch_n, -1, -1)
        context_expanded = context.unsqueeze(1).expand(
            -1,
            self.future_n,
            -1,
        )
        head_input = torch.cat(
            [context_expanded, horizon_embedding],
            dim=2,
        )
        logits = self.hazard_head(head_input).squeeze(2)
        return logits + self.interval_bias.unsqueeze(0)


class LandmarkBalancedSurvivalLoss(nn.Module):
    def __init__(
        self,
        positive_weight: float,
        focal_gamma: float,
        auxiliary_5y_weight: float,
        ranking_weight: float,
        smoothness_weight: float,
    ):
        super().__init__()
        self.register_buffer(
            "positive_weight",
            torch.tensor(float(positive_weight), dtype=torch.float32),
        )
        self.focal_gamma = float(focal_gamma)
        self.auxiliary_5y_weight = float(auxiliary_5y_weight)
        self.ranking_weight = float(ranking_weight)
        self.smoothness_weight = float(smoothness_weight)

    @staticmethod
    def _landmark_equal_mean(
        sample_loss: torch.Tensor,
        landmark_index: torch.Tensor,
    ) -> torch.Tensor:
        landmark_losses = []
        for landmark_value in range(EXPECTED_LANDMARK_N):
            mask = landmark_index == landmark_value
            if mask.any():
                landmark_losses.append(sample_loss[mask].mean())
        if not landmark_losses:
            raise ValueError("当前批次没有有效Landmark。")
        return torch.stack(landmark_losses).mean()

    def forward(
        self,
        logits: torch.Tensor,
        event_target: torch.Tensor,
        at_risk_mask: torch.Tensor,
        landmark_index: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        logits_float = logits.float()
        target = event_target.float()
        risk_mask = at_risk_mask.float()

        interval_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logits_float,
            target,
            reduction="none",
            pos_weight=self.positive_weight,
        )

        if self.focal_gamma > 0:
            probability = torch.sigmoid(logits_float)
            p_t = (
                target * probability
                + (1.0 - target) * (1.0 - probability)
            )
            interval_loss = interval_loss * (
                1.0 - p_t
            ).pow(self.focal_gamma)

        sample_interval_n = risk_mask.sum(dim=1).clamp_min(1.0)
        sample_survival_loss = (
            interval_loss * risk_mask
        ).sum(dim=1) / sample_interval_n
        survival_loss = self._landmark_equal_mean(
            sample_survival_loss,
            landmark_index,
        )

        hazard = torch.sigmoid(logits_float)
        risk_5y = 1.0 - torch.prod(1.0 - hazard, dim=1)
        target_5y = target.sum(dim=1).clamp(0.0, 1.0)
        known_5y = (
            (target_5y > 0.5)
            | (risk_mask[:, -1] > 0.5)
        )

        auxiliary_loss = torch.zeros(
            (),
            dtype=logits_float.dtype,
            device=logits_float.device,
        )
        if self.auxiliary_5y_weight > 0 and known_5y.any():
            # AMP安全的未来5年累计风险BCE。
            #
            # 对离散条件风险h_j：
            #   S_5y = Π_j(1-h_j)
            #        = exp[Σ_j log sigmoid(-logit_j)]
            #   R_5y = 1-S_5y
            #
            # 先计算累计风险的logit，再使用
            # binary_cross_entropy_with_logits，避免AMP下直接对概率
            # 调用binary_cross_entropy所产生的RuntimeError。
            log_survival_5y = torch.nn.functional.logsigmoid(
                -logits_float
            ).sum(dim=1)
            log_risk_5y = torch.log(
                torch.clamp(
                    -torch.expm1(log_survival_5y),
                    min=EPS,
                )
            )
            cumulative_risk_logit_5y = (
                log_risk_5y - log_survival_5y
            )

            auxiliary_sample = (
                torch.nn.functional.binary_cross_entropy_with_logits(
                    cumulative_risk_logit_5y[known_5y],
                    target_5y[known_5y],
                    reduction="none",
                )
            )
            auxiliary_loss = self._landmark_equal_mean(
                auxiliary_sample,
                landmark_index[known_5y],
            )

        ranking_loss = torch.zeros_like(auxiliary_loss)
        if self.ranking_weight > 0:
            case_mask = target_5y > 0.5
            control_mask = (
                (target_5y <= 0.5)
                & (risk_mask[:, -1] > 0.5)
            )
            if case_mask.any() and control_mask.any():
                case_risk = risk_5y[case_mask]
                control_risk = risk_5y[control_mask]
                pairwise_margin = (
                    case_risk[:, None]
                    - control_risk[None, :]
                )
                ranking_loss = torch.nn.functional.softplus(
                    -pairwise_margin / 0.10
                ).mean()

        smoothness_loss = torch.zeros_like(auxiliary_loss)
        if self.smoothness_weight > 0:
            smoothness_loss = (
                logits_float[:, 1:]
                - logits_float[:, :-1]
            ).pow(2).mean()

        total = (
            survival_loss
            + self.auxiliary_5y_weight * auxiliary_loss
            + self.ranking_weight * ranking_loss
            + self.smoothness_weight * smoothness_loss
        )
        parts = {
            "survival_loss": survival_loss.detach(),
            "auxiliary_5y_loss": auxiliary_loss.detach(),
            "ranking_loss": ranking_loss.detach(),
            "smoothness_loss": smoothness_loss.detach(),
        }
        return total, parts


def move_batch_to_device(
    batch: dict[str, torch.Tensor],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    return {
        "dynamic_sequence": batch["dynamic_sequence"].to(
            device=device,
            dtype=torch.float32,
            non_blocking=True,
        ),
        "row_mask": batch["row_mask"].to(
            device=device,
            dtype=torch.bool,
            non_blocking=True,
        ),
        "static_baseline": batch["static_baseline"].to(
            device=device,
            dtype=torch.float32,
            non_blocking=True,
        ),
        "age_at_landmark": batch["age_at_landmark"].to(
            device=device,
            dtype=torch.float32,
            non_blocking=True,
        ),
        "landmark_normalized": batch["landmark_normalized"].to(
            device=device,
            dtype=torch.float32,
            non_blocking=True,
        ),
        "landmark_bin": batch["landmark_bin"].to(
            device=device,
            dtype=torch.long,
            non_blocking=True,
        ),
        "event_target": batch["event_target"].to(
            device=device,
            dtype=torch.float32,
            non_blocking=True,
        ),
        "at_risk_mask": batch["at_risk_mask"].to(
            device=device,
            dtype=torch.float32,
            non_blocking=True,
        ),
        "patient_local": batch["patient_local"],
        "landmark_index": batch["landmark_index"].to(
            device=device,
            dtype=torch.long,
            non_blocking=True,
        ),
    }


def create_data_loaders(
    fold_data: dict[str, Any],
    batch_size: int,
    seed: int,
    device: torch.device,
) -> tuple[DataLoader, DataLoader]:
    generator = torch.Generator()
    generator.manual_seed(int(seed))

    train_landmarks = fold_data["train_sample_pairs"][:, 1]
    landmark_counts = np.bincount(
        train_landmarks,
        minlength=EXPECTED_LANDMARK_N,
    ).astype(np.float64)
    if np.any(landmark_counts <= 0):
        raise ValueError("训练集中存在没有样本的Landmark。")

    sample_weights = (
        1.0 / landmark_counts[train_landmarks]
    )
    sample_weights = (
        sample_weights / sample_weights.mean()
    )

    sampler = WeightedRandomSampler(
        weights=torch.as_tensor(
            sample_weights,
            dtype=torch.double,
        ),
        num_samples=len(sample_weights),
        replacement=True,
        generator=generator,
    )

    train_loader = DataLoader(
        fold_data["train_dataset"],
        batch_size=int(batch_size),
        sampler=sampler,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )
    validation_loader = DataLoader(
        fold_data["validation_dataset"],
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )
    return train_loader, validation_loader


def hazards_to_survival(
    hazard: np.ndarray,
) -> np.ndarray:
    hazard = np.clip(
        np.asarray(hazard, dtype=np.float64),
        EPS,
        1.0 - EPS,
    )
    return np.cumprod(1.0 - hazard, axis=1).astype(np.float32)


def build_model(
    parameters: dict[str, Any],
    device: torch.device,
) -> HybridAttentionLSTMSurvival:
    return HybridAttentionLSTMSurvival(
        dynamic_n=EXPECTED_ENHANCED_DYNAMIC_N,
        static_n=EXPECTED_STATIC_N,
        hidden_size=int(parameters["hidden_size"]),
        num_layers=int(parameters["num_layers"]),
        dropout=float(parameters["dropout"]),
        projection_size=int(parameters["projection_size"]),
        bidirectional=bool(parameters["bidirectional"]),
        pooling_mode=str(parameters["pooling_mode"]),
        static_hidden=int(parameters["static_hidden"]),
        summary_hidden=int(parameters["summary_hidden"]),
        horizon_embed_dim=int(parameters["horizon_embed_dim"]),
        future_n=EXPECTED_FUTURE_INTERVAL_N,
    ).to(device)


def predict_loader(
    model: nn.Module,
    loader: DataLoader,
    loss_function: LandmarkBalancedSurvivalLoss,
    device: torch.device,
    use_amp: bool,
) -> tuple[float, dict[str, np.ndarray]]:
    model.eval()
    total_loss = 0.0
    total_sample_n = 0
    hazards = []
    patient_local = []
    landmark_index = []

    with torch.no_grad():
        for raw_batch in loader:
            batch = move_batch_to_device(raw_batch, device)
            with autocast_context(device, use_amp):
                logits = model(
                    dynamic_sequence=batch["dynamic_sequence"],
                    row_mask=batch["row_mask"],
                    static_baseline=batch["static_baseline"],
                    age_at_landmark=batch["age_at_landmark"],
                    landmark_normalized=batch["landmark_normalized"],
                    landmark_bin=batch["landmark_bin"],
                )
                loss, _ = loss_function(
                    logits,
                    batch["event_target"],
                    batch["at_risk_mask"],
                    batch["landmark_index"],
                )

            batch_n = int(logits.shape[0])
            total_loss += float(loss.item()) * batch_n
            total_sample_n += batch_n
            hazards.append(
                torch.sigmoid(logits.float())
                .cpu()
                .numpy()
                .astype(np.float32)
            )
            patient_local.append(
                batch["patient_local"].numpy().astype(np.int32)
            )
            landmark_index.append(
                batch["landmark_index"]
                .cpu()
                .numpy()
                .astype(np.int8)
            )

    return (
        total_loss / max(total_sample_n, 1),
        {
            "hazard": np.concatenate(hazards, axis=0),
            "patient_local": np.concatenate(patient_local, axis=0),
            "landmark_index": np.concatenate(landmark_index, axis=0),
        },
    )


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_function: LandmarkBalancedSurvivalLoss,
    grad_scaler: Any,
    device: torch.device,
    use_amp: bool,
) -> float:
    model.train()
    total_loss = 0.0
    total_sample_n = 0

    for raw_batch in loader:
        batch = move_batch_to_device(raw_batch, device)
        optimizer.zero_grad(set_to_none=True)

        with autocast_context(device, use_amp):
            logits = model(
                dynamic_sequence=batch["dynamic_sequence"],
                row_mask=batch["row_mask"],
                static_baseline=batch["static_baseline"],
                age_at_landmark=batch["age_at_landmark"],
                landmark_normalized=batch["landmark_normalized"],
                landmark_bin=batch["landmark_bin"],
            )
            loss, _ = loss_function(
                logits,
                batch["event_target"],
                batch["at_risk_mask"],
                batch["landmark_index"],
            )

        if not torch.isfinite(loss):
            raise FloatingPointError("训练损失不是有限数。")

        grad_scaler.scale(loss).backward()
        grad_scaler.unscale_(optimizer)
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=GRADIENT_CLIP_NORM,
        )
        if not torch.isfinite(gradient_norm):
            raise FloatingPointError("梯度范数不是有限数。")

        grad_scaler.step(optimizer)
        grad_scaler.update()

        batch_n = int(logits.shape[0])
        total_loss += float(loss.item()) * batch_n
        total_sample_n += batch_n

    return total_loss / max(total_sample_n, 1)


def clone_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }


def ensemble_snapshot_predictions(
    model: nn.Module,
    snapshot_states: list[dict[str, torch.Tensor]],
    validation_loader: DataLoader,
    loss_function: LandmarkBalancedSurvivalLoss,
    device: torch.device,
    use_amp: bool,
) -> np.ndarray:
    hazard_predictions = []
    for state in snapshot_states:
        model.load_state_dict(state)
        _, prediction = predict_loader(
            model,
            validation_loader,
            loss_function,
            device,
            use_amp,
        )
        hazard_predictions.append(prediction["hazard"])
    return np.mean(
        np.stack(hazard_predictions, axis=0),
        axis=0,
    ).astype(np.float32)


def normalize_parameters(parameters: dict[str, Any]) -> dict[str, Any]:
    return {
        "hidden_size": int(parameters["hidden_size"]),
        "num_layers": int(parameters["num_layers"]),
        "dropout": float(parameters["dropout"]),
        "projection_size": int(parameters["projection_size"]),
        "bidirectional": bool(parameters["bidirectional"]),
        "pooling_mode": str(parameters["pooling_mode"]),
        "static_hidden": int(parameters["static_hidden"]),
        "summary_hidden": int(parameters["summary_hidden"]),
        "horizon_embed_dim": int(parameters["horizon_embed_dim"]),
        "learning_rate": float(parameters["learning_rate"]),
        "weight_decay": float(parameters["weight_decay"]),
        "batch_size": int(parameters["batch_size"]),
        "positive_weight": float(parameters["positive_weight"]),
        "focal_gamma": float(parameters["focal_gamma"]),
        "auxiliary_5y_weight": float(parameters["auxiliary_5y_weight"]),
        "ranking_weight": float(parameters["ranking_weight"]),
        "smoothness_weight": float(parameters["smoothness_weight"]),
    }




# ===================== 已跑通的数据读取/特征构造 =====================
LM = np.array([0, 12, 24, 36, 48, 60], dtype=np.int32)


BINS = LM // 6


GRID = np.arange(11, dtype=np.float64) * 6


TOL = 1e-6


SN_DIR = 'rolling_5y_finaldata_step3_raw_features'


CQ_DIR = '重庆外部验证_step5_GLU更新'


CQ_SOURCE = '重庆_source47_imputed_GLU_updated.csv'


PARAM_TYPES = {
    'hidden_size': int, 'num_layers': int, 'dropout': float,
    'projection_size': int, 'bidirectional': bool, 'pooling_mode': str,
    'static_hidden': int, 'summary_hidden': int, 'horizon_embed_dim': int,
    'learning_rate': float, 'weight_decay': float, 'batch_size': int,
    'positive_weight': float, 'focal_gamma': float,
    'auxiliary_5y_weight': float, 'ranking_weight': float,
    'smoothness_weight': float,
}


LAB = ['HIVRNA_log10', 'CD4', 'CD8', 'Urea', 'WBC', 'PLT', 'HB',
       'TC', 'TG', 'HDL', 'LDL', 'GLU', 'ALT', 'AST', 'eGFR']


STATUS = ['CVD_status', 'diabetes_status', 'hypertension_status',
          'hypercholesterolemia_status', 'HBV_status', 'HCV_status']


MEDS = ['antidiabetic_med', 'antihypertensive_med', 'antilipid_med']


ART = ['TDF_NNRTI_3TC_FTC', 'TDF_PI_3TC_FTC', 'nonTDF_PI', 'BIC_FTC_TAF',
       'EVGc_FTC_TAF', 'TDF_INSTI_3TC_FTC', 'nonTDF_DTG', 'nonTDF_traditional_NNRTI']


CURRENT = ['current_' + x for x in ART]


CUM = [x + '_cum_month' for x in ART]


DYNAMIC = LAB + STATUS + MEDS + CURRENT + CUM


EXPECTED_CONT = ['Age', 'BMI'] + LAB + CUM


EXPECTED_BINARY = ['Oppinfection'] + STATUS[:4] + MEDS + STATUS[4:] + CURRENT


EXPECTED_CATEGORICAL = ['Sex', 'Marriage', 'Course', 'WHOstage']


class BridgeError(RuntimeError):
    """Safe error: descriptions must not contain patient rows/identifiers."""


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise BridgeError('Required JSON is missing: ' + str(path))
    if path.stat().st_size > 10_000_000:
        raise BridgeError('Unexpected JSON size: ' + str(path))
    obj = json.loads(path.read_text(encoding='utf-8-sig'))
    if not isinstance(obj, dict):
        raise BridgeError('Expected a JSON object: ' + str(path))
    return obj


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def normalise_parameters(obj: dict[str, Any], where: str) -> dict[str, Any]:
    missing = sorted(set(PARAM_TYPES) - obj.keys())
    if missing:
        raise BridgeError(where + ' does not contain the full model parameter set: ' + ', '.join(missing))
    p = {}
    for k, kind in PARAM_TYPES.items():
        value = obj[k]
        if kind is bool:
            if not isinstance(value, bool):
                raise BridgeError(where + ': ' + k + ' must be JSON true/false, not a string.')
            p[k] = value
        elif kind is str:
            if not isinstance(value, str):
                raise BridgeError(where + ': invalid string setting ' + k)
            p[k] = value
        elif kind is int:
            if isinstance(value, bool) or not float(value).is_integer() or float(value) <= 0:
                raise BridgeError(where + ': invalid positive integer setting ' + k)
            p[k] = int(value)
        else:
            p[k] = float(value)
            if not np.isfinite(p[k]) or p[k] < 0:
                raise BridgeError(where + ': invalid nonnegative setting ' + k)
    if p['dropout'] >= 1 or p['learning_rate'] <= 0 or p['positive_weight'] <= 0:
        raise BridgeError(where + ': invalid dropout, learning rate, or positive weight.')
    if p['pooling_mode'] not in ('last_attention', 'last_attention_mean'):
        raise BridgeError(where + ': unsupported pooling mode for the recovered source.')
    return p


def read_csv_safe(path: Path, usecols=None) -> pd.DataFrame:
    if not path.is_file():
        raise BridgeError('Required CSV is missing: ' + str(path))
    # Decode strictly; never silently replace characters in identifiers/categories.
    for encoding in ('utf-8-sig', 'gb18030'):
        try:
            return pd.read_csv(path, encoding=encoding, dtype='string', usecols=usecols,
                               keep_default_na=True, low_memory=False)
        except UnicodeDecodeError:
            continue
    raise BridgeError('Cannot decode CSV using UTF-8 or GB18030: ' + str(path))


def numeric(s: pd.Series, label: str) -> np.ndarray:
    try:
        out = pd.to_numeric(s, errors='raise').to_numpy(dtype=np.float64)
    except (ValueError, TypeError):
        raise BridgeError('Invalid numeric data in ' + label) from None
    if not np.isfinite(out).all():
        raise BridgeError('Missing/non-finite numeric data in ' + label)
    return out


def clean_strings(s: pd.Series, label: str) -> np.ndarray:
    s = s.astype('string').str.strip()
    if s.isna().any() or s.eq('').any():
        raise BridgeError('Missing/empty text in ' + label)
    return s.astype(str).to_numpy()


@dataclass
class SourceData:
    tag: str
    meta: pd.DataFrame
    continuous: np.ndarray
    binary: np.ndarray
    categorical: np.ndarray
    row_mask: np.ndarray
    source_directory: str

    @property
    def n(self) -> int:
        return len(self.meta)


def read_meta(path: Path, tag: str) -> pd.DataFrame:
    frame = read_csv_safe(path)
    mapping = ({'center': 'centre', 'event': 'event', 'observed_time_month': 'time'}
               if tag == 'SN' else {'data': 'centre', 'CKDstatus': 'event', 'interval': 'time'})
    required = ['ID', 'patient_index', *mapping]
    if any(c not in frame for c in required):
        raise BridgeError('Patient metadata schema mismatch: ' + str(path))
    ids = clean_strings(frame['ID'], tag + ' patient ID')
    if pd.Index(ids).has_duplicates:
        raise BridgeError(tag + ' patient metadata has duplicate ID values; not automatically deduplicated.')
    index = numeric(frame['patient_index'], tag + ' patient_index')
    if not np.array_equal(index, np.arange(len(frame))):
        raise BridgeError(tag + ' patient_index is not equal to row order; tensor alignment cannot be assumed.')
    centre = clean_strings(frame[mapping_key(mapping, 'centre')], tag + ' centre')
    event = numeric(frame[mapping_key(mapping, 'event')], tag + ' event')
    time_month = numeric(frame[mapping_key(mapping, 'time')], tag + ' time')
    if not np.isin(event, [0, 1]).all() or (time_month <= 0).any():
        raise BridgeError(tag + ' outcome/time values are invalid.')
    archived = frame['analysis_split'].fillna('unspecified').astype(str).to_numpy() if 'analysis_split' in frame else np.repeat('external_source', len(frame))
    return pd.DataFrame({'source': tag, 'source_patient_index': index.astype(np.int32),
                         'ID_LOCAL_ONLY': ids, 'centre_label': centre,
                         'centre_key': [tag + '::' + x for x in centre],
                         'event': event.astype(np.int8), 'time': time_month,
                         'archived_split': archived})


def mapping_key(mapping: dict, target: str) -> str:
    return next(k for k, v in mapping.items() if v == target)


def checked_npy(path: Path, shape: tuple, *, boolean: bool = False) -> np.ndarray:
    if not path.is_file():
        raise BridgeError('Required array is missing: ' + str(path))
    try:
        a = np.load(path, mmap_mode='r', allow_pickle=False)
    except ValueError:
        raise BridgeError('Array cannot be safely memory-mapped without pickle: ' + str(path)) from None
    if a.shape != shape:
        raise BridgeError('Array shape mismatch in ' + str(path) + '; got ' + str(a.shape) + ', expected ' + str(shape))
    if boolean and (a.dtype != np.dtype('bool')):
        raise BridgeError('Expected a Boolean mask: ' + str(path))
    return a


def load_cq_categories(root: Path, meta: pd.DataFrame, mask: np.ndarray) -> np.ndarray:
    # CQ categorical NPY may be object/pickle. Use the exact category columns of
    # the already inventoried source CSV instead; do not load arbitrary pickle.
    path = root / CQ_DIR / CQ_SOURCE
    cols = ['ID', 'time_bin', *EXPECTED_CATEGORICAL]
    frame = read_csv_safe(path, cols)
    ids = clean_strings(frame['ID'], 'CQ panel ID')
    bins = numeric(frame['time_bin'], 'CQ time_bin')
    if not np.equal(bins, np.floor(bins)).all() or (bins < 0).any():
        raise BridgeError('CQ time_bin contains non-integer/negative entries.')
    all_panel_ids = set(ids)
    if all_panel_ids != set(meta['ID_LOCAL_ONLY']):
        raise BridgeError('CQ panel and patient metadata have different ID sets; cannot silently subset.')
    keep = bins <= 10
    selected = frame.loc[keep].copy()
    selected_ids = ids[keep]
    tb = bins[keep].astype(np.int64)
    lookup = pd.Index(meta['ID_LOCAL_ONLY'])
    idx = lookup.get_indexer(selected_ids)
    if (idx < 0).any():
        raise BridgeError('CQ category panel has unmatched patient indices.')
    positions = idx * 11 + tb
    if len(np.unique(positions)) != len(positions):
        raise BridgeError('CQ category panel has duplicate patient/time-bin rows.')
    check_mask = np.zeros_like(mask, dtype=bool)
    check_mask[idx, tb] = True
    if not np.array_equal(check_mask, mask):
        raise BridgeError('CQ category panel row mask differs from the saved tensor mask.')
    cols_arr = [clean_strings(selected[c], 'CQ ' + c) for c in EXPECTED_CATEGORICAL]
    maxlen = max(1, *(max(map(len, c)) for c in cols_arr))
    if maxlen > 128:
        raise BridgeError('Unexpectedly long category strings; inspect CQ coding locally.')
    out = np.full((len(meta), 11, 4), '__NO_TIME_ROW__', dtype=f'<U{max(20, maxlen)}')
    out[idx, tb, :] = np.column_stack(cols_arr)
    return out


def load_sources(root: Path) -> list[SourceData]:
    sn_path, cq_path = root / SN_DIR, root / CQ_DIR
    sn = read_meta(sn_path / 'patient_info_with_fold.csv', 'SN')
    cq = read_meta(cq_path / 'patient_info.csv', 'CQ')
    if sn['centre_label'].nunique() != 2 or cq['centre_label'].nunique() != 1:
        raise BridgeError('Expected two SN source centres and one CQ source centre; centre coding requires review.')
    sources = []
    for tag, path, meta in [('SN', sn_path, sn), ('CQ', cq_path, cq)]:
        n = len(meta)
        cont = checked_npy(path / 'continuous_raw_0_60.npy', (n, 11, 25))
        binary = checked_npy(path / 'binary_raw_0_60.npy', (n, 11, 18))
        mask = checked_npy(path / 'sequence_row_mask.npy', (n, 11), boolean=True)
        if (~mask.any(axis=1)).any():
            raise BridgeError(tag + ': a patient has no history row in 0-60 months.')
        if tag == 'SN':
            cat = checked_npy(path / 'categorical_raw_0_60.npy', (n, 11, 4))
            if cat.dtype.kind not in 'US':
                raise BridgeError('SN category tensor is not a non-pickle text array.')
        else:
            # Saved feature names verify raw continuous/binary column order.
            f = read_csv_safe(path / 'feature_names.csv')
            if 'feature_name' not in f:
                raise BridgeError('CQ feature_names.csv has no feature_name column.')
            names = clean_strings(f['feature_name'], 'CQ feature names').tolist()
            if names[:43] != EXPECTED_CONT + EXPECTED_BINARY:
                raise BridgeError('CQ continuous/binary variable order differs from SN schema.')
            cat = load_cq_categories(root, meta, mask)
        sources.append(SourceData(tag, meta, cont, binary, cat, mask, str(path)))
    return sources


def build_labels(meta: pd.DataFrame, rowmask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = len(meta)
    event = meta['event'].to_numpy(dtype=bool)
    time_month = meta['time'].to_numpy(dtype=np.float64)
    target = np.zeros((n, 6, 10), dtype=np.float32)
    risk = np.zeros((n, 6, 10), dtype=bool)
    eligible = np.zeros((n, 6), dtype=bool)
    for l, month in enumerate(LM):
        valid = (time_month > month + TOL) & rowmask[:, :BINS[l] + 1].any(axis=1)
        eligible[:, l] = valid
        residual = time_month - month
        event_bin = np.ceil((residual - TOL) / 6).astype(np.int64) - 1
        for j in range(10):
            risk[:, l, j] = valid & ((event & (event_bin >= j)) | (~event & (residual >= 6 * (j + 1) - TOL)))
            target[:, l, j] = (valid & event & (event_bin == j)).astype(np.float32)
    origins = eligible & risk.any(axis=2)
    if np.any((target > 0) & ~risk) or (target.sum(axis=2) > 1).any():
        raise BridgeError('Constructed discrete labels failed consistency checks.')
    return target, risk, eligible, origins


def take(sources: list[SourceData], name: str, idx: np.ndarray) -> np.ndarray:
    # Preserve requested global order, without concatenating giant categorical tensors.
    cut = sources[0].n
    pieces = []
    where = []
    for source, lo, hi in [(sources[0], 0, cut), (sources[1], cut, cut + sources[1].n)]:
        pos = np.where((idx >= lo) & (idx < hi))[0]
        if len(pos):
            pieces.append(np.asarray(getattr(source, name)[idx[pos] - lo]))
            where.append(pos)
    if not pieces:
        raise BridgeError('Empty requested input partition.')
    stacked = np.concatenate(pieces, axis=0)
    order = np.concatenate(where)
    return stacked[np.argsort(order)]


def fit_new_preprocessor(sources: list[SourceData], idx: np.ndarray) -> dict:
    # Exactly retain the recovered Step4 numeric treatment: StandardScaler, no
    # invented median imputation. Missing finite input requirements are explicit.
    scaler = StandardScaler()
    levels = [set() for _ in range(4)]
    row_n = 0
    for start in range(0, len(idx), 1024):
        ii = idx[start:start + 1024]
        mask = take(sources, 'row_mask', ii)
        c = take(sources, 'continuous', ii)[mask]
        b = take(sources, 'binary', ii)[mask]
        cat = take(sources, 'categorical', ii)[mask].astype(str)
        if not np.isfinite(c).all():
            raise BridgeError('Non-finite continuous inputs remain. Original Step4 would not produce finite model inputs; no imputation policy is invented in this analysis.')
        if not np.isin(b, [0, 1]).all():
            raise BridgeError('Binary inputs contain missing/invalid values; no automatic replacement.')
        for j in range(4):
            for value in np.unique(cat[:, j]):
                if value.strip().lower() in ('', 'nan', 'none', '<na>', '__no_time_row__'):
                    raise BridgeError('Missing/padding category occurs on a real observation row.')
                levels[j].add(value)
        scaler.partial_fit(c)
        row_n += len(c)
    categories = [np.array(sorted(s), dtype=str) for s in levels]
    try:
        enc = OneHotEncoder(categories=categories, handle_unknown='ignore', sparse_output=False, dtype=np.float32)
    except TypeError:
        enc = OneHotEncoder(categories=categories, handle_unknown='ignore', sparse=False, dtype=np.float32)
    # Categories come ONLY from fit patients. This row simply initialises sklearn.
    enc.fit(np.array([[x[0] for x in categories]], dtype=object))
    names = EXPECTED_CONT + EXPECTED_BINARY + enc.get_feature_names_out(EXPECTED_CATEGORICAL).tolist()
    if len(names) != 57:
        raise BridgeError('Training-only category coding yields ' + str(len(names)) + ' base features, not 57. Do not copy held-out categories to force a match.')
    return {'scaler': scaler, 'encoder': enc, 'names': names, 'fit_rows': row_n}


def make_model_inputs(sources: list[SourceData], ids: np.ndarray, pre: dict) -> dict:
    cont = take(sources, 'continuous', ids).astype(np.float32)
    binary = take(sources, 'binary', ids).astype(np.float32)
    cat = take(sources, 'categorical', ids).astype(str)
    row = take(sources, 'row_mask', ids).astype(bool)
    if not np.isfinite(cont[row]).all() or not np.isin(binary[row], [0, 1]).all():
        raise BridgeError('Selected input rows contain non-finite/invalid input values.')
    x = np.zeros((len(ids), 11, len(pre['names'])), dtype=np.float32)
    x[row] = np.concatenate([pre['scaler'].transform(cont[row]), binary[row],
                            pre['encoder'].transform(cat[row])], axis=1).astype(np.float32)
    names = pre['names']
    dynamic = x[:, :, [names.index(k) for k in DYNAMIC]]
    lab_raw = cont[:, :, [EXPECTED_CONT.index(k) for k in LAB]]
    observed = np.isfinite(lab_raw) & row[:, :, None]
    time_since = np.zeros(observed.shape, dtype=np.float32)
    delta = np.zeros_like(time_since)
    last_step = np.full((len(ids), 15), -1, dtype=np.int16)
    last_value = np.zeros((len(ids), 15), dtype=np.float32)
    seen = np.zeros((len(ids), 15), dtype=bool)
    for t in range(11):
        current = dynamic[:, t, :15]
        now = observed[:, t]
        elapsed = np.clip((t - last_step).astype(np.float32) * 0.5, 0, 5)
        elapsed[~seen] = min((t + 1) * 0.5, 5)
        time_since[:, t] = np.where(row[:, t, None], np.where(now, 0., elapsed / 5.), 0.)
        delta[:, t] = np.where(now & seen, current - last_value, 0.)
        last_value = np.where(now, current, last_value)
        last_step = np.where(now, t, last_step)
        seen |= now
    enhanced = np.concatenate([dynamic, observed.astype(np.float32), time_since, delta], axis=2)
    enhanced[~row] = 0.
    first = np.argmax(row, axis=1)
    static_names = ['BMI', 'Oppinfection'] + [k for k in names if k.startswith(('Sex_', 'Marriage_', 'Course_', 'WHOstage_'))]
    static = x[np.arange(len(ids)), first][:, [names.index(k) for k in static_names]]
    age0 = cont[np.arange(len(ids)), first, EXPECTED_CONT.index('Age')] - first * 0.5
    if static.shape[1] != 16 or enhanced.shape[2] != 85 or not np.isfinite(enhanced).all():
        raise BridgeError('Model input construction failed fixed-dimensional consistency checks.')
    return {'dynamic': enhanced, 'row_mask': row, 'static': static, 'age0': age0.astype(np.float32),
            'age_mean': float(pre['scaler'].mean_[0]), 'age_scale': float(pre['scaler'].scale_[0])}


class LandmarkDataset(Dataset):
    def __init__(self, inputs: dict, pairs: np.ndarray, y: np.ndarray, mask: np.ndarray):
        self.inputs, self.sample_pairs, self.y, self.mask = inputs, pairs, y, mask
    def __len__(self) -> int:
        return len(self.sample_pairs)
    def __getitem__(self, pos: int) -> dict:
        p, l = self.sample_pairs[pos]
        a = self.inputs
        return {'dynamic_sequence': a['dynamic'][p], 'row_mask': a['row_mask'][p],
            'static_baseline': a['static'][p],
            'age_at_landmark': np.float32((a['age0'][p] + LM[l] / 12. - a['age_mean']) / a['age_scale']),
            'landmark_normalized': np.float32(LM[l] / 60.), 'landmark_bin': np.int64(BINS[l]),
            'event_target': self.y[p, l], 'at_risk_mask': self.mask[p, l].astype(np.float32),
            'patient_local': np.int64(p), 'landmark_index': np.int64(l)}




# ===================== 完整中心轮转训练/校准/评价 =====================
VERSION = 'crosscenter_fixed_input_train_v1_20260909'
METRIC_TIMES = np.array([6,12,18,24,30,36,42,48,54,59.999], dtype=float)
SCOPES = {'six_landmarks': [0,1,2,3,4,5], 'landmarks_0_1_3_5': [0,1,3,5]}
METRICS = ['uno_c', 'iauc', 'ibs']
PARAMETER_SOURCE = 'rolling_5y_finaldata_step10e_lstm_v2_final_oof_selected_medium12/lstm_v2_summary.json'
LOG_PATH = None
LIMITATIONS = [
    'Post-hoc centre-rotation sensitivity conditional on already processed/imputed inputs.',
    'Earlier imputation was not refitted: its provenance and possible centre information reuse remain unverified.',
    'Architecture/hyperparameters were previously selected using Shenzhen/Nanning; this is not fully nested model-development validation.',
    'Archived internal test patients are excluded by default; original primary files are not modified.',
    'Numerical IDs and centre labels do not independently verify real-person overlap across hospitals.',
    'Continuous-time evaluation uses the archived valid-origin population; short censored origins with no complete interval are excluded.',
    'The source definition of observation channels uses finite input values; it does not reconstruct original measurement missingness.',
]


def json_ready(value):
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_ready(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_ready(value.tolist())
    if isinstance(value, (np.integer,)): return int(value)
    if isinstance(value, (np.bool_,)): return bool(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, Path): return str(value)
    return value


def save_json_atomic(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(json_ready(value), ensure_ascii=False, indent=2,
                              allow_nan=False), encoding='utf-8')
    os.replace(tmp, path)


def save_npz_atomic(path, **arrays):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('wb') as f: np.savez_compressed(f, **arrays)
    os.replace(tmp, path)


def save_torch_atomic(path, value):
    tmp = Path(str(path) + '.tmp')
    torch.save(value, tmp)
    os.replace(tmp, path)


def csv_save(df, path):
    tmp = Path(str(path) + '.tmp')
    df.to_csv(tmp, index=False, encoding='utf-8-sig')
    os.replace(tmp, path)


def log(text):
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {text}", flush=True)


class TrainingTee:
    def __init__(self, stream, f): self.stream, self.f = stream, f
    def write(self, text):
        self.stream.write(text); self.f.write(text); self.f.flush()
        return len(text)
    def flush(self): self.stream.flush(); self.f.flush()
    def __getattr__(self, key): return getattr(self.stream, key)


def seed_training(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_rng(loader):
    state = np.random.get_state()
    result = {'python': random.getstate(), 'torch': torch.get_rng_state(),
              'numpy': [state[0], state[1].tolist(), state[2], state[3], state[4]],
              'sampler': loader.sampler.generator.get_state()}
    if torch.cuda.is_available(): result['cuda'] = torch.cuda.get_rng_state_all()
    return result


def restore_rng(value, loader):
    random.setstate(value['python']); torch.set_rng_state(value['torch'])
    s = value['numpy']
    np.random.set_state((s[0], np.array(s[1], dtype=np.uint32), s[2], s[3], s[4]))
    loader.sampler.generator.set_state(value['sampler'])
    if 'cuda' in value and torch.cuda.is_available(): torch.cuda.set_rng_state_all(value['cuda'])


def runtime_info(root):
    d = {'python': sys.executable, 'torch': str(torch.__version__),
         'numpy': np.__version__, 'pandas': pd.__version__, 'device': DEVICE,
         'cuda_available': bool(torch.cuda.is_available()),
         'disk_free_GiB': shutil.disk_usage(root).free / 1024**3}
    try:
        import psutil
        d['available_memory_GiB'] = psutil.virtual_memory().available / 1024**3
        d['process_RSS_GiB'] = psutil.Process().memory_info().rss / 1024**3
    except ImportError: pass
    return d


def acquire_lock(out):
    lock = out / 'RUNNING.lock'
    if lock.exists():
        old = read_json(lock)
        old_pid = int(old.get('pid', -1))
        try:
            import psutil
            active = old_pid > 0 and psutil.pid_exists(old_pid)
        except ImportError:
            active = old_pid == os.getpid()
        if active:
            raise RuntimeError('同一结果目录已有任务运行，请勿同时启动两个Notebook。')
        lock.unlink()
    fd = os.open(str(lock), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, 'w', encoding='utf-8') as f:
        json.dump({'pid': os.getpid(), 'created': datetime.now().isoformat()}, f)
    return lock


def load_metric_functions():
    # 使用你已安装的生存评价库，不安装/升级环境。
    try:
        from sksurv.metrics import brier_score, cumulative_dynamic_auc, concordance_index_ipcw
        from sksurv.nonparametric import CensoringDistributionEstimator
        from sksurv.util import Surv
    except ImportError as exc:
        raise ImportError('当前Python缺少scikit-survival；请使用已安装该库的原Jupyter环境。') from exc
    return brier_score, cumulative_dynamic_auc, concordance_index_ipcw, CensoringDistributionEstimator, Surv


def make_surv(event, t):
    y = np.empty(len(t), dtype=[('event', '?'), ('time', '<f8')])
    y['event'] = np.asarray(event, dtype=bool)
    y['time'] = np.asarray(t, dtype=float)
    if len(y) == 0 or not np.isfinite(y['time']).all() or np.any(y['time'] <= 0):
        raise ValueError('评价需要非空、有限且严格为正的随访时间。')
    return y


def km_at(event, t, grid):
    order = np.argsort(t, kind='mergesort')
    ts = np.asarray(t)[order]; ev = np.asarray(event, dtype=int)[order]
    unique, first, counts = np.unique(ts, return_index=True, return_counts=True)
    deaths = np.add.reduceat(ev, first)
    at_risk = len(ts) - np.r_[0, np.cumsum(counts)[:-1]]
    surv = np.cumprod(1.0 - deaths / at_risk)
    pos = np.searchsorted(unique, grid, side='right') - 1
    out = np.ones(len(grid), dtype=float)
    present = pos >= 0
    out[present] = surv[pos[present]]
    return out


def make_references(meta, eligible, indices):
    refs = {}
    for l, month in enumerate(LM):
        ids = np.asarray(indices)[eligible[np.asarray(indices), l]]
        residual = meta['time'].to_numpy()[ids] - month
        ev = meta['event'].to_numpy()[ids].astype(bool)
        # 截至61月作评价参考，不延长任何人的实际随访。
        refs[l] = make_surv(ev & (residual <= 61.0), np.minimum(residual, 61.0))
    return refs


def metric_one(ref, ev, time_values, risk):
    brier_score, dynamic_auc, cindex_ipcw, Censoring, _ = METRIC_FUNCTIONS
    ev = np.asarray(ev, dtype=bool); t = np.asarray(time_values, dtype=float)
    risk = np.asarray(risk, dtype=float)
    answer = {'uno_c': np.nan, 'iauc': np.nan, 'ibs': np.nan,
              'auc': np.full(10, np.nan), 'brier': np.full(10, np.nan), 'notes': []}
    if len(t) < 2 or risk.shape != (len(t), 10):
        answer['notes'].append('insufficient_evaluation_rows'); return answer
    if not np.isfinite(risk).all() or np.any((risk < -1e-7) | (risk > 1+1e-7)):
        raise ValueError('预测风险包含非法值。')
    if np.any(np.diff(risk, axis=1) < -1e-6): raise ValueError('累计风险不单调。')
    y = make_surv(ev, t)
    if t.max() <= METRIC_TIMES[-1] or ref['time'].max() <= METRIC_TIMES[-1]:
        answer['notes'].append('insufficient_followup_for_common_5y_grid'); return answer
    in_support = (METRIC_TIMES >= t.min()) & (METRIC_TIMES < t.max())
    try:
        if in_support.any():
            _, bs = brier_score(ref, y, 1.0-risk[:, in_support], METRIC_TIMES[in_support])
            answer['brier'][in_support] = bs
        early = METRIC_TIMES < t.min()
        if early.any():
            g = Censoring().fit(ref).predict_proba(METRIC_TIMES[early])
            if np.any(g <= 0): raise ValueError('zero censoring survival')
            answer['brier'][early] = (risk[:, early]**2).mean(axis=0) / g
        if np.isfinite(answer['brier']).all():
            integ = np.trapezoid if hasattr(np, 'trapezoid') else np.trapz
            answer['ibs'] = float(integ(answer['brier'], METRIC_TIMES) / (METRIC_TIMES[-1]-METRIC_TIMES[0]))
    except (ValueError, ArithmeticError) as exc:
        answer['notes'].append('brier_unavailable:' + str(exc))
    cases = np.array([np.sum(ev & (t <= q)) for q in METRIC_TIMES])
    controls = np.array([np.sum(t > q) for q in METRIC_TIMES])
    use_auc = in_support & (cases > 0) & (controls > 0)
    try:
        if use_auc.any():
            values, _ = dynamic_auc(ref, y, risk[:, use_auc], METRIC_TIMES[use_auc])
            answer['auc'][use_auc] = values
        s = km_at(ev, t, METRIC_TIMES)
        mass = -np.diff(np.r_[1., s])
        positive_mass = mass > 1e-14
        if (1-s[-1]) > 0 and np.isfinite(answer['auc'][positive_mass]).all():
            # 与scikit-survival的KM事件质量加权均值一致。
            # 首个事件前AUC未定义但其事件质量为0，不虚构AUC=0.5。
            answer['iauc'] = float(np.dot(answer['auc'][positive_mass], mass[positive_mass]) / (1-s[-1]))
        if not use_auc.all(): answer['notes'].append('some_grid_AUCs_not_estimable; zero-event weights are zero')
    except (ValueError, ArithmeticError) as exc:
        answer['notes'].append('auc_unavailable:' + str(exc))
    try:
        if ev.any():
            answer['uno_c'] = float(cindex_ipcw(ref, y, risk[:, -1], tau=float(METRIC_TIMES[-1]))[0])
    except (ValueError, ArithmeticError) as exc:
        answer['notes'].append('cindex_unavailable:' + str(exc))
    return answer


def evaluate_pairs(meta, pairs, risk, refs):
    rows, horizon_rows = [], []
    for l, month in enumerate(LM):
        sel = pairs[:, 1] == l
        ids = pairs[sel, 0]
        residual = meta['time'].to_numpy()[ids] - month
        events = meta['event'].to_numpy()[ids].astype(bool) & (residual <= 60.0 + TOL)
        t = np.minimum(residual, 60.)
        m = metric_one(refs[l], events, t, risk[sel])
        row = {'landmark_index': l, 'landmark_year': float(month/12), 'n': len(ids),
               'events_5y': int(events.sum()), **{k:m[k] for k in METRICS},
               'metric_notes': ';'.join(m['notes'])}
        rows.append(row)
        observed = 1-km_at(events, t, np.array([12.,36.,60.])) if len(t) else np.full(3,np.nan)
        for k, (hor, pos) in enumerate([(12,1),(36,5),(60,9)]):
            horizon_rows.append({'landmark_index': l, 'landmark_year': month/12.,
                'horizon_year': hor/12., 'n': len(ids),
                'events_within_horizon': int(np.sum(events & (t <= hor))),
                'auc': float(m['auc'][pos]), 'brier': float(m['brier'][pos]),
                'mean_predicted_risk': float(risk[sel, pos].mean()) if len(ids) else np.nan,
                'KM_observed_risk': float(observed[k])})
    return pd.DataFrame(rows), pd.DataFrame(horizon_rows)


def strict_summary(table):
    summaries = []
    for scope, lm_indices in SCOPES.items():
        sub = table.set_index('landmark_index').reindex(lm_indices)
        row = {'scope': scope}
        for k in METRICS:
            val = sub[k].to_numpy(float)
            row[k] = float(val.mean()) if np.isfinite(val).all() else np.nan
            row[k+'_estimable_landmarks'] = int(np.isfinite(val).sum())
        summaries.append(row)
    return pd.DataFrame(summaries)


def fold_parts(meta, active, r):
    centres = meta['centre_key'].to_numpy()
    held = np.where(centres == r)[0].astype(np.int32)
    pool = np.where(centres != r)[0].astype(np.int32)
    strata = centres[pool] + '__' + meta['event'].to_numpy()[pool].astype(str)
    skf = StratifiedKFold(n_splits=N_INNER_FOLDS, shuffle=True, random_state=PARTITION_SEED)
    parts = []
    for f, (train_pos, oof_pos) in enumerate(skf.split(pool, strata)):
        rest = pool[train_pos]
        fit, stop = train_test_split(rest, test_size=EARLYSTOP_FRACTION,
            stratify=centres[rest]+'__'+meta['event'].to_numpy()[rest].astype(str),
            random_state=PARTITION_SEED+100*f)
        parts.append({k: np.sort(np.asarray(v,dtype=np.int32)) for k,v in
                      [('fit',fit),('stop',stop),('oof',pool[oof_pos]),('held',held)]})
    return pool, held, parts


def build_all_inputs(sources, active_indices, pre):
    # 按2048人构建，避免复制巨大的Unicode分类张量。
    arrays = None
    for start in range(0, len(active_indices), 2048):
        end = min(start+2048, len(active_indices))
        part = make_model_inputs(sources, active_indices[start:end], pre)
        if arrays is None:
            arrays = {k: np.empty((len(active_indices),)+v.shape[1:], dtype=v.dtype)
                      if isinstance(v,np.ndarray) else v for k,v in part.items()}
        for k,v in part.items():
            if isinstance(v,np.ndarray): arrays[k][start:end] = v
    return arrays


def get_pairs(origins, ids):
    pp = np.argwhere(origins[ids]).astype(np.int32)
    pp[:,0] = np.asarray(ids)[pp[:,0]]
    return pp


def ordinary_loader(ds, batch):
    return DataLoader(ds, batch_size=batch, shuffle=False, num_workers=0, pin_memory=False)


def predict_hazard(model, loader, device):
    model.eval(); h=[]
    with torch.no_grad():
        for raw in loader:
            batch = move_batch_to_device(raw, device)
            logits = model(dynamic_sequence=batch['dynamic_sequence'], row_mask=batch['row_mask'],
                static_baseline=batch['static_baseline'], age_at_landmark=batch['age_at_landmark'],
                landmark_normalized=batch['landmark_normalized'], landmark_bin=batch['landmark_bin'])
            h.append(torch.sigmoid(logits.float()).cpu().numpy())
    if not h: raise RuntimeError('没有可预测的有效landmark记录。')
    return np.concatenate(h).astype(np.float32)


def average_snapshot_hazard(model, snapshots, loader, device):
    total = None
    for s in snapshots:
        model.load_state_dict(s['state_dict'])
        h = predict_hazard(model, loader, device)
        total = h.astype(float) if total is None else total+h
    return (total/len(snapshots)).astype(np.float32)


def metric_sort_key(summary):
    # 主要IBS，iAUC和C仅用于完全相同IBS时的排序。
    row = summary.iloc[0]
    return (float(row.ibs), -float(row.iauc) if np.isfinite(row.iauc) else np.inf,
            -float(row.uno_c) if np.isfinite(row.uno_c) else np.inf)


def train_task(meta, inputs, targets, atrisk, origins, eligible, split, parameters,
               device, seed, task_dir, task_id):
    task_dir.mkdir(parents=True, exist_ok=True)
    done = task_dir/'completed.json'; pred_path = task_dir/'predictions_LOCAL_ONLY.npz'
    if RESUME and done.exists() and pred_path.exists():
        if read_json(done).get('task_id') != task_id: raise RuntimeError('已存任务配置不匹配。')
        log('已完成，复用本次新增分析结果：'+str(task_dir.name)); return
    seed_training(seed)
    pairs = {k:get_pairs(origins, split[k]) for k in ['fit','stop','oof','held']}
    datasets = {k:LandmarkDataset(inputs, v, targets, atrisk) for k,v in pairs.items()}
    fd = {'train_dataset':datasets['fit'], 'validation_dataset':datasets['stop'],
          'train_sample_pairs':pairs['fit']}
    train_loader, stop_loader = create_data_loaders(fd, parameters['batch_size'], seed, device)
    oof_loader = ordinary_loader(datasets['oof'], parameters['batch_size'])
    held_loader = ordinary_loader(datasets['held'], parameters['batch_size'])
    refs = make_references(meta, eligible, split['fit'])
    model = build_model(parameters, device)
    loss_fn = LandmarkBalancedSurvivalLoss(**{k:parameters[k] for k in
        ['positive_weight','focal_gamma','auxiliary_5y_weight','ranking_weight','smoothness_weight']}).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=parameters['learning_rate'],
                                  weight_decay=parameters['weight_decay'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5,
                                                         patience=3, min_lr=1e-6)
    scaler = make_grad_scaler(False)
    last = task_dir/'trainer_last.pt'
    top=[]; history=[]; best=np.inf; wait=0; start_epoch=1; stop_reached=False
    elapsed_previous = 0.
    if RESUME and last.exists():
        ck = torch.load(last, map_location='cpu', weights_only=True)
        if ck['task_id'] != task_id: raise RuntimeError('断点配置不同，禁止混用。')
        model.load_state_dict(ck['model']); optimizer.load_state_dict(ck['optimizer'])
        for state in optimizer.state.values():
            for k,v in state.items():
                if torch.is_tensor(v): state[k]=v.to(device)
        scheduler.load_state_dict(ck['scheduler'])
        top=ck['top']; history=ck['history']; best=ck['best']; wait=ck['wait']
        start_epoch=ck['epoch']+1; stop_reached=ck['stop_reached']
        elapsed_previous=ck.get('elapsed_seconds',0.)
        restore_rng(ck['rng'], train_loader)
        log(f'从epoch {start_epoch}续跑；已完成 {len(history)} epochs。')
        del ck
    started=time.perf_counter()
    try:
        if not stop_reached:
            for epoch in range(start_epoch, MAX_EPOCHS+1):
                e0=time.perf_counter()
                train_loss=train_one_epoch(model, train_loader, optimizer, loss_fn, scaler, device, False)
                stop_h=predict_hazard(model, stop_loader, device)
                table,_=evaluate_pairs(meta,pairs['stop'],1-hazards_to_survival(stop_h),refs)
                summ=strict_summary(table); key=metric_sort_key(summ)
                if not np.isfinite(key[0]):
                    csv_save(table,task_dir/'earlystop_metric_failure.csv')
                    raise RuntimeError('早停风险集不能完整评价共同的5年IBS，原因已保存；没有使用留出中心替代。')
                scheduler.step(key[0])
                top.append({'epoch':epoch,'key':list(key),'state_dict':clone_state_dict(model)})
                top.sort(key=lambda x: tuple(x['key'])); top=top[:TOP_SNAPSHOTS]
                if key[0] < best-MIN_DELTA: best=key[0]; wait=0
                else: wait+=1
                stop_reached=bool(epoch>=MIN_EPOCHS and wait>=PATIENCE)
                hist={'epoch':epoch,'train_loss':train_loss,'earlystop_IBS':key[0],
                      'earlystop_iAUC':float(summ.iloc[0].iauc),'earlystop_UnoC':float(summ.iloc[0].uno_c),
                      'learning_rate':optimizer.param_groups[0]['lr'],
                      'epoch_seconds':time.perf_counter()-e0,
                      'elapsed_seconds':elapsed_previous+time.perf_counter()-started}
                history.append(hist)
                log(f"Epoch {epoch:02d}/{MAX_EPOCHS} | loss={train_loss:.6f} | "
                    f"earlystop IBS={key[0]:.6f} iAUC={hist['earlystop_iAUC']:.4f} "
                    f"UnoC={hist['earlystop_UnoC']:.4f} | {hist['epoch_seconds']:.1f}s")
                csv_save(pd.DataFrame(history),task_dir/'training_history.csv')
                save_torch_atomic(last, {'task_id':task_id,'model':clone_state_dict(model),
                    'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),
                    'top':top,'history':history,'best':best,'wait':wait,'epoch':epoch,
                    'stop_reached':stop_reached,'rng':get_rng(train_loader),
                    'elapsed_seconds':hist['elapsed_seconds']})
                if stop_reached: break
        if not top: raise RuntimeError('未生成有效快照。')
        # 仅用独立早停患者选择快照数量；OOF/留出中心不参与。
        choices=[]; accum=None
        for k,s in enumerate(top,1):
            model.load_state_dict(s['state_dict']); h=predict_hazard(model,stop_loader,device)
            accum=h.astype(float) if accum is None else accum+h
            tt,_=evaluate_pairs(meta,pairs['stop'],1-hazards_to_survival(accum/k),refs)
            choices.append((metric_sort_key(strict_summary(tt)),k))
        choices.sort(key=lambda x:(x[0],x[1])); n_snap=choices[0][1]; selected=top[:n_snap]
        oof_h=average_snapshot_hazard(model,selected,oof_loader,device)
        held_h=average_snapshot_hazard(model,selected,held_loader,device)
        save_torch_atomic(task_dir/'selected_snapshots.pt', {'parameters':parameters,
            'snapshots':selected,'training_seed':seed,'task_id':task_id})
        save_npz_atomic(pred_path,oof_pairs=pairs['oof'],oof_hazard=oof_h,
                        held_pairs=pairs['held'],held_hazard=held_h)
        save_json_atomic(done,{'task_id':task_id,'completed':True,'training_seed':seed,
            'epochs':len(history),'selected_snapshot_epochs':[x['epoch'] for x in selected],
            'oof_origins':len(oof_h),'heldout_origins':len(held_h),
            'fit_patients':len(split['fit']),'earlystop_patients':len(split['stop']),
            'prior_imputation_verified':False,'created':datetime.now().isoformat()})
        log(f'训练任务完成：{len(history)} epochs，快照 {[x["epoch"] for x in selected]}。')
    finally:
        del model,optimizer,scheduler,loss_fn,scaler,train_loader,stop_loader,oof_loader,held_loader,datasets
        gc.collect()
        if device.type=='cuda': torch.cuda.empty_cache()


def fit_calibrator(hazard,target,mask):
    # 原Step11: alpha[10] + beta*logit(h); ridge=1e-6, bounds unchanged.
    h=np.clip(np.asarray(hazard,float),EPS,1-EPS); mask=np.asarray(mask,bool)
    x=np.log(h/(1-h))[mask]; y=np.asarray(target,float)[mask]
    j=np.broadcast_to(np.arange(10),h.shape)[mask]
    if len(y)==0 or y.sum()<=0: raise ValueError('训练中心OOF校准没有事件/有效区间。')
    def objective(p):
        a,b=p[:10],p[-1]; eta=a[j]+b*x
        residual=(expit(eta)-y)/len(y)
        value=np.mean(np.logaddexp(0.,eta)-y*eta)+0.5e-6*(np.mean(a*a)+(b-1)**2)
        grad=np.r_[np.bincount(j,weights=residual,minlength=10)+1e-6*a/10,
                   np.dot(residual,x)+1e-6*(b-1)]
        return float(value),grad
    fit=minimize(objective,np.r_[np.zeros(10),1.],jac=True,method='L-BFGS-B',
                 bounds=[(-8.,8.)]*10+[(.05,5.)],
                 options={'maxiter':300,'ftol':1e-12,'gtol':1e-8,'maxls':50})
    if not fit.success: raise RuntimeError('OOF校准失败：'+str(fit.message))
    return {'alpha':fit.x[:10].tolist(),'beta':float(fit.x[-1]),'n_intervals':len(y),
            'events':int(y.sum()),'iterations':int(fit.nit),'objective':float(fit.fun)}


def apply_calibrator(h,fit):
    x=np.clip(np.asarray(h,float),EPS,1-EPS)
    return np.clip(expit(np.array(fit['alpha'])[None,:]+fit['beta']*np.log(x/(1-x))),EPS,1-EPS).astype(np.float32)


def finalise_rotation(meta,origins,eligible,targets,atrisk,pool,held,rotation_dir):
    refs=make_references(meta,eligible,pool)
    expected=get_pairs(origins,pool); held_pairs=get_pairs(origins,held)
    expected_keys=expected[:,0]*6+expected[:,1]
    row_maps={int(k):i for i,k in enumerate(expected_keys)}
    seed_metrics=[]; seed_risks=[]
    for seed_position,seed in enumerate(SEEDS):
        oof=np.full((len(expected),10),np.nan,np.float32)
        assigned=np.zeros(len(expected),bool); outer_h=[]
        for f in range(N_INNER_FOLDS):
            path=rotation_dir/f'fold_{f}'/f'seed_{seed}'/'predictions_LOCAL_ONLY.npz'
            with np.load(path,allow_pickle=False) as pred:
                pp=pred['oof_pairs']; loc=np.array([row_maps[int(k)] for k in pp[:,0]*6+pp[:,1]])
                if assigned[loc].any(): raise ValueError('重复OOF行。')
                oof[loc]=pred['oof_hazard']; assigned[loc]=True
                if not np.array_equal(pred['held_pairs'],held_pairs): raise ValueError('各折留出中心预测行不一致。')
                outer_h.append(pred['held_hazard'].copy())
        if not assigned.all() or not np.isfinite(oof).all(): raise ValueError('OOF预测不完整。')
        calibrators={}
        for l in range(6):
            select=expected[:,1]==l; ip=expected[select,0]
            calibrators[str(l)]=fit_calibrator(oof[select],targets[ip,l],atrisk[ip,l])
        # 保留快照内部hazard均值；跨fold按累计风险等权平均。
        # 每折先应用同一个训练OOF校准器，再转换累计风险，再集成。
        raw_total=np.zeros((len(held_pairs),10)); cal_total=np.zeros_like(raw_total)
        for h in outer_h:
            hc=np.empty_like(h)
            for l in range(6):
                sel=held_pairs[:,1]==l
                hc[sel]=apply_calibrator(h[sel],calibrators[str(l)])
            raw_total+=1-hazards_to_survival(h)
            cal_total+=1-hazards_to_survival(hc)
        raw_risk=(raw_total/N_INNER_FOLDS).astype(np.float32)
        cal_risk=(cal_total/N_INNER_FOLDS).astype(np.float32)
        seed_risks.append(cal_risk)
        sd=rotation_dir/f'seed_{seed}_summary'; sd.mkdir(exist_ok=True)
        save_json_atomic(sd/'training_OOF_calibrators.json',calibrators)
        save_npz_atomic(sd/'heldout_risk_LOCAL_ONLY.npz',pairs=held_pairs,raw_risk=raw_risk,calibrated_risk=cal_risk)
        for variant,rr in [('raw',raw_risk),('calibrated',cal_risk)]:
            tab,hor=evaluate_pairs(meta,held_pairs,rr,refs)
            csv_save(tab,sd/f'{variant}_landmark_metrics.csv')
            csv_save(hor,sd/f'{variant}_1_3_5y_metrics.csv')
            summ=strict_summary(tab); summ['seed']=seed; summ['variant']=variant
            summ['heldout_centre']=str(meta.iloc[held[0]].centre_key)
            seed_metrics.append(summ)
        last=seed_metrics[-1].iloc[0]
        log(f'留出 {last.heldout_centre} seed={seed} | calibrated UnoC={last.uno_c:.4f} '
            f'iAUC={last.iauc:.4f} IBS={last.ibs:.6f}')
    all_seed=pd.concat(seed_metrics,ignore_index=True)
    csv_save(all_seed,rotation_dir/'seed_level_metrics.csv')
    # 另存三种子集成，供患者bootstrap计算CI；不要与seed均值的CI混淆。
    ensemble=np.mean(np.stack(seed_risks),axis=0).astype(np.float32)
    tab,hor=evaluate_pairs(meta,held_pairs,ensemble,refs)
    csv_save(tab,rotation_dir/'three_seed_ensemble_landmark_metrics.csv')
    csv_save(hor,rotation_dir/'three_seed_ensemble_1_3_5y_metrics.csv')
    save_npz_atomic(rotation_dir/'three_seed_ensemble_LOCAL_ONLY.npz',pairs=held_pairs,risk=ensemble)
    csv_save(strict_summary(tab),rotation_dir/'three_seed_ensemble_point_metrics.csv')
    return all_seed,held_pairs,ensemble,refs


def bootstrap_ensemble(meta,held,pairs,risk,refs,out):
    if BOOTSTRAP_REPS <= 0: return
    log(f'开始患者级bootstrap（{BOOTSTRAP_REPS}次；不重新训练）；CI对象为三种子集成。')
    shape=(BOOTSTRAP_REPS,len(SCOPES),len(METRICS))
    dest=out/'bootstrap_checkpoint.npz'
    draws=np.full(shape,np.nan); completed=0
    if RESUME and dest.exists():
        with np.load(dest,allow_pickle=False) as z:
            if z['metrics'].shape!=shape: raise RuntimeError('bootstrap次数改变，请使用新输出目录。')
            draws[:]=z['metrics']; completed=int(z['completed'])
    rng=np.random.default_rng(BOOTSTRAP_SEED)
    # 重复患者整组抽样：一位患者的全部landmark同时进入。
    lookup=np.full((len(meta),6),-1,np.int32)
    lookup[pairs[:,0],pairs[:,1]]=np.arange(len(pairs))
    for b in range(BOOTSTRAP_REPS):
        sampled=rng.choice(held,size=len(held),replace=True)
        if b<completed: continue
        rows=lookup[sampled].reshape(-1); rows=rows[rows>=0]
        try:
            tab,_=evaluate_pairs(meta,pairs[rows],risk[rows],refs)
            s=strict_summary(tab)
            draws[b]=s[METRICS].to_numpy()
        except (ValueError,ArithmeticError): pass
        if (b+1)%25==0 or b+1==BOOTSTRAP_REPS:
            save_npz_atomic(dest,metrics=draws,completed=np.array(b+1))
            log(f'Bootstrap {b+1}/{BOOTSTRAP_REPS}')
    table=[]
    point=read_csv_safe(out/'three_seed_ensemble_point_metrics.csv')
    for i,scope in enumerate(SCOPES):
        for j,name in enumerate(METRICS):
            v=draws[:,i,j]; good=v[np.isfinite(v)]
            enough=len(good)>=math.ceil(.8*BOOTSTRAP_REPS)
            table.append({'scope':scope,'metric':name,'point':float(pd.to_numeric(point[name], errors='coerce').iloc[i]),
                'lower95':float(np.percentile(good,2.5)) if enough else np.nan,
                'upper95':float(np.percentile(good,97.5)) if enough else np.nan,
                'valid_bootstrap':len(good),'requested_bootstrap':BOOTSTRAP_REPS,
                'target':'fixed_three_seed_prediction_ensemble',
                'uncertainty':'heldout_patient_sampling_conditional_on_fitted_models'})
    csv_save(pd.DataFrame(table),out/'three_seed_ensemble_bootstrap95CI.csv')


def write_overall_tables(out):
    paths=sorted(out.glob('rotation_*/seed_level_metrics.csv'))
    if not paths: return
    long=pd.concat([pd.read_csv(p) for p in paths],ignore_index=True)
    csv_save(long,out/'ALL_centres_seed_level_metrics.csv')
    rows=[]
    for keys,group in long.groupby(['heldout_centre','variant','scope'],sort=False):
        row=dict(zip(['heldout_centre','variant','scope'],keys)); row['seed_n']=len(group)
        for k in METRICS:
            a=group[k].to_numpy(float); complete=np.isfinite(a).all() and len(a)==len(SEEDS)
            row[k+'_mean']=float(a.mean()) if complete else np.nan
            row[k+'_SD']=float(a.std(ddof=1)) if complete and len(a)>1 else np.nan
            row[k+'_valid_seeds']=int(np.isfinite(a).sum())
        rows.append(row)
    table=pd.DataFrame(rows); csv_save(table,out/'FINAL_centres_mean_SD.csv')
    primary=table.loc[(table['variant']=='calibrated') & (table['scope']=='six_landmarks')].copy()
    population_path=out/'centre_population.csv'
    if population_path.exists():
        pop=pd.read_csv(population_path).rename(columns={'centre_key':'heldout_centre',
              'n':'validation_patients','events':'CKD_events_total_followup'})
        primary=primary.merge(pop,on='heldout_centre',how='left',validate='one_to_one')
    csv_save(primary,out/'FINAL_crosscenter_results.csv')
    fmt=table.copy()
    for k in METRICS:
        fmt[k+' mean ± SD']=[f'{m:.6f} ± {s:.6f}' if np.isfinite(m+s) else 'not estimable'
                              for m,s in zip(fmt[k+'_mean'],fmt[k+'_SD'])]
    csv_save(fmt[['heldout_centre','variant','scope','seed_n']+[k+' mean ± SD' for k in METRICS]],
             out/'FINAL_centres_mean_SD_formatted.csv')
    ci_paths=sorted(out.glob('rotation_*/three_seed_ensemble_bootstrap95CI.csv'))
    if ci_paths:
        ci=[]
        for p in ci_paths:
            x=pd.read_csv(p); x['rotation']=p.parent.name; ci.append(x)
        csv_save(pd.concat(ci,ignore_index=True),out/'FINAL_seed_ensemble_bootstrap95CI.csv')


def close_sources(sources):
    for s in sources:
        for name in ['continuous','binary','categorical','row_mask']:
            arr=getattr(s,name,None)
            mapping=getattr(arr,'_mmap',None)
            if mapping is not None:
                try: mapping.close()
                except (OSError,ValueError): pass
    gc.collect()


def _run_all(root,out):
    global METRIC_FUNCTIONS
    METRIC_FUNCTIONS=load_metric_functions()
    torch.set_num_threads(CPU_THREADS)
    device=torch.device(DEVICE)
    if device.type=='cuda' and not torch.cuda.is_available(): raise RuntimeError('该环境没有CUDA；DEVICE请保留cpu。')
    log(f'开始完整训练：3 centres × {N_INNER_FOLDS} folds × {len(SEEDS)} seeds；device={device}')
    log('沿用已整理输入；历史插补未重做，结果标记为固定输入的事后敏感性分析。')
    info=runtime_info(root)
    if info['disk_free_GiB']<1: raise RuntimeError('磁盘剩余空间小于1GiB。')
    if info.get('available_memory_GiB',10)<1.5: raise RuntimeError('可用内存低于1.5GiB，无法安全开始训练。')
    log(f"可用内存={info.get('available_memory_GiB',float('nan')):.2f}GiB；磁盘剩余={info['disk_free_GiB']:.1f}GiB")
    p=normalise_parameters(read_json(root/PARAMETER_SOURCE)['selected_parameters'],'final model')
    sources=load_sources(root)
    try:
        all_meta=pd.concat([s.meta for s in sources],ignore_index=True)
        if PRESERVE_LOCKED_TEST:
            keep=(all_meta['source'].eq('CQ') | (all_meta['source'].eq('SN') & all_meta['archived_split'].eq('development'))).to_numpy()
            if not all_meta.loc[all_meta.source.eq('SN'),'archived_split'].isin(['development','test']).all():
                raise ValueError('历史划分标签不是development/test，不能自动确定训练人群。')
        else: keep=np.ones(len(all_meta),bool)
        active=np.where(keep)[0].astype(np.int32)
        meta=all_meta.iloc[active].reset_index(drop=True)
        if meta['centre_key'].nunique()!=3: raise ValueError('训练人群不是三个中心。')
        rowmask=take(sources,'row_mask',active).astype(bool)
        targets,atrisk,eligible,origins=build_labels(meta,rowmask)
        log(f'本次分析患者={len(meta)}；原内部测试排除={int((~keep).sum())}；有效landmark记录={int(origins.sum())}')
        # 配置指纹防止中断后混用不同训练方案，仅访问明确输入，不扫描磁盘。
        input_paths=[]
        for s in sources:
            for name in ['continuous_raw_0_60.npy','binary_raw_0_60.npy','sequence_row_mask.npy']:
                fp=Path(s.source_directory)/name
                input_paths.append({'path':str(fp),'bytes':fp.stat().st_size,'mtime_ns':fp.stat().st_mtime_ns})
        idhash=hashlib.sha256(('\n'.join(meta['source']+'::'+meta['ID_LOCAL_ONLY'])).encode()).hexdigest()
        datahash=hashlib.sha256(meta[['event','time']].to_numpy().tobytes()+rowmask.tobytes()).hexdigest()
        config={'version':VERSION,'parameters':p,'preserve_locked_test':PRESERVE_LOCKED_TEST,
            'seeds':SEEDS,'inner_folds':N_INNER_FOLDS,'split_seed':PARTITION_SEED,
            'earlystop_fraction':EARLYSTOP_FRACTION,'max_epochs':MAX_EPOCHS,'min_epochs':MIN_EPOCHS,
            'patience':PATIENCE,'min_delta':MIN_DELTA,'top_snapshots':TOP_SNAPSHOTS,
            'device':DEVICE,'threads':CPU_THREADS,'bootstrap_reps':BOOTSTRAP_REPS,
            'bootstrap_seed':BOOTSTRAP_SEED,'metric_times':METRIC_TIMES.tolist(),
            'evaluation_origin_rule':EVALUATION_ORIGIN_RULE,'input_files':input_paths,
            'patient_order_hash':idhash,'outcome_mask_hash':datahash,'n_analysis':len(meta),
            'prior_imputation_verified':False,
            'calibration':'per_landmark; alpha_j+beta*logit(h); ridge=1e-6; training_OOF_only',
            'ensemble':'within_snapshot_hazard_mean; calibrate_each_fold; between_fold_risk_mean',
            'earlystop':'separate 10% of inner-training patients; never OOF or heldout centre',
            'limitations':LIMITATIONS}
        signature=hashlib.sha256(json.dumps(config,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
        cfgpath=out/'analysis_configuration.json'
        if cfgpath.exists():
            old=read_json(cfgpath)
            if old.get('signature')!=signature: raise RuntimeError('已有结果使用不同配置或输入；请修改OUTPUT_NAME，禁止覆盖混算。')
            if not RESUME: raise RuntimeError('输出目录已有任务；RESUME=False时请使用新OUTPUT_NAME。')
        else: save_json_atomic(cfgpath,dict(config,signature=signature,environment_start=info))
        csv_save(meta[['source','source_patient_index','centre_key','archived_split']],out/'patient_index_map_LOCAL_ONLY.csv')
        csv_save(meta.groupby('centre_key').agg(n=('event','size'),events=('event','sum')).reset_index(),out/'centre_population.csv')
        riskrows=[]
        for key in meta['centre_key'].unique():
            cm=meta['centre_key'].eq(key).to_numpy()
            for l,month in enumerate(LM):
                riskrows.append({'heldout_centre':key,'landmark_year':month/12.,
                    'eligible_with_history':int((cm & eligible[:,l]).sum()),
                    'valid_origins':int((cm & origins[:,l]).sum()),
                    'short_censored_excluded':int((cm & eligible[:,l] & ~origins[:,l]).sum()),
                    'events_5y':int(targets[cm,l].sum())})
        csv_save(pd.DataFrame(riskrows),out/'landmark_risk_sets.csv')
        (out/'ANALYSIS_LIMITATIONS.txt').write_text('\n'.join(LIMITATIONS),encoding='utf-8')
        centres=sorted(meta['centre_key'].unique(),key=lambda x:(not x.startswith('CQ::'),x))
        seed_schedule=[{'rotation':r+1,'heldout_centre':key,'inner_fold':f,'seed_position':seed,
                        'actual_training_seed':int(seed+r*10000+f*1000)}
                       for r,key in enumerate(centres) for f in range(N_INNER_FOLDS) for seed in SEEDS]
        csv_save(pd.DataFrame(seed_schedule),out/'training_seed_schedule.csv')
        total_tasks=len(centres)*N_INNER_FOLDS*len(SEEDS); count=0
        for r,key in enumerate(centres):
            rdir=out/f'rotation_{r+1}_{key.split("::")[-1]}'; rdir.mkdir(exist_ok=True)
            pool,held,parts=fold_parts(meta,active,key)
            log(f'========== Rotation {r+1}/3：留出{key}，训练中心人群{len(pool)}，评价{len(held)} ==========')
            for f,split in enumerate(parts):
                fdir=rdir/f'fold_{f}'; fdir.mkdir(exist_ok=True)
                save_npz_atomic(fdir/'patient_splits_LOCAL_ONLY.npz',**split)
                # 对每个训练折重新拟合，仅实际拟合患者参与；没有旧预处理器。
                pre=fit_new_preprocessor(sources,active[split['fit']])
                inputs=build_all_inputs(sources,active,pre)
                save_npz_atomic(fdir/'new_scaler.npz',mean=pre['scaler'].mean_,scale=pre['scaler'].scale_)
                save_json_atomic(fdir/'new_encoding.json',{'categories':[x.tolist() for x in pre['encoder'].categories_],
                    'feature_names':pre['names'],'fitted_patients':len(split['fit'])})
                try:
                    for seed in SEEDS:
                        count+=1
                        task_seed=int(seed + r*10000 + f*1000)
                        task_id=hashlib.sha256(f'{signature}|{r}|{f}|{seed}'.encode()).hexdigest()
                        log(f'任务 {count}/{total_tasks}：中心轮转{r+1} fold={f} seed_position={seed} actual_seed={task_seed}')
                        train_task(meta,inputs,targets,atrisk,origins,eligible,split,p,device,task_seed,
                                   fdir/f'seed_{seed}',task_id)
                finally:
                    del inputs,pre; gc.collect()
                    if device.type=='cuda': torch.cuda.empty_cache()
            finalise_rotation(meta,origins,eligible,targets,atrisk,pool,held,rdir)
            write_overall_tables(out)
        log('全部27个训练任务及逐种子中心评价完成；均值/SD汇总已保存。')
        # 所有训练完成后才bootstrap；中断后不重训已完成任务。
        if BOOTSTRAP_REPS>0:
            for r,key in enumerate(centres):
                rdir=out/f'rotation_{r+1}_{key.split("::")[-1]}'
                held=np.where(meta['centre_key'].to_numpy()==key)[0]
                pool=np.where(meta['centre_key'].to_numpy()!=key)[0]
                refs=make_references(meta,eligible,pool)
                with np.load(rdir/'three_seed_ensemble_LOCAL_ONLY.npz',allow_pickle=False) as z:
                    bootstrap_ensemble(meta,held,z['pairs'],z['risk'],refs,rdir)
            write_overall_tables(out)
        save_json_atomic(out/'analysis_completed.json',{'status':'COMPLETED_POSTHOC_FIXED_INPUT_CENTRE_SENSITIVITY',
            'training_tasks':total_tasks,'source_model_weights_loaded':False,
            'prior_imputation_verified':False,'limitations':LIMITATIONS,
            'completed':datetime.now().isoformat(),'environment_end':runtime_info(root)})
        log('完成。主要结果：'+str(out/'FINAL_crosscenter_results.csv'))
        log('请勿将seed间SD当作患者bootstrap置信区间；后者单独保存在seed-ensemble表。')
    finally:
        close_sources(sources)


def run_all(root=None,out=None):
    root=Path(ROOT if root is None else root).expanduser().resolve()
    if not root.is_dir(): raise FileNotFoundError('项目目录不存在：'+str(root))
    output=Path(out) if out is not None else root/OUTPUT_NAME
    output.mkdir(parents=True,exist_ok=True)
    lock=acquire_lock(output)
    stdout0,stderr0=sys.stdout,sys.stderr
    try:
        with (output/'training_live.log').open('a',encoding='utf-8',buffering=1) as f:
            sys.stdout=TrainingTee(stdout0,f); sys.stderr=TrainingTee(stderr0,f)
            try: _run_all(root,output)
            except KeyboardInterrupt:
                log('已中断；上一个完整epoch已保存。重新运行相同单元格可续跑。')
                raise
            except Exception as exc:
                traceback.print_exc()
                save_json_atomic(output/'last_error.json',{'error_type':type(exc).__name__,
                    'message':str(exc),'time':datetime.now().isoformat()})
                raise
            finally: sys.stdout,sys.stderr=stdout0,stderr0
    finally:
        if lock.exists(): lock.unlink()
        gc.collect()




if __name__ == "__main__":
    run_all()
