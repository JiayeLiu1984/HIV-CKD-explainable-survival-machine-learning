
from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import random
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import optuna
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

try:
    import sksurv
    from sksurv.metrics import (
        concordance_index_censored,
        integrated_brier_score,
    )
except ImportError as exc:
    raise ImportError(
        "Step 9B需要scikit-survival。请在当前环境安装：\n"
        "conda install -c conda-forge scikit-survival optuna\n"
        "或：pip install scikit-survival optuna"
    ) from exc


# ============================================================
# 1. 运行阶段
# ============================================================

# 可选："tune"或"final"
RUN_STAGE = "tune"

# 调参以完成Trial数为目标；以后增大目标值会继续补足。
TUNING_COMPLETE_TRIAL_TARGET = 12
TUNING_MAX_TOTAL_ATTEMPT_N = 30
SEARCH_SPACE_VERSION = "rnn_moderate_v1_20260730"

# 每个Trial和最终OOF共用的训练规则。
MAX_EPOCHS = 40
EARLY_STOPPING_PATIENCE = 6
EARLY_STOPPING_MIN_DELTA = 1e-5
GRADIENT_CLIP_NORM = 5.0
USE_AMP = True
REQUIRE_CUDA = True
NUM_WORKERS = 0
LOG_EVERY_N_BATCHES = 100

# 最终OOF使用最佳参数；每折单独保存检查点。
N_SPLITS = 5
BASE_RANDOM_SEED = 20260730


# ============================================================
# 2. 路径和固定数据参数
# ============================================================

PROJECT_DIR = Path(
    os.getenv("CKD_LSTM_PROJECT_DIR", "__CKD_WORKDIR__")
)

STEP1_DIR = PROJECT_DIR / "rolling_5y_step1_new_split"
STEP2_DIR = PROJECT_DIR / "rolling_5y_step2_folds"
STEP3_DIR = PROJECT_DIR / "rolling_5y_step3_raw_features"
STEP4_DIR = PROJECT_DIR / "rolling_5y_step4_preprocessed"
STEP6_DIR = PROJECT_DIR / "rolling_5y_step6_super_landmark_data"

OUTPUT_DIR = (
    PROJECT_DIR
    / "rolling_5y_step9b_rnn_tune_resume_v1"
)
TUNING_DIR = OUTPUT_DIR / "tuning"
FINAL_DIR = OUTPUT_DIR / "final_oof"

for directory in [OUTPUT_DIR, TUNING_DIR, FINAL_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

LIVE_PROGRESS_FILE = OUTPUT_DIR / "step9b_live_progress.log"
PROGRESS_TABLE_FILE = OUTPUT_DIR / "step9b_progress_table.csv"
OPTUNA_DB_FILE = TUNING_DIR / "rnn_optuna.sqlite3"
OPTUNA_STORAGE_URL = f"sqlite:///{OPTUNA_DB_FILE}"
OPTUNA_STUDY_NAME = "unified_simple_rnn_equal_weight_ibs_v1"

EXPECTED_TOTAL_N = 31911
EXPECTED_DEVELOPMENT_N = 22337
EXPECTED_HISTORY_STEP_N = 11
EXPECTED_FEATURE_N = 57
EXPECTED_STATIC_N = 16
EXPECTED_DYNAMIC_N = 40
EXPECTED_LANDMARK_N = 6
EXPECTED_FUTURE_INTERVAL_N = 10
EXPECTED_DEVELOPMENT_VALID_ORIGIN_N = 100122

LANDMARK_MONTHS = np.asarray(
    [0, 12, 24, 36, 48, 60],
    dtype=np.int32,
)
LANDMARK_BINS = (
    LANDMARK_MONTHS // 6
).astype(np.int64)
FUTURE_END_MONTHS = np.arange(
    6,
    61,
    6,
    dtype=np.float64,
)
IBS_EVALUATION_TIMES = np.asarray(
    [6, 12, 18, 24, 30, 36, 42, 48, 54, 59.999],
    dtype=np.float64,
)
TIME_STEP_YEARS = (
    np.arange(
        EXPECTED_HISTORY_STEP_N,
        dtype=np.float32,
    )
    * 0.5
)
EPS = 1e-7


# ============================================================
# 3. 固定变量分组
# ============================================================

LAB_FEATURES = [
    "HIVRNA_log10",
    "CD4",
    "CD8",
    "Urea",
    "WBC",
    "PLT",
    "HB",
    "TC",
    "TG",
    "HDL",
    "LDL",
    "GLU",
    "ALT",
    "AST",
    "eGFR",
]

PERSISTENT_STATUS_FEATURES = [
    "CVD_status",
    "diabetes_status",
    "hypertension_status",
    "hypercholesterolemia_status",
    "HBV_status",
    "HCV_status",
]

METABOLIC_MED_FEATURES = [
    "antidiabetic_med",
    "antihypertensive_med",
    "antilipid_med",
]

CURRENT_ART_FEATURES = [
    "current_TDF_NNRTI_3TC_FTC",
    "current_TDF_PI_3TC_FTC",
    "current_nonTDF_PI",
    "current_BIC_FTC_TAF",
    "current_EVGc_FTC_TAF",
    "current_TDF_INSTI_3TC_FTC",
    "current_nonTDF_DTG",
    "current_nonTDF_traditional_NNRTI",
]

CUMULATIVE_ART_FEATURES = [
    "TDF_NNRTI_3TC_FTC_cum_month",
    "TDF_PI_3TC_FTC_cum_month",
    "nonTDF_PI_cum_month",
    "BIC_FTC_TAF_cum_month",
    "EVGc_FTC_TAF_cum_month",
    "TDF_INSTI_3TC_FTC_cum_month",
    "nonTDF_DTG_cum_month",
    "nonTDF_traditional_NNRTI_cum_month",
]

DYNAMIC_FEATURES = (
    LAB_FEATURES
    + PERSISTENT_STATUS_FEATURES
    + METABOLIC_MED_FEATURES
    + CURRENT_ART_FEATURES
    + CUMULATIVE_ART_FEATURES
)


# ============================================================
# 4. 通用辅助函数
# ============================================================

def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def format_duration(seconds: float) -> str:
    seconds = max(0, int(round(float(seconds))))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    if hours > 0:
        return f"{hours}小时{minutes:02d}分{seconds:02d}秒"
    if minutes > 0:
        return f"{minutes}分{seconds:02d}秒"
    return f"{seconds}秒"


def progress_print(message: str) -> None:
    line = (
        f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
        f"{message}"
    )
    print(line, flush=True)

    with open(
        LIVE_PROGRESS_FILE,
        "a",
        encoding="utf-8",
    ) as file:
        file.write(line + "\n")
        file.flush()


def append_progress_row(
    phase: str,
    status: str,
    trial_number: int | None = None,
    fold_id: int | None = None,
    metric_name: str | None = None,
    metric_value: float | None = None,
    elapsed_seconds: float | None = None,
    message: str = "",
) -> None:
    row = pd.DataFrame(
        [
            {
                "timestamp": datetime.now().isoformat(
                    timespec="seconds"
                ),
                "phase": phase,
                "status": status,
                "trial_number": trial_number,
                "fold_id": fold_id,
                "metric_name": metric_name,
                "metric_value": metric_value,
                "elapsed_seconds": elapsed_seconds,
                "message": message,
            }
        ]
    )
    row.to_csv(
        PROGRESS_TABLE_FILE,
        mode="a",
        header=not PROGRESS_TABLE_FILE.exists(),
        index=False,
        encoding="utf-8-sig",
    )


def build_survival_array(
    event: np.ndarray,
    time_month: np.ndarray,
) -> np.ndarray:
    event = np.asarray(event, dtype=bool)
    time_month = np.asarray(time_month, dtype=np.float64)

    if event.shape != time_month.shape:
        raise ValueError("事件与生存时间数组形状不一致。")
    if (
        not np.isfinite(time_month).all()
        or np.any(time_month <= 0.0)
    ):
        raise ValueError("生存时间必须为有限正数。")

    result = np.empty(
        len(event),
        dtype=[("event", "?"), ("time", "<f8")],
    )
    result["event"] = event
    result["time"] = time_month
    return result


def parameter_signature(
    parameters: dict[str, Any],
) -> str:
    payload = {
        "parameters": parameters,
        "model_family": "MaskedSimpleRNNSurvival",
        "search_space_version": SEARCH_SPACE_VERSION,
        "max_epochs": MAX_EPOCHS,
        "patience": EARLY_STOPPING_PATIENCE,
        "feature_n": EXPECTED_FEATURE_N,
        "landmark_n": EXPECTED_LANDMARK_N,
        "future_interval_n": EXPECTED_FUTURE_INTERVAL_N,
    }
    text = json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()


def autocast_context(
    device: torch.device,
    enabled: bool,
):
    try:
        return torch.amp.autocast(
            device_type=device.type,
            enabled=enabled,
        )
    except (AttributeError, TypeError):
        return torch.cuda.amp.autocast(
            enabled=enabled,
        )


def make_grad_scaler(enabled: bool):
    try:
        return torch.amp.GradScaler(
            "cuda",
            enabled=enabled,
        )
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(
            enabled=enabled,
        )


# ============================================================
# 5. 核查共同输入
# ============================================================

required_common_files = [
    STEP1_DIR / "development_idx.npy",
    STEP1_DIR / "prediction_origin_mask.npy",
    STEP1_DIR / "future_event_matrix.npy",
    STEP1_DIR / "future_at_risk_mask.npy",
    STEP1_DIR / "landmark_months.npy",
    STEP1_DIR / "landmark_bins.npy",
    STEP2_DIR / "development_fold_id.npy",
    STEP3_DIR / "continuous_raw_0_60.npy",
    STEP3_DIR / "feature_groups.json",
    STEP4_DIR / "development_idx.npy",
    STEP4_DIR / "development_fold_id.npy",
    STEP4_DIR / "sequence_row_mask_development.npy",
    STEP6_DIR / "development_long_row_index_map.npy",
    STEP6_DIR / "development_fold_id_long.npy",
    STEP6_DIR / "development_landmark_month_long.npy",
    STEP6_DIR / "development_analysis_time_month.npy",
    STEP6_DIR / "development_event_within_60m.npy",
]

for fold_id in range(N_SPLITS):
    fold_dir = STEP4_DIR / f"fold_{fold_id}"
    required_common_files.extend(
        [
            fold_dir / "X_development.npy",
            fold_dir / "preprocessor.joblib",
            fold_dir / "feature_names.csv",
            fold_dir / "train_global_idx.npy",
            fold_dir / "validation_global_idx.npy",
        ]
    )

missing_common_files = [
    str(path)
    for path in required_common_files
    if not path.exists()
]
if missing_common_files:
    raise FileNotFoundError(
        "以下Step 9B输入文件不存在：\n"
        + "\n".join(missing_common_files)
    )

development_idx_step1 = np.load(
    STEP1_DIR / "development_idx.npy"
).astype(np.int32)
development_idx = np.load(
    STEP4_DIR / "development_idx.npy"
).astype(np.int32)

development_fold_id_step2 = np.load(
    STEP2_DIR / "development_fold_id.npy"
).astype(np.int8)
development_fold_id = np.load(
    STEP4_DIR / "development_fold_id.npy"
).astype(np.int8)

sequence_row_mask = np.load(
    STEP4_DIR / "sequence_row_mask_development.npy"
).astype(bool)

prediction_origin_mask_all = np.load(
    STEP1_DIR / "prediction_origin_mask.npy"
).astype(bool)
future_event_matrix_all = np.load(
    STEP1_DIR / "future_event_matrix.npy"
).astype(np.float32)
future_at_risk_mask_all = np.load(
    STEP1_DIR / "future_at_risk_mask.npy"
).astype(bool)

landmark_months_file = np.load(
    STEP1_DIR / "landmark_months.npy"
).astype(np.int32)
landmark_bins_file = np.load(
    STEP1_DIR / "landmark_bins.npy"
).astype(np.int64)

continuous_raw = np.load(
    STEP3_DIR / "continuous_raw_0_60.npy",
    mmap_mode="r",
)
with open(
    STEP3_DIR / "feature_groups.json",
    "r",
    encoding="utf-8",
) as file:
    feature_groups = json.load(file)
continuous_vars = list(
    feature_groups["continuous_vars"]
)

long_row_index_map = np.load(
    STEP6_DIR / "development_long_row_index_map.npy"
).astype(np.int32)
fold_id_long = np.load(
    STEP6_DIR / "development_fold_id_long.npy"
).astype(np.int8)
landmark_month_long = np.load(
    STEP6_DIR / "development_landmark_month_long.npy"
).astype(np.int32)
analysis_time_month = np.load(
    STEP6_DIR / "development_analysis_time_month.npy"
).astype(np.float64)
event_within_60m = np.load(
    STEP6_DIR / "development_event_within_60m.npy"
).astype(bool)

if not np.array_equal(
    development_idx_step1,
    development_idx,
):
    raise ValueError(
        "Step 1和Step 4的development_idx顺序不一致。"
    )

if not np.array_equal(
    development_fold_id_step2,
    development_fold_id,
):
    raise ValueError(
        "Step 2和Step 4的development_fold_id不一致。"
    )

if development_idx.shape != (
    EXPECTED_DEVELOPMENT_N,
):
    raise ValueError(
        f"开发集患者索引形状错误：{development_idx.shape}"
    )

if development_fold_id.shape != (
    EXPECTED_DEVELOPMENT_N,
):
    raise ValueError(
        "开发集五折数组形状错误。"
    )

if sequence_row_mask.shape != (
    EXPECTED_DEVELOPMENT_N,
    EXPECTED_HISTORY_STEP_N,
):
    raise ValueError(
        "开发集时间行掩码形状错误。"
    )

if prediction_origin_mask_all.shape != (
    EXPECTED_TOTAL_N,
    EXPECTED_LANDMARK_N,
):
    raise ValueError(
        "prediction_origin_mask形状错误。"
    )

if future_event_matrix_all.shape != (
    EXPECTED_TOTAL_N,
    EXPECTED_LANDMARK_N,
    EXPECTED_FUTURE_INTERVAL_N,
):
    raise ValueError(
        "future_event_matrix形状错误。"
    )

if future_at_risk_mask_all.shape != (
    EXPECTED_TOTAL_N,
    EXPECTED_LANDMARK_N,
    EXPECTED_FUTURE_INTERVAL_N,
):
    raise ValueError(
        "future_at_risk_mask形状错误。"
    )

if not np.array_equal(
    landmark_months_file,
    LANDMARK_MONTHS,
):
    raise ValueError(
        "Step 1的Landmark月份与Step 9B不一致。"
    )

if not np.array_equal(
    landmark_bins_file,
    LANDMARK_BINS,
):
    raise ValueError(
        "Step 1的Landmark行号与Step 9B不一致。"
    )

if long_row_index_map.shape != (
    EXPECTED_DEVELOPMENT_N,
    EXPECTED_LANDMARK_N,
):
    raise ValueError(
        "Step 6长记录映射形状错误。"
    )

if int(
    np.sum(long_row_index_map >= 0)
) != EXPECTED_DEVELOPMENT_VALID_ORIGIN_N:
    raise ValueError(
        "Step 6有效长记录数不是100122。"
    )

if len(analysis_time_month) != (
    EXPECTED_DEVELOPMENT_VALID_ORIGIN_N
):
    raise ValueError(
        "Step 6分析时间记录数错误。"
    )

if (
    np.any(analysis_time_month <= 0)
    or np.any(analysis_time_month > 60.0001)
):
    raise ValueError(
        "Step 6分析时间不在(0, 60]个月。"
    )

prediction_origin_mask = (
    prediction_origin_mask_all[development_idx]
)
future_event_matrix = (
    future_event_matrix_all[development_idx]
)
future_at_risk_mask = (
    future_at_risk_mask_all[development_idx]
)

if int(
    prediction_origin_mask.sum()
) != EXPECTED_DEVELOPMENT_VALID_ORIGIN_N:
    raise ValueError(
        "开发集有效预测起点数不是100122。"
    )

if not np.array_equal(
    prediction_origin_mask,
    long_row_index_map >= 0,
):
    raise ValueError(
        "Step 1有效预测起点与Step 6长记录映射不一致。"
    )

if np.any(
    future_event_matrix.astype(bool)
    & ~future_at_risk_mask
):
    raise ValueError(
        "存在事件标签为1但风险掩码为0的未来区间。"
    )

if np.any(
    future_event_matrix.sum(axis=2) > 1
):
    raise ValueError(
        "部分患者-Landmark存在多个事件区间。"
    )

if "Age" not in continuous_vars:
    raise ValueError(
        "Step 3连续变量列表中缺少Age。"
    )

age_continuous_index = continuous_vars.index("Age")
y_long_all = build_survival_array(
    event_within_60m,
    analysis_time_month,
)

torch.set_num_threads(
    min(8, os.cpu_count() or 1)
)
set_random_seed(BASE_RANDOM_SEED)

cuda_available = torch.cuda.is_available()
if REQUIRE_CUDA and not cuda_available:
    raise RuntimeError(
        "未检测到CUDA。Step 9B要求GPU运行。"
    )

device = torch.device(
    "cuda" if cuda_available else "cpu"
)
use_amp = bool(
    USE_AMP and device.type == "cuda"
)

if device.type == "cuda":
    torch.cuda.empty_cache()


# ============================================================
# 6. 数据集和模型
# ============================================================

class PatientLandmarkDataset(Dataset):
    def __init__(
        self,
        sample_pairs: np.ndarray,
        x_development: np.ndarray,
        sequence_mask: np.ndarray,
        static_baseline_array: np.ndarray,
        baseline_age_array: np.ndarray,
        event_matrix: np.ndarray,
        at_risk_mask: np.ndarray,
        dynamic_indices: np.ndarray,
        age_mean: float,
        age_scale: float,
    ):
        self.sample_pairs = np.asarray(
            sample_pairs,
            dtype=np.int32,
        )
        self.x_development = x_development
        self.sequence_mask = sequence_mask
        self.static_baseline_array = static_baseline_array
        self.baseline_age_array = baseline_age_array
        self.event_matrix = event_matrix
        self.at_risk_mask = at_risk_mask
        self.dynamic_indices = np.asarray(
            dynamic_indices,
            dtype=np.int64,
        )
        self.age_mean = float(age_mean)
        self.age_scale = float(age_scale)

    def __len__(self) -> int:
        return len(self.sample_pairs)

    def __getitem__(
        self,
        index: int,
    ) -> dict[str, np.ndarray]:
        patient_local, landmark_index = (
            self.sample_pairs[index]
        )

        landmark_bin = int(
            LANDMARK_BINS[landmark_index]
        )
        landmark_month = float(
            LANDMARK_MONTHS[landmark_index]
        )

        dynamic_sequence = np.asarray(
            self.x_development[patient_local]
        )[:, self.dynamic_indices].astype(
            np.float32,
            copy=False,
        )

        row_mask = np.asarray(
            self.sequence_mask[patient_local],
            dtype=np.bool_,
        )

        age_at_landmark_raw = (
            float(
                self.baseline_age_array[
                    patient_local
                ]
            )
            + landmark_month / 12.0
        )

        age_at_landmark_standardized = (
            age_at_landmark_raw
            - self.age_mean
        ) / self.age_scale

        return {
            "dynamic_sequence": dynamic_sequence,
            "row_mask": row_mask,
            "static_baseline": np.asarray(
                self.static_baseline_array[
                    patient_local
                ],
                dtype=np.float32,
            ),
            "age_at_landmark": np.float32(
                age_at_landmark_standardized
            ),
            "landmark_normalized": np.float32(
                landmark_month / 60.0
            ),
            "landmark_bin": np.int64(
                landmark_bin
            ),
            "event_target": np.asarray(
                self.event_matrix[
                    patient_local,
                    landmark_index,
                ],
                dtype=np.float32,
            ),
            "at_risk_mask": np.asarray(
                self.at_risk_mask[
                    patient_local,
                    landmark_index,
                ],
                dtype=np.float32,
            ),
            "patient_local": np.int64(
                patient_local
            ),
            "landmark_index": np.int64(
                landmark_index
            ),
        }


class MaskedSimpleRNNSurvival(nn.Module):
    def __init__(
        self,
        dynamic_n: int,
        static_n: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        output_interval_n: int,
    ):
        super().__init__()

        self.dynamic_n = int(dynamic_n)
        self.static_n = int(static_n)
        self.hidden_size = int(hidden_size)
        self.num_layers = int(num_layers)

        cells = []
        for layer_index in range(self.num_layers):
            input_size = (
                self.dynamic_n + 1
                if layer_index == 0
                else self.hidden_size
            )
            cells.append(
                nn.RNNCell(
                    input_size=input_size,
                    hidden_size=self.hidden_size,
                    nonlinearity="tanh",
                )
            )

        self.rnn_cells = nn.ModuleList(cells)
        self.inter_layer_dropout = nn.Dropout(
            float(dropout)
        )

        head_input_n = (
            self.hidden_size
            + self.static_n
            + 1
            + 1
        )

        self.head = nn.Sequential(
            nn.LayerNorm(head_input_n),
            nn.Linear(
                head_input_n,
                self.hidden_size,
            ),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(
                self.hidden_size,
                output_interval_n,
            ),
        )

    def forward(
        self,
        dynamic_sequence: torch.Tensor,
        row_mask: torch.Tensor,
        static_baseline: torch.Tensor,
        age_at_landmark: torch.Tensor,
        landmark_normalized: torch.Tensor,
        landmark_bin: torch.Tensor,
    ) -> torch.Tensor:
        batch_n, time_n, dynamic_n = (
            dynamic_sequence.shape
        )

        if dynamic_n != self.dynamic_n:
            raise ValueError(
                "模型收到的动态特征数不正确。"
            )

        hidden_states = [
            torch.zeros(
                batch_n,
                self.hidden_size,
                dtype=dynamic_sequence.dtype,
                device=dynamic_sequence.device,
            )
            for _ in range(self.num_layers)
        ]

        step_indices = torch.arange(
            time_n,
            device=dynamic_sequence.device,
        )
        time_normalized = (
            step_indices.float()
            / float(max(time_n - 1, 1))
        )

        for step_index in range(time_n):
            active_mask = (
                row_mask[:, step_index]
                & (
                    step_indices[step_index]
                    <= landmark_bin
                )
            ).unsqueeze(1)

            current_dynamic = (
                dynamic_sequence[
                    :,
                    step_index,
                    :,
                ]
            )
            current_time = (
                time_normalized[step_index]
                .expand(batch_n, 1)
            )
            layer_input = torch.cat(
                [
                    current_dynamic,
                    current_time,
                ],
                dim=1,
            )

            for layer_index, cell in enumerate(
                self.rnn_cells
            ):
                candidate_hidden = cell(
                    layer_input,
                    hidden_states[layer_index],
                )
                updated_hidden = torch.where(
                    active_mask,
                    candidate_hidden,
                    hidden_states[layer_index],
                )
                hidden_states[layer_index] = (
                    updated_hidden
                )

                layer_input = updated_hidden
                if layer_index < self.num_layers - 1:
                    layer_input = (
                        self.inter_layer_dropout(
                            layer_input
                        )
                    )

        final_hidden = hidden_states[-1]
        model_context = torch.cat(
            [
                final_hidden,
                static_baseline,
                age_at_landmark.unsqueeze(1),
                landmark_normalized.unsqueeze(1),
            ],
            dim=1,
        )
        return self.head(model_context)


class MaskedDiscreteTimeNLL(nn.Module):
    def __init__(
        self,
        positive_weight: float,
    ):
        super().__init__()
        self.register_buffer(
            "positive_weight",
            torch.tensor(
                float(positive_weight),
                dtype=torch.float32,
            ),
        )

    def forward(
        self,
        logits: torch.Tensor,
        event_target: torch.Tensor,
        at_risk_mask: torch.Tensor,
    ) -> torch.Tensor:
        interval_loss = (
            torch.nn.functional.binary_cross_entropy_with_logits(
                logits,
                event_target,
                reduction="none",
                pos_weight=self.positive_weight,
            )
        )
        masked_loss = (
            interval_loss * at_risk_mask
        )
        denominator = (
            at_risk_mask.sum().clamp_min(1.0)
        )
        return (
            masked_loss.sum() / denominator
        )


# ============================================================
# 7. 每折数据准备
# ============================================================

def prepare_fold_data(
    fold_id: int,
) -> dict[str, Any]:
    fold_dir = STEP4_DIR / f"fold_{fold_id}"

    x_development = np.load(
        fold_dir / "X_development.npy",
        mmap_mode="r",
    )
    feature_names_df = pd.read_csv(
        fold_dir / "feature_names.csv",
        encoding="utf-8-sig",
    )
    feature_names = (
        feature_names_df["feature_name"]
        .astype(str)
        .tolist()
    )
    preprocessor = joblib.load(
        fold_dir / "preprocessor.joblib"
    )

    if x_development.shape != (
        EXPECTED_DEVELOPMENT_N,
        EXPECTED_HISTORY_STEP_N,
        EXPECTED_FEATURE_N,
    ):
        raise ValueError(
            f"第{fold_id}折X_development形状错误："
            f"{x_development.shape}"
        )

    if len(feature_names) != EXPECTED_FEATURE_N:
        raise ValueError(
            f"第{fold_id}折特征数不是57。"
        )

    if not np.isfinite(
        np.asarray(x_development)
    ).all():
        raise ValueError(
            f"第{fold_id}折预处理张量存在NaN或无穷值。"
        )

    if not np.all(
        np.asarray(x_development)[
            ~sequence_row_mask
        ] == 0.0
    ):
        raise ValueError(
            f"第{fold_id}折空缺时间行不是全0。"
        )

    feature_to_index = {
        feature_name: feature_index
        for feature_index, feature_name in enumerate(
            feature_names
        )
    }

    required_names = (
        {"Age", "BMI", "Oppinfection"}
        | set(DYNAMIC_FEATURES)
    )
    missing_names = sorted(
        required_names - set(feature_names)
    )
    if missing_names:
        raise ValueError(
            f"第{fold_id}折缺少特征："
            f"{missing_names}"
        )

    onehot_static_features = [
        feature_name
        for feature_name in feature_names
        if (
            feature_name.startswith("Sex_")
            or feature_name.startswith("Marriage_")
            or feature_name.startswith("Course_")
            or feature_name.startswith("WHOstage_")
        )
    ]
    static_features = [
        "BMI",
        "Oppinfection",
        *onehot_static_features,
    ]

    if len(static_features) != EXPECTED_STATIC_N:
        raise ValueError(
            f"第{fold_id}折静态特征数为"
            f"{len(static_features)}，应为16。"
        )

    if len(DYNAMIC_FEATURES) != (
        EXPECTED_DYNAMIC_N
    ):
        raise ValueError(
            "动态特征固定列表不是40项。"
        )

    assigned_names = (
        {"Age"}
        | set(static_features)
        | set(DYNAMIC_FEATURES)
    )
    if assigned_names != set(feature_names):
        missing_assignment = sorted(
            set(feature_names) - assigned_names
        )
        repeated_n = (
            1
            + len(static_features)
            + len(DYNAMIC_FEATURES)
            - len(assigned_names)
        )
        raise ValueError(
            f"第{fold_id}折特征分组不完整。"
            f"未分配={missing_assignment}，"
            f"重复分配数={repeated_n}。"
        )

    static_feature_indices = np.asarray(
        [
            feature_to_index[name]
            for name in static_features
        ],
        dtype=np.int64,
    )
    dynamic_feature_indices = np.asarray(
        [
            feature_to_index[name]
            for name in DYNAMIC_FEATURES
        ],
        dtype=np.int64,
    )

    first_observed_step = np.argmax(
        sequence_row_mask,
        axis=1,
    ).astype(np.int64)

    if (
        ~sequence_row_mask.any(axis=1)
    ).any():
        raise ValueError(
            "部分开发集患者没有任何真实历史时间行。"
        )

    patient_local_index = np.arange(
        EXPECTED_DEVELOPMENT_N,
        dtype=np.int64,
    )
    static_baseline = np.asarray(
        x_development[
            patient_local_index,
            first_observed_step,
            :,
        ][:, static_feature_indices],
        dtype=np.float32,
    )

    age_at_first_observed = np.asarray(
        continuous_raw[
            development_idx,
            first_observed_step,
            age_continuous_index,
        ],
        dtype=np.float32,
    )
    baseline_age_raw = (
        age_at_first_observed
        - TIME_STEP_YEARS[
            first_observed_step
        ]
    ).astype(np.float32)

    age_scaler_mean = float(
        preprocessor["scaler"].mean_[
            age_continuous_index
        ]
    )
    age_scaler_scale = float(
        preprocessor["scaler"].scale_[
            age_continuous_index
        ]
    )

    if (
        not np.isfinite(age_scaler_mean)
        or not np.isfinite(age_scaler_scale)
        or age_scaler_scale <= 0.0
    ):
        raise ValueError(
            f"第{fold_id}折Age标准化参数无效。"
        )

    train_patient_mask = (
        development_fold_id != fold_id
    )
    validation_patient_mask = (
        development_fold_id == fold_id
    )

    train_sample_pairs = np.argwhere(
        prediction_origin_mask
        & train_patient_mask[:, None]
    ).astype(np.int32)
    validation_sample_pairs = np.argwhere(
        prediction_origin_mask
        & validation_patient_mask[:, None]
    ).astype(np.int32)

    train_long_idx = long_row_index_map[
        train_sample_pairs[:, 0],
        train_sample_pairs[:, 1],
    ]
    validation_long_idx = long_row_index_map[
        validation_sample_pairs[:, 0],
        validation_sample_pairs[:, 1],
    ]

    if np.any(train_long_idx < 0):
        raise ValueError(
            f"第{fold_id}折训练样本存在无效长行号。"
        )
    if np.any(validation_long_idx < 0):
        raise ValueError(
            f"第{fold_id}折验证样本存在无效长行号。"
        )

    if not np.all(
        fold_id_long[validation_long_idx]
        == fold_id
    ):
        raise ValueError(
            f"第{fold_id}折验证长记录折编号错误。"
        )

    if np.intersect1d(
        train_long_idx,
        validation_long_idx,
    ).size:
        raise ValueError(
            f"第{fold_id}折训练和验证长记录重叠。"
        )

    train_dataset = PatientLandmarkDataset(
        sample_pairs=train_sample_pairs,
        x_development=x_development,
        sequence_mask=sequence_row_mask,
        static_baseline_array=static_baseline,
        baseline_age_array=baseline_age_raw,
        event_matrix=future_event_matrix,
        at_risk_mask=future_at_risk_mask,
        dynamic_indices=dynamic_feature_indices,
        age_mean=age_scaler_mean,
        age_scale=age_scaler_scale,
    )
    validation_dataset = PatientLandmarkDataset(
        sample_pairs=validation_sample_pairs,
        x_development=x_development,
        sequence_mask=sequence_row_mask,
        static_baseline_array=static_baseline,
        baseline_age_array=baseline_age_raw,
        event_matrix=future_event_matrix,
        at_risk_mask=future_at_risk_mask,
        dynamic_indices=dynamic_feature_indices,
        age_mean=age_scaler_mean,
        age_scale=age_scaler_scale,
    )

    return {
        "fold_id": int(fold_id),
        "x_development": x_development,
        "train_sample_pairs": train_sample_pairs,
        "validation_sample_pairs":
            validation_sample_pairs,
        "train_long_idx":
            train_long_idx.astype(np.int32),
        "validation_long_idx":
            validation_long_idx.astype(np.int32),
        "train_dataset": train_dataset,
        "validation_dataset": validation_dataset,
        "static_features": static_features,
        "dynamic_features": DYNAMIC_FEATURES,
    }


# ============================================================
# 8. Batch、训练和预测函数
# ============================================================

def move_batch_to_device(
    batch: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    return {
        "dynamic_sequence":
            batch["dynamic_sequence"].to(
                device=device,
                dtype=torch.float32,
                non_blocking=True,
            ),
        "row_mask":
            batch["row_mask"].to(
                device=device,
                dtype=torch.bool,
                non_blocking=True,
            ),
        "static_baseline":
            batch["static_baseline"].to(
                device=device,
                dtype=torch.float32,
                non_blocking=True,
            ),
        "age_at_landmark":
            batch["age_at_landmark"].to(
                device=device,
                dtype=torch.float32,
                non_blocking=True,
            ),
        "landmark_normalized":
            batch["landmark_normalized"].to(
                device=device,
                dtype=torch.float32,
                non_blocking=True,
            ),
        "landmark_bin":
            batch["landmark_bin"].to(
                device=device,
                dtype=torch.long,
                non_blocking=True,
            ),
        "event_target":
            batch["event_target"].to(
                device=device,
                dtype=torch.float32,
                non_blocking=True,
            ),
        "at_risk_mask":
            batch["at_risk_mask"].to(
                device=device,
                dtype=torch.float32,
                non_blocking=True,
            ),
        "patient_local":
            batch["patient_local"],
        "landmark_index":
            batch["landmark_index"],
    }


def create_data_loaders(
    fold_data: dict[str, Any],
    batch_size: int,
    seed: int,
) -> tuple[DataLoader, DataLoader]:
    generator = torch.Generator()
    generator.manual_seed(int(seed))

    train_loader = DataLoader(
        fold_data["train_dataset"],
        batch_size=int(batch_size),
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
        generator=generator,
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


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_function: nn.Module,
    grad_scaler: Any,
    epoch_index: int,
    max_epochs: int,
    log_batches: bool,
) -> float:
    model.train()
    total_weighted_loss = 0.0
    total_valid_intervals = 0.0
    start_time = time.time()

    for batch_index, raw_batch in enumerate(
        loader,
        start=1,
    ):
        batch = move_batch_to_device(raw_batch)
        optimizer.zero_grad(set_to_none=True)

        with autocast_context(
            device,
            use_amp,
        ):
            logits = model(
                dynamic_sequence=(
                    batch["dynamic_sequence"]
                ),
                row_mask=batch["row_mask"],
                static_baseline=(
                    batch["static_baseline"]
                ),
                age_at_landmark=(
                    batch["age_at_landmark"]
                ),
                landmark_normalized=(
                    batch["landmark_normalized"]
                ),
                landmark_bin=(
                    batch["landmark_bin"]
                ),
            )
            loss = loss_function(
                logits,
                batch["event_target"],
                batch["at_risk_mask"],
            )

        if not torch.isfinite(loss):
            raise FloatingPointError(
                "训练损失不是有限数。"
            )

        grad_scaler.scale(loss).backward()
        grad_scaler.unscale_(optimizer)

        gradient_norm = (
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=GRADIENT_CLIP_NORM,
            )
        )
        if not torch.isfinite(gradient_norm):
            raise FloatingPointError(
                "梯度范数不是有限数。"
            )

        grad_scaler.step(optimizer)
        grad_scaler.update()

        valid_interval_n = float(
            batch["at_risk_mask"].sum().item()
        )
        total_weighted_loss += (
            float(loss.item())
            * valid_interval_n
        )
        total_valid_intervals += (
            valid_interval_n
        )

        if (
            log_batches
            and (
                batch_index == 1
                or (
                    batch_index
                    % LOG_EVERY_N_BATCHES
                    == 0
                )
                or batch_index == len(loader)
            )
        ):
            current_loss = (
                total_weighted_loss
                / max(
                    total_valid_intervals,
                    1.0,
                )
            )
            progress_print(
                f"Epoch {epoch_index}/{max_epochs} | "
                f"批次{batch_index}/{len(loader)} | "
                f"训练NLL={current_loss:.6f} | "
                f"耗时={format_duration(time.time() - start_time)}"
            )

    return (
        total_weighted_loss
        / max(total_valid_intervals, 1.0)
    )


@torch.no_grad()
def evaluate_loader(
    model: nn.Module,
    loader: DataLoader,
    loss_function: nn.Module,
    return_predictions: bool,
) -> tuple[float, dict[str, np.ndarray] | None]:
    model.eval()
    total_weighted_loss = 0.0
    total_valid_intervals = 0.0

    hazard_batches = []
    patient_local_batches = []
    landmark_index_batches = []

    for raw_batch in loader:
        batch = move_batch_to_device(raw_batch)

        with autocast_context(
            device,
            use_amp,
        ):
            logits = model(
                dynamic_sequence=(
                    batch["dynamic_sequence"]
                ),
                row_mask=batch["row_mask"],
                static_baseline=(
                    batch["static_baseline"]
                ),
                age_at_landmark=(
                    batch["age_at_landmark"]
                ),
                landmark_normalized=(
                    batch["landmark_normalized"]
                ),
                landmark_bin=(
                    batch["landmark_bin"]
                ),
            )
            loss = loss_function(
                logits,
                batch["event_target"],
                batch["at_risk_mask"],
            )

        if not torch.isfinite(loss):
            raise FloatingPointError(
                "验证损失不是有限数。"
            )

        valid_interval_n = float(
            batch["at_risk_mask"].sum().item()
        )
        total_weighted_loss += (
            float(loss.item())
            * valid_interval_n
        )
        total_valid_intervals += (
            valid_interval_n
        )

        if return_predictions:
            hazard = torch.sigmoid(
                logits.float()
            )
            hazard_batches.append(
                hazard.cpu().numpy().astype(
                    np.float32
                )
            )
            patient_local_batches.append(
                batch["patient_local"]
                .numpy()
                .astype(np.int32)
            )
            landmark_index_batches.append(
                batch["landmark_index"]
                .numpy()
                .astype(np.int8)
            )

    validation_nll = (
        total_weighted_loss
        / max(total_valid_intervals, 1.0)
    )

    if not return_predictions:
        return validation_nll, None

    output = {
        "hazard": np.concatenate(
            hazard_batches,
            axis=0,
        ),
        "patient_local": np.concatenate(
            patient_local_batches,
            axis=0,
        ),
        "landmark_index": np.concatenate(
            landmark_index_batches,
            axis=0,
        ),
    }
    return validation_nll, output


def fit_fold_model(
    fold_data: dict[str, Any],
    parameters: dict[str, Any],
    seed: int,
    log_batches: bool,
) -> dict[str, Any]:
    set_random_seed(seed)

    train_loader, validation_loader = (
        create_data_loaders(
            fold_data,
            batch_size=int(
                parameters["batch_size"]
            ),
            seed=seed,
        )
    )

    model = MaskedSimpleRNNSurvival(
        dynamic_n=EXPECTED_DYNAMIC_N,
        static_n=EXPECTED_STATIC_N,
        hidden_size=int(
            parameters["hidden_size"]
        ),
        num_layers=int(
            parameters["num_layers"]
        ),
        dropout=float(parameters["dropout"]),
        output_interval_n=(
            EXPECTED_FUTURE_INTERVAL_N
        ),
    ).to(device)

    loss_function = MaskedDiscreteTimeNLL(
        positive_weight=float(
            parameters["positive_weight"]
        )
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(parameters["learning_rate"]),
        weight_decay=float(
            parameters["weight_decay"]
        ),
    )
    grad_scaler = make_grad_scaler(
        enabled=use_amp
    )

    best_validation_nll = math.inf
    best_epoch = -1
    best_state_dict = None
    no_improvement_n = 0
    history_rows = []
    fold_start = time.time()

    for epoch_index in range(
        1,
        MAX_EPOCHS + 1,
    ):
        train_nll = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            loss_function=loss_function,
            grad_scaler=grad_scaler,
            epoch_index=epoch_index,
            max_epochs=MAX_EPOCHS,
            log_batches=log_batches,
        )
        validation_nll, _ = evaluate_loader(
            model=model,
            loader=validation_loader,
            loss_function=loss_function,
            return_predictions=False,
        )

        history_rows.append(
            {
                "epoch": int(epoch_index),
                "train_nll": float(train_nll),
                "validation_nll":
                    float(validation_nll),
                "elapsed_seconds":
                    float(time.time() - fold_start),
            }
        )

        if log_batches:
            progress_print(
                f"Epoch {epoch_index}/{MAX_EPOCHS}完成 | "
                f"训练NLL={train_nll:.6f} | "
                f"验证NLL={validation_nll:.6f}"
            )

        improved = (
            validation_nll
            < (
                best_validation_nll
                - EARLY_STOPPING_MIN_DELTA
            )
        )
        if improved:
            best_validation_nll = float(
                validation_nll
            )
            best_epoch = int(epoch_index)
            best_state_dict = {
                key: value.detach().cpu().clone()
                for key, value in (
                    model.state_dict().items()
                )
            }
            no_improvement_n = 0
        else:
            no_improvement_n += 1

        if (
            no_improvement_n
            >= EARLY_STOPPING_PATIENCE
        ):
            break

    if best_state_dict is None:
        raise RuntimeError(
            "未保存任何有效RNN模型状态。"
        )

    model.load_state_dict(best_state_dict)
    final_validation_nll, predictions = (
        evaluate_loader(
            model=model,
            loader=validation_loader,
            loss_function=loss_function,
            return_predictions=True,
        )
    )

    hazard = np.asarray(
        predictions["hazard"],
        dtype=np.float32,
    )
    survival = np.cumprod(
        1.0 - np.clip(
            hazard.astype(np.float64),
            EPS,
            1.0 - EPS,
        ),
        axis=1,
    ).astype(np.float32)
    risk = (
        1.0 - survival
    ).astype(np.float32)

    if hazard.shape != (
        len(fold_data["validation_dataset"]),
        EXPECTED_FUTURE_INTERVAL_N,
    ):
        raise ValueError(
            "RNN验证条件风险形状错误。"
        )
    if not np.isfinite(hazard).all():
        raise ValueError(
            "RNN验证条件风险存在NaN或无穷值。"
        )
    if np.any(
        (hazard < 0.0) | (hazard > 1.0)
    ):
        raise ValueError(
            "RNN验证条件风险超出0～1。"
        )

    survival_violation_n = int(
        np.sum(
            np.diff(
                survival.astype(np.float64),
                axis=1,
            )
            > 1e-7
        )
    )
    risk_violation_n = int(
        np.sum(
            np.diff(
                risk.astype(np.float64),
                axis=1,
            )
            < -1e-7
        )
    )
    if survival_violation_n != 0:
        raise ValueError(
            "RNN验证生存概率不单调。"
        )
    if risk_violation_n != 0:
        raise ValueError(
            "RNN验证累积风险不单调。"
        )

    parameter_n = int(
        sum(
            parameter.numel()
            for parameter in model.parameters()
        )
    )

    result = {
        "model": model,
        "optimizer_state_dict":
            optimizer.state_dict(),
        "best_state_dict": best_state_dict,
        "history": pd.DataFrame(
            history_rows
        ),
        "best_epoch": int(best_epoch),
        "best_validation_nll": float(
            best_validation_nll
        ),
        "reloaded_validation_nll": float(
            final_validation_nll
        ),
        "hazard": hazard,
        "survival": survival,
        "risk": risk,
        "patient_local": np.asarray(
            predictions["patient_local"],
            dtype=np.int32,
        ),
        "landmark_index": np.asarray(
            predictions["landmark_index"],
            dtype=np.int8,
        ),
        "parameter_n": parameter_n,
        "elapsed_seconds": float(
            time.time() - fold_start
        ),
    }

    del train_loader
    del validation_loader
    del loss_function
    del optimizer
    del grad_scaler
    gc.collect()

    return result


# ============================================================
# 9. 评价函数
# ============================================================

def compute_equal_weight_landmark_ibs(
    train_long_idx: np.ndarray,
    validation_long_idx: np.ndarray,
    validation_survival: np.ndarray,
) -> tuple[float, pd.DataFrame]:
    rows = []

    train_landmarks = (
        landmark_month_long[train_long_idx]
    )
    validation_landmarks = (
        landmark_month_long[
            validation_long_idx
        ]
    )

    for landmark_month in LANDMARK_MONTHS:
        train_mask = (
            train_landmarks == landmark_month
        )
        validation_mask = (
            validation_landmarks
            == landmark_month
        )

        y_train = y_long_all[
            train_long_idx[train_mask]
        ]
        y_validation = y_long_all[
            validation_long_idx[
                validation_mask
            ]
        ]
        estimate = np.asarray(
            validation_survival[
                validation_mask
            ],
            dtype=np.float64,
        )

        if (
            len(y_train) == 0
            or len(y_validation) == 0
        ):
            raise ValueError(
                f"Landmark {landmark_month}个月"
                "缺少训练或验证记录。"
            )

        max_supported_time = min(
            float(np.max(y_train["time"])),
            float(
                np.max(y_validation["time"])
            ),
        )
        if (
            max_supported_time
            <= IBS_EVALUATION_TIMES[-1]
        ):
            raise ValueError(
                f"Landmark {landmark_month}个月"
                "不足以评价至59.999个月。"
            )

        ibs_value = float(
            integrated_brier_score(
                y_train,
                y_validation,
                estimate,
                IBS_EVALUATION_TIMES,
            )
        )

        rows.append(
            {
                "landmark_month": int(
                    landmark_month
                ),
                "train_record_n": int(
                    train_mask.sum()
                ),
                "validation_record_n": int(
                    validation_mask.sum()
                ),
                "validation_event_n": int(
                    y_validation["event"].sum()
                ),
                "ibs_0_5_to_5y": ibs_value,
            }
        )

    table = pd.DataFrame(rows)
    return (
        float(
            table["ibs_0_5_to_5y"].mean()
        ),
        table,
    )


def compute_landmark_c_index(
    validation_long_idx: np.ndarray,
    risk_60m: np.ndarray,
) -> pd.DataFrame:
    validation_landmarks = (
        landmark_month_long[
            validation_long_idx
        ]
    )
    rows = []

    for landmark_month in LANDMARK_MONTHS:
        mask = (
            validation_landmarks
            == landmark_month
        )
        y_validation = y_long_all[
            validation_long_idx[mask]
        ]
        score = np.asarray(
            risk_60m[mask],
            dtype=np.float64,
        )

        if (
            int(
                y_validation[
                    "event"
                ].sum()
            )
            == 0
        ):
            c_index = np.nan
        else:
            c_index = float(
                concordance_index_censored(
                    y_validation["event"],
                    y_validation["time"],
                    score,
                )[0]
            )

        rows.append(
            {
                "landmark_month": int(
                    landmark_month
                ),
                "record_n": int(mask.sum()),
                "event_n": int(
                    y_validation["event"].sum()
                ),
                "harrell_c_index":
                    c_index,
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# 10. 调参空间
# ============================================================

def decode_trial_parameters(
    trial: optuna.Trial,
) -> dict[str, Any]:
    return {
        "hidden_size": int(
            trial.suggest_categorical(
                "hidden_size",
                [32, 64, 96, 128],
            )
        ),
        "num_layers": int(
            trial.suggest_categorical(
                "num_layers",
                [1, 2],
            )
        ),
        "dropout": float(
            trial.suggest_categorical(
                "dropout",
                [0.10, 0.20, 0.30],
            )
        ),
        "learning_rate": float(
            trial.suggest_categorical(
                "learning_rate",
                [
                    3e-4,
                    7e-4,
                    1e-3,
                    2e-3,
                ],
            )
        ),
        "weight_decay": float(
            trial.suggest_categorical(
                "weight_decay",
                [
                    1e-6,
                    1e-5,
                    1e-4,
                    1e-3,
                ],
            )
        ),
        "batch_size": int(
            trial.suggest_categorical(
                "batch_size",
                [256, 512, 1024],
            )
        ),
        "positive_weight": float(
            trial.suggest_categorical(
                "positive_weight",
                [1.0, 2.0],
            )
        ),
    }


def normalize_stored_parameters(
    parameters: dict[str, Any],
) -> dict[str, Any]:
    return {
        "hidden_size": int(
            parameters["hidden_size"]
        ),
        "num_layers": int(
            parameters["num_layers"]
        ),
        "dropout": float(
            parameters["dropout"]
        ),
        "learning_rate": float(
            parameters["learning_rate"]
        ),
        "weight_decay": float(
            parameters["weight_decay"]
        ),
        "batch_size": int(
            parameters["batch_size"]
        ),
        "positive_weight": float(
            parameters["positive_weight"]
        ),
    }


# ============================================================
# 11. 调参阶段
# ============================================================

def run_tuning() -> None:
    print("\n" + "=" * 88)
    print(
        "Step 9B-Tune：普通RNN五折适度调参"
    )
    print("=" * 88)
    print("PyTorch版本：", torch.__version__)
    print(
        "scikit-survival版本：",
        sksurv.__version__,
    )
    print("Optuna版本：", optuna.__version__)
    print("运行设备：", device)
    if device.type == "cuda":
        print(
            "GPU：",
            torch.cuda.get_device_name(0),
        )
        print(
            "GPU显存：",
            f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GiB",
        )
    print("AMP：", use_amp)
    print(
        "目标完整Trial数：",
        TUNING_COMPLETE_TRIAL_TARGET,
    )
    print("每个Trial：完整固定五折")
    print(
        "目标函数：6个Landmark等权平均IBS，越低越好"
    )
    print("最大Epoch：", MAX_EPOCHS)
    print(
        "早停耐心：",
        EARLY_STOPPING_PATIENCE,
    )
    print(
        "锁定测试集：未读取"
    )
    print("输出目录：", TUNING_DIR)
    print("=" * 88)

    sampler = optuna.samplers.TPESampler(
        seed=BASE_RANDOM_SEED,
        n_startup_trials=4,
        multivariate=True,
    )
    study = optuna.create_study(
        study_name=OPTUNA_STUDY_NAME,
        direction="minimize",
        sampler=sampler,
        storage=OPTUNA_STORAGE_URL,
        load_if_exists=True,
    )

    def objective(
        trial: optuna.Trial,
    ) -> float:
        parameters = decode_trial_parameters(
            trial
        )
        trial_start = time.time()
        fold_ibs_values = []
        fold_nll_values = []
        fold_epoch_values = []
        fold_metric_frames = []

        progress_print(
            f"Trial {trial.number}开始 | "
            f"参数={parameters}"
        )
        append_progress_row(
            phase="tune",
            status="trial_started",
            trial_number=trial.number,
            message=json.dumps(
                parameters,
                ensure_ascii=False,
            ),
        )

        try:
            for fold_id in range(N_SPLITS):
                fold_start = time.time()
                progress_print(
                    f"Trial {trial.number} | "
                    f"第{fold_id + 1}/{N_SPLITS}折开始"
                )

                fold_data = prepare_fold_data(
                    fold_id
                )
                result = fit_fold_model(
                    fold_data=fold_data,
                    parameters=parameters,
                    seed=(
                        BASE_RANDOM_SEED
                        + trial.number * 1000
                        + fold_id
                    ),
                    log_batches=False,
                )

                fold_mean_ibs, fold_table = (
                    compute_equal_weight_landmark_ibs(
                        train_long_idx=(
                            fold_data[
                                "train_long_idx"
                            ]
                        ),
                        validation_long_idx=(
                            fold_data[
                                "validation_long_idx"
                            ]
                        ),
                        validation_survival=(
                            result["survival"]
                        ),
                    )
                )

                fold_table.insert(
                    0,
                    "fold_id",
                    fold_id,
                )
                fold_metric_frames.append(
                    fold_table
                )
                fold_ibs_values.append(
                    fold_mean_ibs
                )
                fold_nll_values.append(
                    result[
                        "reloaded_validation_nll"
                    ]
                )
                fold_epoch_values.append(
                    result["best_epoch"]
                )

                current_mean_ibs = float(
                    np.mean(
                        fold_ibs_values
                    )
                )
                fold_elapsed = (
                    time.time() - fold_start
                )

                progress_print(
                    f"Trial {trial.number} | "
                    f"第{fold_id + 1}/{N_SPLITS}折完成 | "
                    f"IBS={fold_mean_ibs:.6f} | "
                    f"验证NLL="
                    f"{result['reloaded_validation_nll']:.6f} | "
                    f"最佳Epoch={result['best_epoch']} | "
                    f"当前平均IBS={current_mean_ibs:.6f} | "
                    f"耗时={format_duration(fold_elapsed)}"
                )
                append_progress_row(
                    phase="tune",
                    status="fold_completed",
                    trial_number=trial.number,
                    fold_id=fold_id,
                    metric_name="ibs",
                    metric_value=fold_mean_ibs,
                    elapsed_seconds=fold_elapsed,
                )

                trial.report(
                    current_mean_ibs,
                    step=fold_id,
                )

                del result["model"]
                del result
                del fold_data
                gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()

            objective_value = float(
                np.mean(fold_ibs_values)
            )
            trial.set_user_attr(
                "fold_ibs_values",
                [
                    float(value)
                    for value in fold_ibs_values
                ],
            )
            trial.set_user_attr(
                "mean_validation_nll",
                float(
                    np.mean(
                        fold_nll_values
                    )
                ),
            )
            trial.set_user_attr(
                "median_best_epoch",
                int(
                    round(
                        float(
                            np.median(
                                fold_epoch_values
                            )
                        )
                    )
                ),
            )
            trial.set_user_attr(
                "fold_best_epochs",
                [
                    int(value)
                    for value in fold_epoch_values
                ],
            )
            trial.set_user_attr(
                "search_space_version",
                SEARCH_SPACE_VERSION,
            )

            trial_metric_table = pd.concat(
                fold_metric_frames,
                ignore_index=True,
            )
            trial_metric_table.to_csv(
                TUNING_DIR
                / (
                    f"trial_{trial.number:03d}_"
                    "landmark_metrics.csv"
                ),
                index=False,
                encoding="utf-8-sig",
            )

            elapsed = time.time() - trial_start
            progress_print(
                f"Trial {trial.number}完成 | "
                f"五折平均IBS={objective_value:.6f} | "
                f"耗时={format_duration(elapsed)}"
            )
            append_progress_row(
                phase="tune",
                status="trial_completed",
                trial_number=trial.number,
                metric_name="mean_ibs",
                metric_value=objective_value,
                elapsed_seconds=elapsed,
            )
            return objective_value

        except Exception as exc:
            append_progress_row(
                phase="tune",
                status="trial_failed",
                trial_number=trial.number,
                elapsed_seconds=(
                    time.time() - trial_start
                ),
                message=repr(exc),
            )
            progress_print(
                f"Trial {trial.number}失败："
                f"{type(exc).__name__}: {exc}"
            )
            raise

    total_attempt_n = 0
    while True:
        complete_trials = [
            trial
            for trial in study.trials
            if (
                trial.state
                == optuna.trial.TrialState.COMPLETE
            )
        ]

        if (
            len(complete_trials)
            >= TUNING_COMPLETE_TRIAL_TARGET
        ):
            break

        if (
            total_attempt_n
            >= TUNING_MAX_TOTAL_ATTEMPT_N
        ):
            raise RuntimeError(
                "已达到本次最大尝试Trial数，"
                "但完整Trial仍不足目标。"
            )

        remaining_complete_n = (
            TUNING_COMPLETE_TRIAL_TARGET
            - len(complete_trials)
        )
        study.optimize(
            objective,
            n_trials=1,
            gc_after_trial=True,
            show_progress_bar=False,
        )
        total_attempt_n += 1

        progress_print(
            "当前完整Trial数="
            f"{len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])}"
            f"/{TUNING_COMPLETE_TRIAL_TARGET} | "
            f"尚需约{remaining_complete_n - 1}"
        )

    trials_df = study.trials_dataframe()
    trials_df.to_csv(
        TUNING_DIR / "rnn_optuna_trials.csv",
        index=False,
        encoding="utf-8-sig",
    )

    best_parameters = (
        normalize_stored_parameters(
            study.best_trial.params
        )
    )
    tuning_summary = {
        "study_name": OPTUNA_STUDY_NAME,
        "direction": "minimize",
        "objective": (
            "Five-fold mean of equal-weight "
            "six-landmark IBS"
        ),
        "best_trial_number": int(
            study.best_trial.number
        ),
        "best_mean_ibs": float(
            study.best_value
        ),
        "best_parameters":
            best_parameters,
        "median_best_epoch": int(
            study.best_trial.user_attrs[
                "median_best_epoch"
            ]
        ),
        "fold_best_epochs":
            study.best_trial.user_attrs[
                "fold_best_epochs"
            ],
        "complete_trial_n": int(
            len(
                [
                    trial
                    for trial in study.trials
                    if (
                        trial.state
                        == optuna.trial.TrialState.COMPLETE
                    )
                ]
            )
        ),
        "failed_trial_n": int(
            len(
                [
                    trial
                    for trial in study.trials
                    if (
                        trial.state
                        == optuna.trial.TrialState.FAIL
                    )
                ]
            )
        ),
        "running_trial_n": int(
            len(
                [
                    trial
                    for trial in study.trials
                    if (
                        trial.state
                        == optuna.trial.TrialState.RUNNING
                    )
                ]
            )
        ),
        "search_space_version":
            SEARCH_SPACE_VERSION,
        "locked_test_read": False,
    }
    with open(
        TUNING_DIR / "rnn_tuning_summary.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            tuning_summary,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print("\n" + "=" * 88)
    print("RNN调参完成")
    print("=" * 88)
    print(
        "最佳Trial：",
        study.best_trial.number,
    )
    print(
        "最佳五折平均IBS：",
        f"{study.best_value:.6f}",
    )
    print("最佳参数：", best_parameters)
    print(
        "最佳Trial各折最佳Epoch：",
        study.best_trial.user_attrs[
            "fold_best_epochs"
        ],
    )
    print("输出目录：", TUNING_DIR)
    print(
        "下一步：把RUN_STAGE改为final，"
        "重新运行同一份代码。"
    )


# ============================================================
# 12. 最终五折OOF阶段
# ============================================================

def run_final_oof() -> None:
    if not OPTUNA_DB_FILE.exists():
        raise FileNotFoundError(
            "未找到RNN调参SQLite。"
            "请先运行RUN_STAGE='tune'。"
        )

    study = optuna.load_study(
        study_name=OPTUNA_STUDY_NAME,
        storage=OPTUNA_STORAGE_URL,
    )
    best_parameters = (
        normalize_stored_parameters(
            study.best_trial.params
        )
    )
    signature = parameter_signature(
        best_parameters
    )

    print("\n" + "=" * 88)
    print(
        "Step 9B-Final：最佳参数普通RNN五折OOF"
    )
    print("=" * 88)
    print(
        "最佳Trial：",
        study.best_trial.number,
    )
    print(
        "最佳调参IBS：",
        f"{study.best_value:.6f}",
    )
    print("最佳参数：", best_parameters)
    print("固定五折：是")
    print("折级续跑：是")
    print("锁定测试集：未读取")
    print("输出目录：", FINAL_DIR)
    print("=" * 88)

    with open(
        FINAL_DIR / "stage_configuration.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            {
                "stage": "final_oof",
                "best_trial_number": int(
                    study.best_trial.number
                ),
                "best_tuning_mean_ibs": float(
                    study.best_value
                ),
                "parameters":
                    best_parameters,
                "parameter_signature":
                    signature,
                "max_epochs": MAX_EPOCHS,
                "patience":
                    EARLY_STOPPING_PATIENCE,
                "locked_test_read": False,
            },
            file,
            ensure_ascii=False,
            indent=2,
        )

    oof_hazard_long = np.full(
        (
            EXPECTED_DEVELOPMENT_VALID_ORIGIN_N,
            EXPECTED_FUTURE_INTERVAL_N,
        ),
        np.nan,
        dtype=np.float32,
    )
    oof_survival_long = np.full_like(
        oof_hazard_long,
        np.nan,
    )
    oof_risk_long = np.full_like(
        oof_hazard_long,
        np.nan,
    )

    fold_summary_rows = []
    fold_landmark_tables = []
    final_static_features: list[str] | None = None
    total_start = time.time()

    for fold_id in range(N_SPLITS):
        fold_output_dir = (
            FINAL_DIR / f"fold_{fold_id}"
        )
        fold_output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )
        completed_file = (
            fold_output_dir / "completed.json"
        )

        fold_data = prepare_fold_data(
            fold_id
        )

        if final_static_features is None:
            final_static_features = list(
                fold_data["static_features"]
            )
        elif final_static_features != list(
            fold_data["static_features"]
        ):
            raise ValueError(
                "不同折的静态特征名称或顺序不一致。"
            )

        validation_long_idx = (
            fold_data["validation_long_idx"]
        )

        if completed_file.exists():
            completed = json.loads(
                completed_file.read_text(
                    encoding="utf-8"
                )
            )
            if (
                completed.get(
                    "parameter_signature"
                )
                != signature
            ):
                raise ValueError(
                    f"第{fold_id}折已有结果，"
                    "但参数签名不同。请删除："
                    f"{fold_output_dir}"
                )

            validation_hazard = np.load(
                fold_output_dir
                / "validation_hazard.npy"
            ).astype(np.float32)
            validation_survival = np.load(
                fold_output_dir
                / "validation_survival.npy"
            ).astype(np.float32)
            validation_risk = np.load(
                fold_output_dir
                / "validation_risk.npy"
            ).astype(np.float32)

            expected_shape = (
                len(validation_long_idx),
                EXPECTED_FUTURE_INTERVAL_N,
            )
            if (
                validation_hazard.shape
                != expected_shape
                or validation_survival.shape
                != expected_shape
                or validation_risk.shape
                != expected_shape
            ):
                raise ValueError(
                    f"第{fold_id}折已保存预测形状错误。"
                )

            oof_hazard_long[
                validation_long_idx
            ] = validation_hazard
            oof_survival_long[
                validation_long_idx
            ] = validation_survival
            oof_risk_long[
                validation_long_idx
            ] = validation_risk

            fold_summary_rows.append(
                pd.read_csv(
                    fold_output_dir
                    / "fold_summary.csv",
                    encoding="utf-8-sig",
                ).iloc[0].to_dict()
            )
            fold_landmark_tables.append(
                pd.read_csv(
                    fold_output_dir
                    / "validation_landmark_metrics.csv",
                    encoding="utf-8-sig",
                )
            )
            progress_print(
                f"final_oof第{fold_id + 1}/{N_SPLITS}折"
                "已完成，直接读取检查点。"
            )

            del fold_data
            gc.collect()
            continue

        progress_print(
            f"final_oof第{fold_id + 1}/{N_SPLITS}折开始 | "
            f"训练样本="
            f"{len(fold_data['train_dataset'])} | "
            f"验证样本="
            f"{len(fold_data['validation_dataset'])}"
        )
        fold_start = time.time()

        result = fit_fold_model(
            fold_data=fold_data,
            parameters=best_parameters,
            seed=(
                BASE_RANDOM_SEED
                + 10000
                + fold_id
            ),
            log_batches=True,
        )

        fold_mean_ibs, fold_ibs_table = (
            compute_equal_weight_landmark_ibs(
                train_long_idx=(
                    fold_data[
                        "train_long_idx"
                    ]
                ),
                validation_long_idx=(
                    validation_long_idx
                ),
                validation_survival=(
                    result["survival"]
                ),
            )
        )
        fold_c_table = compute_landmark_c_index(
            validation_long_idx=(
                validation_long_idx
            ),
            risk_60m=result["risk"][:, -1],
        )
        fold_landmark_table = (
            fold_ibs_table.merge(
                fold_c_table,
                on="landmark_month",
                how="inner",
                validate="one_to_one",
            )
        )
        fold_landmark_table.insert(
            0,
            "fold_id",
            fold_id,
        )

        validation_map = pd.DataFrame(
            {
                "prediction_row": np.arange(
                    len(validation_long_idx),
                    dtype=np.int32,
                ),
                "long_row_index":
                    validation_long_idx,
                "patient_local_index":
                    result["patient_local"],
                "patient_global_index":
                    development_idx[
                        result["patient_local"]
                    ],
                "landmark_index":
                    result["landmark_index"],
                "landmark_month":
                    LANDMARK_MONTHS[
                        result[
                            "landmark_index"
                        ]
                    ],
            }
        )

        expected_long_from_prediction = (
            long_row_index_map[
                result["patient_local"],
                result["landmark_index"],
            ]
        )
        if not np.array_equal(
            expected_long_from_prediction,
            validation_long_idx,
        ):
            raise ValueError(
                f"第{fold_id}折预测顺序与"
                "Step 6长记录顺序不一致。"
            )

        fold_elapsed = (
            time.time() - fold_start
        )
        fold_summary = pd.DataFrame(
            [
                {
                    "fold_id": fold_id,
                    "train_patient_n": int(
                        np.sum(
                            development_fold_id
                            != fold_id
                        )
                    ),
                    "validation_patient_n": int(
                        np.sum(
                            development_fold_id
                            == fold_id
                        )
                    ),
                    "train_origin_n": int(
                        len(
                            fold_data[
                                "train_dataset"
                            ]
                        )
                    ),
                    "validation_origin_n": int(
                        len(
                            fold_data[
                                "validation_dataset"
                            ]
                        )
                    ),
                    "best_epoch": int(
                        result["best_epoch"]
                    ),
                    "best_validation_nll": float(
                        result[
                            "best_validation_nll"
                        ]
                    ),
                    "reloaded_validation_nll": float(
                        result[
                            "reloaded_validation_nll"
                        ]
                    ),
                    "mean_ibs": float(
                        fold_mean_ibs
                    ),
                    "mean_harrell_c_index": float(
                        fold_landmark_table[
                            "harrell_c_index"
                        ].mean()
                    ),
                    "model_parameter_n": int(
                        result["parameter_n"]
                    ),
                    "elapsed_seconds": float(
                        fold_elapsed
                    ),
                }
            ]
        )

        torch.save(
            {
                "model_state_dict":
                    result["best_state_dict"],
                "optimizer_state_dict":
                    result[
                        "optimizer_state_dict"
                    ],
                "fold_id": fold_id,
                "parameters":
                    best_parameters,
                "best_epoch": int(
                    result["best_epoch"]
                ),
                "best_validation_nll": float(
                    result[
                        "best_validation_nll"
                    ]
                ),
                "feature_partition": {
                    "static_features":
                        fold_data[
                            "static_features"
                        ],
                    "dynamic_features":
                        fold_data[
                            "dynamic_features"
                        ],
                    "age_context": ["Age"],
                    "landmark_context": [
                        "landmark_month"
                    ],
                },
            },
            fold_output_dir
            / "best_model.pt",
        )
        result["history"].to_csv(
            fold_output_dir
            / "training_history.csv",
            index=False,
            encoding="utf-8-sig",
        )
        np.save(
            fold_output_dir
            / "validation_hazard.npy",
            result["hazard"],
        )
        np.save(
            fold_output_dir
            / "validation_survival.npy",
            result["survival"],
        )
        np.save(
            fold_output_dir
            / "validation_risk.npy",
            result["risk"],
        )
        validation_map.to_csv(
            fold_output_dir
            / "validation_prediction_map.csv",
            index=False,
            encoding="utf-8-sig",
        )
        fold_summary.to_csv(
            fold_output_dir
            / "fold_summary.csv",
            index=False,
            encoding="utf-8-sig",
        )
        fold_landmark_table.to_csv(
            fold_output_dir
            / "validation_landmark_metrics.csv",
            index=False,
            encoding="utf-8-sig",
        )

        with open(
            completed_file,
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                {
                    "fold_id": fold_id,
                    "parameter_signature":
                        signature,
                    "completed_at":
                        datetime.now().isoformat(
                            timespec="seconds"
                        ),
                },
                file,
                ensure_ascii=False,
                indent=2,
            )

        oof_hazard_long[
            validation_long_idx
        ] = result["hazard"]
        oof_survival_long[
            validation_long_idx
        ] = result["survival"]
        oof_risk_long[
            validation_long_idx
        ] = result["risk"]

        fold_summary_rows.append(
            fold_summary.iloc[0].to_dict()
        )
        fold_landmark_tables.append(
            fold_landmark_table
        )

        progress_print(
            f"final_oof第{fold_id + 1}/{N_SPLITS}折完成 | "
            f"IBS={fold_mean_ibs:.6f} | "
            f"平均C-index="
            f"{fold_landmark_table['harrell_c_index'].mean():.6f} | "
            f"最佳Epoch={result['best_epoch']} | "
            f"耗时={format_duration(fold_elapsed)}"
        )
        append_progress_row(
            phase="final_oof",
            status="fold_completed",
            fold_id=fold_id,
            metric_name="ibs",
            metric_value=fold_mean_ibs,
            elapsed_seconds=fold_elapsed,
        )

        del result["model"]
        del result
        del fold_data
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if np.isnan(oof_hazard_long).any():
        raise ValueError(
            "最终RNN OOF条件风险存在缺失。"
        )
    if np.isnan(oof_survival_long).any():
        raise ValueError(
            "最终RNN OOF生存概率存在缺失。"
        )
    if np.isnan(oof_risk_long).any():
        raise ValueError(
            "最终RNN OOF累积风险存在缺失。"
        )

    survival_violation_n = int(
        np.sum(
            np.diff(
                oof_survival_long.astype(
                    np.float64
                ),
                axis=1,
            )
            > 1e-7
        )
    )
    risk_violation_n = int(
        np.sum(
            np.diff(
                oof_risk_long.astype(
                    np.float64
                ),
                axis=1,
            )
            < -1e-7
        )
    )
    identity_max_difference = float(
        np.max(
            np.abs(
                oof_risk_long
                - (
                    1.0
                    - oof_survival_long
                )
            )
        )
    )

    if survival_violation_n != 0:
        raise ValueError(
            "最终RNN OOF生存概率不单调。"
        )
    if risk_violation_n != 0:
        raise ValueError(
            "最终RNN OOF累积风险不单调。"
        )
    if identity_max_difference > 1e-6:
        raise ValueError(
            "最终RNN risk不等于1-survival。"
        )

    oof_hazard_patient = np.full(
        (
            EXPECTED_DEVELOPMENT_N,
            EXPECTED_LANDMARK_N,
            EXPECTED_FUTURE_INTERVAL_N,
        ),
        np.nan,
        dtype=np.float32,
    )
    oof_survival_patient = np.full_like(
        oof_hazard_patient,
        np.nan,
    )
    oof_risk_patient = np.full_like(
        oof_hazard_patient,
        np.nan,
    )

    valid_patient, valid_landmark = np.where(
        long_row_index_map >= 0
    )
    valid_long_idx = long_row_index_map[
        valid_patient,
        valid_landmark,
    ]

    if not np.array_equal(
        np.sort(valid_long_idx),
        np.arange(
            EXPECTED_DEVELOPMENT_VALID_ORIGIN_N
        ),
    ):
        raise ValueError(
            "Step 6长行号不是0～100121的完整排列。"
        )

    oof_hazard_patient[
        valid_patient,
        valid_landmark,
        :,
    ] = oof_hazard_long[
        valid_long_idx
    ]
    oof_survival_patient[
        valid_patient,
        valid_landmark,
        :,
    ] = oof_survival_long[
        valid_long_idx
    ]
    oof_risk_patient[
        valid_patient,
        valid_landmark,
        :,
    ] = oof_risk_long[
        valid_long_idx
    ]

    if np.isnan(
        oof_risk_patient[
            prediction_origin_mask
        ]
    ).any():
        raise ValueError(
            "患者级有效预测起点存在缺失。"
        )
    if not np.isnan(
        oof_risk_patient[
            ~prediction_origin_mask
        ]
    ).all():
        raise ValueError(
            "患者级无效预测起点没有保持NaN。"
        )

    np.save(
        FINAL_DIR / "rnn_oof_hazard_long.npy",
        oof_hazard_long,
    )
    np.save(
        FINAL_DIR / "rnn_oof_survival_long.npy",
        oof_survival_long,
    )
    np.save(
        FINAL_DIR / "rnn_oof_risk_long.npy",
        oof_risk_long,
    )
    np.save(
        FINAL_DIR / "rnn_oof_risk_score_long.npy",
        oof_risk_long[:, -1],
    )

    np.save(
        FINAL_DIR / "rnn_oof_hazard.npy",
        oof_hazard_patient,
    )
    np.save(
        FINAL_DIR / "rnn_oof_survival.npy",
        oof_survival_patient,
    )
    np.save(
        FINAL_DIR / "rnn_oof_risk.npy",
        oof_risk_patient,
    )
    np.save(
        FINAL_DIR / "rnn_oof_risk_score.npy",
        oof_risk_patient[:, :, -1],
    )

    fold_summary_table = pd.DataFrame(
        fold_summary_rows
    ).sort_values(
        "fold_id"
    ).reset_index(drop=True)
    fold_landmark_table = pd.concat(
        fold_landmark_tables,
        ignore_index=True,
    ).sort_values(
        ["fold_id", "landmark_month"]
    ).reset_index(drop=True)

    fold_summary_table.to_csv(
        FINAL_DIR / "rnn_fold_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    fold_landmark_table.to_csv(
        FINAL_DIR
        / "rnn_fold_landmark_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    feature_partition_rows = [
        {
            "feature_name": "Age",
            "feature_group":
                "age_context",
        }
    ]
    if final_static_features is None:
        raise RuntimeError(
            "未获得最终静态特征列表。"
        )

    feature_partition_rows.extend(
        {
            "feature_name": name,
            "feature_group":
                "baseline_static",
        }
        for name in final_static_features
    )
    feature_partition_rows.extend(
        {
            "feature_name": name,
            "feature_group":
                "dynamic_sequence",
        }
        for name in DYNAMIC_FEATURES
    )
    pd.DataFrame(
        feature_partition_rows
    ).drop_duplicates().to_csv(
        FINAL_DIR
        / "rnn_feature_partition.csv",
        index=False,
        encoding="utf-8-sig",
    )

    total_elapsed = (
        time.time() - total_start
    )
    final_summary = {
        "stage": "Step9B_v2_final_oof",
        "best_trial_number": int(
            study.best_trial.number
        ),
        "best_tuning_mean_ibs": float(
            study.best_value
        ),
        "parameters": best_parameters,
        "development_patient_n":
            EXPECTED_DEVELOPMENT_N,
        "development_valid_origin_n":
            EXPECTED_DEVELOPMENT_VALID_ORIGIN_N,
        "landmark_months":
            LANDMARK_MONTHS.tolist(),
        "future_end_months":
            FUTURE_END_MONTHS.tolist(),
        "oof_hazard_long_shape":
            list(oof_hazard_long.shape),
        "oof_patient_shape":
            list(oof_risk_patient.shape),
        "mean_fold_ibs": float(
            fold_summary_table[
                "mean_ibs"
            ].mean()
        ),
        "mean_fold_harrell_c_index": float(
            fold_summary_table[
                "mean_harrell_c_index"
            ].mean()
        ),
        "survival_monotonic_violation_n":
            survival_violation_n,
        "risk_monotonic_violation_n":
            risk_violation_n,
        "risk_identity_max_difference":
            identity_max_difference,
        "total_elapsed_seconds":
            float(total_elapsed),
        "locked_test_read": False,
    }
    with open(
        FINAL_DIR / "rnn_summary.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            final_summary,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print("\n" + "=" * 88)
    print("最终RNN五折OOF完成")
    print("=" * 88)
    print(
        "五折平均IBS：",
        f"{final_summary['mean_fold_ibs']:.6f}",
    )
    print(
        "五折平均C-index：",
        f"{final_summary['mean_fold_harrell_c_index']:.6f}",
    )
    print(
        "OOF长格式形状：",
        oof_risk_long.shape,
    )
    print(
        "OOF患者级形状：",
        oof_risk_patient.shape,
    )
    print(
        "生存概率单调性违反数：",
        survival_violation_n,
    )
    print(
        "累积风险单调性违反数：",
        risk_violation_n,
    )
    print(
        "总耗时：",
        format_duration(total_elapsed),
    )
    print("输出目录：", FINAL_DIR)


# ============================================================
# 13. 启动
# ============================================================

print("\n" + "=" * 88)
print(
    "Step 9B：普通RNN适度调参与可恢复五折OOF"
)
print("=" * 88)
print("当前阶段：", RUN_STAGE)
print("进程PID：", os.getpid())
print("运行设备：", device)
if device.type == "cuda":
    print(
        "GPU：",
        torch.cuda.get_device_name(0),
    )
print("AMP：", use_amp)
print("开发集患者数：", EXPECTED_DEVELOPMENT_N)
print(
    "有效患者-Landmark记录数：",
    EXPECTED_DEVELOPMENT_VALID_ORIGIN_N,
)
print("锁定测试集：未读取")
print("实时日志：", LIVE_PROGRESS_FILE)
print("=" * 88)

if RUN_STAGE == "tune":
    run_tuning()
elif RUN_STAGE == "final":
    run_final_oof()
else:
    raise ValueError(
        "RUN_STAGE只能是'tune'或'final'。"
    )

gc.collect()
if device.type == "cuda":
    torch.cuda.empty_cache()
