#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Step 10E FINAL：基于现有Trial选定参数的LSTM-v2正式五折OOF

本文件已经内置：
- 数据读取与核对；
- 强化纵向特征；
- Hybrid Attention LSTM；
- Landmark平衡损失；
- IBS早停；
- 多快照与多随机种子集成；
- 正式五折OOF输出。

不依赖Step10D_LSTM_v2_common.py。
运行前必须先完成Step 10D单文件独立版。
锁定测试集不读取。
"""

from __future__ import annotations

import gc
import json
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import (
    DataLoader,
    Dataset,
    WeightedRandomSampler,
)

try:
    from sksurv.metrics import (
        concordance_index_ipcw,
        cumulative_dynamic_auc,
        integrated_brier_score,
    )
    from sksurv.util import Surv
except ImportError as exc:
    raise ImportError(
        "需要scikit-survival。请在当前环境运行：\n"
        "conda install -c conda-forge scikit-survival"
    ) from exc


# =============================================================================
# 1. 固定研究配置
# =============================================================================

PROJECT_DIR = Path(
    os.getenv("CKD_LSTM_PROJECT_DIR", "__CKD_WORKDIR__")
)

STEP1_DIR = PROJECT_DIR / "rolling_5y_step1_new_split"
STEP2_DIR = PROJECT_DIR / "rolling_5y_step2_folds"
STEP3_DIR = PROJECT_DIR / "rolling_5y_step3_raw_features"
STEP4_DIR = PROJECT_DIR / "rolling_5y_step4_preprocessed"
STEP6_DIR = PROJECT_DIR / "rolling_5y_step6_super_landmark_data"

EXPECTED_TOTAL_N = 31911
EXPECTED_DEVELOPMENT_N = 22337
EXPECTED_HISTORY_STEP_N = 11
EXPECTED_BASE_FEATURE_N = 57
EXPECTED_STATIC_N = 16
EXPECTED_BASE_DYNAMIC_N = 40
EXPECTED_LAB_N = 15
EXPECTED_EXTRA_DYNAMIC_N = EXPECTED_LAB_N * 3
EXPECTED_ENHANCED_DYNAMIC_N = (
    EXPECTED_BASE_DYNAMIC_N + EXPECTED_EXTRA_DYNAMIC_N
)
EXPECTED_LANDMARK_N = 6
EXPECTED_FUTURE_INTERVAL_N = 10
EXPECTED_DEVELOPMENT_VALID_ORIGIN_N = 100122
N_SPLITS = 5

LANDMARK_MONTHS = np.asarray(
    [0, 12, 24, 36, 48, 60],
    dtype=np.int32,
)
LANDMARK_BINS = (LANDMARK_MONTHS // 6).astype(np.int64)
FUTURE_END_MONTHS = np.arange(6, 61, 6, dtype=np.float64)
METRIC_TIMES = np.asarray(
    [6, 12, 18, 24, 30, 36, 42, 48, 54, 59.999],
    dtype=np.float64,
)
TIME_STEP_YEARS = (
    np.arange(EXPECTED_HISTORY_STEP_N, dtype=np.float32) * 0.5
)
EPS = 1e-7

USE_AMP = True
REQUIRE_CUDA = True
NUM_WORKERS = 0
MAX_EPOCHS = 60
MIN_EPOCHS = 5
EARLY_STOPPING_PATIENCE = 10
EARLY_STOPPING_MIN_DELTA = 1e-5
GRADIENT_CLIP_NORM = 1.0
TOP_SNAPSHOT_N = 3
LR_REDUCE_FACTOR = 0.5
LR_REDUCE_PATIENCE = 3
MIN_LEARNING_RATE = 1e-6

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


# =============================================================================
# 2. 通用函数
# =============================================================================

def set_random_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    torch.cuda.manual_seed_all(int(seed))
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def format_duration(seconds: float) -> str:
    seconds = max(0, int(round(float(seconds))))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}小时{minutes:02d}分{seconds:02d}秒"
    if minutes:
        return f"{minutes}分{seconds:02d}秒"
    return f"{seconds}秒"


def save_json(value: Any, path: Path) -> None:
    path.write_text(
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )


def require_files(paths: list[Path]) -> None:
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "以下必要文件不存在：\n" + "\n".join(missing)
        )


def build_survival_array(
    event: np.ndarray,
    time_month: np.ndarray,
) -> np.ndarray:
    event = np.asarray(event, dtype=bool)
    time_month = np.asarray(time_month, dtype=np.float64)
    if event.shape != time_month.shape:
        raise ValueError("事件与时间形状不一致。")
    if not np.isfinite(time_month).all() or np.any(time_month <= 0):
        raise ValueError("生存时间必须为有限正数。")
    return Surv.from_arrays(event=event, time=time_month)


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


# =============================================================================
# 3. 共同数据
# =============================================================================

@dataclass
class CommonData:
    development_idx: np.ndarray
    development_fold_id: np.ndarray
    sequence_row_mask: np.ndarray
    prediction_origin_mask: np.ndarray
    future_event_matrix: np.ndarray
    future_at_risk_mask: np.ndarray
    continuous_raw: np.ndarray
    continuous_vars: list[str]
    age_continuous_index: int
    lab_continuous_indices: np.ndarray
    long_row_index_map: np.ndarray
    fold_id_long: np.ndarray
    landmark_month_long: np.ndarray
    analysis_time_month: np.ndarray
    event_within_60m: np.ndarray
    y_long_all: np.ndarray


def load_common_data() -> CommonData:
    required = [
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
        required.extend(
            [
                fold_dir / "X_development.npy",
                fold_dir / "preprocessor.joblib",
                fold_dir / "feature_names.csv",
            ]
        )
    require_files(required)

    development_idx_step1 = np.load(
        STEP1_DIR / "development_idx.npy"
    ).astype(np.int32)
    development_idx = np.load(
        STEP4_DIR / "development_idx.npy"
    ).astype(np.int32)
    fold_step2 = np.load(
        STEP2_DIR / "development_fold_id.npy"
    ).astype(np.int8)
    development_fold_id = np.load(
        STEP4_DIR / "development_fold_id.npy"
    ).astype(np.int8)

    if not np.array_equal(development_idx_step1, development_idx):
        raise ValueError("Step 1与Step 4开发集顺序不一致。")
    if not np.array_equal(fold_step2, development_fold_id):
        raise ValueError("Step 2与Step 4固定五折不一致。")

    sequence_row_mask = np.load(
        STEP4_DIR / "sequence_row_mask_development.npy"
    ).astype(bool)

    prediction_origin_mask_all = np.load(
        STEP1_DIR / "prediction_origin_mask.npy"
    ).astype(bool)
    future_event_all = np.load(
        STEP1_DIR / "future_event_matrix.npy"
    ).astype(np.float32)
    future_at_risk_all = np.load(
        STEP1_DIR / "future_at_risk_mask.npy"
    ).astype(bool)

    landmark_months_file = np.load(
        STEP1_DIR / "landmark_months.npy"
    ).astype(np.int32)
    landmark_bins_file = np.load(
        STEP1_DIR / "landmark_bins.npy"
    ).astype(np.int64)

    if not np.array_equal(landmark_months_file, LANDMARK_MONTHS):
        raise ValueError("Landmark月份配置不一致。")
    if not np.array_equal(landmark_bins_file, LANDMARK_BINS):
        raise ValueError("Landmark时间行配置不一致。")

    continuous_raw = np.load(
        STEP3_DIR / "continuous_raw_0_60.npy",
        mmap_mode="r",
    )
    feature_groups = json.loads(
        (STEP3_DIR / "feature_groups.json").read_text(
            encoding="utf-8"
        )
    )
    continuous_vars = list(feature_groups["continuous_vars"])

    if "Age" not in continuous_vars:
        raise ValueError("连续变量中缺少Age。")
    missing_labs = sorted(set(LAB_FEATURES) - set(continuous_vars))
    if missing_labs:
        raise ValueError(f"连续变量中缺少实验室指标：{missing_labs}")

    age_continuous_index = continuous_vars.index("Age")
    lab_continuous_indices = np.asarray(
        [continuous_vars.index(name) for name in LAB_FEATURES],
        dtype=np.int64,
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

    prediction_origin_mask = prediction_origin_mask_all[development_idx]
    future_event_matrix = future_event_all[development_idx]
    future_at_risk_mask = future_at_risk_all[development_idx]

    expected_checks = [
        (
            development_idx.shape,
            (EXPECTED_DEVELOPMENT_N,),
            "development_idx",
        ),
        (
            development_fold_id.shape,
            (EXPECTED_DEVELOPMENT_N,),
            "development_fold_id",
        ),
        (
            sequence_row_mask.shape,
            (EXPECTED_DEVELOPMENT_N, EXPECTED_HISTORY_STEP_N),
            "sequence_row_mask",
        ),
        (
            prediction_origin_mask.shape,
            (EXPECTED_DEVELOPMENT_N, EXPECTED_LANDMARK_N),
            "prediction_origin_mask",
        ),
        (
            future_event_matrix.shape,
            (
                EXPECTED_DEVELOPMENT_N,
                EXPECTED_LANDMARK_N,
                EXPECTED_FUTURE_INTERVAL_N,
            ),
            "future_event_matrix",
        ),
        (
            future_at_risk_mask.shape,
            (
                EXPECTED_DEVELOPMENT_N,
                EXPECTED_LANDMARK_N,
                EXPECTED_FUTURE_INTERVAL_N,
            ),
            "future_at_risk_mask",
        ),
        (
            long_row_index_map.shape,
            (EXPECTED_DEVELOPMENT_N, EXPECTED_LANDMARK_N),
            "long_row_index_map",
        ),
    ]
    for actual, expected, name in expected_checks:
        if actual != expected:
            raise ValueError(f"{name}形状={actual}，预期={expected}。")

    if int(prediction_origin_mask.sum()) != EXPECTED_DEVELOPMENT_VALID_ORIGIN_N:
        raise ValueError("有效患者-Landmark记录数不是100122。")
    if not np.array_equal(
        prediction_origin_mask,
        long_row_index_map >= 0,
    ):
        raise ValueError("标签有效起点与长格式映射不一致。")
    if np.any(future_event_matrix.astype(bool) & ~future_at_risk_mask):
        raise ValueError("存在事件标签为1但风险掩码为0。")
    if np.any(future_event_matrix.sum(axis=2) > 1):
        raise ValueError("同一患者-Landmark存在多个事件区间。")
    if not np.isin(development_fold_id, np.arange(N_SPLITS)).all():
        raise ValueError("开发集固定折编号必须为0～4。")
    if len(analysis_time_month) != EXPECTED_DEVELOPMENT_VALID_ORIGIN_N:
        raise ValueError("Step 6生存时间记录数错误。")

    y_long_all = build_survival_array(
        event_within_60m,
        analysis_time_month,
    )

    return CommonData(
        development_idx=development_idx,
        development_fold_id=development_fold_id,
        sequence_row_mask=sequence_row_mask,
        prediction_origin_mask=prediction_origin_mask,
        future_event_matrix=future_event_matrix,
        future_at_risk_mask=future_at_risk_mask,
        continuous_raw=continuous_raw,
        continuous_vars=continuous_vars,
        age_continuous_index=age_continuous_index,
        lab_continuous_indices=lab_continuous_indices,
        long_row_index_map=long_row_index_map,
        fold_id_long=fold_id_long,
        landmark_month_long=landmark_month_long,
        analysis_time_month=analysis_time_month,
        event_within_60m=event_within_60m,
        y_long_all=y_long_all,
    )


# =============================================================================
# 4. 强化纵向特征
# =============================================================================

def build_enhanced_dynamic_array(
    base_dynamic: np.ndarray,
    lab_raw_development: np.ndarray,
    sequence_row_mask: np.ndarray,
) -> tuple[np.ndarray, list[str]]:
    """
    追加：
    - 15项实验室是否真实测量；
    - 15项实验室距上次真实测量时间（除以5年）；
    - 15项实验室与上次真实测量的标准化变化量。
    """
    base_dynamic = np.asarray(base_dynamic, dtype=np.float32)
    lab_raw_development = np.asarray(
        lab_raw_development,
        dtype=np.float32,
    )
    sequence_row_mask = np.asarray(sequence_row_mask, dtype=bool)

    expected_base_shape = (
        EXPECTED_DEVELOPMENT_N,
        EXPECTED_HISTORY_STEP_N,
        EXPECTED_BASE_DYNAMIC_N,
    )
    expected_lab_shape = (
        EXPECTED_DEVELOPMENT_N,
        EXPECTED_HISTORY_STEP_N,
        EXPECTED_LAB_N,
    )
    if base_dynamic.shape != expected_base_shape:
        raise ValueError(
            f"基础动态特征形状={base_dynamic.shape}，"
            f"预期={expected_base_shape}。"
        )
    if lab_raw_development.shape != expected_lab_shape:
        raise ValueError(
            f"原始实验室形状={lab_raw_development.shape}，"
            f"预期={expected_lab_shape}。"
        )

    lab_observed = (
        np.isfinite(lab_raw_development)
        & sequence_row_mask[:, :, None]
    )
    lab_observed_float = lab_observed.astype(np.float32)

    time_since = np.zeros_like(lab_observed_float, dtype=np.float32)
    lab_delta = np.zeros_like(lab_observed_float, dtype=np.float32)

    standardized_labs = base_dynamic[:, :, :EXPECTED_LAB_N]
    last_seen_step = np.full(
        (EXPECTED_DEVELOPMENT_N, EXPECTED_LAB_N),
        -1,
        dtype=np.int16,
    )
    last_seen_value = np.zeros(
        (EXPECTED_DEVELOPMENT_N, EXPECTED_LAB_N),
        dtype=np.float32,
    )
    has_seen = np.zeros(
        (EXPECTED_DEVELOPMENT_N, EXPECTED_LAB_N),
        dtype=bool,
    )

    for step_index in range(EXPECTED_HISTORY_STEP_N):
        active_row = sequence_row_mask[:, step_index][:, None]
        observed_now = lab_observed[:, step_index, :]
        current_value = standardized_labs[:, step_index, :]

        elapsed_years = (
            step_index - last_seen_step
        ).astype(np.float32) * 0.5
        elapsed_years = np.clip(elapsed_years, 0.0, 5.0)
        elapsed_years[~has_seen] = min(
            (step_index + 1) * 0.5,
            5.0,
        )
        time_since[:, step_index, :] = np.where(
            active_row,
            np.where(observed_now, 0.0, elapsed_years / 5.0),
            0.0,
        )

        delta_now = current_value - last_seen_value
        lab_delta[:, step_index, :] = np.where(
            observed_now & has_seen,
            delta_now,
            0.0,
        )

        last_seen_value = np.where(
            observed_now,
            current_value,
            last_seen_value,
        )
        last_seen_step = np.where(
            observed_now,
            step_index,
            last_seen_step,
        )
        has_seen |= observed_now

    enhanced = np.concatenate(
        [
            base_dynamic,
            lab_observed_float,
            time_since,
            lab_delta,
        ],
        axis=2,
    ).astype(np.float32)

    enhanced[~sequence_row_mask] = 0.0

    if enhanced.shape != (
        EXPECTED_DEVELOPMENT_N,
        EXPECTED_HISTORY_STEP_N,
        EXPECTED_ENHANCED_DYNAMIC_N,
    ):
        raise ValueError("强化动态特征形状错误。")
    if not np.isfinite(enhanced).all():
        raise ValueError("强化动态特征存在NaN或无穷值。")

    names = (
        DYNAMIC_FEATURES
        + [f"{name}_observed" for name in LAB_FEATURES]
        + [f"{name}_time_since_last" for name in LAB_FEATURES]
        + [f"{name}_delta_last_observed" for name in LAB_FEATURES]
    )
    return enhanced, names


# =============================================================================
# 5. Dataset与Fold数据
# =============================================================================

class EnhancedPatientLandmarkDataset(Dataset):
    def __init__(
        self,
        sample_pairs: np.ndarray,
        enhanced_dynamic: np.ndarray,
        sequence_mask: np.ndarray,
        static_baseline: np.ndarray,
        baseline_age_raw: np.ndarray,
        event_matrix: np.ndarray,
        at_risk_mask: np.ndarray,
        age_mean: float,
        age_scale: float,
    ):
        self.sample_pairs = np.asarray(sample_pairs, dtype=np.int32)
        self.enhanced_dynamic = enhanced_dynamic
        self.sequence_mask = sequence_mask
        self.static_baseline = static_baseline
        self.baseline_age_raw = baseline_age_raw
        self.event_matrix = event_matrix
        self.at_risk_mask = at_risk_mask
        self.age_mean = float(age_mean)
        self.age_scale = float(age_scale)

    def __len__(self) -> int:
        return len(self.sample_pairs)

    def __getitem__(self, index: int) -> dict[str, np.ndarray]:
        patient_local, landmark_index = self.sample_pairs[index]
        landmark_month = float(LANDMARK_MONTHS[landmark_index])

        age_raw = (
            float(self.baseline_age_raw[patient_local])
            + landmark_month / 12.0
        )
        age_standardized = (
            age_raw - self.age_mean
        ) / self.age_scale

        return {
            "dynamic_sequence": np.asarray(
                self.enhanced_dynamic[patient_local],
                dtype=np.float32,
            ),
            "row_mask": np.asarray(
                self.sequence_mask[patient_local],
                dtype=np.bool_,
            ),
            "static_baseline": np.asarray(
                self.static_baseline[patient_local],
                dtype=np.float32,
            ),
            "age_at_landmark": np.float32(age_standardized),
            "landmark_normalized": np.float32(landmark_month / 60.0),
            "landmark_bin": np.int64(LANDMARK_BINS[landmark_index]),
            "event_target": np.asarray(
                self.event_matrix[patient_local, landmark_index],
                dtype=np.float32,
            ),
            "at_risk_mask": np.asarray(
                self.at_risk_mask[patient_local, landmark_index],
                dtype=np.float32,
            ),
            "patient_local": np.int64(patient_local),
            "landmark_index": np.int64(landmark_index),
        }


def prepare_fold_data(
    common: CommonData,
    fold_id: int,
) -> dict[str, Any]:
    fold_dir = STEP4_DIR / f"fold_{fold_id}"

    x_development = np.load(
        fold_dir / "X_development.npy",
        mmap_mode="r",
    )
    feature_names = (
        pd.read_csv(
            fold_dir / "feature_names.csv",
            encoding="utf-8-sig",
        )["feature_name"]
        .astype(str)
        .tolist()
    )
    preprocessor = joblib.load(
        fold_dir / "preprocessor.joblib"
    )

    if x_development.shape != (
        EXPECTED_DEVELOPMENT_N,
        EXPECTED_HISTORY_STEP_N,
        EXPECTED_BASE_FEATURE_N,
    ):
        raise ValueError(f"第{fold_id}折预处理张量形状错误。")
    if len(feature_names) != EXPECTED_BASE_FEATURE_N:
        raise ValueError(f"第{fold_id}折预处理特征数不是57。")

    x_array = np.asarray(x_development)
    if not np.isfinite(x_array).all():
        raise ValueError(f"第{fold_id}折预处理张量存在非法值。")
    if not np.all(x_array[~common.sequence_row_mask] == 0.0):
        raise ValueError(f"第{fold_id}折空缺时间行不是全0。")

    feature_to_index = {
        name: index for index, name in enumerate(feature_names)
    }

    onehot_static = [
        name
        for name in feature_names
        if (
            name.startswith("Sex_")
            or name.startswith("Marriage_")
            or name.startswith("Course_")
            or name.startswith("WHOstage_")
        )
    ]
    static_features = ["BMI", "Oppinfection", *onehot_static]
    if len(static_features) != EXPECTED_STATIC_N:
        raise ValueError(
            f"第{fold_id}折静态特征数={len(static_features)}，应为16。"
        )

    required_names = {"Age", *static_features, *DYNAMIC_FEATURES}
    missing = sorted(required_names - set(feature_names))
    if missing:
        raise ValueError(f"第{fold_id}折缺少特征：{missing}")

    static_indices = np.asarray(
        [feature_to_index[name] for name in static_features],
        dtype=np.int64,
    )
    dynamic_indices = np.asarray(
        [feature_to_index[name] for name in DYNAMIC_FEATURES],
        dtype=np.int64,
    )

    first_observed_step = np.argmax(
        common.sequence_row_mask,
        axis=1,
    ).astype(np.int64)
    if (~common.sequence_row_mask.any(axis=1)).any():
        raise ValueError("部分开发集患者没有任何历史时间行。")

    patient_local = np.arange(
        EXPECTED_DEVELOPMENT_N,
        dtype=np.int64,
    )
    static_baseline = np.asarray(
        x_development[
            patient_local,
            first_observed_step,
            :,
        ][:, static_indices],
        dtype=np.float32,
    )

    age_first = np.asarray(
        common.continuous_raw[
            common.development_idx,
            first_observed_step,
            common.age_continuous_index,
        ],
        dtype=np.float32,
    )
    baseline_age_raw = (
        age_first - TIME_STEP_YEARS[first_observed_step]
    ).astype(np.float32)
    if not np.isfinite(baseline_age_raw).all():
        raise ValueError("基线年龄存在缺失或非法值。")

    age_mean = float(
        preprocessor["scaler"].mean_[common.age_continuous_index]
    )
    age_scale = float(
        preprocessor["scaler"].scale_[common.age_continuous_index]
    )
    if (
        not np.isfinite(age_mean)
        or not np.isfinite(age_scale)
        or age_scale <= 0
    ):
        raise ValueError("年龄标准化参数无效。")

    base_dynamic = np.asarray(
        x_development[:, :, dynamic_indices],
        dtype=np.float32,
    )
    lab_raw_dev = np.asarray(
        common.continuous_raw[
            common.development_idx,
            :,
            :,
        ][:, :, common.lab_continuous_indices],
        dtype=np.float32,
    )
    enhanced_dynamic, enhanced_names = build_enhanced_dynamic_array(
        base_dynamic=base_dynamic,
        lab_raw_development=lab_raw_dev,
        sequence_row_mask=common.sequence_row_mask,
    )

    train_patient_mask = common.development_fold_id != fold_id
    validation_patient_mask = common.development_fold_id == fold_id

    train_sample_pairs = np.argwhere(
        common.prediction_origin_mask
        & train_patient_mask[:, None]
    ).astype(np.int32)
    validation_sample_pairs = np.argwhere(
        common.prediction_origin_mask
        & validation_patient_mask[:, None]
    ).astype(np.int32)

    train_long_idx = common.long_row_index_map[
        train_sample_pairs[:, 0],
        train_sample_pairs[:, 1],
    ].astype(np.int32)
    validation_long_idx = common.long_row_index_map[
        validation_sample_pairs[:, 0],
        validation_sample_pairs[:, 1],
    ].astype(np.int32)

    if np.any(train_long_idx < 0) or np.any(validation_long_idx < 0):
        raise ValueError(f"第{fold_id}折存在无效长格式行号。")
    if np.intersect1d(train_long_idx, validation_long_idx).size:
        raise ValueError(f"第{fold_id}折训练和验证记录重叠。")

    train_dataset = EnhancedPatientLandmarkDataset(
        sample_pairs=train_sample_pairs,
        enhanced_dynamic=enhanced_dynamic,
        sequence_mask=common.sequence_row_mask,
        static_baseline=static_baseline,
        baseline_age_raw=baseline_age_raw,
        event_matrix=common.future_event_matrix,
        at_risk_mask=common.future_at_risk_mask,
        age_mean=age_mean,
        age_scale=age_scale,
    )
    validation_dataset = EnhancedPatientLandmarkDataset(
        sample_pairs=validation_sample_pairs,
        enhanced_dynamic=enhanced_dynamic,
        sequence_mask=common.sequence_row_mask,
        static_baseline=static_baseline,
        baseline_age_raw=baseline_age_raw,
        event_matrix=common.future_event_matrix,
        at_risk_mask=common.future_at_risk_mask,
        age_mean=age_mean,
        age_scale=age_scale,
    )

    return {
        "fold_id": int(fold_id),
        "train_dataset": train_dataset,
        "validation_dataset": validation_dataset,
        "train_sample_pairs": train_sample_pairs,
        "validation_sample_pairs": validation_sample_pairs,
        "train_long_idx": train_long_idx,
        "validation_long_idx": validation_long_idx,
        "static_features": static_features,
        "enhanced_dynamic_features": enhanced_names,
    }


# =============================================================================
# 6. 强化LSTM模型
# =============================================================================

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


# =============================================================================
# 7. Landmark平衡生存损失
# =============================================================================

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


# =============================================================================
# 8. DataLoader
# =============================================================================

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


# =============================================================================
# 9. 指标
# =============================================================================

def hazards_to_survival(
    hazard: np.ndarray,
) -> np.ndarray:
    hazard = np.clip(
        np.asarray(hazard, dtype=np.float64),
        EPS,
        1.0 - EPS,
    )
    return np.cumprod(1.0 - hazard, axis=1).astype(np.float32)


def compute_equal_weight_landmark_metrics(
    common: CommonData,
    train_long_idx: np.ndarray,
    validation_long_idx: np.ndarray,
    validation_hazard: np.ndarray,
) -> tuple[dict[str, float], pd.DataFrame]:
    validation_survival = hazards_to_survival(validation_hazard)
    validation_risk = 1.0 - validation_survival

    train_landmarks = common.landmark_month_long[train_long_idx]
    validation_landmarks = common.landmark_month_long[validation_long_idx]

    rows = []
    for landmark_month in LANDMARK_MONTHS:
        train_mask = train_landmarks == landmark_month
        validation_mask = validation_landmarks == landmark_month

        y_train = common.y_long_all[train_long_idx[train_mask]]
        y_validation = common.y_long_all[
            validation_long_idx[validation_mask]
        ]
        survival_estimate = np.asarray(
            validation_survival[validation_mask],
            dtype=np.float64,
        )
        risk_estimate = np.asarray(
            validation_risk[validation_mask],
            dtype=np.float64,
        )

        if len(y_train) == 0 or len(y_validation) == 0:
            raise ValueError(
                f"Landmark {landmark_month}个月缺少训练或验证记录。"
            )
        max_supported = min(
            float(np.max(y_train["time"])),
            float(np.max(y_validation["time"])),
        )
        if max_supported <= METRIC_TIMES[-1]:
            raise ValueError(
                f"Landmark {landmark_month}个月不足以评价至5年。"
            )

        ibs = float(
            integrated_brier_score(
                y_train,
                y_validation,
                survival_estimate,
                METRIC_TIMES,
            )
        )
        auc_values, iauc = cumulative_dynamic_auc(
            y_train,
            y_validation,
            risk_estimate,
            METRIC_TIMES,
        )
        cindex = float(
            concordance_index_ipcw(
                y_train,
                y_validation,
                risk_estimate[:, -1],
                tau=float(METRIC_TIMES[-1]),
            )[0]
        )

        rows.append(
            {
                "landmark_month": int(landmark_month),
                "train_record_n": int(train_mask.sum()),
                "validation_record_n": int(validation_mask.sum()),
                "validation_event_n": int(y_validation["event"].sum()),
                "ibs": ibs,
                "iauc": float(iauc),
                "uno_c_index_5y": cindex,
            }
        )

    table = pd.DataFrame(rows)
    summary = {
        "mean_ibs": float(table["ibs"].mean()),
        "mean_iauc": float(table["iauc"].mean()),
        "mean_uno_c": float(table["uno_c_index_5y"].mean()),
    }
    return summary, table


# =============================================================================
# 10. 训练、快照与集成
# =============================================================================

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


def fit_fold_model(
    common: CommonData,
    fold_data: dict[str, Any],
    parameters: dict[str, Any],
    seed: int,
    device: torch.device,
    use_amp: bool,
    progress_callback=None,
) -> dict[str, Any]:
    set_random_seed(seed)
    train_loader, validation_loader = create_data_loaders(
        fold_data=fold_data,
        batch_size=int(parameters["batch_size"]),
        seed=seed,
        device=device,
    )

    model = build_model(parameters, device)
    loss_function = LandmarkBalancedSurvivalLoss(
        positive_weight=float(parameters["positive_weight"]),
        focal_gamma=float(parameters["focal_gamma"]),
        auxiliary_5y_weight=float(parameters["auxiliary_5y_weight"]),
        ranking_weight=float(parameters["ranking_weight"]),
        smoothness_weight=float(parameters["smoothness_weight"]),
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(parameters["learning_rate"]),
        weight_decay=float(parameters["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=LR_REDUCE_FACTOR,
        patience=LR_REDUCE_PATIENCE,
        min_lr=MIN_LEARNING_RATE,
    )
    grad_scaler = make_grad_scaler(enabled=use_amp)

    top_snapshots: list[dict[str, Any]] = []
    best_monitor_ibs = math.inf
    no_improvement_n = 0
    history_rows = []
    start_time = time.time()

    for epoch in range(1, MAX_EPOCHS + 1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            loss_function,
            grad_scaler,
            device,
            use_amp,
        )
        validation_loss, prediction = predict_loader(
            model,
            validation_loader,
            loss_function,
            device,
            use_amp,
        )
        metric_summary, _ = compute_equal_weight_landmark_metrics(
            common=common,
            train_long_idx=fold_data["train_long_idx"],
            validation_long_idx=fold_data["validation_long_idx"],
            validation_hazard=prediction["hazard"],
        )

        validation_ibs = metric_summary["mean_ibs"]
        validation_iauc = metric_summary["mean_iauc"]
        validation_c = metric_summary["mean_uno_c"]
        scheduler.step(validation_ibs)

        snapshot = {
            "epoch": int(epoch),
            "ibs": float(validation_ibs),
            "iauc": float(validation_iauc),
            "uno_c": float(validation_c),
            "state_dict": clone_state_dict(model),
        }
        top_snapshots.append(snapshot)
        top_snapshots.sort(
            key=lambda item: (
                item["ibs"],
                -item["iauc"],
                -item["uno_c"],
            )
        )
        top_snapshots = top_snapshots[:TOP_SNAPSHOT_N]

        current_lr = float(optimizer.param_groups[0]["lr"])
        history_rows.append(
            {
                "epoch": int(epoch),
                "train_loss": float(train_loss),
                "validation_loss": float(validation_loss),
                "validation_mean_ibs": float(validation_ibs),
                "validation_mean_iauc": float(validation_iauc),
                "validation_mean_uno_c": float(validation_c),
                "learning_rate": current_lr,
                "elapsed_seconds": float(time.time() - start_time),
            }
        )

        if progress_callback is not None:
            progress_callback(
                f"Epoch {epoch}/{MAX_EPOCHS} | "
                f"train={train_loss:.6f} | "
                f"val={validation_loss:.6f} | "
                f"IBS={validation_ibs:.6f} | "
                f"iAUC={validation_iauc:.6f} | "
                f"UnoC={validation_c:.6f} | "
                f"lr={current_lr:.2e}"
            )

        improved = (
            validation_ibs
            < best_monitor_ibs - EARLY_STOPPING_MIN_DELTA
        )
        if improved:
            best_monitor_ibs = float(validation_ibs)
            no_improvement_n = 0
        else:
            no_improvement_n += 1

        if (
            epoch >= MIN_EPOCHS
            and no_improvement_n >= EARLY_STOPPING_PATIENCE
        ):
            break

    if not top_snapshots:
        raise RuntimeError("没有保存任何有效模型快照。")

    ensemble_candidates = []
    sorted_snapshots = sorted(
        top_snapshots,
        key=lambda item: (
            item["ibs"],
            -item["iauc"],
            -item["uno_c"],
        ),
    )
    for snapshot_n in range(1, len(sorted_snapshots) + 1):
        selected_states = [
            item["state_dict"]
            for item in sorted_snapshots[:snapshot_n]
        ]
        hazard = ensemble_snapshot_predictions(
            model=model,
            snapshot_states=selected_states,
            validation_loader=validation_loader,
            loss_function=loss_function,
            device=device,
            use_amp=use_amp,
        )
        summary, table = compute_equal_weight_landmark_metrics(
            common=common,
            train_long_idx=fold_data["train_long_idx"],
            validation_long_idx=fold_data["validation_long_idx"],
            validation_hazard=hazard,
        )
        ensemble_candidates.append(
            {
                "snapshot_n": int(snapshot_n),
                "hazard": hazard,
                "summary": summary,
                "table": table,
                "states": selected_states,
                "epochs": [
                    int(item["epoch"])
                    for item in sorted_snapshots[:snapshot_n]
                ],
            }
        )

    ensemble_candidates.sort(
        key=lambda item: (
            item["summary"]["mean_ibs"],
            -item["summary"]["mean_iauc"],
            -item["summary"]["mean_uno_c"],
        )
    )
    best_ensemble = ensemble_candidates[0]
    hazard = best_ensemble["hazard"]
    survival = hazards_to_survival(hazard)
    risk = (1.0 - survival).astype(np.float32)

    if not np.isfinite(hazard).all():
        raise ValueError("验证条件风险存在NaN或无穷值。")
    if np.any((hazard < 0) | (hazard > 1)):
        raise ValueError("验证条件风险超出0～1。")
    if int(np.sum(np.diff(risk, axis=1) < -1e-7)) != 0:
        raise ValueError("验证累计风险不单调。")

    model_parameter_n = int(
        sum(parameter.numel() for parameter in model.parameters())
    )

    result = {
        "model": model,
        "snapshot_state_dicts": best_ensemble["states"],
        "snapshot_epochs": best_ensemble["epochs"],
        "selected_snapshot_n": best_ensemble["snapshot_n"],
        "history": pd.DataFrame(history_rows),
        "hazard": hazard,
        "survival": survival,
        "risk": risk,
        "metric_summary": best_ensemble["summary"],
        "landmark_metrics": best_ensemble["table"],
        "patient_local": fold_data["validation_sample_pairs"][:, 0].astype(
            np.int32
        ),
        "landmark_index": fold_data["validation_sample_pairs"][:, 1].astype(
            np.int8
        ),
        "parameter_n": model_parameter_n,
        "elapsed_seconds": float(time.time() - start_time),
    }

    del train_loader
    del validation_loader
    del optimizer
    del scheduler
    del grad_scaler
    del loss_function
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return result


# =============================================================================
# 11. 参数标准化
# =============================================================================

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


def default_device() -> tuple[torch.device, bool]:
    cuda_available = torch.cuda.is_available()
    if REQUIRE_CUDA and not cuda_available:
        raise RuntimeError("未检测到CUDA，本步骤要求GPU运行。")
    device = torch.device("cuda" if cuda_available else "cpu")
    use_amp = bool(USE_AMP and device.type == "cuda")
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return device, use_amp
