#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Step 12B：LSTM-v2 Temporal Reference Occlusion（时间维度参考值遮挡）五折 OOF 分析

研究目的
--------
1. 不重新训练模型，不读取锁定内部测试集。
2. 仅使用已经冻结的 Step10E LSTM-v2 五折OOF模型和 Step11 cross-fit calibrator。
3. 在全部Landmark（0、1、2、3、4、5年）中，系统性遮挡历史时间点，量化模型对不同历史时段的依赖。
4. 遮挡并不删除时间行，也不改变row_mask/history length/landmark：
   - 将指定历史时间点的全部85维动态输入替换为该fold训练风险集在该Landmark、
     该历史时间点的均值reference；
   - static、age、landmark context保持不变；
   - 原始cross-fit calibrator保持冻结，不针对遮挡后预测重新校准。
5. 主要结果：
   - 单一6个月历史时间点reference occlusion；
   - 相对Landmark的1年lag-band occlusion；
   - ΔiAUC、ΔIBS、ΔUno C、5年风险变化；
   - 6–60月动态AUC和Brier变化。
6. 全程保持OOF原则：每个患者仅由其held-out fold模型预测。

方法学解释
----------
- 这是post-hoc reference-value temporal occlusion，不是重新训练后的ablation。
- row_mask保持不变，因此不把“某次时间行是否存在”与该时间点内容贡献混在一起。
- 由于模型输入中包含delta和cumulative ART等已经编码历史的信息，
  遮挡单个时间点后，后续时间点的工程特征保持原值。因此本分析量化的是：
    “在其余历史信息保持不变时，该时间点表示本身的条件性预测贡献”
  而不是声称完全删除所有可能由后续累计/变化变量携带的早期历史信息。
- 真正的Baseline-only / Current-only / Full-history重新训练比较应作为后续独立的
  longitudinal incremental value分析，而不能由本步骤替代。
"""

from __future__ import annotations

import gc
import json
import math
import os
import random
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn

import matplotlib.pyplot as plt
from scipy.special import expit
from sksurv.metrics import (
    brier_score,
    concordance_index_ipcw,
    cumulative_dynamic_auc,
    integrated_brier_score,
)
from sksurv.util import Surv


# =============================================================================
# 1. 固定路径与研究配置
# =============================================================================

PROJECT_DIR = Path(
    os.getenv("CKD_LSTM_PROJECT_DIR", "__CKD_WORKDIR__")
)



INTERPRETATION_ROOT = Path(
    os.getenv(
        "CKD_MODEL_INTERPRETATION_ROOT",
        str(PROJECT_DIR / "rolling_5y_model_interpretation_final_6landmarks"),
    )
)
COMMON_FIGURE_DIR = INTERPRETATION_ROOT / "FIGURES_ALL"
COMMON_LOG_DIR = INTERPRETATION_ROOT / "LOGS"
for directory in [INTERPRETATION_ROOT, COMMON_FIGURE_DIR, COMMON_LOG_DIR]:
    directory.mkdir(parents=True, exist_ok=True)
STEP1_DIR = PROJECT_DIR / "rolling_5y_step1_new_split"
STEP2_DIR = PROJECT_DIR / "rolling_5y_step2_folds"
STEP3_DIR = PROJECT_DIR / "rolling_5y_step3_raw_features"
STEP4_DIR = PROJECT_DIR / "rolling_5y_step4_preprocessed"
STEP6_DIR = PROJECT_DIR / "rolling_5y_step6_super_landmark_data"

STEP10E_DIR = (
    PROJECT_DIR
    / "rolling_5y_step10e_lstm_v2_final_oof_selected_existing_trials"
)

STEP11_DIR = (
    PROJECT_DIR
    / "rolling_5y_step11_v2_four_model_oof_comparison_lstm_v2"
)

OUTPUT_DIR = INTERPRETATION_ROOT / "03_TEMPORAL_OCCLUSION"
CHECKPOINT_DIR = OUTPUT_DIR / "prediction_checkpoints"
FIGURE_DIR = COMMON_FIGURE_DIR
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = OUTPUT_DIR / "step12b_live_progress.log"

EXPECTED_TOTAL_N = 31911
EXPECTED_DEVELOPMENT_N = 22337
EXPECTED_HISTORY_STEP_N = 11
EXPECTED_BASE_FEATURE_N = 57
EXPECTED_STATIC_N = 16
EXPECTED_BASE_DYNAMIC_N = 40
EXPECTED_LAB_N = 15
EXPECTED_EXTRA_DYNAMIC_N = EXPECTED_LAB_N * 3
EXPECTED_ENHANCED_DYNAMIC_N = EXPECTED_BASE_DYNAMIC_N + EXPECTED_EXTRA_DYNAMIC_N
EXPECTED_LANDMARK_N = 6
EXPECTED_FUTURE_INTERVAL_N = 10
EXPECTED_DEVELOPMENT_VALID_ORIGIN_N = 100122
N_SPLITS = 5
EPS = 1e-7

LANDMARK_MONTHS = np.asarray([0, 12, 24, 36, 48, 60], dtype=np.int32)
LANDMARK_BINS = (LANDMARK_MONTHS // 6).astype(np.int64)
TIME_STEP_YEARS = np.arange(EXPECTED_HISTORY_STEP_N, dtype=np.float32) * 0.5

PRIMARY_LANDMARK_INDICES = np.arange(EXPECTED_LANDMARK_N, dtype=np.int64)
PRIMARY_LANDMARK_MONTHS = LANDMARK_MONTHS[PRIMARY_LANDMARK_INDICES]

FUTURE_END_MONTHS = np.arange(6, 61, 6, dtype=np.float64)
METRIC_TIMES = np.asarray(
    [6, 12, 18, 24, 30, 36, 42, 48, 54, 59.999],
    dtype=np.float64,
)

# 解释分析采用确定性的全精度FP32前向。
# 与Step12A IG一致：正式AMP保存预测只用于审计；perturbation前后必须使用相同数值路径。
INFERENCE_BATCH_SIZE_OVERRIDE = int(
    os.getenv("CKD_OCCLUSION_BATCH_SIZE", "0")
)
RESUME = os.getenv("CKD_OCCLUSION_RESUME", "1") != "0"
RANDOM_SEED = int(os.getenv("CKD_OCCLUSION_RANDOM_SEED", "20260812"))

# FP32 baseline与正式Step11 AMP保存风险之间允许极小数值漂移。
# 判据使用mean/p99，不让单个极端浮点值误触发。
FP32_BASELINE_MEAN_HARD_TOL = 1e-4
FP32_BASELINE_P99_HARD_TOL = 5e-4

PIPELINE_VERSION = "step12b_temporal_reference_occlusion_v1"

CALIBRATION_FILE = (
    STEP11_DIR / "four_model_hazard_calibration_parameters.csv"
)
CALIBRATED_RISK_FILE = (
    STEP11_DIR / "lstm_v2_crossfit_calibrated_oof_risk_long.npy"
)


# =============================================================================
# 2. 模型输入特征定义——与Step10E完全一致
# =============================================================================

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
# 3. 通用函数
# =============================================================================

def set_random_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    torch.cuda.manual_seed_all(int(seed))
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def require_files(paths: list[Path]) -> None:
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "以下必要文件不存在：\n" + "\n".join(missing)
        )


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


def format_duration(seconds: float) -> str:
    seconds = max(0, int(round(float(seconds))))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}小时{minutes:02d}分{seconds:02d}秒"
    if minutes:
        return f"{minutes}分{seconds:02d}秒"
    return f"{seconds}秒"


def progress_print(message: str) -> None:
    line = (
        f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
        f"{message}"
    )
    print(line, flush=True)
    with LOG_FILE.open("a", encoding="utf-8") as file:
        file.write(line + "\n")
        file.flush()


def torch_load_full(
    path: Path,
    map_location: str | torch.device = "cpu",
):
    try:
        return torch.load(
            path,
            map_location=map_location,
            weights_only=False,
        )
    except TypeError:
        return torch.load(
            path,
            map_location=map_location,
        )


def default_device() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "未检测到CUDA；Step12B建议在GPU环境运行。"
        )
    device = torch.device("cuda")
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    torch.cuda.empty_cache()
    return device


# =============================================================================
# 4. 共同数据
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

    development_idx_step1 = np.load(STEP1_DIR / "development_idx.npy").astype(np.int32)
    development_idx = np.load(STEP4_DIR / "development_idx.npy").astype(np.int32)
    fold_step2 = np.load(STEP2_DIR / "development_fold_id.npy").astype(np.int8)
    development_fold_id = np.load(STEP4_DIR / "development_fold_id.npy").astype(np.int8)

    if not np.array_equal(development_idx_step1, development_idx):
        raise ValueError("Step1与Step4开发集顺序不一致。")
    if not np.array_equal(fold_step2, development_fold_id):
        raise ValueError("Step2与Step4固定五折不一致。")

    sequence_row_mask = np.load(
        STEP4_DIR / "sequence_row_mask_development.npy"
    ).astype(bool)

    prediction_origin_mask_all = np.load(
        STEP1_DIR / "prediction_origin_mask.npy"
    ).astype(bool)
    future_event_all = np.load(STEP1_DIR / "future_event_matrix.npy").astype(np.float32)
    future_at_risk_all = np.load(
        STEP1_DIR / "future_at_risk_mask.npy"
    ).astype(bool)

    landmark_months_file = np.load(STEP1_DIR / "landmark_months.npy").astype(np.int32)
    landmark_bins_file = np.load(STEP1_DIR / "landmark_bins.npy").astype(np.int64)
    if not np.array_equal(landmark_months_file, LANDMARK_MONTHS):
        raise ValueError("Landmark月份配置与Step10E不一致。")
    if not np.array_equal(landmark_bins_file, LANDMARK_BINS):
        raise ValueError("Landmark时间行配置与Step10E不一致。")

    continuous_raw = np.load(
        STEP3_DIR / "continuous_raw_0_60.npy",
        mmap_mode="r",
    )
    feature_groups = json.loads(
        (STEP3_DIR / "feature_groups.json").read_text(encoding="utf-8")
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

    prediction_origin_mask = prediction_origin_mask_all[development_idx]
    future_event_matrix = future_event_all[development_idx]
    future_at_risk_mask = future_at_risk_all[development_idx]

    checks = [
        (development_idx.shape, (EXPECTED_DEVELOPMENT_N,), "development_idx"),
        (development_fold_id.shape, (EXPECTED_DEVELOPMENT_N,), "development_fold_id"),
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
            long_row_index_map.shape,
            (EXPECTED_DEVELOPMENT_N, EXPECTED_LANDMARK_N),
            "long_row_index_map",
        ),
    ]
    for actual, expected, name in checks:
        if actual != expected:
            raise ValueError(f"{name}形状={actual}，预期={expected}。")

    if int(prediction_origin_mask.sum()) != EXPECTED_DEVELOPMENT_VALID_ORIGIN_N:
        raise ValueError("有效患者-Landmark记录数不是100122。")
    if not np.array_equal(prediction_origin_mask, long_row_index_map >= 0):
        raise ValueError("有效Landmark与长格式映射不一致。")

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
    )


# =============================================================================
# 5. 强化纵向特征——与Step10E完全一致
# =============================================================================

def build_enhanced_dynamic_array(
    base_dynamic: np.ndarray,
    lab_raw_development: np.ndarray,
    sequence_row_mask: np.ndarray,
) -> tuple[np.ndarray, list[str]]:
    base_dynamic = np.asarray(base_dynamic, dtype=np.float32)
    lab_raw_development = np.asarray(lab_raw_development, dtype=np.float32)
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
        raise ValueError(f"基础动态特征形状={base_dynamic.shape}，预期={expected_base_shape}。")
    if lab_raw_development.shape != expected_lab_shape:
        raise ValueError(f"原始实验室形状={lab_raw_development.shape}，预期={expected_lab_shape}。")

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
        elapsed_years[~has_seen] = min((step_index + 1) * 0.5, 5.0)

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

        last_seen_value = np.where(observed_now, current_value, last_seen_value)
        last_seen_step = np.where(observed_now, step_index, last_seen_step)
        has_seen |= observed_now

    enhanced = np.concatenate(
        [base_dynamic, lab_observed_float, time_since, lab_delta],
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
    if len(names) != EXPECTED_ENHANCED_DYNAMIC_N:
        raise ValueError("强化动态特征名称数错误。")
    return enhanced, names


# =============================================================================
# 6. Fold输入数据
# =============================================================================

@dataclass
class FoldArrays:
    fold_id: int
    enhanced_dynamic: np.ndarray
    enhanced_dynamic_names: list[str]
    static_baseline: np.ndarray
    static_names: list[str]
    baseline_age_raw: np.ndarray
    age_mean: float
    age_scale: float
    # IG reference按Landmark训练风险集分别构建。
    dynamic_reference_by_landmark: dict[int, np.ndarray]
    static_reference_by_landmark: dict[int, np.ndarray]
    age_reference_by_landmark: dict[int, float]


def prepare_fold_arrays(common: CommonData, fold_id: int) -> FoldArrays:
    fold_dir = STEP4_DIR / f"fold_{fold_id}"
    x_development = np.load(fold_dir / "X_development.npy", mmap_mode="r")
    feature_names = (
        pd.read_csv(fold_dir / "feature_names.csv", encoding="utf-8-sig")["feature_name"]
        .astype(str)
        .tolist()
    )
    preprocessor = joblib.load(fold_dir / "preprocessor.joblib")

    if x_development.shape != (
        EXPECTED_DEVELOPMENT_N,
        EXPECTED_HISTORY_STEP_N,
        EXPECTED_BASE_FEATURE_N,
    ):
        raise ValueError(f"第{fold_id}折预处理张量形状错误：{x_development.shape}")
    if len(feature_names) != EXPECTED_BASE_FEATURE_N:
        raise ValueError(f"第{fold_id}折预处理特征数不是57。")

    feature_to_index = {name: index for index, name in enumerate(feature_names)}
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
    static_names = ["BMI", "Oppinfection", *onehot_static]
    if len(static_names) != EXPECTED_STATIC_N:
        raise ValueError(
            f"第{fold_id}折静态特征数={len(static_names)}，应为16。"
        )

    required_names = {"Age", *static_names, *DYNAMIC_FEATURES}
    missing = sorted(required_names - set(feature_names))
    if missing:
        raise ValueError(f"第{fold_id}折缺少模型输入特征：{missing}")

    static_indices = np.asarray(
        [feature_to_index[name] for name in static_names],
        dtype=np.int64,
    )
    dynamic_indices = np.asarray(
        [feature_to_index[name] for name in DYNAMIC_FEATURES],
        dtype=np.int64,
    )

    first_observed_step = np.argmax(common.sequence_row_mask, axis=1).astype(np.int64)
    if (~common.sequence_row_mask.any(axis=1)).any():
        raise ValueError("部分开发集患者没有任何历史时间行。")

    patient_local = np.arange(EXPECTED_DEVELOPMENT_N, dtype=np.int64)
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

    age_mean = float(preprocessor["scaler"].mean_[common.age_continuous_index])
    age_scale = float(preprocessor["scaler"].scale_[common.age_continuous_index])
    if not np.isfinite(age_mean) or not np.isfinite(age_scale) or age_scale <= 0:
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

    # -------------------------------------------------------------------------
    # IG reference：
    # - 只能使用该fold训练患者；
    # - 并且必须限定在“当前Landmark仍处于风险集”的训练患者。
    # 这样reference代表该预测时点的典型训练风险集患者，而不会混入已经
    # 在更早时间发生CKD/离开风险集的患者。
    # -------------------------------------------------------------------------
    train_patient_mask = common.development_fold_id != fold_id

    dynamic_reference_by_landmark: dict[int, np.ndarray] = {}
    static_reference_by_landmark: dict[int, np.ndarray] = {}
    age_reference_by_landmark: dict[int, float] = {}

    for landmark_index in range(EXPECTED_LANDMARK_N):
        train_origin = (
            train_patient_mask
            & common.prediction_origin_mask[:, landmark_index]
        )
        if not np.any(train_origin):
            raise ValueError(
                f"第{fold_id}折Landmark {LANDMARK_MONTHS[landmark_index]}月"
                "没有训练风险集。"
            )

        landmark_reference = np.zeros(
            (EXPECTED_HISTORY_STEP_N, EXPECTED_ENHANCED_DYNAMIC_N),
            dtype=np.float32,
        )
        max_step = int(LANDMARK_BINS[landmark_index])

        for step_index in range(max_step + 1):
            active_train_origin = (
                train_origin
                & common.sequence_row_mask[:, step_index]
            )
            if not np.any(active_train_origin):
                raise ValueError(
                    f"第{fold_id}折Landmark {LANDMARK_MONTHS[landmark_index]}月、"
                    f"历史时间步{step_index}没有训练参考记录。"
                )
            landmark_reference[step_index] = np.mean(
                enhanced_dynamic[
                    active_train_origin,
                    step_index,
                    :,
                ],
                axis=0,
                dtype=np.float64,
            ).astype(np.float32)

        dynamic_reference_by_landmark[landmark_index] = landmark_reference

        static_reference_by_landmark[landmark_index] = np.mean(
            static_baseline[train_origin],
            axis=0,
            dtype=np.float64,
        ).astype(np.float32)

        age_raw = (
            baseline_age_raw[train_origin]
            + float(LANDMARK_MONTHS[landmark_index]) / 12.0
        )
        age_standardized = (age_raw - age_mean) / age_scale
        age_reference_by_landmark[landmark_index] = float(
            np.mean(age_standardized, dtype=np.float64)
        )

    return FoldArrays(
        fold_id=int(fold_id),
        enhanced_dynamic=enhanced_dynamic,
        enhanced_dynamic_names=enhanced_names,
        static_baseline=static_baseline,
        static_names=static_names,
        baseline_age_raw=baseline_age_raw,
        age_mean=age_mean,
        age_scale=age_scale,
        dynamic_reference_by_landmark=dynamic_reference_by_landmark,
        static_reference_by_landmark=static_reference_by_landmark,
        age_reference_by_landmark=age_reference_by_landmark,
    )


# =============================================================================
# 7. Hybrid Attention LSTM——与Step10E完全一致
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

        if self.pooling_mode not in {"last_attention", "last_attention_mean"}:
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

        step_indices = torch.arange(time_n, device=dynamic_sequence.device)
        active_mask = (
            row_mask
            & (step_indices[None, :] <= landmark_bin[:, None])
        )
        if (~active_mask.any(dim=1)).any():
            raise ValueError("部分样本在Landmark前没有有效历史行。")

        time_normalized = (
            step_indices.float() / float(max(time_n - 1, 1))
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
                (step_index - previous_active).float()
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
            final_hidden = torch.cat([forward_final, backward_final], dim=1)
        else:
            recurrent_history = forward_history
            final_hidden = forward_final

        attention_logits = self.attention_score(
            torch.tanh(
                self.attention_hidden(recurrent_history)
                + self.attention_query(final_hidden).unsqueeze(1)
            )
        ).squeeze(2)
        attention_logits = attention_logits.masked_fill(~active_mask, -1e4)
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
            active_mask.long() * (step_indices[None, :] + 1)
        ).argmax(dim=1)
        batch_index = torch.arange(batch_n, device=dynamic_sequence.device)
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

        horizon_index = torch.arange(self.future_n, device=dynamic_sequence.device)
        horizon_embedding = self.horizon_embedding(
            horizon_index
        ).unsqueeze(0).expand(batch_n, -1, -1)
        context_expanded = context.unsqueeze(1).expand(-1, self.future_n, -1)
        head_input = torch.cat([context_expanded, horizon_embedding], dim=2)
        logits = self.hazard_head(head_input).squeeze(2)
        return logits + self.interval_bias.unsqueeze(0)


def build_model(parameters: dict[str, Any], device: torch.device) -> HybridAttentionLSTMSurvival:
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




def load_crossfit_calibrator(
    calibration_table: pd.DataFrame,
    fold_id: int,
    landmark_month: int,
) -> tuple[np.ndarray, float]:
    sub = calibration_table.loc[
        (calibration_table["model"] == "LSTM-v2")
        & (calibration_table["fit_scope"] == "crossfit")
        & (calibration_table["heldout_fold"] == fold_id)
        & np.isclose(calibration_table["landmark_month"], float(landmark_month))
    ].sort_values("interval_position")

    if len(sub) != EXPECTED_FUTURE_INTERVAL_N:
        raise ValueError(
            f"Fold {fold_id}, Landmark {landmark_month}月校准参数不是10行。"
        )
    expected_positions = np.arange(EXPECTED_FUTURE_INTERVAL_N)
    if not np.array_equal(sub["interval_position"].to_numpy(int), expected_positions):
        raise ValueError("校准区间顺序错误。")

    beta_values = sub["beta_common_slope"].to_numpy(float)
    if not np.allclose(beta_values, beta_values[0], rtol=0, atol=1e-10):
        raise ValueError("同一fold-landmark的校准beta不一致。")

    alpha = sub["alpha_interval"].to_numpy(dtype=np.float64)
    beta = float(beta_values[0])
    return alpha, beta


def build_model_inputs_for_sample_pairs(
    common: CommonData,
    fold_arrays: FoldArrays,
    sample_pairs: np.ndarray,
) -> dict[str, np.ndarray]:
    """
    为任意[patient_local, landmark_index]组合重建Step10E模型输入。
    用于严格的fold级模型重建审计；不构造IG baseline。
    """
    sample_pairs = np.asarray(sample_pairs, dtype=np.int64)
    if sample_pairs.ndim != 2 or sample_pairs.shape[1] != 2:
        raise ValueError("sample_pairs必须为[n,2]。")

    patient_local = sample_pairs[:, 0]
    landmark_index = sample_pairs[:, 1]
    if np.any((landmark_index < 0) | (landmark_index >= EXPECTED_LANDMARK_N)):
        raise ValueError("sample_pairs存在非法landmark_index。")

    landmark_month = LANDMARK_MONTHS[landmark_index].astype(np.float32)

    dynamic = np.asarray(
        fold_arrays.enhanced_dynamic[patient_local],
        dtype=np.float32,
    )
    row_mask = np.asarray(
        common.sequence_row_mask[patient_local],
        dtype=bool,
    )
    static = np.asarray(
        fold_arrays.static_baseline[patient_local],
        dtype=np.float32,
    )
    age_raw = (
        fold_arrays.baseline_age_raw[patient_local]
        + landmark_month / 12.0
    )
    age = (
        (age_raw - fold_arrays.age_mean)
        / fold_arrays.age_scale
    ).astype(np.float32)

    landmark_norm = (
        landmark_month / 60.0
    ).astype(np.float32)
    landmark_bin = LANDMARK_BINS[
        landmark_index
    ].astype(np.int64)

    return {
        "dynamic": dynamic,
        "row_mask": row_mask,
        "static": static,
        "age": age,
        "landmark_norm": landmark_norm,
        "landmark_bin": landmark_bin,
    }


def np_step10e_hazard_to_survival(hazard: np.ndarray) -> np.ndarray:
    """与Step10E hazards_to_survival()一致。"""
    hazard64 = np.clip(
        np.asarray(hazard, dtype=np.float64),
        EPS,
        1.0 - EPS,
    )
    return np.cumprod(
        1.0 - hazard64,
        axis=1,
    ).astype(np.float32)


def np_step11_survival_to_hazard(survival: np.ndarray) -> np.ndarray:
    """与Step11 survival_to_hazard()一致。"""
    survival64 = np.clip(
        np.asarray(survival, dtype=np.float64),
        EPS,
        1.0,
    )
    previous_survival = np.concatenate(
        [
            np.ones(
                (survival64.shape[0], 1),
                dtype=np.float64,
            ),
            survival64[:, :-1],
        ],
        axis=1,
    )
    hazard = (
        1.0
        - survival64
        / np.clip(previous_survival, EPS, 1.0)
    )
    return np.clip(
        hazard,
        EPS,
        1.0 - EPS,
    ).astype(np.float32)


def np_step11_apply_calibrator(
    hazard: np.ndarray,
    alpha: np.ndarray,
    beta: float,
) -> np.ndarray:
    """与Step11 apply_interval_hazard_calibrator()一致。"""
    probability = np.clip(
        np.asarray(hazard, dtype=np.float64),
        EPS,
        1.0 - EPS,
    )
    raw_logit = (
        np.log(probability)
        - np.log1p(-probability)
    )
    alpha64 = np.asarray(alpha, dtype=np.float64)
    calibrated = expit(
        alpha64[None, :]
        + float(beta) * raw_logit
    )
    return np.clip(
        calibrated,
        EPS,
        1.0 - EPS,
    ).astype(np.float32)


def np_step11_hazard_to_risk(hazard: np.ndarray) -> np.ndarray:
    """与Step11 hazard_to_risk()一致。"""
    hazard64 = np.clip(
        np.asarray(hazard, dtype=np.float64),
        EPS,
        1.0 - EPS,
    )
    survival = np.cumprod(
        1.0 - hazard64,
        axis=1,
    )
    return np.clip(
        1.0 - survival,
        0.0,
        1.0,
    ).astype(np.float32)




# =============================================================================
# 10. Step6结局与Step11基准预测
# =============================================================================

@dataclass
class EvaluationData:
    local_patient_idx_long: np.ndarray
    fold_id_long: np.ndarray
    landmark_index_long: np.ndarray
    landmark_month_long: np.ndarray
    analysis_time_month: np.ndarray
    event_within_60m: np.ndarray
    calibrated_risk_saved: np.ndarray


def load_evaluation_data(common: CommonData) -> EvaluationData:
    required = [
        STEP6_DIR / "development_local_patient_idx_long.npy",
        STEP6_DIR / "development_fold_id_long.npy",
        STEP6_DIR / "development_landmark_index_long.npy",
        STEP6_DIR / "development_landmark_month_long.npy",
        STEP6_DIR / "development_analysis_time_month.npy",
        STEP6_DIR / "development_event_within_60m.npy",
        CALIBRATION_FILE,
        CALIBRATED_RISK_FILE,
    ]
    require_files(required)

    local_patient_idx_long = np.load(
        STEP6_DIR / "development_local_patient_idx_long.npy"
    ).astype(np.int32)
    fold_id_long = np.load(
        STEP6_DIR / "development_fold_id_long.npy"
    ).astype(np.int8)
    landmark_index_long = np.load(
        STEP6_DIR / "development_landmark_index_long.npy"
    ).astype(np.int8)
    landmark_month_long = np.load(
        STEP6_DIR / "development_landmark_month_long.npy"
    ).astype(np.float64)
    analysis_time_month = np.load(
        STEP6_DIR / "development_analysis_time_month.npy"
    ).astype(np.float64)
    event_within_60m = np.load(
        STEP6_DIR / "development_event_within_60m.npy"
    ).astype(bool)
    calibrated_risk_saved = np.load(
        CALIBRATED_RISK_FILE,
        mmap_mode="r",
    )

    expected_long_n = EXPECTED_DEVELOPMENT_VALID_ORIGIN_N
    expected_shapes = {
        "local_patient_idx_long": (expected_long_n,),
        "fold_id_long": (expected_long_n,),
        "landmark_index_long": (expected_long_n,),
        "landmark_month_long": (expected_long_n,),
        "analysis_time_month": (expected_long_n,),
        "event_within_60m": (expected_long_n,),
        "calibrated_risk_saved": (
            expected_long_n,
            EXPECTED_FUTURE_INTERVAL_N,
        ),
    }
    actual = {
        "local_patient_idx_long": local_patient_idx_long.shape,
        "fold_id_long": fold_id_long.shape,
        "landmark_index_long": landmark_index_long.shape,
        "landmark_month_long": landmark_month_long.shape,
        "analysis_time_month": analysis_time_month.shape,
        "event_within_60m": event_within_60m.shape,
        "calibrated_risk_saved": calibrated_risk_saved.shape,
    }
    for name, expected in expected_shapes.items():
        if actual[name] != expected:
            raise ValueError(
                f"{name}形状={actual[name]}，预期={expected}。"
            )

    valid_map = common.long_row_index_map >= 0
    landmark_grid = np.broadcast_to(
        np.arange(EXPECTED_LANDMARK_N, dtype=np.int32)[None, :],
        common.long_row_index_map.shape,
    )
    mapped_rows = common.long_row_index_map[valid_map].astype(np.int64)
    if not np.array_equal(
        landmark_index_long[mapped_rows],
        landmark_grid[valid_map],
    ):
        raise ValueError(
            "Step6 landmark_index_long与development_long_row_index_map不一致。"
        )
    if not np.array_equal(
        landmark_month_long.astype(np.int32),
        LANDMARK_MONTHS[landmark_index_long],
    ):
        raise ValueError(
            "Step6 landmark月份与索引不一致。"
        )
    if not np.array_equal(
        local_patient_idx_long[mapped_rows],
        np.broadcast_to(
            np.arange(EXPECTED_DEVELOPMENT_N, dtype=np.int32)[:, None],
            common.long_row_index_map.shape,
        )[valid_map],
    ):
        raise ValueError(
            "Step6 local_patient_idx_long与development_long_row_index_map不一致。"
        )
    if not np.array_equal(
        fold_id_long,
        common.development_fold_id[local_patient_idx_long],
    ):
        raise ValueError(
            "Step6 fold_id_long与开发集固定五折不一致。"
        )

    if not np.isfinite(analysis_time_month).all():
        raise ValueError(
            "Step6 Landmark后随访时间存在NaN或无穷值。"
        )
    if np.any(analysis_time_month <= 0):
        raise ValueError(
            "Step6 Landmark后随访时间必须>0。"
        )

    return EvaluationData(
        local_patient_idx_long=local_patient_idx_long,
        fold_id_long=fold_id_long,
        landmark_index_long=landmark_index_long,
        landmark_month_long=landmark_month_long,
        analysis_time_month=analysis_time_month,
        event_within_60m=event_within_60m,
        calibrated_risk_saved=calibrated_risk_saved,
    )


# =============================================================================
# 11. 冻结模型ensemble读取
# =============================================================================

@dataclass
class SeedCheckpointPayload:
    seed: int
    parameters: dict[str, Any]
    snapshot_states: list[dict[str, torch.Tensor]]


@dataclass
class FoldCheckpointBundle:
    fold_id: int
    parameters: dict[str, Any]
    dynamic_names: list[str]
    static_names: list[str]
    seeds: list[SeedCheckpointPayload]


def load_fold_checkpoint_bundle(
    fold_id: int,
    fold_arrays: FoldArrays,
) -> FoldCheckpointBundle:
    fold_dir = STEP10E_DIR / f"fold_{fold_id}"
    checkpoint_paths = sorted(
        fold_dir.glob("seed_*_snapshot_ensemble.pt")
    )
    if not checkpoint_paths:
        raise FileNotFoundError(
            f"Fold {fold_id}没有找到seed snapshot ensemble检查点。"
        )

    reference_parameters = None
    reference_dynamic_names = None
    reference_static_names = None
    seed_payloads = []

    for checkpoint_path in checkpoint_paths:
        checkpoint = torch_load_full(
            checkpoint_path,
            map_location="cpu",
        )
        if int(checkpoint["fold_id"]) != int(fold_id):
            raise ValueError(
                f"Fold {fold_id}检查点fold_id错误：{checkpoint_path}"
            )

        parameters = dict(checkpoint["parameters"])
        dynamic_names = list(
            checkpoint["feature_names"]["enhanced_dynamic"]
        )
        static_names = list(
            checkpoint["feature_names"]["static"]
        )
        snapshot_states = list(
            checkpoint["snapshot_state_dicts"]
        )

        if not snapshot_states:
            raise ValueError(
                f"Fold {fold_id}检查点没有snapshot state：{checkpoint_path}"
            )

        if reference_parameters is None:
            reference_parameters = parameters
            reference_dynamic_names = dynamic_names
            reference_static_names = static_names
        else:
            if parameters != reference_parameters:
                raise ValueError(
                    f"Fold {fold_id}不同seed模型参数不一致。"
                )
            if dynamic_names != reference_dynamic_names:
                raise ValueError(
                    f"Fold {fold_id}不同seed动态特征顺序不一致。"
                )
            if static_names != reference_static_names:
                raise ValueError(
                    f"Fold {fold_id}不同seed静态特征顺序不一致。"
                )

        seed_payloads.append(
            SeedCheckpointPayload(
                seed=int(checkpoint["seed"]),
                parameters=parameters,
                snapshot_states=snapshot_states,
            )
        )

    if reference_dynamic_names != fold_arrays.enhanced_dynamic_names:
        raise ValueError(
            f"Fold {fold_id}检查点动态特征与当前重建数据不一致。"
        )
    if reference_static_names != fold_arrays.static_names:
        raise ValueError(
            f"Fold {fold_id}检查点静态特征与当前重建数据不一致。"
        )

    if len(reference_dynamic_names) != EXPECTED_ENHANCED_DYNAMIC_N:
        raise ValueError(
            f"Fold {fold_id}动态特征数不是85。"
        )
    if len(reference_static_names) != EXPECTED_STATIC_N:
        raise ValueError(
            f"Fold {fold_id}静态特征数不是16。"
        )

    return FoldCheckpointBundle(
        fold_id=int(fold_id),
        parameters=reference_parameters,
        dynamic_names=reference_dynamic_names,
        static_names=reference_static_names,
        seeds=seed_payloads,
    )


# =============================================================================
# 12. 时间遮挡条件
# =============================================================================

@dataclass(frozen=True)
class OcclusionSpec:
    landmark_index: int
    condition_type: str
    condition_label: str
    occluded_steps: tuple[int, ...]
    history_month: int | None
    lag_month: int | None
    lag_band_start_month: int | None
    lag_band_end_month: int | None


def build_occlusion_specs(
    landmark_index: int,
) -> tuple[list[OcclusionSpec], list[OcclusionSpec]]:
    """
    返回：
    1) single-step：逐个6月历史时间点替换为reference；
    2) lag-band：
       - current_0m：Landmark当前时间点；
       - lag_6_12m：Landmark前6和12个月；
       - lag_18_24m；
       - lag_30_36m；
       - lag_42_48m；
       - lag_54_60m。
    每个非current band最多2个6月时间点，避免“远期组因为时间点更多而天然更大”的混杂。
    """
    landmark_index = int(landmark_index)
    landmark_month = int(LANDMARK_MONTHS[landmark_index])
    max_step = int(LANDMARK_BINS[landmark_index])

    single_specs: list[OcclusionSpec] = []
    for step_index in range(max_step + 1):
        history_month = int(step_index * 6)
        lag_month = int(landmark_month - history_month)
        single_specs.append(
            OcclusionSpec(
                landmark_index=landmark_index,
                condition_type="single_step",
                condition_label=f"history_{history_month:02d}m",
                occluded_steps=(int(step_index),),
                history_month=history_month,
                lag_month=lag_month,
                lag_band_start_month=None,
                lag_band_end_month=None,
            )
        )

    lag_bands = [
        ("current_0m", 0, 0),
        ("lag_6_12m", 6, 12),
        ("lag_18_24m", 18, 24),
        ("lag_30_36m", 30, 36),
        ("lag_42_48m", 42, 48),
        ("lag_54_60m", 54, 60),
    ]

    band_specs: list[OcclusionSpec] = []
    for label, lag_start, lag_end in lag_bands:
        lags = [
            lag
            for lag in range(lag_start, lag_end + 1, 6)
            if lag <= landmark_month
        ]
        if not lags:
            continue

        steps = sorted(
            {
                int((landmark_month - lag) // 6)
                for lag in lags
                if (
                    landmark_month - lag >= 0
                    and (landmark_month - lag) % 6 == 0
                )
            }
        )
        if not steps:
            continue
        if np.any(
            (np.asarray(steps) < 0)
            | (np.asarray(steps) > max_step)
        ):
            raise ValueError(
                f"Landmark {landmark_month}月lag-band生成非法step：{steps}"
            )

        band_specs.append(
            OcclusionSpec(
                landmark_index=landmark_index,
                condition_type="lag_band",
                condition_label=label,
                occluded_steps=tuple(steps),
                history_month=None,
                lag_month=None,
                lag_band_start_month=int(lag_start),
                lag_band_end_month=int(lag_end),
            )
        )

    return single_specs, band_specs


def condition_key(steps: tuple[int, ...]) -> str:
    if not steps:
        return "baseline"
    return "steps_" + "_".join(str(int(x)) for x in steps)


# =============================================================================
# 13. FP32 reference occlusion前向
# =============================================================================

def apply_temporal_reference_occlusion(
    inputs: dict[str, np.ndarray],
    fold_arrays: FoldArrays,
    landmark_index: int,
    occluded_steps: tuple[int, ...],
) -> dict[str, np.ndarray]:
    """
    只改变dynamic_sequence指定时间点的85维内容；
    row_mask/static/age/landmark全部保持原值。
    """
    if not occluded_steps:
        return inputs

    dynamic = np.asarray(
        inputs["dynamic"],
        dtype=np.float32,
    ).copy()
    row_mask = np.asarray(
        inputs["row_mask"],
        dtype=bool,
    )
    landmark_bin = np.asarray(
        inputs["landmark_bin"],
        dtype=np.int64,
    )

    reference = fold_arrays.dynamic_reference_by_landmark[
        int(landmark_index)
    ]

    for step_index in occluded_steps:
        step_index = int(step_index)
        if (
            step_index < 0
            or step_index >= EXPECTED_HISTORY_STEP_N
        ):
            raise ValueError(
                f"非法occluded step：{step_index}"
            )

        # 只替换患者在该时间行真实存在、且位于当前Landmark之前/之内的内容。
        active = (
            row_mask[:, step_index]
            & (step_index <= landmark_bin)
        )
        if np.any(active):
            dynamic[
                active,
                step_index,
                :,
            ] = reference[
                step_index,
                :
            ]

    result = dict(inputs)
    result["dynamic"] = dynamic
    return result


def predict_one_snapshot_fp32(
    model: HybridAttentionLSTMSurvival,
    common: CommonData,
    fold_arrays: FoldArrays,
    sample_pairs: np.ndarray,
    landmark_index: int,
    occluded_steps: tuple[int, ...],
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    hazards = []

    with torch.no_grad():
        for start in range(
            0,
            len(sample_pairs),
            int(batch_size),
        ):
            end = min(
                start + int(batch_size),
                len(sample_pairs),
            )
            batch_pairs = sample_pairs[start:end]

            inputs = build_model_inputs_for_sample_pairs(
                common=common,
                fold_arrays=fold_arrays,
                sample_pairs=batch_pairs,
            )
            inputs = apply_temporal_reference_occlusion(
                inputs=inputs,
                fold_arrays=fold_arrays,
                landmark_index=int(landmark_index),
                occluded_steps=occluded_steps,
            )

            dynamic = torch.as_tensor(
                inputs["dynamic"],
                dtype=torch.float32,
                device=device,
            )
            row_mask = torch.as_tensor(
                inputs["row_mask"],
                dtype=torch.bool,
                device=device,
            )
            static = torch.as_tensor(
                inputs["static"],
                dtype=torch.float32,
                device=device,
            )
            age = torch.as_tensor(
                inputs["age"],
                dtype=torch.float32,
                device=device,
            )
            landmark_norm = torch.as_tensor(
                inputs["landmark_norm"],
                dtype=torch.float32,
                device=device,
            )
            landmark_bin = torch.as_tensor(
                inputs["landmark_bin"],
                dtype=torch.long,
                device=device,
            )

            # 解释性perturbation固定使用FP32，不使用autocast。
            logits = model(
                dynamic_sequence=dynamic,
                row_mask=row_mask,
                static_baseline=static,
                age_at_landmark=age,
                landmark_normalized=landmark_norm,
                landmark_bin=landmark_bin,
            )
            hazard = (
                torch.sigmoid(logits.float())
                .cpu()
                .numpy()
                .astype(np.float32)
            )
            hazards.append(hazard)

            del (
                dynamic,
                row_mask,
                static,
                age,
                landmark_norm,
                landmark_bin,
                logits,
            )

    return np.concatenate(
        hazards,
        axis=0,
    ).astype(np.float32)


def predict_fold_condition_fp32(
    bundle: FoldCheckpointBundle,
    common: CommonData,
    fold_arrays: FoldArrays,
    sample_pairs: np.ndarray,
    landmark_index: int,
    occluded_steps: tuple[int, ...],
    calibration_alpha: np.ndarray,
    calibration_beta: float,
    device: torch.device,
) -> np.ndarray:
    """
    snapshot内、seed间均使用NumPy float32 mean，与Step10E ensemble规则一致；
    唯一区别是解释性perturbation使用FP32前向而非AMP。
    返回10个未来半年累计风险。
    """
    if len(sample_pairs) == 0:
        raise ValueError("sample_pairs为空。")

    if not np.all(
        sample_pairs[:, 1] == int(landmark_index)
    ):
        raise ValueError(
            "predict_fold_condition_fp32要求一个调用只包含一个Landmark。"
        )

    formal_batch_size = int(
        bundle.parameters["batch_size"]
    )
    batch_size = (
        INFERENCE_BATCH_SIZE_OVERRIDE
        if INFERENCE_BATCH_SIZE_OVERRIDE > 0
        else formal_batch_size
    )

    seed_hazards = []

    for seed_payload in bundle.seeds:
        if (
            seed_payload.parameters
            != bundle.parameters
        ):
            raise ValueError(
                "seed参数与bundle参数不一致。"
            )

        model = build_model(
            bundle.parameters,
            device,
        )
        snapshot_hazards = []

        for state in seed_payload.snapshot_states:
            model.load_state_dict(
                state,
                strict=True,
            )
            model.eval()

            snapshot_hazard = (
                predict_one_snapshot_fp32(
                    model=model,
                    common=common,
                    fold_arrays=fold_arrays,
                    sample_pairs=sample_pairs,
                    landmark_index=int(landmark_index),
                    occluded_steps=occluded_steps,
                    batch_size=batch_size,
                    device=device,
                )
            )
            snapshot_hazards.append(
                snapshot_hazard
            )

        seed_hazard = np.mean(
            np.stack(
                snapshot_hazards,
                axis=0,
            ),
            axis=0,
        ).astype(np.float32)
        seed_hazards.append(seed_hazard)

        del model
        gc.collect()
        torch.cuda.empty_cache()

    ensemble_hazard = np.mean(
        np.stack(
            seed_hazards,
            axis=0,
        ),
        axis=0,
    ).astype(np.float32)

    # 完整复现Step10E保存survival -> Step11反推hazard -> cross-fit校准。
    survival_saved_like = (
        np_step10e_hazard_to_survival(
            ensemble_hazard
        )
    )
    step11_raw_hazard = (
        np_step11_survival_to_hazard(
            survival_saved_like
        )
    )
    calibrated_hazard = (
        np_step11_apply_calibrator(
            hazard=step11_raw_hazard,
            alpha=calibration_alpha,
            beta=calibration_beta,
        )
    )
    calibrated_risk = (
        np_step11_hazard_to_risk(
            calibrated_hazard
        )
    )

    if calibrated_risk.shape != (
        len(sample_pairs),
        EXPECTED_FUTURE_INTERVAL_N,
    ):
        raise ValueError(
            "遮挡后累计风险形状错误。"
        )
    if not np.isfinite(
        calibrated_risk
    ).all():
        raise ValueError(
            "遮挡后累计风险出现NaN或无穷值。"
        )
    if np.any(
        (calibrated_risk < -EPS)
        | (calibrated_risk > 1.0 + EPS)
    ):
        raise ValueError(
            "遮挡后累计风险超出0～1。"
        )
    if np.any(
        np.diff(
            calibrated_risk,
            axis=1,
        )
        < -1e-7
    ):
        raise ValueError(
            "遮挡后累计风险不单调。"
        )

    return calibrated_risk.astype(
        np.float32
    )


# =============================================================================
# 14. checkpoint/resume
# =============================================================================

def initialize_checkpoint_metadata() -> None:
    metadata_path = (
        CHECKPOINT_DIR
        / "checkpoint_metadata.json"
    )
    expected = {
        "pipeline_version": PIPELINE_VERSION,
        "primary_landmark_months": (
            PRIMARY_LANDMARK_MONTHS.tolist()
        ),
        "inference_precision": "FP32",
        "reference_scope": (
            "heldout-fold-excluded training risk set, "
            "landmark-specific and history-step-specific"
        ),
        "row_mask_held_fixed": True,
        "calibrator_refit_after_occlusion": False,
    }

    if metadata_path.exists():
        observed = json.loads(
            metadata_path.read_text(
                encoding="utf-8"
            )
        )
        if observed != expected:
            raise ValueError(
                "已有Step12B checkpoint metadata与当前代码不一致。"
                "请不要混用不同版本的occlusion预测。"
            )
        return

    existing_predictions = list(
        CHECKPOINT_DIR.rglob(
            "*_calibrated_risk.npy"
        )
    )
    if existing_predictions:
        raise ValueError(
            "检测到已有Step12B预测checkpoint但缺少版本metadata；"
            "为避免混用旧结果，请先核对/移动旧Step12B输出目录。"
        )

    save_json(
        expected,
        metadata_path,
    )


def landmark_checkpoint_dir(
    fold_id: int,
    landmark_month: int,
) -> Path:
    path = (
        CHECKPOINT_DIR
        / f"fold_{int(fold_id)}"
        / f"landmark_{int(landmark_month)}m"
    )
    path.mkdir(
        parents=True,
        exist_ok=True,
    )
    return path


def condition_prediction_path(
    fold_id: int,
    landmark_month: int,
    steps: tuple[int, ...],
) -> Path:
    return (
        landmark_checkpoint_dir(
            fold_id,
            landmark_month,
        )
        / f"{condition_key(steps)}_calibrated_risk.npy"
    )


def get_or_predict_condition(
    bundle: FoldCheckpointBundle,
    common: CommonData,
    fold_arrays: FoldArrays,
    sample_pairs: np.ndarray,
    patient_local: np.ndarray,
    landmark_index: int,
    occluded_steps: tuple[int, ...],
    calibration_alpha: np.ndarray,
    calibration_beta: float,
    device: torch.device,
) -> np.ndarray:
    landmark_month = int(
        LANDMARK_MONTHS[
            int(landmark_index)
        ]
    )
    fold_id = int(
        bundle.fold_id
    )
    checkpoint_dir = landmark_checkpoint_dir(
        fold_id,
        landmark_month,
    )
    patient_path = (
        checkpoint_dir
        / "patient_local.npy"
    )
    prediction_path = condition_prediction_path(
        fold_id,
        landmark_month,
        occluded_steps,
    )

    if RESUME and prediction_path.exists():
        if not patient_path.exists():
            raise FileNotFoundError(
                f"已有预测但缺少患者顺序文件：{patient_path}"
            )
        saved_patient = np.load(
            patient_path
        ).astype(np.int32)
        if not np.array_equal(
            saved_patient,
            patient_local.astype(np.int32),
        ):
            raise ValueError(
                f"Fold {fold_id}, Landmark {landmark_month}月："
                "已有checkpoint患者顺序与当前不一致。"
            )
        loaded = np.load(
            prediction_path
        ).astype(np.float32)
        if loaded.shape != (
            len(patient_local),
            EXPECTED_FUTURE_INTERVAL_N,
        ):
            raise ValueError(
                f"已有预测形状错误：{prediction_path}"
            )
        return loaded

    risk = predict_fold_condition_fp32(
        bundle=bundle,
        common=common,
        fold_arrays=fold_arrays,
        sample_pairs=sample_pairs,
        landmark_index=int(landmark_index),
        occluded_steps=occluded_steps,
        calibration_alpha=calibration_alpha,
        calibration_beta=calibration_beta,
        device=device,
    )

    if not patient_path.exists():
        np.save(
            patient_path,
            patient_local.astype(np.int32),
        )
    np.save(
        prediction_path,
        risk,
    )
    return risk


# =============================================================================
# 15. 生存评价——与Step11一致
# =============================================================================

def build_survival_array(
    event: np.ndarray,
    time_month: np.ndarray,
) -> np.ndarray:
    event = np.asarray(
        event,
        dtype=bool,
    )
    time_month = np.asarray(
        time_month,
        dtype=np.float64,
    )
    if event.shape != time_month.shape:
        raise ValueError(
            "事件和随访时间形状不一致。"
        )
    if not np.isfinite(
        time_month
    ).all():
        raise ValueError(
            "随访时间存在NaN或无穷值。"
        )
    if np.any(
        time_month <= 0
    ):
        raise ValueError(
            "Landmark后随访时间必须>0。"
        )
    return Surv.from_arrays(
        event=event,
        time=time_month,
    )


def evaluate_survival_predictions(
    survival_reference: np.ndarray,
    risk_matrix: np.ndarray,
) -> dict[str, Any]:
    risk_matrix = np.asarray(
        risk_matrix,
        dtype=np.float64,
    )
    if risk_matrix.ndim != 2:
        raise ValueError(
            "risk_matrix必须为二维。"
        )
    if risk_matrix.shape[1] != EXPECTED_FUTURE_INTERVAL_N:
        raise ValueError(
            "risk_matrix未来区间数不是10。"
        )

    survival_matrix = 1.0 - risk_matrix

    c_index = float(
        concordance_index_ipcw(
            survival_reference,
            survival_reference,
            risk_matrix[:, -1],
            tau=float(METRIC_TIMES[-1]),
        )[0]
    )

    dynamic_auc, mean_auc = (
        cumulative_dynamic_auc(
            survival_reference,
            survival_reference,
            risk_matrix,
            METRIC_TIMES,
        )
    )

    _, brier_values = brier_score(
        survival_reference,
        survival_reference,
        survival_matrix,
        METRIC_TIMES,
    )

    ibs = float(
        integrated_brier_score(
            survival_reference,
            survival_reference,
            survival_matrix,
            METRIC_TIMES,
        )
    )

    return {
        "uno_c_index_5y": c_index,
        "integrated_dynamic_auc": float(
            mean_auc
        ),
        "integrated_brier": ibs,
        "dynamic_auc": np.asarray(
            dynamic_auc,
            dtype=np.float64,
        ),
        "brier": np.asarray(
            brier_values,
            dtype=np.float64,
        ),
    }


# =============================================================================
# 16. 主结果表构建
# =============================================================================

def metric_row(
    spec: OcclusionSpec,
    landmark_month: int,
    risk_set_n: int,
    event_n: int,
    baseline_metrics: dict[str, Any],
    occluded_metrics: dict[str, Any],
    baseline_saved_metrics: dict[str, Any],
    patient_delta_risk5: np.ndarray,
    active_origin_n: int,
    active_step_pair_n: int,
) -> dict[str, Any]:
    baseline_c = float(
        baseline_metrics["uno_c_index_5y"]
    )
    baseline_iauc = float(
        baseline_metrics["integrated_dynamic_auc"]
    )
    baseline_ibs = float(
        baseline_metrics["integrated_brier"]
    )

    occ_c = float(
        occluded_metrics["uno_c_index_5y"]
    )
    occ_iauc = float(
        occluded_metrics["integrated_dynamic_auc"]
    )
    occ_ibs = float(
        occluded_metrics["integrated_brier"]
    )

    delta_risk5 = np.asarray(
        patient_delta_risk5,
        dtype=np.float64,
    )

    return {
        "condition_type": spec.condition_type,
        "condition_label": spec.condition_label,
        "landmark_index": int(spec.landmark_index),
        "landmark_month": int(landmark_month),
        "landmark_year": float(
            landmark_month / 12.0
        ),
        "history_month": spec.history_month,
        "lag_month": spec.lag_month,
        "lag_band_start_month": spec.lag_band_start_month,
        "lag_band_end_month": spec.lag_band_end_month,
        "occluded_step_n": int(
            len(spec.occluded_steps)
        ),
        "occluded_steps": ",".join(
            str(int(x))
            for x in spec.occluded_steps
        ),
        "risk_set_n": int(risk_set_n),
        "future_5y_event_n": int(event_n),
        "active_origin_n": int(active_origin_n),
        "active_origin_fraction": float(
            active_origin_n / max(risk_set_n, 1)
        ),
        "active_step_pair_n": int(active_step_pair_n),
        "active_step_pair_fraction": float(
            active_step_pair_n
            / max(
                risk_set_n * len(spec.occluded_steps),
                1,
            )
        ),

        "saved_step11_baseline_uno_c": float(
            baseline_saved_metrics[
                "uno_c_index_5y"
            ]
        ),
        "saved_step11_baseline_iAUC": float(
            baseline_saved_metrics[
                "integrated_dynamic_auc"
            ]
        ),
        "saved_step11_baseline_IBS": float(
            baseline_saved_metrics[
                "integrated_brier"
            ]
        ),

        "fp32_baseline_uno_c": baseline_c,
        "fp32_baseline_iAUC": baseline_iauc,
        "fp32_baseline_IBS": baseline_ibs,

        "occluded_uno_c": occ_c,
        "occluded_iAUC": occ_iauc,
        "occluded_IBS": occ_ibs,

        # 正值统一表示“遮挡后性能变差”。
        "uno_c_loss": baseline_c - occ_c,
        "iAUC_loss": baseline_iauc - occ_iauc,
        "IBS_increase": occ_ibs - baseline_ibs,

        # 同时保存原始方向差。
        "occluded_minus_baseline_uno_c": occ_c - baseline_c,
        "occluded_minus_baseline_iAUC": occ_iauc - baseline_iauc,
        "occluded_minus_baseline_IBS": occ_ibs - baseline_ibs,

        "mean_delta_risk5": float(
            np.mean(delta_risk5)
        ),
        "median_delta_risk5": float(
            np.median(delta_risk5)
        ),
        "mean_abs_delta_risk5": float(
            np.mean(
                np.abs(delta_risk5)
            )
        ),
        "median_abs_delta_risk5": float(
            np.median(
                np.abs(delta_risk5)
            )
        ),
        "p95_abs_delta_risk5": float(
            np.quantile(
                np.abs(delta_risk5),
                0.95,
            )
        ),
    }


def horizon_rows(
    spec: OcclusionSpec,
    landmark_month: int,
    baseline_metrics: dict[str, Any],
    occluded_metrics: dict[str, Any],
) -> list[dict[str, Any]]:
    rows = []
    for horizon_index, horizon_month in enumerate(
        FUTURE_END_MONTHS
    ):
        baseline_auc = float(
            baseline_metrics["dynamic_auc"][
                horizon_index
            ]
        )
        occluded_auc = float(
            occluded_metrics["dynamic_auc"][
                horizon_index
            ]
        )
        baseline_brier = float(
            baseline_metrics["brier"][
                horizon_index
            ]
        )
        occluded_brier = float(
            occluded_metrics["brier"][
                horizon_index
            ]
        )

        rows.append(
            {
                "condition_type": spec.condition_type,
                "condition_label": spec.condition_label,
                "landmark_index": int(
                    spec.landmark_index
                ),
                "landmark_month": int(
                    landmark_month
                ),
                "landmark_year": float(
                    landmark_month / 12.0
                ),
                "history_month": spec.history_month,
                "lag_month": spec.lag_month,
                "lag_band_start_month": (
                    spec.lag_band_start_month
                ),
                "lag_band_end_month": (
                    spec.lag_band_end_month
                ),
                "horizon_position": int(
                    horizon_index
                ),
                "horizon_month": float(
                    horizon_month
                ),
                "horizon_year": float(
                    horizon_month / 12.0
                ),
                "baseline_dynamic_auc": baseline_auc,
                "occluded_dynamic_auc": occluded_auc,
                "dynamic_auc_loss": (
                    baseline_auc
                    - occluded_auc
                ),
                "baseline_brier": baseline_brier,
                "occluded_brier": occluded_brier,
                "brier_increase": (
                    occluded_brier
                    - baseline_brier
                ),
            }
        )
    return rows


# =============================================================================
# 17. 图形
# =============================================================================

def configure_plot_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "text.color": "black",
            "axes.labelcolor": "black",
            "axes.edgecolor": "black",
            "axes.titlecolor": "black",
            "xtick.color": "black",
            "ytick.color": "black",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save_figure(
    fig: plt.Figure,
    stem: str,
) -> None:
    for extension in [
        "png",
        "pdf",
        "svg",
    ]:
        kwargs = {
            "bbox_inches": "tight",
            "facecolor": "white",
        }
        if extension == "png":
            kwargs["dpi"] = 600
        fig.savefig(
            FIGURE_DIR
            / f"Step12B_{stem}.{extension}",
            **kwargs,
        )
    plt.close(fig)


def plot_single_step_metric(
    single_df: pd.DataFrame,
    y_column: str,
    y_label: str,
    stem: str,
) -> None:
    fig, ax = plt.subplots(
        figsize=(7.4, 5.4)
    )

    for landmark_index in PRIMARY_LANDMARK_INDICES:
        sub = single_df.loc[
            single_df["landmark_index"]
            == int(landmark_index)
        ].sort_values(
            "lag_month"
        )
        if sub.empty:
            continue
        landmark_year = (
            LANDMARK_MONTHS[
                int(landmark_index)
            ]
            / 12.0
        )
        label = (
            "Baseline"
            if landmark_year == 0
            else f"ART year {landmark_year:.0f}"
        )
        ax.plot(
            sub["lag_month"],
            sub[y_column],
            marker="o",
            linewidth=1.8,
            label=label,
        )

    ax.axhline(
        0.0,
        linewidth=1.0,
        linestyle="--",
    )
    ax.set_xlabel(
        "Months before prediction landmark"
    )
    ax.set_ylabel(y_label)
    ax.set_title(
        "Single-timepoint temporal reference occlusion"
    )
    ax.grid(
        alpha=0.20,
    )
    ax.legend(
        frameon=False,
    )
    fig.tight_layout()
    save_figure(
        fig,
        stem,
    )


def plot_lag_band_metric(
    band_df: pd.DataFrame,
    y_column: str,
    y_label: str,
    stem: str,
) -> None:
    order = [
        "current_0m",
        "lag_6_12m",
        "lag_18_24m",
        "lag_30_36m",
        "lag_42_48m",
        "lag_54_60m",
    ]
    label_map = {
        "current_0m": "Current",
        "lag_6_12m": "6–12",
        "lag_18_24m": "18–24",
        "lag_30_36m": "30–36",
        "lag_42_48m": "42–48",
        "lag_54_60m": "54–60",
    }

    fig, ax = plt.subplots(
        figsize=(8.2, 5.4)
    )

    for landmark_index in PRIMARY_LANDMARK_INDICES:
        sub = band_df.loc[
            band_df["landmark_index"]
            == int(landmark_index)
        ].copy()
        if sub.empty:
            continue
        sub["plot_position"] = sub[
            "condition_label"
        ].map(
            {
                key: i
                for i, key in enumerate(order)
            }
        )
        sub = sub.dropna(
            subset=["plot_position"]
        ).sort_values(
            "plot_position"
        )

        landmark_year = (
            LANDMARK_MONTHS[
                int(landmark_index)
            ]
            / 12.0
        )
        label = (
            "Baseline"
            if landmark_year == 0
            else f"ART year {landmark_year:.0f}"
        )
        ax.plot(
            sub["plot_position"],
            sub[y_column],
            marker="o",
            linewidth=1.8,
            label=label,
        )

    ax.axhline(
        0.0,
        linewidth=1.0,
        linestyle="--",
    )
    ax.set_xticks(
        np.arange(
            len(order)
        )
    )
    ax.set_xticklabels(
        [
            label_map[key]
            for key in order
        ]
    )
    ax.set_xlabel(
        "Occluded history lag band (months before landmark)"
    )
    ax.set_ylabel(
        y_label
    )
    ax.set_title(
        "Lag-band temporal reference occlusion"
    )
    ax.grid(
        alpha=0.20,
    )
    ax.legend(
        frameon=False,
    )
    fig.tight_layout()
    save_figure(
        fig,
        stem,
    )


# =============================================================================
# 18. 主流程
# =============================================================================

def run() -> None:
    total_start = time.time()
    set_random_seed(
        RANDOM_SEED
    )
    configure_plot_style()
    device = default_device()

    required = [
        STEP10E_DIR / "lstm_v2_summary.json",
        CALIBRATION_FILE,
        CALIBRATED_RISK_FILE,
    ]
    require_files(required)

    lstm_summary = json.loads(
        (
            STEP10E_DIR
            / "lstm_v2_summary.json"
        ).read_text(
            encoding="utf-8"
        )
    )
    if bool(
        lstm_summary.get(
            "locked_test_used",
            False,
        )
    ):
        raise ValueError(
            "Step10E摘要显示读取过锁定测试集。"
        )

    calibration_table = pd.read_csv(
        CALIBRATION_FILE,
        encoding="utf-8-sig",
    )

    common = load_common_data()
    evaluation = load_evaluation_data(
        common
    )
    initialize_checkpoint_metadata()

    print(
        "\n"
        + "=" * 110
    )
    print(
        "Step 12B：LSTM-v2 Temporal Reference Occlusion 五折OOF分析"
    )
    print(
        "=" * 110
    )
    print(
        "运行设备：",
        device,
    )
    print(
        "主Landmark（月）：",
        PRIMARY_LANDMARK_MONTHS.tolist(),
    )
    print(
        "遮挡方式：指定历史时间点85维动态输入 -> 当前fold训练风险集reference"
    )
    print(
        "row_mask/static/age/landmark：保持不变"
    )
    print(
        "calibrator：使用原Step11 cross-fit calibrator，不重新校准"
    )
    print(
        "perturbation前向：FP32；baseline与occlusion使用完全相同数值路径"
    )
    print(
        "锁定测试集：未读取"
    )
    print(
        "输出目录：",
        OUTPUT_DIR,
    )
    print(
        "=" * 110
    )

    # -------------------------------------------------------------------------
    # 每个主Landmark需要的条件。
    # -------------------------------------------------------------------------
    specs_by_landmark: dict[
        int,
        dict[str, list[OcclusionSpec]],
    ] = {}
    unique_steps_by_landmark: dict[
        int,
        list[tuple[int, ...]],
    ] = {}

    for landmark_index in PRIMARY_LANDMARK_INDICES:
        single_specs, band_specs = (
            build_occlusion_specs(
                int(landmark_index)
            )
        )
        specs_by_landmark[
            int(landmark_index)
        ] = {
            "single": single_specs,
            "band": band_specs,
        }

        unique_steps = sorted(
            {
                spec.occluded_steps
                for spec in (
                    single_specs
                    + band_specs
                )
            },
            key=lambda x: (
                len(x),
                x,
            ),
        )
        unique_steps_by_landmark[
            int(landmark_index)
        ] = unique_steps

    # -------------------------------------------------------------------------
    # 全局prediction容器：只存主Landmark。
    # -------------------------------------------------------------------------
    baseline_fp32_long = np.full(
        (
            EXPECTED_DEVELOPMENT_VALID_ORIGIN_N,
            EXPECTED_FUTURE_INTERVAL_N,
        ),
        np.nan,
        dtype=np.float32,
    )

    condition_risk_long: dict[
        tuple[int, tuple[int, ...]],
        np.ndarray,
    ] = {}

    for landmark_index in PRIMARY_LANDMARK_INDICES:
        for steps in unique_steps_by_landmark[
            int(landmark_index)
        ]:
            condition_risk_long[
                (
                    int(landmark_index),
                    tuple(steps),
                )
            ] = np.full(
                (
                    EXPECTED_DEVELOPMENT_VALID_ORIGIN_N,
                    EXPECTED_FUTURE_INTERVAL_N,
                ),
                np.nan,
                dtype=np.float32,
            )

    audit_rows = []

    # -------------------------------------------------------------------------
    # Fold级OOF预测。
    # -------------------------------------------------------------------------
    for fold_id in range(
        N_SPLITS
    ):
        progress_print(
            f"Fold {fold_id}：准备fold-specific输入和训练风险集reference。"
        )

        fold_arrays = prepare_fold_arrays(
            common,
            fold_id,
        )
        bundle = load_fold_checkpoint_bundle(
            fold_id,
            fold_arrays,
        )

        for landmark_index in PRIMARY_LANDMARK_INDICES:
            landmark_index = int(
                landmark_index
            )
            landmark_month = int(
                LANDMARK_MONTHS[
                    landmark_index
                ]
            )

            patient_mask = (
                common.development_fold_id
                == int(fold_id)
            ) & (
                common.prediction_origin_mask[
                    :,
                    landmark_index,
                ]
            )
            patients = np.where(
                patient_mask
            )[0].astype(
                np.int32
            )

            if len(patients) == 0:
                raise ValueError(
                    f"Fold {fold_id}, Landmark {landmark_month}月没有OOF origins。"
                )

            sample_pairs = np.column_stack(
                [
                    patients,
                    np.full(
                        len(patients),
                        landmark_index,
                        dtype=np.int32,
                    ),
                ]
            ).astype(
                np.int32
            )

            long_rows = common.long_row_index_map[
                patients,
                landmark_index,
            ].astype(
                np.int64
            )
            if np.any(
                long_rows < 0
            ):
                raise ValueError(
                    f"Fold {fold_id}, Landmark {landmark_month}月存在非法long row。"
                )
            if not np.all(
                evaluation.landmark_index_long[
                    long_rows
                ]
                == landmark_index
            ):
                raise ValueError(
                    "patient-landmark到Step6长表映射错误。"
                )

            alpha, beta = load_crossfit_calibrator(
                calibration_table,
                fold_id=fold_id,
                landmark_month=landmark_month,
            )

            # -------------------------------------------------------------
            # FP32 baseline：同一数值路径作为所有occlusion的paired reference。
            # -------------------------------------------------------------
            progress_print(
                f"Fold {fold_id} | Landmark {landmark_month}月："
                f"baseline FP32前向，OOF origins={len(patients)}。"
            )
            baseline_risk = get_or_predict_condition(
                bundle=bundle,
                common=common,
                fold_arrays=fold_arrays,
                sample_pairs=sample_pairs,
                patient_local=patients,
                landmark_index=landmark_index,
                occluded_steps=tuple(),
                calibration_alpha=alpha,
                calibration_beta=beta,
                device=device,
            )
            baseline_fp32_long[
                long_rows,
                :,
            ] = baseline_risk

            saved_baseline = np.asarray(
                evaluation.calibrated_risk_saved[
                    long_rows,
                    :,
                ],
                dtype=np.float32,
            )
            baseline_diff = np.abs(
                baseline_risk.astype(
                    np.float64
                )
                - saved_baseline.astype(
                    np.float64
                )
            )
            mean_diff = float(
                np.mean(
                    baseline_diff
                )
            )
            p99_diff = float(
                np.quantile(
                    baseline_diff,
                    0.99,
                )
            )
            max_diff = float(
                np.max(
                    baseline_diff
                )
            )
            if (
                mean_diff
                > FP32_BASELINE_MEAN_HARD_TOL
                or p99_diff
                > FP32_BASELINE_P99_HARD_TOL
            ):
                raise ValueError(
                    f"Fold {fold_id}, Landmark {landmark_month}月："
                    "FP32 baseline与正式Step11保存风险漂移异常，"
                    f"mean={mean_diff:.3e}, p99={p99_diff:.3e}, max={max_diff:.3e}。"
                )

            audit_rows.append(
                {
                    "fold_id": int(
                        fold_id
                    ),
                    "landmark_index": int(
                        landmark_index
                    ),
                    "landmark_month": int(
                        landmark_month
                    ),
                    "origin_n": int(
                        len(patients)
                    ),
                    "mean_abs_risk_diff_vs_saved_step11": mean_diff,
                    "p99_abs_risk_diff_vs_saved_step11": p99_diff,
                    "max_abs_risk_diff_vs_saved_step11": max_diff,
                }
            )
            progress_print(
                f"Fold {fold_id} | Landmark {landmark_month}月："
                f"baseline审计通过 | mean={mean_diff:.3e} | "
                f"p99={p99_diff:.3e} | max={max_diff:.3e}"
            )

            # -------------------------------------------------------------
            # 所有unique temporal occlusion。
            # -------------------------------------------------------------
            unique_steps = unique_steps_by_landmark[
                landmark_index
            ]

            for condition_position, steps in enumerate(
                unique_steps,
                start=1,
            ):
                history_months = [
                    int(step * 6)
                    for step in steps
                ]
                progress_print(
                    f"Fold {fold_id} | Landmark {landmark_month}月 | "
                    f"condition {condition_position}/{len(unique_steps)} | "
                    f"occlude history months={history_months}"
                )

                condition_risk = get_or_predict_condition(
                    bundle=bundle,
                    common=common,
                    fold_arrays=fold_arrays,
                    sample_pairs=sample_pairs,
                    patient_local=patients,
                    landmark_index=landmark_index,
                    occluded_steps=tuple(
                        steps
                    ),
                    calibration_alpha=alpha,
                    calibration_beta=beta,
                    device=device,
                )

                condition_risk_long[
                    (
                        landmark_index,
                        tuple(steps),
                    )
                ][
                    long_rows,
                    :,
                ] = condition_risk

        del bundle
        del fold_arrays
        gc.collect()
        torch.cuda.empty_cache()

    # -------------------------------------------------------------------------
    # OOF完整性审计。
    # -------------------------------------------------------------------------
    primary_long_mask = np.isin(
        evaluation.landmark_index_long,
        PRIMARY_LANDMARK_INDICES,
    )

    if not np.isfinite(
        baseline_fp32_long[
            primary_long_mask
        ]
    ).all():
        raise ValueError(
            "主Landmark FP32 baseline存在未填充预测。"
        )

    for (
        landmark_index,
        steps,
    ), risk_array in condition_risk_long.items():
        landmark_rows = (
            evaluation.landmark_index_long
            == int(landmark_index)
        )
        if not np.isfinite(
            risk_array[
                landmark_rows
            ]
        ).all():
            raise ValueError(
                f"Landmark {LANDMARK_MONTHS[landmark_index]}月，"
                f"occluded_steps={steps}存在未填充OOF预测。"
            )

    audit_df = pd.DataFrame(
        audit_rows
    )
    audit_df.to_csv(
        OUTPUT_DIR
        / "fp32_baseline_vs_saved_step11_audit.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # -------------------------------------------------------------------------
    # Landmark级指标。
    # -------------------------------------------------------------------------
    metric_rows = []
    horizon_metric_rows = []
    patient_risk_rows = []
    baseline_metric_rows = []

    for landmark_index in PRIMARY_LANDMARK_INDICES:
        landmark_index = int(
            landmark_index
        )
        landmark_month = int(
            LANDMARK_MONTHS[
                landmark_index
            ]
        )
        rows = np.where(
            evaluation.landmark_index_long
            == landmark_index
        )[0]

        survival_reference = build_survival_array(
            evaluation.event_within_60m[
                rows
            ],
            evaluation.analysis_time_month[
                rows
            ],
        )

        saved_baseline_matrix = np.asarray(
            evaluation.calibrated_risk_saved[
                rows,
                :,
            ],
            dtype=np.float32,
        )
        fp32_baseline_matrix = baseline_fp32_long[
            rows,
            :
        ]

        saved_baseline_metrics = (
            evaluate_survival_predictions(
                survival_reference,
                saved_baseline_matrix,
            )
        )
        baseline_metrics = (
            evaluate_survival_predictions(
                survival_reference,
                fp32_baseline_matrix,
            )
        )

        baseline_metric_rows.append(
            {
                "landmark_index": int(
                    landmark_index
                ),
                "landmark_month": int(
                    landmark_month
                ),
                "landmark_year": float(
                    landmark_month / 12.0
                ),
                "risk_set_n": int(
                    len(rows)
                ),
                "future_5y_event_n": int(
                    evaluation.event_within_60m[
                        rows
                    ].sum()
                ),
                "saved_step11_uno_c": float(
                    saved_baseline_metrics[
                        "uno_c_index_5y"
                    ]
                ),
                "fp32_baseline_uno_c": float(
                    baseline_metrics[
                        "uno_c_index_5y"
                    ]
                ),
                "saved_step11_iAUC": float(
                    saved_baseline_metrics[
                        "integrated_dynamic_auc"
                    ]
                ),
                "fp32_baseline_iAUC": float(
                    baseline_metrics[
                        "integrated_dynamic_auc"
                    ]
                ),
                "saved_step11_IBS": float(
                    saved_baseline_metrics[
                        "integrated_brier"
                    ]
                ),
                "fp32_baseline_IBS": float(
                    baseline_metrics[
                        "integrated_brier"
                    ]
                ),
            }
        )

        specs = (
            specs_by_landmark[
                landmark_index
            ]["single"]
            + specs_by_landmark[
                landmark_index
            ]["band"]
        )

        for spec in specs:
            occluded_matrix = (
                condition_risk_long[
                    (
                        landmark_index,
                        spec.occluded_steps,
                    )
                ][
                    rows,
                    :,
                ]
            )
            occluded_metrics = (
                evaluate_survival_predictions(
                    survival_reference,
                    occluded_matrix,
                )
            )

            delta_risk5 = (
                occluded_matrix[
                    :,
                    -1,
                ].astype(
                    np.float64
                )
                - fp32_baseline_matrix[
                    :,
                    -1,
                ].astype(
                    np.float64
                )
            )

            patient_local_for_rows = (
                evaluation.local_patient_idx_long[
                    rows
                ].astype(
                    np.int32
                )
            )
            active_occlusion_matrix = (
                common.sequence_row_mask[
                    patient_local_for_rows
                ][
                    :,
                    list(
                        spec.occluded_steps
                    ),
                ]
            )
            if active_occlusion_matrix.ndim == 1:
                active_occlusion_matrix = (
                    active_occlusion_matrix[
                        :,
                        None,
                    ]
                )
            active_origin_n = int(
                np.any(
                    active_occlusion_matrix,
                    axis=1,
                ).sum()
            )
            active_step_pair_n = int(
                active_occlusion_matrix.sum()
            )

            metric_rows.append(
                metric_row(
                    spec=spec,
                    landmark_month=landmark_month,
                    risk_set_n=len(rows),
                    event_n=int(
                        evaluation.event_within_60m[
                            rows
                        ].sum()
                    ),
                    baseline_metrics=baseline_metrics,
                    occluded_metrics=occluded_metrics,
                    baseline_saved_metrics=saved_baseline_metrics,
                    patient_delta_risk5=delta_risk5,
                    active_origin_n=active_origin_n,
                    active_step_pair_n=active_step_pair_n,
                )
            )
            horizon_metric_rows.extend(
                horizon_rows(
                    spec=spec,
                    landmark_month=landmark_month,
                    baseline_metrics=baseline_metrics,
                    occluded_metrics=occluded_metrics,
                )
            )

            for i, long_row in enumerate(
                rows
            ):
                patient_risk_rows.append(
                    {
                        "long_row": int(
                            long_row
                        ),
                        "patient_local": int(
                            patient_local_for_rows[
                                i
                            ]
                        ),
                        "fold_id": int(
                            common.development_fold_id[
                                patient_local_for_rows[
                                    i
                                ]
                            ]
                        ),
                        "landmark_index": int(
                            landmark_index
                        ),
                        "landmark_month": int(
                            landmark_month
                        ),
                        "landmark_year": float(
                            landmark_month / 12.0
                        ),
                        "condition_type": spec.condition_type,
                        "condition_label": spec.condition_label,
                        "history_month": spec.history_month,
                        "lag_month": spec.lag_month,
                        "lag_band_start_month": (
                            spec.lag_band_start_month
                        ),
                        "lag_band_end_month": (
                            spec.lag_band_end_month
                        ),
                        "occluded_steps": ",".join(
                            str(int(x))
                            for x in spec.occluded_steps
                        ),
                        "baseline_risk_5y": float(
                            fp32_baseline_matrix[
                                i,
                                -1,
                            ]
                        ),
                        "occluded_risk_5y": float(
                            occluded_matrix[
                                i,
                                -1,
                            ]
                        ),
                        "delta_risk_5y": float(
                            delta_risk5[
                                i
                            ]
                        ),
                        "abs_delta_risk_5y": float(
                            abs(
                                delta_risk5[
                                    i
                                ]
                            )
                        ),
                    }
                )

    metrics_df = pd.DataFrame(
        metric_rows
    )
    horizon_df = pd.DataFrame(
        horizon_metric_rows
    )
    patient_df = pd.DataFrame(
        patient_risk_rows
    )
    baseline_metrics_df = pd.DataFrame(
        baseline_metric_rows
    )

    single_df = metrics_df.loc[
        metrics_df["condition_type"]
        == "single_step"
    ].copy()
    band_df = metrics_df.loc[
        metrics_df["condition_type"]
        == "lag_band"
    ].copy()

    single_df["iAUC_loss_rank"] = (
        single_df.groupby(
            "landmark_index"
        )["iAUC_loss"]
        .rank(
            method="first",
            ascending=False,
        )
        .astype(int)
    )
    single_df["IBS_increase_rank"] = (
        single_df.groupby(
            "landmark_index"
        )["IBS_increase"]
        .rank(
            method="first",
            ascending=False,
        )
        .astype(int)
    )
    single_df["risk_change_rank"] = (
        single_df.groupby(
            "landmark_index"
        )["mean_abs_delta_risk5"]
        .rank(
            method="first",
            ascending=False,
        )
        .astype(int)
    )

    # -------------------------------------------------------------------------
    # 保存结果。
    # -------------------------------------------------------------------------
    baseline_metrics_df.to_csv(
        OUTPUT_DIR
        / "temporal_occlusion_baseline_metric_audit.csv",
        index=False,
        encoding="utf-8-sig",
    )
    metrics_df.to_csv(
        OUTPUT_DIR
        / "temporal_occlusion_all_conditions_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    single_df.to_csv(
        OUTPUT_DIR
        / "temporal_occlusion_single_step_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    band_df.to_csv(
        OUTPUT_DIR
        / "temporal_occlusion_lag_band_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    horizon_df.to_csv(
        OUTPUT_DIR
        / "temporal_occlusion_horizon_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    patient_df.to_csv(
        OUTPUT_DIR
        / "temporal_occlusion_patient_level_5y_risk_change.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # -------------------------------------------------------------------------
    # 图形：正值统一表示occlusion造成性能损失/风险变化幅度。
    # -------------------------------------------------------------------------
    plot_single_step_metric(
        single_df,
        y_column="iAUC_loss",
        y_label="iAUC loss after occlusion",
        stem="Figure12B1_single_step_iAUC_loss",
    )
    plot_single_step_metric(
        single_df,
        y_column="IBS_increase",
        y_label="IBS increase after occlusion",
        stem="Figure12B2_single_step_IBS_increase",
    )
    plot_single_step_metric(
        single_df,
        y_column="mean_abs_delta_risk5",
        y_label="Mean absolute change in 5-year CKD risk",
        stem="Figure12B3_single_step_5y_risk_change",
    )
    plot_lag_band_metric(
        band_df,
        y_column="iAUC_loss",
        y_label="iAUC loss after occlusion",
        stem="Figure12B4_lag_band_iAUC_loss",
    )
    plot_lag_band_metric(
        band_df,
        y_column="IBS_increase",
        y_label="IBS increase after occlusion",
        stem="Figure12B5_lag_band_IBS_increase",
    )

    # -------------------------------------------------------------------------
    # 控制台摘要。
    # -------------------------------------------------------------------------
    print(
        "\n"
        + "=" * 110
    )
    print(
        "Step 12B Temporal Reference Occlusion完成"
    )
    print(
        "=" * 110
    )
    print(
        "锁定测试集：未读取"
    )
    print(
        "总耗时：",
        format_duration(
            time.time()
            - total_start
        ),
    )

    print(
        "\nFP32 baseline vs saved Step11 metric audit："
    )
    print(
        baseline_metrics_df.to_string(
            index=False
        )
    )

    print(
        "\n各Landmark单时间点iAUC loss最大的5个历史时间点："
    )
    for landmark_index in PRIMARY_LANDMARK_INDICES:
        sub = single_df.loc[
            single_df["landmark_index"]
            == int(landmark_index)
        ].sort_values(
            "iAUC_loss",
            ascending=False,
        ).head(
            5
        )
        landmark_year = (
            LANDMARK_MONTHS[
                int(landmark_index)
            ]
            / 12.0
        )
        print(
            "\n"
            + "-" * 90
        )
        print(
            f"Landmark {landmark_year:.0f}年"
        )
        print(
            sub[
                [
                    "history_month",
                    "lag_month",
                    "iAUC_loss",
                    "IBS_increase",
                    "uno_c_loss",
                    "mean_abs_delta_risk5",
                ]
            ].to_string(
                index=False
            )
        )

    print(
        "\nLag-band结果："
    )
    print(
        band_df[
            [
                "landmark_year",
                "condition_label",
                "occluded_step_n",
                "active_origin_fraction",
                "iAUC_loss",
                "IBS_increase",
                "uno_c_loss",
                "mean_abs_delta_risk5",
            ]
        ].to_string(
            index=False
        )
    )

    summary = {
        "stage": "Step12B_LSTM_v2_temporal_reference_occlusion",
        "pipeline_version": PIPELINE_VERSION,
        "primary_landmark_months": (
            PRIMARY_LANDMARK_MONTHS.tolist()
        ),
        "future_end_months": (
            FUTURE_END_MONTHS.tolist()
        ),
        "metric_times": (
            METRIC_TIMES.tolist()
        ),
        "occlusion_strategy": {
            "dynamic_input": (
                "replace selected historical timepoint's 85 dynamic channels "
                "with heldout-fold-excluded training-risk-set reference at "
                "the same landmark and history step"
            ),
            "row_mask": "kept unchanged",
            "static_input": "kept unchanged",
            "age": "kept unchanged",
            "landmark_context": "kept unchanged",
            "calibrator": (
                "original Step11 cross-fitted landmark/fold calibrator; "
                "no recalibration after occlusion"
            ),
            "inference_precision": (
                "FP32 for both baseline and occluded prediction"
            ),
            "interpretation_scope": (
                "conditional/direct reliance on the occluded timepoint representation; "
                "later delta/cumulative engineered features are intentionally left unchanged"
            ),
        },
        "single_step_condition_n": int(
            len(
                metrics_df.loc[
                    metrics_df[
                        "condition_type"
                    ]
                    == "single_step"
                ]
            )
        ),
        "lag_band_condition_n": int(
            len(
                metrics_df.loc[
                    metrics_df[
                        "condition_type"
                    ]
                    == "lag_band"
                ]
            )
        ),
        "locked_test_used": False,
        "elapsed_seconds": float(
            time.time()
            - total_start
        ),
        "output_dir": str(
            OUTPUT_DIR
        ),
    }
    save_json(
        summary,
        OUTPUT_DIR
        / "step12b_temporal_occlusion_summary.json",
    )
    save_json(
        {
            "completed": True,
            "completed_at": datetime.now().isoformat(
                timespec="seconds"
            ),
            "locked_test_used": False,
        },
        OUTPUT_DIR
        / "completed.json",
    )

    print(
        "\n输出目录：",
        OUTPUT_DIR,
    )
    print(
        "=" * 110
    )


if __name__ == "__main__":
    run()
