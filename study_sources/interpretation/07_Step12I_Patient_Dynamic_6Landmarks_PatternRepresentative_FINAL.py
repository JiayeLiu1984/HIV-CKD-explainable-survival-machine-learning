#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Step 12A FORMAL：LSTM-v2 Integrated Gradients（IG）五折 OOF 模型解释

目的
----
1. 不重新训练模型，不读取锁定测试集。
2. 直接读取 Step 10E 已保存的 5 折 × 2 seed × snapshot ensemble 权重。
3. 每个患者-Landmark仅由其 held-out fold 的模型解释，保持 OOF 原则。
4. 解释对象固定为：当前 Landmark 后未来 5 年累计 CKD 风险。
5. 默认解释 Step 11 交叉拟合校准后的风险；模型重建审计与IG梯度计算分开处理：审计严格复现Step10E的AMP+NumPy集成，IG本身使用全精度可微前向。
6. 主Landmark：0、1、3、5年。
7. 主解释方法：Integrated Gradients。
8. v4.1修正强化动态组件到临床变量的映射：使用85项输入的精确字典映射，
   不再使用可能发生后缀重叠的字符串截断；可直接复用v4已完成IG检查点。
9. 同时输出：
   - 输入组件级重要性；
   - 临床变量级聚合重要性；
   - signed attribution（推高/降低预测风险）；
   - variable × historical time 的时间归因；
   - IG completeness 误差检查；
   - 专业图形；
   - IG beeswarm；
   - Landmark相对重要性演变；
   - 实验室value/measurement-information组件拆分；
   - selected dependence plots。

重要说明
--------
- 正式分析默认每个主Landmark随机抽取2000个OOF prediction origins，共约8000个origins；IG使用32点Gauss-Legendre积分。
- 默认参数已经用于正式分析；如仅调试，可通过环境变量临时减少样本量，例如：
    CKD_IG_PER_LANDMARK=200 python Step12A_LSTM_v2_IntegratedGradients_OOF_v4_FORMAL.py
- 默认baseline为“对应训练折的经验参考输入”：
    * 动态序列：当前Landmark训练风险集患者在每个历史时间步的特征均值；
    * 静态变量：当前Landmark训练风险集患者静态特征均值；
    * 年龄：该Landmark训练风险集中的标准化年龄均值；
    * row_mask、Landmark位置保持患者真实值，不参与归因。
  这样避免使用 held-out 患者信息构建IG参考值。
- IG只归因可微输入：dynamic_sequence、static_baseline、age_at_landmark。
  Landmark本身是预测场景，不作为“临床风险因素”解释。
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

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from scipy.special import expit


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

FORMAL_STEP12A_DIR = INTERPRETATION_ROOT / "01_GLOBAL_IG"
FORMAL_STEP12A_CHECKPOINT_DIR = FORMAL_STEP12A_DIR / "ig_checkpoints"
FORMAL_STEP12A_PIPELINE_VERSION = "v5_6landmark_formal_ig_explanation"

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

# -----------------------------------------------------------------------------
# 正式分析配置
# -----------------------------------------------------------------------------
EXPLAIN_PER_LANDMARK = 3  # 仅3名按预设轨迹模式客观选择的代表病例；每人使用全部6个正式Landmark
IG_STEPS = int(os.getenv("CKD_IG_STEPS", "32"))
IG_BATCH_SIZE = int(os.getenv("CKD_IG_BATCH_SIZE", "16"))
RANDOM_SEED = int(os.getenv("CKD_IG_RANDOM_SEED", "20260810"))
EXPLAIN_CALIBRATED = True
RESUME = os.getenv("CKD_IG_RESUME", "1") != "0"
TOP_FEATURE_N = int(os.getenv("CKD_IG_TOP_FEATURE_N", "15"))

# 正式解释图配置。
BEESWARM_TOP_N = int(os.getenv("CKD_IG_BEESWARM_TOP_N", "15"))
EVOLUTION_TOP_N = int(os.getenv("CKD_IG_EVOLUTION_TOP_N", "10"))
LAB_COMPONENT_TOP_N = int(os.getenv("CKD_IG_LAB_COMPONENT_TOP_N", "8"))
DEPENDENCE_FEATURES = [
    "eGFR",
    "Age",
    "HDL",
    "TC",
    "LDL",
    "HB",
    "HIVRNA_log10",
    "BMI",
]

# 数值审计阈值。
# 1) checkpoint审计严格按Step10E原始AMP、原batch size及NumPy平均复现。
# 2) Step11校准映射直接从已保存的Step10E survival复现，理论上应接近机器精度。
# 3) IG使用全精度可微前向，因此与当时AMP保存预测允许存在极小数值漂移；
#    该漂移单独报告，不与“模型是否加载正确”混为一谈。
AMP_RECON_MAX_TOL = 2e-4
AMP_RECON_MEAN_TOL = 1e-5
STEP11_MAPPING_TOL = 5e-6
IG_FP32_RISK_DRIFT_HARD_TOL = 2e-3

if EXPLAIN_PER_LANDMARK < 1:
    raise ValueError("CKD_IG_PER_LANDMARK必须>=1。")
if IG_STEPS < 4:
    raise ValueError("CKD_IG_STEPS建议至少为4；正式分析建议32。")
if IG_BATCH_SIZE < 1:
    raise ValueError("CKD_IG_BATCH_SIZE必须>=1。")

MODE_NAME = "calibrated" if EXPLAIN_CALIBRATED else "raw"
PIPELINE_VERSION = "v2_patient_6landmark_pattern_median_local_ig"
POSTPROCESS_REVISION = "v2_prespecified_pattern_subgroup_median_representatives"
OUTPUT_DIR = INTERPRETATION_ROOT / "07_PATIENT_DYNAMIC"
CHECKPOINT_DIR = OUTPUT_DIR / "ig_checkpoints"
FIGURE_DIR = COMMON_FIGURE_DIR
for directory in [OUTPUT_DIR, CHECKPOINT_DIR, FIGURE_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

CASE_SELECTION_FILE = OUTPUT_DIR / "patient_case_selection.csv"
CANDIDATE_TRAJECTORY_FILE = OUTPUT_DIR / "patient_candidate_trajectory_summary.csv"
RISK_TRAJECTORY_FILE = OUTPUT_DIR / "patient_selected_risk_trajectory.csv"
TOP_CONTRIBUTION_FILE = OUTPUT_DIR / "patient_selected_top_contributions.csv"
FEATURE_ID_FILE = OUTPUT_DIR / "patient_selected_feature_id_table.csv"
PATIENT_FIGURE_PNG = FIGURE_DIR / "Step12I_Figure_Patient_Dynamic_Explanation_6Landmarks.png"
PATIENT_FIGURE_PDF = FIGURE_DIR / "Step12I_Figure_Patient_Dynamic_Explanation_6Landmarks.pdf"
TRAJECTORY_OVERVIEW_PNG = FIGURE_DIR / "Step12I_Figure_Patient_Trajectory_Overview_6Landmarks.png"
TRAJECTORY_OVERVIEW_PDF = FIGURE_DIR / "Step12I_Figure_Patient_Trajectory_Overview_6Landmarks.pdf"

CASE_DEFINITIONS = [
    ("Case A", "Representative stable low-risk trajectory"),
    ("Case B", "Representative increasing-risk trajectory"),
    (
        "Case C",
        "Representative decreasing-risk trajectory "
        "(fallback: representative fluctuating-risk trajectory)",
    ),
]

# Minimum number of patients required before a trajectory-pattern subgroup
# is regarded as sufficiently represented for illustrative case selection.
# This threshold is used only for choosing illustrative examples; it has no
# role in model training, calibration, evaluation, or inference.
MIN_PATTERN_SUBGROUP_N = 30

TOP_LOCAL_CONTRIBUTORS_PER_DIRECTION = 3
MAX_FEATURES_IN_CASE_TABLE = 12


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
        raise FileNotFoundError("以下必要文件不存在：\n" + "\n".join(missing))


def save_json(value: Any, path: Path) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=str),
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


def torch_load_full(path: Path, map_location: str | torch.device = "cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def default_device() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("未检测到CUDA；本步骤建议在GPU环境运行。")
    device = torch.device("cuda")
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    torch.cuda.empty_cache()
    return device


def autocast_context(device: torch.device, enabled: bool):
    """与Step10E相同的AMP上下文。仅用于模型重建审计。"""
    try:
        return torch.amp.autocast(
            device_type=device.type,
            enabled=enabled,
        )
    except (AttributeError, TypeError):
        return torch.cuda.amp.autocast(enabled=enabled)


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


# =============================================================================
# 8. LSTM ensemble与IG目标函数
# =============================================================================

class SeedSnapshotEnsemble(nn.Module):
    """
    可微的snapshot ensemble。
    IG阶段使用全精度前向和PyTorch平均，以保证梯度稳定。
    """

    def __init__(self, models: list[nn.Module]):
        super().__init__()
        if not models:
            raise ValueError("Seed ensemble没有snapshot模型。")
        self.models = nn.ModuleList(models)

    def forward_hazard(
        self,
        dynamic_sequence: torch.Tensor,
        row_mask: torch.Tensor,
        static_baseline: torch.Tensor,
        age_at_landmark: torch.Tensor,
        landmark_normalized: torch.Tensor,
        landmark_bin: torch.Tensor,
    ) -> torch.Tensor:
        hazards = []
        for model in self.models:
            logits = model(
                dynamic_sequence=dynamic_sequence,
                row_mask=row_mask,
                static_baseline=static_baseline,
                age_at_landmark=age_at_landmark,
                landmark_normalized=landmark_normalized,
                landmark_bin=landmark_bin,
            )
            hazards.append(torch.sigmoid(logits.float()))
        return torch.stack(hazards, dim=0).mean(dim=0)


class AttributionFoldFiveYearRisk(nn.Module):
    """
    IG解释目标：
      checkpoint权重的全精度可微ensemble
        -> fold/landmark对应cross-fit hazard calibrator
        -> 未来5年累计CKD风险

    这里不人为复现Step10E保存float32 survival的量化舍入。
    原因是IG需要解释一个平滑、可微的模型函数；AMP/float32保存产生的
    微小数值舍入属于计算实现误差，不应该被当作临床归因的一部分。

    与正式Step10E/Step11保存预测之间的差异会单独审计并输出。
    """

    def __init__(
        self,
        seed_ensembles: list[SeedSnapshotEnsemble],
        calibration_alpha: np.ndarray | None,
        calibration_beta: float | None,
    ):
        super().__init__()
        if not seed_ensembles:
            raise ValueError("Fold ensemble没有seed模型。")
        self.seed_ensembles = nn.ModuleList(seed_ensembles)
        self.use_calibration = calibration_alpha is not None

        if self.use_calibration:
            alpha = np.asarray(calibration_alpha, dtype=np.float64)
            if alpha.shape != (EXPECTED_FUTURE_INTERVAL_N,):
                raise ValueError("校准alpha形状不是10。")
            if calibration_beta is None or not np.isfinite(calibration_beta):
                raise ValueError("校准beta无效。")
            self.register_buffer(
                "calibration_alpha",
                torch.as_tensor(alpha, dtype=torch.float64),
            )
            self.register_buffer(
                "calibration_beta",
                torch.tensor(float(calibration_beta), dtype=torch.float64),
            )
        else:
            self.calibration_alpha = None
            self.calibration_beta = None

    def raw_hazard(
        self,
        dynamic_sequence: torch.Tensor,
        row_mask: torch.Tensor,
        static_baseline: torch.Tensor,
        age_at_landmark: torch.Tensor,
        landmark_normalized: torch.Tensor,
        landmark_bin: torch.Tensor,
    ) -> torch.Tensor:
        seed_hazards = []
        for seed_ensemble in self.seed_ensembles:
            seed_hazards.append(
                seed_ensemble.forward_hazard(
                    dynamic_sequence=dynamic_sequence,
                    row_mask=row_mask,
                    static_baseline=static_baseline,
                    age_at_landmark=age_at_landmark,
                    landmark_normalized=landmark_normalized,
                    landmark_bin=landmark_bin,
                )
            )
        return torch.stack(seed_hazards, dim=0).mean(dim=0)

    def calibrated_hazard(
        self,
        dynamic_sequence: torch.Tensor,
        row_mask: torch.Tensor,
        static_baseline: torch.Tensor,
        age_at_landmark: torch.Tensor,
        landmark_normalized: torch.Tensor,
        landmark_bin: torch.Tensor,
    ) -> torch.Tensor:
        raw_hazard = torch.clamp(
            self.raw_hazard(
                dynamic_sequence=dynamic_sequence,
                row_mask=row_mask,
                static_baseline=static_baseline,
                age_at_landmark=age_at_landmark,
                landmark_normalized=landmark_normalized,
                landmark_bin=landmark_bin,
            ),
            EPS,
            1.0 - EPS,
        )

        if not self.use_calibration:
            return raw_hazard

        raw64 = raw_hazard.to(torch.float64)
        raw_logit64 = torch.log(raw64) - torch.log1p(-raw64)
        calibrated_logit64 = (
            self.calibration_alpha.unsqueeze(0)
            + self.calibration_beta * raw_logit64
        )
        return torch.sigmoid(calibrated_logit64)

    def forward(
        self,
        dynamic_sequence: torch.Tensor,
        row_mask: torch.Tensor,
        static_baseline: torch.Tensor,
        age_at_landmark: torch.Tensor,
        landmark_normalized: torch.Tensor,
        landmark_bin: torch.Tensor,
    ) -> torch.Tensor:
        hazard64 = self.calibrated_hazard(
            dynamic_sequence=dynamic_sequence,
            row_mask=row_mask,
            static_baseline=static_baseline,
            age_at_landmark=age_at_landmark,
            landmark_normalized=landmark_normalized,
            landmark_bin=landmark_bin,
        ).to(torch.float64)
        survival5 = torch.prod(1.0 - hazard64, dim=1)
        return 1.0 - survival5


# =============================================================================
# 9. 读取Step11 cross-fit calibrator
# =============================================================================

CALIBRATION_FILE = STEP11_DIR / "four_model_hazard_calibration_parameters.csv"
CALIBRATED_RISK_FILE = STEP11_DIR / "lstm_v2_crossfit_calibrated_oof_risk_long.npy"


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


# =============================================================================
# 10. 读取fold ensemble
# =============================================================================

def load_fold_ensemble(
    fold_id: int,
    landmark_month: int,
    device: torch.device,
    calibration_table: pd.DataFrame | None,
) -> tuple[AttributionFoldFiveYearRisk, list[str], list[str], dict[str, Any]]:
    fold_dir = STEP10E_DIR / f"fold_{fold_id}"
    checkpoint_paths = sorted(fold_dir.glob("seed_*_snapshot_ensemble.pt"))
    if not checkpoint_paths:
        raise FileNotFoundError(
            f"Fold {fold_id}未找到seed snapshot ensemble检查点：{fold_dir}"
        )

    seed_ensembles: list[SeedSnapshotEnsemble] = []
    reference_dynamic_names = None
    reference_static_names = None
    reference_parameters = None
    checkpoint_summary = []

    for checkpoint_path in checkpoint_paths:
        checkpoint = torch_load_full(checkpoint_path, map_location="cpu")
        if int(checkpoint["fold_id"]) != int(fold_id):
            raise ValueError(f"检查点fold不一致：{checkpoint_path}")
        parameters = dict(checkpoint["parameters"])
        dynamic_names = list(checkpoint["feature_names"]["enhanced_dynamic"])
        static_names = list(checkpoint["feature_names"]["static"])
        snapshot_states = list(checkpoint["snapshot_state_dicts"])

        if len(dynamic_names) != EXPECTED_ENHANCED_DYNAMIC_N:
            raise ValueError("检查点动态特征数不是85。")
        if len(static_names) != EXPECTED_STATIC_N:
            raise ValueError("检查点静态特征数不是16。")
        if not snapshot_states:
            raise ValueError(f"检查点没有snapshot state：{checkpoint_path}")

        if reference_dynamic_names is None:
            reference_dynamic_names = dynamic_names
            reference_static_names = static_names
            reference_parameters = parameters
        else:
            if dynamic_names != reference_dynamic_names:
                raise ValueError("同一fold不同seed动态特征顺序不一致。")
            if static_names != reference_static_names:
                raise ValueError("同一fold不同seed静态特征顺序不一致。")
            if parameters != reference_parameters:
                raise ValueError("同一fold不同seed模型参数不一致。")

        snapshot_models = []
        for state in snapshot_states:
            model = build_model(parameters, device)
            model.load_state_dict(state, strict=True)
            model.eval()
            for parameter in model.parameters():
                parameter.requires_grad_(False)
            snapshot_models.append(model)

        seed_ensembles.append(SeedSnapshotEnsemble(snapshot_models).to(device))
        checkpoint_summary.append(
            {
                "checkpoint": str(checkpoint_path),
                "seed": int(checkpoint["seed"]),
                "snapshot_epochs": list(checkpoint["snapshot_epochs"]),
                "snapshot_n": int(len(snapshot_states)),
            }
        )

    if EXPLAIN_CALIBRATED:
        if calibration_table is None:
            raise ValueError("要求解释校准风险，但没有校准参数表。")
        alpha, beta = load_crossfit_calibrator(
            calibration_table,
            fold_id=fold_id,
            landmark_month=landmark_month,
        )
    else:
        alpha, beta = None, None

    wrapper = AttributionFoldFiveYearRisk(
        seed_ensembles=seed_ensembles,
        calibration_alpha=alpha,
        calibration_beta=beta,
    ).to(device)
    wrapper.eval()
    for parameter in wrapper.parameters():
        parameter.requires_grad_(False)

    metadata = {
        "fold_id": int(fold_id),
        "landmark_month": int(landmark_month),
        "seed_ensemble_n": int(len(seed_ensembles)),
        "checkpoints": checkpoint_summary,
        "calibrated": bool(EXPLAIN_CALIBRATED),
        "parameters": reference_parameters,
    }
    return (
        wrapper,
        reference_dynamic_names,
        reference_static_names,
        metadata,
    )


# =============================================================================
# 11. OOF抽样：每个主Landmark从真实风险集均匀随机抽取
# =============================================================================

def build_or_load_sample_selection(
    common: CommonData,
    calibrated_risk_long: np.ndarray,
) -> pd.DataFrame:
    """
    从当前正式6-Landmark LSTM-v2 OOF校准风险中，
    按预设轨迹模式客观选择3名代表病例。

    选择原则
    --------
    1. 仅使用0/1/2/3/4/5年6个正式Landmark的cross-fit calibrated 5-year risk。
    2. 不使用CKD结局、IG大小或任何临床变量值进行病例选择。
    3. 先依据预设分位数规则定义trajectory-pattern subgroup，再在该亚组中
       选择六个Landmark风险轨迹最接近该亚组landmark-wise median trajectory
       的患者，而不是选择最极端患者。
    4. Case A：Representative stable low-risk trajectory。
    5. Case B：Representative increasing-risk trajectory。
    6. Case C：优先选择Representative decreasing-risk trajectory；
       如果下降型亚组样本不足，则自动切换为Representative fluctuating-risk trajectory。
    7. 三名患者必须不同；结局仅在病例选择完成后附加用于描述。
    """

    selection_path = OUTPUT_DIR / "ig_selected_oof_origins.csv"

    calibrated_risk_long = np.asarray(
        calibrated_risk_long,
    )

    if (
        calibrated_risk_long.ndim != 2
        or calibrated_risk_long.shape[1] != EXPECTED_FUTURE_INTERVAL_N
    ):
        raise ValueError(
            "Step11 LSTM-v2 calibrated OOF risk long形状错误："
            f"{calibrated_risk_long.shape}"
        )

    # ---------------------------------------------------------
    # 同时拥有全部6个正式Landmark的候选患者
    # ---------------------------------------------------------
    valid_all_landmarks = np.all(
        common.prediction_origin_mask[:, PRIMARY_LANDMARK_INDICES],
        axis=1,
    )

    candidate_patients = np.where(
        valid_all_landmarks
    )[0].astype(
        np.int64
    )

    if len(candidate_patients) < 3:
        raise RuntimeError(
            "同时具有0/1/2/3/4/5年全部正式Landmark的患者不足3名。"
        )

    risk_matrix = np.full(
        (
            len(candidate_patients),
            len(PRIMARY_LANDMARK_INDICES),
        ),
        np.nan,
        dtype=np.float64,
    )

    for column_index, landmark_index in enumerate(
        PRIMARY_LANDMARK_INDICES
    ):
        long_rows = common.long_row_index_map[
            candidate_patients,
            int(landmark_index),
        ].astype(
            np.int64
        )

        if (long_rows < 0).any():
            raise ValueError(
                "候选患者中存在有效Landmark但long_row_index_map<0。"
            )

        risk_matrix[
            :,
            column_index,
        ] = calibrated_risk_long[
            long_rows,
            -1,
        ].astype(
            np.float64
        )

    if not np.isfinite(
        risk_matrix
    ).all():
        raise ValueError(
            "候选患者6-Landmark calibrated 5-year risk存在非有限值。"
        )

    if (
        (risk_matrix < -1e-8).any()
        or (risk_matrix > 1.0 + 1e-8).any()
    ):
        raise ValueError(
            "候选患者风险超出[0,1]。"
        )

    # ---------------------------------------------------------
    # 构建候选轨迹摘要
    # ---------------------------------------------------------
    candidate_summary = pd.DataFrame(
        {
            "patient_local": candidate_patients,
            "development_global_index": common.development_idx[
                candidate_patients
            ].astype(
                np.int64
            ),
            "fold_id": common.development_fold_id[
                candidate_patients
            ].astype(
                np.int64
            ),
        }
    )

    for column_index, landmark_index in enumerate(
        PRIMARY_LANDMARK_INDICES
    ):
        landmark_month = int(
            LANDMARK_MONTHS[
                int(landmark_index)
            ]
        )
        candidate_summary[
            f"risk_{landmark_month}m"
        ] = risk_matrix[
            :,
            column_index,
        ]

    candidate_summary[
        "risk_mean"
    ] = risk_matrix.mean(
        axis=1
    )

    candidate_summary[
        "risk_sd"
    ] = risk_matrix.std(
        axis=1,
        ddof=0,
    )

    candidate_summary[
        "risk_min"
    ] = risk_matrix.min(
        axis=1
    )

    candidate_summary[
        "risk_max"
    ] = risk_matrix.max(
        axis=1
    )

    candidate_summary[
        "risk_range"
    ] = (
        candidate_summary[
            "risk_max"
        ]
        - candidate_summary[
            "risk_min"
        ]
    )

    candidate_summary[
        "risk_first"
    ] = risk_matrix[
        :,
        0,
    ]

    candidate_summary[
        "risk_last"
    ] = risk_matrix[
        :,
        -1,
    ]

    candidate_summary[
        "risk_delta_first_to_last"
    ] = (
        candidate_summary[
            "risk_last"
        ]
        - candidate_summary[
            "risk_first"
        ]
    )

    # ---------------------------------------------------------
    # 构建用于预设trajectory-pattern选择的客观摘要
    # ---------------------------------------------------------
    mean_risk = candidate_summary[
        "risk_mean"
    ].to_numpy(
        dtype=np.float64
    )

    risk_range = candidate_summary[
        "risk_range"
    ].to_numpy(
        dtype=np.float64
    )

    first_risk = candidate_summary[
        "risk_first"
    ].to_numpy(
        dtype=np.float64
    )

    last_risk = candidate_summary[
        "risk_last"
    ].to_numpy(
        dtype=np.float64
    )

    delta = candidate_summary[
        "risk_delta_first_to_last"
    ].to_numpy(
        dtype=np.float64
    )

    abs_delta = np.abs(
        delta
    )

    # 每个患者六点风险的相邻差分，用于识别真正具有方向变化的fluctuating pattern。
    risk_diff = np.diff(
        risk_matrix,
        axis=1,
    )

    diff_sign = np.sign(
        risk_diff
    )

    # 极小变化视为0，避免浮点噪声造成伪方向切换。
    diff_sign[
        np.abs(
            risk_diff
        )
        < 1e-8
    ] = 0

    direction_change_n = np.zeros(
        len(
            candidate_summary
        ),
        dtype=np.int32,
    )

    for patient_index in range(
        len(
            candidate_summary
        )
    ):
        nonzero_sign = diff_sign[
            patient_index
        ]

        nonzero_sign = nonzero_sign[
            nonzero_sign != 0
        ]

        if len(
            nonzero_sign
        ) >= 2:
            direction_change_n[
                patient_index
            ] = int(
                np.sum(
                    nonzero_sign[
                        1:
                    ]
                    != nonzero_sign[
                        :-1
                    ]
                )
            )

    candidate_summary[
        "risk_abs_delta_first_to_last"
    ] = abs_delta

    candidate_summary[
        "direction_change_n"
    ] = direction_change_n

    # ---------------------------------------------------------
    # 保存关键分位数，所有阈值均由当前完整候选OOF风险轨迹得到。
    # ---------------------------------------------------------
    quantiles = {
        "mean_q25": float(
            np.quantile(
                mean_risk,
                0.25,
            )
        ),
        "mean_q33": float(
            np.quantile(
                mean_risk,
                0.33,
            )
        ),
        "range_q25": float(
            np.quantile(
                risk_range,
                0.25,
            )
        ),
        "range_q50": float(
            np.quantile(
                risk_range,
                0.50,
            )
        ),
        "range_q85": float(
            np.quantile(
                risk_range,
                0.85,
            )
        ),
        "range_q90": float(
            np.quantile(
                risk_range,
                0.90,
            )
        ),
        "first_q50": float(
            np.quantile(
                first_risk,
                0.50,
            )
        ),
        "first_q60": float(
            np.quantile(
                first_risk,
                0.60,
            )
        ),
        "first_q70": float(
            np.quantile(
                first_risk,
                0.70,
            )
        ),
        "first_q75": float(
            np.quantile(
                first_risk,
                0.75,
            )
        ),
        "last_q65": float(
            np.quantile(
                last_risk,
                0.65,
            )
        ),
        "last_q70": float(
            np.quantile(
                last_risk,
                0.70,
            )
        ),
        "last_q75": float(
            np.quantile(
                last_risk,
                0.75,
            )
        ),
        "delta_q10": float(
            np.quantile(
                delta,
                0.10,
            )
        ),
        "delta_q15": float(
            np.quantile(
                delta,
                0.15,
            )
        ),
        "delta_q20": float(
            np.quantile(
                delta,
                0.20,
            )
        ),
        "delta_q80": float(
            np.quantile(
                delta,
                0.80,
            )
        ),
        "delta_q85": float(
            np.quantile(
                delta,
                0.85,
            )
        ),
        "delta_q90": float(
            np.quantile(
                delta,
                0.90,
            )
        ),
        "abs_delta_q50": float(
            np.quantile(
                abs_delta,
                0.50,
            )
        ),
        "abs_delta_q60": float(
            np.quantile(
                abs_delta,
                0.60,
            )
        ),
    }

    for quantile_name, quantile_value in quantiles.items():
        candidate_summary[
            quantile_name
        ] = quantile_value

    # ---------------------------------------------------------
    # percentile audit columns
    # ---------------------------------------------------------
    candidate_summary[
        "mean_risk_percentile"
    ] = candidate_summary[
        "risk_mean"
    ].rank(
        method="average",
        pct=True,
    )

    candidate_summary[
        "first_risk_percentile"
    ] = candidate_summary[
        "risk_first"
    ].rank(
        method="average",
        pct=True,
    )

    candidate_summary[
        "last_risk_percentile"
    ] = candidate_summary[
        "risk_last"
    ].rank(
        method="average",
        pct=True,
    )

    candidate_summary[
        "delta_percentile"
    ] = candidate_summary[
        "risk_delta_first_to_last"
    ].rank(
        method="average",
        pct=True,
    )

    candidate_summary[
        "range_percentile"
    ] = candidate_summary[
        "risk_range"
    ].rank(
        method="average",
        pct=True,
    )

    # ---------------------------------------------------------
    # helper：在预设亚组中选择最接近landmark-wise median trajectory的患者
    # ---------------------------------------------------------
    risk_column_names = [
        f"risk_{int(LANDMARK_MONTHS[int(landmark_index)])}m"
        for landmark_index in PRIMARY_LANDMARK_INDICES
    ]

    def choose_median_trajectory_representative(
        subgroup_mask: np.ndarray,
        *,
        used_patients: set[int],
        case_label: str,
        trajectory_label: str,
        selection_pattern: str,
        selection_tier: str,
        selection_criteria: str,
    ) -> tuple[pd.Series, dict[str, Any]]:
        subgroup_mask = np.asarray(
            subgroup_mask,
            dtype=bool,
        )

        subgroup = candidate_summary.loc[
            subgroup_mask
        ].copy()

        subgroup = subgroup.loc[
            ~subgroup[
                "patient_local"
            ].isin(
                used_patients
            )
        ].copy()

        if len(
            subgroup
        ) == 0:
            raise RuntimeError(
                f"{case_label}亚组在排除已选患者后为空。"
            )

        subgroup_risk_matrix = subgroup[
            risk_column_names
        ].to_numpy(
            dtype=np.float64
        )

        median_trajectory = np.median(
            subgroup_risk_matrix,
            axis=0,
        )

        # 使用六个Landmark原始概率尺度的欧氏距离。
        # 目的不是寻找极端病例，而是寻找该预设pattern subgroup的典型轨迹。
        distance = np.sqrt(
            np.sum(
                (
                    subgroup_risk_matrix
                    - median_trajectory[
                        None,
                        :,
                    ]
                )
                ** 2,
                axis=1,
            )
        )

        subgroup[
            "distance_to_subgroup_median_trajectory"
        ] = distance

        subgroup = subgroup.sort_values(
            [
                "distance_to_subgroup_median_trajectory",
                "patient_local",
            ],
            ascending=[
                True,
                True,
            ],
        )

        chosen = subgroup.iloc[
            0
        ].copy()

        metadata = {
            "case_label": case_label,
            "trajectory_label": trajectory_label,
            "selection_pattern": selection_pattern,
            "selection_tier": selection_tier,
            "selection_criteria": selection_criteria,
            "pattern_subgroup_n_before_excluding_used": int(
                np.sum(
                    subgroup_mask
                )
            ),
            "pattern_subgroup_n_after_excluding_used": int(
                len(
                    subgroup
                )
            ),
            "distance_to_subgroup_median_trajectory": float(
                chosen[
                    "distance_to_subgroup_median_trajectory"
                ]
            ),
        }

        for landmark_position, landmark_index in enumerate(
            PRIMARY_LANDMARK_INDICES
        ):
            landmark_month = int(
                LANDMARK_MONTHS[
                    int(
                        landmark_index
                    )
                ]
            )

            metadata[
                f"subgroup_median_risk_{landmark_month}m"
            ] = float(
                median_trajectory[
                    landmark_position
                ]
            )

        return (
            chosen,
            metadata,
        )

    # ---------------------------------------------------------
    # Case A：Representative stable low-risk trajectory
    #
    # Primary:
    #   mean risk <= Q25 AND trajectory range <= Q25.
    #
    # Fallback:
    #   mean risk <= Q33 AND trajectory range <= Q50.
    # ---------------------------------------------------------
    stable_low_tiers = [
        {
            "tier": "A_primary",
            "mask": (
                candidate_summary[
                    "risk_mean"
                ].to_numpy(
                    dtype=np.float64
                )
                <= quantiles[
                    "mean_q25"
                ]
            )
            & (
                candidate_summary[
                    "risk_range"
                ].to_numpy(
                    dtype=np.float64
                )
                <= quantiles[
                    "range_q25"
                ]
            ),
            "criteria": (
                "risk_mean <= cohort Q25 and risk_range <= cohort Q25"
            ),
        },
        {
            "tier": "A_fallback_1",
            "mask": (
                candidate_summary[
                    "risk_mean"
                ].to_numpy(
                    dtype=np.float64
                )
                <= quantiles[
                    "mean_q33"
                ]
            )
            & (
                candidate_summary[
                    "risk_range"
                ].to_numpy(
                    dtype=np.float64
                )
                <= quantiles[
                    "range_q50"
                ]
            ),
            "criteria": (
                "risk_mean <= cohort Q33 and risk_range <= cohort Q50"
            ),
        },
    ]

    # ---------------------------------------------------------
    # Case B：Representative increasing-risk trajectory
    #
    # Primary:
    #   delta >= Q90, baseline <= Q50, final >= Q75.
    #
    # Progressive fallback only if the strict pattern is underrepresented.
    # ---------------------------------------------------------
    increasing_tiers = [
        {
            "tier": "B_primary",
            "mask": (
                candidate_summary[
                    "risk_delta_first_to_last"
                ].to_numpy(
                    dtype=np.float64
                )
                >= quantiles[
                    "delta_q90"
                ]
            )
            & (
                candidate_summary[
                    "risk_first"
                ].to_numpy(
                    dtype=np.float64
                )
                <= quantiles[
                    "first_q50"
                ]
            )
            & (
                candidate_summary[
                    "risk_last"
                ].to_numpy(
                    dtype=np.float64
                )
                >= quantiles[
                    "last_q75"
                ]
            ),
            "criteria": (
                "delta >= cohort Q90, baseline risk <= Q50, final risk >= Q75"
            ),
        },
        {
            "tier": "B_fallback_1",
            "mask": (
                candidate_summary[
                    "risk_delta_first_to_last"
                ].to_numpy(
                    dtype=np.float64
                )
                >= quantiles[
                    "delta_q85"
                ]
            )
            & (
                candidate_summary[
                    "risk_first"
                ].to_numpy(
                    dtype=np.float64
                )
                <= quantiles[
                    "first_q60"
                ]
            )
            & (
                candidate_summary[
                    "risk_last"
                ].to_numpy(
                    dtype=np.float64
                )
                >= quantiles[
                    "last_q70"
                ]
            ),
            "criteria": (
                "delta >= cohort Q85, baseline risk <= Q60, final risk >= Q70"
            ),
        },
        {
            "tier": "B_fallback_2",
            "mask": (
                candidate_summary[
                    "risk_delta_first_to_last"
                ].to_numpy(
                    dtype=np.float64
                )
                >= quantiles[
                    "delta_q80"
                ]
            )
            & (
                candidate_summary[
                    "risk_last"
                ].to_numpy(
                    dtype=np.float64
                )
                >= quantiles[
                    "last_q65"
                ]
            ),
            "criteria": (
                "delta >= cohort Q80 and final risk >= Q65"
            ),
        },
    ]

    # ---------------------------------------------------------
    # Case C：优先 Representative decreasing-risk trajectory
    #
    # Primary:
    #   baseline >= Q75 AND delta <= Q10 AND delta < 0.
    #
    # If underrepresented, progressively relax.
    #
    # If a clear decreasing subgroup still has < MIN_PATTERN_SUBGROUP_N,
    # switch to fluctuating:
    #   large range + at least one direction change + small net first-to-last change.
    # ---------------------------------------------------------
    decreasing_tiers = [
        {
            "tier": "C_decreasing_primary",
            "mask": (
                candidate_summary[
                    "risk_first"
                ].to_numpy(
                    dtype=np.float64
                )
                >= quantiles[
                    "first_q75"
                ]
            )
            & (
                candidate_summary[
                    "risk_delta_first_to_last"
                ].to_numpy(
                    dtype=np.float64
                )
                <= quantiles[
                    "delta_q10"
                ]
            )
            & (
                candidate_summary[
                    "risk_delta_first_to_last"
                ].to_numpy(
                    dtype=np.float64
                )
                < 0.0
            ),
            "criteria": (
                "baseline risk >= cohort Q75, delta <= cohort Q10, and delta < 0"
            ),
        },
        {
            "tier": "C_decreasing_fallback_1",
            "mask": (
                candidate_summary[
                    "risk_first"
                ].to_numpy(
                    dtype=np.float64
                )
                >= quantiles[
                    "first_q70"
                ]
            )
            & (
                candidate_summary[
                    "risk_delta_first_to_last"
                ].to_numpy(
                    dtype=np.float64
                )
                <= quantiles[
                    "delta_q15"
                ]
            )
            & (
                candidate_summary[
                    "risk_delta_first_to_last"
                ].to_numpy(
                    dtype=np.float64
                )
                < 0.0
            ),
            "criteria": (
                "baseline risk >= cohort Q70, delta <= cohort Q15, and delta < 0"
            ),
        },
        {
            "tier": "C_decreasing_fallback_2",
            "mask": (
                candidate_summary[
                    "risk_delta_first_to_last"
                ].to_numpy(
                    dtype=np.float64
                )
                <= quantiles[
                    "delta_q20"
                ]
            )
            & (
                candidate_summary[
                    "risk_delta_first_to_last"
                ].to_numpy(
                    dtype=np.float64
                )
                < 0.0
            ),
            "criteria": (
                "delta <= cohort Q20 and delta < 0"
            ),
        },
    ]

    fluctuating_tiers = [
        {
            "tier": "C_fluctuating_fallback_1",
            "mask": (
                candidate_summary[
                    "risk_range"
                ].to_numpy(
                    dtype=np.float64
                )
                >= quantiles[
                    "range_q90"
                ]
            )
            & (
                candidate_summary[
                    "risk_abs_delta_first_to_last"
                ].to_numpy(
                    dtype=np.float64
                )
                <= quantiles[
                    "abs_delta_q50"
                ]
            )
            & (
                candidate_summary[
                    "direction_change_n"
                ].to_numpy(
                    dtype=np.int32
                )
                >= 1
            ),
            "criteria": (
                "risk_range >= cohort Q90, |delta| <= cohort median, "
                "and >=1 direction change"
            ),
        },
        {
            "tier": "C_fluctuating_fallback_2",
            "mask": (
                candidate_summary[
                    "risk_range"
                ].to_numpy(
                    dtype=np.float64
                )
                >= quantiles[
                    "range_q85"
                ]
            )
            & (
                candidate_summary[
                    "risk_abs_delta_first_to_last"
                ].to_numpy(
                    dtype=np.float64
                )
                <= quantiles[
                    "abs_delta_q60"
                ]
            )
            & (
                candidate_summary[
                    "direction_change_n"
                ].to_numpy(
                    dtype=np.int32
                )
                >= 1
            ),
            "criteria": (
                "risk_range >= cohort Q85, |delta| <= cohort Q60, "
                "and >=1 direction change"
            ),
        },
    ]

    # ---------------------------------------------------------
    # helper：选择第一个达到MIN_PATTERN_SUBGROUP_N的tier；
    # 若所有tier均不足，则选择非空tier中样本最多的一个，并在输出中明确标记。
    # ---------------------------------------------------------
    def resolve_pattern_tier(
        tiers: list[dict[str, Any]],
        *,
        pattern_name: str,
    ) -> tuple[dict[str, Any], bool]:
        tier_counts = [
            int(
                np.sum(
                    np.asarray(
                        tier[
                            "mask"
                        ],
                        dtype=bool,
                    )
                )
            )
            for tier in tiers
        ]

        for tier, count in zip(
            tiers,
            tier_counts,
        ):
            if count >= MIN_PATTERN_SUBGROUP_N:
                return (
                    tier,
                    True,
                )

        nonempty = [
            (
                tier,
                count,
            )
            for tier, count in zip(
                tiers,
                tier_counts,
            )
            if count > 0
        ]

        if not nonempty:
            raise RuntimeError(
                f"{pattern_name}的所有预设tier均为空。"
            )

        # 若无法达到最低亚组样本数，不隐瞒：选择样本数最多的非空tier，
        # 并在case_selection中记录pattern_subgroup_meets_min_n=False。
        best_tier, _ = max(
            nonempty,
            key=lambda item: item[
                1
            ],
        )

        return (
            best_tier,
            False,
        )

    selected_rows = []
    used_patients: set[int] = set()

    # ---------------------------------------------------------
    # Case A
    # ---------------------------------------------------------
    case_a_tier, case_a_meets_min = resolve_pattern_tier(
        stable_low_tiers,
        pattern_name="stable low-risk trajectory",
    )

    case_a_chosen, case_a_meta = choose_median_trajectory_representative(
        case_a_tier[
            "mask"
        ],
        used_patients=used_patients,
        case_label="Case A",
        trajectory_label="Representative stable low-risk trajectory",
        selection_pattern="stable_low",
        selection_tier=case_a_tier[
            "tier"
        ],
        selection_criteria=case_a_tier[
            "criteria"
        ],
    )

    used_patients.add(
        int(
            case_a_chosen[
                "patient_local"
            ]
        )
    )

    case_a_meta[
        "pattern_subgroup_meets_min_n"
    ] = bool(
        case_a_meets_min
    )

    # ---------------------------------------------------------
    # Case B
    # ---------------------------------------------------------
    case_b_tier, case_b_meets_min = resolve_pattern_tier(
        increasing_tiers,
        pattern_name="increasing-risk trajectory",
    )

    case_b_chosen, case_b_meta = choose_median_trajectory_representative(
        case_b_tier[
            "mask"
        ],
        used_patients=used_patients,
        case_label="Case B",
        trajectory_label="Representative increasing-risk trajectory",
        selection_pattern="increasing",
        selection_tier=case_b_tier[
            "tier"
        ],
        selection_criteria=case_b_tier[
            "criteria"
        ],
    )

    used_patients.add(
        int(
            case_b_chosen[
                "patient_local"
            ]
        )
    )

    case_b_meta[
        "pattern_subgroup_meets_min_n"
    ] = bool(
        case_b_meets_min
    )

    # ---------------------------------------------------------
    # Case C：先尝试decreasing；若没有达到最低样本量，则优先转fluctuating。
    # ---------------------------------------------------------
    decreasing_counts = [
        int(
            np.sum(
                np.asarray(
                    tier[
                        "mask"
                    ],
                    dtype=bool,
                )
            )
        )
        for tier in decreasing_tiers
    ]

    decreasing_has_adequate_tier = any(
        count >= MIN_PATTERN_SUBGROUP_N
        for count in decreasing_counts
    )

    if decreasing_has_adequate_tier:
        case_c_tier, case_c_meets_min = resolve_pattern_tier(
            decreasing_tiers,
            pattern_name="decreasing-risk trajectory",
        )

        case_c_label = "Representative decreasing-risk trajectory"
        case_c_pattern = "decreasing"

    else:
        # 如果下降pattern过少，使用预设fluctuating fallback。
        fluctuating_counts = [
            int(
                np.sum(
                    np.asarray(
                        tier[
                            "mask"
                        ],
                        dtype=bool,
                    )
                )
            )
            for tier in fluctuating_tiers
        ]

        if any(
            count > 0
            for count in fluctuating_counts
        ):
            case_c_tier, case_c_meets_min = resolve_pattern_tier(
                fluctuating_tiers,
                pattern_name="fluctuating-risk trajectory",
            )

            case_c_label = "Representative fluctuating-risk trajectory"
            case_c_pattern = "fluctuating"
        else:
            # 极端情况下fluctuating也不存在，则退回decreasing中最大非空tier，
            # 但会明确标记未达到MIN_PATTERN_SUBGROUP_N。
            case_c_tier, case_c_meets_min = resolve_pattern_tier(
                decreasing_tiers,
                pattern_name="decreasing-risk trajectory",
            )

            case_c_label = "Representative decreasing-risk trajectory"
            case_c_pattern = "decreasing"

    case_c_chosen, case_c_meta = choose_median_trajectory_representative(
        case_c_tier[
            "mask"
        ],
        used_patients=used_patients,
        case_label="Case C",
        trajectory_label=case_c_label,
        selection_pattern=case_c_pattern,
        selection_tier=case_c_tier[
            "tier"
        ],
        selection_criteria=case_c_tier[
            "criteria"
        ],
    )

    used_patients.add(
        int(
            case_c_chosen[
                "patient_local"
            ]
        )
    )

    case_c_meta[
        "pattern_subgroup_meets_min_n"
    ] = bool(
        case_c_meets_min
    )

    # ---------------------------------------------------------
    # Assemble selected-case table
    # ---------------------------------------------------------
    chosen_cases = [
        (
            case_a_chosen,
            case_a_meta,
        ),
        (
            case_b_chosen,
            case_b_meta,
        ),
        (
            case_c_chosen,
            case_c_meta,
        ),
    ]

    for chosen, metadata in chosen_cases:
        selected_row = {
            **metadata,
            "patient_local": int(
                chosen[
                    "patient_local"
                ]
            ),
            "development_global_index": int(
                chosen[
                    "development_global_index"
                ]
            ),
            "fold_id": int(
                chosen[
                    "fold_id"
                ]
            ),
            "risk_first": float(
                chosen[
                    "risk_first"
                ]
            ),
            "risk_last": float(
                chosen[
                    "risk_last"
                ]
            ),
            "risk_mean": float(
                chosen[
                    "risk_mean"
                ]
            ),
            "risk_sd": float(
                chosen[
                    "risk_sd"
                ]
            ),
            "risk_range": float(
                chosen[
                    "risk_range"
                ]
            ),
            "risk_delta_first_to_last": float(
                chosen[
                    "risk_delta_first_to_last"
                ]
            ),
            "risk_abs_delta_first_to_last": float(
                chosen[
                    "risk_abs_delta_first_to_last"
                ]
            ),
            "direction_change_n": int(
                chosen[
                    "direction_change_n"
                ]
            ),
            "mean_risk_percentile": float(
                chosen[
                    "mean_risk_percentile"
                ]
            ),
            "first_risk_percentile": float(
                chosen[
                    "first_risk_percentile"
                ]
            ),
            "last_risk_percentile": float(
                chosen[
                    "last_risk_percentile"
                ]
            ),
            "delta_percentile": float(
                chosen[
                    "delta_percentile"
                ]
            ),
            "range_percentile": float(
                chosen[
                    "range_percentile"
                ]
            ),
            "selection_uses_outcome": False,
            "selection_uses_attribution": False,
            "selection_uses_clinical_values": False,
            "candidate_pool_n": int(
                len(
                    candidate_summary
                )
            ),
            "minimum_pattern_subgroup_n": int(
                MIN_PATTERN_SUBGROUP_N
            ),
        }

        for risk_column_name in risk_column_names:
            selected_row[
                risk_column_name
            ] = float(
                chosen[
                    risk_column_name
                ]
            )

        selected_rows.append(
            selected_row
        )

    selected_cases = pd.DataFrame(
        selected_rows
    )

    # ---------------------------------------------------------
    # 只有选择完成之后才附加结局描述。
    # ---------------------------------------------------------
    last_landmark_index = int(
        PRIMARY_LANDMARK_INDICES[
            -1
        ]
    )

    selected_cases[
        "future_5y_CKD_event_after_5y_landmark"
    ] = [
        int(
            common.future_event_matrix[
                int(patient_local),
                last_landmark_index,
                :,
            ].sum()
            > 0.5
        )
        for patient_local in selected_cases[
            "patient_local"
        ]
    ]

    candidate_summary.to_csv(
        CANDIDATE_TRAJECTORY_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    selected_cases.to_csv(
        CASE_SELECTION_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    # ---------------------------------------------------------
    # 3名病例 × 6个正式Landmark = 18个local IG origins
    # ---------------------------------------------------------
    rows = []

    case_lookup = selected_cases.set_index(
        "patient_local"
    )[
        [
            "case_label",
            "trajectory_label",
        ]
    ].to_dict(
        orient="index"
    )

    for patient_local in selected_cases[
        "patient_local"
    ].astype(
        int
    ):
        for landmark_index in PRIMARY_LANDMARK_INDICES:
            landmark_index = int(
                landmark_index
            )

            if not bool(
                common.prediction_origin_mask[
                    patient_local,
                    landmark_index,
                ]
            ):
                raise ValueError(
                    f"选中患者{patient_local}缺少"
                    f"Landmark {LANDMARK_MONTHS[landmark_index]}月"
                    "有效prediction origin。"
                )

            rows.append(
                {
                    "case_label": case_lookup[
                        patient_local
                    ][
                        "case_label"
                    ],
                    "trajectory_label": case_lookup[
                        patient_local
                    ][
                        "trajectory_label"
                    ],
                    "patient_local": int(
                        patient_local
                    ),
                    "development_global_index": int(
                        common.development_idx[
                            patient_local
                        ]
                    ),
                    "fold_id": int(
                        common.development_fold_id[
                            patient_local
                        ]
                    ),
                    "landmark_index": landmark_index,
                    "landmark_month": int(
                        LANDMARK_MONTHS[
                            landmark_index
                        ]
                    ),
                    "landmark_year": float(
                        LANDMARK_MONTHS[
                            landmark_index
                        ]
                        / 12.0
                    ),
                    "event_within_5y": int(
                        common.future_event_matrix[
                            patient_local,
                            landmark_index,
                            :,
                        ].sum()
                        > 0.5
                    ),
                }
            )

    table = pd.DataFrame(
        rows
    ).sort_values(
        [
            "landmark_index",
            "fold_id",
            "patient_local",
        ]
    ).reset_index(
        drop=True
    )

    if len(
        table
    ) != (
        len(
            selected_cases
        )
        * len(
            PRIMARY_LANDMARK_INDICES
        )
    ):
        raise ValueError(
            "local IG origin数量不是3名病例×6个Landmark。"
        )

    table.to_csv(
        selection_path,
        index=False,
        encoding="utf-8-sig",
    )

    return table


# =============================================================================
# 12. 构建一个fold-landmark样本批次与IG baseline
# =============================================================================

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


def build_inputs_for_pairs(
    common: CommonData,
    fold_arrays: FoldArrays,
    patient_local: np.ndarray,
    landmark_index: int,
) -> dict[str, np.ndarray]:
    patient_local = np.asarray(patient_local, dtype=np.int64)
    landmark_month = int(LANDMARK_MONTHS[landmark_index])

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
        (age_raw - fold_arrays.age_mean) / fold_arrays.age_scale
    ).astype(np.float32)

    landmark_norm = np.full(
        len(patient_local),
        landmark_month / 60.0,
        dtype=np.float32,
    )
    landmark_bin = np.full(
        len(patient_local),
        int(LANDMARK_BINS[landmark_index]),
        dtype=np.int64,
    )

    # -------------------------------------------------------------------------
    # fold训练数据reference；患者自己的row_mask保持不变。
    # -------------------------------------------------------------------------
    baseline_dynamic = np.broadcast_to(
        fold_arrays.dynamic_reference_by_landmark[landmark_index][None, :, :],
        dynamic.shape,
    ).copy().astype(np.float32)
    baseline_dynamic[~row_mask] = 0.0

    baseline_static = np.broadcast_to(
        fold_arrays.static_reference_by_landmark[landmark_index][None, :],
        static.shape,
    ).copy().astype(np.float32)

    baseline_age = np.full(
        len(patient_local),
        float(fold_arrays.age_reference_by_landmark[landmark_index]),
        dtype=np.float32,
    )

    # 当前动态值必须使用模型真正使用的“Landmark前最后一个有效历史行”，
    # 而不是机械取landmark_bin所在行。
    active_mask = (
        row_mask
        & (
            np.arange(EXPECTED_HISTORY_STEP_N, dtype=np.int64)[None, :]
            <= landmark_bin[:, None]
        )
    )
    if (~active_mask.any(axis=1)).any():
        raise ValueError("部分样本在Landmark前没有有效历史行。")
    last_position = (
        active_mask.astype(np.int64)
        * (np.arange(EXPECTED_HISTORY_STEP_N, dtype=np.int64)[None, :] + 1)
    ).argmax(axis=1)
    batch_index = np.arange(len(patient_local), dtype=np.int64)
    current_dynamic = dynamic[
        batch_index,
        last_position,
        :EXPECTED_BASE_DYNAMIC_N,
    ].astype(np.float32)

    return {
        "dynamic": dynamic,
        "row_mask": row_mask,
        "static": static,
        "age": age,
        "age_raw": age_raw.astype(np.float32),
        "current_dynamic": current_dynamic,
        "landmark_norm": landmark_norm,
        "landmark_bin": landmark_bin,
        "baseline_dynamic": baseline_dynamic,
        "baseline_static": baseline_static,
        "baseline_age": baseline_age,
    }


# =============================================================================
# 13. Integrated Gradients
# =============================================================================

def integrated_gradients_batch(
    model: AttributionFoldFiveYearRisk,
    inputs: dict[str, np.ndarray],
    device: torch.device,
    steps: int,
) -> dict[str, np.ndarray]:
    """
    同时对dynamic、static、age做IG。
    使用Gauss-Legendre数值积分计算路径积分。
    row_mask、landmark_norm、landmark_bin保持固定，不参与归因。
    """

    dynamic = torch.as_tensor(inputs["dynamic"], dtype=torch.float32, device=device)
    static = torch.as_tensor(inputs["static"], dtype=torch.float32, device=device)
    age = torch.as_tensor(inputs["age"], dtype=torch.float32, device=device)
    baseline_dynamic = torch.as_tensor(
        inputs["baseline_dynamic"],
        dtype=torch.float32,
        device=device,
    )
    baseline_static = torch.as_tensor(
        inputs["baseline_static"],
        dtype=torch.float32,
        device=device,
    )
    baseline_age = torch.as_tensor(
        inputs["baseline_age"],
        dtype=torch.float32,
        device=device,
    )
    row_mask = torch.as_tensor(inputs["row_mask"], dtype=torch.bool, device=device)
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

    dynamic_delta = dynamic - baseline_dynamic
    static_delta = static - baseline_static
    age_delta = age - baseline_age

    grad_dynamic_integral = torch.zeros_like(dynamic)
    grad_static_integral = torch.zeros_like(static)
    grad_age_integral = torch.zeros_like(age)

    # Gauss-Legendre quadrature on [0,1].
    # 相比等距梯形积分，在相同步数下通常具有更好的IG积分精度和
    # completeness，且不依赖Captum。
    nodes, weights = np.polynomial.legendre.leggauss(int(steps))
    alphas_np = ((nodes + 1.0) / 2.0).astype(np.float64)
    weights_np = (weights / 2.0).astype(np.float64)

    for alpha_value, weight_value in zip(alphas_np, weights_np):
        alpha = torch.tensor(
            float(alpha_value),
            dtype=dynamic.dtype,
            device=device,
        )
        weight = float(weight_value)

        interp_dynamic = (
            baseline_dynamic + alpha * dynamic_delta
        ).detach().requires_grad_(True)
        interp_static = (
            baseline_static + alpha * static_delta
        ).detach().requires_grad_(True)
        interp_age = (
            baseline_age + alpha * age_delta
        ).detach().requires_grad_(True)

        risk = model(
            dynamic_sequence=interp_dynamic,
            row_mask=row_mask,
            static_baseline=interp_static,
            age_at_landmark=interp_age,
            landmark_normalized=landmark_norm,
            landmark_bin=landmark_bin,
        )
        gradients = torch.autograd.grad(
            outputs=risk.sum(),
            inputs=(interp_dynamic, interp_static, interp_age),
            create_graph=False,
            retain_graph=False,
            allow_unused=False,
        )

        grad_dynamic_integral += weight * gradients[0]
        grad_static_integral += weight * gradients[1]
        grad_age_integral += weight * gradients[2]

    average_grad_dynamic = grad_dynamic_integral
    average_grad_static = grad_static_integral
    average_grad_age = grad_age_integral

    attr_dynamic = dynamic_delta * average_grad_dynamic
    attr_static = static_delta * average_grad_static
    attr_age = age_delta * average_grad_age

    with torch.no_grad():
        risk_input = model(
            dynamic_sequence=dynamic,
            row_mask=row_mask,
            static_baseline=static,
            age_at_landmark=age,
            landmark_normalized=landmark_norm,
            landmark_bin=landmark_bin,
        )
        risk_baseline = model(
            dynamic_sequence=baseline_dynamic,
            row_mask=row_mask,
            static_baseline=baseline_static,
            age_at_landmark=baseline_age,
            landmark_normalized=landmark_norm,
            landmark_bin=landmark_bin,
        )

    total_attr = (
        attr_dynamic.flatten(1).sum(dim=1)
        + attr_static.sum(dim=1)
        + attr_age
    )
    target_difference = risk_input - risk_baseline
    completeness_residual = target_difference - total_attr

    return {
        "attr_dynamic": attr_dynamic.detach().cpu().numpy().astype(np.float32),
        "attr_static": attr_static.detach().cpu().numpy().astype(np.float32),
        "attr_age": attr_age.detach().cpu().numpy().astype(np.float32),
        "risk_input": risk_input.detach().cpu().numpy().astype(np.float32),
        "risk_baseline": risk_baseline.detach().cpu().numpy().astype(np.float32),
        "completeness_residual": completeness_residual.detach().cpu().numpy().astype(np.float32),
    }


def run_ig_in_batches(
    model: AttributionFoldFiveYearRisk,
    all_inputs: dict[str, np.ndarray],
    device: torch.device,
) -> dict[str, np.ndarray]:
    n = len(all_inputs["age"])
    outputs: dict[str, list[np.ndarray]] = {
        "attr_dynamic": [],
        "attr_static": [],
        "attr_age": [],
        "risk_input": [],
        "risk_baseline": [],
        "completeness_residual": [],
    }

    for start in range(0, n, IG_BATCH_SIZE):
        end = min(start + IG_BATCH_SIZE, n)
        ig_input_keys = [
            "dynamic",
            "row_mask",
            "static",
            "age",
            "landmark_norm",
            "landmark_bin",
            "baseline_dynamic",
            "baseline_static",
            "baseline_age",
        ]
        batch_inputs = {
            key: all_inputs[key][start:end]
            for key in ig_input_keys
        }
        result = integrated_gradients_batch(
            model=model,
            inputs=batch_inputs,
            device=device,
            steps=IG_STEPS,
        )
        for key in outputs:
            outputs[key].append(result[key])
        print(
            f"    IG batch {start + 1}-{end}/{n}完成",
            flush=True,
        )

    return {
        key: np.concatenate(value, axis=0)
        for key, value in outputs.items()
    }


# =============================================================================
# 14. 三层数值审计
# =============================================================================

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


def derive_validation_sample_pairs(
    common: CommonData,
    fold_id: int,
) -> np.ndarray:
    validation_patient_mask = (
        common.development_fold_id == int(fold_id)
    )
    return np.argwhere(
        common.prediction_origin_mask
        & validation_patient_mask[:, None]
    ).astype(np.int32)


def predict_one_snapshot_amp_original_batches(
    model: HybridAttentionLSTMSurvival,
    common: CommonData,
    fold_arrays: FoldArrays,
    sample_pairs: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    """
    严格模仿Step10E predict_loader()：
    - validation顺序不打乱；
    - 使用原batch_size；
    - forward置于AMP autocast中；
    - sigmoid在logits.float()上计算。
    """
    model.eval()
    hazards = []

    with torch.no_grad():
        for start in range(0, len(sample_pairs), int(batch_size)):
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

            with autocast_context(device, enabled=True):
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

            del dynamic, row_mask, static, age
            del landmark_norm, landmark_bin, logits

    return np.concatenate(
        hazards,
        axis=0,
    ).astype(np.float32)


def audit_step10e_fold_reconstruction(
    common: CommonData,
    fold_arrays: FoldArrays,
    fold_id: int,
    device: torch.device,
) -> dict[str, float]:
    """
    审计A：完整复现该fold的Step10E最终validation_hazard。

    关键点：
    - 不能只抽8个患者重新前向；
    - 必须使用原validation sample顺序和原batch size；
    - 必须使用AMP；
    - snapshot和seed平均必须使用NumPy float32 mean。
    """
    fold_dir = STEP10E_DIR / f"fold_{fold_id}"
    checkpoint_paths = sorted(
        fold_dir.glob("seed_*_snapshot_ensemble.pt")
    )
    if not checkpoint_paths:
        raise FileNotFoundError(
            f"Fold {fold_id}没有snapshot ensemble检查点。"
        )

    saved_patient = np.load(
        fold_dir / "validation_patient_local.npy"
    ).astype(np.int32)
    saved_landmark = np.load(
        fold_dir / "validation_landmark_index.npy"
    ).astype(np.int8)
    saved_hazard = np.load(
        fold_dir / "validation_hazard.npy"
    ).astype(np.float32)

    derived_pairs = derive_validation_sample_pairs(
        common,
        fold_id,
    )
    saved_pairs = np.column_stack(
        [saved_patient, saved_landmark]
    ).astype(np.int32)

    if not np.array_equal(
        derived_pairs,
        saved_pairs,
    ):
        raise ValueError(
            f"Fold {fold_id}：当前重建validation sample顺序与Step10E不一致。"
        )

    seed_hazards = []
    reference_parameters = None
    reference_dynamic_names = None
    reference_static_names = None

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

        if dynamic_names != fold_arrays.enhanced_dynamic_names:
            raise ValueError(
                f"Fold {fold_id}检查点动态特征与重建数据不一致。"
            )
        if static_names != fold_arrays.static_names:
            raise ValueError(
                f"Fold {fold_id}检查点静态特征与重建数据不一致。"
            )

        batch_size = int(parameters["batch_size"])
        snapshot_hazards = []

        for snapshot_position, state in enumerate(
            checkpoint["snapshot_state_dicts"]
        ):
            model = build_model(
                parameters,
                device,
            )
            model.load_state_dict(
                state,
                strict=True,
            )
            model.eval()

            snapshot_hazard = (
                predict_one_snapshot_amp_original_batches(
                    model=model,
                    common=common,
                    fold_arrays=fold_arrays,
                    sample_pairs=derived_pairs,
                    batch_size=batch_size,
                    device=device,
                )
            )
            snapshot_hazards.append(
                snapshot_hazard
            )

            del model
            gc.collect()
            torch.cuda.empty_cache()

        # 与Step10E ensemble_snapshot_predictions()一致：
        # np.mean(float32 stack)。
        seed_hazard = np.mean(
            np.stack(
                snapshot_hazards,
                axis=0,
            ),
            axis=0,
        ).astype(np.float32)
        seed_hazards.append(seed_hazard)

    # 与Step10E不同seed最终平均一致。
    reconstructed_hazard = np.mean(
        np.stack(
            seed_hazards,
            axis=0,
        ),
        axis=0,
    ).astype(np.float32)

    diff = np.abs(
        reconstructed_hazard.astype(np.float64)
        - saved_hazard.astype(np.float64)
    )
    max_diff = float(np.max(diff))
    mean_diff = float(np.mean(diff))
    p99_diff = float(np.quantile(diff, 0.99))

    if (
        max_diff > AMP_RECON_MAX_TOL
        or mean_diff > AMP_RECON_MEAN_TOL
    ):
        raise ValueError(
            f"Fold {fold_id}完整AMP模型重建审计未通过："
            f"max={max_diff:.3e}, mean={mean_diff:.3e}, p99={p99_diff:.3e}。"
            "这才表示模型/数据/计算流程可能真正不一致。"
        )

    return {
        "fold_id": int(fold_id),
        "validation_origin_n": int(len(saved_hazard)),
        "amp_reconstruction_max_abs_diff": max_diff,
        "amp_reconstruction_mean_abs_diff": mean_diff,
        "amp_reconstruction_p99_abs_diff": p99_diff,
    }


def audit_step11_saved_prediction_mapping(
    common: CommonData,
    fold_id: int,
    calibration_table: pd.DataFrame,
    calibrated_risk_long: np.ndarray,
) -> dict[str, float]:
    """
    审计B：完全不重新跑神经网络。
    直接用Step10E已经保存的validation_survival，按Step11原代码
    survival->hazard->crossfit calibration->risk，核对Step11保存结果。
    """
    fold_dir = STEP10E_DIR / f"fold_{fold_id}"
    saved_survival = np.load(
        fold_dir / "validation_survival.npy"
    ).astype(np.float32)
    saved_hazard = np.load(
        fold_dir / "validation_hazard.npy"
    ).astype(np.float32)
    validation_long_idx = np.load(
        fold_dir / "validation_long_idx.npy"
    ).astype(np.int32)
    validation_landmark = np.load(
        fold_dir / "validation_landmark_index.npy"
    ).astype(np.int8)

    # 先核对Step10E hazard->survival内部一致性。
    survival_from_hazard = np_step10e_hazard_to_survival(
        saved_hazard
    )
    step10e_internal_diff = float(
        np.max(
            np.abs(
                survival_from_hazard.astype(np.float64)
                - saved_survival.astype(np.float64)
            )
        )
    )

    raw_hazard_step11 = np_step11_survival_to_hazard(
        saved_survival
    )
    calibrated_hazard = np.full_like(
        raw_hazard_step11,
        np.nan,
        dtype=np.float32,
    )

    for landmark_index, landmark_month in enumerate(
        LANDMARK_MONTHS
    ):
        mask = validation_landmark == landmark_index
        if not np.any(mask):
            continue
        alpha, beta = load_crossfit_calibrator(
            calibration_table,
            fold_id=fold_id,
            landmark_month=int(landmark_month),
        )
        calibrated_hazard[mask] = (
            np_step11_apply_calibrator(
                raw_hazard_step11[mask],
                alpha,
                beta,
            )
        )

    if not np.isfinite(calibrated_hazard).all():
        raise ValueError(
            f"Fold {fold_id} Step11映射后存在NaN。"
        )

    reconstructed_risk = np_step11_hazard_to_risk(
        calibrated_hazard
    )
    expected_risk = np.asarray(
        calibrated_risk_long[
            validation_long_idx,
            :,
        ],
        dtype=np.float32,
    )

    mapping_diff = np.abs(
        reconstructed_risk.astype(np.float64)
        - expected_risk.astype(np.float64)
    )
    max_diff = float(np.max(mapping_diff))
    mean_diff = float(np.mean(mapping_diff))

    if max_diff > STEP11_MAPPING_TOL:
        raise ValueError(
            f"Fold {fold_id} Step11校准映射审计未通过："
            f"max={max_diff:.3e}, mean={mean_diff:.3e}。"
        )

    return {
        "fold_id": int(fold_id),
        "step10e_saved_survival_internal_max_abs_diff": step10e_internal_diff,
        "step11_mapping_max_abs_diff": max_diff,
        "step11_mapping_mean_abs_diff": mean_diff,
    }


def audit_ig_fp32_risk_drift(
    model: AttributionFoldFiveYearRisk,
    common: CommonData,
    fold_arrays: FoldArrays,
    landmark_index: int,
    selected_patients: np.ndarray,
    device: torch.device,
    calibrated_risk_long: np.ndarray | None,
) -> dict[str, float]:
    """
    审计C：比较“IG使用的全精度可微函数”与Step11当时保存的AMP预测。
    这不是模型加载正确性测试，只量化解释函数相对保存预测的数值漂移。
    """
    verify_patients = np.asarray(
        selected_patients[: min(16, len(selected_patients))],
        dtype=np.int64,
    )
    inputs = build_inputs_for_pairs(
        common=common,
        fold_arrays=fold_arrays,
        patient_local=verify_patients,
        landmark_index=landmark_index,
    )

    device_inputs = {
        "dynamic_sequence": torch.as_tensor(
            inputs["dynamic"],
            dtype=torch.float32,
            device=device,
        ),
        "row_mask": torch.as_tensor(
            inputs["row_mask"],
            dtype=torch.bool,
            device=device,
        ),
        "static_baseline": torch.as_tensor(
            inputs["static"],
            dtype=torch.float32,
            device=device,
        ),
        "age_at_landmark": torch.as_tensor(
            inputs["age"],
            dtype=torch.float32,
            device=device,
        ),
        "landmark_normalized": torch.as_tensor(
            inputs["landmark_norm"],
            dtype=torch.float32,
            device=device,
        ),
        "landmark_bin": torch.as_tensor(
            inputs["landmark_bin"],
            dtype=torch.long,
            device=device,
        ),
    }

    with torch.no_grad():
        ig_risk5 = (
            model(**device_inputs)
            .cpu()
            .numpy()
            .astype(np.float64)
        )

    if EXPLAIN_CALIBRATED:
        if calibrated_risk_long is None:
            raise ValueError(
                "要求校准风险漂移审计，但没有Step11校准风险。"
            )
        long_idx = common.long_row_index_map[
            verify_patients,
            landmark_index,
        ]
        expected = np.asarray(
            calibrated_risk_long[
                long_idx,
                -1,
            ],
            dtype=np.float64,
        )
    else:
        # raw模式暂不使用该审计分支。
        expected = ig_risk5.copy()

    diff = np.abs(ig_risk5 - expected)
    max_diff = float(np.max(diff))
    mean_diff = float(np.mean(diff))

    if max_diff > IG_FP32_RISK_DRIFT_HARD_TOL:
        raise ValueError(
            f"Landmark {LANDMARK_MONTHS[landmark_index]}月IG全精度风险"
            f"与正式保存风险差异过大：max={max_diff:.3e}。"
            "这已经超出普通AMP数值漂移范围，需要停止检查。"
        )

    return {
        "ig_fp32_vs_saved_risk_max_abs_diff": max_diff,
        "ig_fp32_vs_saved_risk_mean_abs_diff": mean_diff,
    }


# =============================================================================
# 15. 临床变量聚合规则
# =============================================================================

# -----------------------------------------------------------------------------
# 强化动态输入 -> 临床变量：使用“精确映射”，禁止依赖suffix先后顺序。
#
# 旧版错误原因："*_delta_last_observed" 同时也以 "_observed" 结尾，
# 如果先检查 "_observed"，会被错误截成 "*_delta_last"。
# 这里彻底取消这种启发式字符串截断。
# -----------------------------------------------------------------------------
EXPECTED_ENHANCED_DYNAMIC_NAMES = (
    DYNAMIC_FEATURES
    + [f"{name}_observed" for name in LAB_FEATURES]
    + [f"{name}_time_since_last" for name in LAB_FEATURES]
    + [f"{name}_delta_last_observed" for name in LAB_FEATURES]
)

DYNAMIC_COMPONENT_TO_CLINICAL: dict[str, str] = {
    name: name for name in DYNAMIC_FEATURES
}
for _lab_name in LAB_FEATURES:
    DYNAMIC_COMPONENT_TO_CLINICAL[f"{_lab_name}_observed"] = _lab_name
    DYNAMIC_COMPONENT_TO_CLINICAL[f"{_lab_name}_time_since_last"] = _lab_name
    DYNAMIC_COMPONENT_TO_CLINICAL[f"{_lab_name}_delta_last_observed"] = _lab_name

if len(EXPECTED_ENHANCED_DYNAMIC_NAMES) != EXPECTED_ENHANCED_DYNAMIC_N:
    raise RuntimeError("预期强化动态特征名称数不是85。")
if len(set(EXPECTED_ENHANCED_DYNAMIC_NAMES)) != EXPECTED_ENHANCED_DYNAMIC_N:
    raise RuntimeError("预期强化动态特征名称存在重复。")
if set(DYNAMIC_COMPONENT_TO_CLINICAL) != set(EXPECTED_ENHANCED_DYNAMIC_NAMES):
    raise RuntimeError("强化动态特征精确映射表不完整。")


def dynamic_component_to_clinical(name: str) -> str:
    try:
        return DYNAMIC_COMPONENT_TO_CLINICAL[name]
    except KeyError as exc:
        raise ValueError(
            f"未知强化动态输入组件：{name}。"
            "该名称不在Step10E正式85项输入定义中。"
        ) from exc


def validate_dynamic_feature_partition(dynamic_names: list[str]) -> None:
    """在聚合前一次性审计85项输入名称与临床变量分组。"""
    if len(dynamic_names) != EXPECTED_ENHANCED_DYNAMIC_N:
        raise ValueError(
            f"强化动态特征数={len(dynamic_names)}，应为{EXPECTED_ENHANCED_DYNAMIC_N}。"
        )
    if len(set(dynamic_names)) != len(dynamic_names):
        raise ValueError("强化动态特征名称存在重复。")
    if list(dynamic_names) != list(EXPECTED_ENHANCED_DYNAMIC_NAMES):
        missing = sorted(set(EXPECTED_ENHANCED_DYNAMIC_NAMES) - set(dynamic_names))
        extra = sorted(set(dynamic_names) - set(EXPECTED_ENHANCED_DYNAMIC_NAMES))
        raise ValueError(
            "强化动态特征名称或顺序与Step10E正式定义不一致。"
            f" missing={missing}, extra={extra}"
        )

    group_counts: dict[str, int] = {}
    for component in dynamic_names:
        clinical = dynamic_component_to_clinical(component)
        group_counts[clinical] = group_counts.get(clinical, 0) + 1

    expected_clinical = set(DYNAMIC_FEATURES)
    if set(group_counts) != expected_clinical:
        raise ValueError("强化动态输入聚合后的临床变量集合不正确。")

    for clinical in DYNAMIC_FEATURES:
        expected_n = 4 if clinical in LAB_FEATURES else 1
        actual_n = int(group_counts.get(clinical, 0))
        if actual_n != expected_n:
            raise ValueError(
                f"临床动态变量{clinical}对应输入组件数={actual_n}，"
                f"应为{expected_n}。"
            )


def static_component_to_clinical(name: str) -> str:
    if name.startswith("Sex_"):
        return "Sex"
    if name.startswith("Marriage_"):
        return "Marriage"
    if name.startswith("Course_"):
        return "Course"
    if name.startswith("WHOstage_"):
        return "WHOstage"
    return name


def aggregate_ig_results(
    raw_frames: list[dict[str, Any]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    返回：
      1. sample_clinical_df：每个样本-临床变量 signed/absolute IG
      2. component_df：输入组件级总体重要性
      3. clinical_df：临床变量级总体重要性
      4. temporal_df：动态临床变量 × 历史时间 importance
    """

    sample_clinical_rows = []
    component_rows = []
    temporal_rows = []
    completeness_rows = []

    for block in raw_frames:
        patient_local = block["patient_local"]
        development_global_index = block["development_global_index"]
        fold_id = int(block["fold_id"])
        landmark_index = int(block["landmark_index"])
        landmark_month = int(LANDMARK_MONTHS[landmark_index])
        landmark_year = landmark_month / 12.0
        dynamic_names = block["dynamic_names"]
        static_names = block["static_names"]
        attr_dynamic = block["attr_dynamic"]
        attr_static = block["attr_static"]
        attr_age = block["attr_age"]
        current_dynamic = block["current_dynamic"]
        static_input = block["static_input"]
        age_raw_values = block["age_raw"]
        risk_input = block["risk_input"]
        risk_baseline = block["risk_baseline"]
        residual = block["completeness_residual"]

        if current_dynamic.shape != (len(patient_local), EXPECTED_BASE_DYNAMIC_N):
            raise ValueError("current_dynamic形状错误。")
        if static_input.shape != (len(patient_local), EXPECTED_STATIC_N):
            raise ValueError("static_input形状错误。")
        if age_raw_values.shape != (len(patient_local),):
            raise ValueError("age_raw形状错误。")

        dynamic_value_index = {
            name: index
            for index, name in enumerate(DYNAMIC_FEATURES)
        }

        validate_dynamic_feature_partition(dynamic_names)

        dynamic_group_map: dict[str, list[int]] = {}
        for feature_index, feature_name in enumerate(dynamic_names):
            clinical_name = dynamic_component_to_clinical(feature_name)
            dynamic_group_map.setdefault(clinical_name, []).append(feature_index)

        static_group_map: dict[str, list[int]] = {}
        for feature_index, feature_name in enumerate(static_names):
            clinical_name = static_component_to_clinical(feature_name)
            static_group_map.setdefault(clinical_name, []).append(feature_index)

        # ---------------------------------------------------------------------
        # 每个样本临床变量贡献
        # ---------------------------------------------------------------------
        for sample_index in range(len(patient_local)):
            metadata = {
                "patient_local": int(patient_local[sample_index]),
                "development_global_index": int(development_global_index[sample_index]),
                "fold_id": fold_id,
                "landmark_index": landmark_index,
                "landmark_month": landmark_month,
                "landmark_year": landmark_year,
                "predicted_5y_risk": float(risk_input[sample_index]),
                "reference_5y_risk": float(risk_baseline[sample_index]),
            }

            for clinical_name, indices in dynamic_group_map.items():
                values = attr_dynamic[sample_index, :, indices]
                if clinical_name not in dynamic_value_index:
                    raise ValueError(
                        f"临床动态变量{clinical_name}无法映射回基础模型输入。"
                    )
                current_model_value = float(
                    current_dynamic[
                        sample_index,
                        dynamic_value_index[clinical_name],
                    ]
                )
                sample_clinical_rows.append(
                    {
                        **metadata,
                        "clinical_variable": clinical_name,
                        "source": "dynamic",
                        "signed_ig": float(values.sum()),
                        "absolute_ig": float(np.abs(values).sum()),
                        "current_model_value": current_model_value,
                        "value_definition": "current_model_input_at_landmark",
                    }
                )

            for clinical_name, indices in static_group_map.items():
                values = attr_static[sample_index, indices]
                if len(indices) == 1:
                    current_model_value = float(
                        static_input[sample_index, indices[0]]
                    )
                    value_definition = "baseline_model_input"
                else:
                    # Sex/Marriage/Course/WHOstage为多个one-hot维度，
                    # 聚合后没有自然的连续高低顺序，因此不用于beeswarm颜色。
                    current_model_value = np.nan
                    value_definition = "unordered_categorical_group"
                sample_clinical_rows.append(
                    {
                        **metadata,
                        "clinical_variable": clinical_name,
                        "source": "static",
                        "signed_ig": float(values.sum()),
                        "absolute_ig": float(np.abs(values).sum()),
                        "current_model_value": current_model_value,
                        "value_definition": value_definition,
                    }
                )

            sample_clinical_rows.append(
                {
                    **metadata,
                    "clinical_variable": "Age",
                    "source": "age",
                    "signed_ig": float(attr_age[sample_index]),
                    "absolute_ig": float(abs(attr_age[sample_index])),
                    "current_model_value": float(age_raw_values[sample_index]),
                    "value_definition": "age_years_at_landmark",
                }
            )

            completeness_rows.append(
                {
                    **metadata,
                    "risk_difference_input_minus_reference": float(
                        risk_input[sample_index] - risk_baseline[sample_index]
                    ),
                    "ig_completeness_residual": float(residual[sample_index]),
                    "ig_completeness_abs_residual": float(abs(residual[sample_index])),
                }
            )

        # ---------------------------------------------------------------------
        # 输入组件级重要性
        # ---------------------------------------------------------------------
        for feature_index, feature_name in enumerate(dynamic_names):
            sample_signed = attr_dynamic[:, :, feature_index].sum(axis=1)
            sample_abs = np.abs(attr_dynamic[:, :, feature_index]).sum(axis=1)
            component_rows.append(
                {
                    "fold_id": fold_id,
                    "landmark_index": landmark_index,
                    "landmark_month": landmark_month,
                    "landmark_year": landmark_year,
                    "input_source": "dynamic",
                    "input_component": feature_name,
                    "clinical_variable": dynamic_component_to_clinical(feature_name),
                    "sample_n": len(sample_signed),
                    "sum_abs_ig": float(sample_abs.sum()),
                    "sum_signed_ig": float(sample_signed.sum()),
                }
            )

        for feature_index, feature_name in enumerate(static_names):
            sample_signed = attr_static[:, feature_index]
            sample_abs = np.abs(attr_static[:, feature_index])
            component_rows.append(
                {
                    "fold_id": fold_id,
                    "landmark_index": landmark_index,
                    "landmark_month": landmark_month,
                    "landmark_year": landmark_year,
                    "input_source": "static",
                    "input_component": feature_name,
                    "clinical_variable": static_component_to_clinical(feature_name),
                    "sample_n": len(sample_signed),
                    "sum_abs_ig": float(sample_abs.sum()),
                    "sum_signed_ig": float(sample_signed.sum()),
                }
            )

        component_rows.append(
            {
                "fold_id": fold_id,
                "landmark_index": landmark_index,
                "landmark_month": landmark_month,
                "landmark_year": landmark_year,
                "input_source": "age",
                "input_component": "Age",
                "clinical_variable": "Age",
                "sample_n": len(attr_age),
                "sum_abs_ig": float(np.abs(attr_age).sum()),
                "sum_signed_ig": float(attr_age.sum()),
            }
        )

        # ---------------------------------------------------------------------
        # dynamic clinical variable × historical time
        # ---------------------------------------------------------------------
        max_active_step = int(LANDMARK_BINS[landmark_index])
        for step_index in range(max_active_step + 1):
            history_month = int(step_index * 6)
            for clinical_name, indices in dynamic_group_map.items():
                values = attr_dynamic[:, step_index, indices]
                per_sample_signed = values.sum(axis=1)
                per_sample_abs = np.abs(values).sum(axis=1)
                temporal_rows.append(
                    {
                        "fold_id": fold_id,
                        "landmark_index": landmark_index,
                        "landmark_month": landmark_month,
                        "landmark_year": landmark_year,
                        "history_month": history_month,
                        "history_year": history_month / 12.0,
                        "clinical_variable": clinical_name,
                        "sample_n": len(per_sample_signed),
                        "sum_abs_ig": float(per_sample_abs.sum()),
                        "sum_signed_ig": float(per_sample_signed.sum()),
                    }
                )

    sample_clinical_df = pd.DataFrame(sample_clinical_rows)
    component_block_df = pd.DataFrame(component_rows)
    temporal_block_df = pd.DataFrame(temporal_rows)
    completeness_df = pd.DataFrame(completeness_rows)

    # -------------------------------------------------------------------------
    # 临床变量级总体结果：先在sample级聚合，再按Landmark汇总。
    # -------------------------------------------------------------------------
    clinical_df = (
        sample_clinical_df.groupby(
            ["landmark_index", "landmark_month", "landmark_year", "clinical_variable", "source"],
            as_index=False,
        )
        .agg(
            sample_n=("patient_local", "size"),
            mean_abs_ig=("absolute_ig", "mean"),
            median_abs_ig=("absolute_ig", "median"),
            mean_signed_ig=("signed_ig", "mean"),
            median_signed_ig=("signed_ig", "median"),
            positive_fraction=("signed_ig", lambda x: float(np.mean(np.asarray(x) > 0))),
            negative_fraction=("signed_ig", lambda x: float(np.mean(np.asarray(x) < 0))),
        )
    )
    clinical_df["importance_rank"] = (
        clinical_df.groupby("landmark_index")["mean_abs_ig"]
        .rank(method="first", ascending=False)
        .astype(int)
    )
    clinical_df["relative_importance"] = (
        clinical_df["mean_abs_ig"]
        / clinical_df.groupby("landmark_index")["mean_abs_ig"].transform("sum")
    )
    clinical_df["relative_importance_pct"] = (
        clinical_df["relative_importance"] * 100.0
    )
    clinical_df = clinical_df.sort_values(
        ["landmark_index", "importance_rank", "clinical_variable"]
    ).reset_index(drop=True)

    # -------------------------------------------------------------------------
    # 输入组件级：折块求和后除以样本量。
    # -------------------------------------------------------------------------
    component_df = (
        component_block_df.groupby(
            [
                "landmark_index",
                "landmark_month",
                "landmark_year",
                "input_source",
                "input_component",
                "clinical_variable",
            ],
            as_index=False,
        )
        .agg(
            sample_n=("sample_n", "sum"),
            total_abs_ig=("sum_abs_ig", "sum"),
            total_signed_ig=("sum_signed_ig", "sum"),
        )
    )
    component_df["mean_abs_ig"] = component_df["total_abs_ig"] / component_df["sample_n"]
    component_df["mean_signed_ig"] = component_df["total_signed_ig"] / component_df["sample_n"]

    # -------------------------------------------------------------------------
    # 时间归因：折块求和后除以样本量。
    # -------------------------------------------------------------------------
    temporal_df = (
        temporal_block_df.groupby(
            [
                "landmark_index",
                "landmark_month",
                "landmark_year",
                "history_month",
                "history_year",
                "clinical_variable",
            ],
            as_index=False,
        )
        .agg(
            sample_n=("sample_n", "sum"),
            total_abs_ig=("sum_abs_ig", "sum"),
            total_signed_ig=("sum_signed_ig", "sum"),
        )
    )
    temporal_df["mean_abs_ig"] = temporal_df["total_abs_ig"] / temporal_df["sample_n"]
    temporal_df["mean_signed_ig"] = temporal_df["total_signed_ig"] / temporal_df["sample_n"]

    return sample_clinical_df, component_df, clinical_df, temporal_df, completeness_df


# =============================================================================
# 16. 图形
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


def save_figure(fig: plt.Figure, stem: str) -> None:
    for extension in ["png", "pdf", "svg"]:
        kwargs = {"bbox_inches": "tight", "facecolor": "white"}
        if extension == "png":
            kwargs["dpi"] = 600
        fig.savefig(FIGURE_DIR / f"Step12I_{stem}.{extension}", **kwargs)
    plt.close(fig)


def plot_global_importance(clinical_df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 10), constrained_layout=True)
    axes = axes.ravel()
    for panel_index, landmark_index in enumerate(PRIMARY_LANDMARK_INDICES):
        ax = axes[panel_index]
        sub = clinical_df.loc[
            clinical_df["landmark_index"] == int(landmark_index)
        ].nsmallest(TOP_FEATURE_N, "importance_rank")
        sub = sub.sort_values("mean_abs_ig", ascending=True)
        ax.barh(sub["clinical_variable"], sub["mean_abs_ig"])
        landmark_year = LANDMARK_MONTHS[landmark_index] / 12.0
        title = "Baseline" if landmark_year == 0 else f"ART year {landmark_year:.0f}"
        ax.set_title(f"{title} landmark", fontweight="bold")
        ax.set_xlabel("Mean absolute IG attribution to 5-year CKD risk")
        ax.grid(axis="x", linestyle="--", alpha=0.25)
        ax.text(
            -0.12,
            1.05,
            chr(ord("A") + panel_index),
            transform=ax.transAxes,
            fontsize=14,
            fontweight="bold",
            color="black",
        )
    fig.suptitle(
        "Global LSTM feature attribution by prediction landmark",
        fontsize=15,
        fontweight="bold",
        y=1.02,
    )
    save_figure(fig, "Figure12A_global_clinical_importance")


def plot_signed_importance(clinical_df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 10), constrained_layout=True)
    axes = axes.ravel()
    for panel_index, landmark_index in enumerate(PRIMARY_LANDMARK_INDICES):
        ax = axes[panel_index]
        importance = clinical_df.loc[
            clinical_df["landmark_index"] == int(landmark_index)
        ].nsmallest(TOP_FEATURE_N, "importance_rank")
        importance = importance.sort_values("mean_signed_ig")
        values = importance["mean_signed_ig"].to_numpy(float)
        ax.barh(importance["clinical_variable"], values)
        ax.axvline(0.0, color="black", linewidth=1.0)
        landmark_year = LANDMARK_MONTHS[landmark_index] / 12.0
        title = "Baseline" if landmark_year == 0 else f"ART year {landmark_year:.0f}"
        ax.set_title(f"{title} landmark", fontweight="bold")
        ax.set_xlabel("Mean signed IG attribution to 5-year CKD risk")
        ax.grid(axis="x", linestyle="--", alpha=0.25)
        ax.text(
            -0.12,
            1.05,
            chr(ord("A") + panel_index),
            transform=ax.transAxes,
            fontsize=14,
            fontweight="bold",
            color="black",
        )
    fig.suptitle(
        "Direction of LSTM feature attribution",
        fontsize=15,
        fontweight="bold",
        y=1.02,
    )
    save_figure(fig, "Figure12A_signed_clinical_attribution")


def plot_temporal_heatmaps(
    clinical_df: pd.DataFrame,
    temporal_df: pd.DataFrame,
) -> None:
    # baseline只有一个历史点，因此主时间图展示1/3/5年Landmark。
    selected_landmarks = [1, 3, 5]
    fig, axes = plt.subplots(1, 3, figsize=(18, 7), constrained_layout=True)

    for panel_index, landmark_index in enumerate(selected_landmarks):
        ax = axes[panel_index]
        dynamic_global = clinical_df.loc[
            (clinical_df["landmark_index"] == landmark_index)
            & (clinical_df["source"] == "dynamic")
        ].nsmallest(12, "importance_rank")
        top_variables = dynamic_global["clinical_variable"].tolist()

        sub = temporal_df.loc[
            (temporal_df["landmark_index"] == landmark_index)
            & (temporal_df["clinical_variable"].isin(top_variables))
        ].copy()
        pivot = sub.pivot_table(
            index="clinical_variable",
            columns="history_month",
            values="mean_abs_ig",
            aggfunc="mean",
        )
        # 按global importance排序
        pivot = pivot.reindex(top_variables)
        matrix = pivot.to_numpy(dtype=float)

        im = ax.imshow(matrix, aspect="auto", interpolation="nearest")
        ax.set_yticks(np.arange(len(pivot.index)))
        ax.set_yticklabels(pivot.index)
        ax.set_xticks(np.arange(len(pivot.columns)))
        ax.set_xticklabels([f"{int(x)}" for x in pivot.columns], rotation=45, ha="right")
        ax.set_xlabel("Historical time since ART initiation (months)")
        ax.set_title(
            f"ART year {LANDMARK_MONTHS[landmark_index] / 12:.0f} landmark",
            fontweight="bold",
        )
        ax.text(
            -0.12,
            1.05,
            chr(ord("A") + panel_index),
            transform=ax.transAxes,
            fontsize=14,
            fontweight="bold",
            color="black",
        )
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Mean absolute IG attribution")

    fig.suptitle(
        "Temporal attribution of longitudinal information",
        fontsize=15,
        fontweight="bold",
        y=1.03,
    )
    save_figure(fig, "Figure12A_temporal_attribution_heatmaps")


# =============================================================================
# 17. 正式IG二次整理与论文图
# =============================================================================

def add_within_variable_value_percentile(
    sample_clinical_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    对有自然数值顺序的变量，在每个Landmark-变量内将当前模型输入转换为
    0-1百分位，仅用于beeswarm颜色。IG值本身完全不改变。
    """
    table = sample_clinical_df.copy()
    table["current_value_percentile"] = np.nan

    for (_, _), index in table.groupby(
        ["landmark_index", "clinical_variable"]
    ).groups.items():
        idx = np.asarray(list(index), dtype=np.int64)
        values = table.loc[idx, "current_model_value"].to_numpy(dtype=float)
        finite = np.isfinite(values)
        if finite.sum() < 2:
            continue
        finite_idx = idx[finite]
        finite_values = values[finite]
        # 平均秩避免大量二元值因ties得到任意顺序。
        ranks = pd.Series(finite_values).rank(method="average").to_numpy(dtype=float)
        percentile = (ranks - 1.0) / max(len(ranks) - 1.0, 1.0)
        table.loc[finite_idx, "current_value_percentile"] = percentile

    return table


def build_landmark_evolution_table(
    clinical_df: pd.DataFrame,
) -> pd.DataFrame:
    selected = clinical_df.loc[
        clinical_df["landmark_index"].isin(PRIMARY_LANDMARK_INDICES)
    ].copy()
    overview = (
        selected.groupby("clinical_variable", as_index=False)
        .agg(
            mean_relative_importance_pct=("relative_importance_pct", "mean"),
            mean_rank=("importance_rank", "mean"),
            best_rank=("importance_rank", "min"),
            worst_rank=("importance_rank", "max"),
        )
        .sort_values(
            ["mean_relative_importance_pct", "mean_rank"],
            ascending=[False, True],
        )
        .reset_index(drop=True)
    )
    overview["overall_rank"] = np.arange(1, len(overview) + 1)

    long = selected.merge(
        overview[["clinical_variable", "overall_rank"]],
        on="clinical_variable",
        how="left",
        validate="many_to_one",
    )
    return long.sort_values(
        ["overall_rank", "landmark_year"]
    ).reset_index(drop=True)


LAB_COMPONENT_TYPE_MAP: dict[tuple[str, str], str] = {}
for _lab_name in LAB_FEATURES:
    LAB_COMPONENT_TYPE_MAP[(_lab_name, _lab_name)] = "Value"
    LAB_COMPONENT_TYPE_MAP[(f"{_lab_name}_observed", _lab_name)] = "Observed indicator"
    LAB_COMPONENT_TYPE_MAP[(f"{_lab_name}_time_since_last", _lab_name)] = "Time since last"
    LAB_COMPONENT_TYPE_MAP[(f"{_lab_name}_delta_last_observed", _lab_name)] = "Delta from last observed"


def component_type_for_lab(
    input_component: str,
    clinical_variable: str,
) -> str:
    key = (str(input_component), str(clinical_variable))
    if key not in LAB_COMPONENT_TYPE_MAP:
        raise ValueError(
            f"实验室组件映射失败：input_component={input_component}, "
            f"clinical_variable={clinical_variable}。"
        )
    return LAB_COMPONENT_TYPE_MAP[key]


def build_lab_component_decomposition(
    component_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    lab = component_df.loc[
        component_df["clinical_variable"].isin(LAB_FEATURES)
    ].copy()
    lab["component_type"] = [
        component_type_for_lab(component, clinical)
        for component, clinical in zip(
            lab["input_component"],
            lab["clinical_variable"],
        )
    ]

    expected_types = {
        "Value",
        "Observed indicator",
        "Time since last",
        "Delta from last observed",
    }
    observed_types = set(lab["component_type"].unique())
    if observed_types != expected_types:
        raise ValueError(
            "实验室组件类型集合不完整。"
            f" observed={sorted(observed_types)}, expected={sorted(expected_types)}"
        )

    # 每个Landmark × 每个实验室指标必须恰好包含4个组件，且无重复。
    component_count = (
        lab.groupby(
            ["landmark_index", "clinical_variable"],
            as_index=False,
        )["component_type"]
        .nunique()
    )
    bad = component_count.loc[component_count["component_type"] != 4]
    if not bad.empty:
        raise ValueError(
            "部分Landmark-实验室变量没有完整的4类IG组件：\n"
            + bad.to_string(index=False)
        )

    duplicate_check = lab.duplicated(
        ["landmark_index", "clinical_variable", "component_type"],
        keep=False,
    )
    if duplicate_check.any():
        raise ValueError(
            "实验室组件分解出现重复Landmark-变量-组件记录。"
        )

    total = lab.groupby(
        ["landmark_index", "landmark_month", "landmark_year", "clinical_variable"]
    )["mean_abs_ig"].transform("sum")
    lab["component_share"] = np.divide(
        lab["mean_abs_ig"].to_numpy(dtype=float),
        total.to_numpy(dtype=float),
        out=np.zeros(len(lab), dtype=float),
        where=total.to_numpy(dtype=float) > 0,
    )
    lab["component_share_pct"] = lab["component_share"] * 100.0

    summary_rows = []
    for key, sub in lab.groupby(
        ["landmark_index", "landmark_month", "landmark_year", "clinical_variable"]
    ):
        value_abs = float(
            sub.loc[sub["component_type"] == "Value", "mean_abs_ig"].sum()
        )
        measurement_abs = float(
            sub.loc[sub["component_type"] != "Value", "mean_abs_ig"].sum()
        )
        total_abs = value_abs + measurement_abs
        summary_rows.append(
            {
                "landmark_index": int(key[0]),
                "landmark_month": int(key[1]),
                "landmark_year": float(key[2]),
                "clinical_variable": str(key[3]),
                "value_mean_abs_ig": value_abs,
                "measurement_information_mean_abs_ig": measurement_abs,
                "total_mean_abs_ig": total_abs,
                "value_share_pct": 100.0 * value_abs / total_abs if total_abs > 0 else np.nan,
                "measurement_information_share_pct": 100.0 * measurement_abs / total_abs if total_abs > 0 else np.nan,
            }
        )
    summary = pd.DataFrame(summary_rows)
    return lab, summary


def plot_ig_beeswarm(
    sample_clinical_df: pd.DataFrame,
    clinical_df: pd.DataFrame,
) -> None:
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(14, 12),
        constrained_layout=True,
    )
    axes = axes.ravel()
    rng = np.random.default_rng(RANDOM_SEED + 1201)

    for panel_index, landmark_index in enumerate(PRIMARY_LANDMARK_INDICES):
        ax = axes[panel_index]
        top = clinical_df.loc[
            clinical_df["landmark_index"] == int(landmark_index)
        ].nsmallest(BEESWARM_TOP_N, "importance_rank")
        variables = top["clinical_variable"].tolist()

        # Top-ranked variable at the top.
        y_positions = np.arange(len(variables))[::-1]
        for y, variable in zip(y_positions, variables):
            sub = sample_clinical_df.loc[
                (sample_clinical_df["landmark_index"] == int(landmark_index))
                & (sample_clinical_df["clinical_variable"] == variable)
            ]
            x = sub["signed_ig"].to_numpy(dtype=float)
            percentile = sub["current_value_percentile"].to_numpy(dtype=float)
            jitter = rng.normal(0.0, 0.10, size=len(sub))
            y_value = y + jitter

            finite_color = np.isfinite(percentile)
            if finite_color.any():
                ax.scatter(
                    x[finite_color],
                    y_value[finite_color],
                    c=percentile[finite_color],
                    cmap="coolwarm",
                    vmin=0.0,
                    vmax=1.0,
                    s=10,
                    alpha=0.58,
                    linewidths=0,
                    rasterized=True,
                )
            if (~finite_color).any():
                ax.scatter(
                    x[~finite_color],
                    y_value[~finite_color],
                    c="0.55",
                    s=10,
                    alpha=0.45,
                    linewidths=0,
                    rasterized=True,
                )

        ax.axvline(0.0, color="black", linewidth=0.9)
        ax.set_yticks(y_positions)
        ax.set_yticklabels(variables)
        ax.set_xlabel("Signed IG attribution to predicted 5-year CKD risk")
        landmark_year = LANDMARK_MONTHS[landmark_index] / 12.0
        title = "Baseline" if landmark_year == 0 else f"ART year {landmark_year:.0f}"
        ax.set_title(f"{title} landmark", fontweight="bold")
        ax.grid(axis="x", linestyle="--", alpha=0.18)
        ax.text(
            -0.12,
            1.04,
            chr(ord("A") + panel_index),
            transform=ax.transAxes,
            fontsize=14,
            fontweight="bold",
            color="black",
        )

    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    sm = ScalarMappable(norm=Normalize(0.0, 1.0), cmap="coolwarm")
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes.tolist(), fraction=0.018, pad=0.02)
    cbar.set_label(
        "Current feature value within variable: low → high\n"
        "(grey = unordered categorical group)"
    )
    fig.suptitle(
        "Integrated Gradients distribution and direction of feature contributions",
        fontsize=15,
        fontweight="bold",
        y=1.015,
    )
    save_figure(fig, "Figure12A4_IG_beeswarm_0_1_3_5y")


def plot_landmark_importance_evolution(
    evolution_df: pd.DataFrame,
) -> None:
    top_variables = (
        evolution_df[["clinical_variable", "overall_rank"]]
        .drop_duplicates()
        .nsmallest(EVOLUTION_TOP_N, "overall_rank")["clinical_variable"]
        .tolist()
    )

    fig, ax = plt.subplots(figsize=(10.5, 7.2), constrained_layout=True)
    for variable in top_variables:
        sub = evolution_df.loc[
            evolution_df["clinical_variable"] == variable
        ].sort_values("landmark_year")
        ax.plot(
            sub["landmark_year"],
            sub["relative_importance_pct"],
            marker="o",
            linewidth=2.0,
            markersize=5.0,
            label=variable,
        )

    ax.set_xticks([0, 1, 3, 5])
    ax.set_xlabel("Prediction landmark (years after ART initiation)")
    ax.set_ylabel("Relative IG importance within landmark (%)")
    ax.set_title(
        "Evolution of relative feature importance across prediction landmarks",
        fontweight="bold",
    )
    ax.grid(axis="both", linestyle="--", alpha=0.22)
    ax.legend(
        frameon=False,
        bbox_to_anchor=(1.02, 1.0),
        loc="upper left",
        title=f"Top {EVOLUTION_TOP_N}",
    )
    save_figure(fig, "Figure12A5_landmark_importance_evolution")


def plot_lab_component_decomposition(
    clinical_df: pd.DataFrame,
    lab_component_df: pd.DataFrame,
) -> None:
    component_order = [
        "Value",
        "Observed indicator",
        "Time since last",
        "Delta from last observed",
    ]

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(14, 11),
        constrained_layout=True,
    )
    axes = axes.ravel()

    for panel_index, landmark_index in enumerate(PRIMARY_LANDMARK_INDICES):
        ax = axes[panel_index]
        lab_global = clinical_df.loc[
            (clinical_df["landmark_index"] == int(landmark_index))
            & (clinical_df["clinical_variable"].isin(LAB_FEATURES))
        ].nsmallest(LAB_COMPONENT_TOP_N, "importance_rank")
        variables = lab_global["clinical_variable"].tolist()[::-1]

        y = np.arange(len(variables))
        left = np.zeros(len(variables), dtype=float)
        for component_type in component_order:
            widths = []
            for variable in variables:
                match = lab_component_df.loc[
                    (lab_component_df["landmark_index"] == int(landmark_index))
                    & (lab_component_df["clinical_variable"] == variable)
                    & (lab_component_df["component_type"] == component_type),
                    "component_share_pct",
                ]
                widths.append(float(match.iloc[0]) if len(match) else 0.0)
            widths = np.asarray(widths, dtype=float)
            ax.barh(
                y,
                widths,
                left=left,
                label=component_type,
            )
            left += widths

        ax.set_yticks(y)
        ax.set_yticklabels(variables)
        ax.set_xlim(0, 100)
        ax.set_xlabel("Share of laboratory-variable IG importance (%)")
        landmark_year = LANDMARK_MONTHS[landmark_index] / 12.0
        title = "Baseline" if landmark_year == 0 else f"ART year {landmark_year:.0f}"
        ax.set_title(f"{title} landmark", fontweight="bold")
        ax.grid(axis="x", linestyle="--", alpha=0.18)
        ax.text(
            -0.12,
            1.04,
            chr(ord("A") + panel_index),
            transform=ax.transAxes,
            fontsize=14,
            fontweight="bold",
            color="black",
        )

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, -0.01),
    )
    fig.suptitle(
        "Decomposition of laboratory-variable attribution into value and measurement information",
        fontsize=15,
        fontweight="bold",
        y=1.015,
    )
    save_figure(fig, "Figure12A6_lab_component_decomposition")


def binned_median_xy(
    x: np.ndarray,
    y: np.ndarray,
    bins: int = 12,
) -> tuple[np.ndarray, np.ndarray]:
    finite = np.isfinite(x) & np.isfinite(y)
    x = np.asarray(x[finite], dtype=float)
    y = np.asarray(y[finite], dtype=float)
    if len(x) < max(20, bins * 2):
        return np.asarray([]), np.asarray([])

    quantiles = np.linspace(0.0, 1.0, bins + 1)
    edges = np.unique(np.quantile(x, quantiles))
    if len(edges) < 4:
        return np.asarray([]), np.asarray([])

    centers = []
    medians = []
    for index in range(len(edges) - 1):
        if index == len(edges) - 2:
            mask = (x >= edges[index]) & (x <= edges[index + 1])
        else:
            mask = (x >= edges[index]) & (x < edges[index + 1])
        if mask.sum() < 3:
            continue
        centers.append(float(np.median(x[mask])))
        medians.append(float(np.median(y[mask])))
    return np.asarray(centers), np.asarray(medians)


def plot_selected_dependence(
    sample_clinical_df: pd.DataFrame,
) -> None:
    fig, axes = plt.subplots(
        2,
        4,
        figsize=(18, 9.5),
        constrained_layout=True,
    )
    axes = axes.ravel()

    for panel_index, variable in enumerate(DEPENDENCE_FEATURES):
        ax = axes[panel_index]
        for landmark_index in PRIMARY_LANDMARK_INDICES:
            sub = sample_clinical_df.loc[
                (sample_clinical_df["landmark_index"] == int(landmark_index))
                & (sample_clinical_df["clinical_variable"] == variable)
            ]
            x = sub["current_model_value"].to_numpy(dtype=float)
            y = sub["signed_ig"].to_numpy(dtype=float)
            finite = np.isfinite(x) & np.isfinite(y)
            if finite.sum() == 0:
                continue
            label = f"ART year {LANDMARK_MONTHS[landmark_index] / 12:.0f}"
            scatter_artist = ax.scatter(
                x[finite],
                y[finite],
                s=7,
                alpha=0.10,
                linewidths=0,
                rasterized=True,
                label=label,
            )
            bx, by = binned_median_xy(x[finite], y[finite], bins=12)
            if len(bx):
                line_color = scatter_artist.get_facecolors()[0]
                ax.plot(
                    bx,
                    by,
                    linewidth=2.0,
                    color=line_color,
                )

        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.set_title(variable, fontweight="bold")
        ax.set_xlabel(
            "Age (years)" if variable == "Age" else "Current model input value"
        )
        ax.set_ylabel("Signed IG attribution")
        ax.grid(alpha=0.18, linestyle="--")
        ax.text(
            -0.12,
            1.04,
            chr(ord("A") + panel_index),
            transform=ax.transAxes,
            fontsize=13,
            fontweight="bold",
            color="black",
        )

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles[:4],
            labels[:4],
            loc="lower center",
            ncol=4,
            frameon=False,
            bbox_to_anchor=(0.5, -0.01),
        )
    fig.suptitle(
        "Dependence of IG contributions on current feature values",
        fontsize=15,
        fontweight="bold",
        y=1.015,
    )
    save_figure(fig, "Figure12A7_selected_IG_dependence")

# =============================================================================
# 17. 主流程
# =============================================================================


# =============================================================================
# 20. Step12I：6个正式Landmark患者级动态解释专用函数
# =============================================================================

def build_exact_selected_risk_trajectory(
    common: CommonData,
    calibrated_risk_long: np.ndarray,
    sample_selection: pd.DataFrame,
) -> pd.DataFrame:
    case_info = (
        sample_selection[
            [
                "case_label",
                "trajectory_label",
                "patient_local",
            ]
        ]
        .drop_duplicates(
            "patient_local"
        )
    )

    rows = []

    for case in case_info.itertuples(
        index=False
    ):
        patient_local = int(
            case.patient_local
        )

        for landmark_index in PRIMARY_LANDMARK_INDICES:
            landmark_index = int(
                landmark_index
            )

            long_row = int(
                common.long_row_index_map[
                    patient_local,
                    landmark_index,
                ]
            )

            if long_row < 0:
                raise ValueError(
                    f"患者{patient_local}在Landmark "
                    f"{LANDMARK_MONTHS[landmark_index]}月没有long row。"
                )

            risk = float(
                calibrated_risk_long[
                    long_row,
                    -1,
                ]
            )

            rows.append(
                {
                    "case_label": case.case_label,
                    "trajectory_label": case.trajectory_label,
                    "patient_local": patient_local,
                    "development_global_index": int(
                        common.development_idx[
                            patient_local
                        ]
                    ),
                    "fold_id": int(
                        common.development_fold_id[
                            patient_local
                        ]
                    ),
                    "landmark_index": landmark_index,
                    "landmark_month": int(
                        LANDMARK_MONTHS[
                            landmark_index
                        ]
                    ),
                    "landmark_year": float(
                        LANDMARK_MONTHS[
                            landmark_index
                        ]
                        / 12.0
                    ),
                    "predicted_5y_risk": risk,
                    "event_within_5y": int(
                        common.future_event_matrix[
                            patient_local,
                            landmark_index,
                            :,
                        ].sum()
                        > 0.5
                    ),
                }
            )

    result = pd.DataFrame(
        rows
    ).sort_values(
        [
            "case_label",
            "landmark_index",
        ]
    ).reset_index(
        drop=True
    )

    result.to_csv(
        RISK_TRAJECTORY_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    return result


def build_top_local_contributions(
    sample_clinical_df: pd.DataFrame,
    sample_selection: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    case_info = (
        sample_selection[
            [
                "case_label",
                "trajectory_label",
                "patient_local",
            ]
        ]
        .drop_duplicates(
            "patient_local"
        )
    )

    local_df = sample_clinical_df.merge(
        case_info,
        on="patient_local",
        how="inner",
        validate="many_to_one",
    )

    top_rows = []

    for (
        case_label,
        trajectory_label,
        patient_local,
        landmark_index,
        landmark_month,
        landmark_year,
    ), frame in local_df.groupby(
        [
            "case_label",
            "trajectory_label",
            "patient_local",
            "landmark_index",
            "landmark_month",
            "landmark_year",
        ],
        sort=True,
        dropna=False,
    ):
        frame = frame.loc[
            np.isfinite(
                frame[
                    "signed_ig"
                ]
            )
        ].copy()

        positive = (
            frame.loc[
                frame[
                    "signed_ig"
                ]
                > 0
            ]
            .sort_values(
                [
                    "signed_ig",
                    "clinical_variable",
                ],
                ascending=[
                    False,
                    True,
                ],
            )
            .head(
                TOP_LOCAL_CONTRIBUTORS_PER_DIRECTION
            )
        )

        negative = (
            frame.loc[
                frame[
                    "signed_ig"
                ]
                < 0
            ]
            .sort_values(
                [
                    "signed_ig",
                    "clinical_variable",
                ],
                ascending=[
                    True,
                    True,
                ],
            )
            .head(
                TOP_LOCAL_CONTRIBUTORS_PER_DIRECTION
            )
        )

        positive_total = float(
            positive[
                "signed_ig"
            ].sum()
        )

        negative_total = float(
            np.abs(
                negative[
                    "signed_ig"
                ]
            ).sum()
        )

        for rank, row in enumerate(
            positive.itertuples(
                index=False
            ),
            start=1,
        ):
            top_rows.append(
                {
                    "case_label": case_label,
                    "trajectory_label": trajectory_label,
                    "patient_local": int(
                        patient_local
                    ),
                    "landmark_index": int(
                        landmark_index
                    ),
                    "landmark_month": int(
                        landmark_month
                    ),
                    "landmark_year": float(
                        landmark_year
                    ),
                    "direction": "risk_increase",
                    "rank_within_direction": int(
                        rank
                    ),
                    "clinical_variable": str(
                        row.clinical_variable
                    ),
                    "source": str(
                        row.source
                    ),
                    "signed_ig": float(
                        row.signed_ig
                    ),
                    "absolute_ig": float(
                        row.absolute_ig
                    ),
                    "direction_total_abs_ig": (
                        positive_total
                    ),
                }
            )

        for rank, row in enumerate(
            negative.itertuples(
                index=False
            ),
            start=1,
        ):
            top_rows.append(
                {
                    "case_label": case_label,
                    "trajectory_label": trajectory_label,
                    "patient_local": int(
                        patient_local
                    ),
                    "landmark_index": int(
                        landmark_index
                    ),
                    "landmark_month": int(
                        landmark_month
                    ),
                    "landmark_year": float(
                        landmark_year
                    ),
                    "direction": "risk_decrease",
                    "rank_within_direction": int(
                        rank
                    ),
                    "clinical_variable": str(
                        row.clinical_variable
                    ),
                    "source": str(
                        row.source
                    ),
                    "signed_ig": float(
                        row.signed_ig
                    ),
                    "absolute_ig": float(
                        row.absolute_ig
                    ),
                    "direction_total_abs_ig": (
                        negative_total
                    ),
                }
            )

    top_df = pd.DataFrame(
        top_rows
    )

    if top_df.empty:
        raise RuntimeError(
            "未生成患者局部Top IG贡献。"
        )

    # 每个患者单独建立稳定feature ID。
    feature_pieces = []

    for (
        case_label,
        patient_local,
    ), frame in local_df.groupby(
        [
            "case_label",
            "patient_local",
        ],
        sort=True,
    ):
        feature_summary = (
            frame.groupby(
                [
                    "clinical_variable",
                    "source",
                ],
                as_index=False,
            )
            .agg(
                mean_abs_ig=(
                    "absolute_ig",
                    "mean",
                ),
                mean_signed_ig=(
                    "signed_ig",
                    "mean",
                ),
                max_abs_ig=(
                    "absolute_ig",
                    "max",
                ),
            )
            .sort_values(
                [
                    "mean_abs_ig",
                    "clinical_variable",
                ],
                ascending=[
                    False,
                    True,
                ],
            )
            .reset_index(
                drop=True
            )
        )

        # 只保留至少曾进入某一时间点Top contributor的变量，
        # 然后按六个时间点平均|IG|排序。
        used_variables = set(
            top_df.loc[
                top_df[
                    "case_label"
                ].eq(
                    case_label
                ),
                "clinical_variable",
            ]
        )

        feature_summary = (
            feature_summary.loc[
                feature_summary[
                    "clinical_variable"
                ].isin(
                    used_variables
                )
            ]
            .head(
                MAX_FEATURES_IN_CASE_TABLE
            )
            .copy()
        )

        feature_summary.insert(
            0,
            "feature_id",
            np.arange(
                1,
                len(
                    feature_summary
                )
                + 1,
                dtype=np.int64,
            ),
        )

        feature_summary.insert(
            0,
            "patient_local",
            int(
                patient_local
            ),
        )

        feature_summary.insert(
            0,
            "case_label",
            case_label,
        )

        feature_pieces.append(
            feature_summary
        )

    feature_id_df = pd.concat(
        feature_pieces,
        ignore_index=True,
    )

    id_lookup = {
        (
            row.case_label,
            row.clinical_variable,
        ): int(
            row.feature_id
        )
        for row in feature_id_df.itertuples(
            index=False
        )
    }

    top_df[
        "feature_id"
    ] = [
        id_lookup.get(
            (
                row.case_label,
                row.clinical_variable,
            ),
            np.nan,
        )
        for row in top_df.itertuples(
            index=False
        )
    ]

    top_df.to_csv(
        TOP_CONTRIBUTION_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    feature_id_df.to_csv(
        FEATURE_ID_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    return (
        top_df,
        feature_id_df,
    )


def _trajectory_percent_axis_limits(
    risk_trajectory: pd.DataFrame,
) -> tuple[
    float,
    float,
]:
    risk_pct = (
        100.0
        * risk_trajectory[
            "predicted_5y_risk"
        ].to_numpy(
            dtype=np.float64
        )
    )

    maximum = float(
        np.nanmax(
            risk_pct
        )
    )

    if maximum <= 1.0:
        upper = 1.25
    elif maximum <= 2.0:
        upper = math.ceil(
            maximum
            * 5.0
            / 1.10
        ) / 5.0
    elif maximum <= 5.0:
        upper = math.ceil(
            maximum
            * 2.0
            / 1.08
        ) / 2.0
    elif maximum <= 10.0:
        upper = math.ceil(
            maximum
            * 1.10
        )
    else:
        upper = (
            math.ceil(
                maximum
                * 1.10
                / 5.0
            )
            * 5.0
        )

    upper = max(
        upper,
        maximum
        * 1.10
        + 0.1,
    )

    return (
        0.0,
        float(
            upper
        ),
    )


def plot_candidate_trajectory_overview(
    risk_trajectory: pd.DataFrame,
) -> None:
    candidate = pd.read_csv(
        CANDIDATE_TRAJECTORY_FILE,
        encoding="utf-8-sig",
    )

    risk_columns = [
        f"risk_{int(month)}m"
        for month in PRIMARY_LANDMARK_MONTHS
    ]

    for column in risk_columns:
        if column not in candidate.columns:
            raise ValueError(
                f"候选轨迹表缺少{column}。"
            )

    x = (
        PRIMARY_LANDMARK_MONTHS.astype(
            np.float64
        )
        / 12.0
    )

    fig, ax = plt.subplots(
        figsize=(
            8.6,
            5.8,
        )
    )

    # 避免数万条线使PDF过重；固定抽取最多1200条候选背景轨迹。
    if len(
        candidate
    ) > 1200:
        background = candidate.sample(
            n=1200,
            random_state=RANDOM_SEED,
        )
    else:
        background = candidate

    matrix = (
        100.0
        * background[
            risk_columns
        ].to_numpy(
            dtype=np.float64
        )
    )

    for row in matrix:
        ax.plot(
            x,
            row,
            color="#D0D0D0",
            linewidth=0.65,
            alpha=0.20,
            zorder=1,
        )

    case_colors = {
        "Case A": "#1B9E77",
        "Case B": "#D95F02",
        "Case C": "#7570B3",
    }

    for case_label, frame in risk_trajectory.groupby(
        "case_label",
        sort=True,
    ):
        frame = frame.sort_values(
            "landmark_year"
        )

        trajectory_label = str(
            frame[
                "trajectory_label"
            ].iloc[
                0
            ]
        )

        ax.plot(
            frame[
                "landmark_year"
            ],
            100.0
            * frame[
                "predicted_5y_risk"
            ],
            marker="o",
            markersize=5.0,
            linewidth=2.4,
            color=case_colors[
                case_label
            ],
            label=(
                f"{case_label}: "
                f"{trajectory_label}"
            ),
            zorder=4,
        )

    y_lower, y_upper = (
        _trajectory_percent_axis_limits(
            risk_trajectory
        )
    )

    ax.set_xlim(
        -0.10,
        5.10,
    )

    ax.set_ylim(
        y_lower,
        y_upper,
    )

    ax.set_xticks(
        x
    )

    ax.set_xlabel(
        "Prediction landmark after ART initiation (years)"
    )

    ax.set_ylabel(
        "Predicted CKD risk over the subsequent 5 years (%)"
    )

    ax.set_title(
        "Candidate OOF trajectories and objectively selected illustrative cases"
    )

    ax.grid(
        axis="y",
        linestyle="--",
        alpha=0.25,
    )

    ax.legend(
        frameon=False,
        loc="upper left",
    )

    fig.tight_layout()

    fig.savefig(
        TRAJECTORY_OVERVIEW_PNG,
        dpi=600,
        bbox_inches="tight",
        facecolor="white",
    )

    fig.savefig(
        TRAJECTORY_OVERVIEW_PDF,
        bbox_inches="tight",
        facecolor="white",
    )

    plt.close(
        fig
    )


def plot_patient_dynamic_explanation(
    risk_trajectory: pd.DataFrame,
    top_contributions: pd.DataFrame,
    feature_id_df: pd.DataFrame,
    sample_selection: pd.DataFrame,
) -> None:
    case_colors = {
        "Case A": "#1B9E77",
        "Case B": "#D95F02",
        "Case C": "#7570B3",
    }

    positive_colors = [
        "#D6604D",
        "#F4A582",
        "#FDDBC7",
    ]

    negative_colors = [
        "#2166AC",
        "#67A9CF",
        "#D1E5F0",
    ]

    y_lower, y_upper = (
        _trajectory_percent_axis_limits(
            risk_trajectory
        )
    )

    y_span = max(
        y_upper
        - y_lower,
        1e-6,
    )

    # ribbon只作为相对IG贡献可视化；
    # 跨18个origin统一缩放，使不同时间点band厚度可比较。
    direction_totals = (
        top_contributions[
            [
                "case_label",
                "landmark_year",
                "direction",
                "direction_total_abs_ig",
            ]
        ]
        .drop_duplicates()
    )

    global_max_direction_total = float(
        direction_totals[
            "direction_total_abs_ig"
        ].max()
    )

    if (
        not np.isfinite(
            global_max_direction_total
        )
        or global_max_direction_total
        <= 0
    ):
        global_max_direction_total = 1.0

    max_total_band = (
        0.12
        * y_span
    )

    fig = plt.figure(
        figsize=(
            16.6,
            13.0,
        ),
        constrained_layout=True,
    )

    grid = GridSpec(
        3,
        2,
        figure=fig,
        width_ratios=[
            1.90,
            1.05,
        ],
        height_ratios=[
            1.0,
            1.0,
            1.0,
        ],
    )

    case_selection_unique = (
        sample_selection[
            [
                "case_label",
                "trajectory_label",
                "patient_local",
            ]
        ]
        .drop_duplicates(
            "patient_local"
        )
        .sort_values(
            "case_label"
        )
    )

    for row_index, case in enumerate(
        case_selection_unique.itertuples(
            index=False
        )
    ):
        ax = fig.add_subplot(
            grid[
                row_index,
                0,
            ]
        )

        table_ax = fig.add_subplot(
            grid[
                row_index,
                1,
            ]
        )

        risk_case = (
            risk_trajectory.loc[
                risk_trajectory[
                    "case_label"
                ].eq(
                    case.case_label
                )
            ]
            .sort_values(
                "landmark_year"
            )
        )

        contrib_case = (
            top_contributions.loc[
                top_contributions[
                    "case_label"
                ].eq(
                    case.case_label
                )
            ]
            .copy()
        )

        feature_case = (
            feature_id_df.loc[
                feature_id_df[
                    "case_label"
                ].eq(
                    case.case_label
                )
            ]
            .sort_values(
                "feature_id"
            )
        )

        x = risk_case[
            "landmark_year"
        ].to_numpy(
            dtype=np.float64
        )

        risk_pct = (
            100.0
            * risk_case[
                "predicted_5y_risk"
            ].to_numpy(
                dtype=np.float64
            )
        )

        # -----------------------------------------------------
        # 正/负贡献ribbon
        # -----------------------------------------------------
        for (
            direction,
            colors,
            sign,
        ) in [
            (
                "risk_increase",
                positive_colors,
                +1.0,
            ),
            (
                "risk_decrease",
                negative_colors,
                -1.0,
            ),
        ]:
            direction_df = contrib_case.loc[
                contrib_case[
                    "direction"
                ].eq(
                    direction
                )
            ]

            boundary = risk_pct.copy()

            for rank in range(
                1,
                TOP_LOCAL_CONTRIBUTORS_PER_DIRECTION
                + 1,
            ):
                rank_df = direction_df.loc[
                    direction_df[
                        "rank_within_direction"
                    ].eq(
                        rank
                    )
                ]

                magnitude_lookup = {
                    float(
                        row.landmark_year
                    ): abs(
                        float(
                            row.signed_ig
                        )
                    )
                    for row in rank_df.itertuples(
                        index=False
                    )
                }

                thickness = np.asarray(
                    [
                        (
                            max_total_band
                            * magnitude_lookup.get(
                                float(
                                    year
                                ),
                                0.0,
                            )
                            / global_max_direction_total
                        )
                        for year in x
                    ],
                    dtype=np.float64,
                )

                next_boundary = (
                    boundary
                    + sign
                    * thickness
                )

                ax.fill_between(
                    x,
                    boundary,
                    next_boundary,
                    color=colors[
                        rank
                        - 1
                    ],
                    alpha=0.90,
                    linewidth=0,
                    zorder=1,
                )

                boundary = next_boundary

        # -----------------------------------------------------
        # 风险轨迹
        # -----------------------------------------------------
        case_color = case_colors[
            case.case_label
        ]

        ax.plot(
            x,
            risk_pct,
            color=case_color,
            marker="o",
            markersize=5.2,
            linewidth=2.6,
            zorder=4,
        )

        for year, risk_value in zip(
            x,
            risk_pct,
        ):
            ax.text(
                year,
                risk_value
                + 0.020
                * y_span,
                f"{risk_value:.2f}%",
                ha="center",
                va="bottom",
                color=case_color,
                fontsize=7.8,
                zorder=6,
            )

        # -----------------------------------------------------
        # 每个landmark标记Top-1增加/降低风险的feature ID
        # -----------------------------------------------------
        for (
            direction,
            text_prefix,
            offset,
        ) in [
            (
                "risk_increase",
                "+",
                +0.080
                * y_span,
            ),
            (
                "risk_decrease",
                "−",
                -0.080
                * y_span,
            ),
        ]:
            top1 = contrib_case.loc[
                contrib_case[
                    "direction"
                ].eq(
                    direction
                )
                & contrib_case[
                    "rank_within_direction"
                ].eq(
                    1
                )
            ].sort_values(
                "landmark_year"
            )

            risk_lookup = dict(
                zip(
                    x,
                    risk_pct,
                )
            )

            for row in top1.itertuples(
                index=False
            ):
                if pd.isna(
                    row.feature_id
                ):
                    continue

                y_text = (
                    risk_lookup[
                        float(
                            row.landmark_year
                        )
                    ]
                    + offset
                )

                y_text = min(
                    max(
                        y_text,
                        y_lower
                        + 0.03
                        * y_span,
                    ),
                    y_upper
                    - 0.03
                    * y_span,
                )

                ax.text(
                    float(
                        row.landmark_year
                    ),
                    y_text,
                    (
                        text_prefix
                        + str(
                            int(
                                row.feature_id
                            )
                        )
                    ),
                    ha="center",
                    va="center",
                    fontsize=8.1,
                    fontweight="bold",
                    zorder=7,
                )

        ax.set_xlim(
            -0.10,
            5.10,
        )

        ax.set_ylim(
            y_lower,
            y_upper,
        )

        ax.set_xticks(
            PRIMARY_LANDMARK_MONTHS.astype(
                np.float64
            )
            / 12.0
        )

        ax.set_xlabel(
            "Prediction landmark after ART initiation (years)"
        )

        ax.set_ylabel(
            "Predicted CKD risk over the subsequent 5 years (%)"
        )

        panel_letter = chr(
            ord(
                "A"
            )
            + row_index
        )

        ax.set_title(
            (
                f"{panel_letter}  "
                f"{case.trajectory_label}"
            ),
            loc="left",
            fontweight="bold",
        )

        ax.grid(
            axis="y",
            linestyle="--",
            alpha=0.25,
        )

        event_value = (
            risk_case.loc[
                np.isclose(
                    risk_case[
                        "landmark_year"
                    ],
                    5.0,
                ),
                "event_within_5y",
            ]
        )

        if len(
            event_value
        ) == 1:
            event_text = (
                "Yes"
                if int(
                    event_value.iloc[
                        0
                    ]
                )
                == 1
                else "No"
            )

            ax.text(
                0.99,
                0.97,
                (
                    "Observed CKD within 5 years "
                    "after the 5-year landmark: "
                    f"{event_text}"
                ),
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=8.0,
            )

        if row_index == 0:
            legend_handles = [
                Line2D(
                    [
                        0
                    ],
                    [
                        0
                    ],
                    color=case_color,
                    marker="o",
                    linewidth=2.6,
                    label=(
                        "Cross-fit calibrated "
                        "5-year CKD risk"
                    ),
                ),
                plt.Rectangle(
                    (
                        0,
                        0,
                    ),
                    1,
                    1,
                    facecolor=positive_colors[
                        0
                    ],
                    label=(
                        "Top IG contributors "
                        "increasing risk"
                    ),
                ),
                plt.Rectangle(
                    (
                        0,
                        0,
                    ),
                    1,
                    1,
                    facecolor=negative_colors[
                        0
                    ],
                    label=(
                        "Top IG contributors "
                        "decreasing risk"
                    ),
                ),
            ]

            ax.legend(
                handles=legend_handles,
                frameon=False,
                loc="upper left",
            )

        # -----------------------------------------------------
        # 右侧feature-ID表
        # -----------------------------------------------------
        table_ax.axis(
            "off"
        )

        table_ax.text(
            0.02,
            0.985,
            "ID",
            fontweight="bold",
            ha="left",
            va="top",
            transform=table_ax.transAxes,
        )

        table_ax.text(
            0.12,
            0.985,
            "Feature",
            fontweight="bold",
            ha="left",
            va="top",
            transform=table_ax.transAxes,
        )

        table_ax.text(
            0.96,
            0.985,
            "Mean direction",
            fontweight="bold",
            ha="right",
            va="top",
            transform=table_ax.transAxes,
        )

        rows = feature_case.head(
            MAX_FEATURES_IN_CASE_TABLE
        )

        row_height = (
            0.86
            / max(
                len(
                    rows
                ),
                1,
            )
        )

        y = 0.91

        for row in rows.itertuples(
            index=False
        ):
            if float(
                row.mean_signed_ig
            ) > 0:
                direction_text = "↑ risk"
                direction_color = positive_colors[
                    0
                ]
            elif float(
                row.mean_signed_ig
            ) < 0:
                direction_text = "↓ risk"
                direction_color = negative_colors[
                    0
                ]
            else:
                direction_text = "≈ neutral"
                direction_color = "#666666"

            table_ax.text(
                0.025,
                y,
                str(
                    int(
                        row.feature_id
                    )
                ),
                ha="left",
                va="center",
                transform=table_ax.transAxes,
            )

            table_ax.text(
                0.12,
                y,
                str(
                    row.clinical_variable
                ),
                ha="left",
                va="center",
                transform=table_ax.transAxes,
            )

            table_ax.text(
                0.96,
                y,
                direction_text,
                color=direction_color,
                ha="right",
                va="center",
                transform=table_ax.transAxes,
            )

            table_ax.plot(
                [
                    0.02,
                    0.97,
                ],
                [
                    y
                    - 0.45
                    * row_height,
                    y
                    - 0.45
                    * row_height,
                ],
                color="#DDDDDD",
                linewidth=0.7,
                transform=table_ax.transAxes,
                clip_on=False,
            )

            y -= row_height

        table_ax.text(
            0.02,
            0.015,
            (
                "+ID/−ID on the trajectory denotes the "
                "top risk-increasing/decreasing contributor "
                "at that landmark."
            ),
            fontsize=7.4,
            ha="left",
            va="bottom",
            transform=table_ax.transAxes,
        )

    fig.suptitle(
        (
            "Patient-specific dynamic explanation of "
            "LSTM-predicted subsequent 5-year CKD risk "
            "across all 6 formal prediction landmarks"
        ),
        fontsize=13.5,
        fontweight="bold",
    )

    fig.text(
        0.5,
        -0.004,
        (
            "The model itself is unchanged. Predictions are the existing "
            "cross-fit calibrated OOF risks at 0, 1, 2, 3, 4 and 5 years. "
            "Integrated Gradients were newly calculated only for the 18 "
            "selected patient-landmark origins. Shaded bands visualize "
            "relative local attribution magnitude and are not confidence "
            "intervals or exact probability decompositions."
        ),
        ha="center",
        va="top",
        fontsize=8.0,
    )

    fig.savefig(
        PATIENT_FIGURE_PNG,
        dpi=600,
        bbox_inches="tight",
        facecolor="white",
    )

    fig.savefig(
        PATIENT_FIGURE_PDF,
        bbox_inches="tight",
        facecolor="white",
    )

    plt.close(
        fig
    )


def run() -> None:
    total_start = time.time()
    set_random_seed(RANDOM_SEED)
    configure_plot_style()
    device = default_device()

    required = [STEP10E_DIR / "lstm_v2_summary.json"]
    if EXPLAIN_CALIBRATED:
        required.extend([CALIBRATION_FILE, CALIBRATED_RISK_FILE])
    require_files(required)

    calibration_table = (
        pd.read_csv(CALIBRATION_FILE, encoding="utf-8-sig")
        if EXPLAIN_CALIBRATED
        else None
    )
    calibrated_risk_long = (
        np.load(CALIBRATED_RISK_FILE, mmap_mode="r")
        if EXPLAIN_CALIBRATED
        else None
    )

    common = load_common_data()
    sample_selection = build_or_load_sample_selection(
        common=common,
        calibrated_risk_long=calibrated_risk_long,
    )

    print("=" * 110)
    print("Step 12I：LSTM-v2 6个正式Landmark患者级动态local IG解释")
    print("=" * 110)
    print("运行设备：", device)
    print("解释风险：", "Step11交叉拟合校准后的5年累计CKD风险" if EXPLAIN_CALIBRATED else "Step10E原始5年累计CKD风险")
    print("病例数：3；每名患者使用全部6个正式Landmark（0/1/2/3/4/5年）")
    print("IG积分步数：", IG_STEPS)
    print("IG积分方法：Gauss-Legendre")
    print("IG batch size：", IG_BATCH_SIZE)
    print("模型审计：Step10E原AMP+原batch size+NumPy ensemble")
    print("IG梯度：全精度可微前向；与AMP保存预测的漂移单独报告")
    print("后处理修订：", POSTPROCESS_REVISION)
    if RESUME:
        print("断点续跑：已完成的fold-landmark IG检查点将直接复用，不重新计算IG")
    print("患者动态图使用Landmark（月）：", PRIMARY_LANDMARK_MONTHS.tolist())
    print("锁定测试集：未读取")
    print("输出目录：", OUTPUT_DIR)
    print("=" * 110)
    print("抽样分布：")
    print(
        sample_selection.groupby(["landmark_year", "fold_id"]).size().unstack(fill_value=0)
    )
    print("=" * 110)

    raw_blocks: list[dict[str, Any]] = []
    verification_rows = []

    # -------------------------------------------------------------------------
    # 按fold准备特征，按fold-landmark读取对应OOF ensemble。
    # -------------------------------------------------------------------------
    for fold_id in range(N_SPLITS):
        fold_selection = sample_selection.loc[
            sample_selection["fold_id"] == fold_id
        ]
        if fold_selection.empty:
            continue

        print(f"\n{'-' * 110}")
        print(f"Fold {fold_id}：准备fold-specific输入与Landmark风险集IG reference")
        print(f"{'-' * 110}")
        fold_arrays = prepare_fold_arrays(common, fold_id)

        # ---------------------------------------------------------------------
        # Fold级审计A+B：只做一次，不再用8个小batch预测去硬对Step10E。
        # ---------------------------------------------------------------------
        fold_audit_path = (
            CHECKPOINT_DIR / f"fold_{fold_id}_numerical_audit.json"
        )
        formal_fold_audit_path = (
            FORMAL_STEP12A_CHECKPOINT_DIR
            / f"fold_{fold_id}_numerical_audit.json"
        )

        if (
            RESUME
            and formal_fold_audit_path.exists()
        ):
            # 这些数值审计只验证冻结Step10E/Step11模型映射，
            # 与本次选哪3名患者无关，因此直接复用既有正式Step12A审计。
            fold_audit = json.loads(
                formal_fold_audit_path.read_text(
                    encoding="utf-8"
                )
            )
            if (
                str(
                    fold_audit.get(
                        "pipeline_version",
                        "",
                    )
                )
                != FORMAL_STEP12A_PIPELINE_VERSION
            ):
                raise ValueError(
                    "既有正式Step12A fold数值审计版本不一致。"
                )

            message = (
                f"Fold {fold_id}：复用正式Step12A数值审计 | "
                f"AMP max diff="
                f"{fold_audit['step10e']['amp_reconstruction_max_abs_diff']:.3e}"
            )

            if fold_audit.get(
                "step11"
            ):
                message += (
                    " | Step11 mapping max diff="
                    f"{fold_audit['step11']['step11_mapping_max_abs_diff']:.3e}"
                )

            print(
                message
            )

        elif (
            RESUME
            and fold_audit_path.exists()
        ):
            fold_audit = json.loads(
                fold_audit_path.read_text(
                    encoding="utf-8"
                )
            )

            if (
                str(
                    fold_audit.get(
                        "pipeline_version",
                        "",
                    )
                )
                != PIPELINE_VERSION
            ):
                raise ValueError(
                    "当前Step12I fold数值审计版本不一致。"
                )

            print(
                f"Fold {fold_id}：读取当前Step12I已完成数值审计。"
            )

        else:
            print(
                f"Fold {fold_id}：未找到可复用正式审计，"
                "开始完整Step10E AMP模型重建审计。"
            )

            step10e_audit = audit_step10e_fold_reconstruction(
                common=common,
                fold_arrays=fold_arrays,
                fold_id=fold_id,
                device=device,
            )

            print(
                f"Fold {fold_id} Step10E审计通过 | "
                f"max={step10e_audit['amp_reconstruction_max_abs_diff']:.3e} | "
                f"mean={step10e_audit['amp_reconstruction_mean_abs_diff']:.3e} | "
                f"p99={step10e_audit['amp_reconstruction_p99_abs_diff']:.3e}"
            )

            if EXPLAIN_CALIBRATED:
                print(
                    f"Fold {fold_id}：开始Step11保存预测校准映射审计。"
                )

                step11_audit = audit_step11_saved_prediction_mapping(
                    common=common,
                    fold_id=fold_id,
                    calibration_table=calibration_table,
                    calibrated_risk_long=calibrated_risk_long,
                )

                print(
                    f"Fold {fold_id} Step11审计通过 | "
                    f"mapping max={step11_audit['step11_mapping_max_abs_diff']:.3e} | "
                    f"mean={step11_audit['step11_mapping_mean_abs_diff']:.3e}"
                )
            else:
                step11_audit = {}

            fold_audit = {
                "pipeline_version": PIPELINE_VERSION,
                "fold_id": int(
                    fold_id
                ),
                "step10e": step10e_audit,
                "step11": step11_audit,
            }

            save_json(
                fold_audit,
                fold_audit_path,
            )

        verification_rows.append(
            {
                "audit_scope": "fold_step10e",
                **fold_audit["step10e"],
            }
        )
        if fold_audit.get("step11"):
            verification_rows.append(
                {
                    "audit_scope": "fold_step11",
                    **fold_audit["step11"],
                }
            )

        for landmark_index in PRIMARY_LANDMARK_INDICES:
            landmark_month = int(LANDMARK_MONTHS[landmark_index])
            landmark_selection = fold_selection.loc[
                fold_selection["landmark_index"] == int(landmark_index)
            ].sort_values("patient_local")
            if landmark_selection.empty:
                continue

            checkpoint_path = (
                CHECKPOINT_DIR
                / f"fold_{fold_id}_landmark_{landmark_month}m_ig.npz"
            )
            metadata_path = (
                CHECKPOINT_DIR
                / f"fold_{fold_id}_landmark_{landmark_month}m_metadata.json"
            )

            if RESUME and checkpoint_path.exists() and metadata_path.exists():
                print(
                    f"Fold {fold_id} | Landmark {landmark_month}月："
                    "读取已有IG检查点。"
                )
                loaded = np.load(checkpoint_path, allow_pickle=False)
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

                expected_patients = landmark_selection[
                    "patient_local"
                ].to_numpy(
                    dtype=np.int32
                )

                loaded_patients = loaded[
                    "patient_local"
                ].astype(
                    np.int32
                )

                if not np.array_equal(
                    loaded_patients,
                    expected_patients,
                ):
                    raise ValueError(
                        "已有Step12I IG检查点患者与当前客观选择病例不一致；"
                        "请删除当前Step12I输出目录后重新运行。"
                    )

                if str(metadata.get("pipeline_version", "")) != PIPELINE_VERSION:
                    raise ValueError(
                        "IG检查点pipeline版本与当前代码不一致；"
                        "请使用新的v4_formal_ig_explanation输出目录。"
                    )
                if int(metadata["ig_steps"]) != IG_STEPS:
                    raise ValueError("IG检查点steps与当前配置不一致。")
                if str(metadata.get("integration_method", "")) != "gauss_legendre":
                    raise ValueError("IG检查点积分方法与当前代码不一致。")
                if bool(metadata["calibrated"]) != EXPLAIN_CALIBRATED:
                    raise ValueError("IG检查点校准模式与当前配置不一致。")
                block = {
                    "patient_local": loaded["patient_local"].astype(np.int32),
                    "development_global_index": loaded["development_global_index"].astype(np.int32),
                    "fold_id": int(fold_id),
                    "landmark_index": int(landmark_index),
                    "dynamic_names": list(metadata["dynamic_names"]),
                    "static_names": list(metadata["static_names"]),
                    "attr_dynamic": loaded["attr_dynamic"].astype(np.float32),
                    "attr_static": loaded["attr_static"].astype(np.float32),
                    "attr_age": loaded["attr_age"].astype(np.float32),
                    "risk_input": loaded["risk_input"].astype(np.float32),
                    "risk_baseline": loaded["risk_baseline"].astype(np.float32),
                    "completeness_residual": loaded["completeness_residual"].astype(np.float32),
                    "current_dynamic": loaded["current_dynamic"].astype(np.float32),
                    "static_input": loaded["static_input"].astype(np.float32),
                    "age_raw": loaded["age_raw"].astype(np.float32),
                }
                raw_blocks.append(block)
                continue

            print(
                f"Fold {fold_id} | Landmark {landmark_month}月："
                f"开始解释 {len(landmark_selection)} 个OOF origins。"
            )
            model, dynamic_names, static_names, model_metadata = load_fold_ensemble(
                fold_id=fold_id,
                landmark_month=landmark_month,
                device=device,
                calibration_table=calibration_table,
            )

            if dynamic_names != fold_arrays.enhanced_dynamic_names:
                raise ValueError("检查点与当前重建动态特征名称/顺序不一致。")
            if static_names != fold_arrays.static_names:
                raise ValueError("检查点与当前重建静态特征名称/顺序不一致。")

            patients = landmark_selection["patient_local"].to_numpy(dtype=np.int64)

            drift_audit = audit_ig_fp32_risk_drift(
                model=model,
                common=common,
                fold_arrays=fold_arrays,
                landmark_index=int(landmark_index),
                selected_patients=patients,
                device=device,
                calibrated_risk_long=calibrated_risk_long,
            )
            verification_rows.append(
                {
                    "audit_scope": "ig_fp32_drift",
                    "fold_id": int(fold_id),
                    "landmark_index": int(landmark_index),
                    "landmark_month": landmark_month,
                    **drift_audit,
                }
            )
            print(
                "  IG全精度前向审计 | "
                f"vs saved 5y risk max diff="
                f"{drift_audit['ig_fp32_vs_saved_risk_max_abs_diff']:.3e} | "
                f"mean diff="
                f"{drift_audit['ig_fp32_vs_saved_risk_mean_abs_diff']:.3e}"
            )

            all_inputs = build_inputs_for_pairs(
                common=common,
                fold_arrays=fold_arrays,
                patient_local=patients,
                landmark_index=int(landmark_index),
            )
            ig_result = run_ig_in_batches(
                model=model,
                all_inputs=all_inputs,
                device=device,
            )

            block = {
                "patient_local": patients.astype(np.int32),
                "development_global_index": common.development_idx[patients].astype(np.int32),
                "fold_id": int(fold_id),
                "landmark_index": int(landmark_index),
                "dynamic_names": dynamic_names,
                "static_names": static_names,
                "current_dynamic": all_inputs["current_dynamic"].astype(np.float32),
                "static_input": all_inputs["static"].astype(np.float32),
                "age_raw": all_inputs["age_raw"].astype(np.float32),
                **ig_result,
            }
            raw_blocks.append(block)

            np.savez_compressed(
                checkpoint_path,
                patient_local=block["patient_local"],
                development_global_index=block["development_global_index"],
                attr_dynamic=block["attr_dynamic"],
                attr_static=block["attr_static"],
                attr_age=block["attr_age"],
                risk_input=block["risk_input"],
                risk_baseline=block["risk_baseline"],
                completeness_residual=block["completeness_residual"],
                current_dynamic=block["current_dynamic"],
                static_input=block["static_input"],
                age_raw=block["age_raw"],
            )
            save_json(
                {
                    "pipeline_version": PIPELINE_VERSION,
                    "fold_id": int(fold_id),
                    "landmark_index": int(landmark_index),
                    "landmark_month": landmark_month,
                    "ig_steps": IG_STEPS,
                    "integration_method": "gauss_legendre",
                    "calibrated": bool(EXPLAIN_CALIBRATED),
                    "dynamic_names": dynamic_names,
                    "static_names": static_names,
                    "model_metadata": model_metadata,
                },
                metadata_path,
            )

            del model
            gc.collect()
            torch.cuda.empty_cache()

        del fold_arrays
        gc.collect()
        torch.cuda.empty_cache()

    if not raw_blocks:
        raise RuntimeError("没有得到任何IG结果。")

    # -------------------------------------------------------------------------
    # 聚合与输出
    # -------------------------------------------------------------------------
    (
        sample_clinical_df,
        component_df,
        clinical_df,
        temporal_df,
        completeness_df,
    ) = aggregate_ig_results(raw_blocks)

    sample_clinical_df = add_within_variable_value_percentile(
        sample_clinical_df
    )
    evolution_df = build_landmark_evolution_table(clinical_df)
    lab_component_df, lab_value_measurement_df = (
        build_lab_component_decomposition(component_df)
    )

    sample_clinical_df.to_csv(
        OUTPUT_DIR / "ig_sample_level_clinical_attribution.csv",
        index=False,
        encoding="utf-8-sig",
    )
    evolution_df.to_csv(
        OUTPUT_DIR / "ig_landmark_importance_evolution.csv",
        index=False,
        encoding="utf-8-sig",
    )
    lab_component_df.to_csv(
        OUTPUT_DIR / "ig_lab_component_decomposition.csv",
        index=False,
        encoding="utf-8-sig",
    )
    lab_value_measurement_df.to_csv(
        OUTPUT_DIR / "ig_lab_value_vs_measurement_information.csv",
        index=False,
        encoding="utf-8-sig",
    )
    component_df.to_csv(
        OUTPUT_DIR / "ig_global_component_importance.csv",
        index=False,
        encoding="utf-8-sig",
    )
    clinical_df.to_csv(
        OUTPUT_DIR / "ig_global_clinical_importance.csv",
        index=False,
        encoding="utf-8-sig",
    )
    temporal_df.to_csv(
        OUTPUT_DIR / "ig_temporal_clinical_importance.csv",
        index=False,
        encoding="utf-8-sig",
    )
    completeness_df.to_csv(
        OUTPUT_DIR / "ig_completeness_checks.csv",
        index=False,
        encoding="utf-8-sig",
    )
    if verification_rows:
        pd.DataFrame(verification_rows).to_csv(
            OUTPUT_DIR / "model_reconstruction_checks.csv",
            index=False,
            encoding="utf-8-sig",
        )

    # -------------------------------------------------------------------------
    # completeness审计
    # -------------------------------------------------------------------------
    completeness_summary = (
        completeness_df.groupby(
            ["landmark_index", "landmark_month", "landmark_year"],
            as_index=False,
        )
        .agg(
            sample_n=("patient_local", "size"),
            mean_abs_residual=("ig_completeness_abs_residual", "mean"),
            median_abs_residual=("ig_completeness_abs_residual", "median"),
            p95_abs_residual=("ig_completeness_abs_residual", lambda x: float(np.quantile(x, 0.95))),
            max_abs_residual=("ig_completeness_abs_residual", "max"),
        )
    )
    completeness_summary.to_csv(
        OUTPUT_DIR / "ig_completeness_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    worst_p95 = float(completeness_summary["p95_abs_residual"].max())
    worst_max = float(completeness_summary["max_abs_residual"].max())
    if worst_p95 > 5e-4 or worst_max > 2e-3:
        print(
            "\n警告：IG completeness误差偏大。"
            f" worst p95={worst_p95:.3e}, worst max={worst_max:.3e}。"
            "正式分析前应将CKD_IG_STEPS增加到64后重新核对。"
        )
    else:
        print(
            "\nIG completeness数值积分检查良好 | "
            f"worst p95={worst_p95:.3e} | worst max={worst_max:.3e}"
        )

    # -------------------------------------------------------------------------
    # Step12I患者级6-Landmark动态解释图
    # -------------------------------------------------------------------------
    exact_risk_trajectory = build_exact_selected_risk_trajectory(
        common=common,
        calibrated_risk_long=calibrated_risk_long,
        sample_selection=sample_selection,
    )

    (
        top_local_contributions,
        feature_id_df,
    ) = build_top_local_contributions(
        sample_clinical_df=sample_clinical_df,
        sample_selection=sample_selection,
    )

    plot_candidate_trajectory_overview(
        risk_trajectory=exact_risk_trajectory,
    )

    plot_patient_dynamic_explanation(
        risk_trajectory=exact_risk_trajectory,
        top_contributions=top_local_contributions,
        feature_id_df=feature_id_df,
        sample_selection=sample_selection,
    )

    # -------------------------------------------------------------------------
    # 摘要
    # -------------------------------------------------------------------------
    summary = {
        "stage": "Step12I_LSTM_v2_Patient_Dynamic_Explanation_6Landmarks",
        "pipeline_version": PIPELINE_VERSION,
        "postprocess_revision": POSTPROCESS_REVISION,
        "explain_target": (
            "crossfit_calibrated_5y_cumulative_CKD_risk"
            if EXPLAIN_CALIBRATED
            else "raw_5y_cumulative_CKD_risk"
        ),
        "primary_landmark_months": PRIMARY_LANDMARK_MONTHS.tolist(),
        "selected_case_n": 3,
        "case_landmark_months": PRIMARY_LANDMARK_MONTHS.tolist(),
        "sample_origin_n": int(len(sample_selection)),
        "patient_dynamic_explanation": {
            "selection_uses_outcome": False,
            "selection_uses_attribution": False,
            "selection_uses_clinical_values": False,
            "landmarks": "all formal model landmarks: 0/1/2/3/4/5 years",
            "local_ig_origin_n": int(len(sample_selection)),
        },
        "formal_explanation_outputs": {
            "beeswarm_top_n": BEESWARM_TOP_N,
            "importance_evolution_top_n": EVOLUTION_TOP_N,
            "lab_component_top_n": LAB_COMPONENT_TOP_N,
            "dependence_features": DEPENDENCE_FEATURES,
            "landmark_comparison_uses_relative_importance_pct": True,
        },
        "ig_steps": IG_STEPS,
        "integration_method": "gauss_legendre",
        "ig_batch_size": IG_BATCH_SIZE,
        "random_seed": RANDOM_SEED,
        "numerical_audit_tolerances": {
            "amp_reconstruction_max_abs_diff": AMP_RECON_MAX_TOL,
            "amp_reconstruction_mean_abs_diff": AMP_RECON_MEAN_TOL,
            "step11_saved_mapping_max_abs_diff": STEP11_MAPPING_TOL,
            "ig_fp32_vs_saved_risk_hard_max_abs_diff": IG_FP32_RISK_DRIFT_HARD_TOL,
        },
        "reference_strategy": {
            "dynamic": "heldout-fold-excluded training RISK-SET mean at each landmark and each historical step",
            "static": "heldout-fold-excluded training RISK-SET mean at each landmark",
            "age": "heldout-fold-excluded training RISK-SET mean at each landmark",
            "row_mask": "held fixed at observed patient value",
            "landmark_context": "held fixed and not attributed",
        },
        "locked_test_used": False,
        "completeness": completeness_summary.to_dict(orient="records"),
        "elapsed_seconds": float(time.time() - total_start),
        "output_dir": str(OUTPUT_DIR),
    }
    save_json(summary, OUTPUT_DIR / "step12i_patient_dynamic_summary.json")

    print("\n" + "=" * 110)
    print("Step 12I 6-Landmark患者级动态local IG分析完成")
    print("=" * 110)
    print("local IG origin总数：", len(sample_selection), "（3名病例 × 6个正式Landmark）")
    print("锁定测试集：未读取")
    print("总耗时：", format_duration(time.time() - total_start))
    print("\nIG completeness：")
    print(completeness_summary.to_string(index=False))
    print("\n各Landmark Top 10临床变量：")
    for landmark_index in PRIMARY_LANDMARK_INDICES:
        sub = clinical_df.loc[
            clinical_df["landmark_index"] == int(landmark_index)
        ].nsmallest(10, "importance_rank")
        print("\n" + "-" * 80)
        print(f"Landmark {LANDMARK_MONTHS[landmark_index] / 12:.0f}年")
        print(
            sub[
                [
                    "importance_rank",
                    "clinical_variable",
                    "source",
                    "mean_abs_ig",
                    "mean_signed_ig",
                    "positive_fraction",
                ]
            ].to_string(index=False)
        )
    print("\n输出目录：", OUTPUT_DIR)
    print("=" * 110)


if __name__ == "__main__":
    run()
