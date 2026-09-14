#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Step 11 v2：Cox、RSF、RNN、LSTM-v2 开发集 OOF 统一校准与正式比较
=================================================================

目的
----
1. 读取四个模型完全独立的开发集五折 OOF 预测：
   - Super Landmark Cox
   - Pooled Landmark RSF
   - RNN survival
   - LSTM-v2 survival
2. 所有模型采用完全相同的五折交叉拟合离散风险校准：
      logit(h_cal,j) = alpha_j + beta * logit(h_raw,j)
   其中：
      alpha_j = 10 个未来半年区间特异截距
      beta    = 一个共同斜率
3. 在完全相同的患者、Landmark、结局和预测时间上评价：
   - 未来 5 年 Uno C-index
   - 6–60 月动态 AUC 及 iAUC
   - 6–60 月 Brier score 及 IBS
   - 1、3、5 年 AUC / Brier
   - 1、3、5 年总体和十分位校准
   - 1、3、5 年生存 DCA
4. 患者级 paired bootstrap：
   - 同一个 bootstrap 抽样同时用于全部模型、全部 Landmark
   - 默认 1000 次
   - 支持断点续跑
5. 同时报告：
   - 6 个 Landmark 等权平均：0、1、2、3、4、5 年
   - 主文 4 个 Landmark 等权平均：0、1、3、5 年
6. 预设全部必要配对比较：
   - RSF - Cox
   - RNN - Cox
   - LSTM-v2 - Cox
   - RSF - RNN
   - LSTM-v2 - RSF
   - LSTM-v2 - RNN
7. 保存全开发集 OOF 拟合的最终校准器，供后续冻结模型后使用。
8. 全程不读取 9,574 人锁定内部测试集。

重要说明
--------
- 本步骤不训练 Cox / RSF / RNN / LSTM。
- 本步骤只读取已经生成的 OOF 预测并做统一统计评价。
- LSTM-v2 路径对应已经完成的 Step 10E：
  rolling_5y_step10e_lstm_v2_final_oof_selected_existing_trials
- Bootstrap 默认 1000 次；调试时可临时：
    CKD_BOOTSTRAP_REPS=50 python Step11_v2_....py
  正式论文结果仍建议 1000 次。
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import spearmanr
from sksurv.metrics import (
    brier_score,
    concordance_index_ipcw,
    cumulative_dynamic_auc,
    integrated_brier_score,
)
from sksurv.nonparametric import kaplan_meier_estimator
from sksurv.util import Surv


# =============================================================================
# 1. 固定配置
# =============================================================================

PROJECT_DIR = Path(
    os.getenv(
        "CKD_LSTM_PROJECT_DIR",
        "__CKD_WORKDIR__",
    )
)

STEP6_DIR = (
    PROJECT_DIR
    / "rolling_5y_step6_super_landmark_data"
)

COX_DIR = (
    PROJECT_DIR
    / "rolling_5y_step7_unpenalized_cox_fixed95_v5"
)

RSF_DIR = (
    PROJECT_DIR
    / "rolling_5y_step8_pooled_landmark_rsf_resume_v2"
    / "final_oof"
)

RNN_DIR = (
    PROJECT_DIR
    / "rolling_5y_step9b_rnn_tune_resume_v1"
    / "final_oof"
)

LSTM_V2_DIR = (
    PROJECT_DIR
    / "rolling_5y_step10e_lstm_v2_final_oof_selected_existing_trials"
)

OUTPUT_DIR = (
    PROJECT_DIR
    / "rolling_5y_step11_v2_four_model_oof_comparison_lstm_v2"
)

FIGURE_DIR = OUTPUT_DIR / "figures"
BOOTSTRAP_DIR = OUTPUT_DIR / "bootstrap_checkpoints"

for directory in [
    OUTPUT_DIR,
    FIGURE_DIR,
    BOOTSTRAP_DIR,
]:
    directory.mkdir(
        parents=True,
        exist_ok=True,
    )

BOOTSTRAP_REPS = int(
    os.getenv(
        "CKD_BOOTSTRAP_REPS",
        "1000",
    )
)

RANDOM_SEED = int(
    os.getenv(
        "CKD_RANDOM_SEED",
        "20260804",
    )
)

BOOTSTRAP_SAVE_EVERY = 10
MINIMUM_VALID_BOOTSTRAP_RATE = 0.80

N_SPLITS = 5
EXPECTED_DEVELOPMENT_N = 22337
EXPECTED_LANDMARK_N = 6
EXPECTED_INTERVAL_N = 10
EXPECTED_LONG_N = 100122

LANDMARK_MONTHS = np.asarray(
    [0, 12, 24, 36, 48, 60],
    dtype=np.float64,
)

FUTURE_END_MONTHS = np.arange(
    6,
    61,
    6,
    dtype=np.float64,
)

# sksurv 要求评价时间严格小于最大随访时间。
# 第 10 个风险值仍表示“未来 60 月风险”，
# 但数值评价时使用 59.999 月。
METRIC_TIMES = np.asarray(
    [
        6,
        12,
        18,
        24,
        30,
        36,
        42,
        48,
        54,
        59.999,
    ],
    dtype=np.float64,
)

PRIMARY_LANDMARK_MONTHS = np.asarray(
    [0, 12, 36, 60],
    dtype=np.float64,
)

PRIMARY_LANDMARK_POSITIONS = np.asarray(
    [0, 1, 3, 5],
    dtype=np.int64,
)

ALL_LANDMARK_POSITIONS = np.arange(
    EXPECTED_LANDMARK_N,
    dtype=np.int64,
)

MODEL_ORDER = [
    "Cox",
    "RSF",
    "RNN",
    "LSTM-v2",
]

MODEL_INDEX = {
    model_name: index
    for index, model_name in enumerate(
        MODEL_ORDER
    )
}

PAIRWISE_COMPARISONS = [
    ("RSF", "Cox"),
    ("RNN", "Cox"),
    ("LSTM-v2", "Cox"),
    ("RSF", "RNN"),
    ("LSTM-v2", "RSF"),
    ("LSTM-v2", "RNN"),
]

SUMMARY_SCOPES = {
    "six_landmark_equal_weight_mean": (
        ALL_LANDMARK_POSITIONS
    ),
    "primary_0_1_3_5y_landmark_equal_weight_mean": (
        PRIMARY_LANDMARK_POSITIONS
    ),
}

DCA_THRESHOLD_RANGES = {
    12.0: (0.002, 0.030, 80),
    36.0: (0.005, 0.100, 80),
    60.0: (0.010, 0.200, 80),
}

CALIBRATION_RIDGE = 1e-6
EPS = 1e-7
DPI = 300

METRIC_NAMES = [
    "uno_c_index_5y",
    "integrated_dynamic_auc",
    "integrated_brier",
]

HIGHER_IS_BETTER = {
    "uno_c_index_5y": True,
    "integrated_dynamic_auc": True,
    "integrated_brier": False,
}

if BOOTSTRAP_REPS < 1:
    raise ValueError(
        "BOOTSTRAP_REPS 必须为正整数。"
    )


# =============================================================================
# 2. 通用工具
# =============================================================================

def format_duration(
    seconds: float,
) -> str:
    seconds = max(
        0,
        int(round(float(seconds))),
    )
    hours, remainder = divmod(
        seconds,
        3600,
    )
    minutes, seconds = divmod(
        remainder,
        60,
    )

    if hours:
        return (
            f"{hours}小时"
            f"{minutes:02d}分"
            f"{seconds:02d}秒"
        )
    if minutes:
        return (
            f"{minutes}分"
            f"{seconds:02d}秒"
        )
    return f"{seconds}秒"


def save_json(
    value: Any,
    path: Path,
) -> None:
    path.write_text(
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )


def require_files(
    paths: list[Path],
) -> None:
    missing = [
        str(path)
        for path in paths
        if not path.exists()
    ]

    if missing:
        raise FileNotFoundError(
            "以下必要输入文件不存在：\n"
            + "\n".join(missing)
        )


def display_table(
    frame: pd.DataFrame,
    title: str,
) -> None:
    print(
        "\n"
        + "=" * 110
    )
    print(title)
    print("=" * 110)

    try:
        from IPython.display import display
        display(frame)
    except ImportError:
        print(
            frame.to_string(
                index=False
            )
        )


def save_figure(
    fig: plt.Figure,
    filename_stem: str,
) -> None:
    png_path = (
        FIGURE_DIR
        / f"{filename_stem}.png"
    )
    pdf_path = (
        FIGURE_DIR
        / f"{filename_stem}.pdf"
    )

    fig.savefig(
        png_path,
        dpi=DPI,
        bbox_inches="tight",
        facecolor="white",
    )

    fig.savefig(
        pdf_path,
        bbox_inches="tight",
        facecolor="white",
    )

    plt.close(fig)


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
            "随访时间存在 NaN 或无穷值。"
        )

    if np.any(
        time_month <= 0
    ):
        raise ValueError(
            "Landmark 后随访时间必须 > 0。"
        )

    return Surv.from_arrays(
        event=event,
        time=time_month,
    )


def percentile_interval(
    values: np.ndarray,
    expected_n: int | None = None,
) -> tuple[float, float, int]:
    values = np.asarray(
        values,
        dtype=np.float64,
    )

    valid = values[
        np.isfinite(values)
    ]

    if expected_n is None:
        expected_n = len(values)

    required_n = int(
        np.ceil(
            expected_n
            * MINIMUM_VALID_BOOTSTRAP_RATE
        )
    )

    if len(valid) < required_n:
        raise RuntimeError(
            f"仅获得 {len(valid)}/{expected_n} "
            "个有效 Bootstrap 估计。"
        )

    return (
        float(
            np.percentile(
                valid,
                2.5,
            )
        ),
        float(
            np.percentile(
                valid,
                97.5,
            )
        ),
        int(len(valid)),
    )


def favorable_fraction(
    differences: np.ndarray,
    higher_is_better: bool,
) -> float:
    differences = np.asarray(
        differences,
        dtype=np.float64,
    )

    valid = differences[
        np.isfinite(differences)
    ]

    if len(valid) == 0:
        return np.nan

    if higher_is_better:
        return float(
            np.mean(
                valid > 0
            )
        )

    return float(
        np.mean(
            valid < 0
        )
    )


def two_sided_bootstrap_p(
    differences: np.ndarray,
) -> float:
    differences = np.asarray(
        differences,
        dtype=np.float64,
    )

    valid = differences[
        np.isfinite(differences)
    ]

    if len(valid) == 0:
        return np.nan

    p_lower = (
        np.mean(valid <= 0)
    )
    p_upper = (
        np.mean(valid >= 0)
    )

    return float(
        min(
            1.0,
            2.0
            * min(
                p_lower,
                p_upper,
            ),
        )
    )


def complete_row_mean(
    matrix: np.ndarray,
    positions: np.ndarray,
) -> np.ndarray:
    matrix = np.asarray(
        matrix,
        dtype=np.float64,
    )

    subset = matrix[
        :,
        positions,
    ]

    valid = np.isfinite(
        subset
    ).all(
        axis=1
    )

    result = np.full(
        subset.shape[0],
        np.nan,
        dtype=np.float64,
    )

    result[
        valid
    ] = subset[
        valid
    ].mean(
        axis=1
    )

    return result


def safe_name(
    model_name: str,
) -> str:
    return (
        model_name
        .lower()
        .replace("-", "_")
        .replace(" ", "_")
    )


# =============================================================================
# 3. 读取 Step 6 标签、映射和四模型 OOF
# =============================================================================

required_files = [
    STEP6_DIR
    / "development_long_row_index_map.npy",

    STEP6_DIR
    / "development_future_event_long.npy",

    STEP6_DIR
    / "development_future_at_risk_long.npy",

    STEP6_DIR
    / "development_local_patient_idx_long.npy",

    STEP6_DIR
    / "development_fold_id_long.npy",

    STEP6_DIR
    / "development_landmark_index_long.npy",

    STEP6_DIR
    / "development_landmark_month_long.npy",

    STEP6_DIR
    / "development_analysis_time_month.npy",

    STEP6_DIR
    / "development_event_within_60m.npy",

    STEP6_DIR
    / "landmark_months.npy",

    STEP6_DIR
    / "future_end_months.npy",

    COX_DIR
    / "cox_oof_survival_long.npy",

    COX_DIR
    / "cox_oof_risk_long.npy",

    COX_DIR
    / "cox_oof_log_partial_hazard_long.npy",

    RSF_DIR
    / "rsf_oof_survival_long.npy",

    RSF_DIR
    / "rsf_oof_risk_long.npy",

    RNN_DIR
    / "rnn_oof_survival_long.npy",

    RNN_DIR
    / "rnn_oof_risk_long.npy",

    RNN_DIR
    / "rnn_summary.json",

    LSTM_V2_DIR
    / "lstm_v2_oof_survival_long.npy",

    LSTM_V2_DIR
    / "lstm_v2_oof_risk_long.npy",

    LSTM_V2_DIR
    / "lstm_v2_summary.json",
]

require_files(
    required_files
)

row_index_map = np.load(
    STEP6_DIR
    / "development_long_row_index_map.npy"
).astype(
    np.int32
)

future_event_long = np.load(
    STEP6_DIR
    / "development_future_event_long.npy"
).astype(
    np.float32
)

future_at_risk_long = np.load(
    STEP6_DIR
    / "development_future_at_risk_long.npy"
).astype(
    bool
)

local_patient_idx_long = np.load(
    STEP6_DIR
    / "development_local_patient_idx_long.npy"
).astype(
    np.int32
)

fold_id_long = np.load(
    STEP6_DIR
    / "development_fold_id_long.npy"
).astype(
    np.int8
)

landmark_index_long = np.load(
    STEP6_DIR
    / "development_landmark_index_long.npy"
).astype(
    np.int8
)

landmark_month_long = np.load(
    STEP6_DIR
    / "development_landmark_month_long.npy"
).astype(
    np.float64
)

analysis_time_month = np.load(
    STEP6_DIR
    / "development_analysis_time_month.npy"
).astype(
    np.float64
)

event_within_60m = np.load(
    STEP6_DIR
    / "development_event_within_60m.npy"
).astype(
    bool
)

saved_landmarks = np.load(
    STEP6_DIR
    / "landmark_months.npy"
).astype(
    np.float64
)

saved_future_ends = np.load(
    STEP6_DIR
    / "future_end_months.npy"
).astype(
    np.float64
)

model_survival_raw = {
    "Cox": np.load(
        COX_DIR
        / "cox_oof_survival_long.npy"
    ).astype(
        np.float32
    ),

    "RSF": np.load(
        RSF_DIR
        / "rsf_oof_survival_long.npy"
    ).astype(
        np.float32
    ),

    "RNN": np.load(
        RNN_DIR
        / "rnn_oof_survival_long.npy"
    ).astype(
        np.float32
    ),

    "LSTM-v2": np.load(
        LSTM_V2_DIR
        / "lstm_v2_oof_survival_long.npy"
    ).astype(
        np.float32
    ),
}

model_risk_raw = {
    "Cox": np.load(
        COX_DIR
        / "cox_oof_risk_long.npy"
    ).astype(
        np.float32
    ),

    "RSF": np.load(
        RSF_DIR
        / "rsf_oof_risk_long.npy"
    ).astype(
        np.float32
    ),

    "RNN": np.load(
        RNN_DIR
        / "rnn_oof_risk_long.npy"
    ).astype(
        np.float32
    ),

    "LSTM-v2": np.load(
        LSTM_V2_DIR
        / "lstm_v2_oof_risk_long.npy"
    ).astype(
        np.float32
    ),
}

cox_log_partial_hazard_long = np.load(
    COX_DIR
    / "cox_oof_log_partial_hazard_long.npy"
).astype(
    np.float32
)

rnn_summary = json.loads(
    (
        RNN_DIR
        / "rnn_summary.json"
    ).read_text(
        encoding="utf-8"
    )
)

lstm_v2_summary = json.loads(
    (
        LSTM_V2_DIR
        / "lstm_v2_summary.json"
    ).read_text(
        encoding="utf-8"
    )
)

if bool(
    rnn_summary.get(
        "locked_test_used",
        False,
    )
):
    raise ValueError(
        "RNN 正式 OOF 记录显示使用过锁定测试集。"
    )

if bool(
    lstm_v2_summary.get(
        "locked_test_used",
        False,
    )
):
    raise ValueError(
        "LSTM-v2 正式 OOF 记录显示使用过锁定测试集。"
    )


# =============================================================================
# 4. 输入一致性与 OOF 审计
# =============================================================================

expected_shapes = {
    "row_index_map": (
        EXPECTED_DEVELOPMENT_N,
        EXPECTED_LANDMARK_N,
    ),

    "future_event_long": (
        EXPECTED_LONG_N,
        EXPECTED_INTERVAL_N,
    ),

    "future_at_risk_long": (
        EXPECTED_LONG_N,
        EXPECTED_INTERVAL_N,
    ),

    "local_patient_idx_long": (
        EXPECTED_LONG_N,
    ),

    "fold_id_long": (
        EXPECTED_LONG_N,
    ),

    "landmark_index_long": (
        EXPECTED_LONG_N,
    ),

    "landmark_month_long": (
        EXPECTED_LONG_N,
    ),

    "analysis_time_month": (
        EXPECTED_LONG_N,
    ),

    "event_within_60m": (
        EXPECTED_LONG_N,
    ),
}

actual_arrays = {
    "row_index_map": row_index_map,
    "future_event_long": future_event_long,
    "future_at_risk_long": (
        future_at_risk_long
    ),
    "local_patient_idx_long": (
        local_patient_idx_long
    ),
    "fold_id_long": fold_id_long,
    "landmark_index_long": (
        landmark_index_long
    ),
    "landmark_month_long": (
        landmark_month_long
    ),
    "analysis_time_month": (
        analysis_time_month
    ),
    "event_within_60m": (
        event_within_60m
    ),
}

for name, expected_shape in (
    expected_shapes.items()
):
    actual_shape = (
        actual_arrays[name].shape
    )

    if (
        actual_shape
        != expected_shape
    ):
        raise ValueError(
            f"{name} 形状={actual_shape}，"
            f"预期={expected_shape}。"
        )

if not np.array_equal(
    saved_landmarks,
    LANDMARK_MONTHS,
):
    raise ValueError(
        "Step 6 Landmark 配置不一致。"
    )

if not np.array_equal(
    saved_future_ends,
    FUTURE_END_MONTHS,
):
    raise ValueError(
        "Step 6 未来预测区间配置不一致。"
    )

if not np.isin(
    fold_id_long,
    np.arange(
        N_SPLITS
    ),
).all():
    raise ValueError(
        "长格式 fold_id 必须全部为 0～4。"
    )

if not np.array_equal(
    landmark_month_long,
    LANDMARK_MONTHS[
        landmark_index_long
    ],
):
    raise ValueError(
        "Landmark 索引与月份不一致。"
    )

if (
    np.any(
        analysis_time_month
        <= 0
    )
    or not np.isfinite(
        analysis_time_month
    ).all()
):
    raise ValueError(
        "Landmark 后随访时间非法。"
    )

valid_map = (
    row_index_map
    >= 0
)

patient_grid = np.broadcast_to(
    np.arange(
        EXPECTED_DEVELOPMENT_N,
        dtype=np.int32,
    )[:, None],
    row_index_map.shape,
)

landmark_grid = np.broadcast_to(
    np.arange(
        EXPECTED_LANDMARK_N,
        dtype=np.int32,
    )[None, :],
    row_index_map.shape,
)

mapped_rows = row_index_map[
    valid_map
].astype(
    np.int64
)

if not np.array_equal(
    local_patient_idx_long[
        mapped_rows
    ],
    patient_grid[
        valid_map
    ],
):
    raise ValueError(
        "row_index_map 与 "
        "local_patient_idx_long 不一致。"
    )

if not np.array_equal(
    landmark_index_long[
        mapped_rows
    ],
    landmark_grid[
        valid_map
    ],
):
    raise ValueError(
        "row_index_map 与 "
        "landmark_index_long 不一致。"
    )

for model_name in MODEL_ORDER:
    survival = (
        model_survival_raw[
            model_name
        ]
    )

    risk = (
        model_risk_raw[
            model_name
        ]
    )

    expected_shape = (
        EXPECTED_LONG_N,
        EXPECTED_INTERVAL_N,
    )

    if (
        survival.shape
        != expected_shape
    ):
        raise ValueError(
            f"{model_name} survival "
            f"形状={survival.shape}，"
            f"预期={expected_shape}。"
        )

    if (
        risk.shape
        != expected_shape
    ):
        raise ValueError(
            f"{model_name} risk "
            f"形状={risk.shape}，"
            f"预期={expected_shape}。"
        )

    if (
        not np.isfinite(
            survival
        ).all()
        or not np.isfinite(
            risk
        ).all()
    ):
        raise ValueError(
            f"{model_name} OOF "
            "存在 NaN 或无穷值。"
        )

    if np.any(
        (survival < -EPS)
        | (survival > 1 + EPS)
    ):
        raise ValueError(
            f"{model_name} 生存概率超出 0～1。"
        )

    if np.any(
        (risk < -EPS)
        | (risk > 1 + EPS)
    ):
        raise ValueError(
            f"{model_name} 累积风险超出 0～1。"
        )

    survival_violation_n = int(
        np.sum(
            np.diff(
                survival,
                axis=1,
            )
            > 1e-6
        )
    )

    risk_violation_n = int(
        np.sum(
            np.diff(
                risk,
                axis=1,
            )
            < -1e-6
        )
    )

    if (
        survival_violation_n
        != 0
    ):
        raise ValueError(
            f"{model_name} 生存概率存在 "
            f"{survival_violation_n} "
            "处单调性违反。"
        )

    if (
        risk_violation_n
        != 0
    ):
        raise ValueError(
            f"{model_name} 累积风险存在 "
            f"{risk_violation_n} "
            "处单调性违反。"
        )

    if not np.allclose(
        risk,
        1.0 - survival,
        atol=2e-6,
    ):
        max_diff = float(
            np.max(
                np.abs(
                    risk
                    - (
                        1.0
                        - survival
                    )
                )
            )
        )

        raise ValueError(
            f"{model_name} risk != 1-survival，"
            f"最大差值={max_diff:.3e}。"
        )


# =============================================================================
# 5. Cox 排序审计
# =============================================================================

if not np.isfinite(
    cox_log_partial_hazard_long
).all():
    raise ValueError(
        "Cox log partial hazard "
        "存在 NaN 或无穷值。"
    )

cox_audit_rows: list[
    dict[str, Any]
] = []

for fold_id in range(
    N_SPLITS
):
    for (
        landmark_position,
        landmark_month,
    ) in enumerate(
        LANDMARK_MONTHS
    ):
        rows = np.where(
            (
                fold_id_long
                == fold_id
            )
            & (
                landmark_index_long
                == landmark_position
            )
        )[0]

        survival_reference = (
            build_survival_array(
                event_within_60m[
                    rows
                ],
                analysis_time_month[
                    rows
                ],
            )
        )

        risk_5y = (
            model_risk_raw[
                "Cox"
            ][
                rows,
                -1,
            ].astype(
                np.float64
            )
        )

        linear_score = (
            cox_log_partial_hazard_long[
                rows
            ].astype(
                np.float64
            )
        )

        rank_correlation = float(
            spearmanr(
                risk_5y,
                linear_score,
            ).statistic
        )

        c_risk = float(
            concordance_index_ipcw(
                survival_reference,
                survival_reference,
                risk_5y,
                tau=float(
                    METRIC_TIMES[
                        -1
                    ]
                ),
            )[0]
        )

        c_linear = float(
            concordance_index_ipcw(
                survival_reference,
                survival_reference,
                linear_score,
                tau=float(
                    METRIC_TIMES[
                        -1
                    ]
                ),
            )[0]
        )

        cox_audit_rows.append(
            {
                "fold_id": int(
                    fold_id
                ),
                "landmark_month": float(
                    landmark_month
                ),
                "landmark_year": float(
                    landmark_month
                    / 12.0
                ),
                "record_n": int(
                    len(rows)
                ),
                "future_5y_event_n": int(
                    event_within_60m[
                        rows
                    ].sum()
                ),
                "spearman_risk_vs_linear_score": (
                    rank_correlation
                ),
                "uno_c_from_5y_risk": (
                    c_risk
                ),
                "uno_c_from_linear_score": (
                    c_linear
                ),
                "absolute_c_difference": (
                    abs(
                        c_risk
                        - c_linear
                    )
                ),
            }
        )

cox_audit = pd.DataFrame(
    cox_audit_rows
)

cox_audit.to_csv(
    OUTPUT_DIR
    / "cox_prediction_alignment_audit.csv",
    index=False,
    encoding="utf-8-sig",
)

if (
    cox_audit[
        "spearman_risk_vs_linear_score"
    ].min()
    < 0.999
):
    raise ValueError(
        "Cox 累积风险与线性风险评分"
        "排序审计未通过。"
    )

if (
    cox_audit[
        "absolute_c_difference"
    ].max()
    > 1e-3
):
    raise ValueError(
        "Cox 累积风险与线性评分"
        "Uno C 审计未通过。"
    )


# =============================================================================
# 6. 生存概率 ↔ 条件风险
# =============================================================================

def survival_to_hazard(
    survival: np.ndarray,
) -> np.ndarray:
    survival = np.clip(
        np.asarray(
            survival,
            dtype=np.float64,
        ),
        EPS,
        1.0,
    )

    previous_survival = (
        np.concatenate(
            [
                np.ones(
                    (
                        survival.shape[
                            0
                        ],
                        1,
                    ),
                    dtype=np.float64,
                ),
                survival[
                    :,
                    :-1,
                ],
            ],
            axis=1,
        )
    )

    hazard = (
        1.0
        - (
            survival
            / np.clip(
                previous_survival,
                EPS,
                1.0,
            )
        )
    )

    return np.clip(
        hazard,
        EPS,
        1.0 - EPS,
    ).astype(
        np.float32
    )


def hazard_to_risk(
    hazard: np.ndarray,
) -> np.ndarray:
    hazard = np.clip(
        np.asarray(
            hazard,
            dtype=np.float64,
        ),
        EPS,
        1.0 - EPS,
    )

    survival = np.cumprod(
        1.0 - hazard,
        axis=1,
    )

    return np.clip(
        1.0 - survival,
        0.0,
        1.0,
    ).astype(
        np.float32
    )


def probability_logit(
    probability: np.ndarray,
) -> np.ndarray:
    probability = np.clip(
        np.asarray(
            probability,
            dtype=np.float64,
        ),
        EPS,
        1.0 - EPS,
    )

    return (
        np.log(
            probability
        )
        - np.log1p(
            -probability
        )
    )


model_hazard_raw = {
    model_name: (
        survival_to_hazard(
            model_survival_raw[
                model_name
            ]
        )
    )
    for model_name
    in MODEL_ORDER
}

for model_name in MODEL_ORDER:
    reconstructed_risk = (
        hazard_to_risk(
            model_hazard_raw[
                model_name
            ]
        )
    )

    if not np.allclose(
        reconstructed_risk,
        model_risk_raw[
            model_name
        ],
        atol=2e-5,
    ):
        max_diff = float(
            np.max(
                np.abs(
                    reconstructed_risk
                    - model_risk_raw[
                        model_name
                    ]
                )
            )
        )

        raise ValueError(
            f"{model_name} 条件风险重建"
            "累积风险失败，"
            f"最大差值={max_diff:.3e}。"
        )


# =============================================================================
# 7. 五折交叉拟合 hazard 校准
# =============================================================================

def fit_interval_hazard_calibrator(
    hazard: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
) -> dict[str, Any]:
    """
    模型：
      logit(h_cal,j)
        = alpha_j
        + beta * logit(h_raw,j)

    alpha_j：
      10 个区间特异截距

    beta：
      一个共同斜率
    """

    hazard = np.asarray(
        hazard,
        dtype=np.float64,
    )

    target = np.asarray(
        target,
        dtype=np.float64,
    )

    mask = np.asarray(
        mask,
        dtype=bool,
    )

    if (
        hazard.shape
        != target.shape
        or hazard.shape
        != mask.shape
    ):
        raise ValueError(
            "校准输入形状不一致。"
        )

    raw_logit = (
        probability_logit(
            hazard
        )
    )

    interval_matrix = (
        np.broadcast_to(
            np.arange(
                EXPECTED_INTERVAL_N,
                dtype=np.int16,
            ),
            hazard.shape,
        )
    )

    x = raw_logit[
        mask
    ]

    y = target[
        mask
    ]

    interval_index = (
        interval_matrix[
            mask
        ]
    )

    if len(y) == 0:
        raise ValueError(
            "校准训练数据没有有效风险区间。"
        )

    if float(
        y.sum()
    ) <= 0:
        raise ValueError(
            "校准训练数据没有事件。"
        )

    initial = np.concatenate(
        [
            np.zeros(
                EXPECTED_INTERVAL_N,
                dtype=np.float64,
            ),
            np.asarray(
                [1.0],
                dtype=np.float64,
            ),
        ]
    )

    def objective_and_gradient(
        parameters: np.ndarray,
    ) -> tuple[
        float,
        np.ndarray,
    ]:
        alpha = parameters[
            :EXPECTED_INTERVAL_N
        ]

        beta = parameters[
            -1
        ]

        eta = (
            alpha[
                interval_index
            ]
            + beta
            * x
        )

        loss = np.mean(
            np.logaddexp(
                0.0,
                eta,
            )
            - y
            * eta
        )

        residual = (
            expit(
                eta
            )
            - y
        ) / len(y)

        gradient_alpha = (
            np.bincount(
                interval_index,
                weights=residual,
                minlength=(
                    EXPECTED_INTERVAL_N
                ),
            ).astype(
                np.float64
            )
        )

        gradient_beta = float(
            np.dot(
                residual,
                x,
            )
        )

        if (
            CALIBRATION_RIDGE
            > 0
        ):
            loss += (
                0.5
                * CALIBRATION_RIDGE
                * (
                    float(
                        np.mean(
                            alpha
                            ** 2
                        )
                    )
                    + float(
                        (
                            beta
                            - 1.0
                        )
                        ** 2
                    )
                )
            )

            gradient_alpha += (
                CALIBRATION_RIDGE
                * alpha
                / EXPECTED_INTERVAL_N
            )

            gradient_beta += (
                CALIBRATION_RIDGE
                * (
                    beta
                    - 1.0
                )
            )

        gradient = np.concatenate(
            [
                gradient_alpha,
                np.asarray(
                    [
                        gradient_beta
                    ],
                    dtype=np.float64,
                ),
            ]
        )

        return (
            float(loss),
            gradient,
        )

    bounds = (
        [
            (-8.0, 8.0)
        ]
        * EXPECTED_INTERVAL_N
        + [
            (0.05, 5.0)
        ]
    )

    result = minimize(
        fun=(
            objective_and_gradient
        ),
        x0=initial,
        method="L-BFGS-B",
        jac=True,
        bounds=bounds,
        options={
            "maxiter": 300,
            "ftol": 1e-12,
            "gtol": 1e-8,
            "maxls": 50,
        },
    )

    if not result.success:
        raise RuntimeError(
            "离散风险校准失败："
            + str(
                result.message
            )
        )

    return {
        "alpha": (
            result.x[
                :EXPECTED_INTERVAL_N
            ].astype(
                np.float64
            )
        ),
        "beta": float(
            result.x[
                -1
            ]
        ),
        "valid_interval_n": int(
            len(y)
        ),
        "event_interval_n": int(
            y.sum()
        ),
        "objective": float(
            result.fun
        ),
        "iterations": int(
            result.nit
        ),
    }


def apply_interval_hazard_calibrator(
    hazard: np.ndarray,
    fit: dict[str, Any],
) -> np.ndarray:
    raw_logit = (
        probability_logit(
            hazard
        )
    )

    alpha = np.asarray(
        fit[
            "alpha"
        ],
        dtype=np.float64,
    )

    beta = float(
        fit[
            "beta"
        ]
    )

    calibrated = expit(
        alpha[
            None,
            :,
        ]
        + beta
        * raw_logit
    )

    return np.clip(
        calibrated,
        EPS,
        1.0 - EPS,
    ).astype(
        np.float32
    )


model_hazard_calibrated: dict[
    str,
    np.ndarray,
] = {}

model_risk_calibrated: dict[
    str,
    np.ndarray,
] = {}

model_survival_calibrated: dict[
    str,
    np.ndarray,
] = {}

calibration_parameter_rows: list[
    dict[str, Any]
] = []

final_calibrators: dict[
    str,
    dict[str, Any],
] = {}

for model_name in MODEL_ORDER:
    print(
        "\n"
        + "-" * 110
    )
    print(
        f"{model_name}："
        "开始五折交叉拟合 hazard 校准"
    )
    print(
        "-" * 110
    )

    raw_hazard = (
        model_hazard_raw[
            model_name
        ]
    )

    calibrated_hazard = (
        np.full_like(
            raw_hazard,
            np.nan,
            dtype=np.float32,
        )
    )

    final_calibrators[
        model_name
    ] = {}

    for (
        landmark_position,
        landmark_month,
    ) in enumerate(
        LANDMARK_MONTHS
    ):
        landmark_mask = (
            landmark_index_long
            == landmark_position
        )

        for heldout_fold in range(
            N_SPLITS
        ):
            train_mask = (
                landmark_mask
                & (
                    fold_id_long
                    != heldout_fold
                )
            )

            valid_mask = (
                landmark_mask
                & (
                    fold_id_long
                    == heldout_fold
                )
            )

            fit = (
                fit_interval_hazard_calibrator(
                    raw_hazard[
                        train_mask
                    ],
                    future_event_long[
                        train_mask
                    ],
                    future_at_risk_long[
                        train_mask
                    ],
                )
            )

            calibrated_hazard[
                valid_mask
            ] = (
                apply_interval_hazard_calibrator(
                    raw_hazard[
                        valid_mask
                    ],
                    fit,
                )
            )

            for (
                interval_position,
                interval_month,
            ) in enumerate(
                FUTURE_END_MONTHS
            ):
                calibration_parameter_rows.append(
                    {
                        "model": model_name,
                        "fit_scope": (
                            "crossfit"
                        ),
                        "heldout_fold": int(
                            heldout_fold
                        ),
                        "landmark_month": float(
                            landmark_month
                        ),
                        "landmark_year": float(
                            landmark_month
                            / 12.0
                        ),
                        "interval_position": int(
                            interval_position
                        ),
                        "interval_end_month": float(
                            interval_month
                        ),
                        "alpha_interval": float(
                            fit[
                                "alpha"
                            ][
                                interval_position
                            ]
                        ),
                        "beta_common_slope": float(
                            fit[
                                "beta"
                            ]
                        ),
                        "valid_interval_n": int(
                            fit[
                                "valid_interval_n"
                            ]
                        ),
                        "event_interval_n": int(
                            fit[
                                "event_interval_n"
                            ]
                        ),
                        "objective": float(
                            fit[
                                "objective"
                            ]
                        ),
                        "optimizer_iterations": int(
                            fit[
                                "iterations"
                            ]
                        ),
                    }
                )

            print(
                f"{model_name} | "
                f"Landmark "
                f"{landmark_month / 12:.0f} 年 | "
                f"held-out fold "
                f"{heldout_fold} | "
                f"beta="
                f"{fit['beta']:.4f}"
            )

        # 最终校准器：
        # 仅使用全开发集 OOF 预测 + 开发集真实结局。
        final_fit = (
            fit_interval_hazard_calibrator(
                raw_hazard[
                    landmark_mask
                ],
                future_event_long[
                    landmark_mask
                ],
                future_at_risk_long[
                    landmark_mask
                ],
            )
        )

        final_calibrators[
            model_name
        ][
            str(
                int(
                    landmark_month
                )
            )
        ] = {
            "landmark_month": float(
                landmark_month
            ),
            "alpha": (
                final_fit[
                    "alpha"
                ].tolist()
            ),
            "beta": float(
                final_fit[
                    "beta"
                ]
            ),
            "valid_interval_n": int(
                final_fit[
                    "valid_interval_n"
                ]
            ),
            "event_interval_n": int(
                final_fit[
                    "event_interval_n"
                ]
            ),
        }

        for (
            interval_position,
            interval_month,
        ) in enumerate(
            FUTURE_END_MONTHS
        ):
            calibration_parameter_rows.append(
                {
                    "model": model_name,
                    "fit_scope": (
                        "final_all_development_oof"
                    ),
                    "heldout_fold": -1,
                    "landmark_month": float(
                        landmark_month
                    ),
                    "landmark_year": float(
                        landmark_month
                        / 12.0
                    ),
                    "interval_position": int(
                        interval_position
                    ),
                    "interval_end_month": float(
                        interval_month
                    ),
                    "alpha_interval": float(
                        final_fit[
                            "alpha"
                        ][
                            interval_position
                        ]
                    ),
                    "beta_common_slope": float(
                        final_fit[
                            "beta"
                        ]
                    ),
                    "valid_interval_n": int(
                        final_fit[
                            "valid_interval_n"
                        ]
                    ),
                    "event_interval_n": int(
                        final_fit[
                            "event_interval_n"
                        ]
                    ),
                    "objective": float(
                        final_fit[
                            "objective"
                        ]
                    ),
                    "optimizer_iterations": int(
                        final_fit[
                            "iterations"
                        ]
                    ),
                }
            )

    if not np.isfinite(
        calibrated_hazard
    ).all():
        raise ValueError(
            f"{model_name} 交叉拟合校准后"
            "仍存在缺失值。"
        )

    calibrated_risk = (
        hazard_to_risk(
            calibrated_hazard
        )
    )

    calibrated_survival = (
        1.0
        - calibrated_risk
    ).astype(
        np.float32
    )

    if int(
        np.sum(
            np.diff(
                calibrated_risk,
                axis=1,
            )
            < -1e-7
        )
    ) != 0:
        raise ValueError(
            f"{model_name} 校准后"
            "累积风险不单调。"
        )

    model_hazard_calibrated[
        model_name
    ] = calibrated_hazard

    model_risk_calibrated[
        model_name
    ] = calibrated_risk

    model_survival_calibrated[
        model_name
    ] = calibrated_survival

    model_file_name = (
        safe_name(
            model_name
        )
    )

    np.save(
        OUTPUT_DIR
        / (
            f"{model_file_name}_crossfit_"
            "calibrated_oof_hazard_long.npy"
        ),
        calibrated_hazard,
    )

    np.save(
        OUTPUT_DIR
        / (
            f"{model_file_name}_crossfit_"
            "calibrated_oof_risk_long.npy"
        ),
        calibrated_risk,
    )

    np.save(
        OUTPUT_DIR
        / (
            f"{model_file_name}_crossfit_"
            "calibrated_oof_survival_long.npy"
        ),
        calibrated_survival,
    )

calibration_parameters = pd.DataFrame(
    calibration_parameter_rows
)

calibration_parameters.to_csv(
    OUTPUT_DIR
    / "four_model_hazard_calibration_parameters.csv",
    index=False,
    encoding="utf-8-sig",
)

save_json(
    final_calibrators,
    OUTPUT_DIR
    / "four_model_final_development_oof_calibrators.json",
)


# =============================================================================
# 8. 指标计算函数
# =============================================================================

def evaluate_survival_predictions(
    survival_train: np.ndarray,
    survival_test: np.ndarray,
    risk_matrix: np.ndarray,
) -> dict[str, Any]:
    risk_matrix = np.asarray(
        risk_matrix,
        dtype=np.float64,
    )

    if (
        risk_matrix.ndim
        != 2
        or risk_matrix.shape[
            1
        ]
        != EXPECTED_INTERVAL_N
    ):
        raise ValueError(
            "risk_matrix 形状非法。"
        )

    survival_matrix = (
        1.0
        - risk_matrix
    )

    c_index = float(
        concordance_index_ipcw(
            survival_train,
            survival_test,
            risk_matrix[
                :,
                -1,
            ],
            tau=float(
                METRIC_TIMES[
                    -1
                ]
            ),
        )[0]
    )

    dynamic_auc, mean_auc = (
        cumulative_dynamic_auc(
            survival_train,
            survival_test,
            risk_matrix,
            METRIC_TIMES,
        )
    )

    _, brier_values = (
        brier_score(
            survival_train,
            survival_test,
            survival_matrix,
            METRIC_TIMES,
        )
    )

    ibs = float(
        integrated_brier_score(
            survival_train,
            survival_test,
            survival_matrix,
            METRIC_TIMES,
        )
    )

    return {
        "uno_c_index_5y": (
            c_index
        ),
        "dynamic_auc": (
            np.asarray(
                dynamic_auc,
                dtype=np.float64,
            )
        ),
        "integrated_dynamic_auc": (
            float(
                mean_auc
            )
        ),
        "brier": (
            np.asarray(
                brier_values,
                dtype=np.float64,
            )
        ),
        "integrated_brier": (
            ibs
        ),
    }


def safe_evaluate_survival_predictions(
    survival_train: np.ndarray,
    survival_test: np.ndarray,
    risk_matrix: np.ndarray,
) -> dict[str, Any] | None:
    try:
        return (
            evaluate_survival_predictions(
                survival_train,
                survival_test,
                risk_matrix,
            )
        )
    except (
        ValueError,
        ArithmeticError,
        ZeroDivisionError,
        FloatingPointError,
    ):
        return None


# =============================================================================
# 9. Raw 和 cross-fitted calibrated point performance
# =============================================================================

def calculate_point_tables(
    model_risks: dict[
        str,
        np.ndarray,
    ],
    prediction_stage: str,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    landmark_rows: list[
        dict[str, Any]
    ] = []

    horizon_rows: list[
        dict[str, Any]
    ] = []

    selected_horizon_rows: list[
        dict[str, Any]
    ] = []

    for (
        landmark_position,
        landmark_month,
    ) in enumerate(
        LANDMARK_MONTHS
    ):
        rows = np.where(
            landmark_index_long
            == landmark_position
        )[0]

        survival_reference = (
            build_survival_array(
                event_within_60m[
                    rows
                ],
                analysis_time_month[
                    rows
                ],
            )
        )

        for model_name in MODEL_ORDER:
            risk_matrix = (
                model_risks[
                    model_name
                ][
                    rows,
                    :,
                ]
            )

            metrics = (
                evaluate_survival_predictions(
                    survival_reference,
                    survival_reference,
                    risk_matrix,
                )
            )

            landmark_rows.append(
                {
                    "prediction_stage": (
                        prediction_stage
                    ),
                    "model": (
                        model_name
                    ),
                    "landmark_position": int(
                        landmark_position
                    ),
                    "landmark_month": float(
                        landmark_month
                    ),
                    "landmark_year": float(
                        landmark_month
                        / 12.0
                    ),
                    "risk_set_n": int(
                        len(rows)
                    ),
                    "future_5y_event_n": int(
                        event_within_60m[
                            rows
                        ].sum()
                    ),
                    "uno_c_index_5y": float(
                        metrics[
                            "uno_c_index_5y"
                        ]
                    ),
                    "integrated_dynamic_auc": float(
                        metrics[
                            "integrated_dynamic_auc"
                        ]
                    ),
                    "integrated_brier": float(
                        metrics[
                            "integrated_brier"
                        ]
                    ),
                }
            )

            for (
                horizon_position,
                horizon_month,
            ) in enumerate(
                FUTURE_END_MONTHS
            ):
                row = {
                    "prediction_stage": (
                        prediction_stage
                    ),
                    "model": (
                        model_name
                    ),
                    "landmark_position": int(
                        landmark_position
                    ),
                    "landmark_month": float(
                        landmark_month
                    ),
                    "landmark_year": float(
                        landmark_month
                        / 12.0
                    ),
                    "horizon_position": int(
                        horizon_position
                    ),
                    "horizon_month": float(
                        horizon_month
                    ),
                    "horizon_year": float(
                        horizon_month
                        / 12.0
                    ),
                    "dynamic_auc": float(
                        metrics[
                            "dynamic_auc"
                        ][
                            horizon_position
                        ]
                    ),
                    "brier_score": float(
                        metrics[
                            "brier"
                        ][
                            horizon_position
                        ]
                    ),
                }

                horizon_rows.append(
                    row
                )

                if horizon_month in {
                    12.0,
                    36.0,
                    60.0,
                }:
                    selected_horizon_rows.append(
                        row.copy()
                    )

    return (
        pd.DataFrame(
            landmark_rows
        ),
        pd.DataFrame(
            horizon_rows
        ),
        pd.DataFrame(
            selected_horizon_rows
        ),
    )


(
    raw_landmark_metrics,
    raw_horizon_metrics,
    raw_1y_3y_5y_metrics,
) = calculate_point_tables(
    model_risk_raw,
    prediction_stage="raw_oof",
)

(
    calibrated_landmark_metrics,
    calibrated_horizon_metrics,
    calibrated_1y_3y_5y_metrics,
) = calculate_point_tables(
    model_risk_calibrated,
    prediction_stage=(
        "crossfit_calibrated_oof"
    ),
)

raw_landmark_metrics.to_csv(
    OUTPUT_DIR
    / "four_model_raw_landmark_metrics.csv",
    index=False,
    encoding="utf-8-sig",
)

raw_horizon_metrics.to_csv(
    OUTPUT_DIR
    / "four_model_raw_horizon_metrics.csv",
    index=False,
    encoding="utf-8-sig",
)

raw_1y_3y_5y_metrics.to_csv(
    OUTPUT_DIR
    / "four_model_raw_1y_3y_5y_metrics.csv",
    index=False,
    encoding="utf-8-sig",
)

calibrated_landmark_metrics.to_csv(
    OUTPUT_DIR
    / "four_model_crossfit_calibrated_landmark_metrics.csv",
    index=False,
    encoding="utf-8-sig",
)

calibrated_horizon_metrics.to_csv(
    OUTPUT_DIR
    / "four_model_crossfit_calibrated_horizon_metrics.csv",
    index=False,
    encoding="utf-8-sig",
)

calibrated_1y_3y_5y_metrics.to_csv(
    OUTPUT_DIR
    / "four_model_crossfit_calibrated_1y_3y_5y_metrics.csv",
    index=False,
    encoding="utf-8-sig",
)


# =============================================================================
# 10. 患者级 paired bootstrap
# =============================================================================

BOOTSTRAP_METRIC_INDEX = {
    metric_name: index
    for index, metric_name in enumerate(
        METRIC_NAMES
    )
}

checkpoint_file = (
    BOOTSTRAP_DIR
    / (
        f"paired_bootstrap_"
        f"{BOOTSTRAP_REPS}_replicates.npz"
    )
)

bootstrap_shape = (
    BOOTSTRAP_REPS,
    len(
        MODEL_ORDER
    ),
    EXPECTED_LANDMARK_N,
    len(
        METRIC_NAMES
    ),
)

if checkpoint_file.exists():
    checkpoint = np.load(
        checkpoint_file,
        allow_pickle=False,
    )

    bootstrap_metrics = (
        checkpoint[
            "bootstrap_metrics"
        ].astype(
            np.float64
        )
    )

    completed_replicates = (
        checkpoint[
            "completed_replicates"
        ].astype(
            bool
        )
    )

    if (
        bootstrap_metrics.shape
        != bootstrap_shape
    ):
        raise ValueError(
            "已有 Bootstrap checkpoint "
            "形状与当前配置不一致。"
        )

    if (
        completed_replicates.shape
        != (
            BOOTSTRAP_REPS,
        )
    ):
        raise ValueError(
            "已有 Bootstrap checkpoint "
            "完成标记形状不一致。"
        )

    print(
        "\n检测到 Bootstrap 断点："
        f"{int(completed_replicates.sum())}/"
        f"{BOOTSTRAP_REPS} 已完成。"
    )

else:
    bootstrap_metrics = np.full(
        bootstrap_shape,
        np.nan,
        dtype=np.float64,
    )

    completed_replicates = np.zeros(
        BOOTSTRAP_REPS,
        dtype=bool,
    )


# 每个 Landmark 的完整原始风险集，
# 作为 IPCW 的 survival_train/reference。
landmark_reference_rows: list[
    np.ndarray
] = []

landmark_survival_reference: list[
    np.ndarray
] = []

for landmark_position in range(
    EXPECTED_LANDMARK_N
):
    rows = np.where(
        landmark_index_long
        == landmark_position
    )[0]

    landmark_reference_rows.append(
        rows
    )

    landmark_survival_reference.append(
        build_survival_array(
            event_within_60m[
                rows
            ],
            analysis_time_month[
                rows
            ],
        )
    )


bootstrap_start = time.time()

for bootstrap_index in range(
    BOOTSTRAP_REPS
):
    if completed_replicates[
        bootstrap_index
    ]:
        continue

    # 每个 replicate 使用独立、可复现的 seed，
    # 因此断点续跑不会改变结果。
    rng = np.random.default_rng(
        RANDOM_SEED
        + bootstrap_index
        * 1009
    )

    sampled_patients = (
        rng.integers(
            0,
            EXPECTED_DEVELOPMENT_N,
            size=EXPECTED_DEVELOPMENT_N,
            endpoint=False,
        )
    )

    for landmark_position in range(
        EXPECTED_LANDMARK_N
    ):
        sampled_rows = (
            row_index_map[
                sampled_patients,
                landmark_position,
            ]
        )

        sampled_rows = sampled_rows[
            sampled_rows
            >= 0
        ].astype(
            np.int64
        )

        if (
            len(sampled_rows)
            < 50
        ):
            continue

        sampled_event = (
            event_within_60m[
                sampled_rows
            ]
        )

        if int(
            sampled_event.sum()
        ) < 5:
            continue

        survival_test = (
            build_survival_array(
                sampled_event,
                analysis_time_month[
                    sampled_rows
                ],
            )
        )

        survival_train = (
            landmark_survival_reference[
                landmark_position
            ]
        )

        for (
            model_index,
            model_name,
        ) in enumerate(
            MODEL_ORDER
        ):
            metrics = (
                safe_evaluate_survival_predictions(
                    survival_train,
                    survival_test,
                    model_risk_calibrated[
                        model_name
                    ][
                        sampled_rows,
                        :,
                    ],
                )
            )

            if metrics is None:
                continue

            bootstrap_metrics[
                bootstrap_index,
                model_index,
                landmark_position,
                BOOTSTRAP_METRIC_INDEX[
                    "uno_c_index_5y"
                ],
            ] = (
                metrics[
                    "uno_c_index_5y"
                ]
            )

            bootstrap_metrics[
                bootstrap_index,
                model_index,
                landmark_position,
                BOOTSTRAP_METRIC_INDEX[
                    "integrated_dynamic_auc"
                ],
            ] = (
                metrics[
                    "integrated_dynamic_auc"
                ]
            )

            bootstrap_metrics[
                bootstrap_index,
                model_index,
                landmark_position,
                BOOTSTRAP_METRIC_INDEX[
                    "integrated_brier"
                ],
            ] = (
                metrics[
                    "integrated_brier"
                ]
            )

    completed_replicates[
        bootstrap_index
    ] = True

    completed_n = int(
        completed_replicates.sum()
    )

    if (
        completed_n
        % BOOTSTRAP_SAVE_EVERY
        == 0
        or completed_n
        == BOOTSTRAP_REPS
    ):
        np.savez_compressed(
            checkpoint_file,
            bootstrap_metrics=(
                bootstrap_metrics
            ),
            completed_replicates=(
                completed_replicates
            ),
        )

    if (
        completed_n
        % 25
        == 0
        or completed_n
        == BOOTSTRAP_REPS
    ):
        print(
            "Paired bootstrap："
            f"{completed_n}/"
            f"{BOOTSTRAP_REPS} | "
            "本次已运行 "
            f"{format_duration(time.time() - bootstrap_start)}"
        )


# =============================================================================
# 11. Bootstrap：Landmark 级 CI
# =============================================================================

landmark_bootstrap_rows: list[
    dict[str, Any]
] = []

point_metric_lookup: dict[
    tuple[
        str,
        int,
        str,
    ],
    float,
] = {}

for _, row in (
    calibrated_landmark_metrics.iterrows()
):
    model_name = str(
        row[
            "model"
        ]
    )

    landmark_position = int(
        row[
            "landmark_position"
        ]
    )

    for metric_name in (
        METRIC_NAMES
    ):
        point_metric_lookup[
            (
                model_name,
                landmark_position,
                metric_name,
            )
        ] = float(
            row[
                metric_name
            ]
        )


for (
    model_index,
    model_name,
) in enumerate(
    MODEL_ORDER
):
    for landmark_position in range(
        EXPECTED_LANDMARK_N
    ):
        landmark_month = (
            LANDMARK_MONTHS[
                landmark_position
            ]
        )

        for metric_name in (
            METRIC_NAMES
        ):
            metric_index = (
                BOOTSTRAP_METRIC_INDEX[
                    metric_name
                ]
            )

            values = (
                bootstrap_metrics[
                    :,
                    model_index,
                    landmark_position,
                    metric_index,
                ]
            )

            (
                lower,
                upper,
                valid_n,
            ) = percentile_interval(
                values,
                expected_n=BOOTSTRAP_REPS,
            )

            landmark_bootstrap_rows.append(
                {
                    "model": (
                        model_name
                    ),
                    "landmark_position": int(
                        landmark_position
                    ),
                    "landmark_month": float(
                        landmark_month
                    ),
                    "landmark_year": float(
                        landmark_month
                        / 12.0
                    ),
                    "metric": (
                        metric_name
                    ),
                    "point_estimate": float(
                        point_metric_lookup[
                            (
                                model_name,
                                landmark_position,
                                metric_name,
                            )
                        ]
                    ),
                    "lower_95": (
                        lower
                    ),
                    "upper_95": (
                        upper
                    ),
                    "valid_bootstrap_n": int(
                        valid_n
                    ),
                }
            )

landmark_bootstrap = pd.DataFrame(
    landmark_bootstrap_rows
)

landmark_bootstrap.to_csv(
    OUTPUT_DIR
    / "four_model_landmark_bootstrap_performance.csv",
    index=False,
    encoding="utf-8-sig",
)


# =============================================================================
# 12. 跨 Landmark 等权平均 + Bootstrap CI
# =============================================================================

model_summary_rows: list[
    dict[str, Any]
] = []

scope_bootstrap_cache: dict[
    tuple[
        str,
        str,
        str,
    ],
    np.ndarray,
] = {}

for (
    scope_name,
    landmark_positions,
) in SUMMARY_SCOPES.items():
    for (
        model_index,
        model_name,
    ) in enumerate(
        MODEL_ORDER
    ):
        point_subset = (
            calibrated_landmark_metrics.loc[
                (
                    calibrated_landmark_metrics[
                        "model"
                    ]
                    == model_name
                )
                & (
                    calibrated_landmark_metrics[
                        "landmark_position"
                    ].isin(
                        landmark_positions
                    )
                )
            ]
            .sort_values(
                "landmark_position"
            )
        )

        if (
            len(
                point_subset
            )
            != len(
                landmark_positions
            )
        ):
            raise ValueError(
                f"{scope_name} | "
                f"{model_name} "
                "Landmark 数量不完整。"
            )

        row = {
            "scope": scope_name,
            "model": model_name,
            "landmark_positions": ",".join(
                str(
                    int(x)
                )
                for x in (
                    landmark_positions
                )
            ),
            "landmark_years": ",".join(
                str(
                    int(
                        LANDMARK_MONTHS[
                            x
                        ]
                        / 12
                    )
                )
                for x in (
                    landmark_positions
                )
            ),
        }

        for metric_name in (
            METRIC_NAMES
        ):
            metric_index = (
                BOOTSTRAP_METRIC_INDEX[
                    metric_name
                ]
            )

            point_value = float(
                point_subset[
                    metric_name
                ].mean()
            )

            replicate_by_landmark = (
                bootstrap_metrics[
                    :,
                    model_index,
                    :,
                    metric_index,
                ]
            )

            replicate_mean = (
                complete_row_mean(
                    replicate_by_landmark,
                    landmark_positions,
                )
            )

            scope_bootstrap_cache[
                (
                    scope_name,
                    model_name,
                    metric_name,
                )
            ] = replicate_mean

            (
                lower,
                upper,
                valid_n,
            ) = percentile_interval(
                replicate_mean,
                expected_n=BOOTSTRAP_REPS,
            )

            prefix = {
                "uno_c_index_5y": (
                    "mean_uno_c_index_5y"
                ),
                "integrated_dynamic_auc": (
                    "mean_integrated_dynamic_auc"
                ),
                "integrated_brier": (
                    "mean_integrated_brier"
                ),
            }[
                metric_name
            ]

            row[
                prefix
            ] = point_value

            row[
                (
                    prefix
                    + "_lower_95"
                )
            ] = lower

            row[
                (
                    prefix
                    + "_upper_95"
                )
            ] = upper

            row[
                (
                    prefix
                    + "_valid_bootstrap_n"
                )
            ] = valid_n

        model_summary_rows.append(
            row
        )

model_summary = pd.DataFrame(
    model_summary_rows
)

model_summary.to_csv(
    OUTPUT_DIR
    / "four_model_equal_weight_mean_performance.csv",
    index=False,
    encoding="utf-8-sig",
)


# =============================================================================
# 13. 预设患者级配对 Bootstrap 模型差值
# =============================================================================

pairwise_rows: list[
    dict[str, Any]
] = []

for (
    scope_name,
    landmark_positions,
) in SUMMARY_SCOPES.items():
    for (
        model_a,
        model_b,
    ) in PAIRWISE_COMPARISONS:
        for metric_name in (
            METRIC_NAMES
        ):
            a_boot = (
                scope_bootstrap_cache[
                    (
                        scope_name,
                        model_a,
                        metric_name,
                    )
                ]
            )

            b_boot = (
                scope_bootstrap_cache[
                    (
                        scope_name,
                        model_b,
                        metric_name,
                    )
                ]
            )

            difference = (
                a_boot
                - b_boot
            )

            (
                lower,
                upper,
                valid_n,
            ) = percentile_interval(
                difference,
                expected_n=BOOTSTRAP_REPS,
            )

            point_a = float(
                model_summary.loc[
                    (
                        model_summary[
                            "scope"
                        ]
                        == scope_name
                    )
                    & (
                        model_summary[
                            "model"
                        ]
                        == model_a
                    ),
                    {
                        "uno_c_index_5y": (
                            "mean_uno_c_index_5y"
                        ),
                        "integrated_dynamic_auc": (
                            "mean_integrated_dynamic_auc"
                        ),
                        "integrated_brier": (
                            "mean_integrated_brier"
                        ),
                    }[
                        metric_name
                    ],
                ].iloc[
                    0
                ]
            )

            point_b = float(
                model_summary.loc[
                    (
                        model_summary[
                            "scope"
                        ]
                        == scope_name
                    )
                    & (
                        model_summary[
                            "model"
                        ]
                        == model_b
                    ),
                    {
                        "uno_c_index_5y": (
                            "mean_uno_c_index_5y"
                        ),
                        "integrated_dynamic_auc": (
                            "mean_integrated_dynamic_auc"
                        ),
                        "integrated_brier": (
                            "mean_integrated_brier"
                        ),
                    }[
                        metric_name
                    ],
                ].iloc[
                    0
                ]
            )

            higher = (
                HIGHER_IS_BETTER[
                    metric_name
                ]
            )

            pairwise_rows.append(
                {
                    "scope": (
                        scope_name
                    ),
                    "model_a": (
                        model_a
                    ),
                    "model_b": (
                        model_b
                    ),
                    "comparison_label": (
                        f"{model_a} - {model_b}"
                    ),
                    "metric": (
                        metric_name
                    ),
                    "higher_is_better": bool(
                        higher
                    ),
                    "model_a_value": (
                        point_a
                    ),
                    "model_b_value": (
                        point_b
                    ),
                    "difference_a_minus_b": float(
                        point_a
                        - point_b
                    ),
                    "difference_lower_95": (
                        lower
                    ),
                    "difference_upper_95": (
                        upper
                    ),
                    "valid_bootstrap_n": int(
                        valid_n
                    ),
                    "model_a_favorable_fraction": (
                        favorable_fraction(
                            difference,
                            higher_is_better=(
                                higher
                            ),
                        )
                    ),
                    "bootstrap_two_sided_p": (
                        two_sided_bootstrap_p(
                            difference
                        )
                    ),
                }
            )

pairwise_comparison = pd.DataFrame(
    pairwise_rows
)

pairwise_comparison.to_csv(
    OUTPUT_DIR
    / "four_model_paired_bootstrap_comparison.csv",
    index=False,
    encoding="utf-8-sig",
)


# =============================================================================
# 14. Kaplan–Meier 校准
# =============================================================================

def km_event_risk(
    time_month: np.ndarray,
    event: np.ndarray,
    horizon_month: float,
) -> tuple[
    float,
    float,
    float,
]:
    time_month = np.asarray(
        time_month,
        dtype=np.float64,
    )

    event = np.asarray(
        event,
        dtype=bool,
    )

    if len(
        time_month
    ) == 0:
        return (
            np.nan,
            np.nan,
            np.nan,
        )

    try:
        (
            km_time,
            km_survival,
            km_ci,
        ) = kaplan_meier_estimator(
            event,
            time_month,
            conf_type="log-log",
        )

        eligible = np.where(
            km_time
            <= horizon_month
        )[0]

        if len(
            eligible
        ) == 0:
            survival = 1.0
            lower_survival = 1.0
            upper_survival = 1.0
        else:
            index = int(
                eligible[
                    -1
                ]
            )

            survival = float(
                km_survival[
                    index
                ]
            )

            lower_survival = float(
                km_ci[
                    0,
                    index,
                ]
            )

            upper_survival = float(
                km_ci[
                    1,
                    index,
                ]
            )

        return (
            float(
                1.0
                - survival
            ),
            float(
                1.0
                - upper_survival
            ),
            float(
                1.0
                - lower_survival
            ),
        )

    except TypeError:
        # 兼容旧版 sksurv：
        # 无 conf_type 参数时仍给出点估计。
        (
            km_time,
            km_survival,
        ) = kaplan_meier_estimator(
            event,
            time_month,
        )

        eligible = np.where(
            km_time
            <= horizon_month
        )[0]

        if len(
            eligible
        ) == 0:
            survival = 1.0
        else:
            survival = float(
                km_survival[
                    eligible[
                        -1
                    ]
                ]
            )

        risk = float(
            1.0
            - survival
        )

        return (
            risk,
            np.nan,
            np.nan,
        )


overall_calibration_rows: list[
    dict[str, Any]
] = []

decile_calibration_rows: list[
    dict[str, Any]
] = []

for (
    landmark_position,
    landmark_month,
) in enumerate(
    LANDMARK_MONTHS
):
    rows = np.where(
        landmark_index_long
        == landmark_position
    )[0]

    residual_time = (
        analysis_time_month[
            rows
        ]
    )

    event_indicator = (
        event_within_60m[
            rows
        ]
    )

    for model_name in MODEL_ORDER:
        risk_matrix = (
            model_risk_calibrated[
                model_name
            ][
                rows,
                :,
            ]
        )

        for horizon_month in [
            12.0,
            36.0,
            60.0,
        ]:
            horizon_position = int(
                np.where(
                    np.isclose(
                        FUTURE_END_MONTHS,
                        horizon_month,
                    )
                )[0][0]
            )

            predicted_risk = (
                risk_matrix[
                    :,
                    horizon_position,
                ]
            )

            (
                observed_risk,
                observed_lower,
                observed_upper,
            ) = km_event_risk(
                residual_time,
                event_indicator,
                horizon_month,
            )

            overall_calibration_rows.append(
                {
                    "model": (
                        model_name
                    ),
                    "landmark_month": float(
                        landmark_month
                    ),
                    "landmark_year": float(
                        landmark_month
                        / 12.0
                    ),
                    "horizon_month": float(
                        horizon_month
                    ),
                    "horizon_year": float(
                        horizon_month
                        / 12.0
                    ),
                    "risk_set_n": int(
                        len(rows)
                    ),
                    "mean_predicted_risk": float(
                        np.mean(
                            predicted_risk
                        )
                    ),
                    "km_observed_risk": (
                        observed_risk
                    ),
                    "km_lower_95": (
                        observed_lower
                    ),
                    "km_upper_95": (
                        observed_upper
                    ),
                    "calibration_difference": float(
                        np.mean(
                            predicted_risk
                        )
                        - observed_risk
                    ),
                }
            )

            risk_groups = (
                pd.qcut(
                    pd.Series(
                        predicted_risk
                    ),
                    q=10,
                    labels=False,
                    duplicates="drop",
                ).to_numpy()
            )

            valid_groups = (
                risk_groups[
                    ~pd.isna(
                        risk_groups
                    )
                ]
            )

            if len(
                valid_groups
            ) == 0:
                raise ValueError(
                    f"{model_name} | "
                    f"Landmark "
                    f"{landmark_month / 12:.0f} 年 | "
                    f"未来 "
                    f"{horizon_month / 12:.0f} 年："
                    "无法形成有效风险分组。"
                )

            for risk_group in np.sort(
                np.unique(
                    valid_groups
                )
            ):
                group_mask = (
                    risk_groups
                    == risk_group
                )

                (
                    group_risk,
                    group_lower,
                    group_upper,
                ) = km_event_risk(
                    residual_time[
                        group_mask
                    ],
                    event_indicator[
                        group_mask
                    ],
                    horizon_month,
                )

                decile_calibration_rows.append(
                    {
                        "model": (
                            model_name
                        ),
                        "landmark_month": float(
                            landmark_month
                        ),
                        "landmark_year": float(
                            landmark_month
                            / 12.0
                        ),
                        "horizon_month": float(
                            horizon_month
                        ),
                        "horizon_year": float(
                            horizon_month
                            / 12.0
                        ),
                        "risk_group": int(
                            risk_group
                        )
                        + 1,
                        "patient_n": int(
                            group_mask.sum()
                        ),
                        "mean_predicted_risk": float(
                            np.mean(
                                predicted_risk[
                                    group_mask
                                ]
                            )
                        ),
                        "km_observed_risk": (
                            group_risk
                        ),
                        "km_lower_95": (
                            group_lower
                        ),
                        "km_upper_95": (
                            group_upper
                        ),
                    }
                )

overall_calibration = pd.DataFrame(
    overall_calibration_rows
)

decile_calibration = pd.DataFrame(
    decile_calibration_rows
)

overall_calibration.to_csv(
    OUTPUT_DIR
    / "four_model_calibration_overall.csv",
    index=False,
    encoding="utf-8-sig",
)

decile_calibration.to_csv(
    OUTPUT_DIR
    / "four_model_calibration_deciles.csv",
    index=False,
    encoding="utf-8-sig",
)


# =============================================================================
# 15. 生存 DCA：IPCW net benefit
# =============================================================================

def step_function_value(
    time_grid: np.ndarray,
    probability: np.ndarray,
    query_time: np.ndarray | float,
) -> np.ndarray:
    time_grid = np.asarray(
        time_grid,
        dtype=np.float64,
    )

    probability = np.asarray(
        probability,
        dtype=np.float64,
    )

    query = np.asarray(
        query_time,
        dtype=np.float64,
    )

    indices = np.searchsorted(
        time_grid,
        query,
        side="right",
    ) - 1

    result = np.ones_like(
        query,
        dtype=np.float64,
    )

    valid = (
        indices
        >= 0
    )

    result[
        valid
    ] = probability[
        indices[
            valid
        ]
    ]

    return result


def ipcw_dca_weights(
    event: np.ndarray,
    time_month: np.ndarray,
    horizon_month: float,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    event = np.asarray(
        event,
        dtype=bool,
    )

    time_month = np.asarray(
        time_month,
        dtype=np.float64,
    )

    # reverse KM：
    # 估计 censoring survival G(t)
    (
        censor_time,
        censor_survival,
    ) = kaplan_meier_estimator(
        event,
        time_month,
        reverse=True,
    )

    event_by_horizon = (
        event
        & (
            time_month
            <= horizon_month
        )
    )

    known_non_event = (
        time_month
        > horizon_month
    )

    weights = np.zeros(
        len(
            time_month
        ),
        dtype=np.float64,
    )

    if np.any(
        event_by_horizon
    ):
        query_event_time = (
            np.nextafter(
                time_month[
                    event_by_horizon
                ],
                -np.inf,
            )
        )

        g_event = (
            step_function_value(
                censor_time,
                censor_survival,
                query_event_time,
            )
        )

        weights[
            event_by_horizon
        ] = (
            1.0
            / np.clip(
                g_event,
                1e-6,
                None,
            )
        )

    if np.any(
        known_non_event
    ):
        g_horizon = float(
            step_function_value(
                censor_time,
                censor_survival,
                float(
                    horizon_month
                ),
            )
        )

        weights[
            known_non_event
        ] = (
            1.0
            / max(
                g_horizon,
                1e-6,
            )
        )

    observed_binary = (
        event_by_horizon
        .astype(
            np.int8
        )
    )

    known_mask = (
        event_by_horizon
        | known_non_event
    )

    return (
        observed_binary,
        weights,
        known_mask,
    )


def survival_net_benefit(
    predicted_risk: np.ndarray,
    event: np.ndarray,
    time_month: np.ndarray,
    horizon_month: float,
    thresholds: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    predicted_risk = np.asarray(
        predicted_risk,
        dtype=np.float64,
    )

    thresholds = np.asarray(
        thresholds,
        dtype=np.float64,
    )

    (
        observed_binary,
        weights,
        known_mask,
    ) = ipcw_dca_weights(
        event,
        time_month,
        horizon_month,
    )

    n_total = float(
        len(
            predicted_risk
        )
    )

    weighted_event_total = float(
        np.sum(
            weights[
                known_mask
            ]
            * observed_binary[
                known_mask
            ]
        )
    )

    weighted_nonevent_total = float(
        np.sum(
            weights[
                known_mask
            ]
            * (
                1
                - observed_binary[
                    known_mask
                ]
            )
        )
    )

    treat_all = np.empty(
        len(
            thresholds
        ),
        dtype=np.float64,
    )

    treat_none = np.zeros(
        len(
            thresholds
        ),
        dtype=np.float64,
    )

    model_nb = np.empty(
        len(
            thresholds
        ),
        dtype=np.float64,
    )

    for index, threshold in enumerate(
        thresholds
    ):
        odds = (
            threshold
            / (
                1.0
                - threshold
            )
        )

        positive = (
            predicted_risk
            >= threshold
        )

        true_positive_weight = float(
            np.sum(
                weights[
                    positive
                    & known_mask
                ]
                * observed_binary[
                    positive
                    & known_mask
                ]
            )
        )

        false_positive_weight = float(
            np.sum(
                weights[
                    positive
                    & known_mask
                ]
                * (
                    1
                    - observed_binary[
                        positive
                        & known_mask
                    ]
                )
            )
        )

        model_nb[
            index
        ] = (
            true_positive_weight
            / n_total
            - (
                false_positive_weight
                / n_total
            )
            * odds
        )

        treat_all[
            index
        ] = (
            weighted_event_total
            / n_total
            - (
                weighted_nonevent_total
                / n_total
            )
            * odds
        )

    return (
        model_nb,
        treat_all,
        treat_none,
    )


dca_rows: list[
    dict[str, Any]
] = []

for landmark_month in (
    PRIMARY_LANDMARK_MONTHS
):
    landmark_position = int(
        np.where(
            np.isclose(
                LANDMARK_MONTHS,
                landmark_month,
            )
        )[0][0]
    )

    rows = np.where(
        landmark_index_long
        == landmark_position
    )[0]

    event_indicator = (
        event_within_60m[
            rows
        ]
    )

    residual_time = (
        analysis_time_month[
            rows
        ]
    )

    for horizon_month in [
        12.0,
        36.0,
        60.0,
    ]:
        (
            threshold_min,
            threshold_max,
            threshold_n,
        ) = (
            DCA_THRESHOLD_RANGES[
                horizon_month
            ]
        )

        thresholds = np.linspace(
            threshold_min,
            threshold_max,
            int(
                threshold_n
            ),
        )

        horizon_position = int(
            np.where(
                np.isclose(
                    FUTURE_END_MONTHS,
                    horizon_month,
                )
            )[0][0]
        )

        reference_all = None
        reference_none = None

        for model_name in MODEL_ORDER:
            predicted_risk = (
                model_risk_calibrated[
                    model_name
                ][
                    rows,
                    horizon_position,
                ]
            )

            (
                model_nb,
                treat_all,
                treat_none,
            ) = survival_net_benefit(
                predicted_risk,
                event_indicator,
                residual_time,
                horizon_month,
                thresholds,
            )

            if reference_all is None:
                reference_all = (
                    treat_all
                )
                reference_none = (
                    treat_none
                )

            for (
                threshold,
                net_benefit,
            ) in zip(
                thresholds,
                model_nb,
            ):
                dca_rows.append(
                    {
                        "model": (
                            model_name
                        ),
                        "landmark_month": float(
                            landmark_month
                        ),
                        "landmark_year": float(
                            landmark_month
                            / 12.0
                        ),
                        "horizon_month": float(
                            horizon_month
                        ),
                        "horizon_year": float(
                            horizon_month
                            / 12.0
                        ),
                        "threshold_probability": float(
                            threshold
                        ),
                        "net_benefit": float(
                            net_benefit
                        ),
                    }
                )

        for (
            threshold,
            all_nb,
            none_nb,
        ) in zip(
            thresholds,
            reference_all,
            reference_none,
        ):
            dca_rows.append(
                {
                    "model": (
                        "Treat all"
                    ),
                    "landmark_month": float(
                        landmark_month
                    ),
                    "landmark_year": float(
                        landmark_month
                        / 12.0
                    ),
                    "horizon_month": float(
                        horizon_month
                    ),
                    "horizon_year": float(
                        horizon_month
                        / 12.0
                    ),
                    "threshold_probability": float(
                        threshold
                    ),
                    "net_benefit": float(
                        all_nb
                    ),
                }
            )

            dca_rows.append(
                {
                    "model": (
                        "Treat none"
                    ),
                    "landmark_month": float(
                        landmark_month
                    ),
                    "landmark_year": float(
                        landmark_month
                        / 12.0
                    ),
                    "horizon_month": float(
                        horizon_month
                    ),
                    "horizon_year": float(
                        horizon_month
                        / 12.0
                    ),
                    "threshold_probability": float(
                        threshold
                    ),
                    "net_benefit": float(
                        none_nb
                    ),
                }
            )

dca_table = pd.DataFrame(
    dca_rows
)

dca_table.to_csv(
    OUTPUT_DIR
    / "four_model_survival_dca.csv",
    index=False,
    encoding="utf-8-sig",
)


# =============================================================================
# 16. 图形
# =============================================================================

# 16.1 主文 0/1/3/5 年 Landmark：5 年 Uno C
primary_c = (
    landmark_bootstrap.loc[
        (
            landmark_bootstrap[
                "metric"
            ]
            == "uno_c_index_5y"
        )
        & (
            landmark_bootstrap[
                "landmark_month"
            ].isin(
                PRIMARY_LANDMARK_MONTHS
            )
        )
    ]
    .copy()
)

x_positions = np.arange(
    len(
        PRIMARY_LANDMARK_MONTHS
    ),
    dtype=float,
)

bar_width = 0.18

fig, ax = plt.subplots(
    figsize=(
        11.0,
        6.2,
    )
)

for (
    model_offset,
    model_name,
) in enumerate(
    MODEL_ORDER
):
    subset = (
        primary_c.loc[
            primary_c[
                "model"
            ]
            == model_name
        ]
        .sort_values(
            "landmark_month"
        )
    )

    values = (
        subset[
            "point_estimate"
        ].to_numpy(
            dtype=float
        )
    )

    lower = (
        subset[
            "lower_95"
        ].to_numpy(
            dtype=float
        )
    )

    upper = (
        subset[
            "upper_95"
        ].to_numpy(
            dtype=float
        )
    )

    positions = (
        x_positions
        + (
            model_offset
            - 1.5
        )
        * bar_width
    )

    ax.bar(
        positions,
        values,
        width=bar_width,
        label=model_name,
        yerr=np.vstack(
            [
                values
                - lower,
                upper
                - values,
            ]
        ),
        capsize=3,
    )

ax.set_xticks(
    x_positions
)

ax.set_xticklabels(
    [
        f"ART year "
        f"{int(month / 12)}"
        for month in (
            PRIMARY_LANDMARK_MONTHS
        )
    ]
)

ax.set_xlabel(
    "Prediction landmark"
)

ax.set_ylabel(
    "Uno C-index for future 5-year CKD risk"
)

ax.set_title(
    "Cross-fitted OOF discrimination"
)

ax.grid(
    axis="y",
    alpha=0.20,
)

ax.legend(
    frameon=False
)

fig.tight_layout()

save_figure(
    fig,
    "figure_1_primary_landmark_5y_uno_c",
)


# 16.2 主文 scope 的模型平均性能
primary_summary = (
    model_summary.loc[
        model_summary[
            "scope"
        ]
        == (
            "primary_0_1_3_5y_"
            "landmark_equal_weight_mean"
        )
    ]
    .copy()
)

for (
    metric_column,
    lower_column,
    upper_column,
    ylabel,
    filename,
) in [
    (
        "mean_uno_c_index_5y",
        "mean_uno_c_index_5y_lower_95",
        "mean_uno_c_index_5y_upper_95",
        "Mean Uno C-index",
        "figure_2_primary_mean_uno_c",
    ),
    (
        "mean_integrated_dynamic_auc",
        "mean_integrated_dynamic_auc_lower_95",
        "mean_integrated_dynamic_auc_upper_95",
        "Mean integrated dynamic AUC",
        "figure_3_primary_mean_iauc",
    ),
    (
        "mean_integrated_brier",
        "mean_integrated_brier_lower_95",
        "mean_integrated_brier_upper_95",
        "Mean integrated Brier score",
        "figure_4_primary_mean_ibs",
    ),
]:
    values = (
        primary_summary[
            metric_column
        ].to_numpy(
            dtype=float
        )
    )

    lower = (
        primary_summary[
            lower_column
        ].to_numpy(
            dtype=float
        )
    )

    upper = (
        primary_summary[
            upper_column
        ].to_numpy(
            dtype=float
        )
    )

    x = np.arange(
        len(
            primary_summary
        )
    )

    fig, ax = plt.subplots(
        figsize=(
            8.5,
            5.5,
        )
    )

    ax.bar(
        x,
        values,
        yerr=np.vstack(
            [
                values
                - lower,
                upper
                - values,
            ]
        ),
        capsize=4,
    )

    ax.set_xticks(
        x
    )

    ax.set_xticklabels(
        primary_summary[
            "model"
        ]
    )

    ax.set_ylabel(
        ylabel
    )

    ax.set_title(
        "Equal-weight mean across ART years 0, 1, 3, and 5"
    )

    ax.grid(
        axis="y",
        alpha=0.20,
    )

    fig.tight_layout()

    save_figure(
        fig,
        filename,
    )


# 16.3 动态 AUC 曲线：每个主要 Landmark 单独一张图
for landmark_month in (
    PRIMARY_LANDMARK_MONTHS
):
    subset = (
        calibrated_horizon_metrics.loc[
            calibrated_horizon_metrics[
                "landmark_month"
            ]
            == landmark_month
        ]
    )

    fig, ax = plt.subplots(
        figsize=(
            8.5,
            5.8,
        )
    )

    for model_name in (
        MODEL_ORDER
    ):
        model_subset = (
            subset.loc[
                subset[
                    "model"
                ]
                == model_name
            ]
            .sort_values(
                "horizon_month"
            )
        )

        ax.plot(
            model_subset[
                "horizon_year"
            ],
            model_subset[
                "dynamic_auc"
            ],
            marker="o",
            label=model_name,
        )

    ax.set_xlabel(
        "Years after landmark"
    )

    ax.set_ylabel(
        "Time-dependent AUC"
    )

    ax.set_title(
        "Dynamic AUC | "
        f"ART year "
        f"{int(landmark_month / 12)}"
    )

    ax.grid(
        alpha=0.20
    )

    ax.legend(
        frameon=False
    )

    fig.tight_layout()

    save_figure(
        fig,
        (
            "figure_dynamic_auc_"
            f"landmark_{int(landmark_month)}m"
        ),
    )


# 16.4 Brier 曲线：每个主要 Landmark 单独一张图
for landmark_month in (
    PRIMARY_LANDMARK_MONTHS
):
    subset = (
        calibrated_horizon_metrics.loc[
            calibrated_horizon_metrics[
                "landmark_month"
            ]
            == landmark_month
        ]
    )

    fig, ax = plt.subplots(
        figsize=(
            8.5,
            5.8,
        )
    )

    for model_name in (
        MODEL_ORDER
    ):
        model_subset = (
            subset.loc[
                subset[
                    "model"
                ]
                == model_name
            ]
            .sort_values(
                "horizon_month"
            )
        )

        ax.plot(
            model_subset[
                "horizon_year"
            ],
            model_subset[
                "brier_score"
            ],
            marker="o",
            label=model_name,
        )

    ax.set_xlabel(
        "Years after landmark"
    )

    ax.set_ylabel(
        "Brier score"
    )

    ax.set_title(
        "Brier score | "
        f"ART year "
        f"{int(landmark_month / 12)}"
    )

    ax.grid(
        alpha=0.20
    )

    ax.legend(
        frameon=False
    )

    fig.tight_layout()

    save_figure(
        fig,
        (
            "figure_brier_"
            f"landmark_{int(landmark_month)}m"
        ),
    )


# 16.5 校准图：1/3/5 年，每个 Landmark 单独输出
for landmark_month in (
    PRIMARY_LANDMARK_MONTHS
):
    for horizon_month in [
        12.0,
        36.0,
        60.0,
    ]:
        subset = (
            decile_calibration.loc[
                (
                    decile_calibration[
                        "landmark_month"
                    ]
                    == landmark_month
                )
                & (
                    decile_calibration[
                        "horizon_month"
                    ]
                    == horizon_month
                )
            ]
        )

        fig, ax = plt.subplots(
            figsize=(
                6.5,
                6.2,
            )
        )

        for model_name in (
            MODEL_ORDER
        ):
            model_subset = (
                subset.loc[
                    subset[
                        "model"
                    ]
                    == model_name
                ]
                .sort_values(
                    "mean_predicted_risk"
                )
            )

            ax.plot(
                model_subset[
                    "mean_predicted_risk"
                ],
                model_subset[
                    "km_observed_risk"
                ],
                marker="o",
                label=model_name,
            )

        axis_limit = float(
            max(
                subset[
                    "mean_predicted_risk"
                ].max(),
                subset[
                    "km_observed_risk"
                ].max(),
                0.01,
            )
            * 1.10
        )

        ax.plot(
            [
                0.0,
                axis_limit,
            ],
            [
                0.0,
                axis_limit,
            ],
            linestyle="--",
            linewidth=1.0,
        )

        ax.set_xlim(
            0.0,
            axis_limit,
        )

        ax.set_ylim(
            0.0,
            axis_limit,
        )

        ax.set_xlabel(
            "Mean predicted risk"
        )

        ax.set_ylabel(
            "Kaplan–Meier observed risk"
        )

        ax.set_title(
            f"Calibration | "
            f"ART year "
            f"{int(landmark_month / 12)} | "
            f"future "
            f"{int(horizon_month / 12)} year"
        )

        ax.grid(
            alpha=0.20
        )

        ax.legend(
            frameon=False
        )

        fig.tight_layout()

        save_figure(
            fig,
            (
                "figure_calibration_"
                f"landmark_{int(landmark_month)}m_"
                f"horizon_{int(horizon_month)}m"
            ),
        )


# 16.6 DCA：1/3/5 年，每个主要 Landmark 单独输出
for landmark_month in (
    PRIMARY_LANDMARK_MONTHS
):
    for horizon_month in [
        12.0,
        36.0,
        60.0,
    ]:
        subset = (
            dca_table.loc[
                (
                    dca_table[
                        "landmark_month"
                    ]
                    == landmark_month
                )
                & (
                    dca_table[
                        "horizon_month"
                    ]
                    == horizon_month
                )
            ]
        )

        fig, ax = plt.subplots(
            figsize=(
                8.5,
                5.8,
            )
        )

        for model_name in (
            MODEL_ORDER
            + [
                "Treat all",
                "Treat none",
            ]
        ):
            model_subset = (
                subset.loc[
                    subset[
                        "model"
                    ]
                    == model_name
                ]
                .sort_values(
                    "threshold_probability"
                )
            )

            if len(
                model_subset
            ) == 0:
                continue

            ax.plot(
                model_subset[
                    "threshold_probability"
                ],
                model_subset[
                    "net_benefit"
                ],
                label=model_name,
            )

        ax.xaxis.set_major_formatter(
            PercentFormatter(
                1.0
            )
        )

        ax.set_xlabel(
            "Threshold probability"
        )

        ax.set_ylabel(
            "Net benefit"
        )

        ax.set_title(
            f"Survival DCA | "
            f"ART year "
            f"{int(landmark_month / 12)} | "
            f"future "
            f"{int(horizon_month / 12)} year"
        )

        ax.grid(
            alpha=0.20
        )

        ax.legend(
            frameon=False
        )

        fig.tight_layout()

        save_figure(
            fig,
            (
                "figure_dca_"
                f"landmark_{int(landmark_month)}m_"
                f"horizon_{int(horizon_month)}m"
            ),
        )


# 16.7 主文 scope：配对差值 forest，每个指标一张
primary_pairwise = (
    pairwise_comparison.loc[
        pairwise_comparison[
            "scope"
        ]
        == (
            "primary_0_1_3_5y_"
            "landmark_equal_weight_mean"
        )
    ]
)

for metric_name in (
    METRIC_NAMES
):
    subset = (
        primary_pairwise.loc[
            primary_pairwise[
                "metric"
            ]
            == metric_name
        ]
        .copy()
    )

    values = (
        subset[
            "difference_a_minus_b"
        ].to_numpy(
            dtype=float
        )
    )

    lower = (
        subset[
            "difference_lower_95"
        ].to_numpy(
            dtype=float
        )
    )

    upper = (
        subset[
            "difference_upper_95"
        ].to_numpy(
            dtype=float
        )
    )

    y = np.arange(
        len(
            subset
        )
    )

    fig, ax = plt.subplots(
        figsize=(
            9.0,
            6.0,
        )
    )

    ax.errorbar(
        values,
        y,
        xerr=np.vstack(
            [
                values
                - lower,
                upper
                - values,
            ]
        ),
        fmt="o",
        capsize=4,
    )

    ax.axvline(
        0.0,
        linestyle="--",
        linewidth=1.0,
    )

    ax.set_yticks(
        y
    )

    ax.set_yticklabels(
        subset[
            "comparison_label"
        ]
    )

    ax.set_xlabel(
        (
            "Difference "
            "(model A - model B)"
        )
    )

    ax.set_title(
        "Patient-level paired bootstrap | "
        + metric_name
    )

    ax.grid(
        axis="x",
        alpha=0.20,
    )

    fig.tight_layout()

    save_figure(
        fig,
        (
            "figure_pairwise_forest_"
            + metric_name
        ),
    )


# =============================================================================
# 17. 结果显示
# =============================================================================

display_table(
    model_summary.round(
        6
    ),
    "四模型跨 Landmark 等权平均性能",
)

display_table(
    pairwise_comparison.loc[
        pairwise_comparison[
            "scope"
        ]
        == (
            "primary_0_1_3_5y_"
            "landmark_equal_weight_mean"
        )
    ].round(
        6
    ),
    "主文 0/1/3/5 年 Landmark："
    "患者级配对 Bootstrap 模型差值",
)

display_table(
    calibrated_landmark_metrics.loc[
        calibrated_landmark_metrics[
            "landmark_month"
        ].isin(
            PRIMARY_LANDMARK_MONTHS
        )
    ].round(
        6
    ),
    "0、1、3、5 年 Landmark "
    "cross-fitted calibrated performance",
)


# =============================================================================
# 18. 保存元数据和最终摘要
# =============================================================================

input_fingerprint = hashlib.sha256()

for path in required_files:
    stat = path.stat()

    input_fingerprint.update(
        str(
            path
        ).encode(
            "utf-8"
        )
    )

    input_fingerprint.update(
        str(
            stat.st_size
        ).encode(
            "utf-8"
        )
    )

    input_fingerprint.update(
        str(
            stat.st_mtime_ns
        ).encode(
            "utf-8"
        )
    )

metadata = {
    "step": "11_v2",
    "analysis_name": (
        "Four-model cross-fitted OOF calibration "
        "and paired comparison with LSTM-v2"
    ),
    "models": (
        MODEL_ORDER
    ),
    "pairwise_comparisons": (
        PAIRWISE_COMPARISONS
    ),
    "project_dir": str(
        PROJECT_DIR
    ),
    "step6_dir": str(
        STEP6_DIR
    ),
    "cox_dir": str(
        COX_DIR
    ),
    "rsf_dir": str(
        RSF_DIR
    ),
    "rnn_dir": str(
        RNN_DIR
    ),
    "lstm_v2_dir": str(
        LSTM_V2_DIR
    ),
    "output_dir": str(
        OUTPUT_DIR
    ),
    "bootstrap_reps": int(
        BOOTSTRAP_REPS
    ),
    "bootstrap_unit": (
        "development patient"
    ),
    "bootstrap_pairing": (
        "same sampled development patients "
        "used for all models and all landmarks"
    ),
    "bootstrap_seed": int(
        RANDOM_SEED
    ),
    "calibration_method": (
        "five-fold cross-fitted discrete hazard "
        "recalibration with interval-specific "
        "intercepts and one common slope per "
        "model and landmark"
    ),
    "calibration_ridge": float(
        CALIBRATION_RIDGE
    ),
    "landmark_months": (
        LANDMARK_MONTHS.tolist()
    ),
    "future_end_months": (
        FUTURE_END_MONTHS.tolist()
    ),
    "metric_times": (
        METRIC_TIMES.tolist()
    ),
    "summary_scopes": {
        key: value.tolist()
        for (
            key,
            value,
        ) in SUMMARY_SCOPES.items()
    },
    "lstm_v2_selected_trial": (
        lstm_v2_summary.get(
            "selected_trial_number"
        )
    ),
    "lstm_v2_seed_ensemble_n_per_fold": (
        lstm_v2_summary.get(
            "seed_ensemble_n_per_fold"
        )
    ),
    "input_fingerprint_sha256": (
        input_fingerprint.hexdigest()
    ),
    "locked_test_used": False,
    "warning": (
        "This is development-cohort OOF performance. "
        "It must not be interpreted as locked internal-test "
        "or geographic external-validation performance."
    ),
}

save_json(
    metadata,
    OUTPUT_DIR
    / "step11_v2_metadata.json",
)

primary_summary_records = (
    model_summary.loc[
        model_summary[
            "scope"
        ]
        == (
            "primary_0_1_3_5y_"
            "landmark_equal_weight_mean"
        )
    ]
    .to_dict(
        orient="records"
    )
)

primary_pairwise_records = (
    pairwise_comparison.loc[
        pairwise_comparison[
            "scope"
        ]
        == (
            "primary_0_1_3_5y_"
            "landmark_equal_weight_mean"
        )
    ]
    .to_dict(
        orient="records"
    )
)

summary = {
    "analysis": (
        "development_oof_four_model_comparison"
    ),
    "models": (
        MODEL_ORDER
    ),
    "primary_scope": (
        "ART years 0, 1, 3, 5 equal-weight mean"
    ),
    "primary_model_performance": (
        primary_summary_records
    ),
    "primary_pairwise_comparison": (
        primary_pairwise_records
    ),
    "bootstrap_reps": int(
        BOOTSTRAP_REPS
    ),
    "locked_test_used": False,
    "output_dir": str(
        OUTPUT_DIR
    ),
}

save_json(
    summary,
    OUTPUT_DIR
    / "step11_v2_summary.json",
)


# =============================================================================
# 19. 完成
# =============================================================================

print(
    "\n"
    + "=" * 110
)

print(
    "Step 11 v2 完成："
    "Cox / RSF / RNN / LSTM-v2 "
    "开发集 OOF 统一比较"
)

print(
    "=" * 110
)

print(
    "模型：",
    MODEL_ORDER,
)

print(
    "Bootstrap：",
    BOOTSTRAP_REPS,
)

print(
    "LSTM-v2 Trial：",
    lstm_v2_summary.get(
        "selected_trial_number"
    ),
)

print(
    "锁定测试集：未读取"
)

print(
    "输出目录：",
    OUTPUT_DIR,
)

print(
    "\n最重要结果文件："
)

for filename in [
    "four_model_equal_weight_mean_performance.csv",
    "four_model_paired_bootstrap_comparison.csv",
    "four_model_crossfit_calibrated_landmark_metrics.csv",
    "four_model_crossfit_calibrated_horizon_metrics.csv",
    "four_model_crossfit_calibrated_1y_3y_5y_metrics.csv",
    "four_model_calibration_overall.csv",
    "four_model_calibration_deciles.csv",
    "four_model_survival_dca.csv",
    "step11_v2_summary.json",
]:
    print(
        " -",
        OUTPUT_DIR
        / filename,
    )

print(
    "=" * 110
)
