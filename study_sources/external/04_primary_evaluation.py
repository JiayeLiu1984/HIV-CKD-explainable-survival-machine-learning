# -*- coding: utf-8 -*-

# =============================================================================
# 重庆独立地理外部验证
# Step8 FINAL FIXED
#
# 修正：
#   scikit-survival要求评价时间必须位于外部测试数据随访范围内。
#
# 例如：
#   重庆某Landmark最短随访 = 6.18月
#   固定评价时间 = 6.00月
#
# 则：
#   6月 dynamic AUC：该时点无case，统计上不可估计 → NA
#   iAUC：从第一个支持的评价时间继续计算
#   6月 Brier：所有人均明确event-free且uncensored，
#              精确计算 mean(risk_6m^2)
#   IBS：仍在完整6–59.999月网格上积分
#
# 最终：
#   - Uno C：5年
#   - iAUC：0.5–5年支持网格
#   - IBS：0.5–5年完整网格
#   - 1/3/5年 AUC
#   - 1/3/5年 Brier
#   - 1/3/5年 KM observed risk vs predicted risk
#   - 患者级1000次bootstrap 95%CI
#   - 主文Landmark：0、1、3、5年等权平均
#   - 补充：全部0–5年六个Landmark等权平均
#
# IPCW censoring reference：
#   深圳+南宁开发队列，对应Landmark风险集
#
# 本步骤不：
#   - 训练模型
#   - 修改预测
#   - 使用重庆结果重新校准
#   - 使用重庆结果选择模型
# =============================================================================


from __future__ import annotations

from pathlib import Path
import json
import os
import time
import warnings

import numpy as np
import pandas as pd

from sksurv.metrics import (
    brier_score,
    concordance_index_ipcw,
    cumulative_dynamic_auc,
)

from sksurv.nonparametric import (
    kaplan_meier_estimator,
)

from sksurv.util import Surv


# =============================================================================
# 1. 路径
# =============================================================================

BASE = Path(
    "__CKD_WORKDIR__"
)

STEP1 = (
    BASE
    / "rolling_5y_step1_new_split"
)

STEP6 = (
    BASE
    / "重庆外部验证_step6_LSTM_v2"
)

STEP7R = (
    BASE
    / "重庆外部验证_step7R_FINAL_fold_specific"
)

OUT = (
    BASE
    / "重庆外部验证_step8_FINAL_performance_FIXED"
)

BOOTSTRAP_DIR = (
    OUT
    / "bootstrap"
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)

BOOTSTRAP_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# =============================================================================
# 2. 锁定评价参数
# =============================================================================

LANDMARK_MONTHS = np.asarray(
    [0, 12, 24, 36, 48, 60],
    dtype=np.float64,
)

LANDMARK_YEARS = (
    LANDMARK_MONTHS
    / 12.0
)

PRIMARY_LANDMARK_POSITIONS = np.asarray(
    [0, 1, 3, 5],
    dtype=np.int64,
)

PRIMARY_LANDMARK_MONTHS = (
    LANDMARK_MONTHS[
        PRIMARY_LANDMARK_POSITIONS
    ]
)

# -------------------------------------------------------------------------
# 模型未来风险列
# -------------------------------------------------------------------------

FUTURE_END_MONTHS = np.asarray(
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
        60,
    ],
    dtype=np.float64,
)

# -------------------------------------------------------------------------
# 正式数值评价时间
#
# 第10列仍然代表60月风险；
# 数值评价使用59.999避免sksurv右边界问题。
# -------------------------------------------------------------------------

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

REPORT_HORIZONS = {
    12: 1,
    36: 5,
    60: 9,
}

N_LANDMARK = 6
N_FUTURE = 10

EPS = 1e-7

# 开发reference行政截断稍长于外部测试
REFERENCE_ADMIN_CAP = 61.0

# 外部分析上限
EXTERNAL_ADMIN_CAP = 60.0


# =============================================================================
# 3. Bootstrap
# =============================================================================

BOOTSTRAP_REPS = int(
    os.getenv(
        "CKD_EXTERNAL_BOOTSTRAP_REPS",
        "1000",
    )
)

RANDOM_SEED = int(
    os.getenv(
        "CKD_EXTERNAL_BOOTSTRAP_SEED",
        "20260822",
    )
)

MIN_VALID_BOOTSTRAP_RATE = 0.80


# =============================================================================
# 4. 文件检查
# =============================================================================

required_files = [
    STEP7R
    / "external_calibrated_risk_long.npy",

    STEP7R
    / "external_raw_risk_long.npy",

    STEP6
    / "super_landmark_external_metadata.csv",

    STEP6
    / "prediction_origin_mask.npy",

    STEP6
    / "external_long_row_index_map.npy",

    STEP1
    / "development_idx.npy",

    STEP1
    / "event.npy",

    STEP1
    / "observed_time_month.npy",
]

missing_files = [
    str(path)
    for path in required_files
    if not path.exists()
]

if missing_files:
    raise FileNotFoundError(
        "缺少以下必要文件：\n"
        + "\n".join(missing_files)
    )


# =============================================================================
# 5. 读取重庆最终预测
# =============================================================================

risk_long = np.load(
    STEP7R
    / "external_calibrated_risk_long.npy"
).astype(
    np.float64
)

raw_risk_long = np.load(
    STEP7R
    / "external_raw_risk_long.npy"
).astype(
    np.float64
)

metadata = pd.read_csv(
    STEP6
    / "super_landmark_external_metadata.csv",
    encoding="utf-8-sig",
    dtype={
        "ID": "string",
    },
)

prediction_origin_mask = np.load(
    STEP6
    / "prediction_origin_mask.npy"
).astype(
    bool
)

long_row_index_map = np.load(
    STEP6
    / "external_long_row_index_map.npy"
).astype(
    np.int32
)


# =============================================================================
# 6. 重庆结构检查
# =============================================================================

N_EXTERNAL = (
    prediction_origin_mask.shape[0]
)

if prediction_origin_mask.shape != (
    N_EXTERNAL,
    N_LANDMARK,
):
    raise ValueError(
        "prediction_origin_mask形状错误。"
    )

if long_row_index_map.shape != (
    N_EXTERNAL,
    N_LANDMARK,
):
    raise ValueError(
        "external_long_row_index_map形状错误。"
    )

LONG_N = int(
    prediction_origin_mask.sum()
)

if risk_long.shape != (
    LONG_N,
    N_FUTURE,
):
    raise ValueError(
        f"risk_long形状异常：{risk_long.shape}"
    )

if raw_risk_long.shape != risk_long.shape:
    raise ValueError(
        "raw/calibrated risk形状不一致。"
    )

if len(metadata) != LONG_N:
    raise ValueError(
        "metadata行数与预测行数不一致。"
    )

if not np.isfinite(
    risk_long
).all():
    raise ValueError(
        "最终预测存在NaN/Inf。"
    )

if np.any(
    (risk_long < 0.0)
    | (risk_long > 1.0)
):
    raise ValueError(
        "预测风险超出[0,1]。"
    )

if np.any(
    np.diff(
        risk_long,
        axis=1,
    )
    < -1e-8
):
    raise ValueError(
        "累计风险不是单调递增。"
    )


# =============================================================================
# 7. metadata
# =============================================================================

required_metadata_columns = [
    "local_patient_index",
    "ID",
    "landmark_index",
    "landmark_month",
    "remaining_time_month",
    "analysis_time_month",
    "event_within_60m",
]

missing_metadata_columns = [
    column
    for column in required_metadata_columns
    if column not in metadata.columns
]

if missing_metadata_columns:
    raise ValueError(
        "metadata缺少字段："
        f"{missing_metadata_columns}"
    )

external_patient_index_long = (
    metadata[
        "local_patient_index"
    ]
    .to_numpy(
        dtype=np.int32
    )
)

external_landmark_index_long = (
    metadata[
        "landmark_index"
    ]
    .to_numpy(
        dtype=np.int8
    )
)

external_remaining_time = (
    metadata[
        "remaining_time_month"
    ]
    .to_numpy(
        dtype=np.float64
    )
)

external_analysis_time = (
    metadata[
        "analysis_time_month"
    ]
    .to_numpy(
        dtype=np.float64
    )
)

external_event_60 = (
    metadata[
        "event_within_60m"
    ]
    .to_numpy(
        dtype=np.int8
    )
)

if np.any(
    external_remaining_time <= 0
):
    raise ValueError(
        "存在Landmark后remaining_time<=0。"
    )

if np.any(
    external_analysis_time <= 0
):
    raise ValueError(
        "存在Landmark后analysis_time<=0。"
    )

if np.any(
    external_analysis_time
    > EXTERNAL_ADMIN_CAP
    + EPS
):
    raise ValueError(
        "外部analysis_time超过60月。"
    )

if not np.isin(
    external_event_60,
    [0, 1],
).all():
    raise ValueError(
        "external_event_60不是0/1。"
    )


# =============================================================================
# 8. 长格式映射核对
# =============================================================================

expected_pairs = np.argwhere(
    prediction_origin_mask
)

metadata_pairs = np.column_stack(
    [
        external_patient_index_long,
        external_landmark_index_long,
    ]
)

if not np.array_equal(
    expected_pairs.astype(
        np.int32
    ),
    metadata_pairs.astype(
        np.int32
    ),
):
    raise ValueError(
        "Step6长格式映射顺序不一致。"
    )


# =============================================================================
# 9. 读取深圳+南宁开发队列
# =============================================================================

development_idx = np.load(
    STEP1
    / "development_idx.npy"
).astype(
    np.int64
)

development_event_all = np.load(
    STEP1
    / "event.npy"
).astype(
    np.int8
)

development_time_all = np.load(
    STEP1
    / "observed_time_month.npy"
).astype(
    np.float64
)

development_event = (
    development_event_all[
        development_idx
    ]
)

development_time = (
    development_time_all[
        development_idx
    ]
)

if len(development_idx) != 22337:
    warnings.warn(
        "开发患者数不是预期22337，"
        f"当前={len(development_idx)}。"
    )

if not np.isin(
    development_event,
    [0, 1],
).all():
    raise ValueError(
        "开发队列event异常。"
    )

if not np.isfinite(
    development_time
).all():
    raise ValueError(
        "开发队列随访时间存在NaN/Inf。"
    )


# =============================================================================
# 10. Survival array
# =============================================================================

def build_survival_array(
    event,
    time_month,
):
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
            "event/time形状不一致。"
        )

    if not np.isfinite(
        time_month
    ).all():
        raise ValueError(
            "生存时间存在NaN/Inf。"
        )

    if np.any(
        time_month <= 0
    ):
        raise ValueError(
            "生存时间必须>0。"
        )

    return Surv.from_arrays(
        event=event,
        time=time_month,
    )


# =============================================================================
# 11. 开发队列Landmark IPCW reference
#
# 关键：
# 开发reference行政截断至61月，
# 外部测试行政截断至60月。
#
# 这样保证训练censoring distribution覆盖测试范围。
# =============================================================================

development_survival_reference = []

development_reference_rows = []

for (
    landmark_index,
    landmark_month
) in enumerate(
    LANDMARK_MONTHS
):

    eligible = (
        development_time
        > landmark_month
        + EPS
    )

    original_event = (
        development_event[
            eligible
        ]
    )

    residual_time = (
        development_time[
            eligible
        ]
        - landmark_month
    )

    event_within_60 = (
        (original_event == 1)
        & (
            residual_time
            <= 60.0
            + EPS
        )
    )

    # ----------------------------------------------------------
    # censoring reference稍延长到61个月
    # ----------------------------------------------------------

    reference_time = np.minimum(
        residual_time,
        REFERENCE_ADMIN_CAP,
    )

    y_train = build_survival_array(
        event_within_60,
        reference_time,
    )

    development_survival_reference.append(
        y_train
    )

    development_reference_rows.append(
        {
            "landmark_index":
                landmark_index,

            "landmark_month":
                landmark_month,

            "landmark_year":
                landmark_month
                / 12.0,

            "development_reference_n":
                int(
                    eligible.sum()
                ),

            "development_event_within_60m_n":
                int(
                    event_within_60.sum()
                ),

            "reference_min_time":
                float(
                    reference_time.min()
                ),

            "reference_max_time":
                float(
                    reference_time.max()
                ),
        }
    )


development_reference_summary = pd.DataFrame(
    development_reference_rows
)

development_reference_summary.to_csv(
    OUT
    / "00_development_IPCW_reference.csv",
    index=False,
    encoding="utf-8-sig",
)


# =============================================================================
# 12. 数值积分
# =============================================================================

def trapezoid_integral(
    y,
    x,
):
    if hasattr(
        np,
        "trapezoid",
    ):
        return float(
            np.trapezoid(
                y,
                x,
            )
        )

    return float(
        np.trapz(
            y,
            x,
        )
    )


# =============================================================================
# 13. 核心评价函数
#
# 这是本次修复的关键。
#
# AUC:
#   sksurv仅在test follow-up支持的时间点计算。
#
# 若6个月 < min(test follow-up)：
#   说明6个月以前没有任何event/censor，
#   因而6月AUC无case，定义上不可估计 → NaN。
#
# iAUC:
#   使用可评价时间点。
#   被删除的早期时间点没有event，因此不承担事件权重。
#
# Brier:
#   若t < min(test follow-up)，
#   所有人在t时均明确event-free且uncensored。
#
#   Actual survival(t)=1
#   Pred survival(t)=1-risk(t)
#
#   Brier(t)=mean([1-(1-risk)]^2)
#           =mean(risk^2)
#
# 因此IBS仍可严格在完整6–59.999月区间计算。
# =============================================================================

def evaluate_survival_predictions(
    survival_train,
    event_test,
    time_test,
    risk_matrix,
    require_primary_metrics=False,
):
    event_test = np.asarray(
        event_test,
        dtype=np.int8,
    )

    time_test = np.asarray(
        time_test,
        dtype=np.float64,
    )

    risk_matrix = np.asarray(
        risk_matrix,
        dtype=np.float64,
    )

    if risk_matrix.ndim != 2:
        raise ValueError(
            "risk_matrix必须为二维。"
        )

    if risk_matrix.shape[1] != N_FUTURE:
        raise ValueError(
            "risk_matrix未来区间数不是10。"
        )

    if len(event_test) != len(time_test):
        raise ValueError(
            "event/time长度不一致。"
        )

    if len(event_test) != risk_matrix.shape[0]:
        raise ValueError(
            "event/risk行数不一致。"
        )

    if not np.isfinite(
        time_test
    ).all():
        raise ValueError(
            "test time存在NaN/Inf。"
        )

    if np.any(
        time_test <= 0
    ):
        raise ValueError(
            "test time必须>0。"
        )

    if not np.isfinite(
        risk_matrix
    ).all():
        raise ValueError(
            "risk matrix存在NaN/Inf。"
        )

    y_test = build_survival_array(
        event_test,
        time_test,
    )

    survival_matrix = (
        1.0
        - risk_matrix
    )

    test_min = float(
        time_test.min()
    )

    test_max = float(
        time_test.max()
    )

    # -------------------------------------------------------------------------
    # 检查是否存在右端无法评价的问题
    #
    # 右端不允许动态删点，因为我们要求评价到5年。
    # -------------------------------------------------------------------------

    late_invalid = (
        METRIC_TIMES
        >= test_max
    )

    if late_invalid.any():
        raise ValueError(
            "该样本不足以支持5年评价："
            f"test_max={test_max:.6f}, "
            f"最大评价时间={METRIC_TIMES[-1]:.6f}"
        )

    # -------------------------------------------------------------------------
    # AUC支持时间点
    #
    # 左边允许剔除：
    # 例如6 < 6.18。
    # -------------------------------------------------------------------------

    auc_supported = (
        (METRIC_TIMES >= test_min)
        & (METRIC_TIMES < test_max)
    )

    auc_positions = np.where(
        auc_supported
    )[0]

    auc_full = np.full(
        N_FUTURE,
        np.nan,
        dtype=np.float64,
    )

    iauc = np.nan

    if len(auc_positions) > 0:
        try:
            auc_values, iauc_value = (
                cumulative_dynamic_auc(
                    survival_train,
                    y_test,
                    risk_matrix[
                        :,
                        auc_positions
                    ],
                    METRIC_TIMES[
                        auc_positions
                    ],
                )
            )

            auc_full[
                auc_positions
            ] = np.asarray(
                auc_values,
                dtype=np.float64,
            )

            iauc = float(
                iauc_value
            )

        except (
            ValueError,
            ArithmeticError,
            ZeroDivisionError,
            FloatingPointError,
        ):
            iauc = np.nan

    # -------------------------------------------------------------------------
    # Brier：
    # 左侧unsupported时间点可精确手算。
    # -------------------------------------------------------------------------

    brier_full = np.full(
        N_FUTURE,
        np.nan,
        dtype=np.float64,
    )

    early_positions = np.where(
        METRIC_TIMES
        < test_min
    )[0]

    valid_brier_positions = np.where(
        (METRIC_TIMES >= test_min)
        & (METRIC_TIMES < test_max)
    )[0]

    # -------------------------------------------------------------------------
    # t < min(test_time)
    #
    # 没有任何事件/删失发生：
    # Brier = mean(risk^2)
    # -------------------------------------------------------------------------

    for position in early_positions:
        brier_full[
            position
        ] = float(
            np.mean(
                risk_matrix[
                    :,
                    position
                ]
                ** 2
            )
        )

    # -------------------------------------------------------------------------
    # 正常IPCW Brier
    # -------------------------------------------------------------------------

    if len(
        valid_brier_positions
    ) > 0:

        _, brier_values = brier_score(
            survival_train,
            y_test,
            survival_matrix[
                :,
                valid_brier_positions
            ],
            METRIC_TIMES[
                valid_brier_positions
            ],
        )

        brier_full[
            valid_brier_positions
        ] = np.asarray(
            brier_values,
            dtype=np.float64,
        )

    if not np.isfinite(
        brier_full
    ).all():
        raise ValueError(
            "Brier完整时间网格仍存在NaN。"
        )

    # -------------------------------------------------------------------------
    # IBS：
    # 完整6–59.999个月
    # -------------------------------------------------------------------------

    ibs = (
        trapezoid_integral(
            brier_full,
            METRIC_TIMES,
        )
        /
        (
            METRIC_TIMES[-1]
            - METRIC_TIMES[0]
        )
    )

    # -------------------------------------------------------------------------
    # Uno C：
    # 使用未来5年风险
    # -------------------------------------------------------------------------

    try:
        uno_c = float(
            concordance_index_ipcw(
                survival_train,
                y_test,
                risk_matrix[
                    :,
                    -1
                ],
                tau=float(
                    METRIC_TIMES[-1]
                ),
            )[0]
        )

    except (
        ValueError,
        ArithmeticError,
        ZeroDivisionError,
        FloatingPointError,
    ):
        uno_c = np.nan

    result = {
        "uno_c":
            uno_c,

        "iauc":
            iauc,

        "ibs":
            float(
                ibs
            ),

        "auc":
            auc_full,

        "brier":
            brier_full,

        "test_min":
            test_min,

        "test_max":
            test_max,

        "auc_supported":
            auc_supported,
    }

    # -------------------------------------------------------------------------
    # 正式点估计必须支持：
    # 5年C
    # iAUC
    # IBS
    # 1/3/5年AUC和Brier
    # -------------------------------------------------------------------------

    if require_primary_metrics:

        required_values = [
            result[
                "uno_c"
            ],
            result[
                "iauc"
            ],
            result[
                "ibs"
            ],
            result[
                "auc"
            ][1],
            result[
                "auc"
            ][5],
            result[
                "auc"
            ][9],
            result[
                "brier"
            ][1],
            result[
                "brier"
            ][5],
            result[
                "brier"
            ][9],
        ]

        if not np.isfinite(
            required_values
        ).all():
            raise ValueError(
                "正式1/3/5年或综合性能仍无法估计。"
            )

    return result


# =============================================================================
# 14. KM observed risk
# =============================================================================

def km_event_risk(
    event,
    time_month,
    horizon,
):
    event = np.asarray(
        event,
        dtype=bool,
    )

    time_month = np.asarray(
        time_month,
        dtype=np.float64,
    )

    if len(
        time_month
    ) == 0:
        return np.nan

    km_time, km_survival = (
        kaplan_meier_estimator(
            event,
            time_month,
        )
    )

    position = (
        np.searchsorted(
            km_time,
            float(
                horizon
            ),
            side="right",
        )
        - 1
    )

    if position < 0:
        return 0.0

    return float(
        1.0
        - km_survival[
            position
        ]
    )


# =============================================================================
# 15. Landmark点估计
# =============================================================================

landmark_rows = []
horizon_rows = []
support_rows = []

for (
    landmark_index,
    landmark_month
) in enumerate(
    LANDMARK_MONTHS
):

    rows = np.where(
        external_landmark_index_long
        == landmark_index
    )[0]

    if len(rows) == 0:
        raise ValueError(
            f"Landmark {landmark_month}月没有有效外部记录。"
        )

    y_train = (
        development_survival_reference[
            landmark_index
        ]
    )

    event_test = (
        external_event_60[
            rows
        ]
    )

    time_test = (
        external_analysis_time[
            rows
        ]
    )

    risk = (
        risk_long[
            rows,
            :
        ]
    )

    metrics = (
        evaluate_survival_predictions(
            y_train,
            event_test,
            time_test,
            risk,
            require_primary_metrics=True,
        )
    )

    # -------------------------------------------------------------------------
    # support audit
    # -------------------------------------------------------------------------

    supported_times = (
        METRIC_TIMES[
            metrics[
                "auc_supported"
            ]
        ]
    )

    unsupported_times = (
        METRIC_TIMES[
            ~metrics[
                "auc_supported"
            ]
        ]
    )

    support_rows.append(
        {
            "landmark_index":
                landmark_index,

            "landmark_month":
                landmark_month,

            "landmark_year":
                landmark_month
                / 12.0,

            "external_n":
                len(rows),

            "external_min_followup_month":
                metrics[
                    "test_min"
                ],

            "external_max_followup_month":
                metrics[
                    "test_max"
                ],

            "auc_first_supported_month":
                float(
                    supported_times[0]
                ),

            "auc_last_supported_month":
                float(
                    supported_times[-1]
                ),

            "auc_supported_times":
                ",".join(
                    str(x)
                    for x
                    in supported_times
                ),

            "auc_unsupported_early_times":
                ",".join(
                    str(x)
                    for x
                    in unsupported_times[
                        unsupported_times
                        < metrics[
                            "test_min"
                        ]
                    ]
                ),
        }
    )

    # -------------------------------------------------------------------------
    # Landmark综合指标
    # -------------------------------------------------------------------------

    landmark_rows.append(
        {
            "landmark_index":
                landmark_index,

            "landmark_month":
                landmark_month,

            "landmark_year":
                landmark_month
                / 12.0,

            "risk_set_n":
                int(
                    len(rows)
                ),

            "event_within_5y_n":
                int(
                    event_test.sum()
                ),

            "uno_c_index_5y":
                metrics[
                    "uno_c"
                ],

            "integrated_dynamic_auc":
                metrics[
                    "iauc"
                ],

            "integrated_brier_score":
                metrics[
                    "ibs"
                ],
        }
    )

    # -------------------------------------------------------------------------
    # 每个半年时间点
    # -------------------------------------------------------------------------

    for future_index in range(
        N_FUTURE
    ):

        nominal_horizon = (
            FUTURE_END_MONTHS[
                future_index
            ]
        )

        metric_time = (
            METRIC_TIMES[
                future_index
            ]
        )

        predicted_risk = (
            risk[
                :,
                future_index
            ]
        )

        observed_risk = (
            km_event_risk(
                event_test,
                time_test,
                nominal_horizon,
            )
        )

        horizon_rows.append(
            {
                "landmark_index":
                    landmark_index,

                "landmark_month":
                    landmark_month,

                "landmark_year":
                    landmark_month
                    / 12.0,

                "horizon_position":
                    future_index,

                "nominal_horizon_month":
                    nominal_horizon,

                "metric_time_month":
                    metric_time,

                "horizon_year":
                    nominal_horizon
                    / 12.0,

                "risk_set_n":
                    int(
                        len(rows)
                    ),

                "dynamic_auc":
                    float(
                        metrics[
                            "auc"
                        ][
                            future_index
                        ]
                    )
                    if np.isfinite(
                        metrics[
                            "auc"
                        ][
                            future_index
                        ]
                    )
                    else np.nan,

                "auc_estimable":
                    bool(
                        np.isfinite(
                            metrics[
                                "auc"
                            ][
                                future_index
                            ]
                        )
                    ),

                "brier_score":
                    float(
                        metrics[
                            "brier"
                        ][
                            future_index
                        ]
                    ),

                "mean_predicted_risk":
                    float(
                        predicted_risk.mean()
                    ),

                "median_predicted_risk":
                    float(
                        np.median(
                            predicted_risk
                        )
                    ),

                "km_observed_risk":
                    observed_risk,

                "calibration_difference_pred_minus_obs":
                    float(
                        predicted_risk.mean()
                        - observed_risk
                    ),
            }
        )


landmark_point = pd.DataFrame(
    landmark_rows
)

horizon_point = pd.DataFrame(
    horizon_rows
)

support_audit = pd.DataFrame(
    support_rows
)

support_audit.to_csv(
    OUT
    / "00_metric_time_support_audit.csv",
    index=False,
    encoding="utf-8-sig",
)

landmark_point.to_csv(
    OUT
    / "01_landmark_point_performance.csv",
    index=False,
    encoding="utf-8-sig",
)

horizon_point.to_csv(
    OUT
    / "02_horizon_point_performance.csv",
    index=False,
    encoding="utf-8-sig",
)


# =============================================================================
# 16. 1/3/5年点估计
# =============================================================================

selected_horizon_point = (
    horizon_point.loc[
        horizon_point[
            "nominal_horizon_month"
        ].isin(
            [
                12.0,
                36.0,
                60.0,
            ]
        )
    ]
    .copy()
)

if selected_horizon_point[
    "dynamic_auc"
].isna().any():
    raise ValueError(
        "正式1/3/5年AUC存在不可估计值。"
    )

selected_horizon_point.to_csv(
    OUT
    / "03_1y_3y_5y_point_performance.csv",
    index=False,
    encoding="utf-8-sig",
)


# =============================================================================
# 17. 打印时间支持审计
#
# 在正式跑1000 bootstrap前先让你看到：
# 为什么原代码报错。
# =============================================================================

print()
print("=" * 110)
print("外部评价时间支持审计")
print("=" * 110)

print(
    support_audit[
        [
            "landmark_year",
            "external_n",
            "external_min_followup_month",
            "external_max_followup_month",
            "auc_first_supported_month",
            "auc_last_supported_month",
            "auc_unsupported_early_times",
        ]
    ]
    .round(4)
    .to_string(
        index=False
    )
)

print("=" * 110)


# =============================================================================
# 18. Bootstrap arrays
#
# 患者级抽样：
# 每次从2677患者中有放回抽2677次。
#
# 同一bootstrap患者样本用于全部Landmark。
# =============================================================================

boot_uno = np.full(
    (
        BOOTSTRAP_REPS,
        N_LANDMARK,
    ),
    np.nan,
    dtype=np.float64,
)

boot_iauc = np.full_like(
    boot_uno,
    np.nan,
)

boot_ibs = np.full_like(
    boot_uno,
    np.nan,
)

boot_auc = np.full(
    (
        BOOTSTRAP_REPS,
        N_LANDMARK,
        N_FUTURE,
    ),
    np.nan,
    dtype=np.float64,
)

boot_brier = np.full_like(
    boot_auc,
    np.nan,
)

boot_predicted_mean = np.full_like(
    boot_auc,
    np.nan,
)

boot_observed_km = np.full_like(
    boot_auc,
    np.nan,
)

boot_calibration_difference = np.full_like(
    boot_auc,
    np.nan,
)


# =============================================================================
# 19. Bootstrap
# =============================================================================

rng = np.random.default_rng(
    RANDOM_SEED
)

start_time = time.time()

for bootstrap_index in range(
    BOOTSTRAP_REPS
):

    sampled_patients = rng.integers(
        0,
        N_EXTERNAL,
        size=N_EXTERNAL,
    )

    for landmark_index in range(
        N_LANDMARK
    ):

        sampled_rows = (
            long_row_index_map[
                sampled_patients,
                landmark_index,
            ]
        )

        sampled_rows = (
            sampled_rows[
                sampled_rows >= 0
            ]
            .astype(
                np.int64
            )
        )

        if len(
            sampled_rows
        ) < 20:
            continue

        sampled_event = (
            external_event_60[
                sampled_rows
            ]
        )

        sampled_time = (
            external_analysis_time[
                sampled_rows
            ]
        )

        sampled_risk = (
            risk_long[
                sampled_rows,
                :
            ]
        )

        try:
            metric = (
                evaluate_survival_predictions(
                    development_survival_reference[
                        landmark_index
                    ],
                    sampled_event,
                    sampled_time,
                    sampled_risk,
                    require_primary_metrics=False,
                )
            )

        except (
            ValueError,
            ArithmeticError,
            ZeroDivisionError,
            FloatingPointError,
        ):
            continue

        # ---------------------------------------------------------------------
        # 综合指标分别保存；
        # 某一个指标失败不会污染其他指标。
        # ---------------------------------------------------------------------

        if np.isfinite(
            metric[
                "uno_c"
            ]
        ):
            boot_uno[
                bootstrap_index,
                landmark_index,
            ] = metric[
                "uno_c"
            ]

        if np.isfinite(
            metric[
                "iauc"
            ]
        ):
            boot_iauc[
                bootstrap_index,
                landmark_index,
            ] = metric[
                "iauc"
            ]

        if np.isfinite(
            metric[
                "ibs"
            ]
        ):
            boot_ibs[
                bootstrap_index,
                landmark_index,
            ] = metric[
                "ibs"
            ]

        boot_auc[
            bootstrap_index,
            landmark_index,
            :,
        ] = metric[
            "auc"
        ]

        boot_brier[
            bootstrap_index,
            landmark_index,
            :,
        ] = metric[
            "brier"
        ]

        # ---------------------------------------------------------------------
        # Calibration bootstrap
        # ---------------------------------------------------------------------

        for future_index in range(
            N_FUTURE
        ):

            nominal_horizon = (
                FUTURE_END_MONTHS[
                    future_index
                ]
            )

            predicted_mean = float(
                sampled_risk[
                    :,
                    future_index
                ].mean()
            )

            observed = km_event_risk(
                sampled_event,
                sampled_time,
                nominal_horizon,
            )

            boot_predicted_mean[
                bootstrap_index,
                landmark_index,
                future_index,
            ] = predicted_mean

            boot_observed_km[
                bootstrap_index,
                landmark_index,
                future_index,
            ] = observed

            boot_calibration_difference[
                bootstrap_index,
                landmark_index,
                future_index,
            ] = (
                predicted_mean
                - observed
            )

    if (
        bootstrap_index + 1
    ) % 50 == 0:

        elapsed = (
            time.time()
            - start_time
        )

        print(
            f"Bootstrap "
            f"{bootstrap_index + 1}/"
            f"{BOOTSTRAP_REPS} | "
            f"elapsed="
            f"{elapsed / 60:.1f} min"
        )


# =============================================================================
# 20. 保存Bootstrap原始结果
# =============================================================================

bootstrap_file = (
    BOOTSTRAP_DIR
    / (
        f"external_bootstrap_"
        f"{BOOTSTRAP_REPS}_"
        f"seed_{RANDOM_SEED}.npz"
    )
)

np.savez_compressed(
    bootstrap_file,

    uno_c=
        boot_uno,

    iauc=
        boot_iauc,

    ibs=
        boot_ibs,

    auc=
        boot_auc,

    brier=
        boot_brier,

    predicted_mean=
        boot_predicted_mean,

    observed_km=
        boot_observed_km,

    calibration_difference=
        boot_calibration_difference,

    bootstrap_reps=
        np.asarray(
            BOOTSTRAP_REPS
        ),

    random_seed=
        np.asarray(
            RANDOM_SEED
        ),
)


# =============================================================================
# 21. CI工具
# =============================================================================

def percentile_ci(
    values,
):
    values = np.asarray(
        values,
        dtype=np.float64,
    )

    values = values[
        np.isfinite(
            values
        )
    ]

    if len(
        values
    ) == 0:
        return (
            np.nan,
            np.nan,
            0,
        )

    return (
        float(
            np.quantile(
                values,
                0.025,
            )
        ),

        float(
            np.quantile(
                values,
                0.975,
            )
        ),

        int(
            len(values)
        ),
    )


# =============================================================================
# 22. Landmark 95%CI
# =============================================================================

landmark_ci_rows = []

for landmark_index in range(
    N_LANDMARK
):

    point = landmark_point.iloc[
        landmark_index
    ]

    c_low, c_high, c_n = (
        percentile_ci(
            boot_uno[
                :,
                landmark_index
            ]
        )
    )

    iauc_low, iauc_high, iauc_n = (
        percentile_ci(
            boot_iauc[
                :,
                landmark_index
            ]
        )
    )

    ibs_low, ibs_high, ibs_n = (
        percentile_ci(
            boot_ibs[
                :,
                landmark_index
            ]
        )
    )

    landmark_ci_rows.append(
        {
            "landmark_index":
                landmark_index,

            "landmark_month":
                LANDMARK_MONTHS[
                    landmark_index
                ],

            "landmark_year":
                LANDMARK_YEARS[
                    landmark_index
                ],

            "risk_set_n":
                int(
                    point[
                        "risk_set_n"
                    ]
                ),

            "event_within_5y_n":
                int(
                    point[
                        "event_within_5y_n"
                    ]
                ),

            "uno_c_index_5y":
                point[
                    "uno_c_index_5y"
                ],

            "uno_c_lower_95":
                c_low,

            "uno_c_upper_95":
                c_high,

            "uno_c_valid_bootstrap_n":
                c_n,

            "integrated_dynamic_auc":
                point[
                    "integrated_dynamic_auc"
                ],

            "iauc_lower_95":
                iauc_low,

            "iauc_upper_95":
                iauc_high,

            "iauc_valid_bootstrap_n":
                iauc_n,

            "integrated_brier_score":
                point[
                    "integrated_brier_score"
                ],

            "ibs_lower_95":
                ibs_low,

            "ibs_upper_95":
                ibs_high,

            "ibs_valid_bootstrap_n":
                ibs_n,
        }
    )

landmark_ci = pd.DataFrame(
    landmark_ci_rows
)

landmark_ci.to_csv(
    OUT
    / "04_landmark_performance_with_95CI.csv",
    index=False,
    encoding="utf-8-sig",
)


# =============================================================================
# 23. 1/3/5年 AUC/Brier/Calibration 95%CI
# =============================================================================

horizon_ci_rows = []

for landmark_index in range(
    N_LANDMARK
):

    for (
        nominal_horizon,
        future_index
    ) in REPORT_HORIZONS.items():

        point = (
            horizon_point.loc[
                (
                    horizon_point[
                        "landmark_index"
                    ]
                    == landmark_index
                )
                & (
                    horizon_point[
                        "nominal_horizon_month"
                    ]
                    == float(
                        nominal_horizon
                    )
                )
            ]
            .iloc[0]
        )

        auc_low, auc_high, auc_n = (
            percentile_ci(
                boot_auc[
                    :,
                    landmark_index,
                    future_index,
                ]
            )
        )

        brier_low, brier_high, brier_n = (
            percentile_ci(
                boot_brier[
                    :,
                    landmark_index,
                    future_index,
                ]
            )
        )

        observed_low, observed_high, observed_n = (
            percentile_ci(
                boot_observed_km[
                    :,
                    landmark_index,
                    future_index,
                ]
            )
        )

        calibration_low, calibration_high, calibration_n = (
            percentile_ci(
                boot_calibration_difference[
                    :,
                    landmark_index,
                    future_index,
                ]
            )
        )

        horizon_ci_rows.append(
            {
                "landmark_index":
                    landmark_index,

                "landmark_month":
                    LANDMARK_MONTHS[
                        landmark_index
                    ],

                "landmark_year":
                    LANDMARK_YEARS[
                        landmark_index
                    ],

                "nominal_horizon_month":
                    nominal_horizon,

                "metric_time_month":
                    METRIC_TIMES[
                        future_index
                    ],

                "horizon_year":
                    nominal_horizon
                    / 12.0,

                "risk_set_n":
                    int(
                        point[
                            "risk_set_n"
                        ]
                    ),

                "dynamic_auc":
                    point[
                        "dynamic_auc"
                    ],

                "auc_lower_95":
                    auc_low,

                "auc_upper_95":
                    auc_high,

                "auc_valid_bootstrap_n":
                    auc_n,

                "brier_score":
                    point[
                        "brier_score"
                    ],

                "brier_lower_95":
                    brier_low,

                "brier_upper_95":
                    brier_high,

                "brier_valid_bootstrap_n":
                    brier_n,

                "mean_predicted_risk":
                    point[
                        "mean_predicted_risk"
                    ],

                "km_observed_risk":
                    point[
                        "km_observed_risk"
                    ],

                "km_observed_lower_95":
                    observed_low,

                "km_observed_upper_95":
                    observed_high,

                "km_observed_valid_bootstrap_n":
                    observed_n,

                "calibration_difference_pred_minus_obs":
                    point[
                        "calibration_difference_pred_minus_obs"
                    ],

                "calibration_difference_lower_95":
                    calibration_low,

                "calibration_difference_upper_95":
                    calibration_high,

                "calibration_valid_bootstrap_n":
                    calibration_n,
            }
        )

horizon_ci = pd.DataFrame(
    horizon_ci_rows
)

horizon_ci.to_csv(
    OUT
    / "05_1y_3y_5y_performance_with_95CI.csv",
    index=False,
    encoding="utf-8-sig",
)


# =============================================================================
# 24. Landmark等权平均点估计
# =============================================================================

def equal_weight_point(
    positions,
):
    subset = landmark_point.iloc[
        positions
    ]

    return {
        "uno_c_index_5y":
            float(
                subset[
                    "uno_c_index_5y"
                ].mean()
            ),

        "integrated_dynamic_auc":
            float(
                subset[
                    "integrated_dynamic_auc"
                ].mean()
            ),

        "integrated_brier_score":
            float(
                subset[
                    "integrated_brier_score"
                ].mean()
            ),
    }


primary_point = equal_weight_point(
    PRIMARY_LANDMARK_POSITIONS
)

all6_positions = np.arange(
    N_LANDMARK,
    dtype=np.int64,
)

all6_point = equal_weight_point(
    all6_positions
)


# =============================================================================
# 25. Bootstrap Landmark等权平均
#
# 只有scope内所有Landmark均有效时才纳入该bootstrap replicate。
# 不使用partial nanmean。
# =============================================================================

def complete_case_landmark_mean(
    matrix,
    positions,
):
    subset = np.asarray(
        matrix[
            :,
            positions
        ],
        dtype=np.float64,
    )

    result = np.full(
        subset.shape[0],
        np.nan,
        dtype=np.float64,
    )

    valid = np.all(
        np.isfinite(
            subset
        ),
        axis=1,
    )

    result[
        valid
    ] = np.mean(
        subset[
            valid
        ],
        axis=1,
    )

    return result


primary_boot_uno = (
    complete_case_landmark_mean(
        boot_uno,
        PRIMARY_LANDMARK_POSITIONS,
    )
)

primary_boot_iauc = (
    complete_case_landmark_mean(
        boot_iauc,
        PRIMARY_LANDMARK_POSITIONS,
    )
)

primary_boot_ibs = (
    complete_case_landmark_mean(
        boot_ibs,
        PRIMARY_LANDMARK_POSITIONS,
    )
)

all6_boot_uno = (
    complete_case_landmark_mean(
        boot_uno,
        all6_positions,
    )
)

all6_boot_iauc = (
    complete_case_landmark_mean(
        boot_iauc,
        all6_positions,
    )
)

all6_boot_ibs = (
    complete_case_landmark_mean(
        boot_ibs,
        all6_positions,
    )
)


# =============================================================================
# 26. 主文/全6 Landmark汇总
# =============================================================================

overall_rows = []

for (
    scope,
    point,
    c_values,
    iauc_values,
    ibs_values
) in [
    (
        "primary_0_1_3_5y_landmark_equal_weight_mean",
        primary_point,
        primary_boot_uno,
        primary_boot_iauc,
        primary_boot_ibs,
    ),

    (
        "all_0_1_2_3_4_5y_landmark_equal_weight_mean",
        all6_point,
        all6_boot_uno,
        all6_boot_iauc,
        all6_boot_ibs,
    ),
]:

    c_low, c_high, c_n = (
        percentile_ci(
            c_values
        )
    )

    iauc_low, iauc_high, iauc_n = (
        percentile_ci(
            iauc_values
        )
    )

    ibs_low, ibs_high, ibs_n = (
        percentile_ci(
            ibs_values
        )
    )

    overall_rows.append(
        {
            "scope":
                scope,

            "uno_c_index_5y":
                point[
                    "uno_c_index_5y"
                ],

            "uno_c_lower_95":
                c_low,

            "uno_c_upper_95":
                c_high,

            "uno_c_valid_bootstrap_n":
                c_n,

            "integrated_dynamic_auc":
                point[
                    "integrated_dynamic_auc"
                ],

            "iauc_lower_95":
                iauc_low,

            "iauc_upper_95":
                iauc_high,

            "iauc_valid_bootstrap_n":
                iauc_n,

            "integrated_brier_score":
                point[
                    "integrated_brier_score"
                ],

            "ibs_lower_95":
                ibs_low,

            "ibs_upper_95":
                ibs_high,

            "ibs_valid_bootstrap_n":
                ibs_n,
        }
    )

overall_summary = pd.DataFrame(
    overall_rows
)

overall_summary.to_csv(
    OUT
    / "06_primary_external_validation_summary.csv",
    index=False,
    encoding="utf-8-sig",
)


# =============================================================================
# 27. Bootstrap有效率
# =============================================================================

validity_rows = []

for landmark_index in range(
    N_LANDMARK
):

    metric_specs = [
        (
            "Uno C",
            boot_uno[
                :,
                landmark_index
            ],
        ),
        (
            "iAUC",
            boot_iauc[
                :,
                landmark_index
            ],
        ),
        (
            "IBS",
            boot_ibs[
                :,
                landmark_index
            ],
        ),
        (
            "1y AUC",
            boot_auc[
                :,
                landmark_index,
                1,
            ],
        ),
        (
            "3y AUC",
            boot_auc[
                :,
                landmark_index,
                5,
            ],
        ),
        (
            "5y AUC",
            boot_auc[
                :,
                landmark_index,
                9,
            ],
        ),
    ]

    for metric_name, values in metric_specs:

        valid_n = int(
            np.isfinite(
                values
            ).sum()
        )

        valid_rate = (
            valid_n
            / BOOTSTRAP_REPS
        )

        validity_rows.append(
            {
                "landmark_month":
                    LANDMARK_MONTHS[
                        landmark_index
                    ],

                "landmark_year":
                    LANDMARK_YEARS[
                        landmark_index
                    ],

                "metric":
                    metric_name,

                "valid_bootstrap_n":
                    valid_n,

                "total_bootstrap_n":
                    BOOTSTRAP_REPS,

                "valid_rate":
                    valid_rate,
            }
        )

bootstrap_validity = pd.DataFrame(
    validity_rows
)

bootstrap_validity.to_csv(
    OUT
    / "07_bootstrap_validity.csv",
    index=False,
    encoding="utf-8-sig",
)

minimum_valid_rate = float(
    bootstrap_validity[
        "valid_rate"
    ].min()
)

if minimum_valid_rate < MIN_VALID_BOOTSTRAP_RATE:
    warnings.warn(
        "至少一个正式指标的Bootstrap有效率低于80%。"
    )


# =============================================================================
# 28. 风险集表
# =============================================================================

risk_set_table = (
    metadata
    .groupby(
        [
            "landmark_index",
            "landmark_month",
        ],
        as_index=False,
    )
    .agg(
        risk_set_n=(
            "ID",
            "size",
        ),

        event_within_5y_n=(
            "event_within_60m",
            "sum",
        ),
    )
)

risk_set_table[
    "landmark_year"
] = (
    risk_set_table[
        "landmark_month"
    ]
    / 12.0
)

risk_set_table.to_csv(
    OUT
    / "08_external_landmark_risk_sets.csv",
    index=False,
    encoding="utf-8-sig",
)


# =============================================================================
# 29. 生成主要结果易读表
# =============================================================================

main_landmark_table = (
    landmark_ci.loc[
        landmark_ci[
            "landmark_month"
        ].isin(
            PRIMARY_LANDMARK_MONTHS
        )
    ]
    .copy()
)

main_landmark_table.to_csv(
    OUT
    / "09_MAIN_0_1_3_5y_landmark_performance.csv",
    index=False,
    encoding="utf-8-sig",
)

main_horizon_table = (
    horizon_ci.loc[
        horizon_ci[
            "landmark_month"
        ].isin(
            PRIMARY_LANDMARK_MONTHS
        )
    ]
    .copy()
)

main_horizon_table.to_csv(
    OUT
    / "10_MAIN_0_1_3_5y_landmark_1_3_5y_horizons.csv",
    index=False,
    encoding="utf-8-sig",
)


# =============================================================================
# 30. Manifest
# =============================================================================

manifest = {
    "analysis":
        (
            "Chongqing independent geographic external validation "
            "of frozen LSTM-v2"
        ),

    "external_patient_n":
        int(
            N_EXTERNAL
        ),

    "external_valid_patient_landmark_n":
        int(
            LONG_N
        ),

    "bootstrap_unit":
        "external patient",

    "bootstrap_reps":
        int(
            BOOTSTRAP_REPS
        ),

    "bootstrap_seed":
        int(
            RANDOM_SEED
        ),

    "bootstrap_pairing":
        (
            "same resampled Chongqing patient set "
            "used across all landmarks"
        ),

    "ipcw_reference":
        (
            "Shenzhen and Nanning development risk set "
            "at each landmark"
        ),

    "development_reference_admin_cap_month":
        REFERENCE_ADMIN_CAP,

    "external_admin_cap_month":
        EXTERNAL_ADMIN_CAP,

    "metric_times_month":
        METRIC_TIMES.tolist(),

    "nominal_prediction_horizons_month":
        FUTURE_END_MONTHS.tolist(),

    "primary_landmarks_month":
        PRIMARY_LANDMARK_MONTHS.tolist(),

    "all_landmarks_month":
        LANDMARK_MONTHS.tolist(),

    "early_auc_boundary_rule":
        (
            "If a metric time is earlier than the minimum external "
            "follow-up time at that landmark, dynamic AUC at that "
            "time is not estimable and is recorded as NA. "
            "Such an early time has no observed event/censoring "
            "and therefore contributes no event weight to iAUC."
        ),

    "early_brier_boundary_rule":
        (
            "If metric time t is earlier than minimum external "
            "follow-up, all patients are known event-free and "
            "uncensored at t; therefore Brier(t)=mean(risk(t)^2)."
        ),

    "ibs_interval":
        "6 to 59.999 months after landmark",

    "external_outcome_used_for_training":
        False,

    "external_outcome_used_for_model_selection":
        False,

    "external_outcome_used_for_recalibration":
        False,
}

with open(
    OUT
    / "step8_FINAL_FIXED_manifest.json",
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        manifest,
        file,
        ensure_ascii=False,
        indent=2,
    )


# =============================================================================
# 31. 最终输出
# =============================================================================

elapsed_minutes = (
    time.time()
    - start_time
) / 60.0


print()
print("=" * 120)
print("重庆独立地理外部验证 Step8 FINAL FIXED 完成")
print("=" * 120)

print(
    "外部患者数：",
    N_EXTERNAL
)

print(
    "有效 patient-Landmark：",
    LONG_N
)

print(
    "Bootstrap次数：",
    BOOTSTRAP_REPS
)

print(
    "Bootstrap耗时：",
    f"{elapsed_minutes:.1f} min"
)


print()
print("六个Landmark性能：")

print(
    landmark_ci[
        [
            "landmark_year",
            "risk_set_n",
            "event_within_5y_n",
            "uno_c_index_5y",
            "uno_c_lower_95",
            "uno_c_upper_95",
            "integrated_dynamic_auc",
            "iauc_lower_95",
            "iauc_upper_95",
            "integrated_brier_score",
            "ibs_lower_95",
            "ibs_upper_95",
        ]
    ]
    .round(4)
    .to_string(
        index=False
    )
)


print()
print("主文/总体汇总：")

print(
    overall_summary
    .round(4)
    .to_string(
        index=False
    )
)


print()
print("主文Landmark的1/3/5年结果：")

print(
    main_horizon_table[
        [
            "landmark_year",
            "horizon_year",
            "dynamic_auc",
            "auc_lower_95",
            "auc_upper_95",
            "brier_score",
            "brier_lower_95",
            "brier_upper_95",
            "mean_predicted_risk",
            "km_observed_risk",
            "calibration_difference_pred_minus_obs",
        ]
    ]
    .round(4)
    .to_string(
        index=False
    )
)


print()
print(
    "最小Bootstrap有效率：",
    round(
        minimum_valid_rate,
        4,
    )
)


print()
print(
    "输出目录：",
    OUT
)

print("=" * 120)