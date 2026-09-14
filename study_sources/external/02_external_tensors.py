# -*- coding: utf-8 -*-

# =============================================================================
# 重庆外部验证 Step6 FINAL
#
# Step5 -> 最终 LSTM-v2 输入 + Landmark标签
#
# 输入：
#   __CKD_WORKDIR__/重庆外部验证_step5_GLU更新/
#
# 输出：
#   1. 85维增强动态输入
#   2. 16维静态输入
#   3. 6个Landmark对应Age
#   4. 6个Landmark风险集
#   5. 10个未来半年区间事件/风险标签
#   6. patient-Landmark长格式映射
#
# 本步骤：
#   不填补
#   不重新标准化
#   不训练模型
# =============================================================================

from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd


# =============================================================================
# 可重复性保护：严格阻止跨scikit-learn版本反序列化
#
# 冻结的StandardScaler/OneHotEncoder必须在保存它们的同一scikit-learn
# 版本中加载；否则外部验证立即停止，不能把兼容性警告当作最终结果。
# =============================================================================
import warnings as _sklearn_version_warnings
from sklearn.exceptions import InconsistentVersionWarning

_sklearn_version_warnings.filterwarnings(
    "error",
    category=InconsistentVersionWarning,
)


# =============================================================================
# 1. 路径
# =============================================================================

BASE = Path("__CKD_WORKDIR__")

STEP5 = (
    BASE /
    "重庆外部验证_step5_GLU更新"
)

DEV_STEP3 = (
    BASE /
    "rolling_5y_step3_raw_features"
)

DEV_STEP4 = (
    BASE /
    "rolling_5y_step4_preprocessed"
)

OUT = (
    BASE /
    "重庆外部验证_step6_LSTM_v2"
)

OUT.mkdir(
    parents=True,
    exist_ok=True
)


# =============================================================================
# 2. 正式模型固定参数
# =============================================================================

INTERVAL_WIDTH = 6.0
TIME_TOLERANCE = 1e-6
MAX_PREDICTION_MONTH = 60.0

LANDMARK_MONTHS = np.asarray(
    [0, 12, 24, 36, 48, 60],
    dtype=np.int32,
)

LANDMARK_BINS = (
    LANDMARK_MONTHS / INTERVAL_WIDTH
).astype(np.int32)

FUTURE_END_MONTHS = np.asarray(
    [6, 12, 18, 24, 30, 36, 42, 48, 54, 60],
    dtype=np.int32,
)

N_HISTORY = 11
N_LANDMARK = 6
N_FUTURE = 10

TIME_STEP_YEARS = (
    np.arange(
        N_HISTORY,
        dtype=np.float32,
    )
    * 0.5
)


# =============================================================================
# 3. 正式LSTM-v2变量
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


if len(LAB_FEATURES) != 15:
    raise ValueError("实验室变量数不是15。")

if len(DYNAMIC_FEATURES) != 40:
    raise ValueError("基础动态变量数不是40。")


# =============================================================================
# 4. 读取Step5
# =============================================================================

X = np.load(
    STEP5 /
    "X_full_preprocessor.npy"
).astype(np.float32)


continuous_raw = np.load(
    STEP5 /
    "continuous_raw_0_60.npy"
).astype(np.float32)


sequence_row_mask = np.load(
    STEP5 /
    "sequence_row_mask.npy"
).astype(bool)


patient_info = pd.read_csv(
    STEP5 /
    "patient_info.csv",
    encoding="utf-8-sig",
    dtype={"ID": "string"},
)


feature_names = (
    pd.read_csv(
        STEP5 /
        "feature_names.csv",
        encoding="utf-8-sig",
    )["feature_name"]
    .astype(str)
    .tolist()
)


with open(
    DEV_STEP3 /
    "feature_groups.json",
    "r",
    encoding="utf-8",
) as f:

    feature_groups = json.load(f)


continuous_vars = list(
    feature_groups["continuous_vars"]
)


preprocessor = joblib.load(
    DEV_STEP4 /
    "full_development_preprocessor.joblib"
)


# =============================================================================
# 5. 基础结构检查
# =============================================================================

N = len(patient_info)


if X.shape != (
    N,
    11,
    57,
):
    raise ValueError(
        f"57维输入形状异常：{X.shape}"
    )


if continuous_raw.shape != (
    N,
    11,
    25,
):
    raise ValueError(
        f"连续原始张量形状异常：{continuous_raw.shape}"
    )


if sequence_row_mask.shape != (
    N,
    11,
):
    raise ValueError(
        "sequence_row_mask形状异常。"
    )


if len(feature_names) != 57:
    raise ValueError(
        "feature_names不是57项。"
    )


if not np.isfinite(X).all():
    raise ValueError(
        "X存在NaN/Inf。"
    )


if not np.all(
    X[~sequence_row_mask] == 0
):
    raise ValueError(
        "缺失时间行不是57维全0。"
    )


# 患者顺序检查
if "patient_index" in patient_info.columns:

    expected_index = np.arange(
        N,
        dtype=int,
    )

    actual_index = (
        patient_info["patient_index"]
        .to_numpy(int)
    )

    if not np.array_equal(
        expected_index,
        actual_index,
    ):
        raise ValueError(
            "patient_info与张量患者顺序不一致。"
        )


# =============================================================================
# 6. 找57维变量位置
# =============================================================================

feature_to_index = {
    name: i
    for i, name in enumerate(
        feature_names
    )
}


missing_dynamic = sorted(
    set(DYNAMIC_FEATURES)
    - set(feature_names)
)


if missing_dynamic:
    raise ValueError(
        f"缺少动态变量：{missing_dynamic}"
    )


# =============================================================================
# 7. 16维静态变量
#
# BMI
# Oppinfection
# + 14个One-Hot
# =============================================================================

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


STATIC_FEATURES = [
    "BMI",
    "Oppinfection",
    *onehot_static,
]


if len(STATIC_FEATURES) != 16:

    raise ValueError(
        f"静态变量数={len(STATIC_FEATURES)}，"
        "应为16。"
    )


static_indices = np.asarray(
    [
        feature_to_index[name]
        for name in STATIC_FEATURES
    ],
    dtype=np.int64,
)


dynamic_indices = np.asarray(
    [
        feature_to_index[name]
        for name in DYNAMIC_FEATURES
    ],
    dtype=np.int64,
)


# =============================================================================
# 8. 第一个真实历史时间行
# =============================================================================

if (
    ~sequence_row_mask.any(axis=1)
).any():

    raise ValueError(
        "存在没有任何历史时间行的患者。"
    )


first_observed_step = np.argmax(
    sequence_row_mask,
    axis=1,
).astype(np.int64)


patient_rows = np.arange(
    N,
    dtype=np.int64,
)


# =============================================================================
# 9. 16维静态输入
# =============================================================================

static_baseline = np.asarray(

    X[
        patient_rows,
        first_observed_step,
        :,
    ][
        :,
        static_indices,
    ],

    dtype=np.float32,
)


if static_baseline.shape != (
    N,
    16,
):
    raise ValueError(
        "static_baseline形状错误。"
    )


# =============================================================================
# 10. Age
#
# 正式模型：
# first observed age
# - first observed时间
# = baseline age
#
# Landmark时：
# baseline age + Landmark月份/12
# =============================================================================

if "Age" not in continuous_vars:

    raise ValueError(
        "continuous_vars缺少Age。"
    )


age_continuous_index = (
    continuous_vars.index(
        "Age"
    )
)


age_first = np.asarray(

    continuous_raw[
        patient_rows,
        first_observed_step,
        age_continuous_index,
    ],

    dtype=np.float32,
)


baseline_age_raw = (
    age_first
    - TIME_STEP_YEARS[
        first_observed_step
    ]
).astype(np.float32)


if not np.isfinite(
    baseline_age_raw
).all():

    raise ValueError(
        "baseline_age_raw存在异常。"
    )


# 冻结开发集Age标准化参数
age_mean = float(
    preprocessor["scaler"]
    .mean_[
        age_continuous_index
    ]
)


age_scale = float(
    preprocessor["scaler"]
    .scale_[
        age_continuous_index
    ]
)


if (
    not np.isfinite(age_mean)
    or not np.isfinite(age_scale)
    or age_scale <= 0
):

    raise ValueError(
        "冻结Age标准化参数异常。"
    )


age_at_landmark_raw = (

    baseline_age_raw[:, None]

    + LANDMARK_MONTHS[
        None,
        :
    ] / 12.0
).astype(np.float32)


age_at_landmark_standardized = (

    (
        age_at_landmark_raw
        - age_mean
    )
    / age_scale

).astype(np.float32)


# =============================================================================
# 11. 40维基础动态输入
# =============================================================================

base_dynamic = np.asarray(

    X[
        :,
        :,
        dynamic_indices,
    ],

    dtype=np.float32,
)


if base_dynamic.shape != (
    N,
    11,
    40,
):

    raise ValueError(
        f"base_dynamic形状异常："
        f"{base_dynamic.shape}"
    )


# =============================================================================
# 12. 15项实验室原始张量
#
# 注意：
# continuous_raw已经完成Step5的LOCF及剩余填补。
#
# 这里严格复制最终LSTM-v2训练时的正式定义：
# np.isfinite(lab_raw) & sequence_row_mask
#
# 不使用“observed/LOCF/median审计文件”重新定义模型输入。
# =============================================================================

lab_continuous_indices = np.asarray(
    [
        continuous_vars.index(name)
        for name in LAB_FEATURES
    ],
    dtype=np.int64,
)


lab_raw = np.asarray(

    continuous_raw[
        :,
        :,
        lab_continuous_indices,
    ],

    dtype=np.float32,
)


if lab_raw.shape != (
    N,
    11,
    15,
):

    raise ValueError(
        "lab_raw形状错误。"
    )


# =============================================================================
# 13. 构建增强动态变量
#
# 与Step10E完全一致：
#
# 40基础动态
# +15 observed
# +15 time_since_last
# +15 delta_last_observed
# =85
# =============================================================================

lab_observed = (

    np.isfinite(
        lab_raw
    )

    & sequence_row_mask[
        :,
        :,
        None,
    ]
)


lab_observed_float = (
    lab_observed.astype(
        np.float32
    )
)


time_since = np.zeros_like(
    lab_observed_float,
    dtype=np.float32,
)


lab_delta = np.zeros_like(
    lab_observed_float,
    dtype=np.float32,
)


# base_dynamic最前15项即15项标准化实验室
standardized_labs = (
    base_dynamic[
        :,
        :,
        :15,
    ]
)


last_seen_step = np.full(
    (
        N,
        15,
    ),
    -1,
    dtype=np.int16,
)


last_seen_value = np.zeros(
    (
        N,
        15,
    ),
    dtype=np.float32,
)


has_seen = np.zeros(
    (
        N,
        15,
    ),
    dtype=bool,
)


for step_index in range(
    N_HISTORY
):

    active_row = (
        sequence_row_mask[
            :,
            step_index,
        ][
            :,
            None,
        ]
    )


    observed_now = (
        lab_observed[
            :,
            step_index,
            :
        ]
    )


    current_value = (
        standardized_labs[
            :,
            step_index,
            :
        ]
    )


    elapsed_years = (

        (
            step_index
            - last_seen_step
        )
        .astype(np.float32)

        * 0.5
    )


    elapsed_years = np.clip(
        elapsed_years,
        0.0,
        5.0,
    )


    elapsed_years[
        ~has_seen
    ] = min(
        (
            step_index + 1
        )
        * 0.5,
        5.0,
    )


    time_since[
        :,
        step_index,
        :
    ] = np.where(

        active_row,

        np.where(
            observed_now,
            0.0,
            elapsed_years / 5.0,
        ),

        0.0,
    )


    delta_now = (
        current_value
        - last_seen_value
    )


    lab_delta[
        :,
        step_index,
        :
    ] = np.where(

        observed_now
        & has_seen,

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


# =============================================================================
# 14. 85维最终动态输入
# =============================================================================

enhanced_dynamic = np.concatenate(

    [
        base_dynamic,
        lab_observed_float,
        time_since,
        lab_delta,
    ],

    axis=2,

).astype(np.float32)


enhanced_dynamic[
    ~sequence_row_mask
] = 0.0


ENHANCED_DYNAMIC_NAMES = (

    DYNAMIC_FEATURES

    + [
        f"{name}_observed"
        for name in LAB_FEATURES
    ]

    + [
        f"{name}_time_since_last"
        for name in LAB_FEATURES
    ]

    + [
        f"{name}_delta_last_observed"
        for name in LAB_FEATURES
    ]
)


if enhanced_dynamic.shape != (
    N,
    11,
    85,
):

    raise ValueError(
        f"85维动态输入形状错误："
        f"{enhanced_dynamic.shape}"
    )


if len(
    ENHANCED_DYNAMIC_NAMES
) != 85:

    raise ValueError(
        "动态变量名称不是85项。"
    )


if not np.isfinite(
    enhanced_dynamic
).all():

    raise ValueError(
        "85维动态输入存在NaN/Inf。"
    )


# =============================================================================
# 15. 患者级生存结局
# =============================================================================

event = pd.to_numeric(
    patient_info[
        "CKDstatus"
    ],
    errors="raise",
).to_numpy(
    dtype=np.int8,
)


observed_time_month = pd.to_numeric(
    patient_info[
        "interval"
    ],
    errors="raise",
).to_numpy(
    dtype=np.float32,
)


if not np.isin(
    event,
    [0, 1],
).all():

    raise ValueError(
        "CKDstatus不是0/1。"
    )


if (
    (~np.isfinite(
        observed_time_month
    )).any()
    or (
        observed_time_month
        <= 0
    ).any()
):

    raise ValueError(
        "interval存在非法值。"
    )


# =============================================================================
# 16. Landmark历史可用性
#
# 与正式v2一致：
# Landmark及以前至少存在1个真实时间行即可
# 中间缺行由sequence_row_mask处理
# =============================================================================

landmark_history_mask = np.zeros(
    (
        N_LANDMARK,
        N_HISTORY,
    ),
    dtype=bool,
)


for landmark_index, landmark_bin in enumerate(
    LANDMARK_BINS
):

    landmark_history_mask[
        landmark_index,
        :
        int(landmark_bin) + 1,
    ] = True


history_step_count = np.zeros(
    (
        N,
        N_LANDMARK,
    ),
    dtype=np.int16,
)


for landmark_index, landmark_bin in enumerate(
    LANDMARK_BINS
):

    history_step_count[
        :,
        landmark_index,
    ] = (

        sequence_row_mask[
            :,
            :
            int(landmark_bin) + 1,
        ]
        .sum(axis=1)
        .astype(np.int16)
    )


history_available_mask = (
    history_step_count > 0
)


# =============================================================================
# 17. 构建6个Landmark未来离散生存标签
#
# 完全复制Step1_new_split_landmark_v2逻辑
# =============================================================================

landmark_eligible_mask = np.zeros(
    (
        N,
        N_LANDMARK,
    ),
    dtype=bool,
)


future_event_matrix = np.zeros(
    (
        N,
        N_LANDMARK,
        N_FUTURE,
    ),
    dtype=np.float32,
)


future_at_risk_mask = np.zeros(
    (
        N,
        N_LANDMARK,
        N_FUTURE,
    ),
    dtype=bool,
)


for landmark_index, landmark_month in enumerate(
    LANDMARK_MONTHS
):

    # Landmark时仍处于观察状态
    time_eligible = (

        observed_time_month

        >
        float(
            landmark_month
        )
        + TIME_TOLERANCE
    )


    history_available = (
        history_available_mask[
            :,
            landmark_index,
        ]
    )


    eligible_idx = np.where(

        time_eligible
        & history_available

    )[0]


    landmark_eligible_mask[
        eligible_idx,
        landmark_index,
    ] = True


    remaining_time = (

        observed_time_month[
            eligible_idx
        ]

        - float(
            landmark_month
        )
    )


    eligible_event = (
        event[
            eligible_idx
        ].astype(bool)
    )


    # -------------------------------------------------------------------------
    # CKD所在未来半年区间
    # -------------------------------------------------------------------------

    event_interval = np.full(
        len(
            eligible_idx
        ),
        -1,
        dtype=np.int32,
    )


    event_interval[
        eligible_event
    ] = (

        np.ceil(

            (
                remaining_time[
                    eligible_event
                ]
                - TIME_TOLERANCE
            )

            / INTERVAL_WIDTH

        ).astype(
            np.int32
        )

        - 1
    )


    if (
        event_interval[
            eligible_event
        ] < 0
    ).any():

        raise ValueError(
            f"Landmark {landmark_month}月"
            "存在事件区间<0。"
        )


    # -------------------------------------------------------------------------
    # 10个未来半年区间
    # -------------------------------------------------------------------------

    for future_index in range(
        N_FUTURE
    ):

        interval_end = (
            future_index + 1
        ) * INTERVAL_WIDTH


        # 事件患者：
        # 事件区间以及之前仍处于风险中
        event_at_risk = (

            eligible_event

            & (
                event_interval
                >= future_index
            )
        )


        # 未事件患者：
        # 必须完整随访到这个半年区间结束
        censor_at_risk = (

            ~eligible_event

            & (
                remaining_time
                >=
                interval_end
                - TIME_TOLERANCE
            )
        )


        future_at_risk_mask[
            eligible_idx,
            landmark_index,
            future_index,
        ] = (

            event_at_risk
            | censor_at_risk
        )


        event_here = (

            eligible_event

            & (
                event_interval
                == future_index
            )
        )


        future_event_matrix[
            eligible_idx,
            landmark_index,
            future_index,
        ] = event_here.astype(
            np.float32
        )


# =============================================================================
# 18. 有效预测起点
#
# 至少拥有一个完整未来半年标签
# =============================================================================

prediction_origin_mask = (

    landmark_eligible_mask

    & future_at_risk_mask.any(
        axis=2
    )
)


if np.any(

    (
        future_event_matrix > 0
    )

    & (
        ~future_at_risk_mask
    )
):

    raise ValueError(
        "事件标签出现在风险掩码之外。"
    )


if (
    future_event_matrix
    .sum(axis=2)
    > 1
).any():

    raise ValueError(
        "同一患者同一Landmark出现多个事件区间。"
    )


if not np.array_equal(

    prediction_origin_mask,

    (
        landmark_eligible_mask
        & future_at_risk_mask.any(
            axis=2
        )
    ),
):

    raise ValueError(
        "prediction_origin_mask错误。"
    )


# =============================================================================
# 19. 转成长格式 patient × Landmark
#
# 顺序与正式Step6相同：
# np.where(prediction_origin_mask)
# =============================================================================

local_patient_idx_long, landmark_index_long = np.where(
    prediction_origin_mask
)


local_patient_idx_long = (
    local_patient_idx_long.astype(
        np.int32
    )
)


landmark_index_long = (
    landmark_index_long.astype(
        np.int8
    )
)


LONG_N = len(
    local_patient_idx_long
)


landmark_month_long = (

    LANDMARK_MONTHS[
        landmark_index_long
    ]

).astype(np.int32)


original_event_long = (

    event[
        local_patient_idx_long
    ]

).astype(np.int8)


original_observed_time_long = (

    observed_time_month[
        local_patient_idx_long
    ]

).astype(np.float64)


remaining_time_month = (

    original_observed_time_long

    - landmark_month_long.astype(
        np.float64
    )
)


if (
    remaining_time_month
    <= TIME_TOLERANCE
).any():

    raise ValueError(
        "存在Landmark后剩余随访<=0。"
    )


# =============================================================================
# 20. 正式60个月分析时间
# =============================================================================

analysis_time_month = np.minimum(

    remaining_time_month,

    MAX_PREDICTION_MONTH,

).astype(np.float32)


event_within_60m = (

    (
        original_event_long
        == 1
    )

    & (
        remaining_time_month
        <=
        MAX_PREDICTION_MONTH
        + TIME_TOLERANCE
    )

).astype(np.int8)


administrative_censor_60m = (

    (
        event_within_60m
        == 0
    )

    & (
        remaining_time_month
        >=
        MAX_PREDICTION_MONTH
        - TIME_TOLERANCE
    )

).astype(np.int8)


censor_before_60m = (

    (
        original_event_long
        == 0
    )

    & (
        remaining_time_month
        <
        MAX_PREDICTION_MONTH
        - TIME_TOLERANCE
    )

).astype(np.int8)


if np.any(

    event_within_60m
    + administrative_censor_60m
    + censor_before_60m
    != 1

):

    raise ValueError(
        "事件/删失分类不唯一。"
    )


# =============================================================================
# 21. 长格式未来标签
# =============================================================================

future_event_long = (

    future_event_matrix[
        local_patient_idx_long,
        landmark_index_long,
        :
    ]

).astype(np.float32)


future_at_risk_long = (

    future_at_risk_mask[
        local_patient_idx_long,
        landmark_index_long,
        :
    ]

).astype(bool)


future_at_risk_interval_n = (

    future_at_risk_long
    .sum(axis=1)

).astype(np.int8)


future_event_interval_n = (

    future_event_long
    .sum(axis=1)

).astype(np.int8)


# 精确事件与离散事件必须一致
if not np.array_equal(

    future_event_interval_n,

    event_within_60m,

):

    raise ValueError(
        "精确事件时间与半年离散标签不一致。"
    )


# =============================================================================
# 22. 长记录row map
# =============================================================================

long_row_index_map = np.full(
    (
        N,
        N_LANDMARK,
    ),
    -1,
    dtype=np.int32,
)


long_row_index_map[
    local_patient_idx_long,
    landmark_index_long,
] = np.arange(
    LONG_N,
    dtype=np.int32,
)


if not np.array_equal(

    long_row_index_map
    >= 0,

    prediction_origin_mask,

):

    raise ValueError(
        "long_row_index_map错误。"
    )


# =============================================================================
# 23. Landmark标准化
# =============================================================================

landmark_normalized = (

    LANDMARK_MONTHS.astype(
        np.float32
    )
    / 60.0
)


# =============================================================================
# 24. 长格式metadata
# =============================================================================

metadata = pd.DataFrame({

    "long_row_index":
        np.arange(
            LONG_N,
            dtype=np.int32,
        ),

    "local_patient_index":
        local_patient_idx_long,

    "ID":
        patient_info.loc[
            local_patient_idx_long,
            "ID",
        ]
        .astype("string")
        .to_numpy(),

    "center":
        "Chongqing",

    "landmark_index":
        landmark_index_long,

    "landmark_month":
        landmark_month_long,

    "original_event":
        original_event_long,

    "original_observed_time_month":
        original_observed_time_long.astype(
            np.float32
        ),

    "remaining_time_month":
        remaining_time_month.astype(
            np.float32
        ),

    "analysis_time_month":
        analysis_time_month,

    "event_within_60m":
        event_within_60m,

    "administrative_censor_60m":
        administrative_censor_60m,

    "censor_before_60m":
        censor_before_60m,

    "future_at_risk_interval_n":
        future_at_risk_interval_n,
})


# =============================================================================
# 25. Landmark QC
# =============================================================================

qc_rows = []


for landmark_index, landmark_month in enumerate(
    LANDMARK_MONTHS
):

    eligible = (
        landmark_eligible_mask[
            :,
            landmark_index,
        ]
    )


    valid = (
        prediction_origin_mask[
            :,
            landmark_index,
        ]
    )


    remaining_all = (

        observed_time_month

        - float(
            landmark_month
        )
    )


    event_12 = (

        valid
        & (event == 1)
        & (
            remaining_all
            <= 12
            + TIME_TOLERANCE
        )
    )


    event_36 = (

        valid
        & (event == 1)
        & (
            remaining_all
            <= 36
            + TIME_TOLERANCE
        )
    )


    event_60 = (

        valid
        & (event == 1)
        & (
            remaining_all
            <= 60
            + TIME_TOLERANCE
        )
    )


    qc_rows.append({

        "landmark_month":
            int(
                landmark_month
            ),

        "landmark_year":
            float(
                landmark_month
                / 12
            ),

        "eligible_n":
            int(
                eligible.sum()
            ),

        "valid_origin_n":
            int(
                valid.sum()
            ),

        "event_within_12m_n":
            int(
                event_12.sum()
            ),

        "event_within_36m_n":
            int(
                event_36.sum()
            ),

        "event_within_60m_n":
            int(
                event_60.sum()
            ),

        "future_label_n":
            int(
                future_at_risk_mask[
                    :,
                    landmark_index,
                    :
                ].sum()
            ),

        "median_history_steps":
            float(
                np.median(
                    history_step_count[
                        valid,
                        landmark_index,
                    ]
                )
            )
            if valid.any()
            else np.nan,
    })


landmark_qc = pd.DataFrame(
    qc_rows
)


# =============================================================================
# 26. 保存模型输入
# =============================================================================

np.save(
    OUT /
    "enhanced_dynamic_85.npy",
    enhanced_dynamic
)


np.save(
    OUT /
    "static_baseline_16.npy",
    static_baseline
)


np.save(
    OUT /
    "baseline_age_raw.npy",
    baseline_age_raw
)


np.save(
    OUT /
    "age_at_landmark_raw.npy",
    age_at_landmark_raw
)


np.save(
    OUT /
    "age_at_landmark_standardized.npy",
    age_at_landmark_standardized
)


np.save(
    OUT /
    "sequence_row_mask.npy",
    sequence_row_mask
)


# =============================================================================
# 27. 保存Landmark标签
# =============================================================================

np.save(
    OUT /
    "landmark_months.npy",
    LANDMARK_MONTHS
)


np.save(
    OUT /
    "landmark_bins.npy",
    LANDMARK_BINS
)


np.save(
    OUT /
    "landmark_normalized.npy",
    landmark_normalized
)


np.save(
    OUT /
    "future_end_months.npy",
    FUTURE_END_MONTHS
)


np.save(
    OUT /
    "landmark_history_mask.npy",
    landmark_history_mask
)


np.save(
    OUT /
    "history_step_count.npy",
    history_step_count
)


np.save(
    OUT /
    "history_available_mask.npy",
    history_available_mask
)


np.save(
    OUT /
    "landmark_eligible_mask.npy",
    landmark_eligible_mask
)


np.save(
    OUT /
    "prediction_origin_mask.npy",
    prediction_origin_mask
)


np.save(
    OUT /
    "future_event_matrix.npy",
    future_event_matrix
)


np.save(
    OUT /
    "future_at_risk_mask.npy",
    future_at_risk_mask
)


# =============================================================================
# 28. 保存长格式
# =============================================================================

np.save(
    OUT /
    "external_long_row_index_map.npy",
    long_row_index_map
)


np.save(
    OUT /
    "external_local_patient_idx_long.npy",
    local_patient_idx_long
)


np.save(
    OUT /
    "external_landmark_index_long.npy",
    landmark_index_long
)


np.save(
    OUT /
    "external_landmark_month_long.npy",
    landmark_month_long
)


np.save(
    OUT /
    "external_future_event_long.npy",
    future_event_long
)


np.save(
    OUT /
    "external_future_at_risk_long.npy",
    future_at_risk_long
)


np.save(
    OUT /
    "external_analysis_time_month.npy",
    analysis_time_month
)


np.save(
    OUT /
    "external_event_within_60m.npy",
    event_within_60m
)


metadata.to_csv(
    OUT /
    "super_landmark_external_metadata.csv",
    index=False,
    encoding="utf-8-sig",
)


# =============================================================================
# 29. 保存变量名称
# =============================================================================

pd.DataFrame({

    "dynamic_index":
        np.arange(
            85,
            dtype=np.int32,
        ),

    "feature_name":
        ENHANCED_DYNAMIC_NAMES,

}).to_csv(
    OUT /
    "enhanced_dynamic_feature_names.csv",
    index=False,
    encoding="utf-8-sig",
)


pd.DataFrame({

    "static_index":
        np.arange(
            16,
            dtype=np.int32,
        ),

    "feature_name":
        STATIC_FEATURES,

}).to_csv(
    OUT /
    "static_feature_names.csv",
    index=False,
    encoding="utf-8-sig",
)


landmark_qc.to_csv(
    OUT /
    "landmark_qc.csv",
    index=False,
    encoding="utf-8-sig",
)


# =============================================================================
# 30. Manifest
# =============================================================================

manifest = {

    "patient_n":
        int(N),

    "event_n":
        int(event.sum()),

    "history_step_n":
        11,

    "base_feature_n":
        57,

    "static_feature_n":
        16,

    "base_dynamic_n":
        40,

    "lab_n":
        15,

    "enhanced_dynamic_n":
        85,

    "landmark_months":
        LANDMARK_MONTHS.tolist(),

    "future_end_months":
        FUTURE_END_MONTHS.tolist(),

    "valid_patient_landmark_n":
        int(
            prediction_origin_mask.sum()
        ),

    "long_record_n":
        int(LONG_N),

    "age_mean":
        age_mean,

    "age_scale":
        age_scale,
}


with open(
    OUT /
    "step6_manifest.json",
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        manifest,
        f,
        ensure_ascii=False,
        indent=2,
    )


# =============================================================================
# 31. 最终QC
# =============================================================================

print()
print("=" * 90)
print("重庆外部验证 Step6：LSTM-v2输入和Landmark标签构建完成")
print("=" * 90)

print(
    "患者数：",
    N
)

print(
    "CKD事件：",
    int(
        event.sum()
    )
)

print()
print("模型输入：")

print(
    "57维基础输入：",
    X.shape
)

print(
    "85维动态输入：",
    enhanced_dynamic.shape
)

print(
    "16维静态输入：",
    static_baseline.shape
)

print(
    "Age × 6 Landmark：",
    age_at_landmark_standardized.shape
)

print()
print("标签：")

print(
    "Landmark eligible：",
    landmark_eligible_mask.shape
)

print(
    "prediction_origin：",
    prediction_origin_mask.shape
)

print(
    "future_event：",
    future_event_matrix.shape
)

print(
    "future_at_risk：",
    future_at_risk_mask.shape
)

print()
print(
    "有效 patient-Landmark：",
    int(
        prediction_origin_mask.sum()
    )
)

print(
    "长格式记录：",
    LONG_N
)

print()
print("Landmark QC：")

print(
    landmark_qc.to_string(
        index=False
    )
)

print()
print(
    "85维NaN/Inf：",
    int(
        (
            ~np.isfinite(
                enhanced_dynamic
            )
        ).sum()
    )
)

print(
    "16维NaN/Inf：",
    int(
        (
            ~np.isfinite(
                static_baseline
            )
        ).sum()
    )
)

print(
    "Age NaN/Inf：",
    int(
        (
            ~np.isfinite(
                age_at_landmark_standardized
            )
        ).sum()
    )
)

print()
print(
    "输出目录：",
    OUT
)

print("=" * 90)