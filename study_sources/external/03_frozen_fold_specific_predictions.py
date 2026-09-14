# -*- coding: utf-8 -*-

# =============================================================================
# 重庆独立地理外部验证
# Step7R FINAL — Fold-specific preprocessing + frozen LSTM-v2 ensemble
#
# 这是用于替换旧 Step7 的最终修正版。
#
# 核心原则：
#
#   重庆 Step5 原始张量
#       ↓
#   Fold 0 preprocessor → Fold 0 LSTM ensemble
#   Fold 1 preprocessor → Fold 1 LSTM ensemble
#   Fold 2 preprocessor → Fold 2 LSTM ensemble
#   Fold 3 preprocessor → Fold 3 LSTM ensemble
#   Fold 4 preprocessor → Fold 4 LSTM ensemble
#       ↓
#   五折 raw discrete hazard 等权平均
#       ↓
#   Step11 final_all_development_oof 冻结校准器
#       ↓
#   0.5–5年累计CKD风险
#
# 本步骤：
#   - 不重新训练模型
#   - 不拟合重庆数据
#   - 不重新填补重庆数据
#   - 不利用重庆结局调模型
#   - 不利用重庆结局重新校准
#
# 注意：
#   Step6 中 prediction_origin / labels / metadata 继续使用；
#   但 Step6 中基于 full-preprocessor 生成的
#   enhanced_dynamic_85 / static_baseline / age-z
#   不再用于模型预测。
# =============================================================================


from __future__ import annotations

from pathlib import Path
from contextlib import nullcontext
from typing import Any
import gc
import json

import joblib
import numpy as np
import pandas as pd

import torch
import torch.nn as nn


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

BASE = Path(
    "__CKD_WORKDIR__"
)


# -------------------------------------------------------------------------
# 重庆 Step5：
# 这里读取原始25/18/4张量，而不是读取full-preprocessor后的57维张量
# -------------------------------------------------------------------------

STEP5 = (
    BASE
    / "重庆外部验证_step5_GLU更新"
)


# -------------------------------------------------------------------------
# Step6：
# 只使用Landmark标签、风险集、metadata
# -------------------------------------------------------------------------

STEP6 = (
    BASE
    / "重庆外部验证_step6_LSTM_v2"
)


# -------------------------------------------------------------------------
# 正式开发Step3/Step4
# -------------------------------------------------------------------------

STEP3 = (
    BASE
    / "rolling_5y_step3_raw_features"
)

STEP4 = (
    BASE
    / "rolling_5y_step4_preprocessed"
)


# -------------------------------------------------------------------------
# 冻结LSTM-v2模型
# -------------------------------------------------------------------------

STEP10E = (
    BASE
    / "rolling_5y_step10e_lstm_v2_final_oof_selected_existing_trials"
)


# -------------------------------------------------------------------------
# Step11校准器
# -------------------------------------------------------------------------

STEP11 = (
    BASE
    / "rolling_5y_step11_v2_four_model_oof_comparison_lstm_v2"
)

CALIBRATION_FILE = (
    STEP11
    / "four_model_hazard_calibration_parameters.csv"
)


# -------------------------------------------------------------------------
# 旧错误Step7，仅用于最后比较
# 不参与新预测
# -------------------------------------------------------------------------

OLD_STEP7 = (
    BASE
    / "重庆外部验证_step7_LSTM_v2_prediction"
)


# -------------------------------------------------------------------------
# 新最终结果
# -------------------------------------------------------------------------

OUT = (
    BASE
    / "重庆外部验证_step7R_FINAL_fold_specific"
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)


# =============================================================================
# 2. 固定模型配置
# =============================================================================

N_FOLDS = 5

HISTORY_N = 11

CONTINUOUS_N = 25
BINARY_N = 18
CATEGORICAL_N = 4

BASE_FEATURE_N = 57

STATIC_N = 16
BASE_DYNAMIC_N = 40

LAB_N = 15
ENHANCED_DYNAMIC_N = 85

LANDMARK_N = 6
FUTURE_N = 10

EPS = 1e-7


LANDMARK_MONTHS = np.asarray(
    [
        0,
        12,
        24,
        36,
        48,
        60,
    ],
    dtype=np.int32,
)


LANDMARK_BINS = (
    LANDMARK_MONTHS // 6
).astype(
    np.int64
)


LANDMARK_NORMALIZED = (
    LANDMARK_MONTHS.astype(
        np.float32
    )
    / 60.0
)


FUTURE_END_MONTHS = np.arange(
    6,
    61,
    6,
    dtype=np.float64,
)


TIME_STEP_YEARS = (
    np.arange(
        HISTORY_N,
        dtype=np.float32,
    )
    * 0.5
)


# =============================================================================
# 3. 正式模型变量
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


if len(
    LAB_FEATURES
) != LAB_N:

    raise ValueError(
        "LAB_FEATURES不是15项。"
    )


if len(
    DYNAMIC_FEATURES
) != BASE_DYNAMIC_N:

    raise ValueError(
        "DYNAMIC_FEATURES不是40项。"
    )


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


if len(
    ENHANCED_DYNAMIC_NAMES
) != ENHANCED_DYNAMIC_N:

    raise ValueError(
        "增强动态变量不是85项。"
    )


# =============================================================================
# 4. 文件存在性检查
# =============================================================================

required_files = [

    STEP5
    / "continuous_raw_0_60.npy",

    STEP5
    / "binary_raw_0_60.npy",

    STEP5
    / "categorical_raw_0_60.npy",

    STEP5
    / "sequence_row_mask.npy",

    STEP5
    / "patient_info.csv",

    STEP3
    / "feature_groups.json",

    STEP6
    / "prediction_origin_mask.npy",

    STEP6
    / "external_local_patient_idx_long.npy",

    STEP6
    / "external_landmark_index_long.npy",

    STEP6
    / "super_landmark_external_metadata.csv",

    CALIBRATION_FILE,
]


for fold_id in range(
    N_FOLDS
):

    required_files.extend(

        [

            STEP4
            / f"fold_{fold_id}"
            / "preprocessor.joblib",

            STEP4
            / f"fold_{fold_id}"
            / "feature_names.csv",

        ]
    )


missing_files = [

    str(path)

    for path in required_files

    if not path.exists()
]


if missing_files:

    raise FileNotFoundError(

        "以下必要文件不存在：\n"

        + "\n".join(
            missing_files
        )
    )


# =============================================================================
# 5. 读取重庆Step5原始张量
# =============================================================================

continuous_raw = np.load(

    STEP5
    / "continuous_raw_0_60.npy"

).astype(
    np.float32
)


binary_raw = np.load(

    STEP5
    / "binary_raw_0_60.npy"

).astype(
    np.float32
)


categorical_raw = np.load(

    STEP5
    / "categorical_raw_0_60.npy",

    allow_pickle=True,

)


sequence_row_mask = np.load(

    STEP5
    / "sequence_row_mask.npy"

).astype(
    bool
)


patient_info = pd.read_csv(

    STEP5
    / "patient_info.csv",

    encoding="utf-8-sig",

    dtype={
        "ID": "string"
    },
)


N = len(
    patient_info
)


# =============================================================================
# 6. Step5结构检查
# =============================================================================

if continuous_raw.shape != (
    N,
    HISTORY_N,
    CONTINUOUS_N,
):

    raise ValueError(

        "continuous_raw形状错误："

        f"{continuous_raw.shape}"
    )


if binary_raw.shape != (
    N,
    HISTORY_N,
    BINARY_N,
):

    raise ValueError(

        "binary_raw形状错误："

        f"{binary_raw.shape}"
    )


if categorical_raw.shape != (
    N,
    HISTORY_N,
    CATEGORICAL_N,
):

    raise ValueError(

        "categorical_raw形状错误："

        f"{categorical_raw.shape}"
    )


if sequence_row_mask.shape != (
    N,
    HISTORY_N,
):

    raise ValueError(
        "sequence_row_mask形状错误。"
    )


if not np.isfinite(

    continuous_raw[
        sequence_row_mask
    ]

).all():

    raise ValueError(
        "真实时间行连续变量仍存在NaN/Inf。"
    )


if not np.isfinite(

    binary_raw[
        sequence_row_mask
    ]

).all():

    raise ValueError(
        "真实时间行二分类变量存在NaN/Inf。"
    )


if not np.isin(

    binary_raw[
        sequence_row_mask
    ],

    [
        0.0,
        1.0,
    ],

).all():

    raise ValueError(
        "二分类变量存在非0/1值。"
    )


# =============================================================================
# 7. 正式47变量定义
# =============================================================================

with open(

    STEP3
    / "feature_groups.json",

    "r",

    encoding="utf-8",

) as file:

    groups = json.load(
        file
    )


continuous_vars = list(
    groups[
        "continuous_vars"
    ]
)


binary_vars = list(
    groups[
        "binary_vars"
    ]
)


categorical_vars = list(
    groups[
        "categorical_vars"
    ]
)


if len(
    continuous_vars
) != CONTINUOUS_N:

    raise ValueError(
        "连续变量不是25项。"
    )


if len(
    binary_vars
) != BINARY_N:

    raise ValueError(
        "二分类变量不是18项。"
    )


if len(
    categorical_vars
) != CATEGORICAL_N:

    raise ValueError(
        "分类变量不是4项。"
    )


if "Age" not in continuous_vars:

    raise ValueError(
        "continuous_vars缺少Age。"
    )


AGE_INDEX = continuous_vars.index(
    "Age"
)


LAB_CONTINUOUS_INDICES = np.asarray(

    [

        continuous_vars.index(
            name
        )

        for name in LAB_FEATURES
    ],

    dtype=np.int64,
)


# =============================================================================
# 8. 读取Step6的Landmark定义和长格式
# =============================================================================

prediction_origin_mask = np.load(

    STEP6
    / "prediction_origin_mask.npy"

).astype(
    bool
)


local_patient_idx_long = np.load(

    STEP6
    / "external_local_patient_idx_long.npy"

).astype(
    np.int32
)


landmark_index_long = np.load(

    STEP6
    / "external_landmark_index_long.npy"

).astype(
    np.int8
)


metadata = pd.read_csv(

    STEP6
    / "super_landmark_external_metadata.csv",

    encoding="utf-8-sig",

    dtype={
        "ID": "string"
    },
)


if prediction_origin_mask.shape != (
    N,
    LANDMARK_N,
):

    raise ValueError(
        "prediction_origin_mask形状错误。"
    )


sample_pairs = np.argwhere(
    prediction_origin_mask
).astype(
    np.int32
)


expected_pairs = np.column_stack(

    [
        local_patient_idx_long,
        landmark_index_long,
    ]

).astype(
    np.int32
)


if not np.array_equal(
    sample_pairs,
    expected_pairs,
):

    raise ValueError(
        "Step6长格式患者-Landmark顺序错误。"
    )


LONG_N = len(
    sample_pairs
)


if LONG_N != len(
    metadata
):

    raise ValueError(
        "metadata行数与有效预测起点不一致。"
    )


# =============================================================================
# 9. 第一个真实时间点和原始基线Age
# =============================================================================

if (
    ~sequence_row_mask.any(
        axis=1
    )
).any():

    raise ValueError(
        "存在没有真实历史时间行的患者。"
    )


first_observed_step = np.argmax(

    sequence_row_mask,

    axis=1,

).astype(
    np.int64
)


patient_rows = np.arange(
    N,
    dtype=np.int64,
)


age_first = np.asarray(

    continuous_raw[

        patient_rows,

        first_observed_step,

        AGE_INDEX,

    ],

    dtype=np.float32,
)


baseline_age_raw = (

    age_first

    - TIME_STEP_YEARS[
        first_observed_step
    ]

).astype(
    np.float32
)


if not np.isfinite(
    baseline_age_raw
).all():

    raise ValueError(
        "baseline_age_raw存在NaN/Inf。"
    )


# =============================================================================
# 10. fold-specific 57维转换
# =============================================================================

def transform_external_with_fold_preprocessor(

    preprocessor: dict,

) -> np.ndarray:


    scaler = preprocessor[
        "scaler"
    ]


    encoder = preprocessor[
        "encoder"
    ]


    final_feature_names = list(

        preprocessor[
            "final_feature_names"
        ]
    )


    if len(
        final_feature_names
    ) != BASE_FEATURE_N:

        raise ValueError(
            "fold preprocessor最终特征数不是57。"
        )


    output = np.zeros(

        (
            N,
            HISTORY_N,
            BASE_FEATURE_N,
        ),

        dtype=np.float32,
    )


    continuous_observed = np.asarray(

        continuous_raw[
            sequence_row_mask
        ],

        dtype=np.float32,
    )


    binary_observed = np.asarray(

        binary_raw[
            sequence_row_mask
        ],

        dtype=np.float32,
    )


    categorical_observed = np.asarray(

        categorical_raw[
            sequence_row_mask
        ],

        dtype=object,
    )


    # -------------------------------------------------------------------------
    # 连续变量：当前fold训练集冻结StandardScaler
    # -------------------------------------------------------------------------

    continuous_scaled = (

        scaler.transform(
            continuous_observed
        )
        .astype(
            np.float32
        )
    )


    # -------------------------------------------------------------------------
    # 分类变量：当前fold训练集冻结OneHotEncoder
    # -------------------------------------------------------------------------

    categorical_onehot = (
        encoder.transform(
            categorical_observed
        )
    )


    if hasattr(
        categorical_onehot,
        "toarray",
    ):

        categorical_onehot = (
            categorical_onehot.toarray()
        )


    categorical_onehot = np.asarray(

        categorical_onehot,

        dtype=np.float32,
    )


    observed_features = np.concatenate(

        [

            continuous_scaled,

            binary_observed,

            categorical_onehot,

        ],

        axis=1,

    ).astype(
        np.float32
    )


    if observed_features.shape[1] != BASE_FEATURE_N:

        raise ValueError(

            "fold转换后不是57维："

            f"{observed_features.shape}"
        )


    output[
        sequence_row_mask
    ] = observed_features


    if not np.isfinite(
        output
    ).all():

        raise ValueError(
            "fold-specific 57维张量存在NaN/Inf。"
        )


    if not np.all(

        output[
            ~sequence_row_mask
        ]

        == 0.0

    ):

        raise ValueError(
            "空缺时间行不是57维全0。"
        )


    # -------------------------------------------------------------------------
    # 外部分类变量不能出现未知类别
    #
    # 正式encoder虽然handle_unknown=ignore，
    # 但这里作为外部验证审计，要求4个one-hot组真实时间行和均为1。
    # -------------------------------------------------------------------------

    onehot_start = (
        CONTINUOUS_N
        + BINARY_N
    )


    current_start = (
        onehot_start
    )


    for (
        variable,
        categories
    ) in zip(

        categorical_vars,

        encoder.categories_,

    ):

        current_end = (

            current_start

            + len(
                categories
            )
        )


        group_sum = output[
            :,
            :,
            current_start:
            current_end
        ].sum(
            axis=2
        )


        if not np.allclose(

            group_sum[
                sequence_row_mask
            ],

            1.0,

            atol=1e-6,

        ):

            raise ValueError(

                f"外部数据{variable}"
                "存在开发fold未知类别或One-Hot异常。"
            )


        current_start = (
            current_end
        )


    if current_start != BASE_FEATURE_N:

        raise ValueError(
            "One-Hot边界与57维不一致。"
        )


    return output


# =============================================================================
# 11. 构建85维增强动态输入
#
# 严格复制Step10E定义
# =============================================================================

def build_enhanced_dynamic(

    base_dynamic: np.ndarray,

) -> np.ndarray:


    base_dynamic = np.asarray(

        base_dynamic,

        dtype=np.float32,
    )


    expected_shape = (

        N,
        HISTORY_N,
        BASE_DYNAMIC_N,
    )


    if base_dynamic.shape != expected_shape:

        raise ValueError(

            "base_dynamic形状错误："

            f"{base_dynamic.shape}"
        )


    lab_raw = np.asarray(

        continuous_raw[
            :,
            :,
            LAB_CONTINUOUS_INDICES,
        ],

        dtype=np.float32,
    )


    if lab_raw.shape != (

        N,
        HISTORY_N,
        LAB_N,

    ):

        raise ValueError(
            "lab_raw形状错误。"
        )


    # -------------------------------------------------------------------------
    # 注意：
    #
    # 正式Step10E就是对已填补continuous_raw使用np.isfinite，
    # 因此不能在外部验证阶段改成新的真实抽血mask。
    # -------------------------------------------------------------------------

    lab_observed = (

        np.isfinite(
            lab_raw
        )

        & sequence_row_mask[
            :,
            :,
            None
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


    # 40基础动态变量的前15项就是15个实验室
    standardized_labs = (

        base_dynamic[
            :,
            :,
            :LAB_N
        ]
    )


    last_seen_step = np.full(

        (
            N,
            LAB_N,
        ),

        -1,

        dtype=np.int16,
    )


    last_seen_value = np.zeros(

        (
            N,
            LAB_N,
        ),

        dtype=np.float32,
    )


    has_seen = np.zeros(

        (
            N,
            LAB_N,
        ),

        dtype=bool,
    )


    for step_index in range(
        HISTORY_N
    ):


        active_row = (

            sequence_row_mask[
                :,
                step_index
            ][
                :,
                None
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
            .astype(
                np.float32
            )

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
                step_index
                + 1
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

                elapsed_years
                / 5.0,

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


    enhanced = np.concatenate(

        [

            base_dynamic,

            lab_observed_float,

            time_since,

            lab_delta,

        ],

        axis=2,

    ).astype(
        np.float32
    )


    enhanced[
        ~sequence_row_mask
    ] = 0.0


    if enhanced.shape != (

        N,
        HISTORY_N,
        ENHANCED_DYNAMIC_N,

    ):

        raise ValueError(

            "85维增强动态张量形状错误："

            f"{enhanced.shape}"
        )


    if not np.isfinite(
        enhanced
    ).all():

        raise ValueError(
            "85维动态张量存在NaN/Inf。"
        )


    return enhanced


# =============================================================================
# 12. PyTorch模型
# =============================================================================

class HybridAttentionLSTMSurvival(
    nn.Module
):

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


        self.dynamic_n = int(
            dynamic_n
        )

        self.static_n = int(
            static_n
        )

        self.hidden_size = int(
            hidden_size
        )

        self.num_layers = int(
            num_layers
        )

        self.bidirectional = bool(
            bidirectional
        )

        self.pooling_mode = str(
            pooling_mode
        )

        self.future_n = int(
            future_n
        )


        self.direction_n = (

            2

            if self.bidirectional

            else 1
        )


        self.representation_n = (

            self.hidden_size

            * self.direction_n
        )


        if self.pooling_mode not in {

            "last_attention",

            "last_attention_mean",

        }:

            raise ValueError(
                "pooling_mode无效。"
            )


        self.input_encoder = nn.Sequential(

            nn.LayerNorm(
                self.dynamic_n
                + 2
            ),

            nn.Linear(

                self.dynamic_n
                + 2,

                int(
                    projection_size
                ),
            ),

            nn.SiLU(),

            nn.Dropout(
                float(
                    dropout
                )
            ),
        )


        self.forward_cells = nn.ModuleList(

            [

                nn.LSTMCell(

                    input_size=(

                        int(
                            projection_size
                        )

                        if layer_index == 0

                        else self.hidden_size
                    ),

                    hidden_size=
                    self.hidden_size,
                )

                for layer_index
                in range(
                    self.num_layers
                )
            ]
        )


        if self.bidirectional:

            self.backward_cells = nn.ModuleList(

                [

                    nn.LSTMCell(

                        input_size=(

                            int(
                                projection_size
                            )

                            if layer_index == 0

                            else self.hidden_size
                        ),

                        hidden_size=
                        self.hidden_size,
                    )

                    for layer_index
                    in range(
                        self.num_layers
                    )
                ]
            )

        else:

            self.backward_cells = None


        self.recurrent_dropout = nn.Dropout(

            float(
                dropout
            )
        )


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

            nn.LayerNorm(
                self.static_n
            ),

            nn.Linear(

                self.static_n,

                int(
                    static_hidden
                ),
            ),

            nn.SiLU(),

            nn.Dropout(
                float(
                    dropout
                )
            ),
        )


        self.dynamic_summary_encoder = nn.Sequential(

            nn.LayerNorm(

                self.dynamic_n
                * 2

            ),

            nn.Linear(

                self.dynamic_n
                * 2,

                int(
                    summary_hidden
                ),
            ),

            nn.SiLU(),

            nn.Dropout(
                float(
                    dropout
                )
            ),
        )


        recurrent_context_n = (

            self.representation_n
            * 2
        )


        if (
            self.pooling_mode
            == "last_attention_mean"
        ):

            recurrent_context_n += (
                self.representation_n
            )


        context_n = (

            recurrent_context_n

            + int(
                static_hidden
            )

            + int(
                summary_hidden
            )

            + 2
        )


        self.context_encoder = nn.Sequential(

            nn.LayerNorm(
                context_n
            ),

            nn.Linear(

                context_n,

                self.hidden_size,
            ),

            nn.SiLU(),

            nn.Dropout(
                float(
                    dropout
                )
            ),
        )


        self.horizon_embedding = nn.Embedding(

            self.future_n,

            int(
                horizon_embed_dim
            ),
        )


        head_input_n = (

            self.hidden_size

            + int(
                horizon_embed_dim
            )
        )


        head_hidden_n = max(

            32,

            self.hidden_size
            // 2,
        )


        self.hazard_head = nn.Sequential(

            nn.LayerNorm(
                head_input_n
            ),

            nn.Linear(

                head_input_n,

                head_hidden_n,
            ),

            nn.SiLU(),

            nn.Dropout(
                float(
                    dropout
                )
            ),

            nn.Linear(

                head_hidden_n,

                1,
            ),
        )


        self.interval_bias = nn.Parameter(

            torch.zeros(

                self.future_n,

                dtype=torch.float32,
            )
        )


    # =========================================================================
    # 单方向LSTM
    # =========================================================================

    def _run_direction(

        self,

        encoded_sequence,

        active_mask,

        cells,

        reverse,

    ):


        (
            batch_n,
            time_n,
            _
        ) = encoded_sequence.shape


        hidden = [

            torch.zeros(

                batch_n,

                self.hidden_size,

                dtype=
                encoded_sequence.dtype,

                device=
                encoded_sequence.device,
            )

            for _ in range(
                self.num_layers
            )
        ]


        cell = [

            torch.zeros_like(
                hidden[0]
            )

            for _ in range(
                self.num_layers
            )
        ]


        history = [
            None
        ] * time_n


        indices = (

            range(
                time_n - 1,
                -1,
                -1,
            )

            if reverse

            else range(
                time_n
            )
        )


        for step_index in indices:


            step_active = (

                active_mask[
                    :,
                    step_index
                ]
                .unsqueeze(
                    1
                )
            )


            layer_input = (

                encoded_sequence[
                    :,
                    step_index,
                    :
                ]
            )


            for (
                layer_index,
                recurrent_cell
            ) in enumerate(
                cells
            ):


                (
                    candidate_hidden,
                    candidate_cell,

                ) = recurrent_cell(

                    layer_input,

                    (
                        hidden[
                            layer_index
                        ],

                        cell[
                            layer_index
                        ],
                    ),
                )


                hidden[
                    layer_index
                ] = torch.where(

                    step_active,

                    candidate_hidden,

                    hidden[
                        layer_index
                    ],
                )


                cell[
                    layer_index
                ] = torch.where(

                    step_active,

                    candidate_cell,

                    cell[
                        layer_index
                    ],
                )


                layer_input = (

                    hidden[
                        layer_index
                    ]
                )


                if (

                    layer_index

                    < self.num_layers - 1

                ):

                    layer_input = (

                        self.recurrent_dropout(
                            layer_input
                        )
                    )


            history[
                step_index
            ] = hidden[-1]


        return (

            torch.stack(

                history,

                dim=1,
            ),

            hidden[-1],
        )


    # =========================================================================
    # Forward
    # =========================================================================

    def forward(

        self,

        dynamic_sequence,

        row_mask,

        static_baseline,

        age_at_landmark,

        landmark_normalized,

        landmark_bin,

    ):


        (
            batch_n,
            time_n,
            dynamic_n,
        ) = dynamic_sequence.shape


        if dynamic_n != self.dynamic_n:

            raise ValueError(
                "动态特征数错误。"
            )


        step_indices = torch.arange(

            time_n,

            device=
            dynamic_sequence.device,
        )


        # ---------------------------------------------------------------------
        # Landmark以后历史严格禁止进入网络
        # ---------------------------------------------------------------------

        active_mask = (

            row_mask

            & (

                step_indices[
                    None,
                    :
                ]

                <= landmark_bin[
                    :,
                    None
                ]
            )
        )


        if (

            ~active_mask.any(
                dim=1
            )

        ).any():

            raise ValueError(
                "存在Landmark以前无历史记录的样本。"
            )


        # ---------------------------------------------------------------------
        # 绝对时间
        # ---------------------------------------------------------------------

        time_normalized = (

            step_indices.float()

            / float(

                max(
                    time_n - 1,
                    1
                )
            )

        ).expand(

            batch_n,

            time_n,
        )


        # ---------------------------------------------------------------------
        # 相邻真实时间行gap
        # ---------------------------------------------------------------------

        previous_active = torch.full(

            (
                batch_n,
            ),

            -1,

            dtype=torch.long,

            device=
            dynamic_sequence.device,
        )


        gap_values = []


        for step_index in range(
            time_n
        ):


            current_active = (

                active_mask[
                    :,
                    step_index
                ]
            )


            gap = torch.where(

                current_active
                & (
                    previous_active
                    >= 0
                ),

                (
                    step_index
                    - previous_active
                ).float()

                / float(
                    max(
                        time_n - 1,
                        1
                    )
                ),

                torch.zeros(

                    batch_n,

                    dtype=
                    dynamic_sequence.dtype,

                    device=
                    dynamic_sequence.device,
                ),
            )


            gap_values.append(
                gap
            )


            previous_active = torch.where(

                current_active,

                torch.full_like(

                    previous_active,

                    step_index,
                ),

                previous_active,
            )


        gap_normalized = torch.stack(

            gap_values,

            dim=1,
        )


        # ---------------------------------------------------------------------
        # Input encoder
        # ---------------------------------------------------------------------

        encoded_sequence = (

            self.input_encoder(

                torch.cat(

                    [

                        dynamic_sequence,

                        time_normalized
                        .unsqueeze(
                            2
                        ),

                        gap_normalized
                        .unsqueeze(
                            2
                        ),

                    ],

                    dim=2,
                )
            )
        )


        # ---------------------------------------------------------------------
        # Forward LSTM
        # ---------------------------------------------------------------------

        (
            forward_history,
            forward_final,

        ) = self._run_direction(

            encoded_sequence,

            active_mask,

            self.forward_cells,

            reverse=False,
        )


        # ---------------------------------------------------------------------
        # Backward LSTM
        # ---------------------------------------------------------------------

        if self.bidirectional:


            (
                backward_history,
                backward_final,

            ) = self._run_direction(

                encoded_sequence,

                active_mask,

                self.backward_cells,

                reverse=True,
            )


            recurrent_history = torch.cat(

                [

                    forward_history,

                    backward_history,

                ],

                dim=2,
            )


            final_hidden = torch.cat(

                [

                    forward_final,

                    backward_final,

                ],

                dim=1,
            )


        else:


            recurrent_history = (
                forward_history
            )


            final_hidden = (
                forward_final
            )


        # ---------------------------------------------------------------------
        # Attention
        # ---------------------------------------------------------------------

        attention_logits = (

            self.attention_score(

                torch.tanh(

                    self.attention_hidden(
                        recurrent_history
                    )

                    + self.attention_query(
                        final_hidden
                    )
                    .unsqueeze(
                        1
                    )
                )
            )
            .squeeze(
                2
            )
        )


        attention_logits = (

            attention_logits
            .masked_fill(

                ~active_mask,

                -1e4,
            )
        )


        attention_weight = (

            torch.softmax(

                attention_logits,

                dim=1,
            )
        )


        attention_pool = torch.sum(

            recurrent_history

            * attention_weight
            .unsqueeze(
                2
            ),

            dim=1,
        )


        # ---------------------------------------------------------------------
        # 历史mean
        # ---------------------------------------------------------------------

        active_float = (

            active_mask
            .unsqueeze(
                2
            )
            .to(
                dynamic_sequence.dtype
            )
        )


        active_count = (

            active_float
            .sum(
                dim=1
            )
            .clamp_min(
                1.0
            )
        )


        recurrent_mean = (

            (

                recurrent_history

                * active_float

            )
            .sum(
                dim=1
            )

            / active_count
        )


        dynamic_mean = (

            (

                dynamic_sequence

                * active_float

            )
            .sum(
                dim=1
            )

            / active_count
        )


        # ---------------------------------------------------------------------
        # Landmark前最后有效时间点
        # ---------------------------------------------------------------------

        last_position = (

            active_mask.long()

            * (

                step_indices[
                    None,
                    :
                ]

                + 1
            )

        ).argmax(
            dim=1
        )


        batch_index = torch.arange(

            batch_n,

            device=
            dynamic_sequence.device,
        )


        dynamic_last = (

            dynamic_sequence[

                batch_index,

                last_position,

                :
            ]
        )


        # ---------------------------------------------------------------------
        # Static
        # ---------------------------------------------------------------------

        static_encoded = (

            self.static_encoder(
                static_baseline
            )
        )


        # ---------------------------------------------------------------------
        # Dynamic summary
        # ---------------------------------------------------------------------

        summary_encoded = (

            self.dynamic_summary_encoder(

                torch.cat(

                    [

                        dynamic_last,

                        dynamic_mean,

                    ],

                    dim=1,
                )
            )
        )


        recurrent_parts = [

            final_hidden,

            attention_pool,
        ]


        if (
            self.pooling_mode
            == "last_attention_mean"
        ):

            recurrent_parts.append(
                recurrent_mean
            )


        # ---------------------------------------------------------------------
        # Patient context
        # ---------------------------------------------------------------------

        context = (

            self.context_encoder(

                torch.cat(

                    [

                        *recurrent_parts,

                        static_encoded,

                        summary_encoded,

                        age_at_landmark
                        .unsqueeze(
                            1
                        ),

                        landmark_normalized
                        .unsqueeze(
                            1
                        ),

                    ],

                    dim=1,
                )
            )
        )


        # ---------------------------------------------------------------------
        # 未来10个半年区间
        # ---------------------------------------------------------------------

        horizon_index = torch.arange(

            self.future_n,

            device=
            dynamic_sequence.device,
        )


        horizon_embedding = (

            self.horizon_embedding(
                horizon_index
            )
            .unsqueeze(
                0
            )
            .expand(

                batch_n,

                -1,

                -1,
            )
        )


        context_expanded = (

            context
            .unsqueeze(
                1
            )
            .expand(

                -1,

                self.future_n,

                -1,
            )
        )


        head_input = torch.cat(

            [

                context_expanded,

                horizon_embedding,

            ],

            dim=2,
        )


        logits = (

            self.hazard_head(
                head_input
            )
            .squeeze(
                2
            )
        )


        return (

            logits

            + self.interval_bias
            .unsqueeze(
                0
            )
        )


# =============================================================================
# 13. 建立模型
# =============================================================================

def build_model(

    parameters: dict[str, Any],

    device: torch.device,

):


    model = HybridAttentionLSTMSurvival(

        dynamic_n=
        ENHANCED_DYNAMIC_N,

        static_n=
        STATIC_N,

        hidden_size=int(
            parameters[
                "hidden_size"
            ]
        ),

        num_layers=int(
            parameters[
                "num_layers"
            ]
        ),

        dropout=float(
            parameters[
                "dropout"
            ]
        ),

        projection_size=int(
            parameters[
                "projection_size"
            ]
        ),

        bidirectional=bool(
            parameters[
                "bidirectional"
            ]
        ),

        pooling_mode=str(
            parameters[
                "pooling_mode"
            ]
        ),

        static_hidden=int(
            parameters[
                "static_hidden"
            ]
        ),

        summary_hidden=int(
            parameters[
                "summary_hidden"
            ]
        ),

        horizon_embed_dim=int(
            parameters[
                "horizon_embed_dim"
            ]
        ),

        future_n=
        FUTURE_N,
    )


    return model.to(
        device
    )


# =============================================================================
# 14. torch load
# =============================================================================

def torch_load_full(

    path,

    map_location="cpu",

):


    try:

        return torch.load(

            path,

            map_location=
            map_location,

            weights_only=False,
        )


    except TypeError:

        return torch.load(

            path,

            map_location=
            map_location,
        )


# =============================================================================
# 15. AMP
# =============================================================================

def autocast_context(
    device,
):


    if device.type == "cuda":

        return torch.autocast(

            device_type="cuda",

            dtype=
            torch.float16,

            enabled=True,
        )


    return nullcontext()


# =============================================================================
# 16. 单snapshot预测
# =============================================================================

def predict_snapshot(

    model,

    enhanced_dynamic,

    static_baseline,

    age_at_landmark,

    batch_size,

    device,

):


    model.eval()


    predictions = []


    with torch.no_grad():


        for start in range(

            0,

            LONG_N,

            int(
                batch_size
            ),

        ):


            end = min(

                start
                + int(
                    batch_size
                ),

                LONG_N,
            )


            pairs = (

                sample_pairs[
                    start:end
                ]
            )


            patients = (

                pairs[
                    :,
                    0
                ]
                .astype(
                    np.int64
                )
            )


            landmark_idx = (

                pairs[
                    :,
                    1
                ]
                .astype(
                    np.int64
                )
            )


            dynamic_tensor = torch.as_tensor(

                np.asarray(

                    enhanced_dynamic[
                        patients
                    ],

                    dtype=np.float32,
                ),

                dtype=torch.float32,

                device=device,
            )


            row_mask_tensor = torch.as_tensor(

                np.asarray(

                    sequence_row_mask[
                        patients
                    ],

                    dtype=bool,
                ),

                dtype=torch.bool,

                device=device,
            )


            static_tensor = torch.as_tensor(

                np.asarray(

                    static_baseline[
                        patients
                    ],

                    dtype=np.float32,
                ),

                dtype=torch.float32,

                device=device,
            )


            age_tensor = torch.as_tensor(

                np.asarray(

                    age_at_landmark[

                        patients,

                        landmark_idx,

                    ],

                    dtype=np.float32,
                ),

                dtype=torch.float32,

                device=device,
            )


            landmark_norm_tensor = torch.as_tensor(

                LANDMARK_NORMALIZED[
                    landmark_idx
                ],

                dtype=torch.float32,

                device=device,
            )


            landmark_bin_tensor = torch.as_tensor(

                LANDMARK_BINS[
                    landmark_idx
                ],

                dtype=torch.long,

                device=device,
            )


            with autocast_context(
                device
            ):


                logits = model(

                    dynamic_sequence=
                    dynamic_tensor,

                    row_mask=
                    row_mask_tensor,

                    static_baseline=
                    static_tensor,

                    age_at_landmark=
                    age_tensor,

                    landmark_normalized=
                    landmark_norm_tensor,

                    landmark_bin=
                    landmark_bin_tensor,
                )


            hazard = (

                torch.sigmoid(
                    logits.float()
                )
                .cpu()
                .numpy()
                .astype(
                    np.float32
                )
            )


            predictions.append(
                hazard
            )


            del (
                dynamic_tensor,
                row_mask_tensor,
                static_tensor,
                age_tensor,
                landmark_norm_tensor,
                landmark_bin_tensor,
                logits,
            )


    result = np.concatenate(

        predictions,

        axis=0,

    ).astype(
        np.float32
    )


    if result.shape != (

        LONG_N,

        FUTURE_N,

    ):

        raise ValueError(

            "snapshot hazard形状错误："

            f"{result.shape}"
        )


    if not np.isfinite(
        result
    ).all():

        raise ValueError(
            "snapshot预测存在NaN/Inf。"
        )


    return result


# =============================================================================
# 17. Device
# =============================================================================

DEVICE = torch.device(

    "cuda"

    if torch.cuda.is_available()

    else "cpu"
)


print()
print(
    "预测设备：",
    DEVICE
)


# =============================================================================
# 18. 五折预测
# =============================================================================

fold_raw_hazard_long = np.full(

    (

        N_FOLDS,

        LONG_N,

        FUTURE_N,

    ),

    np.nan,

    dtype=np.float32,
)


fold_audit_rows = []

checkpoint_audit_rows = []


reference_parameters = None

reference_enhanced_names = None

reference_static_names = None

reference_feature_names = None


for fold_id in range(
    N_FOLDS
):


    print()
    print(
        "=" * 100
    )

    print(
        f"开始 Fold {fold_id}"
    )

    print(
        "=" * 100
    )


    fold_step4_dir = (

        STEP4
        / f"fold_{fold_id}"
    )


    # =========================================================================
    # 18.1 当前fold自己的preprocessor
    # =========================================================================

    preprocessor = joblib.load(

        fold_step4_dir
        / "preprocessor.joblib"
    )


    fold_feature_names = (

        pd.read_csv(

            fold_step4_dir
            / "feature_names.csv",

            encoding="utf-8-sig",

        )[
            "feature_name"
        ]
        .astype(
            str
        )
        .tolist()
    )


    if len(
        fold_feature_names
    ) != BASE_FEATURE_N:

        raise ValueError(

            f"Fold {fold_id}"
            " feature_names不是57项。"
        )


    if list(

        preprocessor[
            "final_feature_names"
        ]

    ) != fold_feature_names:

        raise ValueError(

            f"Fold {fold_id}"
            " preprocessor与feature_names.csv顺序不一致。"
        )


    # -------------------------------------------------------------------------
    # 所有fold在正式Step4中应具有同样的最终特征名称顺序
    # -------------------------------------------------------------------------

    if reference_feature_names is None:

        reference_feature_names = (
            fold_feature_names
        )

    else:

        if (
            fold_feature_names
            != reference_feature_names
        ):

            raise ValueError(
                "五折57维特征顺序不一致。"
            )


    # =========================================================================
    # 18.2 重庆数据使用当前fold预处理器
    # =========================================================================

    X_fold = (

        transform_external_with_fold_preprocessor(
            preprocessor
        )
    )


    feature_to_index = {

        name: index

        for (
            index,
            name
        ) in enumerate(
            fold_feature_names
        )
    }


    # =========================================================================
    # 18.3 16维static
    # =========================================================================

    onehot_static = [

        name

        for name
        in fold_feature_names

        if (

            name.startswith(
                "Sex_"
            )

            or name.startswith(
                "Marriage_"
            )

            or name.startswith(
                "Course_"
            )

            or name.startswith(
                "WHOstage_"
            )
        )
    ]


    static_names = [

        "BMI",

        "Oppinfection",

        *onehot_static,
    ]


    if len(
        static_names
    ) != STATIC_N:

        raise ValueError(

            f"Fold {fold_id}"
            f" static数量={len(static_names)}，"
            "应为16。"
        )


    static_indices = np.asarray(

        [

            feature_to_index[
                name
            ]

            for name in static_names
        ],

        dtype=np.int64,
    )


    static_baseline = np.asarray(

        X_fold[

            patient_rows,

            first_observed_step,

            :,

        ][
            :,
            static_indices
        ],

        dtype=np.float32,
    )


    if static_baseline.shape != (

        N,

        STATIC_N,

    ):

        raise ValueError(
            "static_baseline形状错误。"
        )


    # =========================================================================
    # 18.4 Fold-specific Age标准化
    # =========================================================================

    scaler = preprocessor[
        "scaler"
    ]


    age_mean = float(

        scaler.mean_[
            AGE_INDEX
        ]
    )


    age_scale = float(

        scaler.scale_[
            AGE_INDEX
        ]
    )


    if (

        not np.isfinite(
            age_mean
        )

        or not np.isfinite(
            age_scale
        )

        or age_scale <= 0

    ):

        raise ValueError(
            f"Fold {fold_id} Age scaler异常。"
        )


    age_at_landmark_raw = (

        baseline_age_raw[
            :,
            None
        ]

        + LANDMARK_MONTHS[
            None,
            :
        ]
        / 12.0

    ).astype(
        np.float32
    )


    age_at_landmark = (

        (

            age_at_landmark_raw

            - age_mean

        )

        / age_scale

    ).astype(
        np.float32
    )


    # =========================================================================
    # 18.5 当前fold 40维动态变量
    # =========================================================================

    required_names = set(

        [
            "Age",
            *static_names,
            *DYNAMIC_FEATURES,
        ]
    )


    missing = sorted(

        required_names

        - set(
            fold_feature_names
        )
    )


    if missing:

        raise ValueError(

            f"Fold {fold_id}"
            f"缺少模型变量：{missing}"
        )


    dynamic_indices = np.asarray(

        [

            feature_to_index[
                name
            ]

            for name
            in DYNAMIC_FEATURES
        ],

        dtype=np.int64,
    )


    base_dynamic = np.asarray(

        X_fold[
            :,
            :,
            dynamic_indices
        ],

        dtype=np.float32,
    )


    if base_dynamic.shape != (

        N,

        HISTORY_N,

        BASE_DYNAMIC_N,

    ):

        raise ValueError(
            "40维动态输入形状错误。"
        )


    # =========================================================================
    # 18.6 当前fold 85维增强动态输入
    # =========================================================================

    enhanced_dynamic = (

        build_enhanced_dynamic(
            base_dynamic
        )
    )


    if not np.isfinite(

        static_baseline

    ).all():

        raise ValueError(

            f"Fold {fold_id}"
            " static存在NaN/Inf。"
        )


    if not np.isfinite(

        age_at_landmark

    ).all():

        raise ValueError(

            f"Fold {fold_id}"
            " Age存在NaN/Inf。"
        )


    # =========================================================================
    # 18.7 保存当前fold外部输入，便于审计
    # =========================================================================

    fold_out = (

        OUT
        / f"fold_{fold_id}"
    )


    fold_out.mkdir(

        parents=True,

        exist_ok=True,
    )


    np.save(

        fold_out
        / "X_external_57.npy",

        X_fold,
    )


    np.save(

        fold_out
        / "enhanced_dynamic_85.npy",

        enhanced_dynamic,
    )


    np.save(

        fold_out
        / "static_baseline_16.npy",

        static_baseline,
    )


    np.save(

        fold_out
        / "age_at_landmark_standardized.npy",

        age_at_landmark,
    )


    pd.DataFrame({

        "feature_index":
            np.arange(
                BASE_FEATURE_N
            ),

        "feature_name":
            fold_feature_names,

    }).to_csv(

        fold_out
        / "feature_names.csv",

        index=False,

        encoding="utf-8-sig",
    )


    pd.DataFrame({

        "feature_index":
            np.arange(
                ENHANCED_DYNAMIC_N
            ),

        "feature_name":
            ENHANCED_DYNAMIC_NAMES,

    }).to_csv(

        fold_out
        / "enhanced_dynamic_feature_names.csv",

        index=False,

        encoding="utf-8-sig",
    )


    pd.DataFrame({

        "feature_index":
            np.arange(
                STATIC_N
            ),

        "feature_name":
            static_names,

    }).to_csv(

        fold_out
        / "static_feature_names.csv",

        index=False,

        encoding="utf-8-sig",
    )


    # =========================================================================
    # 18.8 找当前fold checkpoint
    # =========================================================================

    checkpoint_dir = (

        STEP10E
        / f"fold_{fold_id}"
    )


    checkpoint_paths = sorted(

        checkpoint_dir.glob(
            "seed_*_snapshot_ensemble.pt"
        )
    )


    if not checkpoint_paths:

        raise FileNotFoundError(

            f"Fold {fold_id}"
            "没有seed snapshot checkpoint。"
        )


    seed_hazards = []


    fold_reference_parameters = None
    fold_reference_dynamic_names = None
    fold_reference_static_names = None


    for checkpoint_path in (
        checkpoint_paths
    ):


        checkpoint = torch_load_full(

            checkpoint_path,

            map_location="cpu",
        )


        checkpoint_fold = int(

            checkpoint[
                "fold_id"
            ]
        )


        if checkpoint_fold != fold_id:

            raise ValueError(

                f"checkpoint fold错误："
                f"{checkpoint_path}"
            )


        seed = int(

            checkpoint[
                "seed"
            ]
        )


        parameters = dict(

            checkpoint[
                "parameters"
            ]
        )


        checkpoint_dynamic_names = list(

            checkpoint[
                "feature_names"
            ][
                "enhanced_dynamic"
            ]
        )


        checkpoint_static_names = list(

            checkpoint[
                "feature_names"
            ][
                "static"
            ]
        )


        snapshot_states = list(

            checkpoint[
                "snapshot_state_dicts"
            ]
        )


        snapshot_epochs = list(

            checkpoint[
                "snapshot_epochs"
            ]
        )


        # ---------------------------------------------------------------------
        # 85维顺序必须完全一致
        # ---------------------------------------------------------------------

        if (
            checkpoint_dynamic_names
            != ENHANCED_DYNAMIC_NAMES
        ):

            raise ValueError(

                f"Fold {fold_id} Seed {seed}："
                "85维变量顺序与重庆当前构建不一致。"
            )


        # ---------------------------------------------------------------------
        # static顺序必须完全一致
        # ---------------------------------------------------------------------

        if (
            checkpoint_static_names
            != static_names
        ):

            raise ValueError(

                f"Fold {fold_id} Seed {seed}："
                "16维static变量顺序不一致。"
            )


        if len(
            snapshot_states
        ) == 0:

            raise ValueError(

                f"Fold {fold_id} Seed {seed}"
                "没有snapshot。"
            )


        # ---------------------------------------------------------------------
        # fold内部不同seed结构一致
        # ---------------------------------------------------------------------

        if fold_reference_parameters is None:


            fold_reference_parameters = dict(
                parameters
            )

            fold_reference_dynamic_names = list(
                checkpoint_dynamic_names
            )

            fold_reference_static_names = list(
                checkpoint_static_names
            )


        else:


            if (
                parameters
                != fold_reference_parameters
            ):

                raise ValueError(

                    f"Fold {fold_id}"
                    "不同seed模型参数不一致。"
                )


            if (
                checkpoint_dynamic_names
                != fold_reference_dynamic_names
            ):

                raise ValueError(
                    "同fold不同seed动态顺序不一致。"
                )


            if (
                checkpoint_static_names
                != fold_reference_static_names
            ):

                raise ValueError(
                    "同fold不同seed静态顺序不一致。"
                )


        # ---------------------------------------------------------------------
        # 五折模型的超参数应相同
        # ---------------------------------------------------------------------

        if reference_parameters is None:


            reference_parameters = dict(
                parameters
            )

            reference_enhanced_names = list(
                checkpoint_dynamic_names
            )

            reference_static_names = list(
                checkpoint_static_names
            )


        else:


            if (
                parameters
                != reference_parameters
            ):

                raise ValueError(
                    "五折最终模型超参数不一致。"
                )


        batch_size = int(

            parameters.get(
                "batch_size",
                256,
            )
        )


        print(

            f"Fold {fold_id} | "
            f"Seed {seed} | "
            f"snapshot数={len(snapshot_states)} | "
            f"epochs={snapshot_epochs}"
        )


        snapshot_hazards = []


        # =====================================================================
        # snapshot ensemble
        # =====================================================================

        for (
            snapshot_position,
            state_dict
        ) in enumerate(
            snapshot_states
        ):


            model = build_model(

                parameters,

                DEVICE,
            )


            model.load_state_dict(

                state_dict,

                strict=True,
            )


            model.eval()


            hazard = predict_snapshot(

                model=
                model,

                enhanced_dynamic=
                enhanced_dynamic,

                static_baseline=
                static_baseline,

                age_at_landmark=
                age_at_landmark,

                batch_size=
                batch_size,

                device=
                DEVICE,
            )


            snapshot_hazards.append(
                hazard
            )


            del model


            gc.collect()


            if DEVICE.type == "cuda":

                torch.cuda.empty_cache()


        # =====================================================================
        # 同seed的snapshot hazard直接平均
        # =====================================================================

        seed_hazard = np.mean(

            np.stack(

                snapshot_hazards,

                axis=0,
            ),

            axis=0,

            dtype=np.float32,

        ).astype(
            np.float32
        )


        seed_hazards.append(
            seed_hazard
        )


        checkpoint_audit_rows.append({

            "fold_id":
                fold_id,

            "seed":
                seed,

            "checkpoint":
                str(
                    checkpoint_path
                ),

            "snapshot_n":
                int(
                    len(
                        snapshot_states
                    )
                ),

            "snapshot_epochs":
                ",".join(
                    map(
                        str,
                        snapshot_epochs
                    )
                ),

            "batch_size":
                batch_size,
        })


        del (
            checkpoint,
            snapshot_hazards,
            seed_hazard,
        )


        gc.collect()


    # =========================================================================
    # 当前fold的两个seed等权平均
    # =========================================================================

    fold_hazard = np.mean(

        np.stack(

            seed_hazards,

            axis=0,
        ),

        axis=0,

        dtype=np.float32,

    ).astype(
        np.float32
    )


    fold_raw_hazard_long[
        fold_id
    ] = fold_hazard


    np.save(

        fold_out
        / "external_raw_hazard_long.npy",

        fold_hazard,
    )


    fold_audit_rows.append({

        "fold_id":
            fold_id,

        "patient_n":
            N,

        "valid_origin_n":
            LONG_N,

        "age_mean":
            age_mean,

        "age_scale":
            age_scale,

        "seed_n":
            len(
                seed_hazards
            ),

        "checkpoint_n":
            len(
                checkpoint_paths
            ),

        "mean_raw_hazard":
            float(
                fold_hazard.mean()
            ),

        "min_raw_hazard":
            float(
                fold_hazard.min()
            ),

        "max_raw_hazard":
            float(
                fold_hazard.max()
            ),
    })


    print(

        f"Fold {fold_id}完成 | "

        f"seed数={len(seed_hazards)} | "

        f"mean hazard="
        f"{fold_hazard.mean():.8f}"
    )


    # -------------------------------------------------------------------------
    # 释放fold输入
    # -------------------------------------------------------------------------

    del (
        preprocessor,
        X_fold,
        base_dynamic,
        enhanced_dynamic,
        static_baseline,
        age_at_landmark,
        seed_hazards,
        fold_hazard,
    )


    gc.collect()


    if DEVICE.type == "cuda":

        torch.cuda.empty_cache()


# =============================================================================
# 19. Fold-level预测完整性
# =============================================================================

if not np.isfinite(

    fold_raw_hazard_long

).all():

    raise ValueError(
        "fold_raw_hazard_long存在NaN/Inf。"
    )


if np.any(

    fold_raw_hazard_long <= 0

) or np.any(

    fold_raw_hazard_long >= 1

):

    raise ValueError(
        "fold hazard不在(0,1)。"
    )


# =============================================================================
# 20. 五折ensemble
#
# 五个fold模型等权平均raw discrete hazard
# =============================================================================

raw_hazard_long = np.mean(

    fold_raw_hazard_long,

    axis=0,

    dtype=np.float32,

).astype(
    np.float32
)


if raw_hazard_long.shape != (

    LONG_N,

    FUTURE_N,

):

    raise ValueError(
        "ensemble hazard形状错误。"
    )


# =============================================================================
# 21. 读取Step11最终开发OOF校准器
# =============================================================================

calibration_table = pd.read_csv(

    CALIBRATION_FILE,

    encoding="utf-8-sig",
)


required_columns = {

    "model",

    "fit_scope",

    "heldout_fold",

    "landmark_month",

    "interval_position",

    "alpha_interval",

    "beta_common_slope",
}


missing_columns = (

    required_columns

    - set(
        calibration_table.columns
    )
)


if missing_columns:

    raise ValueError(

        "校准参数文件缺少："

        f"{sorted(missing_columns)}"
    )


def load_final_calibrator(

    landmark_month,

):


    sub = calibration_table.loc[

        (
            calibration_table[
                "model"
            ]
            == "LSTM-v2"
        )

        & (
            calibration_table[
                "fit_scope"
            ]
            == "final_all_development_oof"
        )

        & (
            pd.to_numeric(

                calibration_table[
                    "heldout_fold"
                ],

                errors="coerce",

            )
            == -1
        )

        & np.isclose(

            pd.to_numeric(

                calibration_table[
                    "landmark_month"
                ],

                errors="coerce",
            ),

            float(
                landmark_month
            ),
        )

    ].copy()


    sub = sub.sort_values(
        "interval_position"
    )


    if len(
        sub
    ) != FUTURE_N:

        raise ValueError(

            f"Landmark {landmark_month}月"
            " final calibrator不是10行。"
        )


    expected_position = np.arange(
        FUTURE_N
    )


    if not np.array_equal(

        sub[
            "interval_position"
        ]
        .to_numpy(
            int
        ),

        expected_position,

    ):

        raise ValueError(
            "calibrator区间顺序错误。"
        )


    alpha = (

        sub[
            "alpha_interval"
        ]
        .to_numpy(
            dtype=np.float64
        )
    )


    beta_values = (

        sub[
            "beta_common_slope"
        ]
        .to_numpy(
            dtype=np.float64
        )
    )


    if not np.allclose(

        beta_values,

        beta_values[0],

        atol=1e-10,

        rtol=0,

    ):

        raise ValueError(
            "同Landmark beta不一致。"
        )


    beta = float(
        beta_values[0]
    )


    if (

        not np.isfinite(
            alpha
        ).all()

        or not np.isfinite(
            beta
        )

    ):

        raise ValueError(
            "calibrator存在非有限参数。"
        )


    return (
        alpha,
        beta,
    )


# =============================================================================
# 22. 校准函数
# =============================================================================

def apply_calibrator(

    hazard,

    alpha,

    beta,

):


    probability = np.clip(

        np.asarray(

            hazard,

            dtype=np.float64,
        ),

        EPS,

        1.0 - EPS,
    )


    raw_logit = (

        np.log(
            probability
        )

        - np.log1p(
            -probability
        )
    )


    calibrated_logit = (

        alpha[
            None,
            :
        ]

        + float(
            beta
        )

        * raw_logit
    )


    # 数值稳定
    calibrated_logit = np.clip(

        calibrated_logit,

        -50.0,

        50.0,
    )


    calibrated = (

        1.0

        / (

            1.0

            + np.exp(
                -calibrated_logit
            )
        )
    )


    return np.clip(

        calibrated,

        EPS,

        1.0 - EPS,

    ).astype(
        np.float32
    )


# =============================================================================
# 23. Primary calibration
#
# 五折raw hazard ensemble → final development OOF calibrator
# =============================================================================

calibrated_hazard_long = np.full(

    (

        LONG_N,

        FUTURE_N,

    ),

    np.nan,

    dtype=np.float32,
)


calibrator_rows = []


for (
    landmark_index,
    landmark_month
) in enumerate(
    LANDMARK_MONTHS
):


    rows = np.where(

        landmark_index_long

        == landmark_index

    )[0]


    (
        alpha,
        beta
    ) = load_final_calibrator(

        int(
            landmark_month
        )
    )


    calibrated_hazard_long[
        rows
    ] = apply_calibrator(

        raw_hazard_long[
            rows
        ],

        alpha,

        beta,
    )


    calibrator_rows.append({

        "landmark_index":
            landmark_index,

        "landmark_month":
            int(
                landmark_month
            ),

        "landmark_year":
            float(
                landmark_month
                / 12.0
            ),

        "record_n":
            int(
                len(
                    rows
                )
            ),

        "beta_common_slope":
            beta,

        **{

            f"alpha_{int(month)}m":
                float(
                    alpha[
                        position
                    ]
                )

            for (
                position,
                month
            ) in enumerate(
                FUTURE_END_MONTHS
            )
        },
    })


if not np.isfinite(

    calibrated_hazard_long

).all():

    raise ValueError(
        "calibrated_hazard存在NaN/Inf。"
    )


# =============================================================================
# 24. 校准顺序敏感性
#
# 每fold先使用同一个final calibrator，再平均
# =============================================================================

fold_calibrated_hazard_long = np.full_like(

    fold_raw_hazard_long,

    np.nan,

    dtype=np.float32,
)


for (
    landmark_index,
    landmark_month
) in enumerate(
    LANDMARK_MONTHS
):


    rows = np.where(

        landmark_index_long

        == landmark_index

    )[0]


    (
        alpha,
        beta
    ) = load_final_calibrator(

        int(
            landmark_month
        )
    )


    for fold_id in range(
        N_FOLDS
    ):


        fold_calibrated_hazard_long[

            fold_id,

            rows,

            :

        ] = apply_calibrator(

            fold_raw_hazard_long[

                fold_id,

                rows,

                :
            ],

            alpha,

            beta,
        )


calibrated_fold_then_mean_hazard = np.mean(

    fold_calibrated_hazard_long,

    axis=0,

    dtype=np.float32,

).astype(
    np.float32
)


# =============================================================================
# 25. Hazard → Survival / Risk
# =============================================================================

def hazard_to_survival_risk(
    hazard,
):


    hazard64 = np.clip(

        np.asarray(

            hazard,

            dtype=np.float64,
        ),

        EPS,

        1.0 - EPS,
    )


    survival = np.cumprod(

        1.0 - hazard64,

        axis=1,
    )


    risk = (

        1.0

        - survival
    )


    return (

        survival.astype(
            np.float32
        ),

        risk.astype(
            np.float32
        ),
    )


(
    raw_survival_long,
    raw_risk_long,

) = hazard_to_survival_risk(

    raw_hazard_long
)


(
    calibrated_survival_long,
    calibrated_risk_long,

) = hazard_to_survival_risk(

    calibrated_hazard_long
)


(
    sensitivity_survival_long,
    sensitivity_risk_long,

) = hazard_to_survival_risk(

    calibrated_fold_then_mean_hazard
)


# =============================================================================
# 26. 单调性QC
# =============================================================================

if (

    np.diff(

        calibrated_risk_long,

        axis=1

    )

    < -1e-7

).any():

    raise ValueError(
        "累计CKD风险不是单调递增。"
    )


if (

    np.diff(

        calibrated_survival_long,

        axis=1

    )

    > 1e-7

).any():

    raise ValueError(
        "Survival不是单调递减。"
    )


# =============================================================================
# 27. 五折模型之间的风险稳定性
# =============================================================================

fold_raw_risk_long = np.empty_like(

    fold_raw_hazard_long,

    dtype=np.float32,
)


for fold_id in range(
    N_FOLDS
):


    _, fold_risk = (

        hazard_to_survival_risk(

            fold_raw_hazard_long[
                fold_id
            ]
        )
    )


    fold_raw_risk_long[
        fold_id
    ] = fold_risk


fold_5y_risk_sd = np.std(

    fold_raw_risk_long[
        :,
        :,
        9
    ],

    axis=0,

    ddof=1,

).astype(
    np.float32
)


# =============================================================================
# 28. 校准顺序差异
# =============================================================================

calibration_order_difference = np.abs(

    calibrated_risk_long

    - sensitivity_risk_long
)


calibration_order_mean_abs_diff = float(

    np.mean(
        calibration_order_difference
    )
)


calibration_order_max_abs_diff = float(

    np.max(
        calibration_order_difference
    )
)


# =============================================================================
# 29. 转回 patient × landmark × horizon
# =============================================================================

def long_to_full(
    matrix_long,
):


    output = np.full(

        (

            N,

            LANDMARK_N,

            FUTURE_N,

        ),

        np.nan,

        dtype=np.float32,
    )


    output[

        local_patient_idx_long,

        landmark_index_long,

        :

    ] = matrix_long


    return output


raw_hazard_full = long_to_full(
    raw_hazard_long
)


raw_risk_full = long_to_full(
    raw_risk_long
)


calibrated_hazard_full = long_to_full(
    calibrated_hazard_long
)


calibrated_survival_full = long_to_full(
    calibrated_survival_long
)


calibrated_risk_full = long_to_full(
    calibrated_risk_long
)


# =============================================================================
# 30. 保存核心预测
# =============================================================================

np.save(

    OUT
    / "external_fold_raw_hazard_long.npy",

    fold_raw_hazard_long,
)


np.save(

    OUT
    / "external_fold_raw_risk_long.npy",

    fold_raw_risk_long,
)


np.save(

    OUT
    / "external_raw_hazard_long.npy",

    raw_hazard_long,
)


np.save(

    OUT
    / "external_raw_survival_long.npy",

    raw_survival_long,
)


np.save(

    OUT
    / "external_raw_risk_long.npy",

    raw_risk_long,
)


np.save(

    OUT
    / "external_calibrated_hazard_long.npy",

    calibrated_hazard_long,
)


np.save(

    OUT
    / "external_calibrated_survival_long.npy",

    calibrated_survival_long,
)


np.save(

    OUT
    / "external_calibrated_risk_long.npy",

    calibrated_risk_long,
)


np.save(

    OUT
    / "external_raw_hazard_full.npy",

    raw_hazard_full,
)


np.save(

    OUT
    / "external_raw_risk_full.npy",

    raw_risk_full,
)


np.save(

    OUT
    / "external_calibrated_hazard_full.npy",

    calibrated_hazard_full,
)


np.save(

    OUT
    / "external_calibrated_survival_full.npy",

    calibrated_survival_full,
)


np.save(

    OUT
    / "external_calibrated_risk_full.npy",

    calibrated_risk_full,
)


# =============================================================================
# 31. 患者-Landmark预测表
# =============================================================================

prediction_table = (

    metadata.copy()
)


for (
    future_index,
    future_month
) in enumerate(
    FUTURE_END_MONTHS
):


    prediction_table[

        f"predicted_risk_{int(future_month)}m"

    ] = (

        calibrated_risk_long[
            :,
            future_index
        ]
    )


prediction_table[

    "predicted_risk_1y"

] = calibrated_risk_long[
    :,
    1
]


prediction_table[

    "predicted_risk_3y"

] = calibrated_risk_long[
    :,
    5
]


prediction_table[

    "predicted_risk_5y"

] = calibrated_risk_long[
    :,
    9
]


prediction_table[

    "raw_predicted_risk_5y"

] = raw_risk_long[
    :,
    9
]


prediction_table[

    "five_fold_raw_5y_risk_sd"

] = fold_5y_risk_sd


prediction_table.to_csv(

    OUT
    / "重庆_LSTM_v2_FINAL_patient_landmark_predictions.csv",

    index=False,

    encoding="utf-8-sig",
)


# =============================================================================
# 32. Landmark预测汇总
# =============================================================================

summary_rows = []


for (
    landmark_index,
    landmark_month
) in enumerate(
    LANDMARK_MONTHS
):


    rows = np.where(

        landmark_index_long

        == landmark_index

    )[0]


    risk_1y = calibrated_risk_long[
        rows,
        1
    ]


    risk_3y = calibrated_risk_long[
        rows,
        5
    ]


    risk_5y = calibrated_risk_long[
        rows,
        9
    ]


    summary_rows.append({

        "landmark_month":
            int(
                landmark_month
            ),

        "landmark_year":
            float(
                landmark_month
                / 12.0
            ),

        "prediction_origin_n":
            int(
                len(
                    rows
                )
            ),

        "event_within_60m_n":
            int(

                metadata.iloc[
                    rows
                ][
                    "event_within_60m"
                ].sum()
            ),

        "mean_predicted_1y_risk":
            float(
                np.mean(
                    risk_1y
                )
            ),

        "median_predicted_1y_risk":
            float(
                np.median(
                    risk_1y
                )
            ),

        "mean_predicted_3y_risk":
            float(
                np.mean(
                    risk_3y
                )
            ),

        "median_predicted_3y_risk":
            float(
                np.median(
                    risk_3y
                )
            ),

        "mean_predicted_5y_risk":
            float(
                np.mean(
                    risk_5y
                )
            ),

        "median_predicted_5y_risk":
            float(
                np.median(
                    risk_5y
                )
            ),

        "mean_fold_5y_prediction_sd":
            float(

                np.mean(

                    fold_5y_risk_sd[
                        rows
                    ]
                )
            ),
    })


prediction_summary = pd.DataFrame(
    summary_rows
)


prediction_summary.to_csv(

    OUT
    / "重庆_LSTM_v2_FINAL_prediction_summary_by_landmark.csv",

    index=False,

    encoding="utf-8-sig",
)


# =============================================================================
# 33. 保存fold/checkpoint/calibrator审计
# =============================================================================

fold_audit = pd.DataFrame(
    fold_audit_rows
)


checkpoint_audit = pd.DataFrame(
    checkpoint_audit_rows
)


calibrator_audit = pd.DataFrame(
    calibrator_rows
)


fold_audit.to_csv(

    OUT
    / "fold_specific_preprocessing_prediction_audit.csv",

    index=False,

    encoding="utf-8-sig",
)


checkpoint_audit.to_csv(

    OUT
    / "checkpoint_audit.csv",

    index=False,

    encoding="utf-8-sig",
)


calibrator_audit.to_csv(

    OUT
    / "final_calibrator_audit.csv",

    index=False,

    encoding="utf-8-sig",
)


# =============================================================================
# 34. 与旧Step7结果比较
#
# 旧Step7使用full-development preprocessor给全部五折模型
# 新Step7R使用每个fold自己的preprocessor
#
# 这里只做审计比较，不影响任何新预测
# =============================================================================

old_comparison = None


old_raw_file = (

    OLD_STEP7
    / "external_raw_hazard_long.npy"
)


old_calibrated_file = (

    OLD_STEP7
    / "external_calibrated_risk_long.npy"
)


if (

    old_raw_file.exists()

    and old_calibrated_file.exists()

):


    old_raw_hazard = np.load(
        old_raw_file
    ).astype(
        np.float32
    )


    old_calibrated_risk = np.load(
        old_calibrated_file
    ).astype(
        np.float32
    )


    if (

        old_raw_hazard.shape
        == raw_hazard_long.shape

        and old_calibrated_risk.shape
        == calibrated_risk_long.shape

    ):


        raw_difference = np.abs(

            raw_hazard_long

            - old_raw_hazard
        )


        risk_difference = np.abs(

            calibrated_risk_long

            - old_calibrated_risk
        )


        risk_5y_difference = np.abs(

            calibrated_risk_long[
                :,
                9
            ]

            - old_calibrated_risk[
                :,
                9
            ]
        )


        old_comparison = pd.DataFrame({

            "metric": [

                "raw_hazard_mean_abs_difference",

                "raw_hazard_max_abs_difference",

                "calibrated_risk_all_horizons_mean_abs_difference",

                "calibrated_risk_all_horizons_max_abs_difference",

                "calibrated_5y_risk_mean_abs_difference",

                "calibrated_5y_risk_median_abs_difference",

                "calibrated_5y_risk_max_abs_difference",

            ],

            "value": [

                float(
                    np.mean(
                        raw_difference
                    )
                ),

                float(
                    np.max(
                        raw_difference
                    )
                ),

                float(
                    np.mean(
                        risk_difference
                    )
                ),

                float(
                    np.max(
                        risk_difference
                    )
                ),

                float(
                    np.mean(
                        risk_5y_difference
                    )
                ),

                float(
                    np.median(
                        risk_5y_difference
                    )
                ),

                float(
                    np.max(
                        risk_5y_difference
                    )
                ),
            ],
        })


        old_comparison.to_csv(

            OUT
            / "AUDIT_old_full_preprocessor_vs_FINAL_fold_specific.csv",

            index=False,

            encoding="utf-8-sig",
        )


# =============================================================================
# 35. Manifest
# =============================================================================

manifest = {

    "analysis":
        (
            "Chongqing geographic external validation "
            "LSTM-v2 FINAL corrected inference"
        ),

    "external_patient_n":
        int(
            N
        ),

    "valid_patient_landmark_n":
        int(
            LONG_N
        ),

    "fold_n":
        int(
            N_FOLDS
        ),

    "checkpoint_n":
        int(
            len(
                checkpoint_audit
            )
        ),

    "base_feature_n":
        BASE_FEATURE_N,

    "enhanced_dynamic_n":
        ENHANCED_DYNAMIC_N,

    "static_n":
        STATIC_N,

    "landmark_months":
        LANDMARK_MONTHS.tolist(),

    "future_end_months":
        FUTURE_END_MONTHS.tolist(),

    "external_preprocessing":
        (
            "each frozen fold model receives external data transformed "
            "with that fold's own training-fold-only StandardScaler "
            "and OneHotEncoder"
        ),

    "ensemble":
        (
            "hazard mean across snapshots within seed; "
            "hazard mean across seeds within fold; "
            "equal-weight raw-hazard mean across five folds"
        ),

    "calibration":
        (
            "Step11 frozen LSTM-v2 "
            "fit_scope=final_all_development_oof "
            "applied after five-fold raw-hazard ensemble"
        ),

    "external_outcome_used_for_training":
        False,

    "external_outcome_used_for_model_selection":
        False,

    "external_outcome_used_for_calibration":
        False,

    "calibration_order_mean_abs_difference":
        calibration_order_mean_abs_diff,

    "calibration_order_max_abs_difference":
        calibration_order_max_abs_diff,

    "model_parameters":
        reference_parameters,

    "warning":
        (
            "Old Step7 full-development-preprocessor predictions "
            "must not be used for final external validation."
        ),
}


with open(

    OUT
    / "step7R_FINAL_manifest.json",

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
# 36. 最终输出
# =============================================================================

print()
print(
    "=" * 110
)

print(
    "重庆外部验证 Step7R FINAL："
    "Fold-specific preprocessing + frozen LSTM-v2 ensemble 完成"
)

print(
    "=" * 110
)


print(
    "患者数：",
    N
)


print(
    "有效 patient-Landmark：",
    LONG_N
)


print(
    "五折模型：",
    N_FOLDS
)


print(
    "Seed checkpoint总数：",
    len(
        checkpoint_audit
    )
)


print()
print(
    "Fold-specific预测审计："
)


print(

    fold_audit
    .round(
        8
    )
    .to_string(
        index=False
    )
)


print()
print(
    "Final raw hazard范围：",
    float(
        raw_hazard_long.min()
    ),
    "~",
    float(
        raw_hazard_long.max()
    )
)


print(
    "Final calibrated hazard范围：",
    float(
        calibrated_hazard_long.min()
    ),
    "~",
    float(
        calibrated_hazard_long.max()
    )
)


print(
    "Final calibrated cumulative risk范围：",
    float(
        calibrated_risk_long.min()
    ),
    "~",
    float(
        calibrated_risk_long.max()
    )
)


print()
print(
    "校准顺序敏感性："
)


print(
    "平均绝对风险差：",
    calibration_order_mean_abs_diff
)


print(
    "最大绝对风险差：",
    calibration_order_max_abs_diff
)


print()
print(
    "Landmark最终预测汇总："
)


print(

    prediction_summary
    .round(
        6
    )
    .to_string(
        index=False
    )
)


if old_comparison is not None:


    print()
    print(
        "旧Step7 full-preprocessor "
        "vs 新Step7R fold-specific："
    )


    print(

        old_comparison
        .to_string(
            index=False
        )
    )


print()
print(
    "最终输出目录："
)


print(
    OUT
)


print(
    "=" * 110
)