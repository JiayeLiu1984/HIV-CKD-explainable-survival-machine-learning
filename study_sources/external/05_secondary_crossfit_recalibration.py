# -*- coding: utf-8 -*-

# =============================================================================
# 重庆独立地理外部验证
# Secondary analysis:
# Patient-level 5-fold cross-fitted intercept-only recalibration
#
# 目的：
#   保留原LSTM全部结构和风险排序信息，
#   仅更新重庆中心的absolute baseline risk。
#
# 重要：
#   这是 post hoc / exploratory secondary recalibration。
#
# Primary external validation:
#   仍然使用原始 frozen model 结果。
#
# Secondary analysis:
#   使用cross-fitted recalibrated prediction。
#
# 不：
#   - 重新训练LSTM
#   - 修改特征
#   - 修改模型超参数
#   - 用重庆选择模型
#
# Recalibration模型：
#
#   logit(h_new) = logit(h_frozen) + delta_L
#
# 每个Landmark只有一个delta_L，
# 10个未来半年hazard共享同一个delta。
#
# =============================================================================


from pathlib import Path
import warnings
import json

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib import font_manager

from sklearn.model_selection import StratifiedKFold

from scipy.optimize import minimize

from sksurv.metrics import brier_score
from sksurv.nonparametric import kaplan_meier_estimator
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
    / "重庆外部验证_secondary_crossfit_recalibration"
)


FIG_DIR = (
    OUT
    / "figures"
)


SOURCE_DIR = (
    OUT
    / "source_data"
)


OUT.mkdir(
    parents=True,
    exist_ok=True,
)

FIG_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

SOURCE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# =============================================================================
# 2. 输入文件
# =============================================================================

ORIGINAL_HAZARD_FILE = (
    STEP7R
    / "external_calibrated_hazard_long.npy"
)


ORIGINAL_RISK_FILE = (
    STEP7R
    / "external_calibrated_risk_long.npy"
)


METADATA_FILE = (
    STEP6
    / "super_landmark_external_metadata.csv"
)


FUTURE_EVENT_FILE = (
    STEP6
    / "external_future_event_long.npy"
)


FUTURE_AT_RISK_FILE = (
    STEP6
    / "external_future_at_risk_long.npy"
)


# =============================================================================
# 3. 固定参数
# =============================================================================

N_FOLDS = 5

RANDOM_SEED = 20260824


LANDMARK_MONTHS = np.asarray(
    [
        0,
        12,
        24,
        36,
        48,
        60,
    ],
    dtype=float,
)


LANDMARK_YEARS = (
    LANDMARK_MONTHS
    / 12.0
)


PRIMARY_LANDMARK_INDEX = [
    0,
    1,
    3,
    5,
]


PRIMARY_LANDMARK_YEARS = [
    0,
    1,
    3,
    5,
]


FUTURE_MONTHS = np.asarray(
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
    dtype=float,
)


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
    dtype=float,
)


REPORT_POSITIONS = {
    12: 1,
    36: 5,
    60: 9,
}


N_LANDMARK = 6
N_FUTURE = 10

EPS = 1e-7


# =============================================================================
# 4. DCA
#
# 外部人群实际5年事件风险约1%–3%
# 因此主secondary图放大0.5%–10%
#
# 原1%–50%仍可保留在原primary external validation supplement。
# =============================================================================

DCA_THRESHOLDS = np.linspace(
    0.005,
    0.10,
    250,
)


# =============================================================================
# 5. 绘图格式
# =============================================================================

ORIGINAL_COLOR = "#E15759"

RECAL_COLOR = "#4C78A8"

OBSERVED_COLOR = "#222222"


def choose_font():

    for name in [
        "Arial",
        "Helvetica",
        "Liberation Sans",
        "DejaVu Sans",
    ]:

        try:

            font_manager.findfont(
                name,
                fallback_to_default=False,
            )

            return name

        except ValueError:

            pass

    return "DejaVu Sans"


plt.style.use(
    "default"
)


plt.rcParams.update({

    "font.family":
        choose_font(),

    "font.size":
        11,

    "figure.facecolor":
        "white",

    "axes.facecolor":
        "white",

    "savefig.facecolor":
        "white",

    "axes.spines.top":
        False,

    "axes.spines.right":
        False,

    "axes.linewidth":
        0.8,

    "pdf.fonttype":
        42,

    "ps.fonttype":
        42,

    "savefig.dpi":
        600,
})


def style_axis(
    ax
):

    ax.grid(
        linestyle="--",
        linewidth=0.6,
        alpha=0.25,
    )

    ax.set_axisbelow(
        True
    )


def panel_label(
    ax,
    text
):

    ax.text(
        -0.10,
        1.07,
        text,
        transform=ax.transAxes,
        fontsize=14,
        fontweight="bold",
        va="top",
    )


def save_figure(
    fig,
    name
):

    fig.savefig(
        FIG_DIR
        / f"{name}.png",
        dpi=600,
        bbox_inches="tight",
        facecolor="white",
    )

    fig.savefig(
        FIG_DIR
        / f"{name}.pdf",
        bbox_inches="tight",
        facecolor="white",
    )

    plt.show()

    plt.close(
        fig
    )


# =============================================================================
# 6. 文件检查
# =============================================================================

required_files = [

    ORIGINAL_HAZARD_FILE,

    ORIGINAL_RISK_FILE,

    METADATA_FILE,

    FUTURE_EVENT_FILE,

    FUTURE_AT_RISK_FILE,

    STEP1
    / "development_idx.npy",

    STEP1
    / "event.npy",

    STEP1
    / "observed_time_month.npy",
]


missing = [
    str(path)
    for path in required_files
    if not path.exists()
]


if missing:

    raise FileNotFoundError(
        "缺少必要文件：\n"
        + "\n".join(
            missing
        )
    )


# =============================================================================
# 7. 读取重庆结果
# =============================================================================

original_hazard = np.load(
    ORIGINAL_HAZARD_FILE
).astype(
    np.float64
)


original_risk = np.load(
    ORIGINAL_RISK_FILE
).astype(
    np.float64
)


future_event = np.load(
    FUTURE_EVENT_FILE
).astype(
    np.float64
)


future_at_risk = np.load(
    FUTURE_AT_RISK_FILE
).astype(
    bool
)


metadata = pd.read_csv(
    METADATA_FILE,
    encoding="utf-8-sig",
    dtype={
        "ID": "string",
    },
)


LONG_N = len(
    metadata
)


# =============================================================================
# 8. QC
# =============================================================================

expected_shape = (
    LONG_N,
    N_FUTURE,
)


for name, array in [

    (
        "original_hazard",
        original_hazard,
    ),

    (
        "original_risk",
        original_risk,
    ),

    (
        "future_event",
        future_event,
    ),

    (
        "future_at_risk",
        future_at_risk,
    ),

]:

    if array.shape != expected_shape:

        raise ValueError(
            f"{name}形状错误："
            f"{array.shape}"
        )


if not np.isfinite(
    original_hazard
).all():

    raise ValueError(
        "original hazard存在NaN/Inf"
    )


if np.any(
    original_hazard <= 0
) or np.any(
    original_hazard >= 1
):

    raise ValueError(
        "original hazard不在(0,1)"
    )


if not np.isin(
    future_event,
    [
        0,
        1,
    ],
).all():

    raise ValueError(
        "future_event不是0/1"
    )


if np.any(
    (future_event > 0)
    &
    (~future_at_risk)
):

    raise ValueError(
        "事件标签位于at-risk mask之外"
    )


required_meta = [

    "local_patient_index",

    "landmark_index",

    "landmark_month",

    "analysis_time_month",

    "event_within_60m",

    "original_event",
]


missing_meta = [

    column

    for column in required_meta

    if column not in metadata.columns
]


if missing_meta:

    raise ValueError(
        f"metadata缺少字段："
        f"{missing_meta}"
    )


patient_index_long = (
    metadata[
        "local_patient_index"
    ]
    .to_numpy(
        int
    )
)


landmark_index_long = (
    metadata[
        "landmark_index"
    ]
    .to_numpy(
        int
    )
)


N_PATIENT = int(
    patient_index_long.max()
    + 1
)


# =============================================================================
# 9. 患者级事件标签
#
# 用于StratifiedKFold。
#
# 同一个患者的所有Landmark必须进入同一fold。
# =============================================================================

patient_event = np.zeros(
    N_PATIENT,
    dtype=int,
)


for patient in range(
    N_PATIENT
):

    rows = np.where(
        patient_index_long
        == patient
    )[0]

    if len(
        rows
    ) == 0:

        continue

    patient_event[
        patient
    ] = int(
        metadata.iloc[
            rows
        ][
            "original_event"
        ].max()
    )


print()
print(
    "重庆患者数：",
    N_PATIENT
)


print(
    "最终CKD事件患者：",
    int(
        patient_event.sum()
    )
)


# =============================================================================
# 10. 基础数学函数
# =============================================================================

def clip_probability(
    p
):

    return np.clip(
        np.asarray(
            p,
            dtype=float,
        ),
        EPS,
        1.0
        - EPS,
    )


def logit(
    p
):

    p = clip_probability(
        p
    )

    return (
        np.log(
            p
        )
        -
        np.log1p(
            -p
        )
    )


def expit(
    x
):

    x = np.asarray(
        x,
        dtype=float,
    )

    result = np.empty_like(
        x
    )

    positive = (
        x >= 0
    )

    result[
        positive
    ] = (
        1.0
        /
        (
            1.0
            +
            np.exp(
                -x[
                    positive
                ]
            )
        )
    )

    exp_x = np.exp(
        x[
            ~positive
        ]
    )

    result[
        ~positive
    ] = (
        exp_x
        /
        (
            1.0
            +
            exp_x
        )
    )

    return result


# =============================================================================
# 11. Intercept-only offset calibration
#
# MLE score equation:
#
#   sum[p_i(delta) - y_i] = 0
#
# 其中：
#   p_i(delta)=sigmoid(logit(h_i)+delta)
#
# 使用二分法，不依赖任何新的ML模型。
# =============================================================================

def fit_intercept_only(
    hazard,
    outcome,
):

    hazard = clip_probability(
        hazard
    )

    outcome = np.asarray(
        outcome,
        dtype=float,
    )


    if len(
        hazard
    ) != len(
        outcome
    ):

        raise ValueError(
            "hazard/outcome长度不一致"
        )


    event_n = float(
        outcome.sum()
    )


    if event_n <= 0:

        raise ValueError(
            "训练数据无事件，无法估计recalibration intercept"
        )


    if event_n >= len(
        outcome
    ):

        raise ValueError(
            "训练数据全部为事件，无法估计recalibration intercept"
        )


    offset = logit(
        hazard
    )


    def score(
        delta
    ):

        return float(
            np.sum(
                expit(
                    offset
                    + delta
                )
                -
                outcome
            )
        )


    lower = -15.0
    upper = 15.0


    if score(
        lower
    ) > 0:

        raise ValueError(
            "recalibration lower bound不足"
        )


    if score(
        upper
    ) < 0:

        raise ValueError(
            "recalibration upper bound不足"
        )


    for _ in range(
        120
    ):

        middle = (
            lower
            + upper
        ) / 2.0


        value = score(
            middle
        )


        if value > 0:

            upper = middle

        else:

            lower = middle


    return float(
        (
            lower
            + upper
        )
        / 2.0
    )


# =============================================================================
# 12. 5-fold patient-level cross-fitting
# =============================================================================

patient_fold = np.full(
    N_PATIENT,
    -1,
    dtype=int,
)


skf = StratifiedKFold(
    n_splits=N_FOLDS,
    shuffle=True,
    random_state=RANDOM_SEED,
)


patient_ids = np.arange(
    N_PATIENT
)


for fold_id, (
    train_patient_idx,
    test_patient_idx,
) in enumerate(
    skf.split(
        patient_ids,
        patient_event,
    )
):

    patient_fold[
        test_patient_idx
    ] = fold_id


if np.any(
    patient_fold < 0
):

    raise ValueError(
        "存在未分配fold的患者"
    )


pd.DataFrame({

    "local_patient_index":
        np.arange(
            N_PATIENT
        ),

    "original_event":
        patient_event,

    "recalibration_fold":
        patient_fold,

}).to_csv(
    OUT
    / "patient_recalibration_folds.csv",
    index=False,
    encoding="utf-8-sig",
)


# =============================================================================
# 13. Cross-fitted recalibration
# =============================================================================

oof_hazard = np.full_like(
    original_hazard,
    np.nan,
    dtype=float,
)


parameter_records = []


for fold_id in range(
    N_FOLDS
):

    test_patients = np.where(
        patient_fold
        == fold_id
    )[0]


    train_patients = np.where(
        patient_fold
        != fold_id
    )[0]


    train_patient_mask = np.isin(
        patient_index_long,
        train_patients,
    )


    test_patient_mask = np.isin(
        patient_index_long,
        test_patients,
    )


    for landmark_index in range(
        N_LANDMARK
    ):

        train_rows = np.where(
            train_patient_mask
            &
            (
                landmark_index_long
                == landmark_index
            )
        )[0]


        test_rows = np.where(
            test_patient_mask
            &
            (
                landmark_index_long
                == landmark_index
            )
        )[0]


        if len(
            train_rows
        ) == 0:

            raise ValueError(
                f"Fold {fold_id}, Landmark "
                f"{landmark_index}无训练记录"
            )


        if len(
            test_rows
        ) == 0:

            raise ValueError(
                f"Fold {fold_id}, Landmark "
                f"{landmark_index}无测试记录"
            )


        at_risk_train = (
            future_at_risk[
                train_rows,
                :
            ]
        )


        hazard_train = (
            original_hazard[
                train_rows,
                :
            ][
                at_risk_train
            ]
        )


        outcome_train = (
            future_event[
                train_rows,
                :
            ][
                at_risk_train
            ]
        )


        delta = fit_intercept_only(
            hazard_train,
            outcome_train,
        )


        # ---------------------------------------------------------------------
        # 只应用于held-out患者
        # ---------------------------------------------------------------------

        test_offset = logit(
            original_hazard[
                test_rows,
                :
            ]
        )


        oof_hazard[
            test_rows,
            :
        ] = expit(
            test_offset
            + delta
        )


        parameter_records.append({

            "fold_id":
                fold_id,

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

            "train_patient_n":
                len(
                    train_patients
                ),

            "test_patient_n":
                len(
                    test_patients
                ),

            "train_origin_n":
                len(
                    train_rows
                ),

            "test_origin_n":
                len(
                    test_rows
                ),

            "train_at_risk_interval_n":
                int(
                    at_risk_train.sum()
                ),

            "train_event_interval_n":
                int(
                    outcome_train.sum()
                ),

            "delta_intercept":
                delta,
        })


if not np.isfinite(
    oof_hazard
).all():

    raise ValueError(
        "OOF recalibrated hazard仍存在NaN"
    )


crossfit_parameters = pd.DataFrame(
    parameter_records
)


crossfit_parameters.to_csv(
    OUT
    / "crossfit_recalibration_parameters.csv",
    index=False,
    encoding="utf-8-sig",
)


# =============================================================================
# 14. Hazard → cumulative risk
# =============================================================================

def hazard_to_risk(
    hazard
):

    hazard = clip_probability(
        hazard
    )

    survival = np.cumprod(
        1.0
        - hazard,
        axis=1,
    )

    return (
        1.0
        - survival
    )


oof_risk = hazard_to_risk(
    oof_hazard
)


if np.any(
    np.diff(
        oof_risk,
        axis=1,
    )
    < -1e-8
):

    raise ValueError(
        "OOF recalibrated累计风险不是单调递增"
    )


np.save(
    OUT
    / "external_crossfit_recalibrated_hazard_long.npy",
    oof_hazard.astype(
        np.float32
    ),
)


np.save(
    OUT
    / "external_crossfit_recalibrated_risk_long.npy",
    oof_risk.astype(
        np.float32
    ),
)


# =============================================================================
# 15. 全重庆中心参数
#
# 只用于将来重庆本地部署。
#
# 不能用于报告本次secondary analysis性能，
# 因为这些参数使用了全部重庆结局。
# =============================================================================

full_parameter_records = []


for landmark_index in range(
    N_LANDMARK
):

    rows = np.where(
        landmark_index_long
        == landmark_index
    )[0]


    mask = future_at_risk[
        rows,
        :
    ]


    hazard = (
        original_hazard[
            rows,
            :
        ][
            mask
        ]
    )


    outcome = (
        future_event[
            rows,
            :
        ][
            mask
        ]
    )


    delta = fit_intercept_only(
        hazard,
        outcome,
    )


    full_parameter_records.append({

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

        "origin_n":
            len(
                rows
            ),

        "at_risk_interval_n":
            int(
                mask.sum()
            ),

        "event_interval_n":
            int(
                outcome.sum()
            ),

        "delta_intercept":
            delta,
    })


full_center_parameters = pd.DataFrame(
    full_parameter_records
)


full_center_parameters.to_csv(
    OUT
    / "FULL_CHONGQING_center_intercepts_FOR_FUTURE_DEPLOYMENT_ONLY.csv",
    index=False,
    encoding="utf-8-sig",
)


# =============================================================================
# 16. Descriptive free intercept + slope
#
# 仅作为诊断：
#
#   logit(h_observed)
#   = alpha + beta * logit(h_frozen)
#
# 这里使用全部重庆数据，只用于描述。
#
# 不用于OOF recalibrated prediction。
# =============================================================================

def fit_intercept_and_slope(
    hazard,
    outcome,
):

    x = logit(
        hazard
    )


    y = np.asarray(
        outcome,
        dtype=float,
    )


    def objective(
        parameter
    ):

        alpha = parameter[
            0
        ]

        beta = parameter[
            1
        ]


        eta = (
            alpha
            +
            beta
            * x
        )


        return float(
            np.sum(
                np.logaddexp(
                    0.0,
                    eta
                )
                -
                y
                * eta
            )
        )


    initial_delta = fit_intercept_only(
        hazard,
        outcome,
    )


    result = minimize(
        objective,
        x0=np.asarray(
            [
                initial_delta,
                1.0,
            ]
        ),
        method="BFGS",
    )


    if not result.success:

        warnings.warn(
            "某Landmark calibration slope优化未完全收敛："
            f"{result.message}"
        )


    return (
        float(
            result.x[
                0
            ]
        ),
        float(
            result.x[
                1
            ]
        ),
    )


diagnostic_records = []


for landmark_index in range(
    N_LANDMARK
):

    rows = np.where(
        landmark_index_long
        == landmark_index
    )[0]


    mask = future_at_risk[
        rows,
        :
    ]


    hazard = (
        original_hazard[
            rows,
            :
        ][
            mask
        ]
    )


    outcome = (
        future_event[
            rows,
            :
        ][
            mask
        ]
    )


    alpha, beta = fit_intercept_and_slope(
        hazard,
        outcome,
    )


    diagnostic_records.append({

        "landmark_index":
            landmark_index,

        "landmark_year":
            LANDMARK_YEARS[
                landmark_index
            ],

        "hazard_calibration_intercept":
            alpha,

        "hazard_calibration_slope":
            beta,

        "event_interval_n":
            int(
                outcome.sum()
            ),

        "at_risk_interval_n":
            int(
                len(
                    outcome
                )
            ),
    })


diagnostic_df = pd.DataFrame(
    diagnostic_records
)


diagnostic_df.to_csv(
    OUT
    / "descriptive_hazard_calibration_intercept_slope.csv",
    index=False,
    encoding="utf-8-sig",
)


# =============================================================================
# 17. KM observed risk
# =============================================================================

def km_risk(
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
        dtype=float,
    )


    (
        km_time,
        km_survival,
    ) = kaplan_meier_estimator(
        event,
        time_month,
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
        -
        km_survival[
            position
        ]
    )


# =============================================================================
# 18. Calibration-in-the-large比较
# =============================================================================

calibration_records = []


for landmark_index in range(
    N_LANDMARK
):

    rows = np.where(
        landmark_index_long
        == landmark_index
    )[0]


    event = (
        metadata.iloc[
            rows
        ][
            "event_within_60m"
        ]
        .to_numpy(
            int
        )
    )


    time_month = (
        metadata.iloc[
            rows
        ][
            "analysis_time_month"
        ]
        .to_numpy(
            float
        )
    )


    for horizon_month, position in (
        REPORT_POSITIONS.items()
    ):

        observed = km_risk(
            event,
            time_month,
            horizon_month,
        )


        original_mean = float(
            original_risk[
                rows,
                position
            ].mean()
        )


        recal_mean = float(
            oof_risk[
                rows,
                position
            ].mean()
        )


        calibration_records.append({

            "landmark_index":
                landmark_index,

            "landmark_year":
                LANDMARK_YEARS[
                    landmark_index
                ],

            "horizon_month":
                horizon_month,

            "horizon_year":
                horizon_month
                / 12.0,

            "origin_n":
                len(
                    rows
                ),

            "km_observed_risk":
                observed,

            "original_mean_predicted_risk":
                original_mean,

            "recalibrated_mean_predicted_risk":
                recal_mean,

            "original_minus_observed":
                original_mean
                - observed,

            "recalibrated_minus_observed":
                recal_mean
                - observed,
        })


calibration_summary = pd.DataFrame(
    calibration_records
)


calibration_summary.to_csv(
    OUT
    / "calibration_in_the_large_original_vs_recalibrated.csv",
    index=False,
    encoding="utf-8-sig",
)


# =============================================================================
# 19. 深圳+南宁development IPCW reference
#
# 用于Brier/IBS。
# 与Step8保持一致。
# =============================================================================

development_idx = np.load(
    STEP1
    / "development_idx.npy"
).astype(
    int
)


development_event_all = np.load(
    STEP1
    / "event.npy"
).astype(
    int
)


development_time_all = np.load(
    STEP1
    / "observed_time_month.npy"
).astype(
    float
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


def survival_array(
    event,
    time
):

    return Surv.from_arrays(
        event=np.asarray(
            event,
            dtype=bool,
        ),
        time=np.asarray(
            time,
            dtype=float,
        ),
    )


development_reference = []


for landmark_month in (
    LANDMARK_MONTHS
):

    eligible = (
        development_time
        >
        landmark_month
        + EPS
    )


    original_event = (
        development_event[
            eligible
        ]
    )


    residual = (
        development_time[
            eligible
        ]
        -
        landmark_month
    )


    event_60 = (
        (original_event == 1)
        &
        (
            residual
            <= 60.0
            + EPS
        )
    )


    # 保证reference覆盖外部60月
    reference_time = np.minimum(
        residual,
        61.0,
    )


    development_reference.append(

        survival_array(
            event_60,
            reference_time,
        )
    )


# =============================================================================
# 20. Brier / IBS函数
#
# 与Step8保持同一个边界逻辑。
# =============================================================================

def brier_curve_and_ibs(
    y_train,
    event_test,
    time_test,
    risk
):

    event_test = np.asarray(
        event_test,
        dtype=bool,
    )


    time_test = np.asarray(
        time_test,
        dtype=float,
    )


    risk = np.asarray(
        risk,
        dtype=float,
    )


    y_test = survival_array(
        event_test,
        time_test,
    )


    survival_prediction = (
        1.0
        -
        risk
    )


    brier = np.full(
        N_FUTURE,
        np.nan,
        dtype=float,
    )


    test_min = float(
        time_test.min()
    )


    test_max = float(
        time_test.max()
    )


    for position, metric_time in enumerate(
        METRIC_TIMES
    ):

        if metric_time < test_min:

            # 所有人明确未发生事件且未删失
            brier[
                position
            ] = float(
                np.mean(
                    risk[
                        :,
                        position
                    ]
                    ** 2
                )
            )

        elif metric_time < test_max:

            _, values = brier_score(
                y_train,
                y_test,
                survival_prediction[
                    :,
                    position:
                    position + 1
                ],
                np.asarray(
                    [
                        metric_time
                    ]
                ),
            )


            brier[
                position
            ] = float(
                values[
                    0
                ]
            )

        else:

            raise ValueError(
                "外部数据不支持评价到该时间点："
                f"{metric_time}"
            )


    if not np.isfinite(
        brier
    ).all():

        raise ValueError(
            "Brier curve存在NaN"
        )


    if hasattr(
        np,
        "trapezoid",
    ):

        integral = np.trapezoid(
            brier,
            METRIC_TIMES,
        )

    else:

        integral = np.trapz(
            brier,
            METRIC_TIMES,
        )


    ibs = float(
        integral
        /
        (
            METRIC_TIMES[
                -1
            ]
            -
            METRIC_TIMES[
                0
            ]
        )
    )


    return (
        brier,
        ibs
    )


# =============================================================================
# 21. Original vs recalibrated Brier / IBS
# =============================================================================

performance_records = []


for landmark_index in range(
    N_LANDMARK
):

    rows = np.where(
        landmark_index_long
        == landmark_index
    )[0]


    event = (
        metadata.iloc[
            rows
        ][
            "event_within_60m"
        ]
        .to_numpy(
            int
        )
    )


    time = (
        metadata.iloc[
            rows
        ][
            "analysis_time_month"
        ]
        .to_numpy(
            float
        )
    )


    original_brier, original_ibs = (
        brier_curve_and_ibs(
            development_reference[
                landmark_index
            ],
            event,
            time,
            original_risk[
                rows
            ],
        )
    )


    recal_brier, recal_ibs = (
        brier_curve_and_ibs(
            development_reference[
                landmark_index
            ],
            event,
            time,
            oof_risk[
                rows
            ],
        )
    )


    performance_records.append({

        "landmark_index":
            landmark_index,

        "landmark_year":
            LANDMARK_YEARS[
                landmark_index
            ],

        "origin_n":
            len(
                rows
            ),

        "original_IBS":
            original_ibs,

        "recalibrated_IBS":
            recal_ibs,

        "IBS_difference_recal_minus_original":
            recal_ibs
            -
            original_ibs,

        "original_Brier_1y":
            original_brier[
                1
            ],

        "recalibrated_Brier_1y":
            recal_brier[
                1
            ],

        "original_Brier_3y":
            original_brier[
                5
            ],

        "recalibrated_Brier_3y":
            recal_brier[
                5
            ],

        "original_Brier_5y":
            original_brier[
                9
            ],

        "recalibrated_Brier_5y":
            recal_brier[
                9
            ],
    })


performance_df = pd.DataFrame(
    performance_records
)


performance_df.to_csv(
    OUT
    / "Brier_IBS_original_vs_crossfit_recalibrated.csv",
    index=False,
    encoding="utf-8-sig",
)


# =============================================================================
# 22. 5年风险十分位校准
#
# 为保证比较的是同一批患者，
# 风险十分位按 ORIGINAL frozen risk 定义。
# =============================================================================

decile_records = []


for landmark_index in (
    PRIMARY_LANDMARK_INDEX
):

    rows = np.where(
        landmark_index_long
        == landmark_index
    )[0]


    temp = pd.DataFrame({

        "original_risk":
            original_risk[
                rows,
                9
            ],

        "recal_risk":
            oof_risk[
                rows,
                9
            ],

        "event":
            metadata.iloc[
                rows
            ][
                "event_within_60m"
            ]
            .to_numpy(
                int
            ),

        "time":
            metadata.iloc[
                rows
            ][
                "analysis_time_month"
            ]
            .to_numpy(
                float
            ),
    })


    temp[
        "risk_decile"
    ] = (
        pd.qcut(
            temp[
                "original_risk"
            ],
            q=10,
            labels=False,
            duplicates="drop",
        )
        +
        1
    )


    for group, data in temp.groupby(
        "risk_decile"
    ):

        observed = km_risk(
            data[
                "event"
            ],
            data[
                "time"
            ],
            60.0,
        )


        decile_records.append({

            "landmark_index":
                landmark_index,

            "landmark_year":
                LANDMARK_YEARS[
                    landmark_index
                ],

            "risk_decile":
                int(
                    group
                ),

            "group_n":
                len(
                    data
                ),

            "original_mean_predicted":
                float(
                    data[
                        "original_risk"
                    ].mean()
                ),

            "recalibrated_mean_predicted":
                float(
                    data[
                        "recal_risk"
                    ].mean()
                ),

            "km_observed":
                observed,
        })


decile_df = pd.DataFrame(
    decile_records
)


decile_df.to_csv(
    SOURCE_DIR
    / "5year_decile_calibration_original_vs_recalibrated.csv",
    index=False,
    encoding="utf-8-sig",
)


# =============================================================================
# 23. IPCW DCA
# =============================================================================

def step_function_value(
    time_grid,
    probability,
    query_time
):

    query = np.atleast_1d(
        np.asarray(
            query_time,
            dtype=float,
        )
    )


    indices = (
        np.searchsorted(
            time_grid,
            query,
            side="right",
        )
        -
        1
    )


    result = np.ones(
        len(
            query
        ),
        dtype=float,
    )


    valid = (
        indices >= 0
    )


    result[
        valid
    ] = probability[
        indices[
            valid
        ]
    ]


    return result


def calculate_dca(
    predicted_risk,
    event,
    time,
    thresholds
):

    predicted_risk = np.asarray(
        predicted_risk,
        dtype=float,
    )


    event = np.asarray(
        event,
        dtype=bool,
    )


    time = np.asarray(
        time,
        dtype=float,
    )


    horizon = 60.0


    case = (
        event
        &
        (
            time <= horizon
        )
    )


    control = (
        (~event)
        &
        (
            time >= horizon
        )
    )


    (
        censor_time,
        censor_survival,
    ) = kaplan_meier_estimator(
        event,
        time,
        reverse=True,
    )


    weights = np.zeros(
        len(
            event
        ),
        dtype=float,
    )


    if np.any(
        case
    ):

        query = np.nextafter(
            time[
                case
            ],
            -np.inf,
        )


        g = step_function_value(
            censor_time,
            censor_survival,
            query,
        )


        weights[
            case
        ] = (
            1.0
            /
            np.clip(
                g,
                1e-6,
                None,
            )
        )


    if np.any(
        control
    ):

        query = np.nextafter(
            horizon,
            -np.inf,
        )


        g = float(
            step_function_value(
                censor_time,
                censor_survival,
                query,
            )[
                0
            ]
        )


        weights[
            control
        ] = (
            1.0
            /
            max(
                g,
                1e-6,
            )
        )


    outcome = case.astype(
        int
    )


    known = (
        case
        |
        control
    )


    n = float(
        len(
            event
        )
    )


    event_total = float(
        np.sum(
            weights[
                known
            ]
            *
            outcome[
                known
            ]
        )
    )


    nonevent_total = float(
        np.sum(
            weights[
                known
            ]
            *
            (
                1
                -
                outcome[
                    known
                ]
            )
        )
    )


    model_nb = []

    treat_all_nb = []


    for threshold in thresholds:

        positive = (
            predicted_risk
            >= threshold
        )


        tp = float(
            np.sum(
                weights[
                    known
                    & positive
                ]
                *
                outcome[
                    known
                    & positive
                ]
            )
        )


        fp = float(
            np.sum(
                weights[
                    known
                    & positive
                ]
                *
                (
                    1
                    -
                    outcome[
                        known
                        & positive
                    ]
                )
            )
        )


        odds = (
            threshold
            /
            (
                1.0
                -
                threshold
            )
        )


        model_nb.append(
            tp
            / n
            -
            fp
            / n
            * odds
        )


        treat_all_nb.append(
            event_total
            / n
            -
            nonevent_total
            / n
            * odds
        )


    return (
        np.asarray(
            model_nb
        ),
        np.asarray(
            treat_all_nb
        ),
    )


# =============================================================================
# 24. DCA source data
# =============================================================================

dca_records = []


for landmark_index in (
    PRIMARY_LANDMARK_INDEX
):

    rows = np.where(
        landmark_index_long
        == landmark_index
    )[0]


    event = (
        metadata.iloc[
            rows
        ][
            "event_within_60m"
        ]
        .to_numpy(
            int
        )
    )


    time = (
        metadata.iloc[
            rows
        ][
            "analysis_time_month"
        ]
        .to_numpy(
            float
        )
    )


    original_nb, treat_all = calculate_dca(
        original_risk[
            rows,
            9
        ],
        event,
        time,
        DCA_THRESHOLDS,
    )


    recal_nb, _ = calculate_dca(
        oof_risk[
            rows,
            9
        ],
        event,
        time,
        DCA_THRESHOLDS,
    )


    for (
        threshold,
        original_value,
        recal_value,
        all_value,
    ) in zip(
        DCA_THRESHOLDS,
        original_nb,
        recal_nb,
        treat_all,
    ):

        dca_records.append({

            "landmark_index":
                landmark_index,

            "landmark_year":
                LANDMARK_YEARS[
                    landmark_index
                ],

            "threshold_probability":
                threshold,

            "original_net_benefit":
                original_value,

            "recalibrated_net_benefit":
                recal_value,

            "treat_all_net_benefit":
                all_value,

            "treat_none_net_benefit":
                0.0,
        })


dca_df = pd.DataFrame(
    dca_records
)


dca_df.to_csv(
    SOURCE_DIR
    / "DCA_original_vs_crossfit_recalibrated_0p5_to_10pct.csv",
    index=False,
    encoding="utf-8-sig",
)


# =============================================================================
# 25. Figure 1
# Original vs recalibrated calibration-in-the-large
# =============================================================================

def plot_calibration_in_large():

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(
            15,
            5,
        ),
        constrained_layout=True,
    )


    for panel_i, horizon_year in enumerate(
        [
            1,
            3,
            5,
        ]
    ):

        ax = axes[
            panel_i
        ]


        panel_label(
            ax,
            chr(
                65
                +
                panel_i
            )
        )


        data = (
            calibration_summary[
                (
                    calibration_summary[
                        "horizon_year"
                    ]
                    == horizon_year
                )
                &
                (
                    calibration_summary[
                        "landmark_index"
                    ].isin(
                        PRIMARY_LANDMARK_INDEX
                    )
                )
            ]
            .sort_values(
                "landmark_year"
            )
        )


        x = data[
            "landmark_year"
        ].to_numpy(
            float
        )


        observed = (
            data[
                "km_observed_risk"
            ]
            .to_numpy(
                float
            )
            * 100
        )


        original = (
            data[
                "original_mean_predicted_risk"
            ]
            .to_numpy(
                float
            )
            * 100
        )


        recal = (
            data[
                "recalibrated_mean_predicted_risk"
            ]
            .to_numpy(
                float
            )
            * 100
        )


        ax.plot(
            x,
            original,
            color=ORIGINAL_COLOR,
            marker="D",
            linewidth=2,
            label="Original frozen model",
        )


        ax.plot(
            x,
            recal,
            color=RECAL_COLOR,
            marker="s",
            linewidth=2,
            label="Cross-fitted recalibration",
        )


        ax.plot(
            x,
            observed,
            color=OBSERVED_COLOR,
            marker="o",
            markerfacecolor="white",
            linestyle="--",
            linewidth=1.8,
            label="KM observed",
        )


        ax.set_xticks(
            [
                0,
                1,
                3,
                5,
            ]
        )


        ax.set_xlabel(
            "ART landmark (years)"
        )


        ax.set_ylabel(
            "CKD risk (%)"
        )


        ax.set_title(
            f"Future {horizon_year}-year CKD risk",
            fontweight="bold",
        )


        ax.set_ylim(
            bottom=0
        )


        style_axis(
            ax
        )


        if panel_i == 0:

            ax.legend(
                frameon=False,
                fontsize=8,
            )


    fig.suptitle(
        "Calibration-in-the-large before and after cross-fitted center recalibration",
        fontsize=14,
        fontweight="bold",
    )


    save_figure(
        fig,
        "Secondary_Figure1_CalibrationInLarge"
    )


# =============================================================================
# 26. Figure 2
# 5-year decile calibration
# =============================================================================

def plot_decile_calibration():

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(
            12,
            10,
        ),
        constrained_layout=True,
    )


    axes = axes.flatten()


    for panel_i, landmark_index in enumerate(
        PRIMARY_LANDMARK_INDEX
    ):

        ax = axes[
            panel_i
        ]


        panel_label(
            ax,
            chr(
                65
                +
                panel_i
            )
        )


        data = (
            decile_df[
                decile_df[
                    "landmark_index"
                ]
                == landmark_index
            ]
            .sort_values(
                "risk_decile"
            )
        )


        original = (
            data[
                "original_mean_predicted"
            ]
            .to_numpy(
                float
            )
            * 100
        )


        recal = (
            data[
                "recalibrated_mean_predicted"
            ]
            .to_numpy(
                float
            )
            * 100
        )


        observed = (
            data[
                "km_observed"
            ]
            .to_numpy(
                float
            )
            * 100
        )


        maximum = max(
            original.max(),
            recal.max(),
            observed.max(),
        )


        maximum = max(
            maximum
            * 1.10,
            1.0,
        )


        ax.plot(
            [
                0,
                maximum,
            ],
            [
                0,
                maximum,
            ],
            "--",
            color="#777777",
            linewidth=1.1,
            label="Perfect calibration",
        )


        ax.plot(
            original,
            observed,
            color=ORIGINAL_COLOR,
            marker="D",
            linewidth=1.8,
            label="Original",
        )


        ax.plot(
            recal,
            observed,
            color=RECAL_COLOR,
            marker="s",
            linewidth=1.8,
            label="Cross-fitted recalibration",
        )


        landmark_year = (
            LANDMARK_YEARS[
                landmark_index
            ]
        )


        title = (
            "Baseline landmark"
            if landmark_year == 0
            else
            f"ART year {int(landmark_year)} landmark"
        )


        ax.set_title(
            title
            +
            "\n5-year CKD risk",
            fontweight="bold",
        )


        ax.set_xlabel(
            "Predicted 5-year risk (%)"
        )


        ax.set_ylabel(
            "Observed 5-year risk (%)"
        )


        ax.set_xlim(
            0,
            maximum,
        )


        ax.set_ylim(
            0,
            maximum,
        )


        ax.set_aspect(
            "equal",
            adjustable="box",
        )


        style_axis(
            ax
        )


        ax.legend(
            frameon=False,
            fontsize=7.5,
        )


    fig.suptitle(
        "Five-year calibration before and after cross-fitted center recalibration",
        fontsize=14,
        fontweight="bold",
    )


    save_figure(
        fig,
        "Secondary_Figure2_5year_Calibration"
    )


# =============================================================================
# 27. Figure 3
# DCA 0.5%–10%
# =============================================================================

def plot_dca():

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(
            12.5,
            9.5,
        ),
        constrained_layout=True,
    )


    axes = axes.flatten()


    for panel_i, landmark_index in enumerate(
        PRIMARY_LANDMARK_INDEX
    ):

        ax = axes[
            panel_i
        ]


        panel_label(
            ax,
            chr(
                65
                +
                panel_i
            )
        )


        data = (
            dca_df[
                dca_df[
                    "landmark_index"
                ]
                == landmark_index
            ]
            .sort_values(
                "threshold_probability"
            )
        )


        x = (
            data[
                "threshold_probability"
            ]
            .to_numpy(
                float
            )
            * 100
        )


        original = data[
            "original_net_benefit"
        ].to_numpy(
            float
        )


        recal = data[
            "recalibrated_net_benefit"
        ].to_numpy(
            float
        )


        treat_all = data[
            "treat_all_net_benefit"
        ].to_numpy(
            float
        )


        # 轻度平滑仅用于显示
        original_display = (
            pd.Series(
                original
            )
            .rolling(
                7,
                center=True,
                min_periods=1,
            )
            .mean()
            .to_numpy()
        )


        recal_display = (
            pd.Series(
                recal
            )
            .rolling(
                7,
                center=True,
                min_periods=1,
            )
            .mean()
            .to_numpy()
        )


        ax.plot(
            x,
            original_display,
            color=ORIGINAL_COLOR,
            linewidth=2.1,
            label="Original frozen model",
        )


        ax.plot(
            x,
            recal_display,
            color=RECAL_COLOR,
            linewidth=2.2,
            label="Cross-fitted recalibration",
        )


        ax.plot(
            x,
            treat_all,
            "--",
            color="#777777",
            linewidth=1.2,
            label="Treat all",
        )


        ax.axhline(
            0,
            color="black",
            linestyle=":",
            linewidth=1.2,
            label="Treat none",
        )


        landmark_year = (
            LANDMARK_YEARS[
                landmark_index
            ]
        )


        title = (
            "Baseline landmark"
            if landmark_year == 0
            else
            f"ART year {int(landmark_year)} landmark"
        )


        ax.set_title(
            title
            +
            "\n5-year CKD risk",
            fontweight="bold",
        )


        ax.set_xlabel(
            "Threshold probability (%)"
        )


        ax.set_ylabel(
            "Net benefit"
        )


        ax.set_xlim(
            0.5,
            10.0,
        )


        relevant = np.concatenate(
            [
                original_display,
                recal_display,
                np.zeros_like(
                    recal_display
                ),
            ]
        )


        ymin = min(
            -0.002,
            relevant.min()
            - 0.001,
        )


        ymax = max(
            0.003,
            relevant.max()
            + 0.002,
        )


        ax.set_ylim(
            ymin,
            ymax,
        )


        style_axis(
            ax
        )


        ax.legend(
            frameon=False,
            fontsize=7.5,
            loc="upper right",
        )


    fig.suptitle(
        "Decision curve analysis before and after cross-fitted center recalibration",
        fontsize=14,
        fontweight="bold",
    )


    save_figure(
        fig,
        "Secondary_Figure3_DCA_0p5_to_10pct"
    )


# =============================================================================
# 28. 运行图
# =============================================================================

plot_calibration_in_large()

plot_decile_calibration()

plot_dca()


# =============================================================================
# 29. Manifest
# =============================================================================

manifest = {

    "analysis":
        (
            "Post hoc exploratory secondary cross-fitted "
            "center recalibration in Chongqing"
        ),

    "recalibration_model":
        (
            "landmark-specific intercept-only update "
            "on discrete-time hazard logits"
        ),

    "formula":
        (
            "logit(h_recalibrated) = "
            "logit(h_frozen) + delta_landmark"
        ),

    "crossfitting":
        "patient-level 5-fold",

    "landmark_n":
        6,

    "parameters_per_landmark":
        1,

    "future_intervals":
        10,

    "primary_external_validation_replaced":
        False,

    "full_external_center_intercepts_used_for_performance":
        False,

    "dca_secondary_threshold_range":
        [
            0.005,
            0.10,
        ],
}


with open(
    OUT
    / "secondary_recalibration_manifest.json",
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
# 30. 最终打印
# =============================================================================

print()
print("=" * 120)

print(
    "重庆 secondary cross-fitted recalibration 完成"
)

print("=" * 120)


print()
print(
    "Cross-fitted intercept："
)


print(
    crossfit_parameters[
        [
            "fold_id",
            "landmark_year",
            "train_event_interval_n",
            "delta_intercept",
        ]
    ]
    .round(
        4
    )
    .to_string(
        index=False
    )
)


print()
print(
    "全重庆 descriptive calibration intercept / slope："
)


print(
    diagnostic_df
    .round(
        4
    )
    .to_string(
        index=False
    )
)


print()
print(
    "Original vs recalibrated calibration-in-the-large："
)


print(
    calibration_summary.loc[
        (
            calibration_summary[
                "landmark_index"
            ].isin(
                PRIMARY_LANDMARK_INDEX
            )
        )
        &
        (
            calibration_summary[
                "horizon_year"
            ].isin(
                [
                    1,
                    3,
                    5,
                ]
            )
        )
    ][
        [
            "landmark_year",
            "horizon_year",
            "km_observed_risk",
            "original_mean_predicted_risk",
            "recalibrated_mean_predicted_risk",
            "original_minus_observed",
            "recalibrated_minus_observed",
        ]
    ]
    .round(
        4
    )
    .to_string(
        index=False
    )
)


print()
print(
    "Original vs recalibrated Brier / IBS："
)


print(
    performance_df.loc[
        performance_df[
            "landmark_index"
        ].isin(
            PRIMARY_LANDMARK_INDEX
        )
    ][
        [
            "landmark_year",
            "original_IBS",
            "recalibrated_IBS",
            "IBS_difference_recal_minus_original",
            "original_Brier_5y",
            "recalibrated_Brier_5y",
        ]
    ]
    .round(
        5
    )
    .to_string(
        index=False
    )
)


print()
print(
    "输出目录："
)

print(
    OUT
)


print("=" * 120)