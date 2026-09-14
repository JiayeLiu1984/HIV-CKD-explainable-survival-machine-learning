# %matplotlib inline

# -*- coding: utf-8 -*-

# =============================================================================
# 重庆独立地理外部验证
# FINAL FIGURES + FINAL NUMERICAL TABLE
#
# 最终仅输出：
#
# Figure 1  5-year Uno C-index + 95% CI
# Figure 2  iAUC + 95% CI
# Figure 3  Dynamic Brier score + IBS
# Figure 4  5-year calibration curve
# Figure 5  5-year DCA
#
# Table 1   External validation numerical summary
#
# 最终统计口径：
#
# C-index / iAUC
#   → primary frozen geographic external validation
#
# Brier / IBS / Calibration / DCA
#   → patient-level cross-fitted intercept-only recalibrated LSTM
#
# Calibration和DCA图中模型统一标注为：
#   LSTM
#
# 不重新训练模型
# 不重新拟合recalibration
# 不显示Original曲线
# =============================================================================


from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib import font_manager
from matplotlib.ticker import MaxNLocator

from sksurv.metrics import brier_score
from sksurv.nonparametric import kaplan_meier_estimator
from sksurv.util import Surv


# =============================================================================
# 1. PATHS
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

STEP8 = (
    BASE
    / "重庆外部验证_step8_FINAL_performance_FIXED"
)

RECAL = (
    BASE
    / "重庆外部验证_secondary_crossfit_recalibration"
)

OUT = (
    BASE
    / "重庆外部验证_FINAL_5Figures_Table"
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)


# =============================================================================
# 2. INPUT FILES
# =============================================================================

LANDMARK_FILE = (
    STEP8
    / "04_landmark_performance_with_95CI.csv"
)

RISK_SET_FILE = (
    STEP8
    / "08_external_landmark_risk_sets.csv"
)

METADATA_FILE = (
    STEP6
    / "super_landmark_external_metadata.csv"
)

RECAL_RISK_FILE = (
    RECAL
    / "external_crossfit_recalibrated_risk_long.npy"
)

RECAL_PERFORMANCE_FILE = (
    RECAL
    / "Brier_IBS_original_vs_crossfit_recalibrated.csv"
)


# =============================================================================
# 3. CONSTANTS
# =============================================================================

PRIMARY_LANDMARKS = np.asarray(
    [0.0, 1.0, 3.0, 5.0],
    dtype=float,
)

PRIMARY_LANDMARK_INDEX = np.asarray(
    [0, 1, 3, 5],
    dtype=int,
)

LANDMARK_LABELS = [
    "Baseline\n(0 y)",
    "1 y",
    "3 y",
    "5 y",
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

FUTURE_YEARS = (
    FUTURE_MONTHS
    / 12.0
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


# -------------------------------------------------------------------------
# 颜色
# -------------------------------------------------------------------------

LSTM_COLOR = "#4C78A8"

PERFECT_COLOR = "#777777"

TREAT_ALL_COLOR = "#888888"

GRID_COLOR = "#D0D0D0"

BLACK = "#000000"

WHITE = "#FFFFFF"

DPI = 600


# -------------------------------------------------------------------------
# DCA最终展示范围
# -------------------------------------------------------------------------

DCA_THRESHOLDS = np.linspace(
    0.005,
    0.10,
    250,
)


# =============================================================================
# 4. FONT + MATPLOTLIB
#
# 强制所有字体黑色
# 强制所有背景白色
# =============================================================================

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

            continue

    return "DejaVu Sans"


plt.style.use(
    "default"
)

plt.rcParams.update({

    "font.family":
        choose_font(),

    "font.size":
        10.5,

    "figure.facecolor":
        WHITE,

    "axes.facecolor":
        WHITE,

    "savefig.facecolor":
        WHITE,

    "text.color":
        BLACK,

    "axes.labelcolor":
        BLACK,

    "axes.edgecolor":
        BLACK,

    "axes.titlecolor":
        BLACK,

    "xtick.color":
        BLACK,

    "ytick.color":
        BLACK,

    "legend.labelcolor":
        BLACK,

    "axes.spines.top":
        False,

    "axes.spines.right":
        False,

    "axes.linewidth":
        0.8,

    "axes.titlesize":
        11.5,

    "axes.labelsize":
        10.5,

    "xtick.labelsize":
        9.5,

    "ytick.labelsize":
        9.5,

    "legend.fontsize":
        8.0,

    "pdf.fonttype":
        42,

    "ps.fonttype":
        42,

    "savefig.dpi":
        DPI,
})


# =============================================================================
# 5. GENERAL FIGURE FUNCTIONS
# =============================================================================

def style_axis(
    ax,
    grid_axis="both",
):

    ax.set_facecolor(
        WHITE
    )

    ax.grid(
        axis=grid_axis,
        linestyle="--",
        linewidth=0.55,
        color=GRID_COLOR,
        alpha=0.55,
    )

    ax.set_axisbelow(
        True
    )

    ax.tick_params(
        colors=BLACK,
    )

    ax.xaxis.label.set_color(
        BLACK
    )

    ax.yaxis.label.set_color(
        BLACK
    )

    ax.title.set_color(
        BLACK
    )

    ax.spines[
        "left"
    ].set_color(
        BLACK
    )

    ax.spines[
        "bottom"
    ].set_color(
        BLACK
    )


def add_panel_label(
    ax,
    label,
):

    ax.text(
        -0.10,
        1.06,
        label,
        transform=ax.transAxes,
        fontsize=13,
        fontweight="bold",
        color=BLACK,
        ha="left",
        va="top",
    )


def landmark_title(
    year,
):

    if np.isclose(
        year,
        0,
    ):

        return (
            "Baseline landmark\n"
            "Predicting ART years 0–5"
        )

    return (
        f"ART year {int(year)} landmark\n"
        f"Predicting ART years "
        f"{int(year)}–{int(year + 5)}"
    )


def save_figure(
    fig,
    name,
):

    fig.patch.set_facecolor(
        WHITE
    )

    for extension in [
        "png",
        "pdf",
        "svg",
    ]:

        kwargs = {

            "bbox_inches":
                "tight",

            "facecolor":
                WHITE,
        }

        if extension == "png":

            kwargs[
                "dpi"
            ] = DPI

        fig.savefig(
            OUT
            / f"{name}.{extension}",
            **kwargs,
        )

    plt.show()

    plt.close(
        fig
    )


# =============================================================================
# 6. FILE CHECK
# =============================================================================

required_files = [

    LANDMARK_FILE,

    RISK_SET_FILE,

    METADATA_FILE,

    RECAL_RISK_FILE,

    RECAL_PERFORMANCE_FILE,

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
# 7. READ RESULTS
# =============================================================================

landmark_df = pd.read_csv(
    LANDMARK_FILE,
    encoding="utf-8-sig",
)

risk_set_df = pd.read_csv(
    RISK_SET_FILE,
    encoding="utf-8-sig",
)

metadata = pd.read_csv(
    METADATA_FILE,
    encoding="utf-8-sig",
)

recal_performance_df = pd.read_csv(
    RECAL_PERFORMANCE_FILE,
    encoding="utf-8-sig",
)

recal_risk = np.load(
    RECAL_RISK_FILE
).astype(
    float
)


# =============================================================================
# 8. BASIC QC
# =============================================================================

if recal_risk.shape != (
    len(
        metadata
    ),
    10,
):

    raise ValueError(
        "recalibrated risk形状错误："
        f"{recal_risk.shape}"
    )


if not np.isfinite(
    recal_risk
).all():

    raise ValueError(
        "recalibrated risk存在NaN/Inf。"
    )


if np.any(
    recal_risk < 0
) or np.any(
    recal_risk > 1
):

    raise ValueError(
        "recalibrated risk不在[0,1]。"
    )


if np.any(
    np.diff(
        recal_risk,
        axis=1,
    )
    < -1e-8
):

    raise ValueError(
        "recalibrated累计风险不是单调递增。"
    )


landmark_primary = (
    landmark_df.loc[
        landmark_df[
            "landmark_year"
        ].isin(
            PRIMARY_LANDMARKS
        )
    ]
    .sort_values(
        "landmark_year"
    )
    .reset_index(
        drop=True
    )
)


recal_performance_primary = (
    recal_performance_df.loc[
        recal_performance_df[
            "landmark_year"
        ].isin(
            PRIMARY_LANDMARKS
        )
    ]
    .sort_values(
        "landmark_year"
    )
    .reset_index(
        drop=True
    )
)


if len(
    landmark_primary
) != 4:

    raise ValueError(
        "主Landmark不是0、1、3、5年。"
    )


# =============================================================================
# 9. DEVELOPMENT IPCW REFERENCE
#
# 为recalibrated dynamic Brier重新计算完整曲线
# 与前面Step8/secondary分析保持一致
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


def make_surv(
    event,
    time,
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


development_reference = {}


for landmark_index, landmark_year in zip(
    PRIMARY_LANDMARK_INDEX,
    PRIMARY_LANDMARKS,
):

    landmark_month = (
        landmark_year
        * 12
    )

    eligible = (
        development_time
        >
        landmark_month
        + 1e-7
    )

    event = (
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
        (event == 1)
        &
        (
            residual
            <= 60.0
            + 1e-7
        )
    )

    reference_time = np.minimum(
        residual,
        61.0,
    )

    development_reference[
        landmark_index
    ] = make_surv(
        event_60,
        reference_time,
    )


# =============================================================================
# 10. RECALIBRATED DYNAMIC BRIER
# =============================================================================

def calculate_brier_curve(
    landmark_index,
):

    rows = np.where(
        metadata[
            "landmark_index"
        ]
        .to_numpy(
            int
        )
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
        .astype(
            bool
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


    risk = (
        recal_risk[
            rows
        ]
    )


    survival = (
        1.0
        -
        risk
    )


    y_test = make_surv(
        event,
        time,
    )


    y_train = (
        development_reference[
            landmark_index
        ]
    )


    test_min = float(
        time.min()
    )

    test_max = float(
        time.max()
    )


    brier = np.full(
        10,
        np.nan,
        dtype=float,
    )


    for position, metric_time in enumerate(
        METRIC_TIMES
    ):

        # -------------------------------------------------------------
        # 评价时间早于所有人的最短随访
        # 所有人均明确event-free
        # -------------------------------------------------------------

        if metric_time < test_min:

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
                survival[
                    :,
                    position:
                    position
                    + 1
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
                f"Landmark {landmark_index}"
                f"不支持评价时间{metric_time}"
            )


    if not np.isfinite(
        brier
    ).all():

        raise ValueError(
            "Brier曲线存在NaN。"
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
            METRIC_TIMES[-1]
            -
            METRIC_TIMES[0]
        )
    )


    return (
        brier,
        ibs,
    )


brier_records = []


for landmark_index, landmark_year in zip(
    PRIMARY_LANDMARK_INDEX,
    PRIMARY_LANDMARKS,
):

    brier, ibs = calculate_brier_curve(
        landmark_index
    )

    for position in range(
        10
    ):

        brier_records.append({

            "landmark_index":
                landmark_index,

            "landmark_year":
                landmark_year,

            "horizon_month":
                FUTURE_MONTHS[
                    position
                ],

            "horizon_year":
                FUTURE_YEARS[
                    position
                ],

            "brier_score":
                brier[
                    position
                ],

            "IBS":
                ibs,
        })


brier_df = pd.DataFrame(
    brier_records
)


brier_df.to_csv(
    OUT
    / "SourceData_Figure3_Brier.csv",
    index=False,
    encoding="utf-8-sig",
)


# =============================================================================
# 11. KM RISK
# =============================================================================

def km_risk(
    event,
    time,
    horizon=60.0,
):

    event = np.asarray(
        event,
        dtype=bool,
    )

    time = np.asarray(
        time,
        dtype=float,
    )


    km_time, km_survival = (
        kaplan_meier_estimator(
            event,
            time,
        )
    )


    position = (
        np.searchsorted(
            km_time,
            horizon,
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
# 12. FINAL CALIBRATION DATA
#
# 只按cross-fitted recalibrated risk重新分十分位
# =============================================================================

calibration_records = []


for landmark_index, landmark_year in zip(
    PRIMARY_LANDMARK_INDEX,
    PRIMARY_LANDMARKS,
):

    rows = np.where(
        metadata[
            "landmark_index"
        ]
        .to_numpy(
            int
        )
        == landmark_index
    )[0]


    temp = pd.DataFrame({

        "risk":
            recal_risk[
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
        "decile"
    ] = (
        pd.qcut(
            temp[
                "risk"
            ],
            q=10,
            labels=False,
            duplicates="drop",
        )
        +
        1
    )


    for decile, group in temp.groupby(
        "decile",
        sort=True,
    ):

        calibration_records.append({

            "landmark_index":
                landmark_index,

            "landmark_year":
                landmark_year,

            "risk_decile":
                int(
                    decile
                ),

            "n":
                len(
                    group
                ),

            "event_n":
                int(
                    group[
                        "event"
                    ].sum()
                ),

            "mean_predicted_risk":
                float(
                    group[
                        "risk"
                    ].mean()
                ),

            "observed_risk":
                km_risk(
                    group[
                        "event"
                    ],
                    group[
                        "time"
                    ],
                    60.0,
                ),
        })


calibration_df = pd.DataFrame(
    calibration_records
)


calibration_df.to_csv(
    OUT
    / "SourceData_Figure4_Calibration.csv",
    index=False,
    encoding="utf-8-sig",
)


# =============================================================================
# 13. DCA FUNCTIONS
# =============================================================================

def step_value(
    time_grid,
    probability,
    query,
):

    query = np.atleast_1d(
        np.asarray(
            query,
            dtype=float,
        )
    )


    index = (
        np.searchsorted(
            time_grid,
            query,
            side="right",
        )
        - 1
    )


    result = np.ones(
        len(
            query
        ),
        dtype=float,
    )


    valid = (
        index >= 0
    )


    result[
        valid
    ] = probability[
        index[
            valid
        ]
    ]


    return result


def calculate_dca(
    risk,
    event,
    time,
    thresholds,
    horizon=60.0,
):

    risk = np.asarray(
        risk,
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


    case = (
        event
        &
        (
            time
            <= horizon
        )
    )


    control = (
        (~event)
        &
        (
            time
            >= horizon
        )
    )


    censor_time, censor_survival = (
        kaplan_meier_estimator(
            event,
            time,
            reverse=True,
        )
    )


    weights = np.zeros(
        len(
            event
        ),
        dtype=float,
    )


    if case.any():

        query = np.nextafter(
            time[
                case
            ],
            -np.inf,
        )

        g = step_value(
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


    if control.any():

        query = np.nextafter(
            horizon,
            -np.inf,
        )

        g = float(
            step_value(
                censor_time,
                censor_survival,
                query,
            )[0]
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


    total_event = np.sum(
        weights[
            known
        ]
        *
        outcome[
            known
        ]
    )


    total_nonevent = np.sum(
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


    model_nb = []

    treat_all_nb = []


    for threshold in thresholds:

        positive = (
            risk
            >= threshold
        )


        tp = np.sum(
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


        fp = np.sum(
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


        odds = (
            threshold
            /
            (
                1
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
            total_event
            / n
            -
            total_nonevent
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
# 14. BUILD DCA DATA
# =============================================================================

dca_records = []


for landmark_index, landmark_year in zip(
    PRIMARY_LANDMARK_INDEX,
    PRIMARY_LANDMARKS,
):

    rows = np.where(
        metadata[
            "landmark_index"
        ]
        .to_numpy(
            int
        )
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


    model_nb, treat_all_nb = calculate_dca(

        recal_risk[
            rows,
            9
        ],

        event,

        time,

        DCA_THRESHOLDS,
    )


    display_nb = (
        pd.Series(
            model_nb
        )
        .rolling(
            window=7,
            center=True,
            min_periods=1,
        )
        .mean()
        .to_numpy()
    )


    for (
        threshold,
        raw,
        display,
        treat_all,
    ) in zip(
        DCA_THRESHOLDS,
        model_nb,
        display_nb,
        treat_all_nb,
    ):

        dca_records.append({

            "landmark_index":
                landmark_index,

            "landmark_year":
                landmark_year,

            "threshold_probability":
                threshold,

            "LSTM_net_benefit":
                raw,

            "LSTM_display_net_benefit":
                display,

            "treat_all_net_benefit":
                treat_all,

            "treat_none_net_benefit":
                0.0,
        })


dca_df = pd.DataFrame(
    dca_records
)


dca_df.to_csv(
    OUT
    / "SourceData_Figure5_DCA.csv",
    index=False,
    encoding="utf-8-sig",
)


# =============================================================================
# FIGURE 1
# C-INDEX
# =============================================================================

def plot_c_index():

    df = landmark_primary

    x = np.arange(
        4
    )


    point = df[
        "uno_c_index_5y"
    ].to_numpy(
        float
    )


    lower = df[
        "uno_c_lower_95"
    ].to_numpy(
        float
    )


    upper = df[
        "uno_c_upper_95"
    ].to_numpy(
        float
    )


    error = np.vstack([

        point
        -
        lower,

        upper
        -
        point,
    ])


    fig, ax = plt.subplots(
        figsize=(
            9.2,
            5.6,
        ),
        facecolor=WHITE,
    )


    bars = ax.bar(
        x,
        point,
        width=0.58,
        color=LSTM_COLOR,
        edgecolor=WHITE,
        yerr=error,
        error_kw={
            "ecolor":
                BLACK,

            "capsize":
                5,

            "elinewidth":
                1.25,

            "capthick":
                1.2,
        },
    )


    for bar, value, high in zip(
        bars,
        point,
        upper,
    ):

        ax.text(
            bar.get_x()
            +
            bar.get_width()
            / 2,
            high
            + 0.006,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            color=BLACK,
            fontweight="bold",
        )


    ax.set_xticks(
        x
    )

    ax.set_xticklabels(
        LANDMARK_LABELS,
        color=BLACK,
    )


    ax.set_xlabel(
        "ART landmark",
        color=BLACK,
    )


    ax.set_ylabel(
        "Uno C-index for future 5-year CKD risk",
        color=BLACK,
    )


    ax.set_title(
        "Five-year discrimination in geographic external validation",
        fontsize=14,
        fontweight="bold",
        color=BLACK,
    )


    ax.set_ylim(
        max(
            0.50,
            lower.min()
            - 0.04,
        ),
        min(
            1.00,
            upper.max()
            + 0.04,
        ),
    )


    style_axis(
        ax,
        "y",
    )


    fig.tight_layout()


    save_figure(
        fig,
        "Figure1_C_index",
    )


# =============================================================================
# FIGURE 2
# iAUC + 95% CI
# =============================================================================

def plot_iauc():

    df = landmark_primary

    x = np.arange(
        4
    )


    point = df[
        "integrated_dynamic_auc"
    ].to_numpy(
        float
    )


    lower = df[
        "iauc_lower_95"
    ].to_numpy(
        float
    )


    upper = df[
        "iauc_upper_95"
    ].to_numpy(
        float
    )


    error = np.vstack([

        point
        -
        lower,

        upper
        -
        point,
    ])


    fig, ax = plt.subplots(
        figsize=(
            9.2,
            5.6,
        ),
        facecolor=WHITE,
    )


    bars = ax.bar(
        x,
        point,
        width=0.58,
        color=LSTM_COLOR,
        edgecolor=WHITE,
        yerr=error,
        error_kw={
            "ecolor":
                BLACK,

            "capsize":
                5,

            "elinewidth":
                1.25,

            "capthick":
                1.2,
        },
    )


    for bar, value, high in zip(
        bars,
        point,
        upper,
    ):

        ax.text(
            bar.get_x()
            +
            bar.get_width()
            / 2,
            high
            + 0.006,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            color=BLACK,
            fontweight="bold",
        )


    ax.set_xticks(
        x
    )

    ax.set_xticklabels(
        LANDMARK_LABELS,
        color=BLACK,
    )


    ax.set_xlabel(
        "ART landmark",
        color=BLACK,
    )


    ax.set_ylabel(
        "Integrated time-dependent AUC",
        color=BLACK,
    )


    ax.set_title(
        "Integrated time-dependent AUC in geographic external validation",
        fontsize=14,
        fontweight="bold",
        color=BLACK,
    )


    ax.set_ylim(
        max(
            0.50,
            lower.min()
            - 0.04,
        ),
        min(
            1.00,
            upper.max()
            + 0.04,
        ),
    )


    style_axis(
        ax,
        "y",
    )


    fig.tight_layout()


    save_figure(
        fig,
        "Figure2_iAUC",
    )


# =============================================================================
# FIGURE 3
# RECALIBRATED DYNAMIC BRIER + IBS
# =============================================================================

def plot_brier():

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(
            12.0,
            8.6,
        ),
        constrained_layout=True,
        facecolor=WHITE,
    )


    axes = axes.flatten()


    for i, landmark in enumerate(
        PRIMARY_LANDMARKS
    ):

        ax = axes[
            i
        ]


        add_panel_label(
            ax,
            chr(
                65
                + i
            ),
        )


        temp = (
            brier_df.loc[
                np.isclose(
                    brier_df[
                        "landmark_year"
                    ],
                    landmark,
                )
            ]
            .sort_values(
                "horizon_year"
            )
        )


        x = temp[
            "horizon_year"
        ].to_numpy(
            float
        )


        y = (
            temp[
                "brier_score"
            ]
            .to_numpy(
                float
            )
            * 1000
        )


        ibs = float(
            temp[
                "IBS"
            ]
            .iloc[0]
        )


        ax.plot(
            x,
            y,
            color=LSTM_COLOR,
            marker="D",
            markerfacecolor=WHITE,
            markeredgecolor=LSTM_COLOR,
            markeredgewidth=1.1,
            markersize=5.5,
            linewidth=2.0,
            label="LSTM",
        )


        ax.text(
            0.96,
            0.95,
            f"IBS={ibs:.4f}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            color=BLACK,
            fontweight="bold",
        )


        span = max(
            y.max()
            -
            y.min(),
            1.0,
        )


        ax.set_ylim(
            max(
                0,
                y.min()
                -
                0.08
                * span,
            ),
            y.max()
            +
            0.15
            * span,
        )


        ax.set_xlim(
            0.5,
            5.0,
        )


        ax.set_xticks(
            np.arange(
                0.5,
                5.1,
                0.5,
            )
        )


        ax.set_xlabel(
            "Years after landmark",
            color=BLACK,
        )


        ax.set_ylabel(
            r"Brier score ($\times 10^3$)",
            color=BLACK,
        )


        ax.set_title(
            landmark_title(
                landmark
            ),
            color=BLACK,
            fontweight="bold",
        )


        ax.yaxis.set_major_locator(
            MaxNLocator(
                nbins=7
            )
        )


        style_axis(
            ax
        )


    fig.suptitle(
        "Dynamic Brier score in geographic external validation",
        fontsize=14,
        fontweight="bold",
        color=BLACK,
    )


    save_figure(
        fig,
        "Figure3_Brier_score",
    )


# =============================================================================
# FIGURE 4
# FINAL CALIBRATION
#
# 只显示cross-fitted recalibrated LSTM
# =============================================================================

def plot_calibration():

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(
            11.4,
            9.0,
        ),
        constrained_layout=True,
        facecolor=WHITE,
    )


    axes = axes.flatten()


    for i, landmark in enumerate(
        PRIMARY_LANDMARKS
    ):

        ax = axes[
            i
        ]


        add_panel_label(
            ax,
            chr(
                65
                + i
            ),
        )


        temp = (
            calibration_df.loc[
                np.isclose(
                    calibration_df[
                        "landmark_year"
                    ],
                    landmark,
                )
            ]
            .sort_values(
                "risk_decile"
            )
        )


        predicted = (
            temp[
                "mean_predicted_risk"
            ]
            .to_numpy(
                float
            )
            * 100
        )


        observed = (
            temp[
                "observed_risk"
            ]
            .to_numpy(
                float
            )
            * 100
        )


        maximum = max(
            predicted.max(),
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
            color=PERFECT_COLOR,
            linewidth=1.15,
            label="Perfect calibration",
        )


        ax.plot(
            predicted,
            observed,
            color=LSTM_COLOR,
            marker="s",
            markerfacecolor=WHITE,
            markeredgecolor=LSTM_COLOR,
            markeredgewidth=1.15,
            markersize=5.5,
            linewidth=2.0,
            label="LSTM",
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


        title = (
            "Baseline landmark"
            if landmark == 0
            else
            f"ART year {int(landmark)} landmark"
        )


        ax.set_title(
            title
            +
            "\n5-year CKD risk",
            fontweight="bold",
            color=BLACK,
        )


        ax.set_xlabel(
            "Predicted 5-year risk (%)",
            color=BLACK,
        )


        ax.set_ylabel(
            "Observed 5-year risk (%)",
            color=BLACK,
        )


        style_axis(
            ax
        )


        legend = ax.legend(
            frameon=False,
            loc="upper left",
        )


        for text in legend.get_texts():

            text.set_color(
                BLACK
            )


    fig.suptitle(
        "Calibration of predicted 5-year CKD risk",
        fontsize=14,
        fontweight="bold",
        color=BLACK,
    )


    save_figure(
        fig,
        "Figure4_Calibration",
    )


# =============================================================================
# FIGURE 5
# FINAL DCA
#
# 只显示cross-fitted recalibrated LSTM
# =============================================================================

def plot_dca():

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(
            12.0,
            8.6,
        ),
        constrained_layout=True,
        facecolor=WHITE,
    )


    axes = axes.flatten()


    for i, landmark in enumerate(
        PRIMARY_LANDMARKS
    ):

        ax = axes[
            i
        ]


        add_panel_label(
            ax,
            chr(
                65
                + i
            ),
        )


        temp = (
            dca_df.loc[
                np.isclose(
                    dca_df[
                        "landmark_year"
                    ],
                    landmark,
                )
            ]
            .sort_values(
                "threshold_probability"
            )
        )


        x = (
            temp[
                "threshold_probability"
            ]
            .to_numpy(
                float
            )
            * 100
        )


        model = (
            temp[
                "LSTM_display_net_benefit"
            ]
            .to_numpy(
                float
            )
        )


        treat_all = (
            temp[
                "treat_all_net_benefit"
            ]
            .to_numpy(
                float
            )
        )


        ax.plot(
            x,
            model,
            color=LSTM_COLOR,
            linewidth=2.2,
            label="LSTM",
            zorder=4,
        )


        ax.plot(
            x,
            treat_all,
            "--",
            color=TREAT_ALL_COLOR,
            linewidth=1.15,
            label="Treat all",
            zorder=2,
        )


        ax.axhline(
            0,
            color=BLACK,
            linestyle=":",
            linewidth=1.15,
            label="Treat none",
            zorder=1,
        )


        relevant = np.concatenate([
            model,
            np.zeros_like(
                model
            ),
        ])


        ymin = min(
            -0.0015,
            relevant.min()
            -
            0.001,
        )


        ymax = max(
            0.0025,
            relevant.max()
            +
            0.0015,
        )


        ax.set_xlim(
            0.5,
            10.0,
        )


        ax.set_ylim(
            ymin,
            ymax,
        )


        title = (
            "Baseline landmark"
            if landmark == 0
            else
            f"ART year {int(landmark)} landmark"
        )


        ax.set_title(
            title
            +
            "\n5-year CKD risk",
            fontweight="bold",
            color=BLACK,
        )


        ax.set_xlabel(
            "Threshold probability (%)",
            color=BLACK,
        )


        ax.set_ylabel(
            "Net benefit",
            color=BLACK,
        )


        style_axis(
            ax
        )


        legend = ax.legend(
            frameon=False,
            loc="upper right",
        )


        for text in legend.get_texts():

            text.set_color(
                BLACK
            )


    fig.suptitle(
        "Decision curve analysis for predicted 5-year CKD risk",
        fontsize=14,
        fontweight="bold",
        color=BLACK,
    )


    save_figure(
        fig,
        "Figure5_DCA",
    )


# =============================================================================
# 15. FINAL NUMERICAL TABLE
#
# 每个主Landmark一行
#
# C-index/iAUC:
# primary frozen model
#
# Brier/IBS/calibration/DCA:
# cross-fitted recalibrated LSTM
# =============================================================================

def nearest_dca_value(
    landmark_year,
    threshold,
):

    temp = dca_df.loc[
        np.isclose(
            dca_df[
                "landmark_year"
            ],
            landmark_year,
        )
    ]


    position = np.argmin(
        np.abs(
            temp[
                "threshold_probability"
            ]
            .to_numpy(
                float
            )
            -
            threshold
        )
    )


    return float(
        temp[
            "LSTM_net_benefit"
        ]
        .iloc[
            position
        ]
    )


table_rows = []


for landmark_index, landmark_year in zip(
    PRIMARY_LANDMARK_INDEX,
    PRIMARY_LANDMARKS,
):

    # -------------------------------------------------------------------------
    # Discrimination
    # -------------------------------------------------------------------------

    perf = landmark_primary.loc[
        np.isclose(
            landmark_primary[
                "landmark_year"
            ],
            landmark_year,
        )
    ].iloc[
        0
    ]


    # -------------------------------------------------------------------------
    # Recalibrated Brier
    # -------------------------------------------------------------------------

    brier_temp = (
        brier_df.loc[
            np.isclose(
                brier_df[
                    "landmark_year"
                ],
                landmark_year,
            )
        ]
        .sort_values(
            "horizon_year"
        )
    )


    brier_1y = float(
        brier_temp.loc[
            np.isclose(
                brier_temp[
                    "horizon_year"
                ],
                1.0,
            ),
            "brier_score",
        ].iloc[
            0
        ]
    )


    brier_3y = float(
        brier_temp.loc[
            np.isclose(
                brier_temp[
                    "horizon_year"
                ],
                3.0,
            ),
            "brier_score",
        ].iloc[
            0
        ]
    )


    brier_5y = float(
        brier_temp.loc[
            np.isclose(
                brier_temp[
                    "horizon_year"
                ],
                5.0,
            ),
            "brier_score",
        ].iloc[
            0
        ]
    )


    ibs = float(
        brier_temp[
            "IBS"
        ]
        .iloc[
            0
        ]
    )


    # -------------------------------------------------------------------------
    # Calibration-in-the-large
    # -------------------------------------------------------------------------

    rows = np.where(
        metadata[
            "landmark_index"
        ]
        .to_numpy(
            int
        )
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


    predicted_5y = float(
        recal_risk[
            rows,
            9
        ].mean()
    )


    observed_5y = km_risk(
        event,
        time,
        60.0,
    )


    # -------------------------------------------------------------------------
    # DCA
    # -------------------------------------------------------------------------

    nb_1 = nearest_dca_value(
        landmark_year,
        0.01,
    )

    nb_3 = nearest_dca_value(
        landmark_year,
        0.03,
    )

    nb_5 = nearest_dca_value(
        landmark_year,
        0.05,
    )


    table_rows.append({

        "Landmark_year":
            landmark_year,

        "At_risk_n":
            int(
                perf[
                    "risk_set_n"
                ]
            ),

        "CKD_events_within_5y_n":
            int(
                perf[
                    "event_within_5y_n"
                ]
            ),

        "Uno_C_index_5y":
            float(
                perf[
                    "uno_c_index_5y"
                ]
            ),

        "Uno_C_lower_95CI":
            float(
                perf[
                    "uno_c_lower_95"
                ]
            ),

        "Uno_C_upper_95CI":
            float(
                perf[
                    "uno_c_upper_95"
                ]
            ),

        "iAUC":
            float(
                perf[
                    "integrated_dynamic_auc"
                ]
            ),

        "iAUC_lower_95CI":
            float(
                perf[
                    "iauc_lower_95"
                ]
            ),

        "iAUC_upper_95CI":
            float(
                perf[
                    "iauc_upper_95"
                ]
            ),

        "Brier_1y":
            brier_1y,

        "Brier_3y":
            brier_3y,

        "Brier_5y":
            brier_5y,

        "IBS":
            ibs,

        "Mean_predicted_5y_risk":
            predicted_5y,

        "KM_observed_5y_risk":
            observed_5y,

        "Calibration_error_pred_minus_obs":
            predicted_5y
            -
            observed_5y,

        "DCA_net_benefit_at_1pct":
            nb_1,

        "DCA_net_benefit_at_3pct":
            nb_3,

        "DCA_net_benefit_at_5pct":
            nb_5,
    })


final_table = pd.DataFrame(
    table_rows
)


# =============================================================================
# 16. SAVE TABLE
# =============================================================================

final_table.to_csv(
    OUT
    / "Table1_External_validation_results.csv",
    index=False,
    encoding="utf-8-sig",
)


final_table.to_excel(
    OUT
    / "Table1_External_validation_results.xlsx",
    index=False,
)


# =============================================================================
# 17. PRINT FORMATTED TABLE
# =============================================================================

display_table = final_table.copy()


for column in [

    "Uno_C_index_5y",
    "Uno_C_lower_95CI",
    "Uno_C_upper_95CI",

    "iAUC",
    "iAUC_lower_95CI",
    "iAUC_upper_95CI",

    "Brier_1y",
    "Brier_3y",
    "Brier_5y",
    "IBS",

    "Mean_predicted_5y_risk",
    "KM_observed_5y_risk",
    "Calibration_error_pred_minus_obs",

    "DCA_net_benefit_at_1pct",
    "DCA_net_benefit_at_3pct",
    "DCA_net_benefit_at_5pct",

]:

    display_table[
        column
    ] = (
        display_table[
            column
        ]
        .astype(
            float
        )
        .round(
            4
        )
    )


# =============================================================================
# 18. RUN
#
# 顺序固定：
# 1 C-index
# 2 iAUC
# 3 Brier
# 4 Calibration
# 5 DCA
# 6 Table
# =============================================================================

plot_c_index()

plot_iauc()

plot_brier()

plot_calibration()

plot_dca()


print()
print("=" * 150)

print(
    "Table 1. External validation results"
)

print("=" * 150)


print(
    display_table.to_string(
        index=False
    )
)


print()
print("=" * 150)

print(
    "最终仅输出5张结果图 + 1张数值表"
)

print()

print(
    "Figure 1: C-index"
)

print(
    "Figure 2: iAUC"
)

print(
    "Figure 3: Brier score"
)

print(
    "Figure 4: Calibration"
)

print(
    "Figure 5: DCA"
)

print(
    "Table 1 : External validation numerical results"
)

print()

print(
    "输出目录："
)

print(
    OUT
)

print("=" * 150)