# %matplotlib inline

# =============================================================================
# Step 11F v5
# 四模型核心论文可视化（Jupyter完整版）
#
# 模型显示名称：
#   Cox      -> Landmark Cox
#   RSF      -> Landmark RSF
#   RNN      -> RNN
#   LSTM-v2  -> LSTM
#
# Figure 1
#   0/1/3/5年Landmark预测未来5年
#   Uno C-index柱状图 + 95%CI
#   数值位于CI上方
#
# Figure 2
#   Dynamic AUC
#   0/1/3/5年Landmark分别预测未来0.5–5年
#   右上角显示iAUC
#
# Figure 3
#   Dynamic Brier
#   与AUC相同的四Panel结构
#   Brier × 10^3
#   四模型均使用不同颜色的平滑实线，不显示离散marker
#   使用PCHIP仅做显示插值，不改变原始Brier结果
#   IBS放在右上角预留空白区，避免与曲线重叠
#
# Figure 4
#   未来5年CKD风险校准
#   0/1/3/5年Landmark
#   四模型
#   无误差线
#
# Figure 5
#   未来5年CKD风险DCA
#   重新计算正确的60个月DCA
#   0/1/3/5年Landmark
#   横轴完整1%–50%
#   四模型曲线仅做轻度显示平滑
#   去除右下角inset，小图不再展示
#
# 输出：
#   Jupyter直接显示
#   PNG 600 dpi
#   PDF
#   SVG
# =============================================================================


# =============================================================================
# 1. 导入
# =============================================================================

from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from scipy.interpolate import PchipInterpolator

from sksurv.nonparametric import kaplan_meier_estimator


# =============================================================================
# 2. 项目路径
# =============================================================================

PROJECT_DIR = Path(
    "__CKD_WORKDIR__"
)


STEP6_DIR = (
    PROJECT_DIR
    / "rolling_5y_step6_super_landmark_data"
)


STEP11_DIR = (
    PROJECT_DIR
    / "rolling_5y_step11_v2_four_model_oof_comparison_lstm_v2"
)


OUT_DIR = (
    STEP11_DIR
    / "professional_figures_core_v5"
)


SOURCE_DIR = (
    OUT_DIR
    / "source_data"
)


OUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


SOURCE_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# =============================================================================
# 3. Step 11结果文件
# =============================================================================

LANDMARK_METRIC_FILE = (
    STEP11_DIR
    / "four_model_crossfit_calibrated_landmark_metrics.csv"
)


LANDMARK_BOOTSTRAP_FILE = (
    STEP11_DIR
    / "four_model_landmark_bootstrap_performance.csv"
)


HORIZON_METRIC_FILE = (
    STEP11_DIR
    / "four_model_crossfit_calibrated_horizon_metrics.csv"
)


CALIBRATION_DECILE_FILE = (
    STEP11_DIR
    / "four_model_calibration_deciles.csv"
)


# =============================================================================
# 4. DCA重新计算所需数据
# =============================================================================

ANALYSIS_TIME_FILE = (
    STEP6_DIR
    / "development_analysis_time_month.npy"
)


EVENT_FILE = (
    STEP6_DIR
    / "development_event_within_60m.npy"
)


LANDMARK_INDEX_FILE = (
    STEP6_DIR
    / "development_landmark_index_long.npy"
)


RISK_FILES = {

    "Cox":
        STEP11_DIR
        / "cox_crossfit_calibrated_oof_risk_long.npy",

    "RSF":
        STEP11_DIR
        / "rsf_crossfit_calibrated_oof_risk_long.npy",

    "RNN":
        STEP11_DIR
        / "rnn_crossfit_calibrated_oof_risk_long.npy",

    "LSTM-v2":
        STEP11_DIR
        / "lstm_v2_crossfit_calibrated_oof_risk_long.npy",
}


# =============================================================================
# 5. 固定设置
# =============================================================================

PRIMARY_LANDMARKS = [
    0.0,
    1.0,
    3.0,
    5.0
]


PRIMARY_LANDMARK_POSITIONS = {

    0.0: 0,

    1.0: 1,

    3.0: 3,

    5.0: 5
}


MODEL_ORDER = [
    "Cox",
    "RSF",
    "RNN",
    "LSTM-v2"
]


# =============================================================================
# 只改变LSTM-v2显示名称
# =============================================================================

MODEL_LABELS = {

    "Cox":
        "Landmark Cox",

    "RSF":
        "Landmark RSF",

    "RNN":
        "RNN",

    "LSTM-v2":
        "LSTM"
}


MODEL_MARKERS = {

    "Cox":
        "o",

    "RSF":
        "s",

    "RNN":
        "^",

    "LSTM-v2":
        "D"
}


MODEL_COLORS = {

    "Cox":
        "#1F77B4",

    "RSF":
        "#FF7F0E",

    "RNN":
        "#2CA02C",

    "LSTM-v2":
        "#D62728"
}


DPI = 600


# =============================================================================
# 6. DCA阈值范围
# =============================================================================

DCA_THRESHOLD_MIN = 0.01

DCA_THRESHOLD_MAX = 0.50

DCA_THRESHOLD_N = 300


# =============================================================================
# 7. 字体
# =============================================================================

def choose_font():

    candidates = [
        "Times New Roman",
        "Times New Roman PS MT",
        "Nimbus Roman",
        "Liberation Serif",
        "DejaVu Serif"
    ]


    for font_name in candidates:

        try:

            font_manager.findfont(
                font_name,
                fallback_to_default=False
            )

            return font_name

        except ValueError:

            continue


    return "DejaVu Serif"


FONT_NAME = choose_font()


# =============================================================================
# 8. 统一论文白底
# =============================================================================

plt.style.use("default")

plt.rcParams.update({

    "font.family":
        FONT_NAME,

    "font.serif": [
        "Times New Roman",
        "Times New Roman PS MT",
        "Nimbus Roman",
        "Liberation Serif",
        "DejaVu Serif"
    ],

    "mathtext.fontset":
        "custom",

    "mathtext.rm":
        FONT_NAME,

    "mathtext.it":
        f"{FONT_NAME}:italic",

    "mathtext.bf":
        f"{FONT_NAME}:bold",

    "mathtext.sf":
        FONT_NAME,

    "mathtext.tt":
        FONT_NAME,

    "mathtext.cal":
        FONT_NAME,

    "font.size":
        11,

    "figure.facecolor":
        "white",

    "axes.facecolor":
        "white",

    "savefig.facecolor":
        "white",

    "text.color":
        "black",

    "axes.labelcolor":
        "black",

    "axes.edgecolor":
        "black",

    "axes.titlecolor":
        "black",

    "legend.labelcolor":
        "black",

    "xtick.color":
        "black",

    "ytick.color":
        "black",

    "axes.titlesize":
        12.5,

    "axes.labelsize":
        11,

    "xtick.labelsize":
        10,

    "ytick.labelsize":
        10,

    "legend.fontsize":
        9,

    "figure.titlesize":
        14,

    "axes.spines.top":
        False,

    "axes.spines.right":
        False,

    "axes.linewidth":
        0.8,

    "lines.linewidth":
        2.0,

    "pdf.fonttype":
        42,

    "ps.fonttype":
        42,

    "savefig.dpi":
        DPI
})


# =============================================================================
# 9. 通用函数
# =============================================================================

def require_file(
    path
):

    if not path.exists():

        raise FileNotFoundError(
            f"找不到必要文件：\n{path}"
        )


def read_csv_checked(
    path
):

    require_file(
        path
    )


    df = pd.read_csv(
        path
    )


    if len(
        df
    ) == 0:

        raise ValueError(
            f"文件为空：\n{path}"
        )


    return df


def force_all_text_black(fig):
    """强制图中所有文字为黑色和统一的Times New Roman字体。"""
    from matplotlib.text import Text

    for text_obj in fig.findobj(match=Text):
        text_obj.set_color("black")
        text_obj.set_fontfamily(FONT_NAME)


def save_and_show(
    fig,
    stem
):

    force_all_text_black(fig)

    png_path = (
        OUT_DIR
        / f"{stem}.png"
    )


    pdf_path = (
        OUT_DIR
        / f"{stem}.pdf"
    )


    svg_path = (
        OUT_DIR
        / f"{stem}.svg"
    )


    fig.savefig(

        png_path,

        dpi=DPI,

        bbox_inches="tight",

        facecolor="white"
    )


    fig.savefig(

        pdf_path,

        bbox_inches="tight",

        facecolor="white"
    )


    fig.savefig(

        svg_path,

        bbox_inches="tight",

        facecolor="white"
    )


    plt.show()


    plt.close(
        fig
    )


    print(
        f"已保存：{png_path}"
    )


def add_panel_label(
    ax,
    label
):

    ax.text(

        -0.11,

        1.07,

        label,

        transform=ax.transAxes,

        fontsize=14,

        fontweight="bold",

        ha="left",

        va="top",

        color="black"
    )


def style_axis(
    ax,
    grid_axis="both"
):

    ax.grid(

        axis=grid_axis,

        linestyle="--",

        linewidth=0.6,

        alpha=0.25,

        color="#A0A0A0"
    )


    ax.set_axisbelow(
        True
    )


# =============================================================================
# 10. 读取结果
# =============================================================================

landmark_df = read_csv_checked(
    LANDMARK_METRIC_FILE
)


bootstrap_df = read_csv_checked(
    LANDMARK_BOOTSTRAP_FILE
)


horizon_df = read_csv_checked(
    HORIZON_METRIC_FILE
)


calibration_df = read_csv_checked(
    CALIBRATION_DECILE_FILE
)


# =============================================================================
# 11. 仅保留cross-fitted calibrated结果
# =============================================================================

if (
    "prediction_stage"
    in landmark_df.columns
):

    landmark_df = (

        landmark_df.loc[

            landmark_df[
                "prediction_stage"
            ]
            == "crossfit_calibrated_oof"
        ]

        .copy()
    )


if (
    "prediction_stage"
    in horizon_df.columns
):

    horizon_df = (

        horizon_df.loc[

            horizon_df[
                "prediction_stage"
            ]
            == "crossfit_calibrated_oof"
        ]

        .copy()
    )


# =============================================================================
# 12. 主Landmark
# =============================================================================

landmark_primary = (

    landmark_df.loc[

        landmark_df[
            "landmark_year"
        ].isin(
            PRIMARY_LANDMARKS
        )
    ]

    .copy()
)


horizon_primary = (

    horizon_df.loc[

        horizon_df[
            "landmark_year"
        ].isin(
            PRIMARY_LANDMARKS
        )
    ]

    .copy()
)


calibration_5y = (

    calibration_df.loc[

        calibration_df[
            "landmark_year"
        ].isin(
            PRIMARY_LANDMARKS
        )

        &

        (
            calibration_df[
                "horizon_year"
            ]
            == 5.0
        )
    ]

    .copy()
)


print("=" * 100)

print(
    "Step 11F v5：可视化数据读取完成"
)

print("=" * 100)

print(
    "主Landmark性能：",
    landmark_primary.shape
)

print(
    "动态AUC/Brier：",
    horizon_primary.shape
)

print(
    "5年校准：",
    calibration_5y.shape
)

print("=" * 100)


# =============================================================================
# FIGURE 1
# C-index柱状图
# =============================================================================

def plot_figure1_cindex():

    fig, ax = plt.subplots(

        figsize=(
            11.8,
            6.6
        )
    )


    x = np.arange(
        len(
            PRIMARY_LANDMARKS
        )
    )


    n_models = len(
        MODEL_ORDER
    )


    total_width = 0.82


    bar_width = (
        total_width
        / n_models
    )


    offsets = (

        np.arange(
            n_models
        )

        -

        (
            n_models
            - 1
        )
        / 2

    ) * bar_width


    all_lower = []

    all_upper = []


    for model_i, model in enumerate(
        MODEL_ORDER
    ):


        model_data = (

            landmark_primary.loc[

                landmark_primary[
                    "model"
                ]
                == model
            ]

            .sort_values(
                "landmark_year"
            )
        )


        ci_data = (

            bootstrap_df.loc[

                (
                    bootstrap_df[
                        "model"
                    ]
                    == model
                )

                &

                (
                    bootstrap_df[
                        "metric"
                    ]
                    == "uno_c_index_5y"
                )

                &

                bootstrap_df[
                    "landmark_year"
                ].isin(
                    PRIMARY_LANDMARKS
                )
            ]

            .sort_values(
                "landmark_year"
            )
        )


        if (
            len(
                model_data
            )
            != 4
            or len(
                ci_data
            )
            != 4
        ):

            raise ValueError(
                f"{model} C-index数据不完整"
            )


        values = (

            model_data[
                "uno_c_index_5y"
            ]

            .to_numpy(
                float
            )
        )


        lower = (

            ci_data[
                "lower_95"
            ]

            .to_numpy(
                float
            )
        )


        upper = (

            ci_data[
                "upper_95"
            ]

            .to_numpy(
                float
            )
        )


        all_lower.extend(
            lower.tolist()
        )


        all_upper.extend(
            upper.tolist()
        )


        yerr = np.vstack([

            values
            - lower,

            upper
            - values
        ])


        bars = ax.bar(

            x
            + offsets[
                model_i
            ],

            values,

            width=
                bar_width
                * 0.90,

            color=
                MODEL_COLORS[
                    model
                ],

            label=
                MODEL_LABELS[
                    model
                ],

            yerr=yerr,

            error_kw={

                "elinewidth":
                    1.4,

                "capsize":
                    4,

                "capthick":
                    1.3,

                "ecolor":
                    "#555555"
            },

            edgecolor="white",

            linewidth=0.7,

            alpha=0.95
        )


        # =====================================================================
        # C-index具体值置于CI上方
        # =====================================================================

        for (
            bar,
            value,
            ci_upper
        ) in zip(
            bars,
            values,
            upper
        ):


            ax.text(

                bar.get_x()
                + bar.get_width()
                / 2,

                ci_upper
                + 0.0042,

                f"{value:.3f}",

                ha="center",

                va="bottom",

                fontsize=9,

                fontweight="bold",

                color="black"
            )


    ax.set_xticks(
        x
    )


    ax.set_xticklabels([

        "Baseline\n(0 y)",

        "1 y",

        "3 y",

        "5 y"
    ])


    ax.set_xlabel(
        "ART landmark"
    )


    ax.set_ylabel(
        "Uno C-index for future 5-year CKD risk"
    )


    ax.set_title(

        "Five-year discrimination across dynamic prediction landmarks",

        fontweight="bold",

        pad=16
    )


    ax.set_ylim(

        min(
            all_lower
        )
        - 0.015,

        max(
            all_upper
        )
        + 0.027
    )


    style_axis(
        ax,
        grid_axis="y"
    )


    ax.legend(

        loc="upper center",

        bbox_to_anchor=(
            0.5,
            -0.12
        ),

        ncol=4,

        frameon=False
    )


    fig.tight_layout()


    save_and_show(

        fig,

        "Figure1_Cindex_bar"
    )


plot_figure1_cindex()


# =============================================================================
# FIGURE 2
# Dynamic AUC
# =============================================================================

def plot_figure2_dynamic_auc():

    fig, axes = plt.subplots(

        2,
        2,

        figsize=(
            12.8,
            9.6
        ),

        constrained_layout=True,

        sharex=True,

        sharey=True
    )


    axes = axes.flatten()


    all_auc = (

        horizon_primary[
            "dynamic_auc"
        ]

        .to_numpy(
            float
        )
    )


    auc_ymin = max(

        0.70,

        all_auc.min()
        - 0.025
    )


    auc_ymax = min(

        1.00,

        all_auc.max()
        + 0.025
    )


    for panel_i, landmark in enumerate(
        PRIMARY_LANDMARKS
    ):


        ax = axes[
            panel_i
        ]


        add_panel_label(

            ax,

            chr(
                65
                + panel_i
            )
        )


        landmark_data = (

            horizon_primary.loc[

                horizon_primary[
                    "landmark_year"
                ]
                == landmark
            ]

            .copy()
        )


        for model in MODEL_ORDER:


            temp = (

                landmark_data.loc[

                    landmark_data[
                        "model"
                    ]
                    == model
                ]

                .sort_values(
                    "horizon_year"
                )
            )


            summary = (

                landmark_primary.loc[

                    (
                        landmark_primary[
                            "model"
                        ]
                        == model
                    )

                    &

                    (
                        landmark_primary[
                            "landmark_year"
                        ]
                        == landmark
                    )
                ]
            )


            iauc = float(

                summary[
                    "integrated_dynamic_auc"
                ]
                .iloc[
                    0
                ]
            )


            ax.plot(

                temp[
                    "horizon_year"
                ],

                temp[
                    "dynamic_auc"
                ],

                color=
                    MODEL_COLORS[
                        model
                    ],

                marker=
                    MODEL_MARKERS[
                        model
                    ],

                markersize=5.5,

                linewidth=2.2,

                label=(

                    f"{MODEL_LABELS[model]}  "
                    f"iAUC={iauc:.3f}"
                )
            )


        if landmark == 0:

            title = (
                "Baseline landmark\n"
                "Predicting ART years 0–5"
            )

        else:

            title = (

                f"ART year {int(landmark)} landmark\n"

                f"Predicting ART years "
                f"{int(landmark)}–"
                f"{int(landmark + 5)}"
            )


        ax.set_title(

            title,

            fontweight="bold"
        )


        ax.set_xlabel(
            "Years after landmark"
        )


        ax.set_ylabel(
            "Time-dependent AUC"
        )


        ax.set_xlim(
            0.5,
            5.0
        )


        ax.set_ylim(
            auc_ymin,
            auc_ymax
        )


        ax.set_xticks(

            np.arange(
                0.5,
                5.1,
                0.5
            )
        )


        style_axis(
            ax
        )


        ax.legend(

            loc="upper right",

            frameon=False,

            fontsize=8.3
        )


    fig.suptitle(

        "Dynamic AUC for CKD prediction over the subsequent 5 years",

        fontsize=15,

        fontweight="bold",

        y=1.02
    )


    save_and_show(

        fig,

        "Figure2_Dynamic_AUC"
    )


plot_figure2_dynamic_auc()


# =============================================================================
# FIGURE 3
#
# Dynamic Brier
#
# 修改重点：
#
# 1. 与AUC完全相同的4个Panel
# 2. Brier乘以1000显示
# 3. 每个Panel独立Y轴
# 4. 紧缩Y轴范围
#
# 目的：
# 放大4模型之间非常小但真实存在的差异
#
# 原始数据和IBS不变。
# =============================================================================

def plot_figure3_dynamic_brier():

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(12.8, 9.6),
        constrained_layout=True,
        sharex=True,
        sharey=False
    )

    axes = axes.flatten()

    # =========================================================================
    # 仅用于可视化的平滑横轴。
    # 原始Brier仍然只来自0.5、1.0、...、5.0年的10个评价时间点；
    # PCHIP插值只用于连接这些原始点，使曲线连续平滑，不修改任何统计结果。
    # =========================================================================
    smooth_horizon_grid = np.linspace(
        0.5,
        5.0,
        300
    )

    for panel_i, landmark in enumerate(PRIMARY_LANDMARKS):

        ax = axes[panel_i]

        add_panel_label(
            ax,
            chr(65 + panel_i)
        )

        landmark_data = (
            horizon_primary.loc[
                horizon_primary["landmark_year"] == landmark
            ]
            .copy()
        )

        # =====================================================================
        # 原始Brier ×1000，仅改变显示尺度，不改变结果。
        # =====================================================================
        panel_brier = (
            landmark_data["brier_score"]
            .to_numpy(float)
            * 1000
        )

        panel_min = float(panel_brier.min())
        panel_max = float(panel_brier.max())
        panel_span = panel_max - panel_min

        if panel_span <= 0:
            panel_span = 1.0

        # 下方仅留少量空白；顶部额外留约22%，专门放置IBS文字框。
        # 不改变Brier值，只改变坐标显示范围。
        brier_ymin = max(
            0,
            panel_min - 0.04 * panel_span
        )

        brier_ymax = (
            panel_max + 0.22 * panel_span
        )

        ibs_lines = []

        for model in MODEL_ORDER:

            temp = (
                landmark_data.loc[
                    landmark_data["model"] == model
                ]
                .sort_values("horizon_year")
            )

            if len(temp) == 0:
                raise ValueError(
                    f"缺少Brier数据：{model}, Landmark={landmark}"
                )

            summary = (
                landmark_primary.loc[
                    (landmark_primary["model"] == model)
                    &
                    (landmark_primary["landmark_year"] == landmark)
                ]
            )

            if len(summary) != 1:
                raise ValueError(
                    f"无法找到唯一IBS：{model}, Landmark={landmark}"
                )

            ibs = float(
                summary["integrated_brier"].iloc[0]
            )

            ibs_lines.append(
                f"{MODEL_LABELS[model]}  IBS={ibs:.4f}"
            )

            x_raw = (
                temp["horizon_year"]
                .to_numpy(dtype=float)
            )

            y_raw = (
                temp["brier_score"]
                .to_numpy(dtype=float)
                * 1000
            )

            # =================================================================
            # PCHIP为形状保持插值：
            # - 平滑通过每一个原始Brier评价点；
            # - 不像普通高阶样条那样容易产生过冲；
            # - 仅用于图形显示。
            # =================================================================
            if len(x_raw) >= 3:
                interpolator = PchipInterpolator(
                    x_raw,
                    y_raw,
                    extrapolate=False
                )

                y_smooth = interpolator(
                    smooth_horizon_grid
                )

                x_plot = smooth_horizon_grid

            else:
                x_plot = x_raw
                y_smooth = y_raw

            # =================================================================
            # 四模型全部使用不同颜色的实线。
            # 不再绘制每个时间点marker，避免“每个点一个符号”的视觉效果。
            # =================================================================
            ax.plot(
                x_plot,
                y_smooth,
                color=MODEL_COLORS[model],
                linestyle="-",
                linewidth=2.25,
                alpha=0.98,
                label=MODEL_LABELS[model],
                zorder=3
            )

        if landmark == 0:
            title = (
                "Baseline landmark\n"
                "Predicting ART years 0–5"
            )
        else:
            title = (
                f"ART year {int(landmark)} landmark\n"
                f"Predicting ART years "
                f"{int(landmark)}–"
                f"{int(landmark + 5)}"
            )

        ax.set_title(
            title,
            fontweight="bold",
            color="black"
        )

        ax.set_xlabel(
            "Years after landmark",
            color="black"
        )

        ax.set_ylabel(
            r"Brier score ($\times 10^3$)",
            color="black"
        )

        ax.set_xlim(
            0.5,
            5.0
        )

        ax.set_ylim(
            brier_ymin,
            brier_ymax
        )

        ax.set_xticks(
            np.arange(
                0.5,
                5.1,
                0.5
            )
        )

        # 适当增加纵轴刻度密度，仅改善读取，不改变Brier数值。
        ax.yaxis.set_major_locator(
            MaxNLocator(nbins=8)
        )

        style_axis(ax)

        # 左上角仅显示模型名称。
        legend = ax.legend(
            loc="upper left",
            frameon=False,
            fontsize=8.0,
            handlelength=2.5,
            borderaxespad=0.4
        )

        for legend_text in legend.get_texts():
            legend_text.set_color("black")

        # =====================================================================
        # IBS单独置于右上角白色注释框。
        # 顶部已预留额外空间，尽量避免与5年Brier曲线重叠。
        # =====================================================================
        ax.text(
            0.985,
            0.985,
            "\n".join(ibs_lines),
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=7.7,
            color="black",
            linespacing=1.20,
            bbox=dict(
                boxstyle="round,pad=0.30",
                facecolor="white",
                edgecolor="#B0B0B0",
                linewidth=0.7,
                alpha=0.96
            ),
            zorder=10
        )

        ax.tick_params(
            axis="both",
            colors="black"
        )

    fig.suptitle(
        "Dynamic Brier score for CKD prediction over the subsequent 5 years",
        fontsize=15,
        fontweight="bold",
        color="black",
        y=1.02
    )

    save_and_show(
        fig,
        "Figure3_Dynamic_Brier_magnified"
    )


plot_figure3_dynamic_brier()


# =============================================================================
# FIGURE 4
# 5年校准
# 无误差线
# =============================================================================

def plot_figure4_calibration():

    fig, axes = plt.subplots(

        2,
        2,

        figsize=(
            12.7,
            10.0
        ),

        constrained_layout=True
    )


    axes = axes.flatten()


    for panel_i, landmark in enumerate(
        PRIMARY_LANDMARKS
    ):


        ax = axes[
            panel_i
        ]


        add_panel_label(

            ax,

            chr(
                65
                + panel_i
            )
        )


        landmark_data = (

            calibration_5y.loc[

                calibration_5y[
                    "landmark_year"
                ]
                == landmark
            ]

            .copy()
        )


        max_value = 0.0


        for model in MODEL_ORDER:


            temp = (

                landmark_data.loc[

                    landmark_data[
                        "model"
                    ]
                    == model
                ]

                .sort_values(
                    "risk_group"
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
                    "km_observed_risk"
                ]

                .to_numpy(
                    float
                )

                * 100
            )


            max_value = max(

                max_value,

                float(
                    predicted.max()
                ),

                float(
                    observed.max()
                )
            )


            # =================================================================
            # 不再添加任何误差线
            # =================================================================

            ax.plot(

                predicted,

                observed,

                color=
                    MODEL_COLORS[
                        model
                    ],

                marker=
                    MODEL_MARKERS[
                        model
                    ],

                markersize=5.5,

                linewidth=1.9,

                label=
                    MODEL_LABELS[
                        model
                    ]
            )


        max_value *= 1.08


        ax.plot(

            [
                0,
                max_value
            ],

            [
                0,
                max_value
            ],

            color="#555555",

            linestyle="--",

            linewidth=1.2,

            label="Perfect calibration"
        )


        if landmark == 0:

            title = (
                "Baseline landmark\n"
                "5-year CKD risk"
            )

        else:

            title = (

                f"ART year {int(landmark)} landmark\n"

                "5-year CKD risk"
            )


        ax.set_title(

            title,

            fontweight="bold"
        )


        ax.set_xlabel(
            "Predicted 5-year risk (%)"
        )


        ax.set_ylabel(
            "Observed 5-year risk (%)"
        )


        ax.set_xlim(
            0,
            max_value
        )


        ax.set_ylim(
            0,
            max_value
        )


        ax.set_aspect(
            "equal",
            adjustable="box"
        )


        style_axis(
            ax
        )


        handles, labels = (
            ax.get_legend_handles_labels()
        )


        perfect_index = labels.index(
            "Perfect calibration"
        )


        ordered_handles = (

            [
                handles[
                    perfect_index
                ]
            ]

            +

            [
                h

                for i, h
                in enumerate(
                    handles
                )

                if i
                != perfect_index
            ]
        )


        ordered_labels = (

            [
                labels[
                    perfect_index
                ]
            ]

            +

            [
                label

                for i, label
                in enumerate(
                    labels
                )

                if i
                != perfect_index
            ]
        )


        ax.legend(

            ordered_handles,

            ordered_labels,

            loc="upper left",

            frameon=False,

            fontsize=8.2
        )


    fig.suptitle(

        "Calibration of predicted 5-year CKD risk",

        fontsize=15,

        fontweight="bold",

        y=1.02
    )


    save_and_show(

        fig,

        "Figure4_5year_Calibration"
    )


plot_figure4_calibration()


# =============================================================================
# 13. DCA输入读取
# =============================================================================

for path in [

    ANALYSIS_TIME_FILE,

    EVENT_FILE,

    LANDMARK_INDEX_FILE

]:

    require_file(
        path
    )


for path in RISK_FILES.values():

    require_file(
        path
    )


analysis_time_month = np.load(
    ANALYSIS_TIME_FILE
).astype(
    float
)


event_within_60m = np.load(
    EVENT_FILE
).astype(
    bool
)


landmark_index_long = np.load(
    LANDMARK_INDEX_FILE
).astype(
    int
)


risk_arrays = {

    model:
        np.load(
            path
        ).astype(
            float
        )

    for model, path
    in RISK_FILES.items()
}


# =============================================================================
# 14. 正确DCA辅助函数
# =============================================================================

def step_function_value(
    time_grid,
    probability,
    query_time
):

    time_grid = np.asarray(
        time_grid,
        dtype=float
    )


    probability = np.asarray(
        probability,
        dtype=float
    )


    query = np.asarray(
        query_time,
        dtype=float
    )


    indices = np.searchsorted(

        time_grid,

        query,

        side="right"

    ) - 1


    result = np.ones_like(
        query,
        dtype=float
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


def build_ipcw_binary_outcome(
    event,
    time_month,
    horizon_month
):

    event = np.asarray(
        event,
        dtype=bool
    )


    time_month = np.asarray(
        time_month,
        dtype=float
    )


    # =========================================================================
    # 5年内发生CKD
    # =========================================================================

    case = (

        event

        &

        (
            time_month
            <= horizon_month
        )
    )


    # =========================================================================
    # 修正后的5年无事件Control：
    # >= 60，而不是 > 60
    # =========================================================================

    control = (

        (~event)

        &

        (
            time_month
            >= horizon_month
        )
    )


    # =========================================================================
    # Reverse KM估计删失分布
    # =========================================================================

    (
        censor_time,
        censor_survival
    ) = kaplan_meier_estimator(

        event,

        time_month,

        reverse=True
    )


    weights = np.zeros(
        len(
            event
        ),
        dtype=float
    )


    # =========================================================================
    # Case：
    # 1/G(T-)
    # =========================================================================

    if np.any(
        case
    ):


        case_query = np.nextafter(

            time_month[
                case
            ],

            -np.inf
        )


        g_case = step_function_value(

            censor_time,

            censor_survival,

            case_query
        )


        weights[
            case
        ] = (

            1.0

            /

            np.clip(
                g_case,
                1e-6,
                None
            )
        )


    # =========================================================================
    # Control：
    # 1/G(60-)
    # =========================================================================

    if np.any(
        control
    ):


        horizon_minus = np.nextafter(

            float(
                horizon_month
            ),

            -np.inf
        )


        g_horizon = float(

            step_function_value(

                censor_time,

                censor_survival,

                horizon_minus
            )
        )


        weights[
            control
        ] = (

            1.0

            /

            max(
                g_horizon,
                1e-6
            )
        )


    outcome = case.astype(
        int
    )


    known = (
        case
        | control
    )


    return (
        outcome,
        weights,
        known
    )


def calculate_dca(
    predicted_risk,
    event,
    time_month,
    horizon_month,
    thresholds
):

    predicted_risk = np.asarray(
        predicted_risk,
        dtype=float
    )


    thresholds = np.asarray(
        thresholds,
        dtype=float
    )


    (
        outcome,
        weights,
        known
    ) = build_ipcw_binary_outcome(

        event,
        time_month,
        horizon_month
    )


    n_total = float(
        len(
            predicted_risk
        )
    )


    weighted_event_total = float(

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


    weighted_nonevent_total = float(

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


    model_nb = np.empty(
        len(
            thresholds
        )
    )


    treat_all_nb = np.empty(
        len(
            thresholds
        )
    )


    treat_none_nb = np.zeros(
        len(
            thresholds
        )
    )


    for i, threshold in enumerate(
        thresholds
    ):


        odds = (

            threshold

            /

            (
                1
                - threshold
            )
        )


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


        model_nb[
            i
        ] = (

            tp
            / n_total

            -

            (
                fp
                / n_total
            )

            * odds
        )


        treat_all_nb[
            i
        ] = (

            weighted_event_total
            / n_total

            -

            (
                weighted_nonevent_total
                / n_total
            )

            * odds
        )


    return (

        model_nb,

        treat_all_nb,

        treat_none_nb
    )


# =============================================================================
# 15. 重新计算DCA
# =============================================================================

thresholds = np.linspace(

    DCA_THRESHOLD_MIN,

    DCA_THRESHOLD_MAX,

    DCA_THRESHOLD_N
)


dca_records = []


for landmark in PRIMARY_LANDMARKS:


    landmark_position = (
        PRIMARY_LANDMARK_POSITIONS[
            landmark
        ]
    )


    rows = np.where(

        landmark_index_long
        == landmark_position

    )[0]


    event = (
        event_within_60m[
            rows
        ]
    )


    time_month = (
        analysis_time_month[
            rows
        ]
    )


    reference_all = None

    reference_none = None


    for model in MODEL_ORDER:


        risk_5y = (

            risk_arrays[
                model
            ][
                rows,
                -1
            ]
        )


        (
            model_nb,
            treat_all,
            treat_none
        ) = calculate_dca(

            predicted_risk=
                risk_5y,

            event=
                event,

            time_month=
                time_month,

            horizon_month=
                60.0,

            thresholds=
                thresholds
        )


        if (
            reference_all
            is None
        ):

            reference_all = (
                treat_all
            )

            reference_none = (
                treat_none
            )


        for threshold, nb in zip(

            thresholds,

            model_nb
        ):


            dca_records.append({

                "landmark_year":
                    landmark,

                "model":
                    model,

                "threshold_probability":
                    float(
                        threshold
                    ),

                "net_benefit":
                    float(
                        nb
                    )
            })


    for threshold, nb_all, nb_none in zip(

        thresholds,

        reference_all,

        reference_none
    ):


        dca_records.append({

            "landmark_year":
                landmark,

            "model":
                "Treat all",

            "threshold_probability":
                float(
                    threshold
                ),

            "net_benefit":
                float(
                    nb_all
                )
        })


        dca_records.append({

            "landmark_year":
                landmark,

            "model":
                "Treat none",

            "threshold_probability":
                float(
                    threshold
                ),

            "net_benefit":
                float(
                    nb_none
                )
        })


corrected_dca_df = pd.DataFrame(
    dca_records
)


# =============================================================================
# FIGURE 5
#
# DCA：
#
# 主图：
#   放大模型Net Benefit区域
#
# inset：
#   展示完整Treat all下降范围
#
# 横轴两者均保持1%–50%
#
# 因此：
#   模型间差异清楚
#   完整性也保留
# =============================================================================

def smooth_dca_for_display(values, window=11):
    """
    仅用于DCA绘图显示的轻度中心移动平均。

    - corrected_dca_df中的原始Net Benefit完全不修改；
    - 只平滑四个模型曲线；
    - Treat all / Treat none保持原始曲线；
    - window=11相对于300个阈值点属于轻度平滑。
    """

    values = np.asarray(values, dtype=float)

    if len(values) < 3:
        return values.copy()

    window = int(window)

    if window < 3:
        return values.copy()

    if window % 2 == 0:
        window += 1

    window = min(window, len(values))

    if window % 2 == 0:
        window -= 1

    if window < 3:
        return values.copy()

    smoothed = (
        pd.Series(values)
        .rolling(
            window=window,
            center=True,
            min_periods=1
        )
        .mean()
        .to_numpy(dtype=float)
    )

    # 保留首尾原始值，避免边缘发生不必要漂移。
    smoothed[0] = values[0]
    smoothed[-1] = values[-1]

    return smoothed


def plot_figure5_dca():

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(13.2, 9.8),
        constrained_layout=True
    )

    axes = axes.flatten()

    for panel_i, landmark in enumerate(PRIMARY_LANDMARKS):

        ax = axes[panel_i]

        add_panel_label(
            ax,
            chr(65 + panel_i)
        )

        landmark_data = (
            corrected_dca_df.loc[
                corrected_dca_df["landmark_year"] == landmark
            ]
            .copy()
        )

        # 用于确定主图Y轴范围：只使用四个模型 + Treat none，
        # 防止Treat all在高阈值处的大负值把模型差异压缩在顶部。
        model_display_values = []

        # =====================================================================
        # 四个模型：仅显示层面轻度平滑
        # =====================================================================
        for model in MODEL_ORDER:

            temp = (
                landmark_data.loc[
                    landmark_data["model"] == model
                ]
                .sort_values("threshold_probability")
            )

            if len(temp) == 0:
                raise ValueError(
                    f"缺少DCA数据：{model}, Landmark={landmark}"
                )

            x = (
                temp["threshold_probability"]
                .to_numpy(float)
                * 100
            )

            y_raw = (
                temp["net_benefit"]
                .to_numpy(float)
            )

            y_plot = smooth_dca_for_display(
                y_raw,
                window=11
            )

            model_display_values.extend(
                y_plot.tolist()
            )

            ax.plot(
                x,
                y_plot,
                color=MODEL_COLORS[model],
                linewidth=2.25,
                alpha=0.98,
                label=MODEL_LABELS[model],
                zorder=3
            )

        # =====================================================================
        # Treat all：保持原始值，不做平滑
        # =====================================================================
        treat_all = (
            landmark_data.loc[
                landmark_data["model"] == "Treat all"
            ]
            .sort_values("threshold_probability")
        )

        ax.plot(
            treat_all["threshold_probability"] * 100,
            treat_all["net_benefit"],
            color="#777777",
            linestyle="--",
            linewidth=1.35,
            alpha=0.90,
            label="Treat all",
            zorder=2
        )

        # =====================================================================
        # Treat none：保持原始值
        # =====================================================================
        treat_none = (
            landmark_data.loc[
                landmark_data["model"] == "Treat none"
            ]
            .sort_values("threshold_probability")
        )

        ax.plot(
            treat_none["threshold_probability"] * 100,
            treat_none["net_benefit"],
            color="black",
            linestyle=":",
            linewidth=1.4,
            alpha=0.95,
            label="Treat none",
            zorder=1
        )

        # =====================================================================
        # 去除原版右下角Full-scale inset。
        # 主图仍保留完整1%–50%阈值横轴。
        # =====================================================================
        model_display_values = np.asarray(
            model_display_values,
            dtype=float
        )

        zoom_min = float(
            min(
                model_display_values.min(),
                0.0
            )
        )

        zoom_max = float(
            model_display_values.max()
        )

        zoom_span = zoom_max - zoom_min

        if zoom_span <= 0:
            zoom_span = 0.01

        main_ymin = min(
            -0.003,
            zoom_min - 0.06 * zoom_span
        )

        main_ymax = (
            zoom_max + 0.08 * zoom_span
        )

        ax.set_ylim(
            main_ymin,
            main_ymax
        )

        ax.set_xlim(
            DCA_THRESHOLD_MIN * 100,
            DCA_THRESHOLD_MAX * 100
        )

        if landmark == 0:
            title = (
                "Baseline landmark\n"
                "5-year CKD risk"
            )
        else:
            title = (
                f"ART year {int(landmark)} landmark\n"
                "5-year CKD risk"
            )

        ax.set_title(
            title,
            fontweight="bold",
            color="black"
        )

        ax.set_xlabel(
            "Threshold probability (%)",
            color="black"
        )

        ax.set_ylabel(
            "Net benefit",
            color="black"
        )

        ax.xaxis.set_major_locator(
            MaxNLocator(nbins=6)
        )

        ax.yaxis.set_major_locator(
            MaxNLocator(nbins=7)
        )

        style_axis(ax)

        ax.tick_params(
            axis="both",
            colors="black"
        )

        legend = ax.legend(
            loc="lower left",
            frameon=False,
            fontsize=7.9,
            ncol=1
        )

        for text in legend.get_texts():
            text.set_color("black")

    fig.suptitle(
        "Decision curve analysis for predicted 5-year CKD risk",
        fontsize=15,
        fontweight="bold",
        color="black",
        y=1.02
    )

    save_and_show(
        fig,
        "Figure5_5year_DCA_zoom_and_full_scale"
    )


plot_figure5_dca()


# =============================================================================
# 16. Source Data
# =============================================================================

landmark_primary.to_csv(

    SOURCE_DIR
    / "Figure1_Cindex_source.csv",

    index=False,

    encoding="utf-8-sig"
)


horizon_primary.to_csv(

    SOURCE_DIR
    / "Figure2_3_AUC_Brier_source.csv",

    index=False,

    encoding="utf-8-sig"
)


calibration_5y.to_csv(

    SOURCE_DIR
    / "Figure4_Calibration_source.csv",

    index=False,

    encoding="utf-8-sig"
)


corrected_dca_df.to_csv(

    SOURCE_DIR
    / "Figure5_DCA_corrected_source.csv",

    index=False,

    encoding="utf-8-sig"
)


# =============================================================================
# 17. 完成
# =============================================================================

print()

print("=" * 100)

print(
    "Step 11F v5 全部完成"
)

print("=" * 100)

print()

print(
    "Figure 1："
    "C-index柱状图"
)

print(
    "Figure 2："
    "Dynamic AUC + iAUC"
)

print(
    "Figure 3："
    "Dynamic Brier ×10^3 + IBS，四模型均为不同颜色平滑实线，无离散marker"
)

print(
    "Figure 4："
    "5年校准，无误差线"
)

print(
    "Figure 5："
    "5年DCA，四模型轻度平滑 + 去除右下角inset"
)

print()

print(
    "模型显示："
)

print(
    "Cox      -> Landmark Cox"
)

print(
    "RSF      -> Landmark RSF"
)

print(
    "RNN      -> RNN"
)

print(
    "LSTM-v2  -> LSTM"
)

print()

print(
    "输出目录："
)

print(
    OUT_DIR
)

print("=" * 100)
