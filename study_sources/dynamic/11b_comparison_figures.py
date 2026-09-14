#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Step 11E v2：四模型开发集 OOF 结果专业可视化
================================================

对应当前正式 Step 11 v2：
  Cox / RSF / RNN / LSTM-v2

主要修正
--------
1. 与当前 Step 11 v2 的真实列名完全对齐：
   mean_uno_c_index_5y_lower_95 / upper_95
   mean_integrated_dynamic_auc_lower_95 / upper_95
   mean_integrated_brier_lower_95 / upper_95

2. 当前 paired bootstrap 为长格式：
   model_a / model_b / metric /
   difference_a_minus_b /
   difference_lower_95 /
   difference_upper_95
   本代码直接支持。

3. 同时保留旧版列名兼容逻辑。
   如果未来读取旧CSV，也可以自动识别。

4. 运行前自动审计6个CSV文件的：
   - 行数
   - 列名
   - 模型
   - Landmark
   - horizon

5. 输出专业论文图：
   Figure 1  四模型主性能 + 95%CI
   Figure 2  患者级配对Bootstrap森林图
   Figure 3  1/3/5年动态AUC和Brier
   Figure 4  Calibration-in-the-large
   Figure 5  LSTM-v2十分位校准（3×4）
   Figure 6  DCA（3×4）

6. 每张图同时输出：
   - PNG 600 dpi
   - PDF
   - SVG

7. 导出每张图对应的 source data。

注意
----
- 本步骤只可视化，不重新训练任何模型。
- 主文重点Landmark：ART后0、1、3、5年。
- 最终候选模型：LSTM-v2。
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, FuncFormatter
import numpy as np
import pandas as pd


# =============================================================================
# 0. 路径与固定设置
# =============================================================================

STEP11_DIR = Path(
    "__CKD_WORKDIR__/"
    "rolling_5y_step11_v2_four_model_oof_comparison_lstm_v2"
)

OUT_DIR = STEP11_DIR / "professional_figures_v2"
SOURCE_DIR = OUT_DIR / "source_data"

OUT_DIR.mkdir(parents=True, exist_ok=True)
SOURCE_DIR.mkdir(parents=True, exist_ok=True)

PRIMARY_SCOPE = (
    "primary_0_1_3_5y_landmark_equal_weight_mean"
)

PRIMARY_LANDMARK_YEARS = [
    0.0,
    1.0,
    3.0,
    5.0,
]

HORIZON_YEARS = [
    1.0,
    3.0,
    5.0,
]

MODEL_ORDER = [
    "Cox",
    "RSF",
    "RNN",
    "LSTM-v2",
]

MODEL_LABELS = {
    "Cox": "Landmark Cox",
    "RSF": "Landmark RSF",
    "RNN": "RNN",
    "LSTM-v2": "LSTM-v2",
}

FINAL_MODEL = "LSTM-v2"

PAIR_ORDER = [
    ("RSF", "Cox"),
    ("RNN", "Cox"),
    ("LSTM-v2", "Cox"),
    ("RSF", "RNN"),
    ("LSTM-v2", "RSF"),
    ("LSTM-v2", "RNN"),
]

PAIR_LABELS = {
    ("RSF", "Cox"): "RSF vs Cox",
    ("RNN", "Cox"): "RNN vs Cox",
    ("LSTM-v2", "Cox"): "LSTM-v2 vs Cox",
    ("RSF", "RNN"): "RSF vs RNN",
    ("LSTM-v2", "RSF"): "LSTM-v2 vs RSF",
    ("LSTM-v2", "RNN"): "LSTM-v2 vs RNN",
}

MODEL_MARKERS = {
    "Cox": "o",
    "RSF": "s",
    "RNN": "^",
    "LSTM-v2": "D",
}

DPI = 600


# =============================================================================
# 1. 绘图风格
# =============================================================================

def choose_font() -> str:
    preferred_fonts = [
        "Arial",
        "Helvetica",
        "Liberation Sans",
        "DejaVu Sans",
    ]

    for font_name in preferred_fonts:
        try:
            font_manager.findfont(
                font_name,
                fallback_to_default=False,
            )
            return font_name
        except ValueError:
            continue

    return "DejaVu Sans"


FONT_NAME = choose_font()

plt.rcParams.update({
    "font.family": FONT_NAME,
    "font.size": 10.5,
    "axes.titlesize": 12.5,
    "axes.labelsize": 11,
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
    "legend.fontsize": 9.5,
    "figure.titlesize": 14,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "lines.linewidth": 2.0,
    "savefig.dpi": DPI,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

DEFAULT_COLORS = (
    plt.rcParams[
        "axes.prop_cycle"
    ].by_key()[
        "color"
    ]
)

MODEL_COLORS = {
    model: DEFAULT_COLORS[
        index
        % len(DEFAULT_COLORS)
    ]
    for index, model
    in enumerate(MODEL_ORDER)
}


# =============================================================================
# 2. 工具函数
# =============================================================================

def require_file(
    path: Path,
) -> None:
    if not path.exists():
        raise FileNotFoundError(
            "找不到必要结果文件：\n"
            f"{path}"
        )


def read_csv_checked(
    path: Path,
) -> pd.DataFrame:
    require_file(path)
    frame = pd.read_csv(path)

    if len(frame) == 0:
        raise ValueError(
            f"结果文件为空：{path}"
        )

    return frame


def resolve_column(
    frame: pd.DataFrame,
    aliases: list[str],
    description: str,
) -> str:
    for column in aliases:
        if column in frame.columns:
            return column

    raise KeyError(
        f"无法找到 {description}。\n"
        f"允许的列名：{aliases}\n"
        f"实际列名：{frame.columns.tolist()}"
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
        path = (
            OUT_DIR
            / f"{stem}.{extension}"
        )

        kwargs = {
            "bbox_inches": "tight",
            "facecolor": "white",
        }

        if extension == "png":
            kwargs["dpi"] = DPI

        fig.savefig(
            path,
            **kwargs,
        )

    plt.close(fig)


def add_panel_label(
    ax,
    label: str,
) -> None:
    ax.text(
        -0.12,
        1.06,
        label,
        transform=ax.transAxes,
        fontsize=13,
        fontweight="bold",
        ha="left",
        va="top",
    )


def style_axis(
    ax,
    grid_axis: str = "y",
) -> None:
    ax.grid(
        axis=grid_axis,
        linestyle="--",
        linewidth=0.6,
        alpha=0.30,
    )
    ax.set_axisbelow(True)


def percent_formatter(
    decimals: int = 0,
):
    return FuncFormatter(
        lambda value, position: (
            f"{value:.{decimals}f}%"
        )
    )


def safe_range_padding(
    lower: float,
    upper: float,
    fraction: float = 0.15,
    minimum: float = 1e-4,
) -> float:
    span = float(
        upper - lower
    )

    if (
        not np.isfinite(span)
        or span <= 0
    ):
        return minimum

    return max(
        span * fraction,
        minimum,
    )


def normalize_pairwise_table(
    pair_df: pd.DataFrame,
) -> pd.DataFrame:
    current_required = {
        "scope",
        "model_a",
        "model_b",
        "metric",
        "difference_a_minus_b",
        "difference_lower_95",
        "difference_upper_95",
    }

    if current_required.issubset(
        pair_df.columns
    ):
        result = pair_df.copy()

        if (
            "comparison_label"
            not in result.columns
        ):
            result[
                "comparison_label"
            ] = (
                result["model_a"]
                + " vs "
                + result["model_b"]
            )

        return result

    legacy_required = {
        "scope",
        "candidate_model",
        "reference_model",
        "delta_cindex_candidate_minus_reference",
        "delta_cindex_lower_95",
        "delta_cindex_upper_95",
        "delta_iauc_candidate_minus_reference",
        "delta_iauc_lower_95",
        "delta_iauc_upper_95",
        "delta_ibs_candidate_minus_reference",
        "delta_ibs_lower_95",
        "delta_ibs_upper_95",
    }

    if legacy_required.issubset(
        pair_df.columns
    ):
        records = []

        metric_specs = [
            (
                "uno_c_index_5y",
                "delta_cindex_candidate_minus_reference",
                "delta_cindex_lower_95",
                "delta_cindex_upper_95",
                "candidate_better_cindex_fraction",
            ),
            (
                "integrated_dynamic_auc",
                "delta_iauc_candidate_minus_reference",
                "delta_iauc_lower_95",
                "delta_iauc_upper_95",
                "candidate_better_iauc_fraction",
            ),
            (
                "integrated_brier",
                "delta_ibs_candidate_minus_reference",
                "delta_ibs_lower_95",
                "delta_ibs_upper_95",
                "candidate_better_ibs_fraction",
            ),
        ]

        for _, row in pair_df.iterrows():
            for (
                metric,
                delta_col,
                lower_col,
                upper_col,
                favorable_col,
            ) in metric_specs:
                records.append({
                    "scope": row["scope"],
                    "model_a": row[
                        "candidate_model"
                    ],
                    "model_b": row[
                        "reference_model"
                    ],
                    "comparison_label": (
                        str(
                            row[
                                "candidate_model"
                            ]
                        )
                        + " vs "
                        + str(
                            row[
                                "reference_model"
                            ]
                        )
                    ),
                    "metric": metric,
                    "difference_a_minus_b": row[
                        delta_col
                    ],
                    "difference_lower_95": row[
                        lower_col
                    ],
                    "difference_upper_95": row[
                        upper_col
                    ],
                    "model_a_favorable_fraction": row.get(
                        favorable_col,
                        np.nan,
                    ),
                    "bootstrap_two_sided_p": np.nan,
                })

        return pd.DataFrame(
            records
        )

    raise ValueError(
        "无法识别 paired bootstrap CSV 格式。\n"
        f"实际列名：{pair_df.columns.tolist()}"
    )


# =============================================================================
# 3. 读取6个结果文件
# =============================================================================

PERF_FILE = (
    STEP11_DIR
    / "four_model_equal_weight_mean_performance.csv"
)

PAIR_FILE = (
    STEP11_DIR
    / "four_model_paired_bootstrap_comparison.csv"
)

METRICS_FILE = (
    STEP11_DIR
    / "four_model_crossfit_calibrated_1y_3y_5y_metrics.csv"
)

CAL_OVERALL_FILE = (
    STEP11_DIR
    / "four_model_calibration_overall.csv"
)

CAL_DECILE_FILE = (
    STEP11_DIR
    / "four_model_calibration_deciles.csv"
)

DCA_FILE = (
    STEP11_DIR
    / "four_model_survival_dca.csv"
)

perf_df = read_csv_checked(
    PERF_FILE
)

pair_raw_df = read_csv_checked(
    PAIR_FILE
)

metrics_df = read_csv_checked(
    METRICS_FILE
)

cal_overall_df = read_csv_checked(
    CAL_OVERALL_FILE
)

cal_decile_df = read_csv_checked(
    CAL_DECILE_FILE
)

dca_df = read_csv_checked(
    DCA_FILE
)

pair_df = normalize_pairwise_table(
    pair_raw_df
)


# =============================================================================
# 4. 输入审计
# =============================================================================

def audit_inputs() -> None:
    audit_lines = []

    objects = [
        ("performance", PERF_FILE, perf_df),
        ("paired_bootstrap", PAIR_FILE, pair_raw_df),
        ("1y_3y_5y_metrics", METRICS_FILE, metrics_df),
        ("calibration_overall", CAL_OVERALL_FILE, cal_overall_df),
        ("calibration_deciles", CAL_DECILE_FILE, cal_decile_df),
        ("dca", DCA_FILE, dca_df),
    ]

    for name, path, frame in objects:
        audit_lines.append(
            "=" * 100
        )
        audit_lines.append(
            name
        )
        audit_lines.append(
            f"path: {path}"
        )
        audit_lines.append(
            f"shape: {frame.shape}"
        )
        audit_lines.append(
            "columns:"
        )
        audit_lines.extend(
            [
                f"  - {column}"
                for column
                in frame.columns
            ]
        )

        if "model" in frame.columns:
            audit_lines.append(
                "models: "
                + str(
                    sorted(
                        frame[
                            "model"
                        ]
                        .dropna()
                        .astype(str)
                        .unique()
                        .tolist()
                    )
                )
            )

        if "landmark_year" in frame.columns:
            audit_lines.append(
                "landmark_year: "
                + str(
                    sorted(
                        frame[
                            "landmark_year"
                        ]
                        .dropna()
                        .unique()
                        .tolist()
                    )
                )
            )

        if "horizon_year" in frame.columns:
            audit_lines.append(
                "horizon_year: "
                + str(
                    sorted(
                        frame[
                            "horizon_year"
                        ]
                        .dropna()
                        .unique()
                        .tolist()
                    )
                )
            )

    audit_path = (
        OUT_DIR
        / "visualization_input_schema_audit.txt"
    )

    audit_path.write_text(
        "\n".join(
            audit_lines
        ),
        encoding="utf-8",
    )

    print(
        "输入审计已保存：",
        audit_path,
    )


audit_inputs()


# =============================================================================
# 5. 列名自动识别
# =============================================================================

C_VALUE = resolve_column(
    perf_df,
    [
        "mean_uno_c_index_5y",
    ],
    "平均5年Uno C",
)

C_LOWER = resolve_column(
    perf_df,
    [
        "mean_uno_c_index_5y_lower_95",
        "cindex_lower_95",
    ],
    "Uno C 95%CI下限",
)

C_UPPER = resolve_column(
    perf_df,
    [
        "mean_uno_c_index_5y_upper_95",
        "cindex_upper_95",
    ],
    "Uno C 95%CI上限",
)

IAUC_VALUE = resolve_column(
    perf_df,
    [
        "mean_integrated_dynamic_auc",
    ],
    "平均iAUC",
)

IAUC_LOWER = resolve_column(
    perf_df,
    [
        "mean_integrated_dynamic_auc_lower_95",
        "iauc_lower_95",
    ],
    "iAUC 95%CI下限",
)

IAUC_UPPER = resolve_column(
    perf_df,
    [
        "mean_integrated_dynamic_auc_upper_95",
        "iauc_upper_95",
    ],
    "iAUC 95%CI上限",
)

IBS_VALUE = resolve_column(
    perf_df,
    [
        "mean_integrated_brier",
    ],
    "平均IBS",
)

IBS_LOWER = resolve_column(
    perf_df,
    [
        "mean_integrated_brier_lower_95",
        "ibs_lower_95",
    ],
    "IBS 95%CI下限",
)

IBS_UPPER = resolve_column(
    perf_df,
    [
        "mean_integrated_brier_upper_95",
        "ibs_upper_95",
    ],
    "IBS 95%CI上限",
)


# =============================================================================
# 6. 主分析数据筛选
# =============================================================================

perf_primary = (
    perf_df.loc[
        perf_df[
            "scope"
        ]
        == PRIMARY_SCOPE
    ]
    .copy()
)

missing_models = (
    set(
        MODEL_ORDER
    )
    - set(
        perf_primary[
            "model"
        ]
        .astype(str)
        .tolist()
    )
)

if missing_models:
    raise ValueError(
        "主性能表缺少模型："
        + str(
            sorted(
                missing_models
            )
        )
    )

perf_primary[
    "model_order"
] = pd.Categorical(
    perf_primary[
        "model"
    ],
    categories=MODEL_ORDER,
    ordered=True,
)

perf_primary = (
    perf_primary
    .sort_values(
        "model_order"
    )
    .drop(
        columns=[
            "model_order"
        ]
    )
    .reset_index(
        drop=True
    )
)

pair_primary = (
    pair_df.loc[
        pair_df[
            "scope"
        ]
        == PRIMARY_SCOPE
    ]
    .copy()
)

metrics_primary = (
    metrics_df.loc[
        (
            metrics_df[
                "landmark_year"
            ]
            .isin(
                PRIMARY_LANDMARK_YEARS
            )
        )
        & (
            metrics_df[
                "horizon_year"
            ]
            .isin(
                HORIZON_YEARS
            )
        )
    ]
    .copy()
)

if (
    "prediction_stage"
    in metrics_primary.columns
):
    metrics_primary = (
        metrics_primary.loc[
            metrics_primary[
                "prediction_stage"
            ]
            == "crossfit_calibrated_oof"
        ]
        .copy()
    )

cal_overall_primary = (
    cal_overall_df.loc[
        (
            cal_overall_df[
                "landmark_year"
            ]
            .isin(
                PRIMARY_LANDMARK_YEARS
            )
        )
        & (
            cal_overall_df[
                "horizon_year"
            ]
            .isin(
                HORIZON_YEARS
            )
        )
    ]
    .copy()
)

cal_decile_primary = (
    cal_decile_df.loc[
        (
            cal_decile_df[
                "landmark_year"
            ]
            .isin(
                PRIMARY_LANDMARK_YEARS
            )
        )
        & (
            cal_decile_df[
                "horizon_year"
            ]
            .isin(
                HORIZON_YEARS
            )
        )
    ]
    .copy()
)

dca_primary = (
    dca_df.loc[
        (
            dca_df[
                "landmark_year"
            ]
            .isin(
                PRIMARY_LANDMARK_YEARS
            )
        )
        & (
            dca_df[
                "horizon_year"
            ]
            .isin(
                HORIZON_YEARS
            )
        )
    ]
    .copy()
)


# =============================================================================
# 7. Figure 1：四模型主性能 + Bootstrap 95%CI
# =============================================================================

def plot_figure1_main_performance() -> None:
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(
            14.8,
            4.9,
        ),
        constrained_layout=True,
    )

    specs = [
        {
            "value": C_VALUE,
            "lower": C_LOWER,
            "upper": C_UPPER,
            "title": "5-year Uno C-index",
            "xlabel": "Mean Uno C-index",
            "better": "Higher is better",
            "digits": 3,
        },
        {
            "value": IAUC_VALUE,
            "lower": IAUC_LOWER,
            "upper": IAUC_UPPER,
            "title": "Integrated dynamic AUC",
            "xlabel": "Mean iAUC",
            "better": "Higher is better",
            "digits": 3,
        },
        {
            "value": IBS_VALUE,
            "lower": IBS_LOWER,
            "upper": IBS_UPPER,
            "title": "Integrated Brier score",
            "xlabel": "Mean IBS",
            "better": "Lower is better",
            "digits": 4,
        },
    ]

    y = np.arange(
        len(
            MODEL_ORDER
        )
    )[::-1]

    for panel_index, (ax, spec) in enumerate(
        zip(
            axes,
            specs,
        )
    ):
        add_panel_label(
            ax,
            chr(
                ord("A")
                + panel_index
            ),
        )

        values = (
            perf_primary[
                spec["value"]
            ]
            .to_numpy(
                dtype=float
            )
        )

        lower = (
            perf_primary[
                spec["lower"]
            ]
            .to_numpy(
                dtype=float
            )
        )

        upper = (
            perf_primary[
                spec["upper"]
            ]
            .to_numpy(
                dtype=float
            )
        )

        for row_index, model_name in enumerate(
            MODEL_ORDER
        ):
            yi = y[
                row_index
            ]

            ax.hlines(
                yi,
                lower[
                    row_index
                ],
                upper[
                    row_index
                ],
                linewidth=2.0,
                alpha=0.85,
            )

            ax.scatter(
                values[
                    row_index
                ],
                yi,
                s=70,
                marker=MODEL_MARKERS[
                    model_name
                ],
                edgecolors="none",
                zorder=3,
            )

        ax.set_yticks(
            y
        )

        if panel_index == 0:
            ax.set_yticklabels(
                [
                    MODEL_LABELS[
                        model
                    ]
                    for model
                    in MODEL_ORDER
                ]
            )
        else:
            ax.set_yticklabels(
                []
            )

        ax.set_title(
            spec[
                "title"
            ]
        )

        ax.set_xlabel(
            spec[
                "xlabel"
            ]
        )

        style_axis(
            ax,
            grid_axis="x",
        )

        xmin = float(
            lower.min()
        )
        xmax = float(
            upper.max()
        )

        pad = safe_range_padding(
            xmin,
            xmax,
            fraction=0.18,
        )

        ax.set_xlim(
            xmin - pad,
            xmax + 2.1 * pad,
        )

        for row_index in range(
            len(
                MODEL_ORDER
            )
        ):
            label_text = format(
                values[
                    row_index
                ],
                f".{spec['digits']}f",
            )

            ax.text(
                upper[
                    row_index
                ]
                + 0.12 * pad,
                y[
                    row_index
                ],
                label_text,
                va="center",
                ha="left",
                fontsize=9,
            )

        ax.text(
            0.02,
            0.03,
            spec[
                "better"
            ],
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=8.5,
        )

    fig.suptitle(
        "Four-model performance across ART years 0, 1, 3, and 5",
        fontweight="bold",
        y=1.04,
    )

    save_figure(
        fig,
        "Figure1_primary_model_performance",
    )


# =============================================================================
# 8. Figure 2：患者级配对Bootstrap森林图
# =============================================================================

def plot_figure2_pairwise_forest() -> None:
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(
            16.2,
            6.0,
        ),
        constrained_layout=True,
    )

    metric_specs = [
        {
            "metric": "uno_c_index_5y",
            "title": "Difference in Uno C-index",
            "xlabel": "Difference (model A − model B)",
            "favor": "Positive favors model A",
            "digits": 4,
        },
        {
            "metric": "integrated_dynamic_auc",
            "title": "Difference in iAUC",
            "xlabel": "Difference (model A − model B)",
            "favor": "Positive favors model A",
            "digits": 4,
        },
        {
            "metric": "integrated_brier",
            "title": "Difference in IBS",
            "xlabel": "Difference (model A − model B)",
            "favor": "Negative favors model A",
            "digits": 5,
        },
    ]

    y = np.arange(
        len(
            PAIR_ORDER
        )
    )[::-1]

    for panel_index, (ax, spec) in enumerate(
        zip(
            axes,
            metric_specs,
        )
    ):
        add_panel_label(
            ax,
            chr(
                ord("A")
                + panel_index
            ),
        )

        ax.axvline(
            0.0,
            linestyle="--",
            linewidth=1.0,
        )

        plot_records = []

        for model_a, model_b in (
            PAIR_ORDER
        ):
            subset = (
                pair_primary.loc[
                    (
                        pair_primary[
                            "model_a"
                        ]
                        == model_a
                    )
                    & (
                        pair_primary[
                            "model_b"
                        ]
                        == model_b
                    )
                    & (
                        pair_primary[
                            "metric"
                        ]
                        == spec[
                            "metric"
                        ]
                    )
                ]
            )

            if len(
                subset
            ) != 1:
                raise ValueError(
                    "配对Bootstrap结果不唯一："
                    f"{model_a} vs {model_b} | "
                    f"{spec['metric']} | "
                    f"找到{len(subset)}行"
                )

            plot_records.append(
                subset.iloc[
                    0
                ]
            )

        plot_frame = pd.DataFrame(
            plot_records
        ).reset_index(
            drop=True
        )

        delta = (
            plot_frame[
                "difference_a_minus_b"
            ]
            .to_numpy(
                dtype=float
            )
        )

        lower = (
            plot_frame[
                "difference_lower_95"
            ]
            .to_numpy(
                dtype=float
            )
        )

        upper = (
            plot_frame[
                "difference_upper_95"
            ]
            .to_numpy(
                dtype=float
            )
        )

        for row_index, (
            model_a,
            model_b,
        ) in enumerate(
            PAIR_ORDER
        ):
            yi = y[
                row_index
            ]

            ax.hlines(
                yi,
                lower[
                    row_index
                ],
                upper[
                    row_index
                ],
                linewidth=2.0,
            )

            ax.scatter(
                delta[
                    row_index
                ],
                yi,
                s=65,
                marker=MODEL_MARKERS[
                    model_a
                ],
                zorder=3,
            )

        ax.set_yticks(
            y
        )

        if panel_index == 0:
            ax.set_yticklabels(
                [
                    PAIR_LABELS[
                        pair
                    ]
                    for pair
                    in PAIR_ORDER
                ]
            )
        else:
            ax.set_yticklabels(
                []
            )

        ax.set_title(
            spec[
                "title"
            ]
        )

        ax.set_xlabel(
            spec[
                "xlabel"
            ]
        )

        style_axis(
            ax,
            grid_axis="x",
        )

        xmin = min(
            float(
                lower.min()
            ),
            0.0,
        )

        xmax = max(
            float(
                upper.max()
            ),
            0.0,
        )

        pad = safe_range_padding(
            xmin,
            xmax,
            fraction=0.15,
        )

        ax.set_xlim(
            xmin - pad,
            xmax + 2.2 * pad,
        )

        for row_index in range(
            len(
                PAIR_ORDER
            )
        ):
            delta_text = (
                "+"
                + format(
                    delta[
                        row_index
                    ],
                    f".{spec['digits']}f",
                )
                if delta[
                    row_index
                ] >= 0
                else format(
                    delta[
                        row_index
                    ],
                    f".{spec['digits']}f",
                )
            )

            ax.text(
                upper[
                    row_index
                ]
                + 0.10 * pad,
                y[
                    row_index
                ],
                delta_text,
                va="center",
                ha="left",
                fontsize=8.5,
            )

        ax.text(
            0.02,
            0.03,
            spec[
                "favor"
            ],
            transform=ax.transAxes,
            fontsize=8.5,
            ha="left",
            va="bottom",
        )

    fig.suptitle(
        "Patient-level paired bootstrap comparison",
        fontweight="bold",
        y=1.03,
    )

    save_figure(
        fig,
        "Figure2_primary_paired_bootstrap_forest",
    )


# =============================================================================
# 9. Figure 3：1/3/5年动态AUC与Brier
# =============================================================================

def plot_figure3_auc_brier() -> None:
    fig, axes = plt.subplots(
        2,
        3,
        figsize=(
            15.8,
            8.3,
        ),
        constrained_layout=True,
        sharex="col",
    )

    for column_index, horizon_year in enumerate(
        HORIZON_YEARS
    ):
        auc_ax = axes[
            0,
            column_index,
        ]

        brier_ax = axes[
            1,
            column_index,
        ]

        add_panel_label(
            auc_ax,
            chr(
                ord("A")
                + column_index
            ),
        )

        add_panel_label(
            brier_ax,
            chr(
                ord("D")
                + column_index
            ),
        )

        subset = (
            metrics_primary.loc[
                metrics_primary[
                    "horizon_year"
                ]
                == horizon_year
            ]
            .copy()
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
                    "landmark_year"
                )
            )

            auc_ax.plot(
                model_subset[
                    "landmark_year"
                ],
                model_subset[
                    "dynamic_auc"
                ],
                marker=MODEL_MARKERS[
                    model_name
                ],
                label=MODEL_LABELS[
                    model_name
                ],
            )

            brier_ax.plot(
                model_subset[
                    "landmark_year"
                ],
                model_subset[
                    "brier_score"
                ],
                marker=MODEL_MARKERS[
                    model_name
                ],
                label=MODEL_LABELS[
                    model_name
                ],
            )

        auc_ax.set_title(
            f"{int(horizon_year)}-year horizon"
        )

        brier_ax.set_title(
            f"{int(horizon_year)}-year horizon"
        )

        auc_ax.set_ylabel(
            "Dynamic AUC"
        )

        brier_ax.set_ylabel(
            "Brier score"
        )

        brier_ax.set_xlabel(
            "ART landmark year"
        )

        for ax in [
            auc_ax,
            brier_ax,
        ]:
            ax.set_xticks(
                PRIMARY_LANDMARK_YEARS
            )

            ax.set_xticklabels(
                [
                    str(
                        int(value)
                    )
                    for value
                    in PRIMARY_LANDMARK_YEARS
                ]
            )

            style_axis(
                ax,
                grid_axis="both",
            )

    handles = [
        Line2D(
            [0],
            [0],
            marker=MODEL_MARKERS[
                model
            ],
            linestyle="-",
            label=MODEL_LABELS[
                model
            ],
        )
        for model
        in MODEL_ORDER
    ]

    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(
            0.5,
            1.02,
        ),
    )

    fig.suptitle(
        "Time-specific discrimination and prediction error",
        fontweight="bold",
        y=1.065,
    )

    save_figure(
        fig,
        "Figure3_dynamic_auc_brier_by_landmark",
    )


# =============================================================================
# 10. Figure 4：Calibration-in-the-large
# =============================================================================

def plot_figure4_calibration_in_the_large() -> None:
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(
            15.6,
            4.9,
        ),
        constrained_layout=True,
    )

    for panel_index, horizon_year in enumerate(
        HORIZON_YEARS
    ):
        ax = axes[
            panel_index
        ]

        add_panel_label(
            ax,
            chr(
                ord("A")
                + panel_index
            ),
        )

        subset = (
            cal_overall_primary.loc[
                cal_overall_primary[
                    "horizon_year"
                ]
                == horizon_year
            ]
            .copy()
        )

        observed = (
            subset.loc[
                subset[
                    "model"
                ]
                == MODEL_ORDER[
                    0
                ]
            ]
            .sort_values(
                "landmark_year"
            )
        )

        ax.plot(
            observed[
                "landmark_year"
            ],
            observed[
                "km_observed_risk"
            ]
            * 100,
            marker="X",
            linestyle="--",
            linewidth=2.2,
            label="Observed (KM)",
        )

        ax.fill_between(
            observed[
                "landmark_year"
            ].to_numpy(
                dtype=float
            ),
            (
                observed[
                    "km_lower_95"
                ]
                * 100
            ).to_numpy(
                dtype=float
            ),
            (
                observed[
                    "km_upper_95"
                ]
                * 100
            ).to_numpy(
                dtype=float
            ),
            alpha=0.10,
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
                    "landmark_year"
                )
            )

            ax.plot(
                model_subset[
                    "landmark_year"
                ],
                model_subset[
                    "mean_predicted_risk"
                ]
                * 100,
                marker=MODEL_MARKERS[
                    model_name
                ],
                label=MODEL_LABELS[
                    model_name
                ],
            )

        ax.set_title(
            f"{int(horizon_year)}-year risk"
        )

        ax.set_xlabel(
            "ART landmark year"
        )

        if panel_index == 0:
            ax.set_ylabel(
                "Risk (%)"
            )

        ax.set_xticks(
            PRIMARY_LANDMARK_YEARS
        )

        ax.set_xticklabels(
            [
                str(
                    int(value)
                )
                for value
                in PRIMARY_LANDMARK_YEARS
            ]
        )

        ax.yaxis.set_major_formatter(
            percent_formatter(
                decimals=1
            )
        )

        style_axis(
            ax,
            grid_axis="both",
        )

    handles = [
        Line2D(
            [0],
            [0],
            marker=MODEL_MARKERS[
                model
            ],
            linestyle="-",
            label=MODEL_LABELS[
                model
            ],
        )
        for model
        in MODEL_ORDER
    ]

    handles.append(
        Line2D(
            [0],
            [0],
            marker="X",
            linestyle="--",
            label="Observed (KM)",
        )
    )

    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=5,
        frameon=False,
        bbox_to_anchor=(
            0.5,
            1.03,
        ),
    )

    fig.suptitle(
        "Calibration-in-the-large across prediction landmarks",
        fontweight="bold",
        y=1.08,
    )

    save_figure(
        fig,
        "Figure4_calibration_in_the_large",
    )


# =============================================================================
# 11. Figure 5：LSTM-v2 十分位校准 3×4
# =============================================================================

def plot_figure5_lstm_decile_calibration() -> None:
    final_frame = (
        cal_decile_primary.loc[
            cal_decile_primary[
                "model"
            ]
            == FINAL_MODEL
        ]
        .copy()
    )

    fig, axes = plt.subplots(
        3,
        4,
        figsize=(
            15.5,
            11.0,
        ),
        constrained_layout=True,
    )

    panel_index = 0

    for row_index, horizon_year in enumerate(
        HORIZON_YEARS
    ):
        for column_index, landmark_year in enumerate(
            PRIMARY_LANDMARK_YEARS
        ):
            ax = axes[
                row_index,
                column_index,
            ]

            add_panel_label(
                ax,
                chr(
                    ord("A")
                    + panel_index
                ),
            )

            panel_index += 1

            subset = (
                final_frame.loc[
                    (
                        final_frame[
                            "horizon_year"
                        ]
                        == horizon_year
                    )
                    & (
                        final_frame[
                            "landmark_year"
                        ]
                        == landmark_year
                    )
                ]
                .sort_values(
                    "risk_group"
                )
            )

            if len(
                subset
            ) == 0:
                raise ValueError(
                    "缺少LSTM-v2十分位校准数据："
                    f"Landmark={landmark_year}, "
                    f"Horizon={horizon_year}"
                )

            predicted = (
                subset[
                    "mean_predicted_risk"
                ]
                .to_numpy(
                    dtype=float
                )
                * 100
            )

            observed = (
                subset[
                    "km_observed_risk"
                ]
                .to_numpy(
                    dtype=float
                )
                * 100
            )

            lower = (
                subset[
                    "km_lower_95"
                ]
                .to_numpy(
                    dtype=float
                )
                * 100
            )

            upper = (
                subset[
                    "km_upper_95"
                ]
                .to_numpy(
                    dtype=float
                )
                * 100
            )

            maximum = max(
                float(
                    predicted.max()
                ),
                float(
                    upper.max()
                ),
                0.5,
            )

            maximum *= 1.08

            ax.plot(
                [
                    0,
                    maximum,
                ],
                [
                    0,
                    maximum,
                ],
                linestyle="--",
                linewidth=1.0,
            )

            ax.errorbar(
                predicted,
                observed,
                yerr=np.vstack(
                    [
                        observed
                        - lower,
                        upper
                        - observed,
                    ]
                ),
                fmt="o-",
                markersize=4.5,
                capsize=2,
                linewidth=1.6,
            )

            ax.set_xlim(
                0,
                maximum,
            )

            ax.set_ylim(
                0,
                maximum,
            )

            ax.set_title(
                (
                    f"Landmark {int(landmark_year)} y | "
                    f"{int(horizon_year)}-y risk"
                ),
                fontsize=10.5,
            )

            if row_index == 2:
                ax.set_xlabel(
                    "Predicted risk (%)"
                )

            if column_index == 0:
                ax.set_ylabel(
                    "Observed risk (%)"
                )

            style_axis(
                ax,
                grid_axis="both",
            )

    fig.suptitle(
        "LSTM-v2 decile-based calibration",
        fontweight="bold",
        y=1.015,
    )

    save_figure(
        fig,
        "Figure5_lstm_v2_decile_calibration",
    )


# =============================================================================
# 12. Figure 6：DCA 3×4
# =============================================================================

def plot_figure6_dca() -> None:
    fig, axes = plt.subplots(
        3,
        4,
        figsize=(
            16.0,
            11.0,
        ),
        constrained_layout=True,
    )

    dca_models = (
        MODEL_ORDER
        + [
            "Treat all",
            "Treat none",
        ]
    )

    line_styles = {
        "Cox": "-",
        "RSF": "-",
        "RNN": "-",
        "LSTM-v2": "-",
        "Treat all": "--",
        "Treat none": ":",
    }

    panel_index = 0

    for row_index, horizon_year in enumerate(
        HORIZON_YEARS
    ):
        for column_index, landmark_year in enumerate(
            PRIMARY_LANDMARK_YEARS
        ):
            ax = axes[
                row_index,
                column_index,
            ]

            add_panel_label(
                ax,
                chr(
                    ord("A")
                    + panel_index
                ),
            )

            panel_index += 1

            subset = (
                dca_primary.loc[
                    (
                        dca_primary[
                            "horizon_year"
                        ]
                        == horizon_year
                    )
                    & (
                        dca_primary[
                            "landmark_year"
                        ]
                        == landmark_year
                    )
                ]
                .copy()
            )

            if len(
                subset
            ) == 0:
                raise ValueError(
                    "缺少DCA数据："
                    f"Landmark={landmark_year}, "
                    f"Horizon={horizon_year}"
                )

            for model_name in (
                dca_models
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

                if model_name in MODEL_ORDER:
                    ax.plot(
                        model_subset[
                            "threshold_probability"
                        ]
                        * 100,
                        model_subset[
                            "net_benefit"
                        ],
                        linestyle=line_styles[
                            model_name
                        ],
                        label=MODEL_LABELS[
                            model_name
                        ],
                    )
                else:
                    ax.plot(
                        model_subset[
                            "threshold_probability"
                        ]
                        * 100,
                        model_subset[
                            "net_benefit"
                        ],
                        linestyle=line_styles[
                            model_name
                        ],
                        linewidth=1.3,
                        label=model_name,
                    )

            ax.set_title(
                (
                    f"Landmark {int(landmark_year)} y | "
                    f"{int(horizon_year)}-y risk"
                ),
                fontsize=10.5,
            )

            if row_index == 2:
                ax.set_xlabel(
                    "Threshold probability (%)"
                )

            if column_index == 0:
                ax.set_ylabel(
                    "Net benefit"
                )

            ax.xaxis.set_major_locator(
                MaxNLocator(
                    nbins=5
                )
            )

            ax.yaxis.set_major_locator(
                MaxNLocator(
                    nbins=5
                )
            )

            style_axis(
                ax,
                grid_axis="both",
            )

    legend_handles = []

    for model_name in (
        MODEL_ORDER
    ):
        legend_handles.append(
            Line2D(
                [0],
                [0],
                linestyle="-",
                marker=MODEL_MARKERS[
                    model_name
                ],
                label=MODEL_LABELS[
                    model_name
                ],
            )
        )

    legend_handles.extend([
        Line2D(
            [0],
            [0],
            linestyle="--",
            label="Treat all",
        ),
        Line2D(
            [0],
            [0],
            linestyle=":",
            label="Treat none",
        ),
    ])

    fig.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=6,
        frameon=False,
        bbox_to_anchor=(
            0.5,
            1.01,
        ),
    )

    fig.suptitle(
        "Decision curve analysis across primary landmarks",
        fontweight="bold",
        y=1.045,
    )

    save_figure(
        fig,
        "Figure6_primary_landmark_dca",
    )


# =============================================================================
# 13. 导出每张图的Source Data
# =============================================================================

def export_source_data() -> None:
    perf_primary.to_csv(
        SOURCE_DIR
        / "Figure1_source.csv",
        index=False,
        encoding="utf-8-sig",
    )

    pair_primary.to_csv(
        SOURCE_DIR
        / "Figure2_source.csv",
        index=False,
        encoding="utf-8-sig",
    )

    metrics_primary.to_csv(
        SOURCE_DIR
        / "Figure3_source.csv",
        index=False,
        encoding="utf-8-sig",
    )

    cal_overall_primary.to_csv(
        SOURCE_DIR
        / "Figure4_source.csv",
        index=False,
        encoding="utf-8-sig",
    )

    cal_decile_primary.loc[
        cal_decile_primary[
            "model"
        ]
        == FINAL_MODEL
    ].to_csv(
        SOURCE_DIR
        / "Figure5_source.csv",
        index=False,
        encoding="utf-8-sig",
    )

    dca_primary.to_csv(
        SOURCE_DIR
        / "Figure6_source.csv",
        index=False,
        encoding="utf-8-sig",
    )


# =============================================================================
# 14. Figure Manifest
# =============================================================================

def write_manifest() -> None:
    manifest = pd.DataFrame([
        {
            "figure": "Figure 1",
            "file_stem": (
                "Figure1_primary_model_performance"
            ),
            "recommended_use": (
                "Main manuscript"
            ),
            "content": (
                "Primary 0/1/3/5-year landmark "
                "mean Uno C, iAUC, IBS with 95% CI"
            ),
        },
        {
            "figure": "Figure 2",
            "file_stem": (
                "Figure2_primary_paired_bootstrap_forest"
            ),
            "recommended_use": (
                "Main or Supplement"
            ),
            "content": (
                "Patient-level paired bootstrap "
                "differences with 95% CI"
            ),
        },
        {
            "figure": "Figure 3",
            "file_stem": (
                "Figure3_dynamic_auc_brier_by_landmark"
            ),
            "recommended_use": (
                "Main manuscript"
            ),
            "content": (
                "1/3/5-year dynamic AUC and "
                "Brier score across landmarks"
            ),
        },
        {
            "figure": "Figure 4",
            "file_stem": (
                "Figure4_calibration_in_the_large"
            ),
            "recommended_use": (
                "Main manuscript"
            ),
            "content": (
                "Observed versus model-predicted "
                "risk across landmarks"
            ),
        },
        {
            "figure": "Figure 5",
            "file_stem": (
                "Figure5_lstm_v2_decile_calibration"
            ),
            "recommended_use": (
                "Supplement"
            ),
            "content": (
                "LSTM-v2 decile calibration "
                "for 12 landmark-horizon combinations"
            ),
        },
        {
            "figure": "Figure 6",
            "file_stem": (
                "Figure6_primary_landmark_dca"
            ),
            "recommended_use": (
                "Main or Supplement"
            ),
            "content": (
                "Decision curve analysis "
                "for 12 landmark-horizon combinations"
            ),
        },
    ])

    manifest.to_csv(
        OUT_DIR
        / "figure_manifest.csv",
        index=False,
        encoding="utf-8-sig",
    )


# =============================================================================
# 15. 主程序
# =============================================================================

def main() -> None:
    print(
        "=" * 100
    )

    print(
        "Step 11E v2："
        "开始生成四模型专业可视化"
    )

    print(
        "=" * 100
    )

    print(
        "结果输入目录：",
        STEP11_DIR,
    )

    print(
        "图片输出目录：",
        OUT_DIR,
    )

    print(
        "字体：",
        FONT_NAME,
    )

    print(
        "主文Landmark：",
        PRIMARY_LANDMARK_YEARS,
    )

    print(
        "最终候选模型：",
        FINAL_MODEL,
    )

    print(
        "=" * 100
    )

    print(
        "\n当前主性能表识别到的95%CI列："
    )

    print(
        "Uno C：",
        C_LOWER,
        "/",
        C_UPPER,
    )

    print(
        "iAUC：",
        IAUC_LOWER,
        "/",
        IAUC_UPPER,
    )

    print(
        "IBS：",
        IBS_LOWER,
        "/",
        IBS_UPPER,
    )

    print(
        "\n当前paired bootstrap格式："
    )

    print(
        pair_primary[
            [
                "model_a",
                "model_b",
                "metric",
            ]
        ]
        .head(
            10
        )
        .to_string(
            index=False
        )
    )

    plot_figure1_main_performance()

    print(
        "Figure 1 完成："
        "四模型主性能 + 95%CI"
    )

    plot_figure2_pairwise_forest()

    print(
        "Figure 2 完成："
        "配对Bootstrap森林图"
    )

    plot_figure3_auc_brier()

    print(
        "Figure 3 完成："
        "动态AUC和Brier"
    )

    plot_figure4_calibration_in_the_large()

    print(
        "Figure 4 完成："
        "Calibration-in-the-large"
    )

    plot_figure5_lstm_decile_calibration()

    print(
        "Figure 5 完成："
        "LSTM-v2十分位校准"
    )

    plot_figure6_dca()

    print(
        "Figure 6 完成："
        "DCA"
    )

    export_source_data()
    write_manifest()

    print(
        "\n"
        + "=" * 100
    )

    print(
        "全部可视化成功完成"
    )

    print(
        "=" * 100
    )

    print(
        "输出目录：",
        OUT_DIR,
    )

    print(
        "\n建议主文优先："
    )

    print(
        "1. Figure 1 "
        "四模型总体性能"
    )

    print(
        "2. Figure 3 "
        "动态AUC/Brier"
    )

    print(
        "3. Figure 4 "
        "总体校准"
    )

    print(
        "4. Figure 6 "
        "DCA"
    )

    print(
        "\n建议补充材料："
    )

    print(
        "1. Figure 2 "
        "配对Bootstrap森林图"
    )

    print(
        "2. Figure 5 "
        "LSTM-v2十分位校准"
    )

    print(
        "=" * 100
    )


if __name__ == "__main__":
    main()
