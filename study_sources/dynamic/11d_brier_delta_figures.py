"""Revised internal-validation Brier visualizations.

This script does not change any Brier or IBS value.  It produces:

1. A main figure retaining the four dynamic Brier curves and adding a paired
   bootstrap delta-IBS panel.  The delta panel magnifies genuine between-model
   differences on their natural difference scale.
2. An optional supplementary boxplot of the paired-bootstrap IBS distributions.

Run this in the same Linux environment used for CKD_dynamic_prediction.ipynb.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator
from scipy.interpolate import PchipInterpolator


# =============================================================================
# 1. Paths and constants
# =============================================================================

PROJECT_DIR = Path(
    os.getenv("CKD_LSTM_PROJECT_DIR", "__CKD_WORKDIR__")
)

STEP11_DIR = (
    PROJECT_DIR
    / "rolling_5y_step11_v2_four_model_oof_comparison_lstm_v2"
)
BOOTSTRAP_DIR = STEP11_DIR / "bootstrap_checkpoints"
OUT_DIR = STEP11_DIR / "professional_figures_core_v5"
SOURCE_DIR = OUT_DIR / "source_data"

LANDMARK_METRIC_FILE = (
    STEP11_DIR / "four_model_crossfit_calibrated_landmark_metrics.csv"
)
HORIZON_METRIC_FILE = (
    STEP11_DIR / "four_model_crossfit_calibrated_horizon_metrics.csv"
)

for directory in [OUT_DIR, SOURCE_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

PRIMARY_LANDMARKS = [0.0, 1.0, 3.0, 5.0]
PRIMARY_LANDMARK_POSITIONS = {0.0: 0, 1.0: 1, 3.0: 3, 5.0: 5}

MODEL_ORDER = ["Cox", "RSF", "RNN", "LSTM-v2"]
MODEL_LABELS = {
    "Cox": "Landmark Cox",
    "RSF": "Landmark RSF",
    "RNN": "RNN",
    "LSTM-v2": "LSTM",
}
MODEL_COLORS = {
    # High-saturation palette matching the vivid style of the reference figure.
    "Cox": "#1F77B4",      # vivid blue
    "RSF": "#FF7F0E",      # vivid orange
    "RNN": "#2CA02C",      # vivid green
    "LSTM-v2": "#D62728",  # vivid red
}
MODEL_INDEX = {model: i for i, model in enumerate(MODEL_ORDER)}

# The Step 11 v2 checkpoint stores these metrics in this fixed order.
BOOTSTRAP_METRIC_NAMES = [
    "uno_c_index_5y",
    "integrated_dynamic_auc",
    "integrated_brier",
]
IBS_INDEX = BOOTSTRAP_METRIC_NAMES.index("integrated_brier")

DPI = 600


# =============================================================================
# 2. Styling and validation helpers
# =============================================================================

def choose_font() -> str:
    # Use Times New Roman throughout all model-performance figures.  The
    # remaining serif fonts are safety fallbacks for Linux environments in
    # which Microsoft core fonts have not been installed.
    for name in [
        "Times New Roman",
        "Times New Roman PS MT",
        "Nimbus Roman",
        "Liberation Serif",
        "DejaVu Serif",
    ]:
        try:
            font_manager.findfont(name, fallback_to_default=False)
            return name
        except ValueError:
            continue
    return "DejaVu Serif"


plt.style.use("default")
plt.rcParams.update(
    {
        "font.family": choose_font(),
        "font.serif": [
            "Times New Roman",
            "Times New Roman PS MT",
            "Nimbus Roman",
            "Liberation Serif",
            "DejaVu Serif",
        ],
        "mathtext.fontset": "custom",
        "mathtext.rm": "Times New Roman",
        "mathtext.it": "Times New Roman:italic",
        "mathtext.bf": "Times New Roman:bold",
        "mathtext.sf": "Times New Roman",
        "mathtext.tt": "Times New Roman",
        "mathtext.cal": "Times New Roman",
        "font.size": 10.5,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.dpi": DPI,
    }
)


def read_csv_checked(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found:\n{path}")
    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError(f"File is empty:\n{path}")
    return frame


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.105,
        1.055,
        label,
        transform=ax.transAxes,
        fontsize=13,
        fontweight="bold",
        ha="left",
        va="top",
    )


def style_axis(ax: plt.Axes, grid_axis: str = "both") -> None:
    ax.grid(
        axis=grid_axis,
        linestyle="--",
        linewidth=0.55,
        alpha=0.25,
        color="#A0A0A0",
    )
    ax.set_axisbelow(True)


def save_figure(fig: plt.Figure, stem: str) -> None:
    for suffix in ["png", "pdf", "svg"]:
        fig.savefig(
            OUT_DIR / f"{stem}.{suffix}",
            dpi=DPI if suffix == "png" else None,
            bbox_inches="tight",
            facecolor="white",
        )
    plt.show()
    plt.close(fig)
    print(f"Saved: {OUT_DIR / (stem + '.pdf')}")


def percentile_interval(values: np.ndarray) -> tuple[float, float, int]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan, np.nan, 0
    return (
        float(np.percentile(values, 2.5)),
        float(np.percentile(values, 97.5)),
        int(len(values)),
    )


def two_sided_bootstrap_p(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan
    return float(min(1.0, 2.0 * min(np.mean(values <= 0), np.mean(values >= 0))))


# =============================================================================
# 3. Load point estimates and the completed paired-bootstrap checkpoint
# =============================================================================

landmark_df = read_csv_checked(LANDMARK_METRIC_FILE)
horizon_df = read_csv_checked(HORIZON_METRIC_FILE)

if "prediction_stage" in landmark_df.columns:
    landmark_df = landmark_df.loc[
        landmark_df["prediction_stage"].eq("crossfit_calibrated_oof")
    ].copy()

if "prediction_stage" in horizon_df.columns:
    horizon_df = horizon_df.loc[
        horizon_df["prediction_stage"].eq("crossfit_calibrated_oof")
    ].copy()

landmark_primary = landmark_df.loc[
    landmark_df["landmark_year"].isin(PRIMARY_LANDMARKS)
].copy()

horizon_primary = horizon_df.loc[
    horizon_df["landmark_year"].isin(PRIMARY_LANDMARKS)
].copy()


def load_best_bootstrap_checkpoint() -> tuple[Path, np.ndarray]:
    candidates = sorted(
        BOOTSTRAP_DIR.glob("paired_bootstrap_*_replicates.npz")
    )
    if not candidates:
        raise FileNotFoundError(
            "No Step 11 v2 paired-bootstrap checkpoint was found in:\n"
            f"{BOOTSTRAP_DIR}"
        )

    best_path: Path | None = None
    best_metrics: np.ndarray | None = None
    best_completed_n = -1

    for path in candidates:
        with np.load(path, allow_pickle=False) as checkpoint:
            metrics = checkpoint["bootstrap_metrics"].astype(float)
            completed = checkpoint["completed_replicates"].astype(bool)

        completed_n = int(completed.sum())
        if completed_n > best_completed_n:
            best_path = path
            best_metrics = metrics[completed]
            best_completed_n = completed_n

    if best_path is None or best_metrics is None or best_completed_n == 0:
        raise ValueError("Bootstrap checkpoint contains no completed replicates.")

    expected_tail = (len(MODEL_ORDER), 6, len(BOOTSTRAP_METRIC_NAMES))
    if best_metrics.shape[1:] != expected_tail:
        raise ValueError(
            "Unexpected bootstrap_metrics shape: "
            f"{best_metrics.shape}; expected (*, {expected_tail})."
        )

    print(f"Bootstrap checkpoint: {best_path}")
    print(f"Completed paired replicates: {best_completed_n}")
    return best_path, best_metrics


CHECKPOINT_FILE, bootstrap_metrics = load_best_bootstrap_checkpoint()


# =============================================================================
# 4. Paired delta audit: comparator IBS - LSTM IBS
#    Positive delta means lower/better IBS for LSTM.
# =============================================================================

delta_rows: list[dict[str, float | int | str]] = []

for landmark in PRIMARY_LANDMARKS:
    landmark_position = PRIMARY_LANDMARK_POSITIONS[landmark]

    lstm_boot = bootstrap_metrics[
        :, MODEL_INDEX["LSTM-v2"], landmark_position, IBS_INDEX
    ]
    lstm_point = float(
        landmark_primary.loc[
            landmark_primary["landmark_year"].eq(landmark)
            & landmark_primary["model"].eq("LSTM-v2"),
            "integrated_brier",
        ].iloc[0]
    )

    for comparator in ["Cox", "RSF", "RNN"]:
        comparator_boot = bootstrap_metrics[
            :, MODEL_INDEX[comparator], landmark_position, IBS_INDEX
        ]
        delta_boot = comparator_boot - lstm_boot
        lower, upper, valid_n = percentile_interval(delta_boot)

        comparator_point = float(
            landmark_primary.loc[
                landmark_primary["landmark_year"].eq(landmark)
                & landmark_primary["model"].eq(comparator),
                "integrated_brier",
            ].iloc[0]
        )

        delta_rows.append(
            {
                "landmark_year": landmark,
                "comparator": comparator,
                "delta_ibs_comparator_minus_lstm": comparator_point - lstm_point,
                "lower_95": lower,
                "upper_95": upper,
                "valid_bootstrap_n": valid_n,
                "lstm_favorable_fraction": float(
                    np.mean(delta_boot[np.isfinite(delta_boot)] > 0)
                ),
                "bootstrap_two_sided_p": two_sided_bootstrap_p(delta_boot),
            }
        )

delta_df = pd.DataFrame(delta_rows)
delta_df.to_csv(
    SOURCE_DIR / "Figure3_paired_delta_IBS_audit.csv",
    index=False,
    encoding="utf-8-sig",
)


# =============================================================================
# 5. Recommended main figure
# =============================================================================

def plot_dynamic_brier_with_paired_delta() -> None:
    fig = plt.figure(figsize=(12.8, 12.2))
    grid = fig.add_gridspec(
        3,
        2,
        height_ratios=[1.0, 1.0, 0.82],
        hspace=0.34,
        wspace=0.18,
    )

    curve_axes = [
        fig.add_subplot(grid[0, 0]),
        fig.add_subplot(grid[0, 1]),
        fig.add_subplot(grid[1, 0]),
        fig.add_subplot(grid[1, 1]),
    ]
    delta_ax = fig.add_subplot(grid[2, :])

    smooth_grid = np.linspace(0.5, 5.0, 300)

    for panel_i, (ax, landmark) in enumerate(
        zip(curve_axes, PRIMARY_LANDMARKS)
    ):
        add_panel_label(ax, chr(65 + panel_i))

        panel = horizon_primary.loc[
            horizon_primary["landmark_year"].eq(landmark)
        ].copy()

        panel_values = panel["brier_score"].to_numpy(float) * 1000.0
        panel_min = float(np.nanmin(panel_values))
        panel_max = float(np.nanmax(panel_values))
        panel_span = max(panel_max - panel_min, 1.0)

        ibs_lines = []
        for model in MODEL_ORDER:
            temp = panel.loc[panel["model"].eq(model)].sort_values("horizon_year")
            if temp.empty:
                raise ValueError(f"Missing Brier data: {model}, landmark={landmark}")

            x_raw = temp["horizon_year"].to_numpy(float)
            y_raw = temp["brier_score"].to_numpy(float) * 1000.0
            y_plot = PchipInterpolator(x_raw, y_raw, extrapolate=False)(smooth_grid)

            ax.plot(
                smooth_grid,
                y_plot,
                color=MODEL_COLORS[model],
                linewidth=2.15,
                alpha=0.98,
                label=MODEL_LABELS[model],
            )

            ibs = float(
                landmark_primary.loc[
                    landmark_primary["landmark_year"].eq(landmark)
                    & landmark_primary["model"].eq(model),
                    "integrated_brier",
                ].iloc[0]
            )
            ibs_lines.append(f"{MODEL_LABELS[model]}  IBS={ibs:.4f}")

        if landmark == 0:
            title = "Baseline landmark\nPredicting ART years 0-5"
        else:
            title = (
                f"ART year {int(landmark)} landmark\n"
                f"Predicting ART years {int(landmark)}-{int(landmark + 5)}"
            )

        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Years after landmark")
        ax.set_ylabel(r"Brier score ($\times 10^3$)")
        ax.set_xlim(0.5, 5.0)
        ax.set_xticks(np.arange(0.5, 5.1, 0.5))
        ax.set_ylim(
            max(0.0, panel_min - 0.04 * panel_span),
            panel_max + 0.18 * panel_span,
        )
        ax.yaxis.set_major_locator(MaxNLocator(nbins=7))
        style_axis(ax)

        ax.legend(loc="upper left", frameon=False, fontsize=7.8)
        ax.text(
            0.985,
            0.985,
            "\n".join(ibs_lines),
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=7.4,
            linespacing=1.18,
            bbox={
                "boxstyle": "round,pad=0.28",
                "facecolor": "white",
                "edgecolor": "#B0B0B0",
                "linewidth": 0.65,
                "alpha": 0.96,
            },
        )

    # -------------------------------------------------------------------------
    # Panel E: paired bootstrap differences on the natural difference scale.
    # -------------------------------------------------------------------------
    add_panel_label(delta_ax, "E")
    comparator_order = ["Cox", "RSF", "RNN"]
    offsets = {"Cox": 0.22, "RSF": 0.0, "RNN": -0.22}

    for landmark_i, landmark in enumerate(PRIMARY_LANDMARKS):
        for comparator in comparator_order:
            row = delta_df.loc[
                delta_df["landmark_year"].eq(landmark)
                & delta_df["comparator"].eq(comparator)
            ].iloc[0]

            point = 1000.0 * float(row["delta_ibs_comparator_minus_lstm"])
            lower = 1000.0 * float(row["lower_95"])
            upper = 1000.0 * float(row["upper_95"])
            y = landmark_i + offsets[comparator]

            delta_ax.errorbar(
                point,
                y,
                xerr=[[point - lower], [upper - point]],
                fmt="o",
                markersize=6.0,
                capsize=3.2,
                linewidth=1.25,
                color=MODEL_COLORS[comparator],
                label=MODEL_LABELS[comparator] if landmark_i == 0 else None,
            )

    delta_ax.axvline(0.0, color="#555555", linestyle="--", linewidth=1.0)
    delta_ax.set_yticks(np.arange(len(PRIMARY_LANDMARKS)))
    delta_ax.set_yticklabels(["Baseline", "ART year 1", "ART year 3", "ART year 5"])
    delta_ax.invert_yaxis()
    delta_ax.set_xlabel(
        r"Paired $\Delta$IBS ($\times 10^3$): comparator - LSTM; positive favors LSTM"
    )
    delta_ax.set_title(
        "Paired-bootstrap differences in integrated Brier score",
        fontweight="bold",
        pad=10,
    )
    style_axis(delta_ax, grid_axis="x")
    delta_ax.legend(loc="upper right", frameon=False, ncol=3)

    fig.suptitle(
        "Dynamic Brier score with paired-bootstrap IBS differences",
        fontsize=15,
        fontweight="bold",
        y=0.995,
    )

    save_figure(fig, "Figure3_Dynamic_Brier_with_paired_delta")


# =============================================================================
# 6. Optional supplementary bootstrap boxplot
# =============================================================================

def plot_bootstrap_ibs_boxplots() -> None:
    fig, axes = plt.subplots(1, 4, figsize=(14.2, 4.9), sharey=False)
    summary_rows: list[dict[str, float | int | str]] = []

    for panel_i, (ax, landmark) in enumerate(zip(axes, PRIMARY_LANDMARKS)):
        landmark_position = PRIMARY_LANDMARK_POSITIONS[landmark]
        distributions = []
        point_values = []
        point_values_raw = []
        lower_ci_values = []
        upper_ci_values = []

        for model in MODEL_ORDER:
            values = bootstrap_metrics[
                :, MODEL_INDEX[model], landmark_position, IBS_INDEX
            ]
            values = values[np.isfinite(values)]
            distributions.append(1000.0 * values)
            lower, upper, valid_n = percentile_interval(values)
            lower_ci_values.append(1000.0 * lower)
            upper_ci_values.append(1000.0 * upper)

            point = float(
                landmark_primary.loc[
                    landmark_primary["landmark_year"].eq(landmark)
                    & landmark_primary["model"].eq(model),
                    "integrated_brier",
                ].iloc[0]
            )
            point_values_raw.append(point)
            point_values.append(1000.0 * point)

            summary_rows.append(
                {
                    "landmark_year": landmark,
                    "model": model,
                    "model_label": MODEL_LABELS[model],
                    "ibs_point_estimate": point,
                    "bootstrap_median_ibs": float(np.median(values)),
                    "bootstrap_lower_95": lower,
                    "bootstrap_upper_95": upper,
                    "valid_bootstrap_n": valid_n,
                }
            )

        boxes = ax.boxplot(
            distributions,
            positions=np.arange(1, 5),
            widths=0.58,
            patch_artist=True,
            showfliers=False,
            whis=(2.5, 97.5),
            medianprops={"color": "black", "linewidth": 1.1},
            whiskerprops={"color": "#555555", "linewidth": 0.9},
            capprops={"color": "#555555", "linewidth": 0.9},
        )

        for patch, model in zip(boxes["boxes"], MODEL_ORDER):
            patch.set_facecolor(MODEL_COLORS[model])
            patch.set_alpha(0.90)
            patch.set_edgecolor("white")

        ax.scatter(
            np.arange(1, 5),
            point_values,
            marker="D",
            s=24,
            color="black",
            zorder=4,
            label="Point estimate",
        )

        # Place each exact IBS point estimate above its 95% bootstrap interval.
        # The axis is multiplied by 1,000, while the printed IBS remains on the
        # conventional 0-1 scale used in the manuscript and result tables.
        for x_position, upper_ci_plot, point_raw in zip(
            np.arange(1, 5), upper_ci_values, point_values_raw
        ):
            ax.annotate(
                f"IBS={point_raw:.4f}",
                xy=(x_position, upper_ci_plot),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=6.2,
                fontweight="bold",
                color="black",
                zorder=5,
            )

        # Reserve headroom so the IBS annotations are not clipped.
        y_min = float(np.nanmin(lower_ci_values + point_values))
        y_max = float(np.nanmax(upper_ci_values + point_values))
        y_span = max(y_max - y_min, 0.25)
        ax.set_ylim(y_min - 0.08 * y_span, y_max + 0.36 * y_span)

        title = "Baseline" if landmark == 0 else f"ART year {int(landmark)}"
        ax.set_title(title, fontweight="bold")
        ax.set_xticks(np.arange(1, 5))
        ax.set_xticklabels(
            ["Cox", "RSF", "RNN", "LSTM"],
            rotation=0,
            ha="center",
        )
        ax.set_ylabel(r"Integrated Brier score ($\times 10^3$)")
        ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
        style_axis(ax, grid_axis="y")
        add_panel_label(ax, chr(65 + panel_i))

    pd.DataFrame(summary_rows).to_csv(
        SOURCE_DIR / "FigureS_bootstrap_IBS_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    handles = [
        Patch(
            facecolor=MODEL_COLORS[m],
            alpha=0.90,
            label=MODEL_LABELS[m],
        )
        for m in MODEL_ORDER
    ]
    handles.append(
        Line2D(
            [0],
            [0],
            marker="D",
            color="black",
            linestyle="none",
            label="Original point estimate",
        )
    )
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False)
    fig.suptitle(
        "Paired-bootstrap distributions of integrated Brier score (IBS)",
        fontsize=14.5,
        fontweight="bold",
        y=1.02,
    )
    fig.subplots_adjust(bottom=0.23, top=0.82, wspace=0.30)

    save_figure(fig, "FigureS_Bootstrap_IBS_boxplots")


if __name__ == "__main__":
    plot_dynamic_brier_with_paired_delta()
    plot_bootstrap_ibs_boxplots()
