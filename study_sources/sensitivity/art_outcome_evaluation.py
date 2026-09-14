from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from sksurv.metrics import (
    brier_score,
    concordance_index_ipcw,
    cumulative_dynamic_auc,
    integrated_brier_score,
)
from sksurv.nonparametric import kaplan_meier_estimator
from sksurv.util import Surv


PROJECT_DIR = Path("__CKD_WORKDIR__")
STEP6_DIR = PROJECT_DIR / "rolling_5y_step6_super_landmark_data"
LSTM_DIR = PROJECT_DIR / "rolling_5y_step10e_lstm_v2_final_oof_selected_existing_trials"
STEP11_DIR = PROJECT_DIR / "rolling_5y_step11_v2_four_model_oof_comparison_lstm_v2"
OUTPUT_DIR = PROJECT_DIR / "comment1_sensitivity_20260903"
COMPARISON_FILE = OUTPUT_DIR / "patient_level_outcome_comparison.csv"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LANDMARK_MONTHS = np.asarray([0, 12, 24, 36, 48, 60], dtype=float)
PRIMARY_LANDMARK_POSITIONS = np.asarray([0, 1, 3, 5], dtype=int)
FUTURE_END_MONTHS = np.arange(6, 61, 6, dtype=float)
METRIC_TIMES = np.asarray([6, 12, 18, 24, 30, 36, 42, 48, 54, 59.999], dtype=float)
HORIZON_MONTHS = (12.0, 36.0, 60.0)
EPS = 1e-7
TIME_TOLERANCE = 1e-6
CALIBRATION_RIDGE = 1e-6


def normalize_id(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip().str.replace(r"\.0$", "", regex=True)


def probability_logit(probability: np.ndarray) -> np.ndarray:
    probability = np.clip(np.asarray(probability, dtype=float), EPS, 1.0 - EPS)
    return np.log(probability) - np.log1p(-probability)


def hazard_to_risk(hazard: np.ndarray) -> np.ndarray:
    hazard = np.clip(np.asarray(hazard, dtype=float), EPS, 1.0 - EPS)
    return 1.0 - np.cumprod(1.0 - hazard, axis=1)


def fit_interval_hazard_calibrator(
    hazard: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
) -> dict[str, Any]:
    raw_logit = probability_logit(hazard)
    interval_matrix = np.broadcast_to(np.arange(10, dtype=np.int16), hazard.shape)
    x = raw_logit[mask]
    y = target[mask].astype(float)
    interval_index = interval_matrix[mask]
    if len(y) == 0 or y.sum() <= 0:
        raise ValueError("Calibration training subset has no valid event interval.")

    initial = np.concatenate([np.zeros(10, dtype=float), np.asarray([1.0])])

    def objective_and_gradient(parameters: np.ndarray):
        alpha = parameters[:10]
        beta = parameters[-1]
        eta = alpha[interval_index] + beta * x
        loss = np.mean(np.logaddexp(0.0, eta) - y * eta)
        residual = (expit(eta) - y) / len(y)
        gradient_alpha = np.bincount(
            interval_index, weights=residual, minlength=10
        ).astype(float)
        gradient_beta = float(np.dot(residual, x))
        loss += 0.5 * CALIBRATION_RIDGE * (
            float(np.mean(alpha**2)) + float((beta - 1.0) ** 2)
        )
        gradient_alpha += CALIBRATION_RIDGE * alpha / 10.0
        gradient_beta += CALIBRATION_RIDGE * (beta - 1.0)
        return float(loss), np.concatenate([gradient_alpha, [gradient_beta]])

    result = minimize(
        fun=objective_and_gradient,
        x0=initial,
        method="L-BFGS-B",
        jac=True,
        bounds=[(-8.0, 8.0)] * 10 + [(0.05, 5.0)],
        options={"maxiter": 300, "ftol": 1e-12, "gtol": 1e-8, "maxls": 50},
    )
    if not result.success:
        raise RuntimeError(str(result.message))
    return {
        "alpha": result.x[:10].astype(float),
        "beta": float(result.x[-1]),
        "valid_interval_n": int(len(y)),
        "event_interval_n": int(y.sum()),
        "objective": float(result.fun),
        "iterations": int(result.nit),
    }


def apply_interval_hazard_calibrator(
    hazard: np.ndarray, parameters: dict[str, Any]
) -> np.ndarray:
    raw_logit = probability_logit(hazard)
    eta = raw_logit * float(parameters["beta"]) + np.asarray(parameters["alpha"])[None, :]
    return np.clip(expit(eta), EPS, 1.0 - EPS)


def build_strict_labels(
    metadata: pd.DataFrame,
    comparison: pd.DataFrame,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    comparison = comparison.copy()
    comparison["ID"] = normalize_id(comparison["ID"])
    selected = comparison[
        [
            "ID",
            "sensitivity_CKDstatus",
            "sensitivity_followup_days",
            "sensitivity_CKDtime",
            "Lastfollowtime",
        ]
    ].copy()
    metadata = metadata.copy()
    metadata["ID"] = normalize_id(metadata["ID"])
    merged = metadata.merge(selected, on="ID", how="left", validate="many_to_one")
    if merged["sensitivity_CKDstatus"].isna().any():
        raise ValueError("Some development origin IDs lack sensitivity outcomes.")

    event_patient = merged["sensitivity_CKDstatus"].astype(int).to_numpy()
    observed_time_patient = (
        pd.to_numeric(merged["sensitivity_followup_days"], errors="raise").to_numpy(float)
        / 30.0
    )
    landmark = merged["landmark_month"].to_numpy(float)
    remaining = observed_time_patient - landmark
    eligible = remaining > TIME_TOLERANCE

    n = len(merged)
    target = np.zeros((n, 10), dtype=np.float32)
    at_risk = np.zeros((n, 10), dtype=bool)
    event_within = np.zeros(n, dtype=np.int8)
    analysis_time = np.full(n, np.nan, dtype=np.float32)

    event_in_60 = eligible & event_patient.astype(bool) & (remaining <= 60.0 + TIME_TOLERANCE)
    analysis_time[eligible] = np.minimum(remaining[eligible], 60.0).astype(np.float32)
    event_within[event_in_60] = 1

    event_interval = np.full(n, -1, dtype=int)
    event_interval[event_in_60] = (
        np.ceil((remaining[event_in_60] - TIME_TOLERANCE) / 6.0).astype(int) - 1
    )
    event_interval[event_in_60] = np.clip(event_interval[event_in_60], 0, 9)

    for future_index, interval_end in enumerate(FUTURE_END_MONTHS):
        at_risk[:, future_index] = eligible & (
            (event_in_60 & (event_interval >= future_index))
            | (~event_in_60 & (remaining >= interval_end - TIME_TOLERANCE))
        )
        target[event_in_60 & (event_interval == future_index), future_index] = 1.0

    merged["sensitivity_remaining_time_month"] = remaining
    merged["sensitivity_origin_eligible"] = eligible
    merged["sensitivity_event_within_60m"] = event_within
    merged["sensitivity_analysis_time_month"] = analysis_time
    merged["sensitivity_event_interval_index"] = event_interval
    return merged, target, at_risk, event_within, analysis_time


def evaluate_landmark(
    event: np.ndarray,
    time_month: np.ndarray,
    risk_matrix: np.ndarray,
) -> dict[str, Any]:
    survival = Surv.from_arrays(event.astype(bool), time_month.astype(float))
    c_index = float(
        concordance_index_ipcw(
            survival, survival, risk_matrix[:, -1], tau=float(METRIC_TIMES[-1])
        )[0]
    )
    dynamic_auc, mean_auc = cumulative_dynamic_auc(
        survival, survival, risk_matrix, METRIC_TIMES
    )
    survival_probability = 1.0 - risk_matrix
    _, brier_values = brier_score(
        survival, survival, survival_probability, METRIC_TIMES
    )
    ibs = float(
        integrated_brier_score(
            survival, survival, survival_probability, METRIC_TIMES
        )
    )
    return {
        "uno_c_index_5y": c_index,
        "dynamic_auc": np.asarray(dynamic_auc, dtype=float),
        "integrated_dynamic_auc": float(mean_auc),
        "brier": np.asarray(brier_values, dtype=float),
        "integrated_brier": ibs,
    }


def km_event_risk(
    time_month: np.ndarray, event: np.ndarray, horizon_month: float
) -> tuple[float, float, float]:
    km_time, km_survival, km_ci = kaplan_meier_estimator(
        event.astype(bool), time_month.astype(float), conf_type="log-log"
    )
    eligible = np.where(km_time <= horizon_month)[0]
    if len(eligible) == 0:
        return 0.0, 0.0, 0.0
    index = int(eligible[-1])
    return (
        float(1.0 - km_survival[index]),
        float(1.0 - km_ci[1, index]),
        float(1.0 - km_ci[0, index]),
    )


def evaluate_stage(
    stage: str,
    eligible: np.ndarray,
    landmark_index: np.ndarray,
    event_within: np.ndarray,
    analysis_time: np.ndarray,
    risk: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metric_rows: list[dict[str, Any]] = []
    horizon_rows: list[dict[str, Any]] = []
    overall_calibration_rows: list[dict[str, Any]] = []
    decile_rows: list[dict[str, Any]] = []

    for landmark_position, landmark_month in enumerate(LANDMARK_MONTHS):
        rows = np.where(eligible & (landmark_index == landmark_position))[0]
        metric = evaluate_landmark(
            event_within[rows], analysis_time[rows], risk[rows]
        )
        metric_rows.append(
            {
                "stage": stage,
                "landmark_position": landmark_position,
                "landmark_month": landmark_month,
                "landmark_year": landmark_month / 12.0,
                "risk_set_n": len(rows),
                "future_5y_event_n": int(event_within[rows].sum()),
                "uno_c_index_5y": metric["uno_c_index_5y"],
                "integrated_dynamic_auc": metric["integrated_dynamic_auc"],
                "integrated_brier": metric["integrated_brier"],
            }
        )
        for j, month in enumerate(FUTURE_END_MONTHS):
            horizon_rows.append(
                {
                    "stage": stage,
                    "landmark_position": landmark_position,
                    "landmark_month": landmark_month,
                    "horizon_month": month,
                    "dynamic_auc": metric["dynamic_auc"][j],
                    "brier": metric["brier"][j],
                }
            )

        for horizon in HORIZON_MONTHS:
            pos = int(np.where(np.isclose(FUTURE_END_MONTHS, horizon))[0][0])
            predicted = risk[rows, pos]
            observed, lower, upper = km_event_risk(
                analysis_time[rows], event_within[rows], horizon
            )
            overall_calibration_rows.append(
                {
                    "stage": stage,
                    "landmark_position": landmark_position,
                    "landmark_month": landmark_month,
                    "landmark_year": landmark_month / 12.0,
                    "horizon_month": horizon,
                    "horizon_year": horizon / 12.0,
                    "risk_set_n": len(rows),
                    "mean_predicted_risk": float(np.mean(predicted)),
                    "km_observed_risk": observed,
                    "km_lower_95": lower,
                    "km_upper_95": upper,
                    "calibration_difference": float(np.mean(predicted) - observed),
                }
            )
            group_number = pd.qcut(
                pd.Series(predicted).rank(method="first"),
                q=10,
                labels=False,
            ).to_numpy() + 1
            for group_id in range(1, 11):
                group_mask = group_number == group_id
                g_observed, g_lower, g_upper = km_event_risk(
                    analysis_time[rows][group_mask],
                    event_within[rows][group_mask],
                    horizon,
                )
                decile_rows.append(
                    {
                        "stage": stage,
                        "landmark_position": landmark_position,
                        "landmark_month": landmark_month,
                        "horizon_month": horizon,
                        "risk_group": group_id,
                        "patient_n": int(group_mask.sum()),
                        "mean_predicted_risk": float(np.mean(predicted[group_mask])),
                        "km_observed_risk": g_observed,
                        "km_lower_95": g_lower,
                        "km_upper_95": g_upper,
                    }
                )

    metric_frame = pd.DataFrame(metric_rows)
    primary_mean = pd.DataFrame(
        [
            {
                "stage": stage,
                "scope": "primary_0_1_3_5y_landmark_equal_weight_mean",
                "mean_uno_c_index_5y": float(
                    metric_frame.loc[
                        metric_frame["landmark_position"].isin(PRIMARY_LANDMARK_POSITIONS),
                        "uno_c_index_5y",
                    ].mean()
                ),
                "mean_integrated_dynamic_auc": float(
                    metric_frame.loc[
                        metric_frame["landmark_position"].isin(PRIMARY_LANDMARK_POSITIONS),
                        "integrated_dynamic_auc",
                    ].mean()
                ),
                "mean_integrated_brier": float(
                    metric_frame.loc[
                        metric_frame["landmark_position"].isin(PRIMARY_LANDMARK_POSITIONS),
                        "integrated_brier",
                    ].mean()
                ),
            }
        ]
    )
    return (
        metric_frame,
        pd.DataFrame(horizon_rows),
        pd.DataFrame(overall_calibration_rows),
        pd.DataFrame(decile_rows),
        primary_mean,
    )


def plot_performance(comparison: pd.DataFrame) -> None:
    metric_specs = [
        ("mean_uno_c_index_5y", "Uno C-index", False),
        ("mean_integrated_dynamic_auc", "Integrated AUC", False),
        ("mean_integrated_brier", "Integrated Brier score", True),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2))
    colors = ["#4C78A8", "#E45756", "#72B7B2"]
    labels = ["Primary outcome", "Strict outcome\n(frozen calibration)", "Strict outcome\n(recalibrated)"]
    for axis, (column, title, lower_better) in zip(axes, metric_specs):
        values = comparison[column].to_numpy(float)
        axis.bar(np.arange(len(values)), values, color=colors, width=0.68)
        for i, value in enumerate(values):
            axis.text(i, value, f"{value:.4f}", ha="center", va="bottom", fontsize=9)
        axis.set_title(title + (" (lower is better)" if lower_better else ""))
        axis.set_xticks(np.arange(len(values)), labels, rotation=18, ha="right")
        axis.grid(axis="y", alpha=0.2)
        ymin = max(0.0, float(values.min()) - max(0.02, float(np.ptp(values)) * 0.8))
        ymax = float(values.max()) + max(0.02, float(np.ptp(values)) * 0.8)
        axis.set_ylim(ymin, ymax)
    fig.suptitle("LSTM-v2 performance under the strict CKD sensitivity outcome", y=1.03)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "figure_model_performance_comparison.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / "figure_model_performance_comparison.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_calibration(deciles: pd.DataFrame) -> None:
    stages = ["strict_frozen_original_calibration", "strict_crossfit_recalibrated"]
    stage_labels = ["Frozen original calibration", "Sensitivity-specific recalibration"]
    colors = ["#E45756", "#72B7B2"]
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 9.0))
    for axis, landmark_position in zip(axes.flat, PRIMARY_LANDMARK_POSITIONS):
        max_value = 0.0
        for stage, label, color in zip(stages, stage_labels, colors):
            sub = deciles[
                (deciles["stage"] == stage)
                & (deciles["landmark_position"] == landmark_position)
                & (deciles["horizon_month"] == 60.0)
            ]
            axis.plot(
                sub["mean_predicted_risk"],
                sub["km_observed_risk"],
                marker="o",
                linewidth=1.7,
                markersize=4,
                label=label,
                color=color,
            )
            max_value = max(
                max_value,
                float(sub[["mean_predicted_risk", "km_observed_risk"]].to_numpy().max()),
            )
        upper = max(0.02, max_value * 1.08)
        axis.plot([0, upper], [0, upper], linestyle="--", color="#555555", linewidth=1)
        axis.set_xlim(0, upper)
        axis.set_ylim(0, upper)
        axis.set_title(f"Landmark {LANDMARK_MONTHS[landmark_position] / 12:.0f} years")
        axis.set_xlabel("Predicted 5-year CKD risk")
        axis.set_ylabel("Observed 5-year CKD risk (KM)")
        axis.grid(alpha=0.2)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False)
    fig.suptitle("Calibration under the strict CKD sensitivity outcome", y=0.98)
    fig.tight_layout(rect=(0, 0.06, 1, 0.96))
    fig.savefig(OUTPUT_DIR / "figure_sensitivity_calibration_5y.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / "figure_sensitivity_calibration_5y.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    metadata = pd.read_csv(
        STEP6_DIR / "super_landmark_development_metadata.csv",
        encoding="utf-8-sig",
        dtype={"ID": "string"},
    )
    comparison = pd.read_csv(
        COMPARISON_FILE,
        encoding="utf-8-sig",
        dtype={"ID": "string"},
    )
    merged, target, mask, event_within, analysis_time = build_strict_labels(
        metadata, comparison
    )
    eligible = merged["sensitivity_origin_eligible"].to_numpy(bool)
    landmark_index = merged["landmark_index"].to_numpy(int)
    fold_id = merged["fold_id"].to_numpy(int)

    raw_hazard = np.load(LSTM_DIR / "lstm_v2_oof_hazard_long.npy").astype(float)
    frozen_risk = np.load(
        STEP11_DIR / "lstm_v2_crossfit_calibrated_oof_risk_long.npy"
    ).astype(float)
    if raw_hazard.shape != target.shape or frozen_risk.shape != target.shape:
        raise ValueError("Prediction and sensitivity-label arrays are misaligned.")

    recalibrated_hazard = np.full_like(raw_hazard, np.nan, dtype=float)
    calibration_parameter_rows = []
    for landmark_position in range(6):
        for heldout_fold in range(5):
            train_rows = eligible & (landmark_index == landmark_position) & (fold_id != heldout_fold)
            validation_rows = eligible & (landmark_index == landmark_position) & (fold_id == heldout_fold)
            parameters = fit_interval_hazard_calibrator(
                raw_hazard[train_rows], target[train_rows], mask[train_rows]
            )
            recalibrated_hazard[validation_rows] = apply_interval_hazard_calibrator(
                raw_hazard[validation_rows], parameters
            )
            for interval_index, alpha in enumerate(parameters["alpha"]):
                calibration_parameter_rows.append(
                    {
                        "landmark_position": landmark_position,
                        "landmark_month": LANDMARK_MONTHS[landmark_position],
                        "heldout_fold": heldout_fold,
                        "interval_index": interval_index,
                        "interval_end_month": FUTURE_END_MONTHS[interval_index],
                        "alpha": alpha,
                        "beta": parameters["beta"],
                        "valid_interval_n": parameters["valid_interval_n"],
                        "event_interval_n": parameters["event_interval_n"],
                    }
                )
    if np.isnan(recalibrated_hazard[eligible]).any():
        raise ValueError("Sensitivity-specific cross-fitted calibration is incomplete.")
    recalibrated_risk = np.full_like(recalibrated_hazard, np.nan, dtype=float)
    recalibrated_risk[eligible] = hazard_to_risk(recalibrated_hazard[eligible])

    stage_results = []
    for stage, risk in [
        ("strict_frozen_original_calibration", frozen_risk),
        ("strict_crossfit_recalibrated", recalibrated_risk),
    ]:
        stage_results.append(
            (stage, evaluate_stage(stage, eligible, landmark_index, event_within, analysis_time, risk))
        )

    landmark_metrics = pd.concat([result[1][0] for result in stage_results], ignore_index=True)
    horizon_metrics = pd.concat([result[1][1] for result in stage_results], ignore_index=True)
    overall_calibration = pd.concat([result[1][2] for result in stage_results], ignore_index=True)
    decile_calibration = pd.concat([result[1][3] for result in stage_results], ignore_index=True)
    strict_means = pd.concat([result[1][4] for result in stage_results], ignore_index=True)

    primary_means = pd.read_csv(
        STEP11_DIR / "four_model_equal_weight_mean_performance.csv",
        encoding="utf-8-sig",
    )
    primary = primary_means[
        (primary_means["model"] == "LSTM-v2")
        & (
            primary_means["scope"]
            == "primary_0_1_3_5y_landmark_equal_weight_mean"
        )
    ][
        [
            "mean_uno_c_index_5y",
            "mean_integrated_dynamic_auc",
            "mean_integrated_brier",
        ]
    ].copy()
    primary.insert(0, "stage", "primary_outcome_crossfit_calibrated")
    primary.insert(1, "scope", "primary_0_1_3_5y_landmark_equal_weight_mean")
    performance_comparison = pd.concat([primary, strict_means], ignore_index=True)
    base = performance_comparison.iloc[0]
    for column in [
        "mean_uno_c_index_5y",
        "mean_integrated_dynamic_auc",
        "mean_integrated_brier",
    ]:
        performance_comparison[f"difference_vs_primary_{column}"] = (
            performance_comparison[column] - float(base[column])
        )

    primary_landmark = pd.read_csv(
        STEP11_DIR / "four_model_crossfit_calibrated_landmark_metrics.csv",
        encoding="utf-8-sig",
    )
    primary_landmark = primary_landmark[primary_landmark["model"] == "LSTM-v2"].copy()
    primary_landmark.insert(0, "stage", "primary_outcome_crossfit_calibrated")
    aligned_landmark_metrics = pd.concat(
        [
            primary_landmark[
                [
                    "stage",
                    "landmark_position",
                    "landmark_month",
                    "landmark_year",
                    "risk_set_n",
                    "future_5y_event_n",
                    "uno_c_index_5y",
                    "integrated_dynamic_auc",
                    "integrated_brier",
                ]
            ],
            landmark_metrics,
        ],
        ignore_index=True,
    )

    merged.to_csv(
        OUTPUT_DIR / "sensitivity_fixed_origin_metadata.csv",
        index=False,
        encoding="utf-8-sig",
    )
    np.save(OUTPUT_DIR / "sensitivity_future_event_long.npy", target)
    np.save(OUTPUT_DIR / "sensitivity_future_at_risk_long.npy", mask)
    np.save(OUTPUT_DIR / "sensitivity_event_within_60m.npy", event_within)
    np.save(OUTPUT_DIR / "sensitivity_analysis_time_month.npy", analysis_time)
    np.save(OUTPUT_DIR / "sensitivity_crossfit_recalibrated_risk_long.npy", recalibrated_risk)
    pd.DataFrame(calibration_parameter_rows).to_csv(
        OUTPUT_DIR / "sensitivity_crossfit_calibration_parameters.csv",
        index=False,
        encoding="utf-8-sig",
    )
    performance_comparison.to_csv(
        OUTPUT_DIR / "performance_comparison.csv", index=False, encoding="utf-8-sig"
    )
    aligned_landmark_metrics.to_csv(
        OUTPUT_DIR / "landmark_performance_comparison.csv",
        index=False,
        encoding="utf-8-sig",
    )
    horizon_metrics.to_csv(
        OUTPUT_DIR / "sensitivity_horizon_metrics.csv", index=False, encoding="utf-8-sig"
    )
    overall_calibration.to_csv(
        OUTPUT_DIR / "sensitivity_calibration_overall.csv",
        index=False,
        encoding="utf-8-sig",
    )
    decile_calibration.to_csv(
        OUTPUT_DIR / "sensitivity_calibration_deciles.csv",
        index=False,
        encoding="utf-8-sig",
    )
    plot_performance(performance_comparison)
    plot_calibration(decile_calibration)

    qc = {
        "development_origin_n_original": int(len(metadata)),
        "development_origin_n_sensitivity_eligible": int(eligible.sum()),
        "excluded_origin_after_sensitivity_end_n": int((~eligible).sum()),
        "development_unique_patient_n_in_existing_origins": int(metadata["ID"].nunique()),
        "strict_future_5y_events_across_origins": int(event_within[eligible].sum()),
        "label_event_outside_risk_mask_n": int(((target > 0) & ~mask).sum()),
        "eligible_recalibrated_prediction_missing_n": int(
            np.isnan(recalibrated_risk[eligible]).sum()
        ),
        "analysis_definition": (
            "Fixed existing prediction origins; LSTM-v2 weights and feature histories unchanged. "
            "Strict outcome labels reconstructed from complete eGFR follow-up."
        ),
    }
    (OUTPUT_DIR / "model_sensitivity_qc.json").write_text(
        json.dumps(qc, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(performance_comparison.to_string(index=False))
    print(json.dumps(qc, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
