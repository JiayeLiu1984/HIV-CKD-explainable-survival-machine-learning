"""
Module 4: Model evaluation
"""

import os
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sksurv.util import Surv
from sksurv.metrics import (
    concordance_index_censored,
    cumulative_dynamic_auc,
    brier_score,
    integrated_brier_score
)
from sksurv.nonparametric import kaplan_meier_estimator

warnings.filterwarnings("ignore")
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["axes.unicode_minus"] = False


# =========================
# Configuration
# =========================
INPUT_DIR = os.path.join("data", "split_data")
FEATURE_FILE = os.path.join("results", "rsf_model", "selected_features_used_in_final_model.csv")
MODEL_FILE = os.path.join("results", "rsf_model", "best_rsf_model.joblib")
OUTPUT_DIR = os.path.join("results", "model_evaluation")

TIME_COL = "interval"
EVENT_COL = "CKDstatus"

REQUESTED_HORIZONS = [12, 36, 60]
THRESHOLDS = np.arange(0.01, 0.81, 0.01)


# =========================
# Utility functions
# =========================
def load_data(input_dir: str):
    """Load split datasets."""
    X_train = pd.read_csv(os.path.join(input_dir, "X_train.csv"), encoding="utf-8-sig")
    X_test = pd.read_csv(os.path.join(input_dir, "X_test.csv"), encoding="utf-8-sig")
    y_train = pd.read_csv(os.path.join(input_dir, "y_train.csv"), encoding="utf-8-sig")
    y_test = pd.read_csv(os.path.join(input_dir, "y_test.csv"), encoding="utf-8-sig")
    return X_train, X_test, y_train, y_test


def load_selected_features(feature_file: str) -> list:
    """Load selected features actually used in the final trained model."""
    if not os.path.exists(feature_file):
        raise FileNotFoundError(f"Feature file not found: {feature_file}")

    feature_df = pd.read_csv(feature_file, encoding="utf-8-sig")
    if "Feature" not in feature_df.columns:
        raise ValueError("Feature file must contain a column named 'Feature'.")

    selected_features = feature_df["Feature"].tolist()
    if len(selected_features) == 0:
        raise ValueError("No features found in selected feature file.")

    return selected_features


def validate_inputs(X_train, X_test, y_train, y_test, selected_features):
    """Validate predictor and survival outcome data."""
    missing_train_features = [col for col in selected_features if col not in X_train.columns]
    missing_test_features = [col for col in selected_features if col not in X_test.columns]

    if missing_train_features:
        raise ValueError(f"Selected features missing in X_train: {missing_train_features}")
    if missing_test_features:
        raise ValueError(f"Selected features missing in X_test: {missing_test_features}")

    for name, y_df in [("y_train", y_train), ("y_test", y_test)]:
        required_cols = [TIME_COL, EVENT_COL]
        missing_cols = [col for col in required_cols if col not in y_df.columns]
        if missing_cols:
            raise ValueError(f"{name} is missing required columns: {missing_cols}")

        if y_df[TIME_COL].isna().any():
            raise ValueError(f"{name} column '{TIME_COL}' contains missing values.")

        if y_df[EVENT_COL].isna().any():
            raise ValueError(f"{name} column '{EVENT_COL}' contains missing values.")

        if not set(y_df[EVENT_COL].dropna().unique()).issubset({0, 1}):
            raise ValueError(f"{name} column '{EVENT_COL}' must contain only 0 and 1.")

        if (pd.to_numeric(y_df[TIME_COL], errors="raise") <= 0).any():
            raise ValueError(f"{name} column '{TIME_COL}' must contain positive values.")


def build_survival_outcome(y_df: pd.DataFrame):
    """Construct structured survival outcome."""
    return Surv.from_arrays(
        event=y_df[EVENT_COL].astype(bool),
        time=y_df[TIME_COL].astype(float)
    )


def save_dataframe(df: pd.DataFrame, output_dir: str, filename: str) -> None:
    """Save dataframe to CSV."""
    df.to_csv(
        os.path.join(output_dir, filename),
        index=False,
        encoding="utf-8-sig"
    )


def step_value(step_times, step_values, t):
    """Return step-function value at time t."""
    idx = np.searchsorted(step_times, t, side="right") - 1
    if idx < 0:
        return 1.0
    return step_values[idx]


def survival_prob_matrix(model, X, times):
    """Return survival probabilities for all samples at given times."""
    surv_fns = model.predict_survival_function(X)
    surv_probs = np.row_stack([[fn(t) for t in times] for fn in surv_fns])
    return surv_probs


def event_prob_matrix(model, X, times):
    """Return event probabilities at given times."""
    return 1.0 - survival_prob_matrix(model, X, times)


def km_event_probability(event, time, horizon):
    """Estimate observed event probability at a given horizon."""
    km_time, km_surv = kaplan_meier_estimator(event.astype(bool), time.astype(float))
    surv_t = step_value(km_time, km_surv, horizon)
    return 1.0 - surv_t


def calibration_by_quantile(event, time, pred_event_prob, horizon, n_bins=5):
    """Generate calibration data using quantile-based grouping."""
    df = pd.DataFrame({
        "event": event.astype(bool),
        "time": time.astype(float),
        "pred": pred_event_prob.astype(float)
    }).copy()

    df["bin"] = pd.qcut(df["pred"], q=n_bins, duplicates="drop")

    rows = []
    for bin_name, sub in df.groupby("bin", observed=False):
        rows.append({
            "bin": str(bin_name),
            "mean_predicted": sub["pred"].mean(),
            "observed_event_probability": km_event_probability(
                sub["event"].values,
                sub["time"].values,
                horizon
            ),
            "n": len(sub)
        })

    return pd.DataFrame(rows)


def censoring_survival_prob(event, time, t):
    """Estimate censoring survival function G(t)."""
    censor_event = ~event.astype(bool)
    km_time, km_surv = kaplan_meier_estimator(censor_event, time.astype(float))
    g_t = step_value(km_time, km_surv, t)
    return max(g_t, 1e-6)


def dca_at_horizon(event, time, pred_event_prob, horizon, thresholds):
    """
    Horizon-specific decision curve analysis for survival outcomes.
    Custom IPCW-based implementation.
    """
    event = event.astype(bool)
    time = time.astype(float)
    pred_event_prob = pred_event_prob.astype(float)

    n = len(time)
    rows = []

    g_h = censoring_survival_prob(event, time, horizon)

    is_case = event & (time <= horizon)
    is_control = time > horizon

    case_weights = np.zeros(n)
    control_weights = np.zeros(n)

    for i in range(n):
        if is_case[i]:
            g_ti = censoring_survival_prob(event, time, time[i])
            case_weights[i] = 1.0 / g_ti
        elif is_control[i]:
            control_weights[i] = 1.0 / g_h

    weighted_case_rate = case_weights.sum() / n
    weighted_control_rate = control_weights.sum() / n

    for pt in thresholds:
        treat = pred_event_prob >= pt

        tp = np.sum(case_weights[treat]) / n
        fp = np.sum(control_weights[treat]) / n

        net_benefit_model = tp - fp * (pt / (1.0 - pt))
        net_benefit_all = weighted_case_rate - weighted_control_rate * (pt / (1.0 - pt))
        net_benefit_none = 0.0

        rows.append({
            "threshold": pt,
            "net_benefit_model": net_benefit_model,
            "net_benefit_all": net_benefit_all,
            "net_benefit_none": net_benefit_none
        })

    return pd.DataFrame(rows)


def get_eval_horizons(y_train: pd.DataFrame, y_test: pd.DataFrame, requested_horizons: list) -> np.ndarray:
    """Generate valid evaluation horizons for AUC/Brier/IBS."""
    max_followup = min(
        float(y_train[TIME_COL].max()),
        float(y_test[TIME_COL].max())
    )

    eval_horizons = [float(t) for t in requested_horizons if 0 < float(t) < max_followup]

    if len(eval_horizons) < 2:
        lower = max(
            float(y_test[TIME_COL].quantile(0.2)),
            1e-6
        )
        upper = min(
            float(y_test[TIME_COL].quantile(0.8)),
            max_followup - 1e-6
        )

        if lower >= upper:
            lower = max_followup * 0.25
            upper = max_followup * 0.75

        eval_horizons = np.linspace(lower, upper, 5).tolist()

    eval_horizons = np.array(sorted(set([float(t) for t in eval_horizons])))

    if len(eval_horizons) < 2:
        raise ValueError("At least two valid evaluation horizons are required for time-dependent metrics.")

    return eval_horizons


def plot_evaluation_summary(
    train_cindex,
    test_cindex,
    auc_df,
    mean_auc,
    brier_df,
    ibs,
    calibration_df,
    eval_horizons,
    dca_df_all,
    dca_horizons,
    output_dir
):
    """Plot summary figure for model evaluation."""
    fig = plt.figure(figsize=(14, 10), facecolor="white")

    # C-index
    ax1 = plt.subplot(2, 3, 1)
    ax1.bar(["Train", "Test"], [train_cindex, test_cindex])
    ax1.set_title("C-index")
    ax1.set_ylabel("C-index")
    ax1.set_ylim(0, 1)

    # Time-dependent AUC
    ax2 = plt.subplot(2, 3, 2)
    ax2.plot(auc_df["time"], auc_df["AUC"], marker="o", linewidth=2)
    ax2.set_title(f"Time-dependent AUC (iAUC={mean_auc:.3f})")
    ax2.set_xlabel("Time")
    ax2.set_ylabel("AUC")
    ax2.set_ylim(0, 1)

    # Brier score
    ax3 = plt.subplot(2, 3, 3)
    ax3.plot(brier_df["time"], brier_df["Brier_score"], marker="o", linewidth=2)
    ax3.set_title(f"Brier score (IBS={ibs:.3f})")
    ax3.set_xlabel("Time")
    ax3.set_ylabel("Brier score")

    # Calibration
    ax4 = plt.subplot(2, 3, 4)
    for horizon in eval_horizons:
        sub = calibration_df[calibration_df["horizon"] == horizon]
        ax4.plot(
            sub["mean_predicted"],
            sub["observed_event_probability"],
            marker="o",
            linewidth=2,
            label=f"{int(horizon)}"
        )

    max_val = max(
        calibration_df["mean_predicted"].max(),
        calibration_df["observed_event_probability"].max()
    )
    ax4.plot([0, max_val], [0, max_val], linestyle="--", color="gray")
    ax4.set_title("Calibration curves")
    ax4.set_xlabel("Predicted event probability")
    ax4.set_ylabel("Observed event probability")
    ax4.legend(title="Time")

    # DCA 1
    if len(dca_horizons) >= 1:
        ax5 = plt.subplot(2, 3, 5)
        sub = dca_df_all[dca_df_all["horizon"] == dca_horizons[0]]
        ax5.plot(sub["threshold"], sub["net_benefit_model"], linewidth=2, label="RSF")
        ax5.plot(sub["threshold"], sub["net_benefit_all"], linestyle="--", label="Treat all")
        ax5.plot(sub["threshold"], sub["net_benefit_none"], linestyle=":", label="Treat none")
        ax5.set_title(f"DCA at {int(dca_horizons[0])}")
        ax5.set_xlabel("Threshold probability")
        ax5.set_ylabel("Net benefit")
        ax5.legend()

    # DCA 2
    if len(dca_horizons) >= 2:
        ax6 = plt.subplot(2, 3, 6)
        sub = dca_df_all[dca_df_all["horizon"] == dca_horizons[1]]
        ax6.plot(sub["threshold"], sub["net_benefit_model"], linewidth=2, label="RSF")
        ax6.plot(sub["threshold"], sub["net_benefit_all"], linestyle="--", label="Treat all")
        ax6.plot(sub["threshold"], sub["net_benefit_none"], linestyle=":", label="Treat none")
        ax6.set_title(f"DCA at {int(dca_horizons[1])}")
        ax6.set_xlabel("Threshold probability")
        ax6.set_ylabel("Net benefit")
        ax6.legend()

    plt.tight_layout()

    plt.savefig(
        os.path.join(output_dir, "model_evaluation_summary.pdf"),
        dpi=600,
        bbox_inches="tight"
    )
    plt.savefig(
        os.path.join(output_dir, "model_evaluation_summary.png"),
        dpi=600,
        bbox_inches="tight"
    )
    plt.close()


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load data
    X_train, X_test, y_train, y_test = load_data(INPUT_DIR)
    selected_features = load_selected_features(FEATURE_FILE)
    validate_inputs(X_train, X_test, y_train, y_test, selected_features)

    # Subset predictors
    X_train_sel = X_train[selected_features].apply(pd.to_numeric, errors="raise").astype(np.float32)
    X_test_sel = X_test[selected_features].apply(pd.to_numeric, errors="raise").astype(np.float32)

    # Build survival outcomes
    y_train_surv = build_survival_outcome(y_train)
    y_test_surv = build_survival_outcome(y_test)

    # Load trained model
    if not os.path.exists(MODEL_FILE):
        raise FileNotFoundError(f"Model file not found: {MODEL_FILE}")
    model = joblib.load(MODEL_FILE)

    # 1. C-index
    train_risk = model.predict(X_train_sel)
    test_risk = model.predict(X_test_sel)

    train_cindex = concordance_index_censored(
        y_train[EVENT_COL].astype(bool),
        y_train[TIME_COL].astype(float),
        train_risk
    )[0]

    test_cindex = concordance_index_censored(
        y_test[EVENT_COL].astype(bool),
        y_test[TIME_COL].astype(float),
        test_risk
    )[0]

    # 2. Time-dependent AUC / iAUC
    eval_horizons = get_eval_horizons(y_train, y_test, REQUESTED_HORIZONS)

    auc_values, mean_auc = cumulative_dynamic_auc(
        y_train_surv,
        y_test_surv,
        test_risk,
        eval_horizons
    )

    auc_df = pd.DataFrame({
        "time": eval_horizons,
        "AUC": auc_values
    })

    # 3. Brier score / IBS
    surv_probs_test = survival_prob_matrix(model, X_test_sel, eval_horizons)

    brier_times, brier_values = brier_score(
        y_train_surv,
        y_test_surv,
        surv_probs_test,
        eval_horizons
    )

    ibs = integrated_brier_score(
        y_train_surv,
        y_test_surv,
        surv_probs_test,
        eval_horizons
    )

    brier_df = pd.DataFrame({
        "time": brier_times,
        "Brier_score": brier_values
    })

    # 4. Calibration
    event_probs_test = event_prob_matrix(model, X_test_sel, eval_horizons)

    calibration_list = []
    for j, horizon in enumerate(eval_horizons):
        cal_df = calibration_by_quantile(
            event=y_test[EVENT_COL].values,
            time=y_test[TIME_COL].values,
            pred_event_prob=event_probs_test[:, j],
            horizon=horizon,
            n_bins=5
        )
        cal_df["horizon"] = horizon
        calibration_list.append(cal_df)

    calibration_df = pd.concat(calibration_list, ignore_index=True)

    # 5. DCA
    dca_horizons = eval_horizons[:2] if len(eval_horizons) >= 2 else eval_horizons

    dca_list = []
    for j, horizon in enumerate(dca_horizons):
        dca_df = dca_at_horizon(
            event=y_test[EVENT_COL].values,
            time=y_test[TIME_COL].values,
            pred_event_prob=event_probs_test[:, j],
            horizon=horizon,
            thresholds=THRESHOLDS
        )
        dca_df["horizon"] = horizon
        dca_list.append(dca_df)

    dca_df_all = pd.concat(dca_list, ignore_index=True)

    # Save evaluation tables
    summary_df = pd.DataFrame({
        "Metric": [
            "Training C-index",
            "Testing C-index",
            "Integrated AUC (iAUC)",
            "Integrated Brier Score (IBS)"
        ],
        "Value": [
            train_cindex,
            test_cindex,
            mean_auc,
            ibs
        ]
    })

    save_dataframe(summary_df, OUTPUT_DIR, "evaluation_summary.csv")
    save_dataframe(auc_df, OUTPUT_DIR, "time_dependent_auc.csv")
    save_dataframe(brier_df, OUTPUT_DIR, "brier_score.csv")
    save_dataframe(calibration_df, OUTPUT_DIR, "calibration_data.csv")
    save_dataframe(dca_df_all, OUTPUT_DIR, "dca_data.csv")

    # Plot figures
    plot_evaluation_summary(
        train_cindex=train_cindex,
        test_cindex=test_cindex,
        auc_df=auc_df,
        mean_auc=mean_auc,
        brier_df=brier_df,
        ibs=ibs,
        calibration_df=calibration_df,
        eval_horizons=eval_horizons,
        dca_df_all=dca_df_all,
        dca_horizons=dca_horizons,
        output_dir=OUTPUT_DIR
    )

    # Print summary
    print("Model evaluation completed successfully.")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Training C-index: {train_cindex:.4f}")
    print(f"Testing C-index: {test_cindex:.4f}")
    print(f"Integrated AUC (iAUC): {mean_auc:.4f}")
    print(f"Integrated Brier Score (IBS): {ibs:.4f}")


if __name__ == "__main__":
    main()
