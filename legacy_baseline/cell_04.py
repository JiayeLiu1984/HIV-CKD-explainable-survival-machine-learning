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
INPUT_DIR = "__CKD_LEGACY_WORKDIR__/split_data"
FEATURE_DIR = "__CKD_LEGACY_WORKDIR__/feature_selection"
MODEL_DIR = "__CKD_LEGACY_WORKDIR__/rsf_model"
OUTPUT_DIR = "__CKD_LEGACY_WORKDIR__/model_evaluation"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Specific time points for different evaluations (months)
CALIB_HORIZONS = [12, 36, 60, 120]  # 1, 3, 5, 10 years
DCA_HORIZONS = [60, 120]  # 5, 10 years
N_INTEGRATED_POINTS = 30  # 30 points for iAUC and IBS

# Threshold range for DCA
THRESHOLDS = np.arange(0.01, 0.81, 0.01)


# =========================
# Helper functions
# =========================
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
    """Return event probabilities = 1 - survival probabilities."""
    return 1.0 - survival_prob_matrix(model, X, times)


def km_event_probability(event, time, horizon):
    """Estimate observed event probability at a given horizon."""
    km_time, km_surv = kaplan_meier_estimator(event, time)
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
    """Horizon-specific decision curve analysis for survival outcomes."""
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


# =========================
# Load data
# =========================
X_train = pd.read_csv(os.path.join(INPUT_DIR, "X_train.csv"), encoding="utf-8-sig")
X_test = pd.read_csv(os.path.join(INPUT_DIR, "X_test.csv"), encoding="utf-8-sig")
y_train = pd.read_csv(os.path.join(INPUT_DIR, "y_train.csv"), encoding="utf-8-sig")
y_test = pd.read_csv(os.path.join(INPUT_DIR, "y_test.csv"), encoding="utf-8-sig")

selected_features = pd.read_csv(
    os.path.join(FEATURE_DIR, "final_top_features.csv"),
    encoding="utf-8-sig"
)["Feature"].tolist()

X_train_sel = X_train[selected_features].astype(np.float32)
X_test_sel = X_test[selected_features].astype(np.float32)

y_train_surv = Surv.from_arrays(
    event=y_train["CKDstatus"].astype(bool),
    time=y_train["interval"].astype(float)
)

y_test_surv = Surv.from_arrays(
    event=y_test["CKDstatus"].astype(bool),
    time=y_test["interval"].astype(float)
)

# =========================
# Load final trained model
# =========================
model = joblib.load(os.path.join(MODEL_DIR, "best_rsf_model.joblib"))

# =========================
# 1. C-index
# =========================
train_risk = model.predict(X_train_sel)
test_risk = model.predict(X_test_sel)

train_cindex = concordance_index_censored(
    y_train["CKDstatus"].astype(bool),
    y_train["interval"].astype(float),
    train_risk
)[0]

test_cindex = concordance_index_censored(
    y_test["CKDstatus"].astype(bool),
    y_test["interval"].astype(float),
    test_risk
)[0]

# =========================
# 2. iAUC / IBS (30 Points)
# =========================
# Select 30 time points within the valid follow-up range
t_min = y_test["interval"].quantile(0.05)
t_max = y_test["interval"].quantile(0.95)
eval_times_30 = np.linspace(t_min, t_max, N_INTEGRATED_POINTS)

auc_values, mean_auc = cumulative_dynamic_auc(
    y_train_surv,
    y_test_surv,
    test_risk,
    eval_times_30
)

surv_probs_test_30 = survival_prob_matrix(model, X_test_sel, eval_times_30)
brier_times, brier_values = brier_score(
    y_train_surv,
    y_test_surv,
    surv_probs_test_30,
    eval_times_30
)

ibs = integrated_brier_score(
    y_train_surv,
    y_test_surv,
    surv_probs_test_30,
    eval_times_30
)

# =========================
# 3. Calibration (1, 3, 5, 10 Years)
# =========================
actual_cal_horizons = [t for t in CALIB_HORIZONS if t < y_test["interval"].max()]
calibration_list = []

for horizon in actual_cal_horizons:
    p_event = event_prob_matrix(model, X_test_sel, [horizon]).flatten()
    cal_df = calibration_by_quantile(
        event=y_test["CKDstatus"].values,
        time=y_test["interval"].values,
        pred_event_prob=p_event,
        horizon=horizon,
        n_bins=5
    )
    cal_df["horizon"] = horizon
    calibration_list.append(cal_df)

calibration_df = pd.concat(calibration_list, ignore_index=True)

# =========================
# 4. DCA (5, 10 Years)
# =========================
actual_dca_horizons = [t for t in DCA_HORIZONS if t < y_test["interval"].max()]
dca_list = []

for horizon in actual_dca_horizons:
    p_event = event_prob_matrix(model, X_test_sel, [horizon]).flatten()
    dca_res = dca_at_horizon(
        event=y_test["CKDstatus"].values,
        time=y_test["interval"].values,
        pred_event_prob=p_event,
        horizon=horizon,
        thresholds=THRESHOLDS
    )
    dca_res["horizon"] = horizon
    dca_list.append(dca_res)

dca_df_all = pd.concat(dca_list, ignore_index=True)

# =========================
# Save evaluation tables
# =========================
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

summary_df.to_csv(os.path.join(OUTPUT_DIR, "evaluation_summary.csv"), index=False, encoding="utf-8-sig")
pd.DataFrame({"time": eval_times_30, "AUC": auc_values}).to_csv(os.path.join(OUTPUT_DIR, "time_dependent_auc.csv"),
                                                                index=False)
pd.DataFrame({"time": brier_times, "Brier_score": brier_values}).to_csv(os.path.join(OUTPUT_DIR, "brier_score.csv"),
                                                                        index=False)
calibration_df.to_csv(os.path.join(OUTPUT_DIR, "calibration_data.csv"), index=False)
dca_df_all.to_csv(os.path.join(OUTPUT_DIR, "dca_data.csv"), index=False)

# =========================
# Plot figures
# =========================
fig = plt.figure(figsize=(16, 10), facecolor="white")

# C-index
ax1 = plt.subplot(2, 3, 1)
ax1.bar(["Train", "Test"], [train_cindex, test_cindex], color=['#2c3e50', '#3498db'])
ax1.set_title("C-index")
ax1.set_ylim(0, 1)

# Time-dependent AUC (30 points)
ax2 = plt.subplot(2, 3, 2)
ax2.plot(eval_times_30, auc_values, marker="o", markersize=4, color='#e67e22')
ax2.set_title(f"Time-dependent AUC (iAUC={mean_auc:.3f})")
ax2.set_xlabel("Time (Months)")
ax2.set_ylabel("AUC")

# Brier score (30 points)
ax3 = plt.subplot(2, 3, 3)
ax3.plot(brier_times, brier_values, marker="o", markersize=4, color='#8e44ad')
ax3.set_title(f"Brier score (IBS={ibs:.3f})")
ax3.set_xlabel("Time (Months)")
ax3.set_ylabel("Brier score")

# Calibration (1, 3, 5, 10 years)
ax4 = plt.subplot(2, 3, 4)
colors = ['#1abc9c', '#f1c40f', '#e67e22', '#e74c3c']
for horizon, color in zip(actual_cal_horizons, colors):
    sub = calibration_df[calibration_df["horizon"] == horizon]
    ax4.plot(sub["mean_predicted"], sub["observed_event_probability"], marker="o", label=f"{int(horizon / 12)} Year",
             color=color)
ax4.plot([0, 1], [0, 1], linestyle="--", color="gray")
ax4.set_title("Calibration Curves")
ax4.set_xlabel("Predicted Probability")
ax4.set_ylabel("Observed Probability")
ax4.legend()

# DCA 5 Year
if 60 in actual_dca_horizons:
    ax5 = plt.subplot(2, 3, 5)
    sub = dca_df_all[dca_df_all["horizon"] == 60]
    ax5.plot(sub["threshold"], sub["net_benefit_model"], label="RSF", color='#2980b9')
    ax5.plot(sub["threshold"], sub["net_benefit_all"], linestyle="--", color='black', label="All")
    ax5.plot(sub["threshold"], sub["net_benefit_none"], linestyle=":", color='gray', label="None")
    ax5.set_title("DCA at 5 Years")
    ax5.set_xlabel("Threshold Probability")
    ax5.set_ylabel("Net Benefit")
    ax5.legend()

# DCA 10 Year
if 120 in actual_dca_horizons:
    ax6 = plt.subplot(2, 3, 6)
    sub = dca_df_all[dca_df_all["horizon"] == 120]
    ax6.plot(sub["threshold"], sub["net_benefit_model"], label="RSF", color='#c0392b')
    ax6.plot(sub["threshold"], sub["net_benefit_all"], linestyle="--", color='black', label="All")
    ax6.plot(sub["threshold"], sub["net_benefit_none"], linestyle=":", color='gray', label="None")
    ax6.set_title("DCA at 10 Years")
    ax6.set_xlabel("Threshold Probability")
    ax6.set_ylabel("Net Benefit")
    ax6.legend()

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "model_evaluation_summary.png"), dpi=600, bbox_inches="tight")
plt.show()

# =========================
# Print summary
# =========================
print("Model evaluation completed successfully.")
print(f"Output directory: {OUTPUT_DIR}")
print(f"Training C-index: {train_cindex:.4f}")
print(f"Testing C-index: {test_cindex:.4f}")
print(f"Integrated AUC (iAUC): {mean_auc:.4f}")
print(f"Integrated Brier Score (IBS): {ibs:.4f}")