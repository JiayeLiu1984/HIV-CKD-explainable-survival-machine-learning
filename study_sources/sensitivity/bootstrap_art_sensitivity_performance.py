from pathlib import Path
import importlib.util
import time

import numpy as np
import pandas as pd


PROJECT = Path(r"__CKD_WORKDIR__")
STEP6 = PROJECT / "rolling_5y_step6_super_landmark_data"
STEP11 = PROJECT / "rolling_5y_step11_v2_four_model_oof_comparison_lstm_v2"
SENS = PROJECT / "comment1_ART_censoring_sensitivity_20260905"
OUT = SENS / "Table_Sensitivity_Model_Performance_with_95CI.csv"
REPS = 1000
SEED = 20260905
PRIMARY_POS = [0, 1, 3, 5]

spec = importlib.util.spec_from_file_location(
    "metric_mod", PROJECT / "comment1_sensitivity_20260903" / "comment1_sensitivity_model_eval.py"
)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

metadata = pd.read_csv(STEP6 / "super_landmark_development_metadata.csv", usecols=["local_patient_index", "landmark_index"])
patient_id = metadata["local_patient_index"].to_numpy(int)
landmark = metadata["landmark_index"].to_numpy(int)
n_patients = int(patient_id.max()) + 1

primary_time = np.load(STEP6 / "development_analysis_time_month.npy").astype(float)
primary_event = np.load(STEP6 / "development_event_within_60m.npy").astype(np.int8)
primary_risk = np.load(STEP11 / "lstm_v2_crossfit_calibrated_oof_risk_long.npy", mmap_mode="r")

sens_time = np.load(SENS / "sensitivity_analysis_time_month.npy").astype(float)
sens_event = np.load(SENS / "sensitivity_event_within_60m.npy").astype(np.int8)
sens_eligible = np.load(SENS / "sensitivity_eligible_long.npy").astype(bool)
sens_risk = np.load(SENS / "sensitivity_crossfit_recalibrated_risk_long.npy", mmap_mode="r")

lm_rows = {lp: np.flatnonzero(landmark == lp) for lp in PRIMARY_POS}


def grouped_ici(time_month, event, pred_5y):
    order = np.argsort(pred_5y, kind="mergesort")
    groups = np.empty(len(pred_5y), dtype=np.int8)
    groups[order] = np.minimum(9, np.arange(len(pred_5y)) * 10 // len(pred_5y))
    value = 0.0
    for g in range(10):
        sel = groups == g
        obs = mod.km_event_risk(time_month[sel], event[sel], 60.0)[0]
        value += sel.mean() * abs(float(pred_5y[sel].mean()) - obs)
    return value


def metric_bundle(time_month, event, risk):
    m = mod.evaluate_landmark(event, time_month, risk)
    ici = grouped_ici(time_month, event, risk[:, -1])
    return np.array([
        m["uno_c_index_5y"], m["integrated_dynamic_auc"], m["integrated_brier"], ici
    ], dtype=float)


# Exact point estimates from the full samples.
point_primary = np.mean([
    metric_bundle(primary_time[lm_rows[lp]], primary_event[lm_rows[lp]], np.asarray(primary_risk[lm_rows[lp]]))
    for lp in PRIMARY_POS
], axis=0)
point_sens = np.mean([
    metric_bundle(sens_time[lm_rows[lp]], sens_event[lm_rows[lp]], np.asarray(sens_risk[lm_rows[lp]]))
    for lp in PRIMARY_POS
], axis=0)

rng = np.random.default_rng(SEED)
boot_primary_ici = np.full(REPS, np.nan)
boot_sens = np.full((REPS, 4), np.nan)
started = time.time()

for b in range(REPS):
    sampled = rng.integers(0, n_patients, size=n_patients)
    mult = np.bincount(sampled, minlength=n_patients)
    sens_lm = []
    primary_ici_lm = []
    try:
        for lp in PRIMARY_POS:
            base_idx = lm_rows[lp]
            reps = mult[patient_id[base_idx]]
            idx = np.repeat(base_idx[reps > 0], reps[reps > 0])
            sens_lm.append(metric_bundle(sens_time[idx], sens_event[idx], np.asarray(sens_risk[idx])))
            primary_ici_lm.append(grouped_ici(primary_time[idx], primary_event[idx], np.asarray(primary_risk[idx, -1])))
        boot_sens[b] = np.mean(sens_lm, axis=0)
        boot_primary_ici[b] = np.mean(primary_ici_lm)
    except Exception:
        pass
    if (b + 1) % 25 == 0:
        print(f"completed {b + 1}/{REPS}; elapsed={time.time() - started:.1f}s", flush=True)

primary_perf = pd.read_csv(STEP11 / "four_model_equal_weight_mean_performance.csv")
primary_perf = primary_perf.loc[
    (primary_perf["model"] == "LSTM-v2")
    & (primary_perf["scope"] == "primary_0_1_3_5y_landmark_equal_weight_mean")
].iloc[0]

def q(v, p):
    v = np.asarray(v); v = v[np.isfinite(v)]
    return float(np.quantile(v, p)), int(v.size)

rows = []
primary_bounds = [
    (primary_perf["mean_uno_c_index_5y_lower_95"], primary_perf["mean_uno_c_index_5y_upper_95"], primary_perf["mean_uno_c_index_5y_valid_bootstrap_n"]),
    (primary_perf["mean_integrated_dynamic_auc_lower_95"], primary_perf["mean_integrated_dynamic_auc_upper_95"], primary_perf["mean_integrated_dynamic_auc_valid_bootstrap_n"]),
    (primary_perf["mean_integrated_brier_lower_95"], primary_perf["mean_integrated_brier_upper_95"], primary_perf["mean_integrated_brier_valid_bootstrap_n"]),
    (*q(boot_primary_ici, .025)[:1],),
]
pici_lo, n_pici = q(boot_primary_ici, .025); pici_hi, _ = q(boot_primary_ici, .975)
rows.append({
    "analysis": "Primary outcome",
    "landmark_years": "0,1,3,5",
    "aggregation": "equal weight across landmarks",
    "C_index": point_primary[0], "C_index_lower_95": primary_perf["mean_uno_c_index_5y_lower_95"], "C_index_upper_95": primary_perf["mean_uno_c_index_5y_upper_95"],
    "iAUC": point_primary[1], "iAUC_lower_95": primary_perf["mean_integrated_dynamic_auc_lower_95"], "iAUC_upper_95": primary_perf["mean_integrated_dynamic_auc_upper_95"],
    "IBS": point_primary[2], "IBS_lower_95": primary_perf["mean_integrated_brier_lower_95"], "IBS_upper_95": primary_perf["mean_integrated_brier_upper_95"],
    "Mean_ICI_5y": point_primary[3], "Mean_ICI_lower_95": pici_lo, "Mean_ICI_upper_95": pici_hi,
    "valid_bootstrap_n": min(1000, n_pici),
})

sens_valid = np.all(np.isfinite(boot_sens), axis=1)
sb = boot_sens[sens_valid]
rows.append({
    "analysis": "ART-censoring sensitivity outcome",
    "landmark_years": "0,1,3,5",
    "aggregation": "equal weight across landmarks",
    "C_index": point_sens[0], "C_index_lower_95": np.quantile(sb[:, 0], .025), "C_index_upper_95": np.quantile(sb[:, 0], .975),
    "iAUC": point_sens[1], "iAUC_lower_95": np.quantile(sb[:, 1], .025), "iAUC_upper_95": np.quantile(sb[:, 1], .975),
    "IBS": point_sens[2], "IBS_lower_95": np.quantile(sb[:, 2], .025), "IBS_upper_95": np.quantile(sb[:, 2], .975),
    "Mean_ICI_5y": point_sens[3], "Mean_ICI_lower_95": np.quantile(sb[:, 3], .025), "Mean_ICI_upper_95": np.quantile(sb[:, 3], .975),
    "valid_bootstrap_n": len(sb),
})

result = pd.DataFrame(rows)
result.to_csv(OUT, index=False, encoding="utf-8-sig")
print(OUT)
print(result.to_string(index=False))
