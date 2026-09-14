from pathlib import Path
import importlib.util
import json

import numpy as np
import pandas as pd


PROJECT = Path(r"__CKD_WORKDIR__")
STEP6 = PROJECT / "rolling_5y_step6_super_landmark_data"
LABELS = PROJECT / "rolling_5y_labels_art_sensitivity"
LSTM = PROJECT / "rolling_5y_step10e_lstm_v2_final_oof_selected_existing_trials"
STEP11 = PROJECT / "rolling_5y_step11_v2_four_model_oof_comparison_lstm_v2"
OUT = PROJECT / "comment1_ART_censoring_sensitivity_20260905"
OUT.mkdir(parents=True, exist_ok=True)

spec = importlib.util.spec_from_file_location(
    "old_eval", PROJECT / "comment1_sensitivity_20260903" / "comment1_sensitivity_model_eval.py"
)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

metadata = pd.read_csv(STEP6 / "super_landmark_development_metadata.csv", encoding="utf-8-sig")
global_idx = metadata["global_patient_index"].to_numpy(int)
landmark_index = metadata["landmark_index"].to_numpy(int)
landmark_month = metadata["landmark_month"].to_numpy(float)
fold_id = metadata["fold_id"].to_numpy(int)

patient_event = np.load(LABELS / "event_sensitivity.npy").astype(np.int8)
patient_time = np.load(LABELS / "observed_time_month.npy").astype(float)
event_patient_long = patient_event[global_idx]
observed_time_long = patient_time[global_idx]
remaining = observed_time_long - landmark_month
eligible = remaining > mod.TIME_TOLERANCE
event_in_60 = eligible & event_patient_long.astype(bool) & (remaining <= 60.0 + mod.TIME_TOLERANCE)
analysis_time = np.full(len(metadata), np.nan, dtype=np.float32)
analysis_time[eligible] = np.minimum(remaining[eligible], 60.0).astype(np.float32)
event_within = event_in_60.astype(np.int8)

target = np.zeros((len(metadata), 10), dtype=np.float32)
mask = np.zeros((len(metadata), 10), dtype=bool)
event_interval = np.full(len(metadata), -1, dtype=int)
event_interval[event_in_60] = (
    np.ceil((remaining[event_in_60] - mod.TIME_TOLERANCE) / 6.0).astype(int) - 1
)
event_interval[event_in_60] = np.clip(event_interval[event_in_60], 0, 9)
for j, interval_end in enumerate(mod.FUTURE_END_MONTHS):
    mask[:, j] = eligible & (
        (event_in_60 & (event_interval >= j))
        | (~event_in_60 & (remaining >= interval_end - mod.TIME_TOLERANCE))
    )
    target[event_in_60 & (event_interval == j), j] = 1.0

raw_hazard = np.load(LSTM / "lstm_v2_oof_hazard_long.npy").astype(float)
frozen_risk = np.load(STEP11 / "lstm_v2_crossfit_calibrated_oof_risk_long.npy").astype(float)
if raw_hazard.shape != target.shape:
    raise ValueError("Raw predictions and ART-sensitivity labels are misaligned")

recalibrated_hazard = np.full_like(raw_hazard, np.nan, dtype=float)
parameter_rows = []
for lp in range(6):
    for heldout in range(5):
        train = eligible & (landmark_index == lp) & (fold_id != heldout)
        valid = eligible & (landmark_index == lp) & (fold_id == heldout)
        pars = mod.fit_interval_hazard_calibrator(raw_hazard[train], target[train], mask[train])
        recalibrated_hazard[valid] = mod.apply_interval_hazard_calibrator(raw_hazard[valid], pars)
        for interval_index, alpha in enumerate(pars["alpha"]):
            parameter_rows.append({
                "landmark_position": lp,
                "landmark_month": float(mod.LANDMARK_MONTHS[lp]),
                "heldout_fold": heldout,
                "interval_index": interval_index,
                "interval_end_month": float(mod.FUTURE_END_MONTHS[interval_index]),
                "alpha": float(alpha),
                "beta": float(pars["beta"]),
                "valid_interval_n": int(pars["valid_interval_n"]),
                "event_interval_n": int(pars["event_interval_n"]),
            })
if np.isnan(recalibrated_hazard[eligible]).any():
    raise ValueError("Cross-fitted recalibration is incomplete")
recalibrated_risk = np.full_like(recalibrated_hazard, np.nan, dtype=float)
recalibrated_risk[eligible] = mod.hazard_to_risk(recalibrated_hazard[eligible])

stage_results = []
for stage, risk in [
    ("ART_sensitivity_frozen_primary_calibration", frozen_risk),
    ("ART_sensitivity_crossfit_recalibrated", recalibrated_risk),
]:
    stage_results.append((stage, mod.evaluate_stage(
        stage, eligible, landmark_index, event_within, analysis_time, risk
    )))

landmark_metrics = pd.concat([x[1][0] for x in stage_results], ignore_index=True)
horizon_metrics = pd.concat([x[1][1] for x in stage_results], ignore_index=True)
overall_calibration = pd.concat([x[1][2] for x in stage_results], ignore_index=True)
decile_calibration = pd.concat([x[1][3] for x in stage_results], ignore_index=True)
sensitivity_means = pd.concat([x[1][4] for x in stage_results], ignore_index=True)

primary_all = pd.read_csv(STEP11 / "four_model_equal_weight_mean_performance.csv")
primary = primary_all.loc[
    (primary_all["model"] == "LSTM-v2")
    & (primary_all["scope"] == "primary_0_1_3_5y_landmark_equal_weight_mean"),
    ["mean_uno_c_index_5y", "mean_integrated_dynamic_auc", "mean_integrated_brier"],
].copy()
primary.insert(0, "stage", "primary_outcome_crossfit_calibrated")
primary.insert(1, "scope", "primary_0_1_3_5y_landmark_equal_weight_mean")
comparison = pd.concat([primary, sensitivity_means], ignore_index=True)
base = comparison.iloc[0]
for col in ["mean_uno_c_index_5y", "mean_integrated_dynamic_auc", "mean_integrated_brier"]:
    comparison[f"difference_vs_primary_{col}"] = comparison[col] - float(base[col])

np.save(OUT / "sensitivity_event_within_60m.npy", event_within)
np.save(OUT / "sensitivity_analysis_time_month.npy", analysis_time)
np.save(OUT / "sensitivity_eligible_long.npy", eligible)
np.save(OUT / "sensitivity_crossfit_recalibrated_risk_long.npy", recalibrated_risk)
pd.DataFrame(parameter_rows).to_csv(OUT / "sensitivity_crossfit_calibration_parameters.csv", index=False, encoding="utf-8-sig")
comparison.to_csv(OUT / "performance_comparison.csv", index=False, encoding="utf-8-sig")
landmark_metrics.to_csv(OUT / "landmark_performance_comparison.csv", index=False, encoding="utf-8-sig")
horizon_metrics.to_csv(OUT / "sensitivity_horizon_metrics.csv", index=False, encoding="utf-8-sig")
overall_calibration.to_csv(OUT / "sensitivity_calibration_overall.csv", index=False, encoding="utf-8-sig")
decile_calibration.to_csv(OUT / "sensitivity_calibration_deciles.csv", index=False, encoding="utf-8-sig")

qc = {
    "outcome_definition": "eGFR-based CKD during relevant ART exposure censored at original CKD time; other outcomes retained",
    "original_events_all_patients": 2491,
    "sensitivity_events_all_patients": int(patient_event.sum()),
    "censored_potential_ART_CKD_all_patients": int(2491 - patient_event.sum()),
    "development_origins": int(len(metadata)),
    "eligible_development_origins": int(eligible.sum()),
    "sensitivity_future_5y_events_across_origins": int(event_within[eligible].sum()),
    "same_predictions_and_features": True,
    "calibration": "five-fold cross-fitted interval-hazard recalibration under sensitivity outcome",
}
(OUT / "model_sensitivity_qc.json").write_text(json.dumps(qc, indent=2), encoding="utf-8")
print(comparison.to_string(index=False))
print(json.dumps(qc, indent=2))
