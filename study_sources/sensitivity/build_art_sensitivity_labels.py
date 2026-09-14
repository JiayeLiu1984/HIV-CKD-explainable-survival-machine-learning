from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT = Path(r"__CKD_WORKDIR__")
TENSOR = PROJECT / "tensor_outputs_fixed_0_10y"
ORIG_LABEL = PROJECT / "rolling_5y_labels_origins_0_5y"
OUT = PROJECT / "rolling_5y_labels_art_sensitivity"
QC = Path(r"__CKD_LOCAL_INPUT__/Documents/深度学习/outputs/comment1-art-current-exposure-20260904/Patient_level_ART_CKD_QC.csv")
OUT.mkdir(parents=True, exist_ok=True)

patient_info = pd.read_csv(TENSOR / "patient_info.csv", encoding="utf-8-sig", dtype={"ID": "string"})
event_original = np.load(TENSOR / "event.npy").astype(np.int8)
observed_time = np.load(TENSOR / "observed_time_month.npy").astype(np.float64)
valid_mask = np.load(TENSOR / "valid_mask.npy").astype(bool)
all_origins = np.load(TENSOR / "origin_months.npy").astype(float)
split = np.load(TENSOR / "split_indices.npz")
development_idx = split["development_idx"].astype(np.int64)
test_idx = split["test_idx"].astype(np.int64)

qc = pd.read_csv(QC, encoding="utf-8-sig", dtype={"ID": "string"})
if qc["ID"].duplicated().any() or len(qc) != len(patient_info):
    raise ValueError("Patient QC must contain one unique row per model patient")
aligned = patient_info[["ID"]].merge(
    qc[["ID", "CKDstatus_original", "CKDstatus_sensitivity", "potential_ART_CKD"]],
    on="ID", how="left", validate="one_to_one",
)
if aligned["CKDstatus_sensitivity"].isna().any():
    raise ValueError("Patient IDs are not fully aligned")
if not np.array_equal(aligned["CKDstatus_original"].to_numpy(dtype=np.int8), event_original):
    raise ValueError("Original outcomes in patient QC do not match frozen model arrays")
event = aligned["CKDstatus_sensitivity"].to_numpy(dtype=np.int8)
if event.sum() != 2022 or (event > event_original).any():
    raise ValueError("Sensitivity event count/monotonicity failed")

selected_origin_months = np.asarray([0, 12, 24, 36, 48, 60], dtype=float)
selected_origin_indices = np.asarray([
    int(np.where(np.isclose(all_origins, month))[0][0]) for month in selected_origin_months
], dtype=np.int64)
selected_valid = valid_mask[:, selected_origin_indices]
relative_start = np.arange(10, dtype=float) * 6.0
relative_end = np.arange(1, 11, dtype=float) * 6.0
n = len(event)
rolling_event = np.zeros((n, 6, 10), dtype=np.float32)
at_risk = np.zeros((n, 6, 10), dtype=bool)
event_bin = np.full((n, 6), -1, dtype=np.int16)
eps = 1e-8

for op, origin in enumerate(selected_origin_months):
    origin_valid = selected_valid[:, op] & (observed_time > origin + eps)
    residual = observed_time - origin
    within = origin_valid & (event == 1) & (residual <= 60.0 + eps)
    ids = np.where(within)[0]
    if len(ids):
        bins = np.ceil((residual[ids] - eps) / 6.0).astype(np.int64) - 1
        event_bin[ids, op] = np.clip(bins, 0, 9)
    for j in range(10):
        event_observed = origin_valid & (event == 1) & (residual > relative_start[j] + eps)
        censor_observed = origin_valid & (event == 0) & (residual >= relative_end[j] - eps)
        at_risk[:, op, j] = event_observed | censor_observed
        rolling_event[within & (event_bin[:, op] == j), op, j] = 1.0

prediction_origin = at_risk.any(axis=2)
valid_origin_count = prediction_origin.sum(axis=1).astype(np.int16)
valid_interval_count = at_risk.reshape(n, -1).sum(axis=1).astype(np.int32)
effective = valid_origin_count > 0

if np.any((rolling_event == 1) & ~at_risk):
    raise ValueError("Positive event label outside at-risk mask")
if (rolling_event.sum(axis=2) > 1).any():
    raise ValueError("Multiple event intervals at a landmark")
if np.any(prediction_origin & ~selected_valid):
    raise ValueError("Prediction origin without valid history")

np.save(OUT / "event_sensitivity.npy", event)
np.save(OUT / "observed_time_month.npy", observed_time.astype(np.float32))
np.save(OUT / "rolling_event_matrix.npy", rolling_event)
np.save(OUT / "rolling_at_risk_mask.npy", at_risk)
np.save(OUT / "rolling_prediction_origin_mask.npy", prediction_origin)
np.save(OUT / "selected_origin_indices.npy", selected_origin_indices)
np.save(OUT / "selected_origin_months.npy", selected_origin_months)
np.save(OUT / "relative_start_months.npy", relative_start)
np.save(OUT / "relative_end_months.npy", relative_end)
np.save(OUT / "event_bin_relative.npy", event_bin)
np.save(OUT / "valid_origin_count.npy", valid_origin_count)
np.save(OUT / "valid_interval_label_count.npy", valid_interval_count)
np.save(OUT / "effective_patient_mask.npy", effective)
shutil.copyfile(TENSOR / "feature_names.csv", OUT / "feature_names.csv")

summary = {
    "total_patients": n,
    "original_events": int(event_original.sum()),
    "sensitivity_events": int(event.sum()),
    "censored_potential_ART_CKD": int(aligned["potential_ART_CKD"].sum()),
    "development_patients": int(len(development_idx)),
    "test_patients": int(len(test_idx)),
    "effective_development_patients": int(effective[development_idx].sum()),
    "effective_test_patients": int(effective[test_idx].sum()),
    "positive_rolling_event_labels": int(rolling_event.sum()),
    "valid_interval_labels": int(at_risk.sum()),
    "same_X_and_preprocessing": True,
    "same_split": True,
    "observed_time_unchanged": bool(np.array_equal(observed_time.astype(np.float32), np.load(TENSOR / "observed_time_month.npy").astype(np.float32))),
}
(OUT / "label_build_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(summary, ensure_ascii=False, indent=2))
