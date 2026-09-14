#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Step 12J: audit representative-case selection and draw 6-landmark IG heatmaps."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_DIR = Path(os.getenv("CKD_LSTM_PROJECT_DIR", "__CKD_WORKDIR__"))
INTERPRETATION_ROOT = Path(
    os.getenv(
        "CKD_MODEL_INTERPRETATION_ROOT",
        str(PROJECT_DIR / "rolling_5y_model_interpretation_final_6landmarks"),
    )
)
COMMON_FIGURE_DIR = INTERPRETATION_ROOT / "FIGURES_ALL"
OUTPUT_DIR = INTERPRETATION_ROOT / "08_PATIENT_CASE_AUDIT"
STEP12I_DIR = INTERPRETATION_ROOT / "07_PATIENT_DYNAMIC"
for directory in [INTERPRETATION_ROOT, COMMON_FIGURE_DIR, OUTPUT_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

SELECTION_FILE = STEP12I_DIR / "patient_case_selection.csv"
CANDIDATE_FILE = STEP12I_DIR / "patient_candidate_trajectory_summary.csv"
RISK_FILE = STEP12I_DIR / "patient_selected_risk_trajectory.csv"
IG_FILE = STEP12I_DIR / "ig_sample_level_clinical_attribution.csv"
AUDIT_FILE = OUTPUT_DIR / "patient_case_selection_audit.csv"
HEATMAP_SOURCE_FILE = OUTPUT_DIR / "patient_case_signed_ig_heatmap_source.csv"
SUMMARY_FILE = OUTPUT_DIR / "step12j_patient_case_audit_summary.json"
HEATMAP_PNG = COMMON_FIGURE_DIR / "Step12J_Patient_Case_Signed_IG_Heatmap_6Landmarks.png"
HEATMAP_PDF = COMMON_FIGURE_DIR / "Step12J_Patient_Case_Signed_IG_Heatmap_6Landmarks.pdf"
EXPECTED_LANDMARKS = np.asarray([0, 1, 2, 3, 4, 5], dtype=float)
SELECTION_RISK_COLUMNS = [
    "risk_0m",
    "risk_12m",
    "risk_24m",
    "risk_36m",
    "risk_48m",
    "risk_60m",
]
TOP_VARIABLE_N = int(os.getenv("CKD_PATIENT_HEATMAP_TOP_N", "12"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_files(paths: list[Path]) -> None:
    missing = [str(path) for path in paths if not path.is_file() or path.stat().st_size == 0]
    if missing:
        raise FileNotFoundError("Missing Step12I outputs:\n" + "\n".join(missing))


def require_columns(frame: pd.DataFrame, columns: set[str], label: str) -> None:
    missing = sorted(columns.difference(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing columns: {missing}")


def configure_plot_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 9,
        }
    )


def main() -> None:
    require_files([SELECTION_FILE, CANDIDATE_FILE, RISK_FILE, IG_FILE])
    selection = pd.read_csv(SELECTION_FILE, encoding="utf-8-sig")
    candidates = pd.read_csv(CANDIDATE_FILE, encoding="utf-8-sig")
    risk = pd.read_csv(RISK_FILE, encoding="utf-8-sig")
    ig = pd.read_csv(IG_FILE, encoding="utf-8-sig")

    require_columns(
        selection,
        {"case_label", "patient_local", *SELECTION_RISK_COLUMNS},
        "patient_case_selection",
    )
    require_columns(
        risk,
        {"case_label", "patient_local", "landmark_year", "predicted_5y_risk"},
        "patient_selected_risk_trajectory",
    )
    require_columns(
        ig,
        {"patient_local", "landmark_year", "clinical_variable", "signed_ig", "absolute_ig"},
        "ig_sample_level_clinical_attribution",
    )

    for frame in [risk, ig]:
        frame["landmark_year"] = pd.to_numeric(frame["landmark_year"], errors="raise")
    for column in SELECTION_RISK_COLUMNS:
        selection[column] = pd.to_numeric(selection[column], errors="raise")
    for column in ["signed_ig", "absolute_ig"]:
        ig[column] = pd.to_numeric(ig[column], errors="raise")

    selection_unique = selection[["case_label", "patient_local"]].drop_duplicates()
    if len(selection_unique) != 3 or selection_unique["patient_local"].nunique() != 3:
        raise ValueError("Step12I must select exactly three distinct representative patients.")

    candidate_ids = set(pd.to_numeric(candidates["patient_local"], errors="raise").astype(int))
    audit_rows: list[dict[str, object]] = []
    for row in selection_unique.sort_values("case_label").itertuples(index=False):
        case_label = str(row.case_label)
        patient_local = int(row.patient_local)
        selected_rows = selection.loc[
            selection["case_label"].eq(case_label)
            & pd.to_numeric(selection["patient_local"], errors="raise").eq(patient_local)
        ]
        risk_rows = risk.loc[
            risk["case_label"].eq(case_label)
            & pd.to_numeric(risk["patient_local"], errors="raise").eq(patient_local)
        ]
        risk_landmarks = np.sort(risk_rows["landmark_year"].unique())
        audit_rows.append(
            {
                "case_label": case_label,
                "patient_local": patient_local,
                "candidate_pool_contains_patient": patient_local in candidate_ids,
                "selection_patient_is_constant_across_landmarks": (
                    selected_rows["patient_local"].nunique() == 1
                    and risk_rows["patient_local"].nunique() == 1
                    and int(risk_rows["patient_local"].iloc[0]) == patient_local
                ),
                "selection_has_all_6_landmarks": bool(
                    len(selected_rows) == 1
                    and selected_rows[SELECTION_RISK_COLUMNS].notna().all(axis=None)
                ),
                "risk_trajectory_has_all_6_landmarks": np.array_equal(risk_landmarks, EXPECTED_LANDMARKS),
                "selection_row_n": int(len(selected_rows)),
                "risk_row_n": int(len(risk_rows)),
                "risk_min": float(risk_rows["predicted_5y_risk"].min()),
                "risk_max": float(risk_rows["predicted_5y_risk"].max()),
                "risk_first_to_last_change": float(
                    risk_rows.sort_values("landmark_year")["predicted_5y_risk"].iloc[-1]
                    - risk_rows.sort_values("landmark_year")["predicted_5y_risk"].iloc[0]
                ),
                "selection_uses_outcome": False,
                "selection_uses_attribution": False,
                "selection_uses_clinical_values": False,
            }
        )

    audit = pd.DataFrame(audit_rows)
    bool_columns = [
        "candidate_pool_contains_patient",
        "selection_patient_is_constant_across_landmarks",
        "selection_has_all_6_landmarks",
        "risk_trajectory_has_all_6_landmarks",
    ]
    if not audit[bool_columns].all(axis=None):
        raise ValueError("Representative-case audit failed; inspect the audit table.")
    audit.to_csv(AUDIT_FILE, index=False, encoding="utf-8-sig")

    patient_to_case = dict(zip(selection_unique["patient_local"].astype(int), selection_unique["case_label"]))
    ig = ig.loc[ig["patient_local"].astype(int).isin(patient_to_case)].copy()
    ig["case_label"] = ig["patient_local"].astype(int).map(patient_to_case)
    ig = ig.loc[ig["landmark_year"].isin(EXPECTED_LANDMARKS)].copy()

    heatmap_rows: list[pd.DataFrame] = []
    configure_plot_style()
    fig, axes = plt.subplots(1, 3, figsize=(18, 9), constrained_layout=True)
    image = None
    for axis, case_label in zip(axes, sorted(selection_unique["case_label"].astype(str))):
        case = ig.loc[ig["case_label"].eq(case_label)].copy()
        variable_order = (
            case.groupby("clinical_variable", as_index=False)["absolute_ig"]
            .sum()
            .sort_values(["absolute_ig", "clinical_variable"], ascending=[False, True])
            .head(TOP_VARIABLE_N)["clinical_variable"]
            .tolist()
        )
        signed = (
            case.loc[case["clinical_variable"].isin(variable_order)]
            .groupby(["clinical_variable", "landmark_year"], as_index=False)["signed_ig"]
            .sum()
        )
        signed["case_label"] = case_label
        heatmap_rows.append(signed)
        matrix = (
            signed.pivot(index="clinical_variable", columns="landmark_year", values="signed_ig")
            .reindex(index=variable_order, columns=EXPECTED_LANDMARKS)
            .to_numpy(dtype=float)
        )
        vmax = float(np.nanmax(np.abs(matrix))) if np.isfinite(matrix).any() else 1.0
        if vmax <= 0:
            vmax = 1.0
        image = axis.imshow(matrix, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
        axis.set_xticks(range(len(EXPECTED_LANDMARKS)), [f"{year:g} y" for year in EXPECTED_LANDMARKS])
        axis.set_yticks(range(len(variable_order)), variable_order)
        axis.set_xlabel("Prediction landmark after ART initiation")
        axis.set_title(case_label, loc="left", fontweight="bold")
        axis.set_xticks(np.arange(-0.5, len(EXPECTED_LANDMARKS), 1), minor=True)
        axis.set_yticks(np.arange(-0.5, len(variable_order), 1), minor=True)
        axis.grid(which="minor", color="white", linewidth=0.6)
        axis.tick_params(which="minor", bottom=False, left=False)
        fig.colorbar(image, ax=axis, fraction=0.035, pad=0.02, label="Summed signed IG")

    fig.suptitle(
        "Representative-patient signed Integrated Gradients across all 6 formal landmarks",
        fontsize=13,
        fontweight="bold",
    )
    fig.savefig(HEATMAP_PNG, dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(HEATMAP_PDF, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    heatmap_source = pd.concat(heatmap_rows, ignore_index=True)
    heatmap_source.to_csv(HEATMAP_SOURCE_FILE, index=False, encoding="utf-8-sig")

    summary = {
        "stage": "Step12J_Patient_Case_Selection_Audit_and_Heatmap",
        "case_n": 3,
        "landmarks_years": EXPECTED_LANDMARKS.tolist(),
        "top_variables_per_case": TOP_VARIABLE_N,
        "selection_uses_outcome": False,
        "selection_uses_attribution": False,
        "selection_uses_clinical_values": False,
        "audit_passed": True,
        "locked_test_used": False,
        "inputs": {str(path): sha256_file(path) for path in [SELECTION_FILE, CANDIDATE_FILE, RISK_FILE, IG_FILE]},
        "outputs": [str(AUDIT_FILE), str(HEATMAP_SOURCE_FILE), str(HEATMAP_PNG), str(HEATMAP_PDF)],
    }
    SUMMARY_FILE.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(audit.to_string(index=False))
    print("Step12J completed. Output:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
