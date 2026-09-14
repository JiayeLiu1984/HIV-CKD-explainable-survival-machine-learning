# ============================================================
# Figure 3 FINAL — CURRENT-STEP + EARLIER-HISTORY explanation
# Same selected-case visualization, aligned with the corrected Figure 2 definition
#
# Key design:
#   Panel A: ONLY continuous / trajectory-like variables
#            (no comorbidity status, no current ART one-hot, no cumulative ART)
#   Panel B: complete year-5 force decomposition using
#            CURRENT-STEP clinical IG + one residual term named
#            'Earlier longitudinal history' + direct static inputs.
#            This aligns the attribution time scope with displayed current values.
#
# Patient selection is automatic and uses only frozen model outputs/inputs:
#   - moderate-high but non-extreme year-5 CKD risk
#   - eGFR, HIV RNA and CD4 are model-relevant and sufficiently observed
#   - at least one major comorbidity is present at year 5
#   - at least two lipid variables are available
#   - one active ART regimen and its paired cumulative exposure are available
#   - continuous trajectories are reasonably complete/smooth
#   - excessive ART switching and abrupt risk jumps are penalized
#
# NO retraining
# NO IG recomputation
# ============================================================

from pathlib import Path
import json
import math
import textwrap
import warnings

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon, Patch, Ellipse, ConnectionPatch


# ============================================================
# 1. Paths
# ============================================================

PROJECT_DIR = Path("__CKD_WORKDIR__")

STEP1_DIR = PROJECT_DIR / "rolling_5y_step1_new_split"
STEP2_DIR = PROJECT_DIR / "rolling_5y_step2_folds"
STEP3_DIR = PROJECT_DIR / "rolling_5y_step3_raw_features"
STEP4_DIR = PROJECT_DIR / "rolling_5y_step4_preprocessed"
STEP6_DIR = PROJECT_DIR / "rolling_5y_step6_super_landmark_data"
STEP11_DIR = PROJECT_DIR / "rolling_5y_step11_v2_four_model_oof_comparison_lstm_v2"

V5_IG_DIR = (
    PROJECT_DIR
    / "rolling_5y_step12a_ig_oof_n2000_steps32_calibrated_v5_6landmark_formal_ig_explanation"
)
V4_IG_DIR = (
    PROJECT_DIR
    / "rolling_5y_step12a_ig_oof_n2000_steps32_calibrated_v4_formal_ig_explanation"
)

FORMAL_IG_CANDIDATES = [
    PROJECT_DIR
    / "rolling_5y_model_interpretation_final_6landmarks"
    / "01_GLOBAL_IG"
    / "ig_sample_level_clinical_attribution.csv",
    V5_IG_DIR
    / "ig_sample_level_clinical_attribution.csv",
    V4_IG_DIR
    / "ig_sample_level_clinical_attribution.csv",
]

IG_CHECKPOINT_DIR_CANDIDATES = [
    PROJECT_DIR
    / "rolling_5y_model_interpretation_final_6landmarks"
    / "01_GLOBAL_IG"
    / "ig_checkpoints",
    V5_IG_DIR / "ig_checkpoints",
    V4_IG_DIR / "ig_checkpoints",
]

RISK_FILE = STEP11_DIR / "lstm_v2_crossfit_calibrated_oof_risk_long.npy"
LONG_MAP_FILE = STEP6_DIR / "development_long_row_index_map.npy"

OUTPUT_DIR = (
    PROJECT_DIR
    / "rolling_5y_model_interpretation_final_6landmarks"
    / "FIGURE3_CURRENT_STEP_HISTORY_FINAL"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PNG = OUTPUT_DIR / "Figure3_CurrentStep_History_FINAL.png"
OUTPUT_PDF = OUTPUT_DIR / "Figure3_CurrentStep_History_FINAL.pdf"
OUTPUT_SOURCE = OUTPUT_DIR / "Figure3_CurrentStep_History_FINAL_source.csv"
OUTPUT_CANDIDATE_TABLE = OUTPUT_DIR / "Figure3_CurrentStep_History_FINAL_candidate_ranking.csv"
OUTPUT_A_PROFILE = OUTPUT_DIR / "Figure3_CurrentStep_History_FINAL_PanelA_variables.csv"
OUTPUT_B_PROFILE = OUTPUT_DIR / "Figure3_CurrentStep_History_FINAL_PanelB_variables.csv"

# Previous Figure-3 ranking is used only to exclude the patient already shown in v10.
PREVIOUS_V10_CANDIDATE_TABLE = (
    PROJECT_DIR
    / "rolling_5y_model_interpretation_final_6landmarks"
    / "FIGURE3_REFERENCE_STYLE_FINAL_V10_BALANCED_ONEROW_LABELS"
    / "Figure3_ReferenceStyle_FINAL_v10_BalancedOneRowLabels_candidate_ranking.csv"
)

# Also exclude the patient selected by the immediately preceding high-risk v11 run.
PREVIOUS_V11_CANDIDATE_TABLE = (
    PROJECT_DIR
    / "rolling_5y_model_interpretation_final_6landmarks"
    / "FIGURE3_REFERENCE_STYLE_FINAL_V11_ALTERNATIVE_HIGH_RISK"
    / "Figure3_ReferenceStyle_FINAL_v11_AlternativeHighRiskPatient_candidate_ranking.csv"
)

# Exclude the patient selected by v12 as well, because v13 intentionally searches
# for a different case with richer longitudinal dynamics.
PREVIOUS_V12_CANDIDATE_TABLE = (
    PROJECT_DIR
    / "rolling_5y_model_interpretation_final_6landmarks"
    / "FIGURE3_REFERENCE_STYLE_FINAL_V12_HIGH_RISK_LOWER_EGFR_IMPACT"
    / "Figure3_ReferenceStyle_FINAL_v12_HighRisk_LowerEGFRImpact_candidate_ranking.csv"
)

# Exclude the v13 patient as well. v15 intentionally searches for a new case
# with sustained dynamics in both the early and late follow-up periods and a
# more balanced multi-feature local explanation.
PREVIOUS_V13_CANDIDATE_TABLE = (
    PROJECT_DIR
    / "rolling_5y_model_interpretation_final_6landmarks"
    / "FIGURE3_REFERENCE_STYLE_FINAL_V13_HIGH_RISK_DYNAMIC_BALANCED"
    / "Figure3_ReferenceStyle_FINAL_v13_HighRisk_DynamicBalanced_candidate_ranking.csv"
)

# The v14 patient is retained as a benchmark for comparison, but is excluded
# from the new v15 selection so that this script actively searches for a
# different patient with a more consistent late risk trajectory.
PREVIOUS_V14_CANDIDATE_TABLE = (
    PROJECT_DIR
    / "rolling_5y_model_interpretation_final_6landmarks"
    / "FIGURE3_REFERENCE_STYLE_FINAL_V14_HIGH_RISK_MULTIFACTOR_SUSTAINED_DYNAMIC"
    / "Figure3_ReferenceStyle_FINAL_v14_HighRisk_Multifactor_SustainedDynamic_candidate_ranking.csv"
)

# Exclude the v15 case as well. v16 intentionally searches for another patient
# with a more gradual risk increase and fewer long plateau-like trajectories.
PREVIOUS_V15_CANDIDATE_TABLE = (
    PROJECT_DIR
    / "rolling_5y_model_interpretation_final_6landmarks"
    / "FIGURE3_REFERENCE_STYLE_FINAL_V15_HIGH_RISK_LATE_RISK_CONSISTENT"
    / "Figure3_ReferenceStyle_FINAL_v15_HighRisk_LateRiskConsistent_candidate_ranking.csv"
)

# Optional manual exclusions. Add patient_local IDs here if you want to avoid
# additional previously inspected cases, e.g. a locally selected synthetic case set.
MANUAL_EXCLUDE_PATIENT_LOCALS = set()


# ============================================================
# 2. Configuration
# ============================================================

LANDMARK_MONTHS = np.array([0, 12, 24, 36, 48, 60], dtype=int)
LANDMARK_YEARS = LANDMARK_MONTHS / 12.0

HISTORY_MONTHS = np.arange(0, 61, 6, dtype=int)
HISTORY_YEARS = HISTORY_MONTHS / 12.0

TARGET_LANDMARK_MONTH = 60
TOP_A_N = 10
TOP_B_LABEL_N = 8

# Figure-3 Panel B now follows the same temporal definition as corrected Figure 2.
# Dynamic feature labels refer to CURRENT-STEP contributions. All dynamic
# contributions before the current step are retained as one net force term so
# the force decomposition still reconstructs f(x).
EARLIER_HISTORY_FEATURE = "__earlier_longitudinal_history__"
EARLIER_HISTORY_DISPLAY = "Earlier longitudinal history"
PANEL_B_RECONSTRUCTION_TOL_PP = 0.08
PANEL_B_DYNAMIC_RECONCILIATION_TOL_PP = 0.15

# Panel B final display strategy:
# - draw ALL model features in the force plot;
# - label exactly 10 major force-bar segments;
# - DO NOT exclude ART, medication, Course, demographic, or otherwise
#   "difficult-to-explain" variables from consideration;
# - guarantee that the TWO largest positive/red contributors and the SINGLE
#   largest negative/blue contributor are labelled;
# - fill the remaining label slots strictly by overall |IG| magnitude.
#
# This avoids cherry-picking and ensures that visually dominant force segments
# are always identified, even when their clinical interpretation is complex.
PANEL_B_TOP_LABEL_N = 10
PANEL_B_REQUIRED_TOP_POSITIVE_N = 2
PANEL_B_REQUIRED_TOP_NEGATIVE_N = 1
PANEL_B_UNLABELLED_ALPHA = 0.92
PANEL_B_LABELLED_ALPHA = 0.98
PANEL_B_TINY_SEGMENT_FRAC = 0.0045
DISPLAY_FX_X = 75.0
DISPLAY_CONNECTOR_X = 77.0
PANEL_B_SEPARATOR_FRAC = 0.010
PANEL_B_SEPARATOR_MIN = 0.40
FINAL_SHORTLIST_N = 20

# Panel A must resemble the reference: 10 genuinely continuous, displayable trajectories.
# Selection is based on fold-standardized model values, not within-patient z-scoring.
A_MIN_OBS_ANCHOR = {"eGFR": 3, "HIVRNA_log10": 2, "CD4": 3}
A_MIN_OBS_OTHER = 4
A_MIN_OBSERVED_Z_RANGE_OTHER = 0.90
A_MIN_ALL_Z_RANGE_OTHER = 1.00
A_MAX_ABS_Z_SOFT = 4.5

# v16: sustained longitudinal dynamics + plateau control.
# A good Figure-3 patient must have genuine movement over the full 0-5 year
# history, including the later 2.5-5 year period. This prevents selection of a
# patient whose trajectories move only early and then become long LOCF plateaus.
A_DYNAMIC_RANGE_MIN = 1.00
A_DYNAMIC_SD_MIN = 0.38
A_DYNAMIC_CHANGE_MIN = 3
A_HALF_DYNAMIC_RANGE_MIN = 0.55
A_HALF_DYNAMIC_CHANGE_MIN = 1

A_STRICT_MEDIAN_RANGE_MIN = 1.15
A_STRICT_MEDIAN_SD_MIN = 0.48
A_STRICT_MIN_DYNAMIC_VARS = 6
A_STRICT_MIN_EARLY_DYNAMIC_VARS = 5
A_STRICT_MIN_LATE_DYNAMIC_VARS = 5
A_STRICT_MIN_SUSTAINED_DYNAMIC_VARS = 4
A_STRICT_MEAN_OBS_FRACTION_MIN = 0.38
A_STRICT_MAX_ABS_Z = 4.5

A_RELAXED_MEDIAN_RANGE_MIN = 0.95
A_RELAXED_MEDIAN_SD_MIN = 0.38
A_RELAXED_MIN_DYNAMIC_VARS = 5
A_RELAXED_MIN_EARLY_DYNAMIC_VARS = 4
A_RELAXED_MIN_LATE_DYNAMIC_VARS = 4
A_RELAXED_MIN_SUSTAINED_DYNAMIC_VARS = 3
A_RELAXED_MEAN_OBS_FRACTION_MIN = 0.30
A_RELAXED_MAX_ABS_Z = 5.5

# v16 plateau-control objective.
# A "plateau" step is a 6-month change smaller than this value on the frozen
# fold-standardized model-input scale. We do not demand constant movement at
# every visit, but we explicitly avoid cases dominated by long LOCF/flat runs.
A_PLATEAU_STEP_EPS_Z = 0.12
A_LOW_PLATEAU_FRACTION_MAX = 0.60
A_STRICT_MEDIAN_PLATEAU_FRACTION_MAX = 0.55
A_STRICT_MEAN_PLATEAU_FRACTION_MAX = 0.58
A_STRICT_MIN_LOW_PLATEAU_VARS = 6
A_STRICT_MAX_SINGLE_PLATEAU_FRACTION = 0.82
A_RELAXED_MEDIAN_PLATEAU_FRACTION_MAX = 0.65
A_RELAXED_MEAN_PLATEAU_FRACTION_MAX = 0.68
A_RELAXED_MIN_LOW_PLATEAU_VARS = 5
A_RELAXED_MAX_SINGLE_PLATEAU_FRACTION = 0.90

# v16 gradual-risk objective.
# The example should remain high risk, but the risk curve should rise through
# multiple landmarks rather than being created by one very large terminal jump.
RISK_STRICT_MIN_Y5_PCT = 30.0
RISK_MODERATE_MIN_Y5_PCT = 25.0
RISK_FALLBACK_MIN_Y5_PCT = 22.0
RISK_STRICT_MAX_ANNUAL_JUMP_PP = 18.0
RISK_MODERATE_MAX_ANNUAL_JUMP_PP = 23.0
RISK_FALLBACK_MAX_ANNUAL_JUMP_PP = 30.0
RISK_STRICT_MAX_TERMINAL_JUMP_PP = 15.0
RISK_MODERATE_MAX_TERMINAL_JUMP_PP = 20.0
RISK_FALLBACK_MAX_TERMINAL_JUMP_PP = 27.0
RISK_STRICT_MIN_UPWARD_STEPS = 3
RISK_MODERATE_MIN_UPWARD_STEPS = 3
RISK_FALLBACK_MIN_UPWARD_STEPS = 2
RISK_STRICT_MIN_TOTAL_GAIN_PP = 15.0
RISK_MODERATE_MIN_TOTAL_GAIN_PP = 10.0
RISK_FALLBACK_MIN_TOTAL_GAIN_PP = 5.0

# High-risk alternative-patient search.
# Percentiles are calculated against the FULL development cohort, not the IG sample.
# Primary search: 90th-99.5th percentile, targeting approximately the 95th percentile.
RISK_QUANTILE_LOW = 0.90
RISK_QUANTILE_HIGH = 0.995
RISK_TARGET_PERCENTILE = 0.95
RISK_TARGET_BANDWIDTH = 0.10

# v15 late-risk consistency objective.
# We do NOT require perfectly monotonic risk. We only avoid cases with a large
# terminal reversal (e.g. 43% at year 4 -> 31% at year 5) because that weakens
# the longitudinal narrative of the example figure.
LATE_RISK_STRICT_MAX_DROP_4_TO_5_PP = 5.0
LATE_RISK_MODERATE_MAX_DROP_4_TO_5_PP = 8.0
LATE_RISK_FALLBACK_MAX_DROP_4_TO_5_PP = 12.0
LATE_RISK_REQUIRE_Y5_GE_Y3_STRICT = True
LATE_RISK_REQUIRE_Y5_GE_Y3_MODERATE = True
LATE_RISK_MIN_GAIN_Y3_TO_Y5_STRICT_PP = 0.0
LATE_RISK_MIN_GAIN_Y3_TO_Y5_MODERATE_PP = -2.0
LATE_RISK_DROP_PENALTY_WEIGHT = 0.10
LATE_RISK_GAIN_REWARD_WEIGHT = 0.025
LATE_RISK_Y5_WEIGHT = 0.035

# ------------------------------------------------------------------
# NEW v12 objective: keep the patient high-risk but avoid an explanation
# dominated by eGFR.  The selector uses staged thresholds so that it first
# searches for a genuinely non-dominant eGFR contribution, then relaxes only
# if necessary.  eGFR is still required in Panel A and remains available in B.
# ------------------------------------------------------------------
EGFR_STRICT_MAX_ABS_PP = 12.0
EGFR_STRICT_MAX_SHARE = 0.25
EGFR_MODERATE_MAX_ABS_PP = 20.0
EGFR_MODERATE_MAX_SHARE = 0.35
EGFR_FALLBACK_MAX_ABS_PP = 30.0
EGFR_MIN_ABS_PP = 0.05
EGFR_PENALTY_WEIGHT_SHARE = 2.75
EGFR_PENALTY_WEIGHT_ABS_PP = 0.035

# v16: no single feature should dominate the local explanation. The strict
# pool favors a genuinely multi-factor risk profile; a moderate fallback is
# available only if the formal IG sample is too sparse.
MULTIFACTOR_STRICT_MAX_SINGLE_ABS_PP = 10.0
MULTIFACTOR_STRICT_MAX_SINGLE_SHARE = 0.25
MULTIFACTOR_STRICT_MAX_TOP3_SHARE = 0.62
MULTIFACTOR_MODERATE_MAX_SINGLE_ABS_PP = 15.0
MULTIFACTOR_MODERATE_MAX_SINGLE_SHARE = 0.32
MULTIFACTOR_MODERATE_MAX_TOP3_SHARE = 0.72
MULTIFACTOR_MIN_POSITIVE_TERMS = 2
MULTIFACTOR_MIN_NEGATIVE_TERMS = 2
MULTIFACTOR_MIN_ABS_TERM_PP = 0.50

# Static demographic descriptors can remain in the complete force bar, but
# they are not used as the sparse text annotations in Panel B.
PANEL_B_LABEL_EXCLUDE_TOKENS = (
    "marriage",
    "marital",
    "sex",
    "gender",
    "education",
)

# If the formal IG sample contains too few eligible cases, broaden only to the
# upper quartile. We still remain in a clearly high-risk population.
RELAXED_RISK_QUANTILE_LOW = 0.80
RELAXED_RISK_QUANTILE_HIGH = 1.00
MIN_PRIMARY_POOL = 8

ANCHOR_FEATURES = [
    "eGFR",
    "HIVRNA_log10",
    "CD4",
]

COMORBIDITY_FEATURES = [
    "CVD_status",
    "diabetes_status",
    "hypertension_status",
    "hypercholesterolemia_status",
]

# Medication indicator variables used in Panel B label formatting.
# These were previously referenced as MEDICATION_FEATURES before being defined.
MEDICATION_FEATURES = [
    "antidiabetic_med",
    "antihypertensive_med",
    "antilipid_med",
]

LIPID_FEATURES = [
    "TC",
    "TG",
    "HDL",
    "LDL",
]

# Panel A is restricted to continuous / naturally trajectory-like variables.
# No categorical status variables and no ART variables are allowed here.
PANEL_A_CONTINUOUS_CANDIDATES = [
    "eGFR",
    "HIVRNA_log10",
    "CD4",
    "CD8",
    "Urea",
    "WBC",
    "PLT",
    "HB",
    "TC",
    "TG",
    "HDL",
    "LDL",
    "GLU",
    "ALT",
    "AST",
    "BMI",
]

# Variables that can fill the final two labelled slots in Panel B.
PANEL_B_FILLER_FEATURES = [
    "HB",
    "Urea",
    "WBC",
    "PLT",
    "GLU",
    "CD8",
    "BMI",
    "Age",
    "HBV_status",
    "HCV_status",
    "antidiabetic_med",
    "antihypertensive_med",
    "antilipid_med",
]

CURRENT_ART_FEATURES = [
    "current_TDF_NNRTI_3TC_FTC",
    "current_TDF_PI_3TC_FTC",
    "current_nonTDF_PI",
    "current_BIC_FTC_TAF",
    "current_EVGc_FTC_TAF",
    "current_TDF_INSTI_3TC_FTC",
    "current_nonTDF_DTG",
    "current_nonTDF_traditional_NNRTI",
]

CUMULATIVE_ART_FEATURES = [
    "TDF_NNRTI_3TC_FTC_cum_month",
    "TDF_PI_3TC_FTC_cum_month",
    "nonTDF_PI_cum_month",
    "BIC_FTC_TAF_cum_month",
    "EVGc_FTC_TAF_cum_month",
    "TDF_INSTI_3TC_FTC_cum_month",
    "nonTDF_DTG_cum_month",
    "nonTDF_traditional_NNRTI_cum_month",
]


LAB_FEATURES = [
    "HIVRNA_log10",
    "CD4",
    "CD8",
    "Urea",
    "WBC",
    "PLT",
    "HB",
    "TC",
    "TG",
    "HDL",
    "LDL",
    "GLU",
    "ALT",
    "AST",
    "eGFR",
]

PERSISTENT_STATUS_FEATURES = [
    "CVD_status",
    "diabetes_status",
    "hypertension_status",
    "hypercholesterolemia_status",
    "HBV_status",
    "HCV_status",
]

DYNAMIC_FEATURES = (
    LAB_FEATURES
    + PERSISTENT_STATUS_FEATURES
    + MEDICATION_FEATURES
    + CURRENT_ART_FEATURES
    + CUMULATIVE_ART_FEATURES
)

if len(DYNAMIC_FEATURES) != 40:
    raise RuntimeError(
        f"Expected 40 base dynamic clinical variables, got {len(DYNAMIC_FEATURES)}."
    )

EXPECTED_ENHANCED_DYNAMIC_NAMES = (
    DYNAMIC_FEATURES
    + [f"{name}_observed" for name in LAB_FEATURES]
    + [f"{name}_time_since_last" for name in LAB_FEATURES]
    + [f"{name}_delta_last_observed" for name in LAB_FEATURES]
)

if len(EXPECTED_ENHANCED_DYNAMIC_NAMES) != 85:
    raise RuntimeError(
        f"Expected 85 enhanced dynamic model inputs, got "
        f"{len(EXPECTED_ENHANCED_DYNAMIC_NAMES)}."
    )

DYNAMIC_COMPONENT_TO_CLINICAL = {
    name: name
    for name in DYNAMIC_FEATURES
}
for _lab in LAB_FEATURES:
    DYNAMIC_COMPONENT_TO_CLINICAL[f"{_lab}_observed"] = _lab
    DYNAMIC_COMPONENT_TO_CLINICAL[f"{_lab}_time_since_last"] = _lab
    DYNAMIC_COMPONENT_TO_CLINICAL[f"{_lab}_delta_last_observed"] = _lab

if set(DYNAMIC_COMPONENT_TO_CLINICAL) != set(EXPECTED_ENHANCED_DYNAMIC_NAMES):
    raise RuntimeError("The 85-channel dynamic-to-clinical mapping is incomplete.")

CURRENT_TO_CUMULATIVE = {
    "current_TDF_NNRTI_3TC_FTC": "TDF_NNRTI_3TC_FTC_cum_month",
    "current_TDF_PI_3TC_FTC": "TDF_PI_3TC_FTC_cum_month",
    "current_nonTDF_PI": "nonTDF_PI_cum_month",
    "current_BIC_FTC_TAF": "BIC_FTC_TAF_cum_month",
    "current_EVGc_FTC_TAF": "EVGc_FTC_TAF_cum_month",
    "current_TDF_INSTI_3TC_FTC": "TDF_INSTI_3TC_FTC_cum_month",
    "current_nonTDF_DTG": "nonTDF_DTG_cum_month",
    "current_nonTDF_traditional_NNRTI": "nonTDF_traditional_NNRTI_cum_month",
}

# Smoothness penalty is not applied to HIV RNA because a rapid early fall after
# ART initiation is clinically plausible and should not disqualify a good case.
SMOOTHNESS_FEATURES = {
    "eGFR",
    "CD4",
    "CD8",
    "Urea",
    "WBC",
    "PLT",
    "HB",
    "TC",
    "TG",
    "HDL",
    "LDL",
    "GLU",
    "BMI",
}

DISPLAY_NAME = {
    "HIVRNA_log10": "HIV RNA",
    "CD4": "CD4",
    "CD8": "CD8",
    "Urea": "Urea",
    "WBC": "WBC",
    "PLT": "PLT",
    "HB": "Hb",
    "TC": "TC",
    "TG": "TG",
    "HDL": "HDL-C",
    "LDL": "LDL-C",
    "GLU": "Glu",
    "ALT": "ALT",
    "AST": "AST",
    "eGFR": "eGFR",
    "Age": "Age",
    "BMI": "BMI",
    "CVD_status": "CVD",
    "diabetes_status": "DM",
    "hypertension_status": "HTN",
    "hypercholesterolemia_status": "Hyperchol.",
    "HBV_status": "HBV",
    "HCV_status": "HCV",
    "antidiabetic_med": "DM med",
    "antihypertensive_med": "HTN med",
    "antilipid_med": "Lipid med",
    "current_TDF_NNRTI_3TC_FTC": "TDF-NNRTI",
    "current_TDF_PI_3TC_FTC": "TDF-PI",
    "current_nonTDF_PI": "nonTDF-PI",
    "current_BIC_FTC_TAF": "BIC/FTC/TAF",
    "current_EVGc_FTC_TAF": "EVG/c/FTC/TAF",
    "current_TDF_INSTI_3TC_FTC": "TDF-INSTI",
    "current_nonTDF_DTG": "nonTDF-DTG",
    "current_nonTDF_traditional_NNRTI": "nonTDF-NNRTI",
    "TDF_NNRTI_3TC_FTC_cum_month": "Cum TDF-NNRTI",
    "TDF_PI_3TC_FTC_cum_month": "Cum TDF-PI",
    "nonTDF_PI_cum_month": "Cum PI",
    "BIC_FTC_TAF_cum_month": "Cum BIC/FTC/TAF",
    "EVGc_FTC_TAF_cum_month": "Cum EVG/c/FTC/TAF",
    "TDF_INSTI_3TC_FTC_cum_month": "Cum TDF-INSTI",
    "nonTDF_DTG_cum_month": "Cum nonTDF-DTG",
    "nonTDF_traditional_NNRTI_cum_month": "Cum nonTDF-NNRTI",
    "__earlier_longitudinal_history__": "Earlier longitudinal history",
}

RISK_UP = "#F21864"
RISK_DOWN = "#278CD7"
RISK_LINE = "#D62728"
HIGHLIGHT_COLOR = "#E91E63"

MARKERS = ["o", "s", "D", "^", "v", "p", "X", "<", ">", "*"]


# ============================================================
# 3. Helpers
# ============================================================

def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found:\n{path}")


def first_existing(paths: list[Path], label: str) -> Path:
    for path in paths:
        if path.exists():
            return path
    raise FileNotFoundError(
        f"Could not find {label}:\n" + "\n".join(str(p) for p in paths)
    )


def feature_display(variable: str) -> str:
    return DISPLAY_NAME.get(str(variable), str(variable).replace("_", " "))


def allow_panel_b_text_label(variable: str) -> bool:
    """
    Final no-cherry-picking rule.

    Every force-plot variable is eligible for a text label, including ART,
    concomitant medication, Course, demographic inputs, and variables whose
    clinical direction is not straightforward.

    Interpretation difficulty affects DISCUSSION wording, not whether a major
    force segment is identified in the figure.
    """
    return True



def select_panel_b_label_variables(
    B_all: pd.DataFrame,
) -> list[str]:
    """
    Final Panel-B label selection with NO clinical cherry-picking.

    Rules:
      1) all non-zero force terms are eligible;
      2) force-include the two largest positive/red contributors;
      3) force-include the single largest negative/blue contributor;
      4) fill all remaining slots using the largest |IG| terms overall;
      5) return at most PANEL_B_TOP_LABEL_N variables.

    Therefore, the largest blue segment and the second-largest red segment
    cannot be omitted simply because another variable is easier to explain.
    """
    eligible = B_all.loc[
        B_all["abs_ig_pp"] > 1e-12
    ].copy()

    if eligible.empty:
        return []

    eligible = eligible.sort_values(
        ["abs_ig_pp", "clinical_variable"],
        ascending=[False, True],
    ).reset_index(drop=True)

    mandatory: list[str] = []

    top_positive = (
        eligible.loc[eligible["ig_pp"] > 0]
        .sort_values(
            ["abs_ig_pp", "clinical_variable"],
            ascending=[False, True],
        )
        .head(PANEL_B_REQUIRED_TOP_POSITIVE_N)
        ["clinical_variable"]
        .astype(str)
        .tolist()
    )
    mandatory.extend(top_positive)

    top_negative = (
        eligible.loc[eligible["ig_pp"] < 0]
        .sort_values(
            ["abs_ig_pp", "clinical_variable"],
            ascending=[False, True],
        )
        .head(PANEL_B_REQUIRED_TOP_NEGATIVE_N)
        ["clinical_variable"]
        .astype(str)
        .tolist()
    )
    mandatory.extend(top_negative)

    mandatory = list(dict.fromkeys(mandatory))

    # Fill remaining positions by overall absolute contribution.
    selected = list(mandatory)
    for variable in eligible["clinical_variable"].astype(str).tolist():
        if variable not in selected:
            selected.append(variable)
        if len(selected) >= PANEL_B_TOP_LABEL_N:
            break

    # Keep only the requested number.
    selected = selected[:PANEL_B_TOP_LABEL_N]

    # Stable output order:
    # positive/red first, then negative/blue; within each sign by |IG|.
    selected_table = eligible.loc[
        eligible["clinical_variable"].isin(selected)
    ].copy()
    selected_table["_sign_order"] = np.where(
        selected_table["ig_pp"] > 0,
        0,
        1,
    )
    selected_table = selected_table.sort_values(
        ["_sign_order", "abs_ig_pp", "clinical_variable"],
        ascending=[True, False, True],
    )

    return (
        selected_table["clinical_variable"]
        .astype(str)
        .tolist()
    )


def zscore_within_variable(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    out = np.full(values.shape, np.nan, dtype=float)
    finite = np.isfinite(values)
    if finite.sum() == 0:
        return out
    mean = float(np.mean(values[finite]))
    sd = float(np.std(values[finite], ddof=0))
    if sd < 1e-12:
        out[finite] = 0.0
    else:
        out[finite] = (values[finite] - mean) / sd
    return out


def count_sign_reversals(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 3:
        return 0.0
    diffs = np.diff(values)
    diffs = diffs[np.abs(diffs) > 1e-12]
    if len(diffs) < 2:
        return 0.0
    signs = np.sign(diffs)
    return float(np.sum(signs[1:] * signs[:-1] < 0))


def mean_abs_step_change_z(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return np.nan
    z = zscore_within_variable(values)
    z = z[np.isfinite(z)]
    if len(z) < 2:
        return 0.0
    return float(np.mean(np.abs(np.diff(z))))


def aggregate_target_feature(target: pd.DataFrame, variable: str):
    hit = target.loc[target["clinical_variable"].eq(variable)]
    if hit.empty:
        return None
    return {
        "clinical_variable": variable,
        "signed_ig": float(hit["signed_ig"].sum()),
        "absolute_ig": float(hit["absolute_ig"].sum()),
        "current_model_value": (
            float(hit["current_model_value"].dropna().iloc[0])
            if hit["current_model_value"].notna().any()
            else np.nan
        ),
    }





def _target_landmark_step_index(patient_mask: np.ndarray, target_month: int) -> int:
    candidates = [
        i
        for i, month in enumerate(HISTORY_MONTHS)
        if int(month) <= int(target_month) and bool(patient_mask[i])
    ]
    if not candidates:
        raise ValueError(
            f"No valid step is available on/before landmark_month={target_month}."
        )
    return int(candidates[-1])



def resolve_5y_ig_checkpoint_dir() -> Path:
    """Return the first complete 5-fold 60-month formal IG checkpoint directory."""
    for directory in IG_CHECKPOINT_DIR_CANDIDATES:
        complete = all(
            (
                directory
                / f"fold_{fold_id}_landmark_{TARGET_LANDMARK_MONTH}m_ig.npz"
            ).exists()
            and (
                directory
                / f"fold_{fold_id}_landmark_{TARGET_LANDMARK_MONTH}m_metadata.json"
            ).exists()
            for fold_id in range(5)
        )
        if complete:
            return directory

    raise FileNotFoundError(
        "Could not find a complete 5-fold 60-month formal IG checkpoint set:\n"
        + "\n".join(str(path) for path in IG_CHECKPOINT_DIR_CANDIDATES)
    )


_STATIC_CATEGORY_CACHE: dict[tuple[int, int, str], str] = {}


def _pretty_static_level(parent_variable: str, component_name: str) -> str:
    """
    Convert the active one-hot component name into a publication-ready category.

    Examples:
      Marriage_Married or cohabiting -> Married/cohabiting
      Marriage_Never married         -> Never married
      WHOstage_2                     -> 2
      Course_Drugs                   -> Injection drug use
    """
    parent_variable = str(parent_variable)
    component_name = str(component_name)

    prefix = f"{parent_variable}_"
    if not component_name.startswith(prefix):
        raise ValueError(
            f"Static component {component_name!r} does not match parent "
            f"{parent_variable!r}."
        )

    level = component_name[len(prefix):].strip()
    level = level.replace("_", " ")
    level = " ".join(level.split())

    exact_replacements = {
        "Divorced-separated-or-widowed": "Divorced/separated/widowed",
        "Divorced separated or widowed": "Divorced/separated/widowed",
        "Married or cohabiting": "Married/cohabiting",
        "Male to male": "Male-to-male sexual contact",
        "Male-to-male": "Male-to-male sexual contact",
        "Drugs": "Injection drug use",
        "Injection drug": "Injection drug use",
        "Others": "Other",
    }
    return exact_replacements.get(level, level)


def decode_static_parent_category(
    *,
    patient_local: int,
    fold_id: int,
    parent_variable: str,
) -> str:
    """
    Decode the ACTUAL active category for a static parent variable from the
    exact formal year-5 IG checkpoint.

    This is used for Sex, Marriage, Course and WHOstage. It does not guess from
    the parent-level IG table. Instead it reads:
        metadata['static_names']
        checkpoint['static_input']
    and requires exactly one active one-hot component for the requested parent.

    If the one-hot structure is not valid, the script raises an error rather
    than printing an ambiguous category.
    """
    key = (int(fold_id), int(patient_local), str(parent_variable))
    if key in _STATIC_CATEGORY_CACHE:
        return _STATIC_CATEGORY_CACHE[key]

    checkpoint_dir = resolve_5y_ig_checkpoint_dir()
    npz_path = (
        checkpoint_dir
        / f"fold_{int(fold_id)}_landmark_{TARGET_LANDMARK_MONTH}m_ig.npz"
    )
    metadata_path = (
        checkpoint_dir
        / f"fold_{int(fold_id)}_landmark_{TARGET_LANDMARK_MONTH}m_metadata.json"
    )

    loaded = np.load(npz_path, allow_pickle=False)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    if "patient_local" not in loaded.files:
        raise ValueError(f"Missing patient_local in checkpoint: {npz_path}")
    if "static_input" not in loaded.files:
        raise ValueError(
            "The formal IG checkpoint does not contain static_input, so the "
            f"actual {parent_variable} category cannot be decoded safely."
        )

    static_names = list(metadata.get("static_names", []))
    static_input = loaded["static_input"].astype(np.float64)

    if static_input.ndim != 2:
        raise ValueError(
            f"Unexpected static_input shape in {npz_path}: {static_input.shape}"
        )
    if static_input.shape[1] != len(static_names):
        raise ValueError(
            "static_input width does not match metadata static_names: "
            f"{static_input.shape[1]} vs {len(static_names)}."
        )

    checkpoint_patients = loaded["patient_local"].astype(np.int64)
    patient_hits = np.where(
        checkpoint_patients == int(patient_local)
    )[0]
    if len(patient_hits) != 1:
        raise ValueError(
            f"Expected exactly one checkpoint row for patient_local={patient_local}; "
            f"found {len(patient_hits)}."
        )
    sample_index = int(patient_hits[0])

    prefix = f"{parent_variable}_"
    component_indices = [
        index
        for index, name in enumerate(static_names)
        if str(name).startswith(prefix)
    ]
    if not component_indices:
        raise ValueError(
            f"No static one-hot components found for parent={parent_variable!r}. "
            f"Available static_names={static_names}"
        )

    values = static_input[
        sample_index,
        component_indices,
    ]
    if not np.all(np.isfinite(values)):
        raise ValueError(
            f"Non-finite one-hot values found for {parent_variable}: {values}"
        )

    # Strict one-hot audit.
    active_local = np.where(values >= 0.5)[0]
    if len(active_local) != 1:
        raise ValueError(
            f"{parent_variable} is not exactly one-hot for patient_local="
            f"{patient_local}: values={values.tolist()}"
        )

    active_component_index = component_indices[
        int(active_local[0])
    ]
    active_component_name = str(
        static_names[active_component_index]
    )

    category = _pretty_static_level(
        parent_variable,
        active_component_name,
    )
    _STATIC_CATEGORY_CACHE[key] = category
    return category


def build_current_step_force_terms(
    *,
    target: pd.DataFrame,
    patient_local: int,
    fold_id: int,
    patient_mask: np.ndarray,
):
    """
    Construct a COMPLETE Panel-B decomposition with aligned temporal meaning.

    Dynamic part:
        current-step clinical IG for all 40 dynamic clinical variables
        + one net term representing all earlier longitudinal dynamic history.

    Static part:
        the original formal year-5 clinical IG rows for variables with no
        longitudinal time axis (Age/BMI/Oppinfection/Sex/Marriage/Course/WHOstage).

    By construction:
        current dynamic + earlier dynamic history + direct static
        == original complete formal year-5 clinical IG decomposition.

    The raw checkpoint is also audited against the formal dynamic total.
    """
    checkpoint_dir = resolve_5y_ig_checkpoint_dir()
    npz_path = (
        checkpoint_dir
        / f"fold_{int(fold_id)}_landmark_{TARGET_LANDMARK_MONTH}m_ig.npz"
    )
    metadata_path = (
        checkpoint_dir
        / f"fold_{int(fold_id)}_landmark_{TARGET_LANDMARK_MONTH}m_metadata.json"
    )

    loaded = np.load(npz_path, allow_pickle=False)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    required = {"patient_local", "attr_dynamic"}
    missing = sorted(required - set(loaded.files))
    if missing:
        raise ValueError(
            f"Current-step Figure-3 checkpoint is missing arrays: {missing}"
        )

    checkpoint_patients = loaded["patient_local"].astype(np.int64)
    matches = np.where(checkpoint_patients == int(patient_local))[0]
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one checkpoint row for patient_local={patient_local}; "
            f"found {len(matches)}."
        )
    sample_index = int(matches[0])

    dynamic_names = list(metadata.get("dynamic_names", []))
    if dynamic_names != EXPECTED_ENHANCED_DYNAMIC_NAMES:
        missing_names = sorted(
            set(EXPECTED_ENHANCED_DYNAMIC_NAMES) - set(dynamic_names)
        )
        extra_names = sorted(
            set(dynamic_names) - set(EXPECTED_ENHANCED_DYNAMIC_NAMES)
        )
        raise ValueError(
            "Checkpoint 85-channel dynamic input names/order do not match the "
            "locked model definition."
            f"\nmissing={missing_names}"
            f"\nextra={extra_names}"
        )

    attr_dynamic = loaded["attr_dynamic"].astype(np.float64)
    if attr_dynamic.ndim != 3 or attr_dynamic.shape[2] != 85:
        raise ValueError(
            f"Unexpected attr_dynamic shape: {attr_dynamic.shape}"
        )

    current_step = _target_landmark_step_index(
        np.asarray(patient_mask, dtype=bool),
        TARGET_LANDMARK_MONTH,
    )
    current_month = int(HISTORY_MONTHS[current_step])

    attr_current = attr_dynamic[
        sample_index,
        current_step,
        :,
    ]

    group_map: dict[str, list[int]] = {}
    for component_index, component_name in enumerate(dynamic_names):
        clinical_name = DYNAMIC_COMPONENT_TO_CLINICAL[component_name]
        group_map.setdefault(clinical_name, []).append(component_index)

    if set(group_map) != set(DYNAMIC_FEATURES):
        raise RuntimeError(
            "Current-step dynamic clinical grouping is not exactly the locked 40 variables."
        )

    current_rows = []
    for variable in DYNAMIC_FEATURES:
        indices = group_map[variable]
        selected = attr_current[indices]
        signed = float(np.sum(selected))
        current_rows.append(
            {
                "clinical_variable": variable,
                "signed_ig": signed,
                "absolute_ig": abs(signed),
                "attribution_scope": "current_step_dynamic",
                "current_step_index": int(current_step),
                "current_history_month": int(current_month),
                "component_n": int(len(indices)),
            }
        )

    current_dynamic_df = pd.DataFrame(current_rows)

    formal_dynamic = (
        target.loc[target["clinical_variable"].isin(DYNAMIC_FEATURES)]
        .groupby("clinical_variable", as_index=False)
        .agg(
            signed_ig=("signed_ig", "sum"),
            absolute_ig=("absolute_ig", "sum"),
        )
    )
    if set(formal_dynamic["clinical_variable"]) != set(DYNAMIC_FEATURES):
        missing_vars = sorted(
            set(DYNAMIC_FEATURES) - set(formal_dynamic["clinical_variable"])
        )
        raise ValueError(
            f"Formal year-5 IG is missing dynamic clinical variables: {missing_vars}"
        )

    formal_dynamic_total = float(formal_dynamic["signed_ig"].sum())
    raw_checkpoint_dynamic_total = float(
        attr_dynamic[sample_index].sum()
    )
    dynamic_reconciliation_diff_pp = 100.0 * (
        raw_checkpoint_dynamic_total - formal_dynamic_total
    )

    if abs(dynamic_reconciliation_diff_pp) > PANEL_B_DYNAMIC_RECONCILIATION_TOL_PP:
        raise RuntimeError(
            "Raw checkpoint dynamic IG does not reconcile to the formal clinical "
            "dynamic IG within tolerance. "
            f"Difference={dynamic_reconciliation_diff_pp:+.4f} pp."
        )

    current_dynamic_total = float(current_dynamic_df["signed_ig"].sum())

    # Net contribution of ALL longitudinal dynamic information before the current
    # step. Using the formal dynamic total preserves the exact clinical-level
    # completeness of the original Step12A explanation.
    earlier_history_signed = (
        formal_dynamic_total
        - current_dynamic_total
    )

    earlier_history_df = pd.DataFrame(
        [
            {
                "clinical_variable": EARLIER_HISTORY_FEATURE,
                "signed_ig": float(earlier_history_signed),
                "absolute_ig": abs(float(earlier_history_signed)),
                "attribution_scope": "earlier_longitudinal_history_net",
                "current_step_index": int(current_step),
                "current_history_month": int(current_month),
                "component_n": np.nan,
            }
        ]
    )

    static_df = (
        target.loc[~target["clinical_variable"].isin(DYNAMIC_FEATURES)]
        .groupby("clinical_variable", as_index=False)
        .agg(
            signed_ig=("signed_ig", "sum"),
            absolute_ig=("absolute_ig", "sum"),
        )
    )
    static_df["attribution_scope"] = "direct_static_or_age"
    static_df["current_step_index"] = np.nan
    static_df["current_history_month"] = np.nan
    static_df["component_n"] = np.nan

    force_terms = pd.concat(
        [
            current_dynamic_df,
            earlier_history_df,
            static_df,
        ],
        ignore_index=True,
        sort=False,
    )

    original_total = float(target["signed_ig"].sum())
    hybrid_total = float(force_terms["signed_ig"].sum())
    hybrid_diff_pp = 100.0 * (hybrid_total - original_total)

    if abs(hybrid_diff_pp) > 1e-7:
        raise RuntimeError(
            "Current-step + earlier-history + static hybrid decomposition does "
            "not exactly equal the original formal clinical IG total. "
            f"Difference={hybrid_diff_pp:+.8f} pp."
        )

    audit = {
        "checkpoint_dir": str(checkpoint_dir),
        "checkpoint_file": str(npz_path),
        "current_step_index": int(current_step),
        "current_history_month": int(current_month),
        "formal_dynamic_total_pp": 100.0 * formal_dynamic_total,
        "raw_checkpoint_dynamic_total_pp": 100.0 * raw_checkpoint_dynamic_total,
        "dynamic_reconciliation_diff_pp": dynamic_reconciliation_diff_pp,
        "current_dynamic_net_pp": 100.0 * current_dynamic_total,
        "earlier_history_net_pp": 100.0 * earlier_history_signed,
        "original_complete_ig_total_pp": 100.0 * original_total,
        "hybrid_complete_ig_total_pp": 100.0 * hybrid_total,
        "hybrid_minus_original_pp": hybrid_diff_pp,
    }

    return force_terms, audit


PANEL_B_BINARY_STYLE_FEATURES = set(
    COMORBIDITY_FEATURES
    + MEDICATION_FEATURES
    + CURRENT_ART_FEATURES
    + ["HBV_status", "HCV_status"]
)


def _format_panel_b_value(variable: str, value: float) -> str:
    variable = str(variable)
    if not np.isfinite(value):
        return "NA"
    if variable in {"Age"}:
        return f"{value:.0f}"
    if variable in {"CD4", "CD8", "WBC", "PLT", "HB"}:
        return f"{value:.0f}"
    if variable.endswith("_cum_month"):
        return f"{value:.0f}"
    if variable in {"eGFR", "BMI", "Urea", "TC", "TG", "HDL", "LDL", "GLU", "ALT", "AST", "HIVRNA_log10"}:
        return f"{value:.1f}"
    return f"{value:.2f}"


def panel_b_label_text(
    variable: str,
    *,
    target: pd.DataFrame,
    patient_local: int,
    fold_id: int,
    global_index: int,
    patient_mask: np.ndarray,
    feature_to_index: dict,
    X_development,
) -> str:
    """Reference-style label text: show the feature name and this patient's
    year-5 feature value when possible, otherwise fall back to the variable name.
    """
    disp = feature_display(variable)
    variable = str(variable)

    if variable == EARLIER_HISTORY_FEATURE:
        return EARLIER_HISTORY_DISPLAY

    # 0) Decode the ACTUAL active static category from the formal checkpoint.
    #    This avoids ambiguous labels such as only "Marriage" or "WHOstage".
    if variable in {"Sex", "Marriage", "Course", "WHOstage"}:
        category = decode_static_parent_category(
            patient_local=patient_local,
            fold_id=fold_id,
            parent_variable=variable,
        )
        parent_label = {
            "Sex": "Sex",
            "Marriage": "Marriage",
            "Course": "HIV transmission route",
            "WHOstage": "WHO stage",
        }[variable]
        return f"{parent_label} = {category}"

    # 1) Raw continuous value from the longitudinal feature matrix.
    if variable in continuous_to_index:
        step_idx = _target_landmark_step_index(patient_mask, TARGET_LANDMARK_MONTH)
        ci = int(continuous_to_index[variable])
        value = np.nan
        for step in range(step_idx, -1, -1):
            if not bool(patient_mask[step]):
                continue
            candidate = float(continuous_raw[global_index, step, ci])
            if np.isfinite(candidate):
                value = candidate
                break
        if np.isfinite(value):

            formatted = _format_panel_b_value(variable, value)
            if variable == "Age":
                return f"{disp} = {formatted} y"
            if variable == "eGFR":
                return f"{disp} = {formatted}"
            if variable == "HIVRNA_log10":
                return f"{disp} = {formatted} log10"
            if variable in {"TC", "TG", "HDL", "LDL", "GLU"}:
                return f"{disp} = {formatted}"
            if variable in {"CD4", "CD8"}:
                return f"{disp} = {formatted}"
            if variable.endswith("_cum_month"):
                return f"{disp} = {formatted} months"
            return f"{disp} = {formatted}"

    # 2) Binary / indicator variables from the model input row.
    if variable in feature_to_index:
        step_idx = _target_landmark_step_index(patient_mask, TARGET_LANDMARK_MONTH)
        model_value = float(X_development[patient_local, step_idx, feature_to_index[variable]])
        if variable in PANEL_B_BINARY_STYLE_FEATURES:
            return f"{disp} = {'Yes' if model_value >= 0.5 else 'No'}"
        if np.isfinite(model_value) and abs(model_value - round(model_value)) < 1e-6 and abs(model_value) <= 1.0:
            return f"{disp} = {'Yes' if model_value >= 0.5 else 'No'}"

    # 3) Fall back to current_model_value from the IG table if available.
    agg = aggregate_target_feature(target, variable)
    if agg is not None:
        current_model_value = float(agg.get("current_model_value", np.nan))
        if np.isfinite(current_model_value):
            if variable in PANEL_B_BINARY_STYLE_FEATURES:
                return f"{disp} = {'Yes' if current_model_value >= 0.5 else 'No'}"
            return f"{disp} = {_format_panel_b_value(variable, current_model_value)}"

    return disp


def year5_importance_table(target: pd.DataFrame) -> pd.DataFrame:
    out = (
        target.groupby("clinical_variable", as_index=False)
        .agg(
            signed_ig=("signed_ig", "sum"),
            absolute_ig=("absolute_ig", "sum"),
        )
        .copy()
    )
    out["abs_signed_ig"] = np.abs(out["signed_ig"])
    out = out.sort_values(
        ["absolute_ig", "clinical_variable"],
        ascending=[False, True],
    ).reset_index(drop=True)
    out["rank"] = np.arange(1, len(out) + 1)
    out["pct_rank"] = out["absolute_ig"].rank(
        method="average",
        pct=True,
        ascending=True,
    )
    return out


def patient_mean_importance(patient_ig: pd.DataFrame) -> dict[str, float]:
    """Mean |IG| across all available formal landmarks for Panel A ranking."""
    grouped = (
        patient_ig.groupby("clinical_variable", as_index=False)
        .agg(mean_abs_ig=("absolute_ig", "mean"))
    )
    return dict(
        zip(
            grouped["clinical_variable"].astype(str),
            grouped["mean_abs_ig"].astype(float),
        )
    )


def choose_profiles(
    patient_ig: pd.DataFrame,
    target: pd.DataFrame,
    patient_local: int,
    global_index: int,
    fold_id: int,
    patient_mask: np.ndarray,
    relaxed: bool = False,
):
    """
    Build the variables used in the two panels.

    Panel A:
      - exactly 10 continuous variables;
      - eGFR, HIV RNA and CD4 are required;
      - at least one lipid variable is required;
      - the remaining variables must have enough genuinely observed measurements
        and enough variation on the fold-standardized model scale;
      - ranking combines patient-specific mean |IG| and display quality.

    Panel B:
      - full year-5 IG decomposition is retained;
      - only the final top-10 major terms are labelled, matching the sparse
        annotation density of the reference figure;
      - labels include eGFR, HIV RNA, CD4, one present comorbidity, one lipid,
        current ART, matched cumulative ART, and one additional strong feature.
    """
    imp_y5 = year5_importance_table(target)
    y5_abs = dict(zip(imp_y5["clinical_variable"], imp_y5["absolute_ig"]))
    mean_abs = patient_mean_importance(patient_ig)

    if any(v not in y5_abs for v in ANCHOR_FEATURES):
        return None

    display_table = continuous_display_table(
        patient_local=patient_local,
        global_index=global_index,
        fold_id=fold_id,
        patient_mask=patient_mask,
        mean_abs=mean_abs,
    )
    if display_table.empty:
        return None

    display_lookup = display_table.set_index("clinical_variable").to_dict("index")

    # Anchors must be genuinely measured often enough. We intentionally use a
    # looser variation threshold for HIV RNA because suppression after ART can
    # produce a clinically meaningful early drop followed by a plateau.
    anchor_min_obs = dict(A_MIN_OBS_ANCHOR)
    other_min_obs = A_MIN_OBS_OTHER
    min_obs_range = A_MIN_OBSERVED_Z_RANGE_OTHER
    min_all_range = A_MIN_ALL_Z_RANGE_OTHER
    if relaxed:
        anchor_min_obs = {"eGFR": 2, "HIVRNA_log10": 1, "CD4": 2}
        other_min_obs = 3
        min_obs_range = 0.45
        min_all_range = 0.55

    for v in ANCHOR_FEATURES:
        if v not in display_lookup:
            return None
        if int(display_lookup[v]["n_observed"]) < int(anchor_min_obs[v]):
            return None

    # Other variables must actually be drawable rather than median-imputed flats.
    eligible_other = display_table.loc[
        (~display_table["clinical_variable"].isin(ANCHOR_FEATURES))
        & (display_table["n_observed"] >= other_min_obs)
        & (display_table["observed_z_range"] >= min_obs_range)
        & (display_table["all_z_range"] >= min_all_range)
    ].copy()

    # At least one lipid with a real longitudinal signal.
    eligible_lipids = eligible_other.loc[
        eligible_other["clinical_variable"].isin(LIPID_FEATURES)
    ].copy()
    if eligible_lipids.empty:
        return None

    eligible_lipids = eligible_lipids.sort_values(
        ["joint_score", "mean_abs_ig"], ascending=[False, False]
    )
    selected_lipids = eligible_lipids["clinical_variable"].astype(str).tolist()[:2]

    # Always keep the strongest lipid; keep a second lipid only if it is among
    # the upper half of all eligible continuous variables by joint score.
    selected_lipids = selected_lipids[:1] + [
        v for v in selected_lipids[1:2]
        if float(display_lookup[v]["joint_score"])
        >= float(eligible_other["joint_score"].median())
    ]

    a_selected = list(ANCHOR_FEATURES) + selected_lipids
    remaining = eligible_other.loc[
        ~eligible_other["clinical_variable"].isin(a_selected)
    ].sort_values(
        ["joint_score", "mean_abs_ig", "visual_score"],
        ascending=[False, False, False],
    )
    a_selected += remaining["clinical_variable"].astype(str).tolist()[
        : TOP_A_N - len(a_selected)
    ]
    a_selected = list(dict.fromkeys(a_selected))
    if len(a_selected) != TOP_A_N:
        return None

    # -------------------------
    # Present major comorbidity
    # -------------------------
    present_comorbidities = []
    for v in COMORBIDITY_FEATURES:
        row = aggregate_target_feature(target, v)
        if row is None:
            continue
        if np.isfinite(row["current_model_value"]) and row["current_model_value"] > 0.5:
            present_comorbidities.append(row)
    if not present_comorbidities:
        return None
    present_comorbidities = sorted(
        present_comorbidities,
        key=lambda x: x["absolute_ig"],
        reverse=True,
    )
    selected_comorbidity = str(present_comorbidities[0]["clinical_variable"])

    # -------------------------
    # Active current ART regimen
    # -------------------------
    active_art = []
    for v in CURRENT_ART_FEATURES:
        row = aggregate_target_feature(target, v)
        if row is None:
            continue
        if np.isfinite(row["current_model_value"]) and row["current_model_value"] > 0.5:
            active_art.append(row)
    if not active_art:
        return None
    active_art = sorted(
        active_art,
        key=lambda x: (x["absolute_ig"], x["current_model_value"]),
        reverse=True,
    )
    selected_current_art = str(active_art[0]["clinical_variable"])
    selected_cumulative_art = CURRENT_TO_CUMULATIVE.get(selected_current_art)
    if selected_cumulative_art is None or selected_cumulative_art not in y5_abs:
        return None

    # -------------------------
    # Panel B labelled set: top major terms, deliberately sparse like the reference.
    # -------------------------
    strongest_lipid_for_b = max(
        [v for v in LIPID_FEATURES if v in y5_abs],
        key=lambda v: y5_abs[v],
    )

    b_selected = (
        list(ANCHOR_FEATURES)
        + [selected_comorbidity]
        + [strongest_lipid_for_b]
        + [selected_current_art, selected_cumulative_art]
    )

    b_fillers = [
        v for v in PANEL_B_FILLER_FEATURES
        if v in y5_abs and v not in b_selected
    ]
    b_fillers = sorted(b_fillers, key=lambda v: y5_abs[v], reverse=True)
    if len(b_fillers) < TOP_B_LABEL_N - len(b_selected):
        return None
    b_selected += b_fillers[: TOP_B_LABEL_N - len(b_selected)]
    b_selected = list(dict.fromkeys(b_selected))
    if len(b_selected) != TOP_B_LABEL_N:
        return None

    selected_display = display_table.loc[
        display_table["clinical_variable"].isin(a_selected)
    ].copy()

    return {
        "A_variables": a_selected,
        "B_variables": b_selected,
        "lipids": selected_lipids,
        "comorbidity": selected_comorbidity,
        "current_art": selected_current_art,
        "cumulative_art": selected_cumulative_art,
        "active_art_count_year5": len(active_art),
        "A_display_table": selected_display,
        "A_visual_score": float(selected_display["visual_score"].mean()),
        "A_joint_score": float(selected_display["joint_score"].mean()),
        "A_min_observed": int(selected_display["n_observed"].min()),
        "A_median_observed_z_range": float(selected_display["observed_z_range"].median()),
        "A_median_all_z_range": float(selected_display["all_z_range"].median()),
        "A_median_temporal_sd": float(selected_display["temporal_sd_z"].median()),
        "A_n_dynamic": int(selected_display["is_dynamic"].sum()),
        "A_n_early_dynamic": int(selected_display["is_early_dynamic"].sum()),
        "A_n_late_dynamic": int(selected_display["is_late_dynamic"].sum()),
        "A_n_sustained_dynamic": int(selected_display["is_sustained_dynamic"].sum()),
        "A_median_early_z_range": float(selected_display["early_z_range"].median()),
        "A_median_late_z_range": float(selected_display["late_z_range"].median()),
        "A_mean_obs_fraction": float(selected_display["obs_fraction"].mean()),
        "A_max_abs_z": float(selected_display["max_abs_z"].max()),
        "A_median_plateau_fraction": float(selected_display["plateau_fraction"].median()),
        "A_mean_plateau_fraction": float(selected_display["plateau_fraction"].mean()),
        "A_max_plateau_fraction": float(selected_display["plateau_fraction"].max()),
        "A_mean_longest_plateau_fraction": float(selected_display["longest_plateau_fraction"].mean()),
        "A_n_low_plateau": int(selected_display["low_plateau"].sum()),
        "A_n_lipids": int(sum(v in LIPID_FEATURES for v in a_selected)),
    }


_FOLD_CACHE = {}


def get_fold_model_inputs(fold_id: int):
    fold_id = int(fold_id)
    if fold_id not in _FOLD_CACHE:
        fold_dir = STEP4_DIR / f"fold_{fold_id}"
        feature_names = (
            pd.read_csv(
                fold_dir / "feature_names.csv",
                encoding="utf-8-sig",
            )["feature_name"]
            .astype(str)
            .tolist()
        )
        X = np.load(fold_dir / "X_development.npy", mmap_mode="r")
        _FOLD_CACHE[fold_id] = (
            X,
            {name: i for i, name in enumerate(feature_names)},
        )
    return _FOLD_CACHE[fold_id]



def continuous_display_table(
    patient_local: int,
    global_index: int,
    fold_id: int,
    patient_mask: np.ndarray,
    mean_abs: dict[str, float],
) -> pd.DataFrame:
    """Quantify whether continuous variables are suitable for Panel A.

    Crucially, trajectory variation is evaluated on the *fold-standardized model
    input scale*, the same scale that is plotted in Panel A. This avoids the
    artificial two-level shapes created by within-patient z-scoring.
    """
    X, feature_to_index = get_fold_model_inputs(fold_id)
    rows = []

    for variable in PANEL_A_CONTINUOUS_CANDIDATES:
        if variable not in continuous_to_index or variable not in feature_to_index:
            continue

        ci = int(continuous_to_index[variable])
        fi = int(feature_to_index[variable])

        valid = np.asarray(patient_mask, dtype=bool).copy()
        z_all = np.asarray(X[patient_local, :, fi], dtype=float).copy()
        raw = np.asarray(continuous_raw[global_index, :, ci], dtype=float).copy()
        observed = valid & np.isfinite(raw)
        z_valid = z_all[valid & np.isfinite(z_all)]
        z_obs = z_all[observed & np.isfinite(z_all)]

        n_observed = int(observed.sum())
        n_valid = int(valid.sum())
        obs_fraction = float(n_observed / max(n_valid, 1))

        all_z_range = (
            float(np.max(z_valid) - np.min(z_valid)) if len(z_valid) >= 2 else 0.0
        )
        observed_z_range = (
            float(np.max(z_obs) - np.min(z_obs)) if len(z_obs) >= 2 else 0.0
        )
        max_abs_z = float(np.max(np.abs(z_valid))) if len(z_valid) else np.inf

        if n_observed >= 2:
            raw_obs = raw[observed]
            unique_observed = int(len(np.unique(np.round(raw_obs, 8))))
        else:
            unique_observed = n_observed

        # v15: quantify total, early (0-2.5 y), and late (2.5-5 y) movement.
        if len(z_valid) >= 3:
            diffs = np.diff(z_valid)
            abs_diffs = np.abs(diffs)
            mean_step = float(np.mean(abs_diffs))
            temporal_sd = float(np.std(z_valid, ddof=0))
            n_material_changes = int(np.sum(abs_diffs >= 0.25))
        else:
            mean_step = np.nan
            temporal_sd = 0.0
            n_material_changes = 0

        months = np.asarray(HISTORY_MONTHS, dtype=float)
        finite_model = valid & np.isfinite(z_all)
        early_mask = finite_model & (months <= 30)
        late_mask = finite_model & (months >= 30)
        z_early = z_all[early_mask]
        z_late = z_all[late_mask]

        early_range = (
            float(np.max(z_early) - np.min(z_early)) if len(z_early) >= 2 else 0.0
        )
        late_range = (
            float(np.max(z_late) - np.min(z_late)) if len(z_late) >= 2 else 0.0
        )
        early_changes = (
            int(np.sum(np.abs(np.diff(z_early)) >= 0.20)) if len(z_early) >= 2 else 0
        )
        late_changes = (
            int(np.sum(np.abs(np.diff(z_late)) >= 0.20)) if len(z_late) >= 2 else 0
        )
        is_early_dynamic = int(
            early_range >= A_HALF_DYNAMIC_RANGE_MIN
            and early_changes >= A_HALF_DYNAMIC_CHANGE_MIN
        )
        is_late_dynamic = int(
            late_range >= A_HALF_DYNAMIC_RANGE_MIN
            and late_changes >= A_HALF_DYNAMIC_CHANGE_MIN
        )
        is_sustained_dynamic = int(is_early_dynamic and is_late_dynamic)

        # v16 plateau audit. This is calculated on every valid 6-month model
        # input step so repeated LOCF/flat segments are visible to the selector.
        if len(z_valid) >= 2:
            valid_step_diffs = np.abs(np.diff(z_valid))
            plateau_fraction = float(
                np.mean(valid_step_diffs < A_PLATEAU_STEP_EPS_Z)
            )

            # Longest consecutive plateau run, expressed as a fraction of all
            # available transition steps.
            plateau_flags = valid_step_diffs < A_PLATEAU_STEP_EPS_Z
            longest_plateau_run = 0
            current_run = 0
            for flag in plateau_flags:
                if bool(flag):
                    current_run += 1
                    longest_plateau_run = max(longest_plateau_run, current_run)
                else:
                    current_run = 0
            longest_plateau_fraction = float(
                longest_plateau_run / max(len(valid_step_diffs), 1)
            )
        else:
            plateau_fraction = 1.0
            longest_plateau_fraction = 1.0

        low_plateau = int(plateau_fraction <= A_LOW_PLATEAU_FRACTION_MAX)
        nonplateau_reward = max(0.0, 1.0 - plateau_fraction)

        # Moderate movement is desirable; only very abrupt oscillation is penalized.
        movement_reward = min(mean_step / 0.45, 1.0) if np.isfinite(mean_step) else 0.0
        abrupt_penalty = (
            1.0
            if (not np.isfinite(mean_step) or mean_step <= 1.50)
            else 1.0 / (1.0 + 0.60 * (mean_step - 1.50))
        )
        extreme_penalty = (
            1.0 if max_abs_z <= A_MAX_ABS_Z_SOFT
            else 1.0 / (1.0 + 0.8 * (max_abs_z - A_MAX_ABS_Z_SOFT))
        )
        plateau_penalty = max(0.30, 1.0 - 0.55 * plateau_fraction)

        visual_score = (
            0.28 * min(obs_fraction / 0.65, 1.0)
            + 0.23 * min(observed_z_range / 1.6, 1.0)
            + 0.18 * min(all_z_range / 2.0, 1.0)
            + 0.12 * min(temporal_sd / 0.65, 1.0)
            + 0.08 * min(n_material_changes / 5.0, 1.0)
            + 0.05 * min(unique_observed / 5.0, 1.0)
            + 0.10 * movement_reward
            + 0.12 * nonplateau_reward
        ) * abrupt_penalty * extreme_penalty * plateau_penalty

        rows.append(
            {
                "clinical_variable": variable,
                "mean_abs_ig": float(mean_abs.get(variable, 0.0)),
                "n_observed": n_observed,
                "n_valid": n_valid,
                "obs_fraction": obs_fraction,
                "unique_observed": unique_observed,
                "observed_z_range": observed_z_range,
                "all_z_range": all_z_range,
                "max_abs_z": max_abs_z,
                "mean_abs_step_z": mean_step,
                "temporal_sd_z": temporal_sd,
                "n_material_changes": n_material_changes,
                "early_z_range": early_range,
                "late_z_range": late_range,
                "early_material_changes": early_changes,
                "late_material_changes": late_changes,
                "is_early_dynamic": is_early_dynamic,
                "is_late_dynamic": is_late_dynamic,
                "is_sustained_dynamic": is_sustained_dynamic,
                "plateau_fraction": plateau_fraction,
                "longest_plateau_fraction": longest_plateau_fraction,
                "low_plateau": low_plateau,
                "is_dynamic": int(
                    (all_z_range >= A_DYNAMIC_RANGE_MIN)
                    and (temporal_sd >= A_DYNAMIC_SD_MIN)
                    and (n_material_changes >= A_DYNAMIC_CHANGE_MIN)
                ),
                "visual_score": float(visual_score),
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    out["importance_pct"] = out["mean_abs_ig"].rank(
        method="average", pct=True, ascending=True
    )
    out["visual_pct"] = out["visual_score"].rank(
        method="average", pct=True, ascending=True
    )
    out["joint_score"] = (
        0.58 * out["importance_pct"]
        + 0.42 * out["visual_pct"]
    )
    return out.sort_values(
        ["joint_score", "mean_abs_ig"], ascending=[False, False]
    ).reset_index(drop=True)


def regimen_sequence_and_changes(
    patient_local: int,
    fold_id: int,
    patient_mask: np.ndarray,
):
    X, feature_to_index = get_fold_model_inputs(fold_id)
    formal_steps = [int(m // 6) for m in LANDMARK_MONTHS]
    sequence = []

    for step in formal_steps:
        if step >= len(patient_mask) or not bool(patient_mask[step]):
            sequence.append(None)
            continue

        active = []
        for v in CURRENT_ART_FEATURES:
            if v not in feature_to_index:
                continue
            value = float(X[patient_local, step, feature_to_index[v]])
            if np.isfinite(value) and value > 0.5:
                active.append(v)

        sequence.append(active[0] if active else None)

    observed = [x for x in sequence if x is not None]
    changes = int(sum(a != b for a, b in zip(observed[:-1], observed[1:])))
    return sequence, changes


def spread_positions(values, lower, upper, min_gap):
    values = np.asarray(values, dtype=float)
    if len(values) <= 1:
        return values.copy()

    order = np.argsort(values)
    x = values[order].copy()
    available = max(upper - lower, 1e-6)
    if len(x) > 1:
        min_gap = min(min_gap, available / (len(x) - 1) * 0.88)

    for i in range(1, len(x)):
        x[i] = max(x[i], x[i - 1] + min_gap)

    if x[-1] > upper:
        x -= x[-1] - upper

    for i in range(len(x) - 2, -1, -1):
        x[i] = min(x[i], x[i + 1] - min_gap)

    if x[0] < lower:
        x += lower - x[0]

    out = np.empty_like(x)
    out[order] = x
    return np.clip(out, lower, upper)


def wrap_feature_label(text: str, width: int = 18) -> str:
    return "\n".join(
        textwrap.wrap(
            str(text),
            width=width,
            break_long_words=False,
            break_on_hyphens=False,
        )
    )


# ============================================================
# 4. Load formal IG + frozen development data
# ============================================================

FORMAL_IG_FILE = first_existing(
    FORMAL_IG_CANDIDATES,
    "formal/global sample-level IG file",
)

for path in [RISK_FILE, LONG_MAP_FILE]:
    require_file(path)

ig = pd.read_csv(FORMAL_IG_FILE, encoding="utf-8-sig")

required_ig_columns = {
    "patient_local",
    "landmark_month",
    "clinical_variable",
    "signed_ig",
    "absolute_ig",
    "predicted_5y_risk",
    "reference_5y_risk",
    "current_model_value",
}
missing = sorted(required_ig_columns - set(ig.columns))
if missing:
    raise ValueError(f"Formal IG file missing columns: {missing}")

for col in [
    "patient_local",
    "landmark_month",
    "signed_ig",
    "absolute_ig",
    "predicted_5y_risk",
    "reference_5y_risk",
    "current_model_value",
]:
    ig[col] = pd.to_numeric(ig[col], errors="coerce")

ig = ig.dropna(subset=["patient_local", "landmark_month"]).copy()
ig["patient_local"] = ig["patient_local"].astype(int)
ig["landmark_month"] = ig["landmark_month"].astype(int)
ig["clinical_variable"] = ig["clinical_variable"].astype(str)

development_idx = np.load(STEP1_DIR / "development_idx.npy").astype(int)
development_fold_id = np.load(STEP2_DIR / "development_fold_id.npy").astype(int)
row_mask = np.load(STEP4_DIR / "sequence_row_mask_development.npy").astype(bool)

continuous_raw = np.load(
    STEP3_DIR / "continuous_raw_0_60.npy",
    mmap_mode="r",
)

feature_groups = json.loads(
    (STEP3_DIR / "feature_groups.json").read_text(encoding="utf-8")
)
continuous_vars = list(feature_groups["continuous_vars"])
continuous_to_index = {name: i for i, name in enumerate(continuous_vars)}

long_map = np.load(LONG_MAP_FILE).astype(int)
risk_long = np.load(RISK_FILE, mmap_mode="r")

if long_map.shape[0] != len(development_idx):
    raise ValueError(
        f"long_map patient count={long_map.shape[0]} but development cohort={len(development_idx)}"
    )
if long_map.shape[1] < len(LANDMARK_MONTHS):
    raise ValueError(
        f"long_map has {long_map.shape[1]} landmarks; expected at least {len(LANDMARK_MONTHS)}"
    )

print("=" * 100)
print("Formal IG file loaded:", FORMAL_IG_FILE)
print("Formal IG shape:", ig.shape)
print("Unique patients in formal IG:", ig["patient_local"].nunique())
print(
    "Unique patients with year-5 formal IG:",
    ig.loc[
        ig["landmark_month"].eq(TARGET_LANDMARK_MONTH),
        "patient_local",
    ].nunique(),
)
print("=" * 100)


# ============================================================
# 5. Build FULL development 0/1/2/3/4/5-year risk table
# ============================================================

risk_rows = []
for patient_local in range(len(development_idx)):
    for landmark_index, landmark_month in enumerate(LANDMARK_MONTHS):
        long_row = int(long_map[patient_local, landmark_index])
        if long_row < 0:
            continue
        risk_value = float(risk_long[long_row, -1])
        if not np.isfinite(risk_value):
            continue
        risk_rows.append(
            {
                "patient_local": int(patient_local),
                "landmark_index": int(landmark_index),
                "landmark_month": int(landmark_month),
                "predicted_5y_risk": risk_value,
            }
        )

risk_table = pd.DataFrame(risk_rows)
if risk_table.empty:
    raise ValueError("Full development risk table is empty.")

landmark_counts = risk_table.groupby("patient_local")["landmark_month"].nunique()
complete_patients = set(
    landmark_counts[
        landmark_counts == len(LANDMARK_MONTHS)
    ].index.astype(int)
)

risk_5_all = (
    risk_table.loc[
        risk_table["landmark_month"].eq(TARGET_LANDMARK_MONTH),
        ["patient_local", "predicted_5y_risk"],
    ]
    .drop_duplicates("patient_local")
    .copy()
)
risk_5_all = risk_5_all.loc[
    risk_5_all["patient_local"].isin(complete_patients)
].copy()

if risk_5_all.empty:
    raise ValueError("No development patient has a complete 0-5y risk trajectory.")

risk_5_all["risk_percentile"] = risk_5_all["predicted_5y_risk"].rank(
    method="average",
    pct=True,
    ascending=True,
)

formal_year5_patients = set(
    ig.loc[
        ig["landmark_month"].eq(TARGET_LANDMARK_MONTH),
        "patient_local",
    ]
    .astype(int)
    .unique()
)

risk_5 = risk_5_all.loc[
    risk_5_all["patient_local"].isin(formal_year5_patients)
].copy()

print("All development patients with complete 0-5y risk trajectory:", len(risk_5_all))
print("Year-5 formal-IG patients with complete 0-5y risk trajectory:", len(risk_5))

if risk_5.empty:
    raise ValueError(
        "No overlap between year-5 formal IG patients and the frozen development risk table."
    )

# ------------------------------------------------------------------
# Exclude patients already displayed in previous Figure-3 versions.
# v14 is handled specially: we keep its selected patient as a benchmark and
# exclude it only from the new v15 search so that a genuinely alternative case
# can be found and compared against it.
excluded_patient_ids = set(int(x) for x in MANUAL_EXCLUDE_PATIENT_LOCALS)
benchmark_v14_patient_local = None

for previous_table in [
    PREVIOUS_V10_CANDIDATE_TABLE,
    PREVIOUS_V11_CANDIDATE_TABLE,
    PREVIOUS_V12_CANDIDATE_TABLE,
    PREVIOUS_V13_CANDIDATE_TABLE,
]:
    if not previous_table.exists():
        continue
    try:
        previous_ranking = pd.read_csv(previous_table, encoding="utf-8-sig")
        if {"patient_local", "selected_for_figure"}.issubset(previous_ranking.columns):
            selected_mask = (
                previous_ranking["selected_for_figure"]
                .astype(str)
                .str.strip()
                .str.lower()
                .isin(["true", "1", "yes"])
            )
            previous_selected = (
                pd.to_numeric(
                    previous_ranking.loc[selected_mask, "patient_local"],
                    errors="coerce",
                )
                .dropna()
                .astype(int)
                .tolist()
            )
            excluded_patient_ids.update(previous_selected)
    except Exception as exc:
        warnings.warn(
            f"Could not read previous Figure-3 candidate table for exclusion: "
            f"{previous_table} ({exc})"
        )

if PREVIOUS_V14_CANDIDATE_TABLE.exists():
    try:
        v14_ranking = pd.read_csv(PREVIOUS_V14_CANDIDATE_TABLE, encoding="utf-8-sig")
        if {"patient_local", "selected_for_figure"}.issubset(v14_ranking.columns):
            v14_selected_mask = (
                v14_ranking["selected_for_figure"]
                .astype(str)
                .str.strip()
                .str.lower()
                .isin(["true", "1", "yes"])
            )
            v14_selected_ids = (
                pd.to_numeric(
                    v14_ranking.loc[v14_selected_mask, "patient_local"],
                    errors="coerce",
                )
                .dropna()
                .astype(int)
                .tolist()
            )
            if v14_selected_ids:
                benchmark_v14_patient_local = int(v14_selected_ids[0])
                excluded_patient_ids.add(benchmark_v14_patient_local)
                print(
                    "v14 benchmark patient_local retained for comparison but "
                    "excluded from v15 selection:",
                    benchmark_v14_patient_local,
                )
    except Exception as exc:
        warnings.warn(f"Could not read v14 benchmark table: {exc}")

# v15 is not used as the benchmark here; it is simply excluded so v16 searches
# for a genuinely different trajectory.
if PREVIOUS_V15_CANDIDATE_TABLE.exists():
    try:
        v15_ranking = pd.read_csv(PREVIOUS_V15_CANDIDATE_TABLE, encoding="utf-8-sig")
        if {"patient_local", "selected_for_figure"}.issubset(v15_ranking.columns):
            v15_selected_mask = (
                v15_ranking["selected_for_figure"]
                .astype(str)
                .str.strip()
                .str.lower()
                .isin(["true", "1", "yes"])
            )
            v15_selected_ids = (
                pd.to_numeric(
                    v15_ranking.loc[v15_selected_mask, "patient_local"],
                    errors="coerce",
                )
                .dropna()
                .astype(int)
                .tolist()
            )
            excluded_patient_ids.update(v15_selected_ids)
            if v15_selected_ids:
                print("v15 patient_local excluded from v16 selection:", v15_selected_ids)
    except Exception as exc:
        warnings.warn(f"Could not read v15 candidate table: {exc}")

if excluded_patient_ids:
    print("Excluded from new v16 selection patient_local IDs:", sorted(excluded_patient_ids))
    risk_5 = risk_5.loc[
        ~risk_5["patient_local"].isin(excluded_patient_ids)
    ].copy()

if risk_5.empty:
    raise ValueError(
        "No year-5 formal-IG patients remain after excluding previously displayed cases."
    )

primary_pool = risk_5.loc[
    risk_5["risk_percentile"].between(
        RISK_QUANTILE_LOW,
        RISK_QUANTILE_HIGH,
        inclusive="both",
    )
].copy()

if len(primary_pool) >= MIN_PRIMARY_POOL:
    candidate_pool = primary_pool.copy()
    pool_description = (
        f"{RISK_QUANTILE_LOW:.0%}-{RISK_QUANTILE_HIGH:.0%} development-risk percentile"
    )
else:
    candidate_pool = risk_5.loc[
        risk_5["risk_percentile"].between(
            RELAXED_RISK_QUANTILE_LOW,
            RELAXED_RISK_QUANTILE_HIGH,
            inclusive="both",
        )
    ].copy()
    pool_description = (
        f"relaxed {RELAXED_RISK_QUANTILE_LOW:.0%}-{RELAXED_RISK_QUANTILE_HIGH:.1%} "
        "development-risk percentile"
    )

candidate_ids = candidate_pool["patient_local"].astype(int).tolist()

print("\nCandidate risk pool:", pool_description)
print("Candidate patient count before clinical constraints:", len(candidate_ids))


# ============================================================
# 6. Candidate evaluation
# ============================================================

def evaluate_candidates(candidate_ids, relaxed=False):
    rows = []

    for patient_local in candidate_ids:
        patient_local = int(patient_local)
        if patient_local < 0 or patient_local >= len(development_idx):
            continue

        global_index = int(development_idx[patient_local])
        fold_id = int(development_fold_id[patient_local])
        patient_mask = np.asarray(row_mask[patient_local], dtype=bool)

        patient_ig = ig.loc[ig["patient_local"].eq(patient_local)].copy()
        target = patient_ig.loc[
            patient_ig["landmark_month"].eq(TARGET_LANDMARK_MONTH)
        ].copy()
        if target.empty:
            continue

        profiles = choose_profiles(
            patient_ig=patient_ig,
            target=target,
            patient_local=patient_local,
            global_index=global_index,
            fold_id=fold_id,
            patient_mask=patient_mask,
            relaxed=relaxed,
        )
        if profiles is None:
            continue

        A_variables = profiles["A_variables"]
        B_variables = profiles["B_variables"]

        # Risk trajectory
        patient_risk = (
            risk_table.loc[risk_table["patient_local"].eq(patient_local)]
            .sort_values("landmark_month")
            .copy()
        )
        if len(patient_risk) != len(LANDMARK_MONTHS):
            continue

        yearly_risk = patient_risk["predicted_5y_risk"].to_numpy(dtype=float)
        yearly_diff = np.diff(yearly_risk)
        yearly_diff_pp = 100.0 * yearly_diff
        max_risk_jump_pp = float(np.max(np.abs(yearly_diff_pp)))
        terminal_jump_pp = float(abs(yearly_diff_pp[-1]))
        upward_steps = int(np.sum(yearly_diff_pp > 0.5))
        downward_steps = int(np.sum(yearly_diff_pp < -0.5))
        total_risk_gain_pp = float(100.0 * (yearly_risk[-1] - yearly_risk[0]))
        positive_gain_pp = float(np.sum(np.maximum(yearly_diff_pp, 0.0)))
        largest_jump_share = float(
            max_risk_jump_pp / max(positive_gain_pp, 1e-6)
        )

        # v16 gradualness score: reward several upward steps and penalize a single
        # dominant jump. This score is used after the hard filters below.
        risk_gradual_score = float(
            0.45 * min(upward_steps / 4.0, 1.0)
            + 0.30 * (1.0 / (1.0 + max_risk_jump_pp / 15.0))
            + 0.25 * max(0.0, 1.0 - min(largest_jump_share, 1.0))
        )

        # v15/v16 late-risk trajectory audit at year 3, year 4 and year 5.
        risk_y0_pct = 100.0 * float(yearly_risk[0])
        risk_y1_pct = 100.0 * float(yearly_risk[1])
        risk_y2_pct = 100.0 * float(yearly_risk[2])
        risk_y3_pct = 100.0 * float(yearly_risk[3])
        risk_y4_pct = 100.0 * float(yearly_risk[4])
        risk_y5_pct = 100.0 * float(yearly_risk[5])
        late_drop_4_to_5_pp = max(0.0, risk_y4_pct - risk_y5_pct)
        late_gain_3_to_5_pp = risk_y5_pct - risk_y3_pct
        late_gain_2_to_5_pp = risk_y5_pct - risk_y2_pct
        late_risk_consistency_score = (
            1.0 / (1.0 + late_drop_4_to_5_pp / 5.0)
            + 0.35 * np.clip(late_gain_3_to_5_pp / 15.0, -1.0, 1.0)
        )

        risk_percentile = float(
            risk_5_all.loc[
                risk_5_all["patient_local"].eq(patient_local),
                "risk_percentile",
            ].iloc[0]
        )
        risk_target_score = max(
            0.0,
            1.0 - abs(risk_percentile - RISK_TARGET_PERCENTILE) / RISK_TARGET_BANDWIDTH,
        )
        risk_smooth_score = 1.0 / (1.0 + max_risk_jump_pp / 8.0)

        # Year-5 importance
        imp_table = year5_importance_table(target)
        abs_lookup = dict(zip(imp_table["clinical_variable"], imp_table["absolute_ig"]))
        pct_lookup = dict(zip(imp_table["clinical_variable"], imp_table["pct_rank"]))
        rank_lookup = dict(zip(imp_table["clinical_variable"], imp_table["rank"]))

        total_abs = float(imp_table["absolute_ig"].sum())
        A_share = (
            float(sum(abs_lookup.get(v, 0.0) for v in A_variables)) / total_abs
            if total_abs > 0
            else 0.0
        )
        B_share = (
            float(sum(abs_lookup.get(v, 0.0) for v in B_variables)) / total_abs
            if total_abs > 0
            else 0.0
        )

        # Year-5 eGFR contribution audit.
        egfr_row = aggregate_target_feature(target, "eGFR")
        if egfr_row is None:
            continue
        eGFR_signed_pp = 100.0 * float(egfr_row["signed_ig"])
        eGFR_abs_pp = abs(eGFR_signed_pp)
        eGFR_abs_share = (
            float(egfr_row["absolute_ig"]) / total_abs
            if total_abs > 0
            else 1.0
        )
        eGFR_is_top1 = int(rank_lookup.get("eGFR", 999) == 1)

        # v16 multi-factor explanation audit across ALL year-5 variables.
        imp_for_balance = imp_table.copy()
        imp_for_balance["abs_pp"] = 100.0 * imp_for_balance["absolute_ig"].astype(float)
        imp_for_balance["signed_pp"] = 100.0 * imp_for_balance["signed_ig"].astype(float)
        imp_for_balance = imp_for_balance.sort_values("abs_pp", ascending=False).reset_index(drop=True)
        max_feature_name = str(imp_for_balance.iloc[0]["clinical_variable"])
        max_feature_abs_pp = float(imp_for_balance.iloc[0]["abs_pp"])
        max_feature_abs_share = (
            float(imp_for_balance.iloc[0]["absolute_ig"]) / total_abs
            if total_abs > 0 else 1.0
        )
        top3_abs_share = (
            float(imp_for_balance.head(3)["absolute_ig"].sum()) / total_abs
            if total_abs > 0 else 1.0
        )
        material_terms = imp_for_balance.loc[
            imp_for_balance["abs_pp"] >= MULTIFACTOR_MIN_ABS_TERM_PP
        ].copy()
        n_material_positive = int((material_terms["signed_pp"] > 0).sum())
        n_material_negative = int((material_terms["signed_pp"] < 0).sum())
        multi_factor_balance_score = float(
            max(0.0, 1.0 - max_feature_abs_share)
            * max(0.0, 1.0 - top3_abs_share)
        )

        # Keep eGFR clinically relevant but not completely negligible.
        if eGFR_abs_pp < EGFR_MIN_ABS_PP:
            continue

        anchor_pct = np.array(
            [float(pct_lookup.get(v, 0.0)) for v in ANCHOR_FEATURES],
            dtype=float,
        )
        anchor_relevance = float(anchor_pct.mean())

        comorbidity_pct = float(pct_lookup.get(profiles["comorbidity"], 0.0))
        lipid_pct = float(
            np.mean([pct_lookup.get(v, 0.0) for v in profiles["lipids"]])
        )
        art_pct = float(
            np.mean(
                [
                    pct_lookup.get(profiles["current_art"], 0.0),
                    pct_lookup.get(profiles["cumulative_art"], 0.0),
                ]
            )
        )

        # Panel A display quality is computed on the actual fold-standardized
        # values that will be plotted, not on within-patient re-standardization.
        panelA_visual_score = float(profiles["A_visual_score"])
        panelA_joint_score = float(profiles["A_joint_score"])
        A_min_observed = int(profiles["A_min_observed"])
        A_median_observed_z_range = float(profiles["A_median_observed_z_range"])
        A_median_all_z_range = float(profiles["A_median_all_z_range"])
        A_median_temporal_sd = float(profiles["A_median_temporal_sd"])
        A_n_dynamic = int(profiles["A_n_dynamic"])
        A_n_early_dynamic = int(profiles["A_n_early_dynamic"])
        A_n_late_dynamic = int(profiles["A_n_late_dynamic"])
        A_n_sustained_dynamic = int(profiles["A_n_sustained_dynamic"])
        A_median_early_z_range = float(profiles["A_median_early_z_range"])
        A_median_late_z_range = float(profiles["A_median_late_z_range"])
        A_mean_obs_fraction = float(profiles["A_mean_obs_fraction"])
        A_max_abs_z = float(profiles["A_max_abs_z"])
        A_median_plateau_fraction = float(profiles["A_median_plateau_fraction"])
        A_mean_plateau_fraction = float(profiles["A_mean_plateau_fraction"])
        A_max_plateau_fraction = float(profiles["A_max_plateau_fraction"])
        A_mean_longest_plateau_fraction = float(profiles["A_mean_longest_plateau_fraction"])
        A_n_low_plateau = int(profiles["A_n_low_plateau"])
        A_n_lipids = int(profiles["A_n_lipids"])

        # Hard guards: require sustained movement, not just early movement followed
        # by a long plateau, and reject extreme standardized trajectories.
        if relaxed:
            max_regimen_changes = 4
            if A_median_all_z_range < A_RELAXED_MEDIAN_RANGE_MIN:
                continue
            if A_median_temporal_sd < A_RELAXED_MEDIAN_SD_MIN:
                continue
            if A_n_dynamic < A_RELAXED_MIN_DYNAMIC_VARS:
                continue
            if A_n_early_dynamic < A_RELAXED_MIN_EARLY_DYNAMIC_VARS:
                continue
            if A_n_late_dynamic < A_RELAXED_MIN_LATE_DYNAMIC_VARS:
                continue
            if A_n_sustained_dynamic < A_RELAXED_MIN_SUSTAINED_DYNAMIC_VARS:
                continue
            if A_mean_obs_fraction < A_RELAXED_MEAN_OBS_FRACTION_MIN:
                continue
            if A_max_abs_z > A_RELAXED_MAX_ABS_Z:
                continue
            if A_median_plateau_fraction > A_RELAXED_MEDIAN_PLATEAU_FRACTION_MAX:
                continue
            if A_mean_plateau_fraction > A_RELAXED_MEAN_PLATEAU_FRACTION_MAX:
                continue
            if A_n_low_plateau < A_RELAXED_MIN_LOW_PLATEAU_VARS:
                continue
            if A_max_plateau_fraction > A_RELAXED_MAX_SINGLE_PLATEAU_FRACTION:
                continue
            if max_feature_abs_pp > MULTIFACTOR_MODERATE_MAX_SINGLE_ABS_PP:
                continue
            if max_feature_abs_share > MULTIFACTOR_MODERATE_MAX_SINGLE_SHARE:
                continue
            if top3_abs_share > MULTIFACTOR_MODERATE_MAX_TOP3_SHARE:
                continue
        else:
            max_regimen_changes = 3
            if A_median_all_z_range < A_STRICT_MEDIAN_RANGE_MIN:
                continue
            if A_median_temporal_sd < A_STRICT_MEDIAN_SD_MIN:
                continue
            if A_n_dynamic < A_STRICT_MIN_DYNAMIC_VARS:
                continue
            if A_n_early_dynamic < A_STRICT_MIN_EARLY_DYNAMIC_VARS:
                continue
            if A_n_late_dynamic < A_STRICT_MIN_LATE_DYNAMIC_VARS:
                continue
            if A_n_sustained_dynamic < A_STRICT_MIN_SUSTAINED_DYNAMIC_VARS:
                continue
            if A_mean_obs_fraction < A_STRICT_MEAN_OBS_FRACTION_MIN:
                continue
            if A_max_abs_z > A_STRICT_MAX_ABS_Z:
                continue
            if A_median_plateau_fraction > A_STRICT_MEDIAN_PLATEAU_FRACTION_MAX:
                continue
            if A_mean_plateau_fraction > A_STRICT_MEAN_PLATEAU_FRACTION_MAX:
                continue
            if A_n_low_plateau < A_STRICT_MIN_LOW_PLATEAU_VARS:
                continue
            if A_max_plateau_fraction > A_STRICT_MAX_SINGLE_PLATEAU_FRACTION:
                continue
            if max_feature_abs_pp > MULTIFACTOR_STRICT_MAX_SINGLE_ABS_PP:
                continue
            if max_feature_abs_share > MULTIFACTOR_STRICT_MAX_SINGLE_SHARE:
                continue
            if top3_abs_share > MULTIFACTOR_STRICT_MAX_TOP3_SHARE:
                continue

        if n_material_positive < MULTIFACTOR_MIN_POSITIVE_TERMS:
            continue
        if n_material_negative < MULTIFACTOR_MIN_NEGATIVE_TERMS:
            continue

        regimen_seq, regimen_changes = regimen_sequence_and_changes(
            patient_local=patient_local,
            fold_id=fold_id,
            patient_mask=patient_mask,
        )
        if regimen_changes > max_regimen_changes:
            continue
        regimen_stability_score = 1.0 / (1.0 + 0.75 * regimen_changes)

        active_art_consistency = (
            1.0 if profiles["active_art_count_year5"] == 1 else 0.60
        )

        # Prefer a force plot with contributions in both directions.
        b_signed = []
        for v in B_variables:
            row = aggregate_target_feature(target, v)
            if row is not None:
                b_signed.append(float(row["signed_ig"]))
        n_pos = int(sum(x > 0 for x in b_signed))
        n_neg = int(sum(x < 0 for x in b_signed))
        direction_balance = min(n_pos, n_neg) / max(1.0, TOP_B_LABEL_N / 2.0)

        # No outcome label is used here.
        # High-risk Figure-3 score: risk level is now an explicit major component,
        # while the same clinical coherence and display-quality requirements remain.
        high_risk_strength = float(
            np.clip(
                (risk_percentile - RISK_QUANTILE_LOW)
                / max(RISK_QUANTILE_HIGH - RISK_QUANTILE_LOW, 1e-6),
                0.0,
                1.0,
            )
        )

        composite_score = (
            1.10 * A_share
            + 1.25 * B_share
            + 1.30 * anchor_relevance
            + 0.55 * comorbidity_pct
            + 0.60 * lipid_pct
            + 0.70 * art_pct
            + 1.20 * risk_target_score
            + 1.75 * high_risk_strength
            + 0.45 * risk_smooth_score
            + 1.10 * panelA_visual_score
            + 0.65 * panelA_joint_score
            + 0.85 * min(A_median_all_z_range / 2.0, 1.25)
            + 0.75 * min(A_median_temporal_sd / 0.75, 1.25)
            + 0.10 * A_n_dynamic
            + 0.18 * A_n_sustained_dynamic
            + 0.10 * min(A_median_early_z_range / 1.0, 1.25)
            + 0.16 * min(A_median_late_z_range / 1.0, 1.25)
            + 0.65 * multi_factor_balance_score
            + 0.30 * direction_balance
            + 0.35 * regimen_stability_score
            + 0.15 * active_art_consistency
            - EGFR_PENALTY_WEIGHT_SHARE * eGFR_abs_share
            - EGFR_PENALTY_WEIGHT_ABS_PP * eGFR_abs_pp
            - 1.80 * max_feature_abs_share
            - 0.025 * max_feature_abs_pp
            - 0.70 * top3_abs_share
            - 0.25 * eGFR_is_top1
            + 0.90 * late_risk_consistency_score
            + 1.10 * risk_gradual_score
            + 0.80 * max(0.0, 1.0 - A_mean_plateau_fraction)
            + 0.08 * A_n_low_plateau
            - 0.80 * A_median_plateau_fraction
            - 0.55 * A_mean_longest_plateau_fraction
            + LATE_RISK_GAIN_REWARD_WEIGHT * max(late_gain_3_to_5_pp, 0.0)
            + LATE_RISK_Y5_WEIGHT * risk_y5_pct
            - LATE_RISK_DROP_PENALTY_WEIGHT * late_drop_4_to_5_pp
            - 0.035 * max_risk_jump_pp
            - 0.025 * terminal_jump_pp
        )

        rows.append(
            {
                "patient_local": patient_local,
                "year5_risk_pct": 100.0 * float(yearly_risk[-1]),
                "risk_percentile": risk_percentile,
                "year0_risk_pct": 100.0 * float(yearly_risk[0]),
                "risk_delta_0_to_5_pp": total_risk_gain_pp,
                "max_annual_risk_jump_pp": max_risk_jump_pp,
                "terminal_risk_jump_pp": terminal_jump_pp,
                "risk_upward_steps": upward_steps,
                "risk_downward_steps": downward_steps,
                "risk_positive_gain_pp": positive_gain_pp,
                "largest_jump_share_of_positive_gain": largest_jump_share,
                "risk_gradual_score": risk_gradual_score,
                "risk_y0_pct": risk_y0_pct,
                "risk_y1_pct": risk_y1_pct,
                "risk_y2_pct": risk_y2_pct,
                "risk_y3_pct": risk_y3_pct,
                "risk_y4_pct": risk_y4_pct,
                "risk_y5_pct": risk_y5_pct,
                "late_drop_4_to_5_pp": late_drop_4_to_5_pp,
                "late_gain_3_to_5_pp": late_gain_3_to_5_pp,
                "late_gain_2_to_5_pp": late_gain_2_to_5_pp,
                "late_risk_consistency_score": late_risk_consistency_score,
                "high_risk_strength": high_risk_strength,
                "eGFR_signed_pp": eGFR_signed_pp,
                "eGFR_abs_pp": eGFR_abs_pp,
                "eGFR_abs_share": eGFR_abs_share,
                "eGFR_is_top1": eGFR_is_top1,
                "max_feature_name": max_feature_name,
                "max_feature_abs_pp": max_feature_abs_pp,
                "max_feature_abs_share": max_feature_abs_share,
                "top3_abs_share": top3_abs_share,
                "n_material_positive": n_material_positive,
                "n_material_negative": n_material_negative,
                "multi_factor_balance_score": multi_factor_balance_score,
                "A_continuous_importance_share": A_share,
                "B_labelled_importance_share": B_share,
                "anchor_relevance_score": anchor_relevance,
                "eGFR_rank": int(rank_lookup.get("eGFR", 999)),
                "HIVRNA_rank": int(rank_lookup.get("HIVRNA_log10", 999)),
                "CD4_rank": int(rank_lookup.get("CD4", 999)),
                "comorbidity": profiles["comorbidity"],
                "lipid_1": profiles["lipids"][0],
                "lipid_2": (
                    profiles["lipids"][1] if len(profiles["lipids"]) > 1 else ""
                ),
                "current_art": profiles["current_art"],
                "cumulative_art": profiles["cumulative_art"],
                "regimen_changes_0_to_5": regimen_changes,
                "regimen_sequence": " | ".join(
                    feature_display(x) if x is not None else "NA"
                    for x in regimen_seq
                ),
                "panelA_visual_score": panelA_visual_score,
                "panelA_joint_score": panelA_joint_score,
                "A_min_observed": A_min_observed,
                "A_median_observed_z_range": A_median_observed_z_range,
                "A_median_all_z_range": A_median_all_z_range,
                "A_median_temporal_sd": A_median_temporal_sd,
                "A_n_dynamic": A_n_dynamic,
                "A_n_early_dynamic": A_n_early_dynamic,
                "A_n_late_dynamic": A_n_late_dynamic,
                "A_n_sustained_dynamic": A_n_sustained_dynamic,
                "A_median_early_z_range": A_median_early_z_range,
                "A_median_late_z_range": A_median_late_z_range,
                "A_mean_obs_fraction": A_mean_obs_fraction,
                "A_max_abs_z": A_max_abs_z,
                "A_median_plateau_fraction": A_median_plateau_fraction,
                "A_mean_plateau_fraction": A_mean_plateau_fraction,
                "A_max_plateau_fraction": A_max_plateau_fraction,
                "A_mean_longest_plateau_fraction": A_mean_longest_plateau_fraction,
                "A_n_low_plateau": A_n_low_plateau,
                "A_n_lipids": A_n_lipids,
                "B_direction_balance": direction_balance,
                "A_variables": " | ".join(A_variables),
                "A_display_names": " | ".join(feature_display(v) for v in A_variables),
                "B_variables": " | ".join(B_variables),
                "B_display_names": " | ".join(feature_display(v) for v in B_variables),
                "composite_score": composite_score,
                "selection_relaxed": bool(relaxed),
            }
        )

    return pd.DataFrame(rows)


candidate_df = evaluate_candidates(candidate_ids, relaxed=False)

if candidate_df.empty:
    print(
        "\n[Selection] No patient passed the primary clinical/measurement constraints. "
        "Running one predefined relaxed pass."
    )
    relaxed_ids = risk_5.loc[
        risk_5["risk_percentile"].between(
            RELAXED_RISK_QUANTILE_LOW,
            RELAXED_RISK_QUANTILE_HIGH,
            inclusive="both",
        ),
        "patient_local",
    ].astype(int).tolist()
    candidate_df = evaluate_candidates(relaxed_ids, relaxed=True)

if candidate_df.empty:
    raise ValueError(
        "No candidate could be selected even after the predefined relaxed pass. "
        "Please inspect formal IG coverage and year-5 current_model_value fields."
    )

candidate_df = candidate_df.sort_values(
    [
        "risk_percentile",
        "eGFR_abs_share",
        "eGFR_abs_pp",
        "composite_score",
        "panelA_visual_score",
    ],
    ascending=[False, True, True, False, False],
).reset_index(drop=True)

# ------------------------------------------------------------------
# v16 final selection: high risk + multi-factor balance + sustained dynamics
# + low plateau burden + gradual risk increase.
# ------------------------------------------------------------------
high_risk_df = candidate_df.loc[
    candidate_df["risk_percentile"] >= RISK_QUANTILE_LOW
].copy()
if high_risk_df.empty:
    high_risk_df = candidate_df.copy()

common_strict_mask = (
    (high_risk_df["eGFR_abs_pp"] <= EGFR_STRICT_MAX_ABS_PP)
    & (high_risk_df["eGFR_abs_share"] <= EGFR_STRICT_MAX_SHARE)
    & (high_risk_df["eGFR_is_top1"] == 0)
    & (high_risk_df["max_feature_abs_pp"] <= MULTIFACTOR_STRICT_MAX_SINGLE_ABS_PP)
    & (high_risk_df["max_feature_abs_share"] <= MULTIFACTOR_STRICT_MAX_SINGLE_SHARE)
    & (high_risk_df["top3_abs_share"] <= MULTIFACTOR_STRICT_MAX_TOP3_SHARE)
    & (high_risk_df["A_n_early_dynamic"] >= A_STRICT_MIN_EARLY_DYNAMIC_VARS)
    & (high_risk_df["A_n_late_dynamic"] >= A_STRICT_MIN_LATE_DYNAMIC_VARS)
    & (high_risk_df["A_n_sustained_dynamic"] >= A_STRICT_MIN_SUSTAINED_DYNAMIC_VARS)
    & (high_risk_df["A_mean_obs_fraction"] >= A_STRICT_MEAN_OBS_FRACTION_MIN)
    & (high_risk_df["A_max_abs_z"] <= A_STRICT_MAX_ABS_Z)
    & (high_risk_df["A_median_plateau_fraction"] <= A_STRICT_MEDIAN_PLATEAU_FRACTION_MAX)
    & (high_risk_df["A_mean_plateau_fraction"] <= A_STRICT_MEAN_PLATEAU_FRACTION_MAX)
    & (high_risk_df["A_n_low_plateau"] >= A_STRICT_MIN_LOW_PLATEAU_VARS)
    & (high_risk_df["A_max_plateau_fraction"] <= A_STRICT_MAX_SINGLE_PLATEAU_FRACTION)
)

strict_pool = high_risk_df.loc[
    common_strict_mask
    & (high_risk_df["risk_y5_pct"] >= RISK_STRICT_MIN_Y5_PCT)
    & (high_risk_df["max_annual_risk_jump_pp"] <= RISK_STRICT_MAX_ANNUAL_JUMP_PP)
    & (high_risk_df["terminal_risk_jump_pp"] <= RISK_STRICT_MAX_TERMINAL_JUMP_PP)
    & (high_risk_df["risk_upward_steps"] >= RISK_STRICT_MIN_UPWARD_STEPS)
    & (high_risk_df["risk_delta_0_to_5_pp"] >= RISK_STRICT_MIN_TOTAL_GAIN_PP)
    & (high_risk_df["late_drop_4_to_5_pp"] <= LATE_RISK_STRICT_MAX_DROP_4_TO_5_PP)
    & (high_risk_df["late_gain_3_to_5_pp"] >= LATE_RISK_MIN_GAIN_Y3_TO_Y5_STRICT_PP)
].copy()

common_moderate_mask = (
    (high_risk_df["eGFR_abs_pp"] <= EGFR_MODERATE_MAX_ABS_PP)
    & (high_risk_df["eGFR_abs_share"] <= EGFR_MODERATE_MAX_SHARE)
    & (high_risk_df["max_feature_abs_pp"] <= MULTIFACTOR_MODERATE_MAX_SINGLE_ABS_PP)
    & (high_risk_df["max_feature_abs_share"] <= MULTIFACTOR_MODERATE_MAX_SINGLE_SHARE)
    & (high_risk_df["top3_abs_share"] <= MULTIFACTOR_MODERATE_MAX_TOP3_SHARE)
    & (high_risk_df["A_n_early_dynamic"] >= A_RELAXED_MIN_EARLY_DYNAMIC_VARS)
    & (high_risk_df["A_n_late_dynamic"] >= A_RELAXED_MIN_LATE_DYNAMIC_VARS)
    & (high_risk_df["A_n_sustained_dynamic"] >= A_RELAXED_MIN_SUSTAINED_DYNAMIC_VARS)
    & (high_risk_df["A_mean_obs_fraction"] >= A_RELAXED_MEAN_OBS_FRACTION_MIN)
    & (high_risk_df["A_max_abs_z"] <= A_RELAXED_MAX_ABS_Z)
    & (high_risk_df["A_median_plateau_fraction"] <= A_RELAXED_MEDIAN_PLATEAU_FRACTION_MAX)
    & (high_risk_df["A_mean_plateau_fraction"] <= A_RELAXED_MEAN_PLATEAU_FRACTION_MAX)
    & (high_risk_df["A_n_low_plateau"] >= A_RELAXED_MIN_LOW_PLATEAU_VARS)
    & (high_risk_df["A_max_plateau_fraction"] <= A_RELAXED_MAX_SINGLE_PLATEAU_FRACTION)
)

moderate_pool = high_risk_df.loc[
    common_moderate_mask
    & (high_risk_df["risk_y5_pct"] >= RISK_MODERATE_MIN_Y5_PCT)
    & (high_risk_df["max_annual_risk_jump_pp"] <= RISK_MODERATE_MAX_ANNUAL_JUMP_PP)
    & (high_risk_df["terminal_risk_jump_pp"] <= RISK_MODERATE_MAX_TERMINAL_JUMP_PP)
    & (high_risk_df["risk_upward_steps"] >= RISK_MODERATE_MIN_UPWARD_STEPS)
    & (high_risk_df["risk_delta_0_to_5_pp"] >= RISK_MODERATE_MIN_TOTAL_GAIN_PP)
    & (high_risk_df["late_drop_4_to_5_pp"] <= LATE_RISK_MODERATE_MAX_DROP_4_TO_5_PP)
    & (high_risk_df["late_gain_3_to_5_pp"] >= LATE_RISK_MIN_GAIN_Y3_TO_Y5_MODERATE_PP)
].copy()

fallback_pool = high_risk_df.loc[
    common_moderate_mask
    & (high_risk_df["risk_y5_pct"] >= RISK_FALLBACK_MIN_Y5_PCT)
    & (high_risk_df["max_annual_risk_jump_pp"] <= RISK_FALLBACK_MAX_ANNUAL_JUMP_PP)
    & (high_risk_df["terminal_risk_jump_pp"] <= RISK_FALLBACK_MAX_TERMINAL_JUMP_PP)
    & (high_risk_df["risk_upward_steps"] >= RISK_FALLBACK_MIN_UPWARD_STEPS)
    & (high_risk_df["risk_delta_0_to_5_pp"] >= RISK_FALLBACK_MIN_TOTAL_GAIN_PP)
    & (high_risk_df["late_drop_4_to_5_pp"] <= LATE_RISK_FALLBACK_MAX_DROP_4_TO_5_PP)
].copy()

if not strict_pool.empty:
    shortlist = strict_pool.copy()
    selection_mode = "strict high-risk + gradual risk + low plateau + multifactor sustained dynamics"
elif not moderate_pool.empty:
    shortlist = moderate_pool.copy()
    selection_mode = "moderate high-risk + gradual risk + controlled plateau + multifactor dynamics"
elif not fallback_pool.empty:
    shortlist = fallback_pool.copy()
    selection_mode = "fallback high-risk + limited jump + controlled plateau"
else:
    shortlist = high_risk_df.copy()
    selection_mode = "best available high-risk case with gradual-risk and plateau penalties"

shortlist = shortlist.sort_values(
    [
        "max_annual_risk_jump_pp",
        "terminal_risk_jump_pp",
        "A_median_plateau_fraction",
        "A_mean_longest_plateau_fraction",
        "A_n_sustained_dynamic",
        "multi_factor_balance_score",
        "risk_percentile",
        "max_feature_abs_share",
    ],
    ascending=[True, True, True, True, False, False, False, True],
).head(min(FINAL_SHORTLIST_N, len(shortlist))).copy()

shortlist["final_dynamic_figure_score"] = (
    2.00 * shortlist["risk_percentile"]
    + 1.10 * shortlist["panelA_visual_score"]
    + 0.70 * shortlist["panelA_joint_score"]
    + 0.18 * shortlist["A_n_sustained_dynamic"]
    + 0.10 * shortlist["A_n_late_dynamic"]
    + 0.08 * shortlist["A_n_early_dynamic"]
    + 0.10 * shortlist["A_n_low_plateau"]
    + 0.45 * np.minimum(shortlist["A_median_late_z_range"] / 1.0, 1.25)
    + 0.34 * np.minimum(shortlist["A_median_early_z_range"] / 1.0, 1.25)
    + 1.10 * shortlist["multi_factor_balance_score"]
    + 0.25 * shortlist["B_labelled_importance_share"]
    + 0.14 * shortlist["A_continuous_importance_share"]
    + 1.10 * shortlist["late_risk_consistency_score"]
    + 1.25 * shortlist["risk_gradual_score"]
    + 0.022 * shortlist["risk_y5_pct"]
    + 0.018 * np.maximum(shortlist["late_gain_3_to_5_pp"], 0.0)
    - 0.055 * shortlist["max_annual_risk_jump_pp"]
    - 0.040 * shortlist["terminal_risk_jump_pp"]
    - 1.15 * shortlist["A_median_plateau_fraction"]
    - 0.70 * shortlist["A_mean_plateau_fraction"]
    - 0.60 * shortlist["A_mean_longest_plateau_fraction"]
    - 1.90 * shortlist["max_feature_abs_share"]
    - 0.025 * shortlist["max_feature_abs_pp"]
    - 1.00 * shortlist["top3_abs_share"]
    - 1.80 * shortlist["eGFR_abs_share"]
    - 0.016 * shortlist["eGFR_abs_pp"]
    - 0.15 * shortlist["eGFR_is_top1"]
    - 0.24 * np.maximum(shortlist["A_max_abs_z"] - 4.0, 0.0)
)

selected_row = shortlist.sort_values(
    [
        "final_dynamic_figure_score",
        "max_annual_risk_jump_pp",
        "terminal_risk_jump_pp",
        "A_median_plateau_fraction",
        "A_n_sustained_dynamic",
        "multi_factor_balance_score",
        "risk_percentile",
    ],
    ascending=[False, True, True, True, False, False, False],
).reset_index(drop=True).iloc[0]

print("\n" + "=" * 100)
print("v16 selection mode:", selection_mode)
print("Selected candidate risk / multifactor / sustained-dynamics / plateau / gradual-risk audit")
print("=" * 100)
print(f"patient_local        : {int(selected_row['patient_local'])}")
print(f"year-5 risk          : {float(selected_row['year5_risk_pct']):.2f}%")
print(f"risk percentile      : {100*float(selected_row['risk_percentile']):.1f}th")
print(f"risk y3/y4/y5        : {float(selected_row['risk_y3_pct']):.1f}% / {float(selected_row['risk_y4_pct']):.1f}% / {float(selected_row['risk_y5_pct']):.1f}%")
print(f"late drop y4->y5     : {float(selected_row['late_drop_4_to_5_pp']):.2f} pp")
print(f"late gain y3->y5     : {float(selected_row['late_gain_3_to_5_pp']):+.2f} pp")
print(f"eGFR |IG|            : {float(selected_row['eGFR_abs_pp']):.2f} pp")
print(f"eGFR |IG| share      : {100*float(selected_row['eGFR_abs_share']):.1f}%")
print(f"largest IG feature   : {selected_row['max_feature_name']}")
print(f"largest |IG|         : {float(selected_row['max_feature_abs_pp']):.2f} pp")
print(f"largest |IG| share   : {100*float(selected_row['max_feature_abs_share']):.1f}%")
print(f"top-3 |IG| share     : {100*float(selected_row['top3_abs_share']):.1f}%")
print(f"material +/- terms   : {int(selected_row['n_material_positive'])}/{int(selected_row['n_material_negative'])}")
print(f"A early dynamic vars : {int(selected_row['A_n_early_dynamic'])}/{TOP_A_N}")
print(f"A late dynamic vars  : {int(selected_row['A_n_late_dynamic'])}/{TOP_A_N}")
print(f"A sustained vars     : {int(selected_row['A_n_sustained_dynamic'])}/{TOP_A_N}")
print(f"A mean obs fraction  : {100*float(selected_row['A_mean_obs_fraction']):.1f}%")
print(f"A max |Z|            : {float(selected_row['A_max_abs_z']):.2f}")
print(f"A median plateau     : {100*float(selected_row['A_median_plateau_fraction']):.1f}%")
print(f"A mean plateau       : {100*float(selected_row['A_mean_plateau_fraction']):.1f}%")
print(f"A low-plateau vars   : {int(selected_row['A_n_low_plateau'])}/{TOP_A_N}")
print(f"A longest plateau avg: {100*float(selected_row['A_mean_longest_plateau_fraction']):.1f}%")
print(f"max annual risk jump : {float(selected_row['max_annual_risk_jump_pp']):.2f} pp")
print(f"terminal risk jump   : {float(selected_row['terminal_risk_jump_pp']):.2f} pp")
print(f"risk upward steps    : {int(selected_row['risk_upward_steps'])}/5")
print(f"risk gradual score   : {float(selected_row['risk_gradual_score']):.3f}")

# Compare against the retained v14 benchmark patient if available.
if benchmark_v14_patient_local is not None:
    benchmark_risk = risk_table.loc[
        risk_table["patient_local"].eq(benchmark_v14_patient_local)
    ].sort_values("landmark_month")
    if len(benchmark_risk) == len(LANDMARK_MONTHS):
        bvals = 100.0 * benchmark_risk["predicted_5y_risk"].to_numpy(dtype=float)
        benchmark_drop = max(0.0, float(bvals[4] - bvals[5]))
        benchmark_gain = float(bvals[5] - bvals[3])
        print("\nRetained v14 benchmark comparison:")
        print(f"benchmark patient    : {benchmark_v14_patient_local}")
        print(f"benchmark y3/y4/y5   : {bvals[3]:.1f}% / {bvals[4]:.1f}% / {bvals[5]:.1f}%")
        print(f"benchmark y4->y5 drop: {benchmark_drop:.2f} pp")
        print(f"benchmark y3->y5 gain: {benchmark_gain:+.2f} pp")
        print(f"new v16 y4->y5 drop  : {float(selected_row['late_drop_4_to_5_pp']):.2f} pp")
        print(f"new v16 y3->y5 gain  : {float(selected_row['late_gain_3_to_5_pp']):+.2f} pp")

candidate_df["final_dynamic_figure_score"] = np.nan
score_map = dict(
    zip(
        shortlist["patient_local"].astype(int),
        shortlist["final_dynamic_figure_score"].astype(float),
    )
)
candidate_df["final_dynamic_figure_score"] = candidate_df["patient_local"].map(score_map)

candidate_df["selected_for_figure"] = candidate_df["patient_local"].eq(
    int(selected_row["patient_local"])
)
candidate_df.to_csv(
    OUTPUT_CANDIDATE_TABLE,
    index=False,
    encoding="utf-8-sig",
)

PATIENT_LOCAL = int(selected_row["patient_local"])

print("\n" + "=" * 100)
print("Top high-risk gradual plateau-controlled candidates")
print("=" * 100)
print(
    candidate_df[
        [
            "patient_local",
            "year5_risk_pct",
            "risk_percentile",
            "high_risk_strength",
            "eGFR_signed_pp",
            "eGFR_abs_pp",
            "eGFR_abs_share",
            "eGFR_is_top1",
            "max_feature_name",
            "max_feature_abs_pp",
            "max_feature_abs_share",
            "top3_abs_share",
            "A_n_early_dynamic",
            "A_n_late_dynamic",
            "A_n_sustained_dynamic",
            "A_mean_obs_fraction",
            "A_median_plateau_fraction",
            "A_mean_plateau_fraction",
            "A_n_low_plateau",
            "max_annual_risk_jump_pp",
            "terminal_risk_jump_pp",
            "risk_upward_steps",
            "risk_gradual_score",
            "final_dynamic_figure_score",
            "A_continuous_importance_share",
            "B_labelled_importance_share",
            "eGFR_rank",
            "HIVRNA_rank",
            "CD4_rank",
            "comorbidity",
            "lipid_1",
            "lipid_2",
            "current_art",
            "regimen_changes_0_to_5",
            "A_min_observed",
            "A_median_observed_z_range",
            "A_median_all_z_range",
            "A_median_temporal_sd",
            "A_n_dynamic",
            "A_max_abs_z",
            "A_n_lipids",
            "panelA_visual_score",
            "panelA_joint_score",
            "composite_score",
            "selected_for_figure",
        ]
    ]
    .head(15)
    .to_string(index=False)
)

print("\nSelected patient_local:", PATIENT_LOCAL)


# ============================================================
# 7. Prepare selected patient
# ============================================================

patient_local = PATIENT_LOCAL
global_index = int(development_idx[patient_local])
fold_id = int(development_fold_id[patient_local])
patient_mask = np.asarray(row_mask[patient_local], dtype=bool)

patient_ig = ig.loc[ig["patient_local"].eq(patient_local)].copy()
target = patient_ig.loc[
    patient_ig["landmark_month"].eq(TARGET_LANDMARK_MONTH)
].copy()

profiles = choose_profiles(
    patient_ig=patient_ig,
    target=target,
    patient_local=patient_local,
    global_index=global_index,
    fold_id=fold_id,
    patient_mask=patient_mask,
    relaxed=bool(selected_row.get("selection_relaxed", False)),
)
if profiles is None:
    raise RuntimeError("Selected patient unexpectedly failed profile reconstruction.")

A_VARIABLES = profiles["A_variables"]
B_LABEL_VARIABLES = profiles["B_variables"]

risk_df = (
    risk_table.loc[
        risk_table["patient_local"].eq(patient_local),
        ["landmark_month", "predicted_5y_risk"],
    ]
    .drop_duplicates()
    .sort_values("landmark_month")
    .reset_index(drop=True)
)
risk_df["landmark_year"] = risk_df["landmark_month"] / 12.0
risk_df["risk_pct"] = 100.0 * risk_df["predicted_5y_risk"]

fold_dir = STEP4_DIR / f"fold_{fold_id}"
feature_names = (
    pd.read_csv(
        fold_dir / "feature_names.csv",
        encoding="utf-8-sig",
    )["feature_name"]
    .astype(str)
    .tolist()
)
feature_to_index = {name: i for i, name in enumerate(feature_names)}

X_development = np.load(fold_dir / "X_development.npy", mmap_mode="r")
preprocessor = joblib.load(fold_dir / "preprocessor.joblib")
scaler = preprocessor["scaler"]

print("\nSelected patient fold_id:", fold_id)
print("Selected patient development_global_index:", global_index)

print("\nPanel A continuous variables:")
for i, v in enumerate(A_VARIABLES, 1):
    print(f"{i:02d}. {v:<20s} -> {feature_display(v)}")

print("\nPanel B labelled variables:")
for i, v in enumerate(B_LABEL_VARIABLES, 1):
    print(f"{i:02d}. {v:<38s} -> {feature_display(v)}")


# ============================================================
# 8. Build Panel A continuous trajectories only
#
# IMPORTANT: the reference figure uses standardized feature values. Therefore
# Panel A plots the fold-standardized model input directly. We do NOT re-zscore
# each variable within this patient, because that can turn a nearly flat/imputed
# trajectory into an artificial two-level pattern.
# ============================================================

valid_steps = np.where(patient_mask)[0]
if len(valid_steps) == 0:
    raise ValueError("Selected patient has no valid historical rows.")

trajectory_rows = []

for variable in A_VARIABLES:
    if variable not in continuous_to_index:
        raise ValueError(
            f"Panel A variable is not in continuous_raw definition: {variable}"
        )
    if variable not in feature_to_index:
        raise ValueError(
            f"Panel A variable is not in fold feature_names: {variable}"
        )

    continuous_i = int(continuous_to_index[variable])
    feature_i = int(feature_to_index[variable])

    for step, month in enumerate(HISTORY_MONTHS):
        if not patient_mask[step]:
            trajectory_rows.append(
                {
                    "panel": "A_continuous_trajectory",
                    "patient_local": patient_local,
                    "clinical_variable": variable,
                    "display_name": feature_display(variable),
                    "month": int(month),
                    "year": float(month / 12.0),
                    "raw_value": np.nan,
                    "model_z": np.nan,
                    "plot_z": np.nan,
                    "observed": False,
                }
            )
            continue

        raw_value = float(continuous_raw[global_index, step, continuous_i])
        observed = np.isfinite(raw_value)
        model_z = float(X_development[patient_local, step, feature_i])

        trajectory_rows.append(
            {
                "panel": "A_continuous_trajectory",
                "patient_local": patient_local,
                "clinical_variable": variable,
                "display_name": feature_display(variable),
                "month": int(month),
                "year": float(month / 12.0),
                "raw_value": raw_value if observed else np.nan,
                "model_z": model_z,
                "plot_z": model_z,
                "observed": bool(observed),
            }
        )

trajectory_df = pd.DataFrame(trajectory_rows)

# Audit: no categorical or ART feature is permitted in Panel A.
for variable in A_VARIABLES:
    if variable in COMORBIDITY_FEATURES or variable in CURRENT_ART_FEATURES or variable in CUMULATIVE_ART_FEATURES:
        raise RuntimeError(f"Categorical/ART feature leaked into Panel A: {variable}")

# ============================================================
# 9. Panel B: COMPLETE current-step + earlier-history force decomposition
#
# Dynamic clinical variables:
#   current-step IG only, aligned with the current values used in labels.
#
# Earlier dynamic information:
#   retained as one net "Earlier longitudinal history" force term.
#
# Static/age variables:
#   retain their direct formal year-5 IG contributions.
#
# The complete force sum is audited against the original formal Step12A total.
# ============================================================

B_all, PANEL_B_CURRENT_AUDIT = build_current_step_force_terms(
    target=target,
    patient_local=patient_local,
    fold_id=fold_id,
    patient_mask=patient_mask,
)

B_all["ig_pp"] = 100.0 * B_all["signed_ig"]
B_all["abs_ig_pp"] = np.abs(B_all["ig_pp"])
B_all["display_name"] = B_all["clinical_variable"].map(feature_display)

print("\n" + "=" * 100)
print("Panel B current-step / earlier-history audit")
print("=" * 100)
for key, value in PANEL_B_CURRENT_AUDIT.items():
    print(f"{key:<38s}: {value}")

# ------------------------------------------------------------------
# Final label selection for Panel B
# ------------------------------------------------------------------
# The complete force ribbon still contains every current-step dynamic feature,
# the net Earlier longitudinal history term, and every direct static/age term.
# Only the text annotations are restricted to 10 major contributors.
# No variable class is excluded on clinical-interpretability grounds.
FINAL_B_LABEL_VARIABLES = (
    select_panel_b_label_variables(B_all)
)

B_all["labelled"] = B_all[
    "clinical_variable"
].isin(FINAL_B_LABEL_VARIABLES)

print("\nPanel B labelled major contributors:")
print(
    "  selection rule: no cherry-picking; force-include top 2 positive/red "
    "and top 1 negative/blue, then fill to 10 by overall |IG|."
)
for i, variable in enumerate(
    FINAL_B_LABEL_VARIABLES,
    1,
):
    row = B_all.loc[
        B_all["clinical_variable"].eq(
            variable
        )
    ].iloc[0]
    print(
        f"{i:02d}. "
        f"{feature_display(variable):<44s} "
        f"{float(row['ig_pp']):+7.3f} pp"
    )

# Mandatory-rank audit requested for the final figure.
_positive_ranked_audit = (
    B_all.loc[B_all["ig_pp"] > 0]
    .sort_values(
        ["abs_ig_pp", "clinical_variable"],
        ascending=[False, True],
    )
    .reset_index(drop=True)
)
_negative_ranked_audit = (
    B_all.loc[B_all["ig_pp"] < 0]
    .sort_values(
        ["abs_ig_pp", "clinical_variable"],
        ascending=[False, True],
    )
    .reset_index(drop=True)
)

print("\nMandatory dominant-segment label audit:")
for rank_idx in range(
    min(PANEL_B_REQUIRED_TOP_POSITIVE_N, len(_positive_ranked_audit))
):
    variable = str(
        _positive_ranked_audit.loc[
            rank_idx,
            "clinical_variable",
        ]
    )
    print(
        f"  Positive/red rank {rank_idx + 1}: "
        f"{feature_display(variable)} "
        f"({float(_positive_ranked_audit.loc[rank_idx, 'ig_pp']):+.3f} pp) "
        f"labelled={variable in FINAL_B_LABEL_VARIABLES}"
    )

if len(_negative_ranked_audit) > 0:
    variable = str(
        _negative_ranked_audit.loc[
            0,
            "clinical_variable",
        ]
    )
    print(
        f"  Negative/blue rank 1: "
        f"{feature_display(variable)} "
        f"({float(_negative_ranked_audit.loc[0, 'ig_pp']):+.3f} pp) "
        f"labelled={variable in FINAL_B_LABEL_VARIABLES}"
    )

# Hard fail if a mandatory visually dominant segment is ever omitted.
_required_dominant = set(
    _positive_ranked_audit.head(
        PANEL_B_REQUIRED_TOP_POSITIVE_N
    )["clinical_variable"].astype(str).tolist()
    + _negative_ranked_audit.head(
        PANEL_B_REQUIRED_TOP_NEGATIVE_N
    )["clinical_variable"].astype(str).tolist()
)
_missing_required = sorted(
    _required_dominant - set(FINAL_B_LABEL_VARIABLES)
)
if _missing_required:
    raise RuntimeError(
        "Mandatory dominant Panel-B segments were not labelled: "
        + ", ".join(_missing_required)
    )

reference_risk = float(target["reference_5y_risk"].iloc[0])
predicted_risk = float(target["predicted_5y_risk"].iloc[0])
base_pct = 100.0 * reference_risk
predicted_pct = 100.0 * predicted_risk

# Reference-style order: positive first, then negative; within each direction
# use contribution magnitude only. Label status does not alter bar geometry.
pos = B_all.loc[B_all["ig_pp"] > 0].sort_values(
    ["abs_ig_pp", "clinical_variable"],
    ascending=[False, True],
).copy()
neg = B_all.loc[B_all["ig_pp"] < 0].sort_values(
    ["abs_ig_pp", "clinical_variable"],
    ascending=[False, True],
).copy()
zero = B_all.loc[np.isclose(B_all["ig_pp"], 0.0)].copy()
B_plot = pd.concat([pos, neg, zero], ignore_index=True)

segments = []
current = base_pct
for row in B_plot.itertuples(index=False):
    delta = float(row.ig_pp)
    start = float(current)
    end = float(start + delta)
    segments.append(
        {
            "clinical_variable": row.clinical_variable,
            "display_name": row.display_name,
            "delta_pp": delta,
            "abs_delta_pp": abs(delta),
            "start": start,
            "end": end,
            "midpoint": (start + end) / 2.0,
            "labelled": bool(row.labelled),
        }
    )
    current = end

segment_df = pd.DataFrame(segments)
reconstructed_pct = float(segment_df["end"].iloc[-1])
reconstruction_residual = reconstructed_pct - predicted_pct

print("\n" + "=" * 100)
print("Panel B complete IG reconstruction audit")
print("=" * 100)
print(f"Reference/base risk : {base_pct:.4f}%")
print(f"Predicted risk      : {predicted_pct:.4f}%")
print(f"Reconstructed       : {reconstructed_pct:.4f}%")
print(f"Residual            : {reconstruction_residual:.8f} percentage points")

if abs(reconstruction_residual) > PANEL_B_RECONSTRUCTION_TOL_PP:
    warnings.warn(
        "IG completeness residual exceeds the predefined Figure-3 tolerance. "
        "The figure will still be produced; inspect the formal IG audit."
    )

# Save feature profiles.
A_profile = (
    patient_ig.groupby("clinical_variable", as_index=False)
    .agg(mean_abs_ig_across_landmarks=("absolute_ig", "mean"))
)
A_profile = A_profile.loc[
    A_profile["clinical_variable"].isin(A_VARIABLES)
].copy()
A_profile["display_name"] = A_profile["clinical_variable"].map(feature_display)
A_profile["panel_order"] = A_profile["clinical_variable"].map(
    {v: i + 1 for i, v in enumerate(A_VARIABLES)}
)
A_profile = A_profile.sort_values("panel_order")
# Add the actual displayability metrics used for patient/variable selection.
A_display_audit = profiles["A_display_table"].copy()
A_profile = A_profile.merge(
    A_display_audit,
    on="clinical_variable",
    how="left",
    validate="one_to_one",
)
A_profile.to_csv(OUTPUT_A_PROFILE, index=False, encoding="utf-8-sig")

B_profile = B_all.loc[
    B_all["clinical_variable"].isin(
        FINAL_B_LABEL_VARIABLES
    )
].copy()
B_profile["panel_order"] = (
    B_profile["clinical_variable"].map(
        {
            variable: i + 1
            for i, variable in enumerate(
                FINAL_B_LABEL_VARIABLES
            )
        }
    )
)
B_profile["label_selection_rule"] = (
    "no cherry-picking: top 2 positive + top 1 negative mandatory, then overall |IG| to 10"
)
B_profile["label_max_n"] = PANEL_B_TOP_LABEL_N
B_profile["required_top_positive_n"] = PANEL_B_REQUIRED_TOP_POSITIVE_N
B_profile["required_top_negative_n"] = PANEL_B_REQUIRED_TOP_NEGATIVE_N
B_profile = B_profile.sort_values(
    "panel_order"
)
B_profile.to_csv(
    OUTPUT_B_PROFILE,
    index=False,
    encoding="utf-8-sig",
)

PANEL_B_CURRENT_AUDIT_FILE = (
    OUTPUT_DIR / "Figure3_CurrentStep_History_FINAL_PanelB_audit.csv"
)
pd.DataFrame([PANEL_B_CURRENT_AUDIT]).to_csv(
    PANEL_B_CURRENT_AUDIT_FILE,
    index=False,
    encoding="utf-8-sig",
)


# ============================================================
# 10. Plot style
# ============================================================

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7.0,
        "axes.labelsize": 7.7,
        "axes.titlesize": 7.7,
        "xtick.labelsize": 6.9,
        "ytick.labelsize": 6.9,
        "legend.fontsize": 6.8,
        "axes.linewidth": 0.85,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "text.color": "black",
        "axes.labelcolor": "black",
        "xtick.color": "black",
        "ytick.color": "black",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

fig = plt.figure(figsize=(11.6, 7.1))
gs = fig.add_gridspec(
    2,
    1,
    height_ratios=[2.30, 1.28],
    hspace=0.47,
)


# ============================================================
# 11. Panel A -- CONTINUOUS VARIABLES ONLY
# ============================================================

ax = fig.add_subplot(gs[0, 0])
variable_legend_handles = []

for i, variable in enumerate(A_VARIABLES):
    sub = (
        trajectory_df.loc[
            trajectory_df["clinical_variable"].eq(variable)
        ]
        .sort_values("month")
        .copy()
    )

    x = sub["year"].to_numpy(dtype=float)
    y = sub["plot_z"].to_numpy(dtype=float)
    observed = sub["observed"].to_numpy(dtype=bool)

    line = ax.plot(
        x,
        y,
        linewidth=0.85,
        alpha=0.53,
        zorder=2,
    )[0]
    color = line.get_color()
    marker = MARKERS[i % len(MARKERS)]

    solid = observed & np.isfinite(y)
    hollow = (~observed) & np.isfinite(y)

    ax.scatter(
        x[solid],
        y[solid],
        marker=marker,
        s=31,
        color=color,
        edgecolor="black",
        linewidth=0.35,
        zorder=4,
    )

    ax.scatter(
        x[hollow],
        y[hollow],
        marker=marker,
        s=31,
        facecolors="white",
        edgecolors=color,
        linewidth=0.95,
        zorder=4,
    )

    variable_legend_handles.append(
        Line2D(
            [0],
            [0],
            linestyle="none",
            marker=marker,
            markersize=6.1,
            markerfacecolor=color,
            markeredgecolor="black",
            markeredgewidth=0.35,
            label=feature_display(variable),
        )
    )

ax.axhline(
    0.0,
    linestyle="--",
    linewidth=0.55,
    color="0.72",
    alpha=0.55,
    zorder=1,
)

finite_z = trajectory_df["plot_z"].to_numpy(dtype=float)
finite_z = finite_z[np.isfinite(finite_z)]
if len(finite_z) > 0:
    y_lim = max(
        2.5,
        float(np.ceil((np.max(np.abs(finite_z)) + 0.20) * 2.0) / 2.0),
    )
else:
    y_lim = 2.5

ax.set_ylim(-y_lim, y_lim)
ax.set_xlim(-0.05, 5.05)
ax.set_xticks(np.arange(0, 5.1, 1.0))
ax.set_xticks(np.arange(0.5, 5.0, 1.0), minor=True)
ax.set_xlabel("Time after ART initiation (years)")
ax.set_ylabel("Standardized value (Z-score)")

ax.grid(
    axis="y",
    linestyle="-",
    linewidth=0.35,
    color="0.88",
    alpha=0.90,
)
ax.tick_params(which="major", length=3.0, width=0.8)
ax.tick_params(which="minor", length=1.6, width=0.6)

# Risk axis
ax_risk = ax.twinx()
risk_line_handle = Line2D(
    [0],
    [0],
    color=RISK_LINE,
    linestyle="--",
    marker="^",
    markersize=6.2,
    linewidth=1.25,
    label="5y CKD risk",
)

ax_risk.plot(
    risk_df["landmark_year"],
    risk_df["risk_pct"],
    color=RISK_LINE,
    linestyle="--",
    marker="^",
    markersize=6.2,
    linewidth=1.25,
    zorder=6,
)

risk_x = risk_df["landmark_year"].to_numpy(dtype=float)
risk_y = risk_df["risk_pct"].to_numpy(dtype=float)

for j, (xv, yv) in enumerate(zip(risk_x, risk_y)):
    is_final = j == len(risk_x) - 1
    dx = 3 if j == 0 else (-3 if is_final else 0)
    ha = "left" if j == 0 else ("right" if is_final else "center")
    ax_risk.annotate(
        f"{yv:.1f}%",
        (xv, yv),
        xytext=(dx, 5),
        textcoords="offset points",
        ha=ha,
        va="bottom",
        fontsize=7.15 if is_final else 6.8,
        fontweight="bold" if is_final else "normal",
        color="0.22" if is_final else "0.30",
        clip_on=False,
    )

risk_upper = max(
    10,
    math.ceil(float(risk_df["risk_pct"].max()) * 1.18 / 5.0) * 5,
)
ax_risk.set_ylim(0, risk_upper)
ax_risk.set_ylabel("Subsequent 5-year CKD risk (%)", color=RISK_LINE)
ax_risk.tick_params(axis="y", colors=RISK_LINE, length=3.0, width=0.8)
ax_risk.spines["right"].set_color(RISK_LINE)

observed_handle = Line2D(
    [0], [0],
    linestyle="none",
    marker="o",
    markersize=6.4,
    markerfacecolor="0.35",
    markeredgecolor="black",
    markeredgewidth=0.35,
    label="Obs (solid)",
)
imputed_handle = Line2D(
    [0], [0],
    linestyle="none",
    marker="o",
    markersize=6.4,
    markerfacecolor="white",
    markeredgecolor="0.45",
    markeredgewidth=0.95,
    label="Imp (hollow)",
)

legend_handles = (
    [observed_handle, imputed_handle]
    + variable_legend_handles
    + [risk_line_handle]
)
ax.legend(
    legend_handles,
    [h.get_label() for h in legend_handles],
    loc="lower center",
    bbox_to_anchor=(0.5, 1.025),
    ncol=len(legend_handles),
    frameon=False,
    fontsize=5.35,
    handlelength=1.25,
    handletextpad=0.34,
    columnspacing=0.72,
    borderaxespad=0.0,
)

# Highlight year-5 explanation point.
x_highlight = float(risk_df["landmark_year"].iloc[-1])
y_highlight = float(risk_df["risk_pct"].iloc[-1])
ax_risk.add_patch(
    Ellipse(
        (x_highlight, y_highlight),
        width=0.28,
        height=max(2.0, risk_upper * 0.10),
        edgecolor=HIGHLIGHT_COLOR,
        facecolor="none",
        linestyle=(0, (4, 3)),
        linewidth=1.35,
        zorder=7,
    )
)

ax.text(
    -0.045,
    1.025,
    "a",
    transform=ax.transAxes,
    fontsize=12.5,
    fontweight="bold",
    ha="left",
    va="bottom",
)


# ============================================================
# 12. Panel B -- CURRENT-STEP + EARLIER-HISTORY COMPLETE LOCAL EXPLANATION
# ============================================================

ax2 = fig.add_subplot(gs[1, 0])
ax2.set_yticks([])

for side in ["left", "right", "bottom"]:
    ax2.spines[side].set_visible(False)
ax2.spines["top"].set_visible(True)
ax2.spines["top"].set_linewidth(1.15)
ax2.xaxis.set_ticks_position("top")
ax2.tick_params(
    axis="x",
    top=True,
    labeltop=True,
    bottom=False,
    labelbottom=False,
    length=3.2,
    width=0.9,
    pad=2.2,
)

# ------------------------------------------------------------
# Panel B: TRUE-SCALE current-step + earlier-history force-style explanation.
# IMPORTANT: unlike earlier display-only versions, the x-axis here is the
# true predicted-risk (%) scale. Therefore base value and f(x) are both shown
# at their real coordinates, matching the scientific meaning of the reference.
# ------------------------------------------------------------

pos_terms = B_all.loc[B_all["ig_pp"] > 0].copy().sort_values(
    ["abs_ig_pp", "clinical_variable"],
    ascending=[False, True],
)
neg_terms = B_all.loc[B_all["ig_pp"] < 0].copy().sort_values(
    ["abs_ig_pp", "clinical_variable"],
    ascending=[False, True],
)

pos_total = float(pos_terms["ig_pp"].sum())
neg_total = float(np.abs(neg_terms["ig_pp"]).sum())

red_left = predicted_pct - pos_total
red_right = predicted_pct
blue_left = predicted_pct
blue_right = predicted_pct + neg_total

xmin_raw = min(red_left, base_pct, predicted_pct)
xmax_raw = max(blue_right, base_pct, predicted_pct)
pad = max(2.5, 0.075 * max(xmax_raw - xmin_raw, 8.0))

# DO NOT clip xmin to zero. Integrated-gradient additive coordinates may extend
# below 0% or above 100% because intermediate cumulative attributions are not
# probabilities themselves. Clipping would hide part of the red ribbon and
# break the visual correspondence between feature width and IG contribution.
xmin = xmin_raw - pad
xmax = xmax_raw + pad
span = max(xmax - xmin, 8.0)

ax2.set_xlim(xmin, xmax)
ax2.set_ylim(-0.64, 1.12)

y_bar = 0.70
bar_height = 0.17

plot_records = []

# Positive terms occupy the left side of f(x) on the true risk scale.
current_x = red_left
for row in pos_terms.itertuples(index=False):
    width = float(row.ig_pp)
    start_x = current_x
    end_x = current_x + width
    plot_records.append(
        {
            "clinical_variable": row.clinical_variable,
            "display_name": row.display_name,
            "delta_pp": float(row.ig_pp),
            "abs_delta_pp": float(abs(row.ig_pp)),
            "labelled": bool(row.labelled),
            "sign": 1,
            "plot_start": start_x,
            "plot_end": end_x,
            "plot_mid": (start_x + end_x) / 2.0,
            "label_text": panel_b_label_text(
                row.clinical_variable,
                target=target,
                patient_local=patient_local,
                fold_id=fold_id,
                global_index=global_index,
                patient_mask=patient_mask,
                feature_to_index=feature_to_index,
                X_development=X_development,
            ),
        }
    )
    current_x = end_x

# Negative terms occupy the right side of f(x) on the true risk scale.
current_x = blue_left
for row in neg_terms.itertuples(index=False):
    width = float(abs(row.ig_pp))
    start_x = current_x
    end_x = current_x + width
    plot_records.append(
        {
            "clinical_variable": row.clinical_variable,
            "display_name": row.display_name,
            "delta_pp": float(row.ig_pp),
            "abs_delta_pp": float(abs(row.ig_pp)),
            "labelled": bool(row.labelled),
            "sign": -1,
            "plot_start": start_x,
            "plot_end": end_x,
            "plot_mid": (start_x + end_x) / 2.0,
            "label_text": panel_b_label_text(
                row.clinical_variable,
                target=target,
                patient_local=patient_local,
                fold_id=fold_id,
                global_index=global_index,
                patient_mask=patient_mask,
                feature_to_index=feature_to_index,
                X_development=X_development,
            ),
        }
    )
    current_x = end_x

plot_segment_df = pd.DataFrame(plot_records)

# Continuous red/blue ribbons, with interior white chevrons at variable boundaries.
def _draw_background_ribbon(ax, left, right, sign, color):
    width = max(right - left, 1e-9)
    head = min(width * 0.10, span * 0.014)
    head = max(head, 0.55)
    if sign > 0:
        points = [
            (left, y_bar),
            (left + 0.65 * head, y_bar - bar_height / 2),
            (right - head, y_bar - bar_height / 2),
            (right, y_bar),
            (right - head, y_bar + bar_height / 2),
            (left + 0.65 * head, y_bar + bar_height / 2),
        ]
    else:
        points = [
            (left, y_bar),
            (left + head, y_bar - bar_height / 2),
            (right - 0.65 * head, y_bar - bar_height / 2),
            (right, y_bar),
            (right - 0.65 * head, y_bar + bar_height / 2),
            (left + head, y_bar + bar_height / 2),
        ]
    ax.add_patch(
        Polygon(
            points,
            closed=True,
            facecolor=color,
            edgecolor="none",
            alpha=0.98,
            zorder=2,
        )
    )


def _draw_boundary_chevron(ax, boundary_x, sign):
    sep = max(0.55, span * 0.010)
    if sign > 0:
        chevron = [
            (boundary_x - sep, y_bar - bar_height / 2),
            (boundary_x - 0.18 * sep, y_bar - bar_height / 2),
            (boundary_x + 0.45 * sep, y_bar),
            (boundary_x - 0.18 * sep, y_bar + bar_height / 2),
            (boundary_x - sep, y_bar + bar_height / 2),
            (boundary_x - 0.36 * sep, y_bar),
        ]
    else:
        chevron = [
            (boundary_x + sep, y_bar - bar_height / 2),
            (boundary_x + 0.18 * sep, y_bar - bar_height / 2),
            (boundary_x - 0.45 * sep, y_bar),
            (boundary_x + 0.18 * sep, y_bar + bar_height / 2),
            (boundary_x + sep, y_bar + bar_height / 2),
            (boundary_x + 0.36 * sep, y_bar),
        ]
    ax.add_patch(
        Polygon(
            chevron,
            closed=True,
            facecolor="white",
            edgecolor="none",
            alpha=0.93,
            zorder=4,
        )
    )

if pos_total > 1e-10:
    _draw_background_ribbon(ax2, red_left, red_right, 1, RISK_UP)
if neg_total > 1e-10:
    _draw_background_ribbon(ax2, blue_left, blue_right, -1, RISK_DOWN)

for sign_value in (1, -1):
    sub = (
        plot_segment_df.loc[plot_segment_df["sign"] == sign_value]
        .copy()
        .sort_values("plot_start")
    )
    if len(sub) <= 1:
        continue
    for boundary_x in sub["plot_start"].to_list()[1:]:
        _draw_boundary_chevron(ax2, float(boundary_x), int(sign_value))

# Top-axis ticks represent the complete additive model-output coordinate in
# percentage points. Base value and f(x) are actual risks; intermediate ribbon
# coordinates are additive IG decomposition coordinates and may lie outside
# the conventional 0%-100% probability interval.
if span >= 90:
    tick_step = 10.0
elif span >= 55:
    tick_step = 5.0
else:
    tick_step = 2.5

tick_start = math.floor(xmin / tick_step) * tick_step
tick_end = math.ceil(xmax / tick_step) * tick_step
ax2.set_xticks(np.arange(tick_start, tick_end + 1e-9, tick_step))

# Strict geometry audit: the entire IG range and the base/final correspondence
# must be retained in the rendered panel.
base_from_geometry = red_left + neg_total
fx_from_ig = base_pct + pos_total - neg_total
print("\n" + "=" * 100)
print("Panel B full-range geometry audit")
print("=" * 100)
print(f"Positive IG total          : {pos_total:+.4f} pp")
print(f"Negative IG magnitude      : {neg_total:.4f} pp")
print(f"Red ribbon range           : [{red_left:.4f}, {red_right:.4f}]")
print(f"Blue ribbon range          : [{blue_left:.4f}, {blue_right:.4f}]")
print(f"Displayed full x-range     : [{xmin:.4f}, {xmax:.4f}]")
print(f"Base value (model)         : {base_pct:.4f}%")
print(f"Base implied by geometry   : {base_from_geometry:.4f}%")
print(f"f(x) (model)               : {predicted_pct:.4f}%")
print(f"f(x) implied by IG         : {fx_from_ig:.4f}%")

if abs(base_from_geometry - base_pct) > 0.08:
    raise RuntimeError(
        "Panel B geometry/base mismatch: the additive ribbon no longer corresponds "
        "to the model reference value."
    )
if abs(fx_from_ig - predicted_pct) > 0.08:
    raise RuntimeError(
        "Panel B geometry/f(x) mismatch: the additive ribbon no longer reconstructs "
        "the year-5 predicted risk."
    )
if xmin > red_left or xmax < blue_right:
    raise RuntimeError(
        "Panel B x-range clips part of the IG ribbon. Increase the plotting range."
    )

# Base value marker.
ax2.axvline(
    base_pct,
    ymin=0.48,
    ymax=0.82,
    color="0.45",
    linestyle="--",
    linewidth=0.90,
    zorder=5,
)
ax2.text(
    base_pct,
    0.955,
    "base value",
    ha="center",
    va="bottom",
    fontsize=6.8,
    color="0.40",
)
ax2.text(
    base_pct,
    0.865,
    f"{base_pct:.2f}%",
    ha="center",
    va="bottom",
    fontsize=7.0,
)

# f(x) / year-5 risk marker at the TRUE predicted-risk coordinate.
fx_display_x = predicted_pct
ax2.axvline(
    fx_display_x,
    ymin=0.44,
    ymax=0.70,
    color="black",
    linewidth=1.15,
    zorder=6,
)
ax2.text(
    fx_display_x - 0.95,
    0.935,
    "higher",
    ha="right",
    va="center",
    fontsize=6.8,
    color=RISK_UP,
)
ax2.text(
    fx_display_x,
    0.935,
    "↔",
    ha="center",
    va="center",
    fontsize=7.4,
    color="0.35",
)
ax2.text(
    fx_display_x + 0.95,
    0.935,
    "lower",
    ha="left",
    va="center",
    fontsize=6.8,
    color=RISK_DOWN,
)
ax2.text(
    fx_display_x,
    0.855,
    "f(x)",
    ha="center",
    va="bottom",
    fontsize=6.8,
    color="0.35",
)
ax2.text(
    fx_display_x,
    0.760,
    f"{predicted_pct:.2f}%",
    ha="center",
    va="bottom",
    fontsize=8.2,
    fontweight="bold",
)

ax2.legend(
    handles=[
        Patch(facecolor=RISK_UP, edgecolor="none", label="Higher CKD risk"),
        Patch(facecolor=RISK_DOWN, edgecolor="none", label="Lower CKD risk"),
    ],
    loc="lower center",
    bbox_to_anchor=(0.5, 1.105),
    ncol=2,
    frameon=False,
    fontsize=6.55,
    handlelength=1.9,
    columnspacing=2.4,
)
# One-row reference-style labels: values only for major contributors.
label_df = plot_segment_df.loc[plot_segment_df["labelled"]].copy()
label_y = -0.29
connector_start_y = y_bar - bar_height / 2 - 0.012
connector_end_y = label_y + 0.055

pos_lab = label_df.loc[label_df["sign"] > 0].copy().sort_values("plot_mid").reset_index(drop=True)
neg_lab = label_df.loc[label_df["sign"] < 0].copy().sort_values("plot_mid").reset_index(drop=True)

label_artists = []
connector_specs = []

left_margin = xmin + max(2.0, 0.025 * span)
right_margin = xmax - max(2.0, 0.025 * span)
protected_fx_gap = max(3.2, 0.055 * span)
left_end = fx_display_x - protected_fx_gap
right_start = fx_display_x + protected_fx_gap

if len(pos_lab) > 0:
    left_positions = np.linspace(left_margin, max(left_margin, left_end), len(pos_lab))
    for x_lab, row in zip(left_positions, pos_lab.itertuples(index=False)):
        connector_specs.append((float(row.plot_mid), float(x_lab), RISK_UP))
        label_artists.append(
            ax2.text(
                float(x_lab),
                label_y,
                str(row.label_text),
                ha="center",
                va="top",
                fontsize=5.35,
                color=RISK_UP,
                clip_on=False,
                zorder=5,
            )
        )

if len(neg_lab) > 0:
    right_positions = np.linspace(min(right_start, right_margin), right_margin, len(neg_lab))
    for x_lab, row in zip(right_positions, neg_lab.itertuples(index=False)):
        connector_specs.append((float(row.plot_mid), float(x_lab), RISK_DOWN))
        label_artists.append(
            ax2.text(
                float(x_lab),
                label_y,
                str(row.label_text),
                ha="center",
                va="top",
                fontsize=5.35,
                color=RISK_DOWN,
                clip_on=False,
                zorder=5,
            )
        )

for midpoint, x_lab, color in connector_specs:
    ax2.plot(
        [midpoint, x_lab],
        [connector_start_y, connector_end_y],
        color=color,
        linewidth=0.72,
        alpha=0.95,
        solid_capstyle="round",
        zorder=1,
        clip_on=False,
    )

if label_artists:
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    fontsize_now = 5.35
    for _ in range(12):
        boxes = [artist.get_window_extent(renderer=renderer) for artist in label_artists]
        overlap_pairs = [
            (i, i + 1)
            for i in range(len(boxes) - 1)
            if boxes[i].x1 + 1.0 > boxes[i + 1].x0
        ]
        if not overlap_pairs or fontsize_now <= 4.35:
            break
        fontsize_now -= 0.15
        for artist in label_artists:
            artist.set_fontsize(fontsize_now)
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
    final_boxes = [artist.get_window_extent(renderer=renderer) for artist in label_artists]
    final_overlap_pairs = [
        (i, i + 1)
        for i in range(len(final_boxes) - 1)
        if final_boxes[i].x1 + 1.0 > final_boxes[i + 1].x0
    ]
    print(
        f"Panel B one-row label audit: n={len(label_artists)}, "
        f"font={fontsize_now:.2f}, overlap_pairs={len(final_overlap_pairs)}"
    )
    if final_overlap_pairs:
        warnings.warn(
            "Panel B labels still have pixel-level overlap at the minimum font size. "
            "Consider reducing the number of labelled contributors."
        )

ax2.text(
    -0.045,
    1.025,
    "b",
    transform=ax2.transAxes,
    fontsize=12.5,
    fontweight="bold",
    ha="left",
    va="bottom",
)

# Connect the year-5 point in Panel A to the explained year-5 prediction in
# Panel B. The end point lands slightly to the right of f(x) to keep the
# dashed guide from overlapping the bold risk annotation.
connector_x_b = min(
    max(fx_display_x + max(2.5, 0.055 * span), xmin + 1.0),
    xmax - 1.0,
)

connector = ConnectionPatch(
    xyA=(x_highlight, y_highlight),
    coordsA=ax_risk.transData,
    xyB=(connector_x_b, 1.01),
    coordsB=ax2.transData,
    arrowstyle='-|>',
    mutation_scale=13,
    lw=1.1,
    linestyle=(0, (3, 3)),
    color=HIGHLIGHT_COLOR,
    shrinkA=10,
    shrinkB=2,
)
fig.add_artist(connector)


# ============================================================
# 13. Layout + source data
# ============================================================

plt.subplots_adjust(
    top=0.84,
    bottom=0.115,
    left=0.085,
    right=0.915,
)

source_A = trajectory_df.copy()
source_A["selected_patient_year5_risk_pct"] = predicted_pct

source_B = segment_df.copy()
source_B["panel"] = "B_year5_current_step_plus_earlier_history_force"
source_B["patient_local"] = patient_local
source_B["reference_risk_pct"] = base_pct
source_B["predicted_risk_pct"] = predicted_pct
source_B["current_step_index"] = PANEL_B_CURRENT_AUDIT["current_step_index"]
source_B["current_history_month"] = PANEL_B_CURRENT_AUDIT["current_history_month"]
source_B["earlier_history_net_pp"] = PANEL_B_CURRENT_AUDIT["earlier_history_net_pp"]
source_B["dynamic_reconciliation_diff_pp"] = PANEL_B_CURRENT_AUDIT[
    "dynamic_reconciliation_diff_pp"
]
source_B["display_fx_x"] = fx_display_x
source_B["display_connector_x"] = connector_x_b
source_B["panel_b_x_axis_meaning"] = "complete_additive_current_step_plus_earlier_history_IG_coordinate_percentage_points"
source_B = source_B.merge(
    plot_segment_df[["clinical_variable", "plot_start", "plot_end", "plot_mid", "sign", "label_text"]],
    on="clinical_variable",
    how="left",
)

source_risk = risk_df.copy()
source_risk["panel"] = "A_formal_risk_trajectory"
source_risk["patient_local"] = patient_local

pd.concat(
    [source_A, source_B, source_risk],
    ignore_index=True,
    sort=False,
).to_csv(
    OUTPUT_SOURCE,
    index=False,
    encoding="utf-8-sig",
)


# ============================================================
# 14. Save figure
# ============================================================

fig.savefig(
    OUTPUT_PNG,
    dpi=600,
    bbox_inches="tight",
    facecolor="white",
)
fig.savefig(
    OUTPUT_PDF,
    bbox_inches="tight",
    facecolor="white",
)

plt.show()
plt.close(fig)


# ============================================================
# 15. Final audit output
# ============================================================

selected_candidate = candidate_df.loc[candidate_df["patient_local"].eq(PATIENT_LOCAL)].iloc[0]

print("\n" + "=" * 100)
print("Figure completed")
print("=" * 100)
print(f"Selected patient_local: {patient_local}")
print(f"Year-5 predicted CKD risk: {predicted_pct:.2f}%")
print(
    f"Risk percentile: {100 * float(selected_candidate['risk_percentile']):.1f}th percentile"
)
print(
    f"Panel A continuous importance share: "
    f"{100 * float(selected_candidate['A_continuous_importance_share']):.1f}%"
)
print(
    f"Panel B labelled importance share: "
    f"{100 * float(selected_candidate['B_labelled_importance_share']):.1f}%"
)
print(
    f"Panel A visual score: {float(selected_candidate['panelA_visual_score']):.3f}; "
    f"median observed Z-range: {float(selected_candidate['A_median_observed_z_range']):.2f}; "
    f"minimum observed measurements: {int(selected_candidate['A_min_observed'])}"
)
print(f"Major comorbidity: {feature_display(profiles['comorbidity'])}")
print(f"Current ART (selection descriptor only): {feature_display(profiles['current_art'])}")
print(f"Matched cumulative ART (selection descriptor only): {feature_display(profiles['cumulative_art'])}")
print(
    f"Regimen changes across 0-5y landmarks: "
    f"{int(selected_candidate['regimen_changes_0_to_5'])}"
)

print("\nPanel A -- continuous trajectories only:")
for i, v in enumerate(A_VARIABLES, 1):
    print(f"{i:02d}. {feature_display(v)}")

print("\nPanel B -- labelled current-step / history contributors:")
for i, v in enumerate(FINAL_B_LABEL_VARIABLES, 1):
    contribution = float(
        B_all.loc[B_all["clinical_variable"].eq(v), "ig_pp"].sum()
    )
    print(f"{i:02d}. {feature_display(v):<44s} {contribution:+.3f} pp")

print("\nPanel B static-category label audit:")
for parent_variable in ["Sex", "Marriage", "Course", "WHOstage"]:
    if parent_variable in FINAL_B_LABEL_VARIABLES:
        decoded = decode_static_parent_category(
            patient_local=patient_local,
            fold_id=fold_id,
            parent_variable=parent_variable,
        )
        print(
            f"  {parent_variable:<10s} -> {decoded}"
        )

print("\nSaved files:")
print("Figure PNG       :", OUTPUT_PNG)
print("Figure PDF       :", OUTPUT_PDF)
print("Source CSV       :", OUTPUT_SOURCE)
print("Candidate ranking:", OUTPUT_CANDIDATE_TABLE)
print("Panel A profile  :", OUTPUT_A_PROFILE)
print("Panel B profile  :", OUTPUT_B_PROFILE)
print("Panel B audit    :", PANEL_B_CURRENT_AUDIT_FILE)
