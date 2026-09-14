from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd


ROOT = Path(r"__CKD_LOCAL_INPUT__/Documents/深度学习")
OUT = ROOT / "outputs" / "comment1-sensitivity-20260903"
BASELINE = ROOT / "multi_center_ML_csv" / "multi_center_ML_05_纳入排除后.csv"
VALID_EGFR = OUT / "complete_valid_egfr_records.csv"
PATIENT_COMPARISON = OUT / "patient_level_outcome_comparison.csv"

MODEL_CSV = OUT / "严格CKD_建模随访表_原字段结构.csv"
LONG_CSV = OUT / "严格CKD_逐次eGFR随访核验长表.csv"
AUDIT_CSV = OUT / "严格CKD_患者级结局核验表.csv"
META_JSON = ROOT / "tmp" / "reviewer_revision" / "strict_followup_workbook_metadata.json"

PERSISTENCE_DAYS = 90
EGFR_THRESHOLD = 60.0
DECLINE_THRESHOLD = 0.25
DAYS_PER_MONTH = 30.0


def parse_dates(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    for column in columns:
        frame[column] = pd.to_datetime(frame[column], errors="coerce")
    return frame


def iso_dates(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    result = frame.copy()
    for column in columns:
        result[column] = pd.to_datetime(result[column], errors="coerce").dt.strftime("%Y-%m-%d")
    return result


def build_long_audit(records: pd.DataFrame, patient: pd.DataFrame) -> pd.DataFrame:
    records = records.sort_values(["ID", "lab_date"], kind="mergesort").reset_index(drop=True)
    records["days_from_ART"] = (records["lab_date"] - records["ARTtime"]).dt.days
    records["absolute_eGFR_change_from_reference"] = records["eGFR"] - records["reference_eGFR"]
    records["relative_eGFR_decline"] = (
        records["reference_eGFR"] - records["eGFR"]
    ) / records["reference_eGFR"]
    records["eGFR_below_60"] = records["eGFR"].lt(EGFR_THRESHOLD)
    records["decline_gt_25pct"] = records["relative_eGFR_decline"].gt(DECLINE_THRESHOLD)
    records["strict_abnormal_observation"] = (
        records["eGFR_below_60"] & records["decline_gt_25pct"]
    )
    records["below60_but_decline_le25pct"] = (
        records["eGFR_below_60"] & ~records["decline_gt_25pct"]
    )

    previous_strict = records.groupby("ID", sort=False)["strict_abnormal_observation"].shift(fill_value=False)
    starts_episode = records["strict_abnormal_observation"] & ~previous_strict
    episode_counter = starts_episode.groupby(records["ID"], sort=False).cumsum().astype("Int64")
    records["strict_episode_id"] = episode_counter.where(records["strict_abnormal_observation"])

    strict_rows = records["strict_episode_id"].notna()
    strict_index = records.loc[strict_rows].index
    episode_group = records.loc[strict_rows].groupby(
        ["ID", "strict_episode_id"], sort=False, dropna=False
    )
    records.loc[strict_index, "strict_episode_start_date"] = episode_group["lab_date"].transform("min")
    records["strict_episode_start_date"] = pd.to_datetime(
        records["strict_episode_start_date"], errors="coerce"
    )
    records["days_since_strict_episode_start"] = (
        records["lab_date"] - records["strict_episode_start_date"]
    ).dt.days.astype("Int64")
    episode_group = records.loc[strict_rows].groupby(
        ["ID", "strict_episode_id"], sort=False, dropna=False
    )
    records.loc[strict_index, "strict_episode_max_observed_days"] = episode_group[
        "days_since_strict_episode_start"
    ].transform("max")
    records["strict_episode_max_observed_days"] = records[
        "strict_episode_max_observed_days"
    ].astype("Int64")
    records["episode_ever_reaches_90_days"] = (
        records["strict_episode_max_observed_days"].ge(PERSISTENCE_DAYS).fillna(False)
    )
    records["persistence_met_at_this_visit"] = (
        records["strict_abnormal_observation"]
        & records["days_since_strict_episode_start"].ge(PERSISTENCE_DAYS).fillna(False)
    )
    confirmation_order = records.loc[records["persistence_met_at_this_visit"]].groupby(
        ["ID", "strict_episode_id"], sort=False
    ).cumcount()
    records["first_confirmation_in_episode"] = False
    records.loc[confirmation_order.index, "first_confirmation_in_episode"] = confirmation_order.eq(0)

    patient_keep = patient[
        [
            "ID",
            "original_CKDstatus",
            "original_CKDtime",
            "sensitivity_CKDstatus",
            "sensitivity_CKDtime",
            "confirmation_date",
            "sensitivity_end_date",
            "ascertainment_status",
        ]
    ].copy()
    records = records.merge(patient_keep, on="ID", how="left", validate="many_to_one")
    records["selected_event_episode"] = (
        records["sensitivity_CKDstatus"].eq(1)
        & records["strict_episode_start_date"].eq(records["sensitivity_CKDtime"])
    )
    records["strict_event_onset_row"] = (
        records["selected_event_episode"]
        & records["lab_date"].eq(records["sensitivity_CKDtime"])
    )
    records["strict_event_confirmation_row"] = (
        records["selected_event_episode"]
        & records["lab_date"].eq(records["confirmation_date"])
    )
    records["after_selected_event_onset"] = (
        records["sensitivity_CKDstatus"].eq(1)
        & records["lab_date"].gt(records["sensitivity_CKDtime"])
    )
    records["after_selected_event_confirmation"] = (
        records["sensitivity_CKDstatus"].eq(1)
        & records["lab_date"].gt(records["confirmation_date"])
    )

    status = np.full(len(records), "不满足严格异常", dtype=object)
    status[records["below60_but_decline_le25pct"].to_numpy()] = "eGFR<60但下降≤25%，继续随访"
    status[records["strict_abnormal_observation"].to_numpy()] = "严格异常，尚未达到90天"
    status[records["episode_ever_reaches_90_days"].to_numpy()] = "属于达到90天的严格异常段"
    status[records["strict_event_onset_row"].to_numpy()] = "最终严格CKD起始记录"
    status[records["strict_event_confirmation_row"].to_numpy()] = "最终严格CKD确认记录"
    status[records["after_selected_event_confirmation"].to_numpy()] = "确认后记录（保留用于完整审计）"
    records["record_interpretation"] = status

    columns = [
        "ID",
        "data",
        "source_center",
        "ARTtime",
        "Lastfollowtime",
        "lab_date",
        "days_from_ART",
        "reference_eGFR",
        "eGFR",
        "absolute_eGFR_change_from_reference",
        "relative_eGFR_decline",
        "eGFR_below_60",
        "decline_gt_25pct",
        "strict_abnormal_observation",
        "below60_but_decline_le25pct",
        "strict_episode_id",
        "strict_episode_start_date",
        "days_since_strict_episode_start",
        "strict_episode_max_observed_days",
        "episode_ever_reaches_90_days",
        "persistence_met_at_this_visit",
        "first_confirmation_in_episode",
        "selected_event_episode",
        "strict_event_onset_row",
        "strict_event_confirmation_row",
        "after_selected_event_onset",
        "after_selected_event_confirmation",
        "original_CKDstatus",
        "original_CKDtime",
        "sensitivity_CKDstatus",
        "sensitivity_CKDtime",
        "confirmation_date",
        "sensitivity_end_date",
        "ascertainment_status",
        "record_interpretation",
    ]
    return records[columns]


def build_model_table(base: pd.DataFrame, patient: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    original = base[["ID", "Lastfollowtime", "CKDstatus", "CKDtime", "interval"]].copy()
    original = original.rename(
        columns={
            "Lastfollowtime": "original_Lastfollowtime",
            "CKDstatus": "original_CKDstatus_from_input",
            "CKDtime": "original_CKDtime_from_input",
            "interval": "original_interval_from_input",
        }
    )
    mapping = patient[
        [
            "ID",
            "Lastfollowtime",
            "sensitivity_CKDstatus",
            "sensitivity_CKDtime",
            "sensitivity_end_date",
            "sensitivity_followup_days",
            "confirmation_date",
            "confirmation_gap_days",
            "n_valid_followup_eGFR",
            "reclassified_to_non_CKD",
            "new_sensitivity_event",
            "onset_delay_days",
            "ascertainment_status",
        ]
    ].copy()
    result = base.drop(columns=["Lastfollowtime", "CKDstatus", "CKDtime", "interval"]).merge(
        mapping, on="ID", how="left", validate="one_to_one"
    )
    result = result.rename(
        columns={
            "sensitivity_CKDstatus": "CKDstatus",
            "sensitivity_CKDtime": "CKDtime",
        }
    )
    result["interval"] = result["sensitivity_followup_days"] / DAYS_PER_MONTH
    original_columns = base.columns.tolist()
    result = result[original_columns]

    audit = result.merge(original, on="ID", how="left", validate="one_to_one").merge(
        mapping[
            [
                "ID",
                "sensitivity_end_date",
                "confirmation_date",
                "confirmation_gap_days",
                "n_valid_followup_eGFR",
                "reclassified_to_non_CKD",
                "new_sensitivity_event",
                "onset_delay_days",
                "ascertainment_status",
            ]
        ],
        on="ID",
        how="left",
        validate="one_to_one",
    )
    return result, audit


def main() -> None:
    base = pd.read_csv(BASELINE, encoding="utf-8-sig", dtype={"ID": "string"}, low_memory=False)
    patient = pd.read_csv(
        PATIENT_COMPARISON,
        encoding="utf-8-sig",
        dtype={"ID": "string"},
        low_memory=False,
    )
    patient = parse_dates(
        patient,
        [
            "ARTtime",
            "Lastfollowtime",
            "original_CKDtime",
            "sensitivity_CKDtime",
            "confirmation_date",
            "sensitivity_end_date",
        ],
    )
    records = pd.read_csv(
        VALID_EGFR,
        encoding="utf-8-sig",
        dtype={"ID": "string"},
        low_memory=False,
    )
    records = parse_dates(records, ["ARTtime", "Lastfollowtime", "lab_date"])

    model, audit = build_model_table(base, patient)
    long_audit = build_long_audit(records, patient)

    model_dates = ["Birthday", "ARTtime", "Lastfollowtime", "CKDtime"]
    audit_dates = model_dates + [
        "original_Lastfollowtime",
        "original_CKDtime_from_input",
        "sensitivity_end_date",
        "confirmation_date",
    ]
    long_dates = [
        "ARTtime",
        "Lastfollowtime",
        "lab_date",
        "strict_episode_start_date",
        "original_CKDtime",
        "sensitivity_CKDtime",
        "confirmation_date",
        "sensitivity_end_date",
    ]
    model_out = iso_dates(model, model_dates)
    audit_out = iso_dates(audit, audit_dates)
    long_out = iso_dates(long_audit, long_dates)

    model_out.to_csv(MODEL_CSV, index=False, encoding="utf-8-sig")
    audit_out.to_csv(AUDIT_CSV, index=False, encoding="utf-8-sig")
    long_out.to_csv(LONG_CSV, index=False, encoding="utf-8-sig")

    model_event_n = int(model["CKDstatus"].sum())
    onset_rows = int(long_audit["strict_event_onset_row"].sum())
    confirmation_rows = int(long_audit["strict_event_confirmation_row"].sum())
    status_mismatch = int(
        model[["ID", "CKDstatus"]]
        .merge(patient[["ID", "sensitivity_CKDstatus"]], on="ID", validate="one_to_one")
        .eval("CKDstatus != sensitivity_CKDstatus")
        .sum()
    )
    date_check = model[["ID", "CKDtime"]].merge(
        patient[["ID", "sensitivity_CKDtime"]], on="ID", validate="one_to_one"
    )
    model_event_date = pd.to_datetime(date_check["CKDtime"], errors="coerce")
    event_date_match = model_event_date.eq(date_check["sensitivity_CKDtime"]) | (
        model_event_date.isna() & date_check["sensitivity_CKDtime"].isna()
    )
    date_mismatch = int((~event_date_match).sum())
    interval_expected = (
        pd.to_datetime(patient["sensitivity_end_date"]) - pd.to_datetime(patient["ARTtime"])
    ).dt.days / DAYS_PER_MONTH
    interval_check = model[["ID", "interval"]].merge(
        pd.DataFrame({"ID": patient["ID"], "expected": interval_expected}),
        on="ID",
        validate="one_to_one",
    )
    max_interval_diff = float((interval_check["interval"] - interval_check["expected"]).abs().max())

    summary = {
        "patient_rows": int(len(model)),
        "original_columns_preserved": int(len(base.columns)),
        "strict_event_n": model_event_n,
        "non_event_n": int(len(model) - model_event_n),
        "long_followup_rows": int(len(long_audit)),
        "patients_with_valid_post_art_egfr": int(long_audit["ID"].nunique()),
        "patients_without_valid_post_art_egfr": int(len(model) - long_audit["ID"].nunique()),
        "strict_onset_row_n": onset_rows,
        "strict_confirmation_row_n": confirmation_rows,
        "status_mismatch_n": status_mismatch,
        "event_date_mismatch_n": date_mismatch,
        "max_interval_difference_days_equivalent": max_interval_diff * DAYS_PER_MONTH,
        "exact_90_day_confirmation_n": int(
            patient["confirmation_gap_days"].eq(PERSISTENCE_DAYS).sum()
        ),
        "model_csv": str(MODEL_CSV),
        "audit_csv": str(AUDIT_CSV),
        "long_csv": str(LONG_CSV),
    }
    META_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
