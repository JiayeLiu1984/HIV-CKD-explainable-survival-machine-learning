# ============================================================
# Step 6：构建Super Landmark长格式数据并核查标签一致性
#
# 代码逻辑：
# 1. 读取Step 1的患者级结局、Landmark掩码和未来半年标签。
# 2. 读取Step 2固定的患者级五折划分。
# 3. 读取Step 5生成的146项原始历史汇总特征。
# 4. 仅将prediction_origin_mask=True的患者-Landmark记录堆叠成长格式。
# 5. 每条记录的分析时间从Landmark重新起算，最长截断为60个月。
# 6. 60个月内发生CKD记为事件；超过60个月的事件按60个月行政删失。
# 7. 保存长格式特征、精确生存结局、离散半年标签和行号映射。
# 8. 五折仍按患者划分，同一患者的全部Landmark记录保持在同一折。
# 9. 本步骤只构建数据，不拟合Cox、RSF、RNN或LSTM。
# 10. 锁定测试集仅完成结构化保存，不参与调参或模型选择。
# ============================================================

import json
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# 1. 路径和固定参数
# ============================================================

STEP1_DIR = Path(
    "__CKD_WORKDIR__/rolling_5y_step1_new_split"
)

STEP2_DIR = Path(
    "__CKD_WORKDIR__/rolling_5y_step2_folds"
)

STEP5_DIR = Path(
    "__CKD_WORKDIR__/rolling_5y_step5_landmark_summary"
)

OUTPUT_DIR = Path(
    "__CKD_WORKDIR__/rolling_5y_step6_super_landmark_data"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

N_SPLITS = 5
INTERVAL_WIDTH_MONTH = 6.0
MAX_PREDICTION_MONTH = 60.0
TIME_TOLERANCE = 1e-6

EXPECTED_TOTAL_N = 31911
EXPECTED_DEVELOPMENT_N = 22337
EXPECTED_TEST_N = 9574
EXPECTED_LANDMARK_N = 6
EXPECTED_FUTURE_INTERVAL_N = 10
EXPECTED_SUMMARY_FEATURE_N = 146

EXPECTED_LANDMARK_MONTHS = np.array(
    [0, 12, 24, 36, 48, 60],
    dtype=np.int32,
)

EXPECTED_FUTURE_END_MONTHS = np.arange(
    6,
    61,
    6,
    dtype=np.int32,
)


# ============================================================
# 2. 检查输入文件
# ============================================================

required_files = [
    STEP1_DIR / "development_idx.npy",
    STEP1_DIR / "test_idx.npy",
    STEP1_DIR / "event.npy",
    STEP1_DIR / "observed_time_month.npy",
    STEP1_DIR / "landmark_months.npy",
    STEP1_DIR / "future_end_months.npy",
    STEP1_DIR / "landmark_eligible_mask.npy",
    STEP1_DIR / "prediction_origin_mask.npy",
    STEP1_DIR / "future_event_matrix.npy",
    STEP1_DIR / "future_at_risk_mask.npy",
    STEP1_DIR / "patient_info_new_split.csv",
    STEP2_DIR / "fold_id_all.npy",
    STEP5_DIR / "X_summary_development_raw.npy",
    STEP5_DIR / "X_summary_test_raw.npy",
    STEP5_DIR / "summary_feature_names.csv",
    STEP5_DIR / "step5_manifest.json",
]

missing_files = [
    str(path) for path in required_files if not path.exists()
]

if missing_files:
    raise FileNotFoundError(
        "以下输入文件不存在：\n" + "\n".join(missing_files)
    )


# ============================================================
# 3. 读取患者划分、结局和Landmark标签
# ============================================================

development_idx = np.load(
    STEP1_DIR / "development_idx.npy"
).astype(np.int32)

test_idx = np.load(
    STEP1_DIR / "test_idx.npy"
).astype(np.int32)

event_all = np.load(
    STEP1_DIR / "event.npy"
).astype(np.int8)

observed_time_month_all = np.load(
    STEP1_DIR / "observed_time_month.npy"
).astype(np.float64)

landmark_months = np.load(
    STEP1_DIR / "landmark_months.npy"
).astype(np.int32)

future_end_months = np.load(
    STEP1_DIR / "future_end_months.npy"
).astype(np.int32)

landmark_eligible_mask_all = np.load(
    STEP1_DIR / "landmark_eligible_mask.npy"
).astype(bool)

prediction_origin_mask_all = np.load(
    STEP1_DIR / "prediction_origin_mask.npy"
).astype(bool)

future_event_matrix_all = np.load(
    STEP1_DIR / "future_event_matrix.npy"
).astype(np.float32)

future_at_risk_mask_all = np.load(
    STEP1_DIR / "future_at_risk_mask.npy"
).astype(bool)

fold_id_all = np.load(
    STEP2_DIR / "fold_id_all.npy"
).astype(np.int8)

patient_info = pd.read_csv(
    STEP1_DIR / "patient_info_new_split.csv",
    encoding="utf-8-sig",
    dtype={
        "ID": "string",
        "center": "string",
        "analysis_split": "string",
    },
)


# ============================================================
# 4. 读取Step 5汇总特征和特征字典
# ============================================================

X_summary_development = np.load(
    STEP5_DIR / "X_summary_development_raw.npy",
    mmap_mode="r",
)

X_summary_test = np.load(
    STEP5_DIR / "X_summary_test_raw.npy",
    mmap_mode="r",
)

summary_feature_table = pd.read_csv(
    STEP5_DIR / "summary_feature_names.csv",
    encoding="utf-8-sig",
)

with open(
    STEP5_DIR / "step5_manifest.json",
    "r",
    encoding="utf-8",
) as file:
    step5_manifest = json.load(file)


# ============================================================
# 5. 核对患者顺序、数组形状和固定定义
# ============================================================

expected_shapes = {
    "event_all": (EXPECTED_TOTAL_N,),
    "observed_time_month_all": (EXPECTED_TOTAL_N,),
    "landmark_eligible_mask_all": (
        EXPECTED_TOTAL_N,
        EXPECTED_LANDMARK_N,
    ),
    "prediction_origin_mask_all": (
        EXPECTED_TOTAL_N,
        EXPECTED_LANDMARK_N,
    ),
    "future_event_matrix_all": (
        EXPECTED_TOTAL_N,
        EXPECTED_LANDMARK_N,
        EXPECTED_FUTURE_INTERVAL_N,
    ),
    "future_at_risk_mask_all": (
        EXPECTED_TOTAL_N,
        EXPECTED_LANDMARK_N,
        EXPECTED_FUTURE_INTERVAL_N,
    ),
    "fold_id_all": (EXPECTED_TOTAL_N,),
    "X_summary_development": (
        EXPECTED_DEVELOPMENT_N,
        EXPECTED_LANDMARK_N,
        EXPECTED_SUMMARY_FEATURE_N,
    ),
    "X_summary_test": (
        EXPECTED_TEST_N,
        EXPECTED_LANDMARK_N,
        EXPECTED_SUMMARY_FEATURE_N,
    ),
}

actual_shapes = {
    "event_all": event_all.shape,
    "observed_time_month_all": observed_time_month_all.shape,
    "landmark_eligible_mask_all": landmark_eligible_mask_all.shape,
    "prediction_origin_mask_all": prediction_origin_mask_all.shape,
    "future_event_matrix_all": future_event_matrix_all.shape,
    "future_at_risk_mask_all": future_at_risk_mask_all.shape,
    "fold_id_all": fold_id_all.shape,
    "X_summary_development": X_summary_development.shape,
    "X_summary_test": X_summary_test.shape,
}

for name, expected_shape in expected_shapes.items():
    if actual_shapes[name] != expected_shape:
        raise ValueError(
            f"{name}形状为{actual_shapes[name]}，"
            f"应为{expected_shape}。"
        )

if len(patient_info) != EXPECTED_TOTAL_N:
    raise ValueError(
        f"patient_info患者数为{len(patient_info)}，"
        f"应为{EXPECTED_TOTAL_N}。"
    )

if not np.array_equal(
    patient_info["patient_index"].to_numpy(dtype=np.int32),
    np.arange(EXPECTED_TOTAL_N, dtype=np.int32),
):
    raise ValueError("patient_info中的patient_index顺序不连续。")

if not np.array_equal(
    landmark_months,
    EXPECTED_LANDMARK_MONTHS,
):
    raise ValueError(
        f"Landmark月份为{landmark_months.tolist()}，"
        f"应为{EXPECTED_LANDMARK_MONTHS.tolist()}。"
    )

if not np.array_equal(
    future_end_months,
    EXPECTED_FUTURE_END_MONTHS,
):
    raise ValueError(
        f"未来预测时间为{future_end_months.tolist()}，"
        f"应为{EXPECTED_FUTURE_END_MONTHS.tolist()}。"
    )

if len(development_idx) != EXPECTED_DEVELOPMENT_N:
    raise ValueError("开发集患者数不正确。")

if len(test_idx) != EXPECTED_TEST_N:
    raise ValueError("测试集患者数不正确。")

if np.intersect1d(development_idx, test_idx).size > 0:
    raise ValueError("开发集和测试集患者索引存在交叉。")

if not np.array_equal(
    np.sort(np.concatenate([development_idx, test_idx])),
    np.arange(EXPECTED_TOTAL_N, dtype=np.int32),
):
    raise ValueError("开发集和测试集未完整覆盖全部患者。")

if not np.all(
    patient_info.loc[development_idx, "analysis_split"].to_numpy()
    == "development"
):
    raise ValueError("development_idx与patient_info划分不一致。")

if not np.all(
    patient_info.loc[test_idx, "analysis_split"].to_numpy()
    == "test"
):
    raise ValueError("test_idx与patient_info划分不一致。")

if not np.isin(fold_id_all[development_idx], np.arange(N_SPLITS)).all():
    raise ValueError("开发集fold_id必须为0～4。")

if not np.all(fold_id_all[test_idx] == -1):
    raise ValueError("锁定测试集fold_id应全部为-1。")

if len(summary_feature_table) != EXPECTED_SUMMARY_FEATURE_N:
    raise ValueError("Step 5特征字典行数不是146。")

if int(step5_manifest["summary_feature_n"]) != EXPECTED_SUMMARY_FEATURE_N:
    raise ValueError("Step 5 manifest中的特征数不是146。")

if not np.array_equal(
    prediction_origin_mask_all,
    landmark_eligible_mask_all
    & future_at_risk_mask_all.any(axis=2),
):
    raise ValueError(
        "prediction_origin_mask与Landmark资格或未来风险掩码不一致。"
    )

if np.any(
    (future_event_matrix_all > 0)
    & (~future_at_risk_mask_all)
):
    raise ValueError("存在事件标签位于风险掩码之外。")

if np.any(future_event_matrix_all.sum(axis=2) > 1):
    raise ValueError("同一患者同一Landmark存在多个未来事件区间。")


# ============================================================
# 6. 构建单个队列的Super Landmark长格式数据
# ============================================================

def build_super_landmark_dataset(
    dataset_name,
    cohort_global_idx,
    X_summary_cohort,
):
    """将患者×Landmark张量转换为有效预测起点的长格式数据。"""

    cohort_origin_mask = prediction_origin_mask_all[cohort_global_idx]
    local_patient_idx, landmark_index = np.where(cohort_origin_mask)

    local_patient_idx = local_patient_idx.astype(np.int32)
    landmark_index = landmark_index.astype(np.int8)
    global_patient_idx = cohort_global_idx[local_patient_idx].astype(np.int32)

    long_row_n = len(local_patient_idx)

    if long_row_n != int(cohort_origin_mask.sum()):
        raise ValueError(f"{dataset_name}长记录数与有效预测起点数不一致。")

    X_long = np.asarray(
        X_summary_cohort[local_patient_idx, landmark_index, :],
        dtype=np.float32,
    )

    if X_long.shape != (
        long_row_n,
        EXPECTED_SUMMARY_FEATURE_N,
    ):
        raise ValueError(f"{dataset_name}长格式特征形状异常。")

    if not np.isfinite(X_long).all():
        raise ValueError(f"{dataset_name}长格式特征存在NaN或无穷值。")

    landmark_month = landmark_months[landmark_index].astype(np.int32)
    original_event = event_all[global_patient_idx].astype(np.int8)
    original_observed_time = observed_time_month_all[
        global_patient_idx
    ].astype(np.float64)

    remaining_time_month = (
        original_observed_time - landmark_month.astype(np.float64)
    )

    if np.any(remaining_time_month <= TIME_TOLERANCE):
        raise ValueError(
            f"{dataset_name}存在Landmark后剩余随访时间小于或等于0的记录。"
        )

    analysis_time_month = np.minimum(
        remaining_time_month,
        MAX_PREDICTION_MONTH,
    ).astype(np.float32)

    event_within_60m = (
        (original_event == 1)
        & (
            remaining_time_month
            <= MAX_PREDICTION_MONTH + TIME_TOLERANCE
        )
    ).astype(np.int8)

    administrative_censor_60m = (
        (event_within_60m == 0)
        & (
            remaining_time_month
            >= MAX_PREDICTION_MONTH - TIME_TOLERANCE
        )
    ).astype(np.int8)

    censor_before_60m = (
        (original_event == 0)
        & (
            remaining_time_month
            < MAX_PREDICTION_MONTH - TIME_TOLERANCE
        )
    ).astype(np.int8)

    future_event_long = future_event_matrix_all[
        global_patient_idx,
        landmark_index,
        :,
    ].astype(np.float32)

    future_at_risk_long = future_at_risk_mask_all[
        global_patient_idx,
        landmark_index,
        :,
    ].astype(bool)

    future_at_risk_interval_n = future_at_risk_long.sum(
        axis=1
    ).astype(np.int8)

    future_event_interval_n = future_event_long.sum(
        axis=1
    ).astype(np.int8)

    event_interval_index = np.full(
        long_row_n,
        -1,
        dtype=np.int8,
    )
    has_discrete_event = future_event_interval_n == 1
    event_interval_index[has_discrete_event] = np.argmax(
        future_event_long[has_discrete_event],
        axis=1,
    ).astype(np.int8)

    # 事件患者：风险区间数等于事件区间序号+1，超过60个月则保留10个区间。
    event_interval_full = np.full(
        long_row_n,
        -1,
        dtype=np.int32,
    )
    event_rows = original_event == 1
    event_interval_full[event_rows] = (
        np.ceil(
            (
                remaining_time_month[event_rows]
                - TIME_TOLERANCE
            )
            / INTERVAL_WIDTH_MONTH
        ).astype(np.int32)
        - 1
    )

    expected_at_risk_interval_n = np.zeros(
        long_row_n,
        dtype=np.int32,
    )

    expected_at_risk_interval_n[event_rows] = np.clip(
        event_interval_full[event_rows] + 1,
        0,
        EXPECTED_FUTURE_INTERVAL_N,
    )

    censor_rows = ~event_rows
    expected_at_risk_interval_n[censor_rows] = np.clip(
        np.floor(
            (
                remaining_time_month[censor_rows]
                + TIME_TOLERANCE
            )
            / INTERVAL_WIDTH_MONTH
        ).astype(np.int32),
        0,
        EXPECTED_FUTURE_INTERVAL_N,
    )

    if not np.array_equal(
        future_at_risk_interval_n.astype(np.int32),
        expected_at_risk_interval_n,
    ):
        mismatch_idx = np.flatnonzero(
            future_at_risk_interval_n.astype(np.int32)
            != expected_at_risk_interval_n
        )[:20]
        raise ValueError(
            f"{dataset_name}长格式剩余时间与未来风险掩码不一致，"
            f"示例长行：{mismatch_idx.tolist()}"
        )

    if np.any(future_at_risk_interval_n < 1):
        raise ValueError(
            f"{dataset_name}存在没有完整未来半年区间的长记录。"
        )

    if not np.array_equal(
        future_event_interval_n,
        event_within_60m,
    ):
        mismatch_idx = np.flatnonzero(
            future_event_interval_n != event_within_60m
        )[:20]
        raise ValueError(
            f"{dataset_name}精确60个月事件与离散事件标签不一致，"
            f"示例长行：{mismatch_idx.tolist()}"
        )

    if np.any(event_within_60m == 1):
        expected_event_interval = event_interval_full[
            event_within_60m == 1
        ]
        observed_event_interval = event_interval_index[
            event_within_60m == 1
        ].astype(np.int32)

        if not np.array_equal(
            expected_event_interval,
            observed_event_interval,
        ):
            raise ValueError(
                f"{dataset_name}精确事件时间与离散事件区间不一致。"
            )

    if np.any(
        (analysis_time_month <= 0)
        | (analysis_time_month > MAX_PREDICTION_MONTH + TIME_TOLERANCE)
    ):
        raise ValueError(f"{dataset_name}分析时间不在(0, 60]个月内。")

    if np.any(
        (event_within_60m == 1)
        & (
            np.abs(
                analysis_time_month.astype(np.float64)
                - remaining_time_month
            )
            > 1e-4
        )
    ):
        raise ValueError(f"{dataset_name}60个月内事件时间被错误截断。")

    if np.any(
        (administrative_censor_60m == 1)
        & (
            np.abs(
                analysis_time_month.astype(np.float64)
                - MAX_PREDICTION_MONTH
            )
            > 1e-4
        )
    ):
        raise ValueError(f"{dataset_name}行政删失时间不是60个月。")

    if np.any(
        event_within_60m
        + administrative_censor_60m
        + censor_before_60m
        != 1
    ):
        raise ValueError(
            f"{dataset_name}长记录没有唯一对应事件或删失类型。"
        )

    row_index_map = np.full(
        (
            len(cohort_global_idx),
            EXPECTED_LANDMARK_N,
        ),
        -1,
        dtype=np.int32,
    )
    row_index_map[
        local_patient_idx,
        landmark_index,
    ] = np.arange(long_row_n, dtype=np.int32)

    if not np.array_equal(
        row_index_map >= 0,
        cohort_origin_mask,
    ):
        raise ValueError(f"{dataset_name}长行号映射与有效起点掩码不一致。")

    fold_id_long = fold_id_all[global_patient_idx].astype(np.int8)

    metadata = pd.DataFrame({
        "long_row_index": np.arange(long_row_n, dtype=np.int32),
        "local_patient_index": local_patient_idx,
        "global_patient_index": global_patient_idx,
        "ID": patient_info.loc[
            global_patient_idx,
            "ID",
        ].astype("string").to_numpy(),
        "center": patient_info.loc[
            global_patient_idx,
            "center",
        ].astype("string").to_numpy(),
        "fold_id": fold_id_long,
        "landmark_index": landmark_index,
        "landmark_month": landmark_month,
        "original_event": original_event,
        "original_observed_time_month": original_observed_time.astype(
            np.float32
        ),
        "remaining_time_month": remaining_time_month.astype(np.float32),
        "analysis_time_month": analysis_time_month,
        "event_within_60m": event_within_60m,
        "administrative_censor_60m": administrative_censor_60m,
        "censor_before_60m": censor_before_60m,
        "future_at_risk_interval_n": future_at_risk_interval_n,
        "event_interval_index": event_interval_index,
    })

    return {
        "X_long": X_long,
        "metadata": metadata,
        "row_index_map": row_index_map,
        "future_event_long": future_event_long,
        "future_at_risk_long": future_at_risk_long,
        "local_patient_idx": local_patient_idx,
        "global_patient_idx": global_patient_idx,
        "fold_id_long": fold_id_long,
        "landmark_index": landmark_index,
        "landmark_month": landmark_month,
        "analysis_time_month": analysis_time_month,
        "event_within_60m": event_within_60m,
    }


# ============================================================
# 7. 构建开发集和锁定测试集长格式数据
# ============================================================

development_long = build_super_landmark_dataset(
    "development",
    development_idx,
    X_summary_development,
)

test_long = build_super_landmark_dataset(
    "test",
    test_idx,
    X_summary_test,
)


# ============================================================
# 8. 保存单个队列的长格式输出
# ============================================================

def save_super_landmark_dataset(dataset_name, dataset_dict):
    """保存特征、结局、离散标签、元数据和行号映射。"""

    np.save(
        OUTPUT_DIR / f"X_super_landmark_{dataset_name}_raw.npy",
        dataset_dict["X_long"],
    )
    np.save(
        OUTPUT_DIR / f"{dataset_name}_long_row_index_map.npy",
        dataset_dict["row_index_map"],
    )
    np.save(
        OUTPUT_DIR / f"{dataset_name}_future_event_long.npy",
        dataset_dict["future_event_long"],
    )
    np.save(
        OUTPUT_DIR / f"{dataset_name}_future_at_risk_long.npy",
        dataset_dict["future_at_risk_long"],
    )
    np.save(
        OUTPUT_DIR / f"{dataset_name}_local_patient_idx_long.npy",
        dataset_dict["local_patient_idx"],
    )
    np.save(
        OUTPUT_DIR / f"{dataset_name}_global_patient_idx_long.npy",
        dataset_dict["global_patient_idx"],
    )
    np.save(
        OUTPUT_DIR / f"{dataset_name}_fold_id_long.npy",
        dataset_dict["fold_id_long"],
    )
    np.save(
        OUTPUT_DIR / f"{dataset_name}_landmark_index_long.npy",
        dataset_dict["landmark_index"],
    )
    np.save(
        OUTPUT_DIR / f"{dataset_name}_landmark_month_long.npy",
        dataset_dict["landmark_month"],
    )
    np.save(
        OUTPUT_DIR / f"{dataset_name}_analysis_time_month.npy",
        dataset_dict["analysis_time_month"],
    )
    np.save(
        OUTPUT_DIR / f"{dataset_name}_event_within_60m.npy",
        dataset_dict["event_within_60m"],
    )

    dataset_dict["metadata"].to_csv(
        OUTPUT_DIR / f"super_landmark_{dataset_name}_metadata.csv",
        index=False,
        encoding="utf-8-sig",
    )


save_super_landmark_dataset(
    "development",
    development_long,
)

save_super_landmark_dataset(
    "test",
    test_long,
)

summary_feature_table.to_csv(
    OUTPUT_DIR / "summary_feature_names.csv",
    index=False,
    encoding="utf-8-sig",
)

np.save(
    OUTPUT_DIR / "landmark_months.npy",
    landmark_months,
)
np.save(
    OUTPUT_DIR / "future_end_months.npy",
    future_end_months,
)


# ============================================================
# 9. 生成开发集五折长记录索引
# ============================================================

fold_summary_rows = []
fold_landmark_rows = []

development_fold_id_long = development_long["fold_id_long"]
development_global_patient_idx_long = development_long[
    "global_patient_idx"
]

development_patient_fold = fold_id_all[development_idx]

for fold_id in range(N_SPLITS):
    fold_dir = OUTPUT_DIR / f"fold_{fold_id}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    validation_patient_local_idx = np.flatnonzero(
        development_patient_fold == fold_id
    ).astype(np.int32)
    train_patient_local_idx = np.flatnonzero(
        development_patient_fold != fold_id
    ).astype(np.int32)

    validation_long_idx = np.flatnonzero(
        development_fold_id_long == fold_id
    ).astype(np.int32)
    train_long_idx = np.flatnonzero(
        development_fold_id_long != fold_id
    ).astype(np.int32)

    train_global_patients = np.unique(
        development_global_patient_idx_long[train_long_idx]
    )
    validation_global_patients = np.unique(
        development_global_patient_idx_long[validation_long_idx]
    )

    if np.intersect1d(
        train_global_patients,
        validation_global_patients,
    ).size > 0:
        raise ValueError(
            f"第{fold_id}折训练和验证长记录存在患者交叉。"
        )

    expected_train_global_patients = development_idx[
        train_patient_local_idx
    ]
    expected_validation_global_patients = development_idx[
        validation_patient_local_idx
    ]

    # 无有效预测起点的患者不会出现在长格式中，因此只要求出现的患者是预期集合的子集。
    if not np.isin(
        train_global_patients,
        expected_train_global_patients,
    ).all():
        raise ValueError(f"第{fold_id}折训练长记录患者归属错误。")

    if not np.isin(
        validation_global_patients,
        expected_validation_global_patients,
    ).all():
        raise ValueError(f"第{fold_id}折验证长记录患者归属错误。")

    np.save(
        fold_dir / "train_patient_local_idx.npy",
        train_patient_local_idx,
    )
    np.save(
        fold_dir / "validation_patient_local_idx.npy",
        validation_patient_local_idx,
    )
    np.save(
        fold_dir / "train_long_idx.npy",
        train_long_idx,
    )
    np.save(
        fold_dir / "validation_long_idx.npy",
        validation_long_idx,
    )

    fold_summary_rows.append({
        "fold_id": fold_id,
        "train_patient_n": int(len(train_patient_local_idx)),
        "validation_patient_n": int(len(validation_patient_local_idx)),
        "train_patient_with_valid_origin_n": int(len(train_global_patients)),
        "validation_patient_with_valid_origin_n": int(
            len(validation_global_patients)
        ),
        "train_long_record_n": int(len(train_long_idx)),
        "validation_long_record_n": int(len(validation_long_idx)),
        "train_event_within_60m_n": int(
            development_long["event_within_60m"][train_long_idx].sum()
        ),
        "validation_event_within_60m_n": int(
            development_long["event_within_60m"][validation_long_idx].sum()
        ),
        "patient_overlap_n": 0,
    })

    for landmark_index_value, landmark_month_value in enumerate(
        landmark_months
    ):
        train_landmark_mask = (
            development_long["landmark_index"][train_long_idx]
            == landmark_index_value
        )
        validation_landmark_mask = (
            development_long["landmark_index"][validation_long_idx]
            == landmark_index_value
        )

        fold_landmark_rows.append({
            "fold_id": fold_id,
            "landmark_month": int(landmark_month_value),
            "train_long_record_n": int(train_landmark_mask.sum()),
            "validation_long_record_n": int(
                validation_landmark_mask.sum()
            ),
            "train_event_within_60m_n": int(
                development_long["event_within_60m"][train_long_idx][
                    train_landmark_mask
                ].sum()
            ),
            "validation_event_within_60m_n": int(
                development_long["event_within_60m"][validation_long_idx][
                    validation_landmark_mask
                ].sum()
            ),
        })


# ============================================================
# 10. 生成总体和各Landmark核查表
# ============================================================

def summarize_long_dataset(dataset_name, dataset_dict, cohort_global_idx):
    """汇总长记录数、事件数、删失类型和未来区间标签。"""

    metadata = dataset_dict["metadata"]
    rows = []

    for landmark_index_value, landmark_month_value in enumerate(
        landmark_months
    ):
        landmark_rows = metadata[
            metadata["landmark_index"] == landmark_index_value
        ]

        expected_valid_origin_n = int(
            prediction_origin_mask_all[
                cohort_global_idx,
                landmark_index_value,
            ].sum()
        )

        if len(landmark_rows) != expected_valid_origin_n:
            raise ValueError(
                f"{dataset_name} Landmark {landmark_month_value}个月"
                "长记录数与有效预测起点数不一致。"
            )

        rows.append({
            "dataset": dataset_name,
            "landmark_month": int(landmark_month_value),
            "valid_origin_n": expected_valid_origin_n,
            "long_record_n": int(len(landmark_rows)),
            "unique_patient_n": int(landmark_rows["ID"].nunique()),
            "event_within_12m_n": int((
                (landmark_rows["original_event"] == 1)
                & (landmark_rows["remaining_time_month"] <= 12 + TIME_TOLERANCE)
            ).sum()),
            "event_within_36m_n": int((
                (landmark_rows["original_event"] == 1)
                & (landmark_rows["remaining_time_month"] <= 36 + TIME_TOLERANCE)
            ).sum()),
            "event_within_60m_n": int(
                landmark_rows["event_within_60m"].sum()
            ),
            "censor_before_60m_n": int(
                landmark_rows["censor_before_60m"].sum()
            ),
            "administrative_censor_60m_n": int(
                landmark_rows["administrative_censor_60m"].sum()
            ),
            "median_analysis_time_month": float(
                landmark_rows["analysis_time_month"].median()
            ),
            "median_future_at_risk_interval_n": float(
                landmark_rows["future_at_risk_interval_n"].median()
            ),
            "min_future_at_risk_interval_n": int(
                landmark_rows["future_at_risk_interval_n"].min()
            ),
            "max_future_at_risk_interval_n": int(
                landmark_rows["future_at_risk_interval_n"].max()
            ),
        })

    return rows


long_audit_rows = []
long_audit_rows.extend(
    summarize_long_dataset(
        "development",
        development_long,
        development_idx,
    )
)
long_audit_rows.extend(
    summarize_long_dataset(
        "test",
        test_long,
        test_idx,
    )
)

long_audit_table = pd.DataFrame(long_audit_rows)
fold_summary_table = pd.DataFrame(fold_summary_rows)
fold_landmark_table = pd.DataFrame(fold_landmark_rows)

long_audit_table.to_csv(
    OUTPUT_DIR / "super_landmark_audit.csv",
    index=False,
    encoding="utf-8-sig",
)
fold_summary_table.to_csv(
    OUTPUT_DIR / "fold_long_summary.csv",
    index=False,
    encoding="utf-8-sig",
)
fold_landmark_table.to_csv(
    OUTPUT_DIR / "fold_landmark_long_summary.csv",
    index=False,
    encoding="utf-8-sig",
)


# ============================================================
# 11. 最终完整性核查
# ============================================================

if not np.all(development_long["fold_id_long"] >= 0):
    raise ValueError("开发集长记录存在无效fold_id。")

if not np.all(test_long["fold_id_long"] == -1):
    raise ValueError("测试集长记录fold_id应全部为-1。")

if development_long["X_long"].shape[0] != int(
    prediction_origin_mask_all[development_idx].sum()
):
    raise ValueError("开发集长记录总数不正确。")

if test_long["X_long"].shape[0] != int(
    prediction_origin_mask_all[test_idx].sum()
):
    raise ValueError("测试集长记录总数不正确。")

# 使用行号映射重建有效起点掩码。
if not np.array_equal(
    development_long["row_index_map"] >= 0,
    prediction_origin_mask_all[development_idx],
):
    raise ValueError("开发集行号映射无法重建有效起点掩码。")

if not np.array_equal(
    test_long["row_index_map"] >= 0,
    prediction_origin_mask_all[test_idx],
):
    raise ValueError("测试集行号映射无法重建有效起点掩码。")

# 每个开发集患者的所有长记录必须只有一个fold_id。
development_fold_per_patient = (
    development_long["metadata"]
    .groupby("global_patient_index")["fold_id"]
    .nunique()
)

if (development_fold_per_patient > 1).any():
    raise ValueError("同一开发集患者的长记录被分配到多个折。")


# ============================================================
# 12. 保存配置和数据接口说明
# ============================================================

manifest = {
    "step": 6,
    "task": "build_super_landmark_long_format_data",
    "summary_feature_n": EXPECTED_SUMMARY_FEATURE_N,
    "development_long_shape": list(
        development_long["X_long"].shape
    ),
    "test_long_shape": list(test_long["X_long"].shape),
    "landmark_months": landmark_months.tolist(),
    "future_end_months": future_end_months.tolist(),
    "record_inclusion_rule": (
        "prediction_origin_mask=True; at risk after landmark, "
        "history available, and at least one complete future 6-month interval"
    ),
    "analysis_time_origin": "time since each landmark",
    "maximum_prediction_month": MAX_PREDICTION_MONTH,
    "event_definition": (
        "CKD event within 60 months after landmark"
    ),
    "administrative_censoring": (
        "remaining follow-up or CKD event beyond 60 months is censored at 60 months"
    ),
    "cross_validation_rule": (
        "patient-level fixed 5-fold; all landmarks from one patient stay in one fold"
    ),
    "test_usage_rule": (
        "test data are only structured and saved; no tuning or model selection"
    ),
    "cox_note": (
        "landmark_month will be used as the baseline-hazard stratum; "
        "the 146 clinical summary features remain the predictor matrix"
    ),
    "rsf_note": (
        "landmark_month will be added as an explicit predictor during RSF modeling"
    ),
    "prediction_restore_rule": (
        "use *_long_row_index_map.npy to restore long predictions to "
        "[patient, landmark, future interval]"
    ),
}

with open(
    OUTPUT_DIR / "step6_manifest.json",
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        manifest,
        file,
        ensure_ascii=False,
        indent=2,
    )


# ============================================================
# 13. 输出最终核查结果
# ============================================================

print("=" * 40)
print("Step 6完成：Super Landmark长格式数据已生成")
print("=" * 40)
print()
print(
    "开发集长格式特征：",
    development_long["X_long"].shape,
)
print(
    "锁定测试集长格式特征：",
    test_long["X_long"].shape,
)
print("汇总特征数：", EXPECTED_SUMMARY_FEATURE_N)
print()
print("Super Landmark核查：")
print(long_audit_table.to_string(index=False))
print()
print("五折长格式核查：")
print(fold_summary_table.to_string(index=False))
print()
print("处理原则：")
print("1. 仅prediction_origin_mask=True的患者-Landmark记录进入长格式。")
print("2. 生存时间从各Landmark重新起算，并在60个月行政截断。")
print("3. 60个月内CKD记为事件，60个月后的CKD不作为当前窗口事件。")
print("4. 精确生存结局与原10个半年离散标签已逐条核对一致。")
print("5. 同一患者全部Landmark记录固定属于同一交叉验证折。")
print("6. 测试集只完成结构化保存，不参与调参或模型选择。")
print()
print("输出目录：", OUTPUT_DIR)
print("Step 6运行完成。")
