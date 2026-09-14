
# ============================================================
# Step 2：固定开发集患者级五折交叉验证划分
#
# 代码逻辑：
# 1. 读取Step 1生成的患者信息、开发集索引和动态标签。
# 2. 仅在70%开发集中按“中心 × CKD结局”进行患者级分层五折。
# 3. 同一患者的全部Landmark始终位于同一折。
# 4. 保存固定fold编号和每折训练/验证患者索引。
# 5. 核对各折患者数、事件率、中心和Landmark风险集是否平衡。
# ============================================================

from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold


# ============================================================
# 1. 路径和固定参数
# ============================================================

STEP1_DIR = Path(
    "__CKD_WORKDIR__/rolling_5y_step1_new_split"
)

OUTPUT_DIR = Path(
    "__CKD_WORKDIR__/rolling_5y_step2_folds"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

N_SPLITS = 5
RANDOM_SEED = 20260727

EXPECTED_TOTAL_N = 31911
EXPECTED_DEVELOPMENT_N = 22337
EXPECTED_TEST_N = 9574
EXPECTED_EVENT_N = 2491


# ============================================================
# 2. 检查并读取Step 1输出
#
# 代码逻辑：
# 先核对Step 1目录及所有必需文件是否存在；
# 文件齐全后再开始读取，避免运行到中途才报错。
# ============================================================

required_step1_files = [
    "patient_info_new_split.csv",
    "development_idx.npy",
    "test_idx.npy",
    "landmark_months.npy",
    "landmark_eligible_mask.npy",
    "prediction_origin_mask.npy",
    "future_event_matrix.npy",
    "future_at_risk_mask.npy",
]

if not STEP1_DIR.exists():
    raise FileNotFoundError(
        f"Step 1输出目录不存在：{STEP1_DIR}"
    )

missing_files = [
    file_name
    for file_name in required_step1_files
    if not (STEP1_DIR / file_name).exists()
]

if missing_files:
    existing_files = sorted(
        path.name for path in STEP1_DIR.iterdir()
    )
    raise FileNotFoundError(
        "Step 1输出文件不完整。\n"
        f"缺少文件：{missing_files}\n"
        f"当前目录已有文件：{existing_files}"
    )

patient_info = pd.read_csv(
    STEP1_DIR / "patient_info_new_split.csv",
    encoding="utf-8-sig",
    dtype={"ID": "string", "center": "string"}
)

development_idx = np.load(
    STEP1_DIR / "development_idx.npy"
).astype(np.int32)

test_idx = np.load(
    STEP1_DIR / "test_idx.npy"
).astype(np.int32)

landmark_months = np.load(
    STEP1_DIR / "landmark_months.npy"
).astype(np.int32)

landmark_eligible_mask = np.load(
    STEP1_DIR / "landmark_eligible_mask.npy"
).astype(bool)

prediction_origin_mask = np.load(
    STEP1_DIR / "prediction_origin_mask.npy"
).astype(bool)

future_event_matrix = np.load(
    STEP1_DIR / "future_event_matrix.npy"
).astype(np.float32)

future_at_risk_mask = np.load(
    STEP1_DIR / "future_at_risk_mask.npy"
).astype(bool)


# ============================================================
# 3. 核对患者顺序、数据划分和数组维度
# ============================================================

required_cols = [
    "patient_index",
    "ID",
    "center",
    "analysis_split",
    "event",
    "observed_time_month"
]

missing_cols = [
    col for col in required_cols
    if col not in patient_info.columns
]

if missing_cols:
    raise ValueError(
        f"patient_info_new_split.csv缺少变量：{missing_cols}"
    )

n_patients = len(patient_info)
n_landmarks = len(landmark_months)

if n_patients != EXPECTED_TOTAL_N:
    raise ValueError(
        f"患者总数为{n_patients}，应为{EXPECTED_TOTAL_N}。"
    )

if len(development_idx) != EXPECTED_DEVELOPMENT_N:
    raise ValueError(
        f"开发集人数为{len(development_idx)}，"
        f"应为{EXPECTED_DEVELOPMENT_N}。"
    )

if len(test_idx) != EXPECTED_TEST_N:
    raise ValueError(
        f"测试集人数为{len(test_idx)}，"
        f"应为{EXPECTED_TEST_N}。"
    )

if int(patient_info["event"].sum()) != EXPECTED_EVENT_N:
    raise ValueError(
        f"CKD事件数为{int(patient_info['event'].sum())}，"
        f"应为{EXPECTED_EVENT_N}。"
    )

if not np.array_equal(
    patient_info["patient_index"].to_numpy(dtype=np.int32),
    np.arange(n_patients, dtype=np.int32)
):
    raise ValueError(
        "patient_info中的patient_index与当前行顺序不一致。"
    )

if np.intersect1d(development_idx, test_idx).size > 0:
    raise ValueError(
        "开发集与测试集患者存在交叉。"
    )

if len(development_idx) + len(test_idx) != n_patients:
    raise ValueError(
        "开发集和测试集未覆盖全部患者。"
    )

expected_shapes = {
    "landmark_eligible_mask": (n_patients, n_landmarks),
    "prediction_origin_mask": (n_patients, n_landmarks),
    "future_event_matrix": (n_patients, n_landmarks, 10),
    "future_at_risk_mask": (n_patients, n_landmarks, 10)
}

actual_shapes = {
    "landmark_eligible_mask": landmark_eligible_mask.shape,
    "prediction_origin_mask": prediction_origin_mask.shape,
    "future_event_matrix": future_event_matrix.shape,
    "future_at_risk_mask": future_at_risk_mask.shape
}

for name, expected_shape in expected_shapes.items():
    if actual_shapes[name] != expected_shape:
        raise ValueError(
            f"{name}形状为{actual_shapes[name]}，"
            f"应为{expected_shape}。"
        )

if not patient_info.loc[
    development_idx, "analysis_split"
].eq("development").all():
    raise ValueError(
        "development_idx中包含非开发集患者。"
    )

if not patient_info.loc[
    test_idx, "analysis_split"
].eq("test").all():
    raise ValueError(
        "test_idx中包含非测试集患者。"
    )


# ============================================================
# 4. 构建“中心 × CKD结局”分层变量
#
# 说明：
# 只在开发集中分层五折；
# 测试集保持锁定，不参与任何交叉验证。
# ============================================================

development_info = (
    patient_info.loc[development_idx]
    .copy()
    .reset_index(drop=True)
)

development_info["center"] = (
    development_info["center"]
    .astype("string")
    .str.strip()
)

development_info["event"] = pd.to_numeric(
    development_info["event"],
    errors="raise"
).astype(np.int8)

development_info["stratum"] = (
    development_info["center"].astype(str)
    + "_event"
    + development_info["event"].astype(str)
)

stratum_counts = (
    development_info["stratum"]
    .value_counts()
    .sort_index()
)

if (stratum_counts < N_SPLITS).any():
    raise ValueError(
        "部分“中心 × CKD结局”分层人数少于5，"
        "无法进行分层五折。"
    )

print("开发集分层人数：")
print(stratum_counts)


# ============================================================
# 5. 固定患者级五折
#
# 说明：
# fold_id为0～4；
# 每名开发集患者只属于一个验证折；
# 同一患者的全部Landmark自动属于同一折。
# ============================================================

skf = StratifiedKFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=RANDOM_SEED
)

development_fold_id = np.full(
    len(development_idx),
    -1,
    dtype=np.int8
)

for fold_id, (_, validation_position) in enumerate(
    skf.split(
        np.zeros(len(development_idx)),
        development_info["stratum"]
    )
):
    development_fold_id[validation_position] = fold_id

if (development_fold_id < 0).any():
    raise ValueError(
        "部分开发集患者未获得fold编号。"
    )

fold_counts = np.bincount(
    development_fold_id,
    minlength=N_SPLITS
)

if fold_counts.max() - fold_counts.min() > 1:
    raise ValueError(
        f"五折人数差异异常：{fold_counts.tolist()}"
    )


# ============================================================
# 6. 保存全体患者fold编号和每折索引
#
# 说明：
# 开发集fold_id为0～4；
# 锁定测试集fold_id固定为-1。
# ============================================================

fold_id_all = np.full(
    n_patients,
    -1,
    dtype=np.int8
)

fold_id_all[development_idx] = development_fold_id

patient_info_with_fold = patient_info.copy()
patient_info_with_fold["fold_id"] = fold_id_all

fold_manifest = {}

for fold_id in range(N_SPLITS):
    validation_idx = development_idx[
        development_fold_id == fold_id
    ].astype(np.int32)

    training_idx = development_idx[
        development_fold_id != fold_id
    ].astype(np.int32)

    if np.intersect1d(
        training_idx,
        validation_idx
    ).size > 0:
        raise ValueError(
            f"Fold {fold_id}训练集和验证集存在患者交叉。"
        )

    np.save(
        OUTPUT_DIR / f"fold_{fold_id}_train_idx.npy",
        training_idx
    )

    np.save(
        OUTPUT_DIR / f"fold_{fold_id}_validation_idx.npy",
        validation_idx
    )

    fold_manifest[str(fold_id)] = {
        "training_n": int(len(training_idx)),
        "validation_n": int(len(validation_idx))
    }


# ============================================================
# 7. 汇总各折总体和中心分布
# ============================================================

development_info["fold_id"] = development_fold_id

fold_summary = (
    development_info.groupby("fold_id")
    .agg(
        patient_n=("ID", "size"),
        event_n=("event", "sum"),
        event_rate=("event", "mean"),
        median_followup_month=(
            "observed_time_month",
            "median"
        )
    )
    .reset_index()
)

fold_center_summary = (
    development_info.groupby(
        ["fold_id", "center"],
        observed=True
    )
    .agg(
        patient_n=("ID", "size"),
        event_n=("event", "sum"),
        event_rate=("event", "mean"),
        median_followup_month=(
            "observed_time_month",
            "median"
        )
    )
    .reset_index()
    .sort_values(["fold_id", "center"])
)


# ============================================================
# 8. 汇总各折Landmark风险集
#
# 说明：
# known_12m、known_36m、known_60m表示相应时间点
# 已发生事件或已完整观察到该预测时间的患者数。
# ============================================================

horizon_intervals = {
    12: 2,
    36: 6,
    60: 10
}

landmark_rows = []

for fold_id in range(N_SPLITS):
    fold_global_idx = development_idx[
        development_fold_id == fold_id
    ]

    for landmark_index, landmark_month in enumerate(
        landmark_months
    ):
        row = {
            "fold_id": fold_id,
            "landmark_month": int(landmark_month),
            "eligible_n": int(
                landmark_eligible_mask[
                    fold_global_idx,
                    landmark_index
                ].sum()
            ),
            "valid_origin_n": int(
                prediction_origin_mask[
                    fold_global_idx,
                    landmark_index
                ].sum()
            ),
            "future_interval_label_n": int(
                future_at_risk_mask[
                    fold_global_idx,
                    landmark_index,
                    :
                ].sum()
            ),
            "positive_interval_label_n": int(
                future_event_matrix[
                    fold_global_idx,
                    landmark_index,
                    :
                ].sum()
            )
        }

        for horizon_month, interval_count in horizon_intervals.items():
            event_within_horizon = (
                future_event_matrix[
                    fold_global_idx,
                    landmark_index,
                    :interval_count
                ].sum(axis=1) > 0
            )

            known_at_horizon = (
                event_within_horizon
                |
                future_at_risk_mask[
                    fold_global_idx,
                    landmark_index,
                    interval_count - 1
                ]
            )

            row[f"event_{horizon_month}m_n"] = int(
                event_within_horizon.sum()
            )

            row[f"known_{horizon_month}m_n"] = int(
                known_at_horizon.sum()
            )

        landmark_rows.append(row)

fold_landmark_summary = pd.DataFrame(
    landmark_rows
)


# ============================================================
# 9. 最终完整性检查
#
# 说明：
# 每名开发集患者必须恰好作为一次验证患者；
# 测试集患者不能进入任何开发折。
# ============================================================

validation_count = np.zeros(
    n_patients,
    dtype=np.int8
)

for fold_id in range(N_SPLITS):
    validation_idx = np.load(
        OUTPUT_DIR
        / f"fold_{fold_id}_validation_idx.npy"
    )

    validation_count[validation_idx] += 1

if not np.all(
    validation_count[development_idx] == 1
):
    raise ValueError(
        "部分开发集患者未恰好作为一次验证患者。"
    )

if not np.all(
    validation_count[test_idx] == 0
):
    raise ValueError(
        "锁定测试集患者进入了开发集五折。"
    )


# ============================================================
# 10. 保存Step 2结果
# ============================================================

np.save(
    OUTPUT_DIR / "development_fold_id.npy",
    development_fold_id
)

np.save(
    OUTPUT_DIR / "fold_id_all.npy",
    fold_id_all
)

patient_info_with_fold.to_csv(
    OUTPUT_DIR / "patient_info_with_fold.csv",
    index=False,
    encoding="utf-8-sig"
)

fold_summary.to_csv(
    OUTPUT_DIR / "fold_summary.csv",
    index=False,
    encoding="utf-8-sig"
)

fold_center_summary.to_csv(
    OUTPUT_DIR / "fold_center_summary.csv",
    index=False,
    encoding="utf-8-sig"
)

fold_landmark_summary.to_csv(
    OUTPUT_DIR / "fold_landmark_summary.csv",
    index=False,
    encoding="utf-8-sig"
)

with open(
    OUTPUT_DIR / "fold_manifest.json",
    "w",
    encoding="utf-8"
) as file:
    json.dump(
        {
            "n_splits": N_SPLITS,
            "random_seed": RANDOM_SEED,
            "stratification": "center × CKD event",
            "development_n": int(len(development_idx)),
            "test_n": int(len(test_idx)),
            "folds": fold_manifest
        },
        file,
        ensure_ascii=False,
        indent=2
    )


# ============================================================
# 11. 输出最终结果
# ============================================================

print("\n========================================")
print("Step 2完成：开发集患者级五折已固定")
print("========================================")

print("\n五折总体分布：")
print(
    fold_summary.to_string(index=False)
)

print("\n五折中心分布：")
print(
    fold_center_summary.to_string(index=False)
)

print("\n五折Landmark风险集：")
print(
    fold_landmark_summary.to_string(index=False)
)

print("\n开发集患者数：", len(development_idx))
print("锁定测试集患者数：", len(test_idx))
print(
    "每名开发集患者恰好作为一次验证患者：",
    bool(
        np.all(
            validation_count[development_idx] == 1
        )
    )
)

print("\n输出目录：", OUTPUT_DIR)
print("Step 2运行完成。")
