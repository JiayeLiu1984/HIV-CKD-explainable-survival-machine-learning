# ============================================================
# Step 1：重新划分开发集/测试集并构建Landmark动态标签
#
# 代码逻辑：
# 1. 读取已经完成半年Landmark整理的纵向长表。
# 2. 本步骤只使用ID、中心、时间、结局等结构变量，不使用模型特征。
# 3. 忽略旧split，按患者级“中心 × CKD结局”重新分层随机划分7:3。
# 4. 设置0、12、24、36、48、60个月共6个Landmark。
# 5. 每个Landmark预测未来0～60个月，共10个半年区间。
# 6. 空缺半年时间行允许存在，由sequence_row_mask记录，不再要求历史时间行完整。
# 7. 保存固定划分、动态标签、风险掩码和核查表。
#
# 重要说明：
# - 本步骤不读取旧的标准化编码文件。
# - 本步骤不进行填补、标准化或One-Hot。
# - 新划分后的正式特征预处理将在后续训练折内完成。
# ============================================================

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


# ============================================================
# 1. 路径和固定参数
# 代码逻辑：仅需确认输入文件路径，其余参数固定后不再修改。
# ============================================================

INPUT_FILE = Path(
    "__CKD_WORKDIR__/深圳南宁随访数据表2.csv"
)

OUTPUT_DIR = Path(
    "__CKD_WORKDIR__/rolling_5y_step1_new_split"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 20260727
TEST_SIZE = 0.30
INTERVAL_WIDTH = 6.0
TIME_TOLERANCE = 1e-6

LANDMARK_MONTHS = np.array(
    [0, 12, 24, 36, 48, 60],
    dtype=np.int32,
)
LANDMARK_BINS = (LANDMARK_MONTHS / INTERVAL_WIDTH).astype(np.int32)

N_FUTURE_INTERVALS = 10
FUTURE_END_MONTHS = (
    np.arange(1, N_FUTURE_INTERVALS + 1) * INTERVAL_WIDTH
).astype(np.int32)

EXPECTED_TOTAL_N = 31911
EXPECTED_EVENT_N = 2491
EXPECTED_DEVELOPMENT_N = 22337
EXPECTED_TEST_N = 9574


# ============================================================
# 2. 读取纵向长表
# 代码逻辑：Step 1只读取构建划分和标签所需的结构变量。
# ============================================================

if not INPUT_FILE.exists():
    raise FileNotFoundError(f"未找到输入文件：{INPUT_FILE}")

usecols = [
    "ID",
    "data",
    "time_bin",
    "month",
    "CKDstatus",
    "interval",
]

try:
    df = pd.read_csv(
        INPUT_FILE,
        encoding="gb18030",
        low_memory=False,
        dtype={"ID": "string"},
        usecols=usecols,
    )
except UnicodeDecodeError:
    df = pd.read_csv(
        INPUT_FILE,
        encoding="utf-8-sig",
        low_memory=False,
        dtype={"ID": "string"},
        usecols=usecols,
    )

missing_cols = [col for col in usecols if col not in df.columns]
if missing_cols:
    raise ValueError(f"输入数据缺少必要变量：{missing_cols}")

print("原始长表形状：", df.shape)


# ============================================================
# 3. 整理结构变量格式
# 代码逻辑：只统一ID、中心、时间和结局格式，不改变临床特征。
# ============================================================

df["ID"] = (
    df["ID"]
    .astype("string")
    .str.strip()
    .str.replace(r"\.0$", "", regex=True)
)

df["data"] = df["data"].astype("string").str.strip()

df["time_bin"] = pd.to_numeric(df["time_bin"], errors="raise")
df["month"] = pd.to_numeric(df["month"], errors="raise")
df["CKDstatus"] = pd.to_numeric(df["CKDstatus"], errors="raise")
df["interval"] = pd.to_numeric(df["interval"], errors="raise")

if not np.allclose(df["time_bin"], np.round(df["time_bin"])):
    raise ValueError("time_bin中存在非整数值。")

df["time_bin"] = df["time_bin"].astype(np.int32)
df["CKDstatus"] = df["CKDstatus"].astype(np.int8)

df = (
    df.sort_values(["ID", "time_bin"])
    .reset_index(drop=True)
)


# ============================================================
# 4. 检查结构变量
# 代码逻辑：确认患者ID、时间、结局和随访时间可以用于标签构建。
# ============================================================

missing_summary = df[usecols].isna().sum()
missing_summary = missing_summary[missing_summary > 0]
if not missing_summary.empty:
    raise ValueError(f"结构变量中存在缺失值：\n{missing_summary}")

if df["ID"].eq("").any():
    raise ValueError("ID中存在空字符串。")

if df["data"].eq("").any():
    raise ValueError("data中存在空字符串。")

if not df["CKDstatus"].isin([0, 1]).all():
    raise ValueError("CKDstatus只能取0或1。")

if (df["interval"] <= 0).any():
    raise ValueError("interval中存在小于或等于0的值。")

if df.duplicated(["ID", "time_bin"]).any():
    duplicate_rows = df.loc[
        df.duplicated(["ID", "time_bin"], keep=False),
        ["ID", "time_bin"],
    ].head(20)
    raise ValueError(
        "存在重复的ID + time_bin，示例：\n"
        f"{duplicate_rows.to_string(index=False)}"
    )

if not np.allclose(
    df["month"].to_numpy(dtype=float),
    df["time_bin"].to_numpy(dtype=float) * INTERVAL_WIDTH,
):
    raise ValueError("month与time_bin不一致，应满足month = time_bin × 6。")

for col in ["data", "CKDstatus", "interval"]:
    inconsistent = df.groupby("ID")[col].nunique(dropna=False).gt(1)
    if inconsistent.any():
        bad_ids = inconsistent[inconsistent].index[:20].tolist()
        raise ValueError(
            f"同一患者的{col}不一致，示例ID：{bad_ids}"
        )


# ============================================================
# 5. 建立患者级信息表
# 代码逻辑：每位患者只保留一个中心、结局和随访时间。
# ============================================================

patient_info = (
    df.groupby("ID", sort=True)
    .agg(
        center=("data", "first"),
        event=("CKDstatus", "first"),
        observed_time_month=("interval", "first"),
        first_time_bin=("time_bin", "min"),
        last_time_bin=("time_bin", "max"),
        observed_row_n=("time_bin", "size"),
    )
    .reset_index()
)

patient_info.insert(
    0,
    "patient_index",
    np.arange(len(patient_info), dtype=np.int32),
)

n_patients = len(patient_info)
n_events = int(patient_info["event"].sum())

if n_patients != EXPECTED_TOTAL_N:
    raise ValueError(
        f"患者总数为{n_patients}，与预期{EXPECTED_TOTAL_N}不一致。"
    )

if n_events != EXPECTED_EVENT_N:
    raise ValueError(
        f"CKD事件数为{n_events}，与预期{EXPECTED_EVENT_N}不一致。"
    )

print("患者总数：", n_patients)
print("CKD事件数：", n_events)
print("中心分布：")
print(patient_info["center"].value_counts())


# ============================================================
# 6. 重新进行患者级7:3分层随机划分
# 代码逻辑：按“中心 × CKD结局”分层，固定随机种子，只划分一次。
# ============================================================

patient_info["split_stratum"] = (
    patient_info["center"].astype(str)
    + "__event_"
    + patient_info["event"].astype(str)
)

stratum_counts = patient_info["split_stratum"].value_counts().sort_index()
if stratum_counts.min() < 2:
    raise ValueError(
        "至少一个‘中心 × CKD结局’分层人数不足2人：\n"
        f"{stratum_counts}"
    )

all_indices = patient_info.index.to_numpy(dtype=np.int32)

development_idx, test_idx = train_test_split(
    all_indices,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    shuffle=True,
    stratify=patient_info["split_stratum"],
)

development_idx = np.sort(development_idx.astype(np.int32))
test_idx = np.sort(test_idx.astype(np.int32))

patient_info["analysis_split"] = ""
patient_info.loc[development_idx, "analysis_split"] = "development"
patient_info.loc[test_idx, "analysis_split"] = "test"

if len(development_idx) != EXPECTED_DEVELOPMENT_N:
    raise ValueError(
        f"开发集人数为{len(development_idx)}，"
        f"与预期{EXPECTED_DEVELOPMENT_N}不一致。"
    )

if len(test_idx) != EXPECTED_TEST_N:
    raise ValueError(
        f"测试集人数为{len(test_idx)}，与预期{EXPECTED_TEST_N}不一致。"
    )

if np.intersect1d(development_idx, test_idx).size > 0:
    raise ValueError("开发集和测试集患者索引存在交叉。")

if len(development_idx) + len(test_idx) != n_patients:
    raise ValueError("部分患者未进入开发集或测试集。")


# ============================================================
# 7. 核对新划分的患者、事件和中心分布
# 代码逻辑：确认开发集与测试集的事件率和中心构成基本一致。
# ============================================================

split_overall_summary = (
    patient_info.groupby("analysis_split", observed=True)
    .agg(
        patient_n=("ID", "size"),
        event_n=("event", "sum"),
        event_rate=("event", "mean"),
        median_followup_month=("observed_time_month", "median"),
    )
    .reset_index()
)

split_center_summary = (
    patient_info.groupby(
        ["analysis_split", "center"],
        observed=True,
    )
    .agg(
        patient_n=("ID", "size"),
        event_n=("event", "sum"),
        event_rate=("event", "mean"),
        median_followup_month=("observed_time_month", "median"),
    )
    .reset_index()
)

print("\n新7:3患者级划分：")
print(split_overall_summary.to_string(index=False))

print("\n按中心分层的划分结果：")
print(split_center_summary.to_string(index=False))


# ============================================================
# 8. 构建0～60个月历史时间行掩码
# 代码逻辑：空缺半年时间行保留为False，不再要求每个时间行都存在。
# ============================================================

patient_ids = patient_info["ID"].astype(str).to_numpy()
patient_to_index = {
    patient_id: index
    for index, patient_id in enumerate(patient_ids)
}

n_history_steps = int(LANDMARK_BINS.max()) + 1
sequence_row_mask = np.zeros(
    (n_patients, n_history_steps),
    dtype=bool,
)

df_history = df.loc[
    df["time_bin"].between(0, LANDMARK_BINS.max()),
    ["ID", "time_bin"],
].copy()

history_patient_idx = (
    df_history["ID"]
    .map(patient_to_index)
    .to_numpy(dtype=np.int32)
)
history_time_idx = df_history["time_bin"].to_numpy(dtype=np.int32)
sequence_row_mask[history_patient_idx, history_time_idx] = True

landmark_history_mask = np.zeros(
    (len(LANDMARK_MONTHS), n_history_steps),
    dtype=bool,
)
for landmark_index, landmark_bin in enumerate(LANDMARK_BINS):
    landmark_history_mask[
        landmark_index,
        : int(landmark_bin) + 1,
    ] = True

# 每位患者在每个Landmark前实际具有的历史时间行数
history_step_count = np.zeros(
    (n_patients, len(LANDMARK_MONTHS)),
    dtype=np.int16,
)

for landmark_index, landmark_bin in enumerate(LANDMARK_BINS):
    history_step_count[:, landmark_index] = (
        sequence_row_mask[:, : int(landmark_bin) + 1]
        .sum(axis=1)
        .astype(np.int16)
    )

# 至少存在一个Landmark及以前的可用历史时间行
history_available_mask = history_step_count > 0


# ============================================================
# 9. 初始化动态标签和风险掩码
# 代码逻辑：输出固定形状[患者, 6个Landmark, 10个未来半年区间]。
# ============================================================

event = patient_info["event"].to_numpy(dtype=np.int8)
observed_time_month = patient_info["observed_time_month"].to_numpy(
    dtype=np.float32
)
analysis_split = patient_info["analysis_split"].to_numpy()

n_landmarks = len(LANDMARK_MONTHS)

landmark_eligible_mask = np.zeros(
    (n_patients, n_landmarks),
    dtype=bool,
)
future_event_matrix = np.zeros(
    (n_patients, n_landmarks, N_FUTURE_INTERVALS),
    dtype=np.float32,
)
future_at_risk_mask = np.zeros(
    (n_patients, n_landmarks, N_FUTURE_INTERVALS),
    dtype=bool,
)


# ============================================================
# 10. 生成6个Landmark的未来离散生存标签
# 代码逻辑：
# 1. Landmark时仍在随访且尚未发生CKD。
# 2. Landmark及以前至少存在一个可用历史时间行。
# 3. 缺少中间半年时间行不排除患者，由sequence_row_mask处理。
# 4. 事件患者保留至事件区间；删失患者仅保留完整观察区间。
# ============================================================

for landmark_index, landmark_month in enumerate(LANDMARK_MONTHS):
    time_eligible = (
        observed_time_month
        > float(landmark_month) + TIME_TOLERANCE
    )

    history_available = history_available_mask[:, landmark_index]

    eligible_idx = np.where(
        time_eligible & history_available
    )[0]

    landmark_eligible_mask[
        eligible_idx,
        landmark_index,
    ] = True

    remaining_time = (
        observed_time_month[eligible_idx]
        - float(landmark_month)
    )
    eligible_event = event[eligible_idx].astype(bool)

    event_interval = np.full(
        len(eligible_idx),
        -1,
        dtype=np.int32,
    )

    event_interval[eligible_event] = (
        np.ceil(
            (
                remaining_time[eligible_event]
                - TIME_TOLERANCE
            )
            / INTERVAL_WIDTH
        ).astype(np.int32)
        - 1
    )

    if (event_interval[eligible_event] < 0).any():
        raise ValueError(
            f"Landmark {landmark_month}个月后"
            "存在事件区间小于0的患者。"
        )

    for future_index in range(N_FUTURE_INTERVALS):
        interval_end = (
            future_index + 1
        ) * INTERVAL_WIDTH

        # 事件患者在事件发生区间及以前均处于风险中
        event_at_risk = (
            eligible_event
            & (event_interval >= future_index)
        )

        # 未发生事件者仅保留完整观察到区间末的标签
        censor_at_risk = (
            ~eligible_event
            & (
                remaining_time
                >= interval_end - TIME_TOLERANCE
            )
        )

        future_at_risk_mask[
            eligible_idx,
            landmark_index,
            future_index,
        ] = event_at_risk | censor_at_risk

        event_here = (
            eligible_event
            & (event_interval == future_index)
        )

        future_event_matrix[
            eligible_idx,
            landmark_index,
            future_index,
        ] = event_here.astype(np.float32)


# ============================================================
# 11. 生成有效预测起点并检查标签
# 代码逻辑：至少有一个完整未来半年区间时，该Landmark才用于建模。
# ============================================================

prediction_origin_mask = (
    landmark_eligible_mask
    & future_at_risk_mask.any(axis=2)
)

if np.any(
    (future_event_matrix > 0)
    & (~future_at_risk_mask)
):
    raise ValueError("存在事件标签位于风险掩码之外。")

if (future_event_matrix.sum(axis=2) > 1).any():
    raise ValueError("同一患者同一Landmark出现多个事件区间。")

if not np.array_equal(
    prediction_origin_mask,
    future_at_risk_mask.any(axis=2),
):
    raise ValueError(
        "prediction_origin_mask与"
        "future_at_risk_mask不一致。"
    )


# ============================================================
# 12. 生成Landmark风险集核查表
# 代码逻辑：分别统计总体、开发集和测试集的历史可用性及1/3/5年结果。
# ============================================================

cohort_masks = {
    "overall": np.ones(n_patients, dtype=bool),
    "development": analysis_split == "development",
    "test": analysis_split == "test",
}

risk_set_rows = []

for landmark_index, landmark_month in enumerate(LANDMARK_MONTHS):
    for cohort_name, cohort_mask in cohort_masks.items():
        followed_at_landmark = (
            observed_time_month
            > float(landmark_month) + TIME_TOLERANCE
        ) & cohort_mask

        history_available = (
            history_available_mask[:, landmark_index]
            & cohort_mask
        )

        eligible = (
            landmark_eligible_mask[:, landmark_index]
            & cohort_mask
        )

        valid_origin = (
            prediction_origin_mask[:, landmark_index]
            & cohort_mask
        )

        row = {
            "cohort": cohort_name,
            "landmark_month": int(landmark_month),
            "followed_at_landmark_n": int(
                followed_at_landmark.sum()
            ),
            "history_available_n": int(
                history_available.sum()
            ),
            "no_history_at_landmark_n": int(
                (
                    followed_at_landmark
                    & ~history_available_mask[:, landmark_index]
                ).sum()
            ),
            "eligible_n": int(eligible.sum()),
            "valid_origin_n": int(valid_origin.sum()),
            "no_complete_future_interval_n": int(
                (eligible & ~valid_origin).sum()
            ),
            "median_history_step_n": float(
                np.median(
                    history_step_count[
                        eligible,
                        landmark_index,
                    ]
                )
            ) if eligible.any() else np.nan,
            "future_interval_label_n": int(
                future_at_risk_mask[
                    cohort_mask,
                    landmark_index,
                    :,
                ].sum()
            ),
            "positive_interval_label_n": int(
                future_event_matrix[
                    cohort_mask,
                    landmark_index,
                    :,
                ].sum()
            ),
        }

        for horizon_month in [12, 36, 60]:
            horizon_end = landmark_month + horizon_month

            event_within_horizon = (
                eligible
                & (event == 1)
                & (
                    observed_time_month
                    <= horizon_end + TIME_TOLERANCE
                )
            )

            known_at_horizon = (
                eligible
                & (
                    event_within_horizon
                    | (
                        observed_time_month
                        >= horizon_end - TIME_TOLERANCE
                    )
                )
            )

            censored_before_horizon = (
                eligible
                & (event == 0)
                & (
                    observed_time_month
                    < horizon_end - TIME_TOLERANCE
                )
            )

            row[f"event_{horizon_month}m_n"] = int(
                event_within_horizon.sum()
            )
            row[f"known_{horizon_month}m_n"] = int(
                known_at_horizon.sum()
            )
            row[f"censored_before_{horizon_month}m_n"] = int(
                censored_before_horizon.sum()
            )

        risk_set_rows.append(row)

landmark_risk_set_summary = pd.DataFrame(risk_set_rows)


# ============================================================
# 13. 生成历史时间行核查表
# 代码逻辑：逐Landmark统计患者拥有0、1、2……个历史时间行的分布。
# ============================================================

history_availability_rows = []

for landmark_index, landmark_month in enumerate(LANDMARK_MONTHS):
    max_possible_steps = int(LANDMARK_BINS[landmark_index]) + 1

    for cohort_name, cohort_mask in cohort_masks.items():
        counts = history_step_count[
            cohort_mask,
            landmark_index,
        ]

        for available_step_n in range(max_possible_steps + 1):
            history_availability_rows.append({
                "cohort": cohort_name,
                "landmark_month": int(landmark_month),
                "available_history_step_n": int(available_step_n),
                "patient_n": int(
                    (counts == available_step_n).sum()
                ),
            })

history_availability_summary = pd.DataFrame(
    history_availability_rows
)


# ============================================================
# 14. 生成未来半年区间核查表
# 代码逻辑：逐Landmark、逐半年区间统计风险人数和事件人数。
# ============================================================

future_interval_rows = []

for landmark_index, landmark_month in enumerate(LANDMARK_MONTHS):
    for future_index, future_end_month in enumerate(FUTURE_END_MONTHS):
        for cohort_name, cohort_mask in cohort_masks.items():
            future_interval_rows.append({
                "cohort": cohort_name,
                "landmark_month": int(landmark_month),
                "future_interval_index": int(future_index),
                "future_start_month": int(
                    future_end_month - INTERVAL_WIDTH
                ),
                "future_end_month": int(future_end_month),
                "at_risk_n": int(
                    future_at_risk_mask[
                        cohort_mask,
                        landmark_index,
                        future_index,
                    ].sum()
                ),
                "event_n": int(
                    future_event_matrix[
                        cohort_mask,
                        landmark_index,
                        future_index,
                    ].sum()
                ),
            })

future_interval_summary = pd.DataFrame(
    future_interval_rows
)


# ============================================================
# 15. 保存正式输出
# 代码逻辑：保存患者划分、Landmark标签、掩码和核查结果。
# ============================================================

arrays_to_save = {
    "development_idx": development_idx,
    "test_idx": test_idx,
    "event": event,
    "observed_time_month": observed_time_month,
    "landmark_months": LANDMARK_MONTHS,
    "landmark_bins": LANDMARK_BINS,
    "future_end_months": FUTURE_END_MONTHS,
    "sequence_row_mask": sequence_row_mask,
    "landmark_history_mask": landmark_history_mask,
    "history_step_count": history_step_count,
    "history_available_mask": history_available_mask,
    "landmark_eligible_mask": landmark_eligible_mask,
    "prediction_origin_mask": prediction_origin_mask,
    "future_event_matrix": future_event_matrix,
    "future_at_risk_mask": future_at_risk_mask,
}

for file_name, array in arrays_to_save.items():
    np.save(
        OUTPUT_DIR / f"{file_name}.npy",
        array,
    )

patient_info.to_csv(
    OUTPUT_DIR / "patient_info_new_split.csv",
    index=False,
    encoding="utf-8-sig",
)

patient_info[
    [
        "patient_index",
        "ID",
        "center",
        "event",
        "observed_time_month",
        "analysis_split",
    ]
].to_csv(
    OUTPUT_DIR / "patient_split_new_7_3.csv",
    index=False,
    encoding="utf-8-sig",
)

split_overall_summary.to_csv(
    OUTPUT_DIR / "split_overall_summary.csv",
    index=False,
    encoding="utf-8-sig",
)

split_center_summary.to_csv(
    OUTPUT_DIR / "split_center_summary.csv",
    index=False,
    encoding="utf-8-sig",
)

landmark_risk_set_summary.to_csv(
    OUTPUT_DIR / "landmark_risk_set_summary.csv",
    index=False,
    encoding="utf-8-sig",
)

history_availability_summary.to_csv(
    OUTPUT_DIR / "history_availability_summary.csv",
    index=False,
    encoding="utf-8-sig",
)

future_interval_summary.to_csv(
    OUTPUT_DIR / "future_interval_summary.csv",
    index=False,
    encoding="utf-8-sig",
)

config = {
    "input_file": str(INPUT_FILE),
    "output_dir": str(OUTPUT_DIR),
    "random_state": RANDOM_STATE,
    "test_size": TEST_SIZE,
    "interval_width_month": INTERVAL_WIDTH,
    "landmark_months": LANDMARK_MONTHS.tolist(),
    "future_end_months": FUTURE_END_MONTHS.tolist(),
    "n_patients": int(n_patients),
    "n_events": int(n_events),
    "n_development": int(len(development_idx)),
    "n_test": int(len(test_idx)),
    "standardized_data_used": False,
    "feature_preprocessing_performed": False,
}

with open(
    OUTPUT_DIR / "step1_config.json",
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        config,
        file,
        ensure_ascii=False,
        indent=2,
    )


# ============================================================
# 16. 输出最终核查结果
# 代码逻辑：确认划分、数组形状、风险集和保存位置均正确。
# ============================================================

print("\n========================================")
print("Step 1完成：新7:3划分和Landmark标签已生成")
print("========================================")

print("开发集患者数：", len(development_idx))
print("锁定测试集患者数：", len(test_idx))
print("患者级CKD事件数：", int(event.sum()))

print("\nsequence_row_mask形状：", sequence_row_mask.shape)
print("history_step_count形状：", history_step_count.shape)
print("future_event_matrix形状：", future_event_matrix.shape)
print("future_at_risk_mask形状：", future_at_risk_mask.shape)
print("landmark_eligible_mask形状：", landmark_eligible_mask.shape)
print("prediction_origin_mask形状：", prediction_origin_mask.shape)

print(
    "\n有效预测起点总数：",
    int(prediction_origin_mask.sum()),
)
print(
    "有效未来区间标签总数：",
    int(future_at_risk_mask.sum()),
)
print(
    "阳性未来区间标签总数：",
    int(future_event_matrix.sum()),
)

print("\n各Landmark总体风险集：")
print(
    landmark_risk_set_summary.loc[
        landmark_risk_set_summary["cohort"].eq("overall")
    ].to_string(index=False)
)

print("\n注意：空缺半年时间行已保留并由sequence_row_mask表示。")
print("本步骤未读取旧标准化数据，也未进行特征预处理。")
print("\n输出目录：", OUTPUT_DIR)
print("Step 1运行完成。")
