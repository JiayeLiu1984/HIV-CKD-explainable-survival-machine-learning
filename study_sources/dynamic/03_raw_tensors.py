# ============================================================
# Step 3：构建统一的0～60个月原始特征张量
#
# 代码逻辑：
# 1. 读取已完成半年整理和缺失处理的原始长表。
# 2. 沿用Step 1的新7:3划分和Step 2固定五折。
# 3. 构建0、6、12……60个月共11个时间点的原始特征张量。
# 4. 空缺半年时间行保留，由sequence_row_mask识别。
# 5. 本步骤不标准化、不One-Hot、不训练模型。
# ============================================================

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder


# ============================================================
# 1. 路径和固定参数
# ============================================================

INPUT_FILE = Path(
    "__CKD_WORKDIR__/深圳南宁随访数据表2.csv"
)

STEP1_DIR = Path(
    "__CKD_WORKDIR__/rolling_5y_step1_new_split"
)

STEP2_DIR = Path(
    "__CKD_WORKDIR__/rolling_5y_step2_folds"
)

OUTPUT_DIR = Path(
    "__CKD_WORKDIR__/rolling_5y_step3_raw_features"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

INTERVAL_WIDTH = 6
N_HISTORY_STEPS = 11
EXPECTED_TOTAL_N = 31911
EXPECTED_DEVELOPMENT_N = 22337
EXPECTED_TEST_N = 9574
EXPECTED_ONEHOT_N = 14
EXPECTED_FINAL_FEATURE_N = 57
PADDING_CATEGORY = "__NO_TIME_ROW__"


# ============================================================
# 2. 固定原始候选变量
# ============================================================

continuous_vars = [
    "Age", "BMI", "HIVRNA_log10", "CD4", "CD8", "Urea", "WBC",
    "PLT", "HB", "TC", "TG", "HDL", "LDL", "GLU", "ALT", "AST",
    "eGFR", "TDF_NNRTI_3TC_FTC_cum_month",
    "TDF_PI_3TC_FTC_cum_month", "nonTDF_PI_cum_month",
    "BIC_FTC_TAF_cum_month", "EVGc_FTC_TAF_cum_month",
    "TDF_INSTI_3TC_FTC_cum_month", "nonTDF_DTG_cum_month",
    "nonTDF_traditional_NNRTI_cum_month",
]

binary_vars = [
    "Oppinfection", "CVD_status", "diabetes_status",
    "hypertension_status", "hypercholesterolemia_status",
    "antidiabetic_med", "antihypertensive_med", "antilipid_med",
    "HBV_status", "HCV_status", "current_TDF_NNRTI_3TC_FTC",
    "current_TDF_PI_3TC_FTC", "current_nonTDF_PI",
    "current_BIC_FTC_TAF", "current_EVGc_FTC_TAF",
    "current_TDF_INSTI_3TC_FTC", "current_nonTDF_DTG",
    "current_nonTDF_traditional_NNRTI",
]

categorical_vars = ["Sex", "Marriage", "Course", "WHOstage"]

structure_vars = [
    "ID", "data", "time_bin", "month", "CKDstatus", "interval"
]

raw_feature_vars = continuous_vars + binary_vars + categorical_vars

if len(continuous_vars) != 25:
    raise ValueError("连续变量数不是25。")
if len(binary_vars) != 18:
    raise ValueError("二分类变量数不是18。")
if len(categorical_vars) != 4:
    raise ValueError("多分类变量数不是4。")


# ============================================================
# 3. 核对并读取Step 1和Step 2输出
# ============================================================

required_files = [
    INPUT_FILE,
    STEP1_DIR / "sequence_row_mask.npy",
    STEP1_DIR / "development_idx.npy",
    STEP1_DIR / "test_idx.npy",
    STEP2_DIR / "patient_info_with_fold.csv",
    STEP2_DIR / "fold_id_all.npy",
]

missing_files = [str(path) for path in required_files if not path.exists()]
if missing_files:
    raise FileNotFoundError(
        "以下输入文件不存在：\n" + "\n".join(missing_files)
    )

patient_info = pd.read_csv(
    STEP2_DIR / "patient_info_with_fold.csv",
    encoding="utf-8-sig",
    dtype={
        "ID": "string",
        "center": "string",
        "analysis_split": "string",
    },
)

sequence_row_mask = np.load(
    STEP1_DIR / "sequence_row_mask.npy"
).astype(bool)

development_idx = np.load(
    STEP1_DIR / "development_idx.npy"
).astype(np.int32)

test_idx = np.load(
    STEP1_DIR / "test_idx.npy"
).astype(np.int32)

fold_id_all = np.load(
    STEP2_DIR / "fold_id_all.npy"
).astype(np.int8)

required_patient_cols = [
    "patient_index", "ID", "center", "analysis_split",
    "event", "observed_time_month", "fold_id",
]
missing_cols = [
    col for col in required_patient_cols if col not in patient_info.columns
]
if missing_cols:
    raise ValueError(
        f"patient_info_with_fold.csv缺少变量：{missing_cols}"
    )

n_patients = len(patient_info)

if n_patients != EXPECTED_TOTAL_N:
    raise ValueError(
        f"患者数为{n_patients}，应为{EXPECTED_TOTAL_N}。"
    )
if len(development_idx) != EXPECTED_DEVELOPMENT_N:
    raise ValueError(
        f"开发集人数为{len(development_idx)}，"
        f"应为{EXPECTED_DEVELOPMENT_N}。"
    )
if len(test_idx) != EXPECTED_TEST_N:
    raise ValueError(
        f"测试集人数为{len(test_idx)}，应为{EXPECTED_TEST_N}。"
    )
if sequence_row_mask.shape != (n_patients, N_HISTORY_STEPS):
    raise ValueError(
        f"sequence_row_mask形状为{sequence_row_mask.shape}。"
    )
if not np.array_equal(
    patient_info["patient_index"].to_numpy(dtype=np.int32),
    np.arange(n_patients, dtype=np.int32),
):
    raise ValueError("patient_index与患者表行顺序不一致。")
if not np.array_equal(
    patient_info["fold_id"].to_numpy(dtype=np.int8),
    fold_id_all,
):
    raise ValueError("患者表fold_id与fold_id_all.npy不一致。")
if not np.all(fold_id_all[development_idx] >= 0):
    raise ValueError("部分开发集患者没有五折编号。")
if not np.all(fold_id_all[test_idx] == -1):
    raise ValueError("锁定测试集患者进入了开发集五折。")


# ============================================================
# 4. 读取未标准化完整长表
# ============================================================

usecols = structure_vars + raw_feature_vars

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
    raise ValueError(f"原始长表缺少变量：{missing_cols}")

print("读取长表形状：", df.shape)


# ============================================================
# 5. 整理格式并检查原始候选变量
# ============================================================

df["ID"] = (
    df["ID"].astype("string").str.strip()
    .str.replace(r"\.0$", "", regex=True)
)
df["data"] = df["data"].astype("string").str.strip()

numeric_vars = [
    "time_bin", "month", "CKDstatus", "interval"
] + continuous_vars + binary_vars

df[numeric_vars] = df[numeric_vars].apply(
    pd.to_numeric, errors="raise"
)

if not np.allclose(df["time_bin"], np.round(df["time_bin"])):
    raise ValueError("time_bin中存在非整数值。")

df["time_bin"] = df["time_bin"].astype(np.int32)
df["CKDstatus"] = df["CKDstatus"].astype(np.int8)

for col in categorical_vars:
    df[col] = df[col].astype("string").str.strip()

df = df.sort_values(["ID", "time_bin"]).reset_index(drop=True)

if df.duplicated(["ID", "time_bin"]).any():
    raise ValueError("存在重复的ID + time_bin。")

if not np.allclose(
    df["month"].to_numpy(dtype=float),
    df["time_bin"].to_numpy(dtype=float) * INTERVAL_WIDTH,
):
    raise ValueError("month与time_bin不一致。")

missing_counts = df[raw_feature_vars].isna().sum()
missing_counts = missing_counts[missing_counts > 0]
if not missing_counts.empty:
    raise ValueError(
        "候选变量中仍存在缺失值：\n"
        f"{missing_counts.to_string()}"
    )

if not np.isfinite(
    df[continuous_vars].to_numpy(dtype=np.float64)
).all():
    raise ValueError("连续变量中存在NaN或无穷值。")

for col in binary_vars:
    values = set(df[col].unique().tolist())
    if not values.issubset({0, 1}):
        raise ValueError(
            f"{col}不是标准0/1变量，当前取值：{sorted(values)}"
        )

for col in categorical_vars:
    if df[col].eq("").any():
        raise ValueError(f"{col}中存在空字符串。")


# ============================================================
# 6. 将长表连接到固定患者顺序
# ============================================================

patient_id_to_index = dict(
    zip(
        patient_info["ID"].astype(str),
        patient_info["patient_index"].astype(int),
    )
)

df["patient_index"] = (
    df["ID"].astype(str).map(patient_id_to_index)
)

if df["patient_index"].isna().any():
    bad_ids = (
        df.loc[df["patient_index"].isna(), "ID"]
        .drop_duplicates().head(20).tolist()
    )
    raise ValueError(
        f"存在Step 2患者表未包含的ID：{bad_ids}"
    )

df["patient_index"] = df["patient_index"].astype(np.int32)

raw_patient_n = df["ID"].nunique()
if raw_patient_n != n_patients:
    raise ValueError(
        f"长表患者数为{raw_patient_n}，患者表为{n_patients}。"
    )

raw_patient_info = (
    df.groupby("ID", sort=True)
    .agg(
        center=("data", "first"),
        event=("CKDstatus", "first"),
        observed_time_month=("interval", "first"),
    )
    .reset_index()
)

check = raw_patient_info.merge(
    patient_info[
        ["ID", "center", "event", "observed_time_month"]
    ],
    on="ID",
    how="outer",
    suffixes=("_raw", "_saved"),
    validate="one_to_one",
)

if check.isna().any().any():
    raise ValueError("长表与患者表的ID集合不一致。")
if not check["center_raw"].eq(check["center_saved"]).all():
    raise ValueError("长表与患者表的中心不一致。")
if not np.array_equal(
    check["event_raw"].to_numpy(dtype=np.int8),
    check["event_saved"].to_numpy(dtype=np.int8),
):
    raise ValueError("长表与患者表的CKD结局不一致。")
if not np.allclose(
    check["observed_time_month_raw"].to_numpy(dtype=float),
    check["observed_time_month_saved"].to_numpy(dtype=float),
):
    raise ValueError("长表与患者表的随访时间不一致。")


# ============================================================
# 7. 构建0～60个月原始特征张量
# ============================================================

df_history = df.loc[
    df["time_bin"].between(0, N_HISTORY_STEPS - 1)
].copy()

continuous_raw = np.full(
    (n_patients, N_HISTORY_STEPS, len(continuous_vars)),
    np.nan,
    dtype=np.float32,
)

binary_raw = np.full(
    (n_patients, N_HISTORY_STEPS, len(binary_vars)),
    np.nan,
    dtype=np.float32,
)

categorical_raw = np.full(
    (n_patients, N_HISTORY_STEPS, len(categorical_vars)),
    PADDING_CATEGORY,
    dtype="<U64",
)

observed_row_mask = np.zeros(
    (n_patients, N_HISTORY_STEPS),
    dtype=bool,
)

patient_index = df_history["patient_index"].to_numpy(dtype=np.int32)
time_index = df_history["time_bin"].to_numpy(dtype=np.int32)

continuous_raw[patient_index, time_index, :] = (
    df_history[continuous_vars].to_numpy(dtype=np.float32)
)
binary_raw[patient_index, time_index, :] = (
    df_history[binary_vars].to_numpy(dtype=np.float32)
)
categorical_raw[patient_index, time_index, :] = (
    df_history[categorical_vars].astype(str).to_numpy(dtype="<U64")
)
observed_row_mask[patient_index, time_index] = True


# ============================================================
# 8. 核对时间行掩码和张量内容
# ============================================================

if not np.array_equal(observed_row_mask, sequence_row_mask):
    mismatch = np.argwhere(
        observed_row_mask != sequence_row_mask
    )[:20]
    raise ValueError(
        "Step 3时间行掩码与Step 1不一致，"
        f"示例位置：{mismatch.tolist()}"
    )

if not np.isfinite(
    continuous_raw[observed_row_mask]
).all():
    raise ValueError("真实时间行连续特征存在异常值。")

if not np.isfinite(
    binary_raw[observed_row_mask]
).all():
    raise ValueError("真实时间行二分类特征存在异常值。")

if not np.isin(
    binary_raw[observed_row_mask], [0.0, 1.0]
).all():
    raise ValueError("真实时间行二分类特征存在非0/1值。")

if np.any(
    categorical_raw[observed_row_mask] == PADDING_CATEGORY
):
    raise ValueError("真实时间行多分类特征出现填充标记。")

if not np.isnan(
    continuous_raw[~observed_row_mask]
).all():
    raise ValueError("空缺时间行连续特征不是NaN。")

if not np.isnan(
    binary_raw[~observed_row_mask]
).all():
    raise ValueError("空缺时间行二分类特征不是NaN。")

if not np.all(
    categorical_raw[~observed_row_mask] == PADDING_CATEGORY
):
    raise ValueError("空缺时间行多分类填充不一致。")


# ============================================================
# 9. 使用开发集核对One-Hot类别空间
#
# 说明：
# 本段只核对最终应有14个One-Hot变量；
# 正式标准化和One-Hot仍在后续训练折内完成。
# ============================================================

development_patient_mask = np.zeros(n_patients, dtype=bool)
development_patient_mask[development_idx] = True

development_row_mask = (
    observed_row_mask & development_patient_mask[:, None]
)
development_categories = categorical_raw[development_row_mask]

try:
    encoder = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False,
        dtype=np.float32,
    )
except TypeError:
    encoder = OneHotEncoder(
        handle_unknown="ignore",
        sparse=False,
        dtype=np.float32,
    )

encoder.fit(development_categories)

onehot_feature_names = (
    encoder.get_feature_names_out(categorical_vars).tolist()
)
onehot_n = len(onehot_feature_names)
final_feature_n = len(continuous_vars) + len(binary_vars) + onehot_n

if onehot_n != EXPECTED_ONEHOT_N:
    raise ValueError(
        f"开发集One-Hot变量数为{onehot_n}，"
        f"应为{EXPECTED_ONEHOT_N}。\n"
        f"实际变量：{onehot_feature_names}"
    )

if final_feature_n != EXPECTED_FINAL_FEATURE_N:
    raise ValueError(
        f"最终特征数为{final_feature_n}，"
        f"应为{EXPECTED_FINAL_FEATURE_N}。"
    )


# ============================================================
# 10. 检查测试集是否出现开发集未见类别
# ============================================================

test_patient_mask = np.zeros(n_patients, dtype=bool)
test_patient_mask[test_idx] = True
test_row_mask = observed_row_mask & test_patient_mask[:, None]
test_categories = categorical_raw[test_row_mask]

category_rows = []
unknown_rows = []

for j, variable in enumerate(categorical_vars):
    development_levels = sorted(
        set(development_categories[:, j].tolist())
    )
    test_levels = sorted(set(test_categories[:, j].tolist()))
    unknown_levels = sorted(
        set(test_levels) - set(development_levels)
    )

    for level in development_levels:
        category_rows.append({
            "variable": variable,
            "category": level,
            "source": "development",
        })

    for level in unknown_levels:
        unknown_rows.append({
            "variable": variable,
            "category": level,
        })

category_level_summary = pd.DataFrame(category_rows)
unknown_category_summary = pd.DataFrame(
    unknown_rows,
    columns=["variable", "category"],
)


# ============================================================
# 11. 生成核查表
# ============================================================

tensor_summary = pd.DataFrame({
    "item": [
        "patient_n",
        "history_step_n",
        "continuous_n",
        "binary_n",
        "categorical_n",
        "onehot_n",
        "final_feature_n",
        "observed_time_row_n",
        "missing_time_row_n",
    ],
    "value": [
        n_patients,
        N_HISTORY_STEPS,
        len(continuous_vars),
        len(binary_vars),
        len(categorical_vars),
        onehot_n,
        final_feature_n,
        int(observed_row_mask.sum()),
        int((~observed_row_mask).sum()),
    ],
})

split_rows = []
for split_name, indices in [
    ("development", development_idx),
    ("test", test_idx),
]:
    split_rows.append({
        "analysis_split": split_name,
        "patient_n": int(len(indices)),
        "event_n": int(patient_info.loc[indices, "event"].sum()),
        "observed_time_row_n": int(observed_row_mask[indices].sum()),
        "missing_time_row_n": int((~observed_row_mask[indices]).sum()),
    })

split_tensor_summary = pd.DataFrame(split_rows)

fold_rows = []
for fold_id in range(5):
    indices = np.where(fold_id_all == fold_id)[0]
    fold_rows.append({
        "fold_id": fold_id,
        "patient_n": int(len(indices)),
        "event_n": int(patient_info.loc[indices, "event"].sum()),
        "observed_time_row_n": int(observed_row_mask[indices].sum()),
        "missing_time_row_n": int((~observed_row_mask[indices]).sum()),
    })

fold_tensor_summary = pd.DataFrame(fold_rows)


# ============================================================
# 12. 保存Step 3输出
# ============================================================

np.save(
    OUTPUT_DIR / "continuous_raw_0_60.npy",
    continuous_raw,
)
np.save(
    OUTPUT_DIR / "binary_raw_0_60.npy",
    binary_raw,
)
np.save(
    OUTPUT_DIR / "categorical_raw_0_60.npy",
    categorical_raw,
)
np.save(
    OUTPUT_DIR / "sequence_row_mask.npy",
    observed_row_mask,
)
np.save(
    OUTPUT_DIR / "development_idx.npy",
    development_idx,
)
np.save(
    OUTPUT_DIR / "test_idx.npy",
    test_idx,
)
np.save(
    OUTPUT_DIR / "fold_id_all.npy",
    fold_id_all,
)

patient_info.to_csv(
    OUTPUT_DIR / "patient_info_with_fold.csv",
    index=False,
    encoding="utf-8-sig",
)
tensor_summary.to_csv(
    OUTPUT_DIR / "tensor_summary.csv",
    index=False,
    encoding="utf-8-sig",
)
split_tensor_summary.to_csv(
    OUTPUT_DIR / "split_tensor_summary.csv",
    index=False,
    encoding="utf-8-sig",
)
fold_tensor_summary.to_csv(
    OUTPUT_DIR / "fold_tensor_summary.csv",
    index=False,
    encoding="utf-8-sig",
)
category_level_summary.to_csv(
    OUTPUT_DIR / "category_level_summary.csv",
    index=False,
    encoding="utf-8-sig",
)
unknown_category_summary.to_csv(
    OUTPUT_DIR / "unknown_test_categories.csv",
    index=False,
    encoding="utf-8-sig",
)

feature_groups = {
    "continuous_vars": continuous_vars,
    "binary_vars": binary_vars,
    "categorical_vars": categorical_vars,
    "onehot_feature_names_development": onehot_feature_names,
    "continuous_n": len(continuous_vars),
    "binary_n": len(binary_vars),
    "categorical_n": len(categorical_vars),
    "onehot_n": onehot_n,
    "final_feature_n": final_feature_n,
}

with open(
    OUTPUT_DIR / "feature_groups.json",
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        feature_groups,
        file,
        ensure_ascii=False,
        indent=2,
    )

manifest = {
    "input_file": str(INPUT_FILE),
    "step1_dir": str(STEP1_DIR),
    "step2_dir": str(STEP2_DIR),
    "output_dir": str(OUTPUT_DIR),
    "history_months": list(range(0, 61, 6)),
    "n_patients": int(n_patients),
    "development_n": int(len(development_idx)),
    "test_n": int(len(test_idx)),
    "observed_time_row_n": int(observed_row_mask.sum()),
    "missing_time_row_n": int((~observed_row_mask).sum()),
    "preprocessing_status": (
        "raw tensors only; no scaling or one-hot transformation"
    ),
}

with open(
    OUTPUT_DIR / "step3_manifest.json",
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
# 13. 输出最终结果
# ============================================================

print("\n========================================")
print("Step 3完成：统一原始特征张量已生成")
print("========================================")

print("\n连续变量张量：", continuous_raw.shape)
print("二分类变量张量：", binary_raw.shape)
print("多分类变量张量：", categorical_raw.shape)
print("时间行掩码：", observed_row_mask.shape)

print("\n连续变量数：", len(continuous_vars))
print("二分类变量数：", len(binary_vars))
print("多分类原始变量数：", len(categorical_vars))
print("开发集One-Hot变量数：", onehot_n)
print("预计最终模型特征数：", final_feature_n)

print("\n时间行统计：")
print(tensor_summary.to_string(index=False))

print("\n开发集和测试集统计：")
print(split_tensor_summary.to_string(index=False))

print("\n五折统计：")
print(fold_tensor_summary.to_string(index=False))

if unknown_category_summary.empty:
    print("\n测试集没有出现开发集未见的分类水平。")
else:
    print("\n测试集出现开发集未见的分类水平：")
    print(unknown_category_summary.to_string(index=False))

print("\n本步骤未进行标准化或One-Hot转换。")
print("输出目录：", OUTPUT_DIR)
print("Step 3运行完成。")
