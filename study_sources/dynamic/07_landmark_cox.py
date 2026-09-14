# ============================================================
# Step 7：固定95项特征的未惩罚Super Landmark Cox五折OOF预测（v4）
#
# 代码逻辑：
# 1. 读取Step 6生成的开发集Super Landmark长格式数据。
# 2. 使用预先固定的95项非重复特征，不按结局自动筛选变量。
# 3. 15项实验室仅保留baseline、current和slope_per_year。
# 4. 删除history_observed_step_count，保留missing_ratio和最近观测间隔。
# 5. 删除ART_regimen_changed，只保留ART_regimen_change_count。
# 6. 基线分类变量各删除一个固定参考水平。
# 7. 当前ART和累计ART显式构造Other，并以TDF+NNRTI方案为参考。
# 8. 每折仅用训练长记录拟合连续变量标准化参数。
# 9. Cox不调参、不加惩罚；五折仅生成无泄漏OOF预测。
# 10. 六个Landmark共享回归系数，Landmark作为strata估计不同基线风险。
# 11. 使用唯一长记录索引恢复lifelines分层预测的原始行顺序。
# 12. 强制核对5年累计风险与Cox线性风险评分的排序完全一致。
# ============================================================

import inspect
import json
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler

try:
    import lifelines
    from lifelines import CoxPHFitter
    from lifelines.exceptions import ConvergenceError, ConvergenceWarning
    from lifelines.utils import concordance_index
except ImportError as exc:
    raise ImportError(
        "Step 7需要lifelines。请先在当前环境运行：\n"
        "pip install lifelines\n"
        "安装完成后重新从头运行本代码。"
    ) from exc


# ============================================================
# 1. 路径和固定参数
# ============================================================

STEP5_DIR = Path(
    "__CKD_WORKDIR__/rolling_5y_step5_landmark_summary"
)

STEP6_DIR = Path(
    "__CKD_WORKDIR__/rolling_5y_step6_super_landmark_data"
)

OUTPUT_DIR = Path(
    "__CKD_WORKDIR__/rolling_5y_step7_unpenalized_cox_fixed95_v5"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

N_SPLITS = 5
EXPECTED_DEVELOPMENT_N = 22337
EXPECTED_LANDMARK_N = 6
EXPECTED_FUTURE_INTERVAL_N = 10
EXPECTED_SUMMARY_FEATURE_N = 146
EXPECTED_LONG_RECORD_N = 100122
EXPECTED_FIXED_COX_FEATURE_N = 95

LANDMARK_MONTHS = np.array(
    [0, 12, 24, 36, 48, 60],
    dtype=np.int32,
)

FUTURE_END_MONTHS = np.arange(
    6,
    61,
    6,
    dtype=np.float64,
)

DURATION_COLUMN = "analysis_time_month"
EVENT_COLUMN = "event_within_60m"
STRATA_COLUMN = "landmark_month"

CONSTANT_TOLERANCE = 1e-12
SURVIVAL_TOLERANCE = 1e-6
# Cox固定时点累计风险与线性预测子理论上应保持相同排序。
# 由于生存概率计算、截断及并列值处理存在浮点误差，审计使用
# 足以识别预测错位、但不会因微小数值差异误报的阈值。
RISK_RANK_CORRELATION_MIN = 0.9999
C_INDEX_AUDIT_TOLERANCE = 1e-4
STANDARDIZATION_TOLERANCE = 1e-5
EXACT_CORRELATION_TOLERANCE = 1.0 - 1e-10
ART_MONTH_TOLERANCE = 0.25
OPTIMIZER_STEP_SIZES = [0.10, 0.05, 0.01]

# 基线分类变量的固定参考水平。
REFERENCE_BASELINE_ONEHOT = [
    "Sex_Female_baseline",
    "Marriage_Married or cohabiting_baseline",
    "Course_Heterosexual_baseline",
    "WHOstage_1_baseline",
]

# ART当前方案和累计暴露均以该方案作为参考。
REFERENCE_CURRENT_ART = "current_TDF_NNRTI_3TC_FTC_at_landmark"
REFERENCE_CUMULATIVE_ART = (
    "TDF_NNRTI_3TC_FTC_cum_month_at_landmark"
)

CURRENT_ART_SOURCE_NAMES = [
    "current_TDF_NNRTI_3TC_FTC",
    "current_TDF_PI_3TC_FTC",
    "current_nonTDF_PI",
    "current_BIC_FTC_TAF",
    "current_EVGc_FTC_TAF",
    "current_TDF_INSTI_3TC_FTC",
    "current_nonTDF_DTG",
    "current_nonTDF_traditional_NNRTI",
]

ART_CUMULATIVE_SOURCE_NAMES = [
    "TDF_NNRTI_3TC_FTC_cum_month",
    "TDF_PI_3TC_FTC_cum_month",
    "nonTDF_PI_cum_month",
    "BIC_FTC_TAF_cum_month",
    "EVGc_FTC_TAF_cum_month",
    "TDF_INSTI_3TC_FTC_cum_month",
    "nonTDF_DTG_cum_month",
    "nonTDF_traditional_NNRTI_cum_month",
]


# ============================================================
# 2. 检查并读取Step 6输入
# ============================================================

required_files = [
    STEP5_DIR / "summary_feature_names.csv",
    STEP5_DIR / "summary_column_semantic_audit.csv",
    STEP6_DIR / "X_super_landmark_development_raw.npy",
    STEP6_DIR / "super_landmark_development_metadata.csv",
    STEP6_DIR / "development_long_row_index_map.npy",
    STEP6_DIR / "development_future_event_long.npy",
    STEP6_DIR / "development_future_at_risk_long.npy",
    STEP6_DIR / "development_local_patient_idx_long.npy",
    STEP6_DIR / "development_fold_id_long.npy",
    STEP6_DIR / "development_landmark_index_long.npy",
    STEP6_DIR / "development_landmark_month_long.npy",
    STEP6_DIR / "development_analysis_time_month.npy",
    STEP6_DIR / "development_event_within_60m.npy",
    STEP6_DIR / "summary_feature_names.csv",
    STEP6_DIR / "landmark_months.npy",
    STEP6_DIR / "future_end_months.npy",
]

for fold_id in range(N_SPLITS):
    required_files.extend([
        STEP6_DIR / f"fold_{fold_id}" / "train_long_idx.npy",
        STEP6_DIR / f"fold_{fold_id}" / "validation_long_idx.npy",
    ])

missing_files = [
    str(path) for path in required_files if not path.exists()
]

if missing_files:
    raise FileNotFoundError(
        "以下Step 6输入文件不存在：\n" + "\n".join(missing_files)
    )

X_long_raw = np.load(
    STEP6_DIR / "X_super_landmark_development_raw.npy",
    mmap_mode="r",
)

metadata = pd.read_csv(
    STEP6_DIR / "super_landmark_development_metadata.csv",
    encoding="utf-8-sig",
)

row_index_map = np.load(
    STEP6_DIR / "development_long_row_index_map.npy"
).astype(np.int32)

future_event_long = np.load(
    STEP6_DIR / "development_future_event_long.npy"
).astype(np.float32)

future_at_risk_long = np.load(
    STEP6_DIR / "development_future_at_risk_long.npy"
).astype(bool)

local_patient_idx_long = np.load(
    STEP6_DIR / "development_local_patient_idx_long.npy"
).astype(np.int32)

fold_id_long = np.load(
    STEP6_DIR / "development_fold_id_long.npy"
).astype(np.int8)

landmark_index_long = np.load(
    STEP6_DIR / "development_landmark_index_long.npy"
).astype(np.int8)

landmark_month_long = np.load(
    STEP6_DIR / "development_landmark_month_long.npy"
).astype(np.int32)

analysis_time_month = np.load(
    STEP6_DIR / "development_analysis_time_month.npy"
).astype(np.float64)

event_within_60m = np.load(
    STEP6_DIR / "development_event_within_60m.npy"
).astype(np.int8)

feature_table = pd.read_csv(
    STEP6_DIR / "summary_feature_names.csv",
    encoding="utf-8-sig",
)

step5_feature_table = pd.read_csv(
    STEP5_DIR / "summary_feature_names.csv",
    encoding="utf-8-sig",
)

step5_semantic_audit = pd.read_csv(
    STEP5_DIR / "summary_column_semantic_audit.csv",
    encoding="utf-8-sig",
)

saved_landmark_months = np.load(
    STEP6_DIR / "landmark_months.npy"
).astype(np.int32)

saved_future_end_months = np.load(
    STEP6_DIR / "future_end_months.npy"
).astype(np.float64)


# ============================================================
# 3. 核对Step 6数据结构
# ============================================================

expected_shapes = {
    "X_long_raw": (
        EXPECTED_LONG_RECORD_N,
        EXPECTED_SUMMARY_FEATURE_N,
    ),
    "metadata": (EXPECTED_LONG_RECORD_N,),
    "row_index_map": (
        EXPECTED_DEVELOPMENT_N,
        EXPECTED_LANDMARK_N,
    ),
    "future_event_long": (
        EXPECTED_LONG_RECORD_N,
        EXPECTED_FUTURE_INTERVAL_N,
    ),
    "future_at_risk_long": (
        EXPECTED_LONG_RECORD_N,
        EXPECTED_FUTURE_INTERVAL_N,
    ),
    "local_patient_idx_long": (EXPECTED_LONG_RECORD_N,),
    "fold_id_long": (EXPECTED_LONG_RECORD_N,),
    "landmark_index_long": (EXPECTED_LONG_RECORD_N,),
    "landmark_month_long": (EXPECTED_LONG_RECORD_N,),
    "analysis_time_month": (EXPECTED_LONG_RECORD_N,),
    "event_within_60m": (EXPECTED_LONG_RECORD_N,),
}

actual_shapes = {
    "X_long_raw": X_long_raw.shape,
    "metadata": (len(metadata),),
    "row_index_map": row_index_map.shape,
    "future_event_long": future_event_long.shape,
    "future_at_risk_long": future_at_risk_long.shape,
    "local_patient_idx_long": local_patient_idx_long.shape,
    "fold_id_long": fold_id_long.shape,
    "landmark_index_long": landmark_index_long.shape,
    "landmark_month_long": landmark_month_long.shape,
    "analysis_time_month": analysis_time_month.shape,
    "event_within_60m": event_within_60m.shape,
}

for name, expected_shape in expected_shapes.items():
    if actual_shapes[name] != expected_shape:
        raise ValueError(
            f"{name}形状为{actual_shapes[name]}，应为{expected_shape}。"
        )

required_feature_columns = {
    "summary_feature_index",
    "summary_feature_name",
    "feature_role",
    "source_variable",
}

if not required_feature_columns.issubset(feature_table.columns):
    raise ValueError(
        "summary_feature_names.csv缺少字段："
        f"{sorted(required_feature_columns - set(feature_table.columns))}"
    )

if len(feature_table) != EXPECTED_SUMMARY_FEATURE_N:
    raise ValueError(
        f"Step 5特征字典有{len(feature_table)}行，"
        f"应为{EXPECTED_SUMMARY_FEATURE_N}行。"
    )

if not np.array_equal(
    feature_table["summary_feature_index"].to_numpy(dtype=np.int32),
    np.arange(EXPECTED_SUMMARY_FEATURE_N, dtype=np.int32),
):
    raise ValueError("Step 5特征字典索引不是0到145的固定顺序。")

# Step 6必须由最新Step 5重新生成，避免旧特征名称与矩阵列错位。
comparison_columns = [
    "summary_feature_index",
    "summary_feature_name",
    "feature_role",
    "source_variable",
]

if not feature_table[comparison_columns].equals(
    step5_feature_table[comparison_columns]
):
    raise ValueError(
        "Step 6中的特征字典与最新Step 5不一致。"
        "请先运行Step5_build_landmark_summary_features_v5.py，"
        "再从头重新运行Step6_build_super_landmark_long_data.py。"
    )

if "invalid_n" not in step5_semantic_audit.columns:
    raise ValueError("Step 5语义核查文件缺少invalid_n字段。")

if int(step5_semantic_audit["invalid_n"].sum()) != 0:
    raise ValueError(
        "Step 5特征名称与矩阵列语义核查未通过，不能进入Cox建模。"
    )

if not np.array_equal(saved_landmark_months, LANDMARK_MONTHS):
    raise ValueError("Step 6保存的Landmark月份与当前固定定义不一致。")

if not np.array_equal(saved_future_end_months, FUTURE_END_MONTHS):
    raise ValueError("Step 6保存的未来区间与当前固定定义不一致。")

if not np.isfinite(np.asarray(X_long_raw)).all():
    raise ValueError("Step 6开发集长格式特征包含NaN或无穷值。")

if np.any(analysis_time_month <= 0.0):
    raise ValueError("Super Landmark分析时间必须全部大于0。")

if np.any(analysis_time_month > 60.0 + 1e-6):
    raise ValueError("Super Landmark分析时间不能超过60个月。")

if not np.isin(event_within_60m, [0, 1]).all():
    raise ValueError("事件标签不是0/1。")

if not np.array_equal(
    LANDMARK_MONTHS[landmark_index_long],
    landmark_month_long,
):
    raise ValueError("Landmark索引与月份映射错误。")

expected_long_row = row_index_map[
    local_patient_idx_long,
    landmark_index_long,
]

if not np.array_equal(
    expected_long_row,
    np.arange(EXPECTED_LONG_RECORD_N, dtype=np.int32),
):
    raise ValueError("患者-Landmark长行映射错误。")


# ============================================================
# 4. 构建固定95项Cox特征
# ============================================================

all_feature_names = feature_table[
    "summary_feature_name"
].astype(str).tolist()

name_to_index = {
    name: index for index, name in enumerate(all_feature_names)
}


def require_feature_names(names):
    missing = [name for name in names if name not in name_to_index]
    if missing:
        raise ValueError(
            "以下预期Step 5特征不存在：" + str(missing)
        )


require_feature_names(REFERENCE_BASELINE_ONEHOT)
require_feature_names([REFERENCE_CURRENT_ART, REFERENCE_CUMULATIVE_ART])

# 保留的原始特征角色。
keep_roles = {
    "deterministic_time_updated",
    "baseline_static",
    "baseline_static_onehot",
    "longitudinal_baseline",
    "longitudinal_current",
    "longitudinal_slope",
    "persistent_status_current",
    "persistent_status_duration",
    "medication_status_current",
    "medication_exposure_duration",
    "cumulative_art_exposure",
    "current_art_regimen",
    "history_availability",
    "history_recency",
    "art_regimen_history",
}

# 明确删除的冗余特征。
remove_names = set(REFERENCE_BASELINE_ONEHOT)
remove_names.update({
    "history_observed_step_count",
    "ART_regimen_changed",
    REFERENCE_CURRENT_ART,
    REFERENCE_CUMULATIVE_ART,
})

original_keep_rows = feature_table.loc[
    feature_table["feature_role"].astype(str).isin(keep_roles)
].copy()

original_keep_rows = original_keep_rows.loc[
    ~original_keep_rows["summary_feature_name"].astype(str).isin(
        remove_names
    )
].copy()

# 明确排除均值、标准差和变化值，即使未来角色名称发生误配也不能进入。
forbidden_roles = {
    "longitudinal_change",
    "longitudinal_mean",
    "longitudinal_variability",
}

if original_keep_rows["feature_role"].astype(str).isin(
    forbidden_roles
).any():
    raise ValueError("固定Cox特征中误纳入变化值、均值或标准差。")

# 读取8项当前ART和8项累计ART，用于构造Other。
current_art_feature_names = [
    f"{name}_at_landmark" for name in CURRENT_ART_SOURCE_NAMES
]

cumulative_art_feature_names = [
    f"{name}_at_landmark" for name in ART_CUMULATIVE_SOURCE_NAMES
]

require_feature_names(current_art_feature_names)
require_feature_names(cumulative_art_feature_names)

current_art_indices = np.array(
    [name_to_index[name] for name in current_art_feature_names],
    dtype=np.int32,
)

cumulative_art_indices = np.array(
    [name_to_index[name] for name in cumulative_art_feature_names],
    dtype=np.int32,
)

current_art_matrix = np.asarray(
    X_long_raw[:, current_art_indices],
    dtype=np.float64,
)

if not np.all(
    np.isclose(current_art_matrix, 0.0, atol=1e-7)
    | np.isclose(current_art_matrix, 1.0, atol=1e-7)
):
    raise ValueError("当前ART方案变量不是严格0/1。")

current_art_sum = np.sum(current_art_matrix, axis=1)

if np.any(current_art_sum > 1.0 + 1e-7):
    overlap_n = int(np.sum(current_art_sum > 1.0 + 1e-7))
    raise ValueError(
        f"存在{overlap_n}条记录同时属于多个当前ART方案。"
    )

current_art_other = np.isclose(
    current_art_sum,
    0.0,
    atol=1e-7,
).astype(np.float64)

cumulative_art_matrix = np.asarray(
    X_long_raw[:, cumulative_art_indices],
    dtype=np.float64,
)

if np.any(cumulative_art_matrix < -1e-7):
    raise ValueError("ART累计暴露月数存在负值。")

cumulative_art_sum = np.sum(cumulative_art_matrix, axis=1)
other_cumulative_art = (
    landmark_month_long.astype(np.float64) - cumulative_art_sum
)

minimum_other_cumulative = float(np.min(other_cumulative_art))

if minimum_other_cumulative < -ART_MONTH_TOLERANCE:
    bad_n = int(np.sum(
        other_cumulative_art < -ART_MONTH_TOLERANCE
    ))
    raise ValueError(
        f"存在{bad_n}条记录的8类ART累计月数之和超过"
        f"Landmark月份；最小Other累计月数为"
        f"{minimum_other_cumulative:.6f}。"
    )

# 只修正浮点误差范围内的小负值。
other_cumulative_art = np.maximum(other_cumulative_art, 0.0)

# 按Step 5固定顺序保留原始特征，并插入2项派生ART特征。
original_keep_indices = original_keep_rows[
    "summary_feature_index"
].to_numpy(dtype=np.int32)

original_keep_names = original_keep_rows[
    "summary_feature_name"
].astype(str).tolist()

original_keep_roles = original_keep_rows[
    "feature_role"
].astype(str).tolist()

X_fixed_original = np.asarray(
    X_long_raw[:, original_keep_indices],
    dtype=np.float64,
)

X_fixed = np.column_stack([
    X_fixed_original,
    current_art_other,
    other_cumulative_art,
]).astype(np.float64)

fixed_feature_names = original_keep_names + [
    "current_ART_Other_at_landmark",
    "ART_Other_cum_month_at_landmark",
]

fixed_feature_roles = original_keep_roles + [
    "current_art_regimen",
    "cumulative_art_exposure",
]

fixed_source_variables = original_keep_rows[
    "source_variable"
].astype(str).tolist() + [
    "current_ART_Other",
    "ART_Other_cum_month",
]

if X_fixed.shape != (
    EXPECTED_LONG_RECORD_N,
    EXPECTED_FIXED_COX_FEATURE_N,
):
    raise ValueError(
        f"固定Cox特征矩阵形状为{X_fixed.shape}，应为"
        f"({EXPECTED_LONG_RECORD_N}, {EXPECTED_FIXED_COX_FEATURE_N})。"
    )

if len(set(fixed_feature_names)) != EXPECTED_FIXED_COX_FEATURE_N:
    raise ValueError("固定95项Cox特征名称存在重复。")

if not np.isfinite(X_fixed).all():
    raise ValueError("固定95项Cox特征包含NaN或无穷值。")

# 二分类变量使用固定特征名称识别，避免依据角色误判时长或累计暴露。
EXPECTED_BINARY_FEATURE_NAMES = {
    "Oppinfection_baseline",

    # 基线分类变量删除参考水平后保留的10项One-Hot。
    "Sex_Male_baseline",
    "Marriage_Divorced-separated-or-widowed_baseline",
    "Marriage_Never married_baseline",
    "Marriage_Others_baseline",
    "Course_Drugs_baseline",
    "Course_Male to male_baseline",
    "Course_Others_baseline",
    "WHOstage_2_baseline",
    "WHOstage_3_baseline",
    "WHOstage_4_baseline",

    # 6项持续性疾病在Landmark时的状态。
    "CVD_status_at_landmark",
    "diabetes_status_at_landmark",
    "hypertension_status_at_landmark",
    "hypercholesterolemia_status_at_landmark",
    "HBV_status_at_landmark",
    "HCV_status_at_landmark",

    # 3项代谢用药在Landmark时的状态。
    "antidiabetic_med_at_landmark",
    "antihypertensive_med_at_landmark",
    "antilipid_med_at_landmark",

    # 当前ART删除TDF+NNRTI参考方案后保留7项，再加入Other。
    "current_TDF_PI_3TC_FTC_at_landmark",
    "current_nonTDF_PI_at_landmark",
    "current_BIC_FTC_TAF_at_landmark",
    "current_EVGc_FTC_TAF_at_landmark",
    "current_TDF_INSTI_3TC_FTC_at_landmark",
    "current_nonTDF_DTG_at_landmark",
    "current_nonTDF_traditional_NNRTI_at_landmark",
    "current_ART_Other_at_landmark",
}

missing_binary_features = sorted(
    EXPECTED_BINARY_FEATURE_NAMES - set(fixed_feature_names)
)
present_binary_features = sorted(
    set(fixed_feature_names) & EXPECTED_BINARY_FEATURE_NAMES
)

if missing_binary_features:
    raise ValueError(
        "固定Cox二分类特征缺失："
        f"{missing_binary_features}"
    )

if len(present_binary_features) != 28:
    raise ValueError(
        "固定Cox二分类特征数量异常："
        f"{len(present_binary_features)}，应为28。"
    )

binary_mask = np.array([
    name in EXPECTED_BINARY_FEATURE_NAMES
    for name in fixed_feature_names
], dtype=bool)

continuous_mask = ~binary_mask

# 逐列核查并保存实际取值范围，避免只返回笼统错误。
binary_audit_rows = []
invalid_binary_features = []

for feature_name in sorted(EXPECTED_BINARY_FEATURE_NAMES):
    position = fixed_feature_names.index(feature_name)
    values = X_fixed[:, position].astype(np.float64, copy=False)
    close_zero = np.isclose(values, 0.0, atol=1e-6)
    close_one = np.isclose(values, 1.0, atol=1e-6)
    invalid_mask = ~(close_zero | close_one)

    binary_audit_rows.append({
        "feature_name": feature_name,
        "minimum": float(np.min(values)),
        "maximum": float(np.max(values)),
        "unique_value_n": int(np.unique(values).size),
        "zero_n": int(close_zero.sum()),
        "one_n": int(close_one.sum()),
        "invalid_n": int(invalid_mask.sum()),
        "invalid_value_examples": ";".join(
            map(str, np.unique(values[invalid_mask])[:10].tolist())
        ) if np.any(invalid_mask) else "",
    })

    if np.any(invalid_mask):
        invalid_binary_features.append(feature_name)
    else:
        # 将float32写入产生的极小误差统一吸附为严格0或1。
        X_fixed[:, position] = np.where(close_one, 1.0, 0.0)

binary_audit_table = pd.DataFrame(binary_audit_rows)
binary_audit_table.to_csv(
    OUTPUT_DIR / "cox_binary_feature_audit.csv",
    index=False,
    encoding="utf-8-sig",
)

if invalid_binary_features:
    details = binary_audit_table.loc[
        binary_audit_table["feature_name"].isin(invalid_binary_features),
        [
            "feature_name",
            "minimum",
            "maximum",
            "unique_value_n",
            "invalid_n",
            "invalid_value_examples",
        ],
    ].to_dict("records")

    raise ValueError(
        "以下明确指定的二分类特征存在非0/1值："
        f"{details}。详细结果已保存至cox_binary_feature_audit.csv。"
    )

fixed_feature_dictionary = pd.DataFrame({
    "cox_feature_index": np.arange(
        EXPECTED_FIXED_COX_FEATURE_N,
        dtype=np.int32,
    ),
    "cox_feature_name": fixed_feature_names,
    "feature_role": fixed_feature_roles,
    "source_variable": fixed_source_variables,
    "is_binary_unscaled": binary_mask,
})

fixed_feature_dictionary.to_csv(
    OUTPUT_DIR / "cox_fixed_feature_dictionary_95.csv",
    index=False,
    encoding="utf-8-sig",
)

removed_feature_table = feature_table.loc[
    ~feature_table["summary_feature_index"].isin(
        original_keep_indices
    )
].copy()

removal_reason_map = {}

for name in feature_table.loc[
    feature_table["feature_role"].astype(str)
    == "longitudinal_change",
    "summary_feature_name",
].astype(str):
    removal_reason_map[name] = "remove_exact_change_dependency"

for name in feature_table.loc[
    feature_table["feature_role"].astype(str)
    == "longitudinal_mean",
    "summary_feature_name",
].astype(str):
    removal_reason_map[name] = "remove_highly_redundant_history_mean"

for name in feature_table.loc[
    feature_table["feature_role"].astype(str)
    == "longitudinal_variability",
    "summary_feature_name",
].astype(str):
    removal_reason_map[name] = "remove_unstable_history_sd"

for name in REFERENCE_BASELINE_ONEHOT:
    removal_reason_map[name] = "baseline_category_reference_level"

removal_reason_map["history_observed_step_count"] = (
    "exactly_determined_by_missing_ratio_within_landmark"
)
removal_reason_map["ART_regimen_changed"] = (
    "redundant_with_ART_regimen_change_count"
)
removal_reason_map[REFERENCE_CURRENT_ART] = (
    "current_ART_reference_level"
)
removal_reason_map[REFERENCE_CUMULATIVE_ART] = (
    "cumulative_ART_reference_exposure"
)

removed_feature_table["cox_removal_reason"] = (
    removed_feature_table["summary_feature_name"]
    .astype(str)
    .map(removal_reason_map)
    .fillna("not_in_prespecified_fixed_cox_set")
)

removed_feature_table.to_csv(
    OUTPUT_DIR / "cox_removed_features.csv",
    index=False,
    encoding="utf-8-sig",
)


# ============================================================
# 5. 定义折内预处理和诊断函数
# ============================================================


def stratified_center_matrix(X, strata_values):
    """在每个Landmark内部中心化，用于分层Cox可识别性诊断。"""

    X = np.asarray(X, dtype=np.float64)
    strata_values = np.asarray(strata_values, dtype=np.int32)
    centered = np.empty_like(X, dtype=np.float64)

    for landmark_month in LANDMARK_MONTHS:
        mask = strata_values == landmark_month

        if int(mask.sum()) == 0:
            raise ValueError(
                f"训练数据中Landmark {landmark_month}个月没有记录。"
            )

        centered[mask] = (
            X[mask]
            - np.mean(X[mask], axis=0, dtype=np.float64)
        )

    return centered



def calculate_design_diagnostics(X_train, strata_values):
    """检查固定95项特征在分层部分似然中的秩和条件数。"""

    centered = stratified_center_matrix(X_train, strata_values)
    column_norm = np.sqrt(np.sum(centered ** 2, axis=0))

    no_within_stratum_variation = np.flatnonzero(
        column_norm <= CONSTANT_TOLERANCE
    ).astype(np.int32)

    if len(no_within_stratum_variation) > 0:
        names = [
            fixed_feature_names[index]
            for index in no_within_stratum_variation
        ]
        raise ValueError(
            "以下固定Cox特征在全部Landmark分层内均无变异："
            f"{names}"
        )

    normalized = centered / column_norm[None, :]
    gram = normalized.T @ normalized
    eigenvalues = np.linalg.eigvalsh(gram)
    eigenvalues = np.maximum(eigenvalues, 0.0)

    largest = float(np.max(eigenvalues))
    exact_tolerance = (
        max(gram.shape)
        * np.finfo(np.float64).eps
        * largest
        * 100.0
    )
    rank = int(np.sum(eigenvalues > exact_tolerance))

    positive = eigenvalues[eigenvalues > exact_tolerance]
    condition_number = (
        float(np.sqrt(np.max(positive) / np.min(positive)))
        if len(positive) > 0
        else np.inf
    )

    if rank != EXPECTED_FIXED_COX_FEATURE_N:
        raise ValueError(
            f"固定95项特征在Landmark分层内的数值秩为{rank}，"
            f"应为{EXPECTED_FIXED_COX_FEATURE_N}。"
        )

    # 仅报告近乎完全相关，不基于普通相关性自动删除变量。
    correlation = np.corrcoef(normalized, rowvar=False)
    upper_i, upper_j = np.triu_indices_from(correlation, k=1)
    exact_pair_mask = np.abs(correlation[upper_i, upper_j]) > (
        EXACT_CORRELATION_TOLERANCE
    )

    exact_pairs = []
    for i, j in zip(
        upper_i[exact_pair_mask],
        upper_j[exact_pair_mask],
    ):
        exact_pairs.append({
            "feature_1": fixed_feature_names[int(i)],
            "feature_2": fixed_feature_names[int(j)],
            "correlation": float(correlation[i, j]),
        })

    if exact_pairs:
        raise ValueError(
            "固定95项特征中仍存在近乎完全相关变量："
            f"{exact_pairs[:10]}"
        )

    return {
        "rank": rank,
        "condition_number": condition_number,
        "minimum_eigenvalue": float(np.min(positive)),
        "maximum_eigenvalue": float(np.max(positive)),
    }



def check_binary_event_separation(
    X_train,
    event_train,
    binary_feature_mask,
):
    """检查二分类变量是否存在明显的完全事件分离。"""

    rows = []
    critical = []

    binary_positions = np.flatnonzero(
        binary_feature_mask
    ).astype(np.int32)

    for position in binary_positions:
        values = X_train[:, position]

        level_0 = np.isclose(values, 0.0, atol=1e-7)
        level_1 = np.isclose(values, 1.0, atol=1e-7)

        level_0_n = int(level_0.sum())
        level_1_n = int(level_1.sum())
        level_0_event_n = int(event_train[level_0].sum())
        level_1_event_n = int(event_train[level_1].sum())

        row = {
            "feature_name": fixed_feature_names[position],
            "level_0_n": level_0_n,
            "level_1_n": level_1_n,
            "level_0_event_n": level_0_event_n,
            "level_1_event_n": level_1_event_n,
        }
        rows.append(row)

        if (
            level_0_n > 0
            and level_1_n > 0
            and (
                level_0_event_n == 0
                or level_1_event_n == 0
            )
        ):
            critical.append(row)

    return pd.DataFrame(rows), critical



def fit_fold_scaler(train_long_idx):
    """仅使用对应训练折拟合连续变量标准化参数。"""

    X_train_raw = X_fixed[train_long_idx].copy()

    global_range = (
        np.max(X_train_raw, axis=0)
        - np.min(X_train_raw, axis=0)
    )

    constant_positions = np.flatnonzero(
        global_range <= CONSTANT_TOLERANCE
    ).astype(np.int32)

    if len(constant_positions) > 0:
        names = [
            fixed_feature_names[index]
            for index in constant_positions
        ]
        raise ValueError(
            "固定95项特征在当前训练折中出现全局常数："
            f"{names}"
        )

    scaler = StandardScaler(
        with_mean=True,
        with_std=True,
    )
    scaler.fit(X_train_raw[:, continuous_mask])

    return scaler



def transform_fold_features(long_idx, scaler):
    """使用训练折标准化参数转换指定长记录。"""

    X_transformed = X_fixed[long_idx].copy()
    X_transformed[:, continuous_mask] = scaler.transform(
        X_transformed[:, continuous_mask]
    )

    if not np.isfinite(X_transformed).all():
        raise ValueError("折内转换后的Cox特征包含NaN或无穷值。")

    return X_transformed



def build_cox_dataframe(
    X_transformed,
    long_idx,
    include_outcome=True,
):
    """构建lifelines分层Cox需要的DataFrame。"""

    frame = pd.DataFrame(
        X_transformed,
        columns=fixed_feature_names,
    )
    frame[STRATA_COLUMN] = landmark_month_long[long_idx].astype(np.int32)

    if include_outcome:
        frame[DURATION_COLUMN] = analysis_time_month[long_idx]
        frame[EVENT_COLUMN] = event_within_60m[long_idx].astype(np.int8)

    return frame


# ============================================================
# 6. 定义未惩罚分层Cox拟合函数
# ============================================================


def fit_unpenalized_cox(train_frame):
    """固定模型定义，仅尝试更保守的Newton-Raphson步长。"""

    failed_attempts = []

    for step_size in OPTIMIZER_STEP_SIZES:
        model = CoxPHFitter(penalizer=0.0)

        fit_kwargs = {
            "duration_col": DURATION_COLUMN,
            "event_col": EVENT_COLUMN,
            "strata": [STRATA_COLUMN],
            "show_progress": False,
            "robust": False,
            "batch_mode": True,
        }

        if "fit_options" in inspect.signature(model.fit).parameters:
            fit_kwargs["fit_options"] = {
                "step_size": step_size,
                "precision": 1e-7,
                "r_precision": 1e-9,
                "max_steps": 1000,
            }

        try:
            with warnings.catch_warnings(record=True) as caught_warnings:
                warnings.simplefilter("always")
                model.fit(
                    train_frame,
                    **fit_kwargs,
                )
        except ConvergenceError as exc:
            failed_attempts.append({
                "step_size": step_size,
                "error": str(exc),
            })
            continue

        convergence_messages = []
        other_warning_messages = []

        for warning_item in caught_warnings:
            message = str(warning_item.message)

            if issubclass(warning_item.category, ConvergenceWarning):
                convergence_messages.append(message)
            else:
                other_warning_messages.append(
                    f"{warning_item.category.__name__}: {message}"
                )

        critical_terms = [
            "failed to converge",
            "convergence halted",
            "delta contains nan",
            "matrix inversion problems",
            "ill-conditioned",
        ]

        critical_messages = [
            message
            for message in convergence_messages
            if any(term in message.lower() for term in critical_terms)
        ]

        if critical_messages:
            failed_attempts.append({
                "step_size": step_size,
                "error": " | ".join(critical_messages),
            })
            continue

        if not np.isfinite(
            model.params_.to_numpy(dtype=np.float64)
        ).all():
            failed_attempts.append({
                "step_size": step_size,
                "error": "回归系数包含NaN或无穷值",
            })
            continue

        if not np.isfinite(
            model.variance_matrix_.to_numpy(dtype=np.float64)
        ).all():
            failed_attempts.append({
                "step_size": step_size,
                "error": "方差矩阵包含NaN或无穷值",
            })
            continue

        return (
            model,
            convergence_messages,
            other_warning_messages,
            step_size,
            failed_attempts,
        )

    failure_text = "\n".join([
        f"step_size={item['step_size']}: {item['error']}"
        for item in failed_attempts
    ])

    raise RuntimeError(
        "固定95项未惩罚Super Landmark Cox仍无法收敛。"
        "此时不应继续自动删除不同折的变量；建议将Cox主模型"
        "改为固定特征的Ridge Cox，并通过开发集五折选择惩罚强度。\n"
        + failure_text
    )


# ============================================================
# 7. 五折拟合并生成开发集OOF预测
# ============================================================

print("=" * 72)
print("Step 7：开始固定95项未惩罚Super Landmark Cox五折OOF（预测行顺序与数值审计修正版）预测")
print("=" * 72)
print("lifelines版本：", lifelines.__version__)
print("Step 5原始汇总特征数：", EXPECTED_SUMMARY_FEATURE_N)
print("固定传统Cox特征数：", EXPECTED_FIXED_COX_FEATURE_N)
print("实验室保留：baseline、current、slope_per_year")
print("实验室删除：change、history_mean、history_sd")
print("ART当前方案参考：", REFERENCE_CURRENT_ART)
print("ART累计暴露参考：", REFERENCE_CUMULATIVE_ART)
print("Cox惩罚：无")
print("超参数调优：无")
print("固定二分类特征数：", int(binary_mask.sum()))
print("折内标准化连续特征数：", int(continuous_mask.sum()))

cox_oof_survival_long = np.full(
    (
        EXPECTED_LONG_RECORD_N,
        EXPECTED_FUTURE_INTERVAL_N,
    ),
    np.nan,
    dtype=np.float32,
)

cox_oof_risk_long = np.full_like(
    cox_oof_survival_long,
    np.nan,
)

cox_oof_log_partial_hazard_long = np.full(
    EXPECTED_LONG_RECORD_N,
    np.nan,
    dtype=np.float32,
)

fold_summary_rows = []
fold_landmark_rows = []

for fold_id in range(N_SPLITS):
    fold_input_dir = STEP6_DIR / f"fold_{fold_id}"
    fold_output_dir = OUTPUT_DIR / f"fold_{fold_id}"
    fold_output_dir.mkdir(parents=True, exist_ok=True)

    train_long_idx = np.load(
        fold_input_dir / "train_long_idx.npy"
    ).astype(np.int32)

    validation_long_idx = np.load(
        fold_input_dir / "validation_long_idx.npy"
    ).astype(np.int32)

    if np.intersect1d(
        train_long_idx,
        validation_long_idx,
    ).size > 0:
        raise ValueError(f"第{fold_id}折训练和验证长记录交叉。")

    if not np.all(fold_id_long[validation_long_idx] == fold_id):
        raise ValueError(f"第{fold_id}折验证长记录归属错误。")

    if np.any(fold_id_long[train_long_idx] == fold_id):
        raise ValueError(f"第{fold_id}折训练长记录归属错误。")

    train_patients = np.unique(
        local_patient_idx_long[train_long_idx]
    )
    validation_patients = np.unique(
        local_patient_idx_long[validation_long_idx]
    )

    if np.intersect1d(
        train_patients,
        validation_patients,
    ).size > 0:
        raise ValueError(f"第{fold_id}折训练和验证存在患者交叉。")

    scaler = fit_fold_scaler(train_long_idx)

    X_train = transform_fold_features(
        train_long_idx,
        scaler,
    )
    X_validation = transform_fold_features(
        validation_long_idx,
        scaler,
    )

    if np.any(continuous_mask):
        max_abs_scaled_mean = float(np.max(np.abs(
            np.mean(
                X_train[:, continuous_mask],
                axis=0,
                dtype=np.float64,
            )
        )))
        max_abs_scaled_sd_difference = float(np.max(np.abs(
            np.std(
                X_train[:, continuous_mask],
                axis=0,
                dtype=np.float64,
            ) - 1.0
        )))
    else:
        max_abs_scaled_mean = 0.0
        max_abs_scaled_sd_difference = 0.0

    if max_abs_scaled_mean > STANDARDIZATION_TOLERANCE:
        raise ValueError(
            f"第{fold_id}折训练连续特征标准化均值异常："
            f"{max_abs_scaled_mean}"
        )

    if max_abs_scaled_sd_difference > STANDARDIZATION_TOLERANCE:
        raise ValueError(
            f"第{fold_id}折训练连续特征标准差异常："
            f"{max_abs_scaled_sd_difference}"
        )

    diagnostics = calculate_design_diagnostics(
        X_train,
        landmark_month_long[train_long_idx],
    )

    separation_table, critical_separation = (
        check_binary_event_separation(
            X_train,
            event_within_60m[train_long_idx],
            binary_mask,
        )
    )

    separation_table.to_csv(
        fold_output_dir / "binary_event_separation_audit.csv",
        index=False,
        encoding="utf-8-sig",
    )

    if critical_separation:
        raise ValueError(
            f"第{fold_id}折存在二分类变量完全事件分离："
            f"{critical_separation}"
        )

    train_frame = build_cox_dataframe(
        X_train,
        train_long_idx,
        include_outcome=True,
    )
    validation_frame = build_cox_dataframe(
        X_validation,
        validation_long_idx,
        include_outcome=True,
    )

    # 使用真实长记录编号作为唯一索引。lifelines分层预测可能按strata
    # 分组返回列，因此不能直接丢弃DataFrame列标签后依赖当前列顺序。
    train_frame.index = pd.Index(
        train_long_idx.astype(np.int64),
        name="long_row_index",
    )
    validation_frame.index = pd.Index(
        validation_long_idx.astype(np.int64),
        name="long_row_index",
    )

    print(
        f"\n第{fold_id}折开始：训练{len(train_long_idx)}条，"
        f"验证{len(validation_long_idx)}条，"
        f"固定特征{EXPECTED_FIXED_COX_FEATURE_N}项，"
        f"分层条件数{diagnostics['condition_number']:.3e}"
    )

    (
        model,
        convergence_messages,
        other_warning_messages,
        optimizer_step_size,
        failed_optimizer_attempts,
    ) = fit_unpenalized_cox(train_frame)

    validation_survival_covariates = validation_frame[
        fixed_feature_names + [STRATA_COLUMN]
    ].copy()

    validation_linear_covariates = validation_frame[
        fixed_feature_names
    ].copy()

    survival_frame = model.predict_survival_function(
        validation_survival_covariates,
        times=FUTURE_END_MONTHS,
    )

    # predict_survival_function返回“时间×患者”的DataFrame。分层模型
    # 内部可按strata分组处理，因此必须依据唯一长记录索引显式还原
    # 为validation_long_idx的原始顺序，不能直接使用to_numpy().T。
    expected_prediction_columns = pd.Index(
        validation_long_idx.astype(np.int64),
        name="long_row_index",
    )
    if survival_frame.columns.has_duplicates:
        raise ValueError(
            f"第{fold_id}折生存预测列索引存在重复值。"
        )

    missing_prediction_columns = expected_prediction_columns.difference(
        survival_frame.columns
    )
    unexpected_prediction_columns = survival_frame.columns.difference(
        expected_prediction_columns
    )
    if (
        len(missing_prediction_columns) > 0
        or len(unexpected_prediction_columns) > 0
    ):
        raise ValueError(
            f"第{fold_id}折生存预测列索引与验证长记录不一致。"
            f"缺失{len(missing_prediction_columns)}列，"
            f"额外{len(unexpected_prediction_columns)}列。"
        )

    survival_frame = survival_frame.reindex(
        columns=expected_prediction_columns
    )
    survival_prediction = survival_frame.to_numpy(
        dtype=np.float64
    ).T

    expected_prediction_shape = (
        len(validation_long_idx),
        EXPECTED_FUTURE_INTERVAL_N,
    )

    if survival_prediction.shape != expected_prediction_shape:
        raise ValueError(
            f"第{fold_id}折生存概率形状为"
            f"{survival_prediction.shape}，应为"
            f"{expected_prediction_shape}。"
        )

    survival_prediction = np.clip(
        survival_prediction,
        0.0,
        1.0,
    )
    survival_prediction = np.minimum.accumulate(
        survival_prediction,
        axis=1,
    )
    risk_prediction = 1.0 - survival_prediction

    log_partial_hazard_series = model.predict_log_partial_hazard(
        validation_linear_covariates
    )
    log_partial_hazard = log_partial_hazard_series.reindex(
        expected_prediction_columns
    ).to_numpy(dtype=np.float64)

    if log_partial_hazard.shape != (len(validation_long_idx),):
        raise ValueError(
            f"第{fold_id}折线性风险评分形状错误："
            f"{log_partial_hazard.shape}。"
        )

    if not np.isfinite(survival_prediction).all():
        raise ValueError(f"第{fold_id}折生存概率包含NaN或无穷值。")

    if not np.isfinite(log_partial_hazard).all():
        raise ValueError(f"第{fold_id}折风险评分包含NaN或无穷值。")

    cox_oof_survival_long[validation_long_idx] = (
        survival_prediction.astype(np.float32)
    )
    cox_oof_risk_long[validation_long_idx] = (
        risk_prediction.astype(np.float32)
    )
    cox_oof_log_partial_hazard_long[validation_long_idx] = (
        log_partial_hazard.astype(np.float32)
    )

    joblib.dump(
        model,
        fold_output_dir / "unpenalized_cox_supermodel_fixed95.joblib",
    )
    joblib.dump(
        scaler,
        fold_output_dir / "continuous_feature_scaler.joblib",
    )

    np.save(
        fold_output_dir / "validation_long_idx.npy",
        validation_long_idx,
    )
    np.save(
        fold_output_dir / "binary_feature_mask.npy",
        binary_mask,
    )

    fixed_feature_dictionary.to_csv(
        fold_output_dir / "active_cox_features.csv",
        index=False,
        encoding="utf-8-sig",
    )

    coefficient_table = model.summary.reset_index()
    if "covariate" in coefficient_table.columns:
        coefficient_table = coefficient_table.rename(
            columns={"covariate": "feature_name"}
        )
    coefficient_table.to_csv(
        fold_output_dir / "cox_coefficient_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    model.baseline_survival_.to_csv(
        fold_output_dir / "cox_baseline_survival_by_landmark.csv",
        encoding="utf-8-sig",
    )

    warning_table = pd.DataFrame({
        "warning_type": (
            ["ConvergenceWarning"] * len(convergence_messages)
            + ["OtherWarning"] * len(other_warning_messages)
        ),
        "warning_message": (
            convergence_messages + other_warning_messages
        ),
    })
    warning_table.to_csv(
        fold_output_dir / "cox_fit_warnings.csv",
        index=False,
        encoding="utf-8-sig",
    )

    fold_monotonic_violation_n = int(np.sum(
        np.diff(survival_prediction, axis=1)
        > SURVIVAL_TOLERANCE
    ))

    fold_summary_rows.append({
        "fold_id": fold_id,
        "train_patient_n": int(len(train_patients)),
        "validation_patient_n": int(len(validation_patients)),
        "train_long_record_n": int(len(train_long_idx)),
        "validation_long_record_n": int(len(validation_long_idx)),
        "train_event_within_60m_n": int(
            event_within_60m[train_long_idx].sum()
        ),
        "validation_event_within_60m_n": int(
            event_within_60m[validation_long_idx].sum()
        ),
        "fixed_feature_n": EXPECTED_FIXED_COX_FEATURE_N,
        "binary_unscaled_feature_n": int(binary_mask.sum()),
        "standardized_feature_n": int(continuous_mask.sum()),
        "design_matrix_rank": int(diagnostics["rank"]),
        "design_condition_number": float(
            diagnostics["condition_number"]
        ),
        "max_abs_train_scaled_mean": max_abs_scaled_mean,
        "max_abs_train_scaled_sd_difference": (
            max_abs_scaled_sd_difference
        ),
        "optimizer_step_size": float(optimizer_step_size),
        "failed_optimizer_attempt_n": int(
            len(failed_optimizer_attempts)
        ),
        "convergence_warning_n": int(len(convergence_messages)),
        "other_warning_n": int(len(other_warning_messages)),
        "survival_monotonic_violation_n": (
            fold_monotonic_violation_n
        ),
        "min_predicted_survival": float(
            survival_prediction.min()
        ),
        "max_predicted_survival": float(
            survival_prediction.max()
        ),
    })

    for landmark_month in LANDMARK_MONTHS:
        landmark_mask = (
            landmark_month_long[validation_long_idx]
            == landmark_month
        )
        landmark_validation_idx = validation_long_idx[landmark_mask]

        duration = analysis_time_month[landmark_validation_idx]
        event = event_within_60m[landmark_validation_idx]
        # 直接使用当前验证折尚未降精度的float64预测进行审计。
        # 不从已保存为float32的全局OOF数组回读，避免极小概率值
        # 因精度压缩形成额外并列，从而造成C-index微小差异。
        local_log_hazard = log_partial_hazard[
            landmark_mask
        ].astype(np.float64, copy=False)

        if int(event.sum()) == 0:
            raise ValueError(
                f"第{fold_id}折Landmark {landmark_month}个月没有事件。"
            )

        local_risk_60m = risk_prediction[
            landmark_mask,
            -1,
        ].astype(np.float64, copy=False)

        c_index = float(concordance_index(
            duration,
            -local_log_hazard,
            event,
        ))
        c_index_from_60m_risk = float(concordance_index(
            duration,
            -local_risk_60m,
            event,
        ))
        rank_correlation = float(
            spearmanr(
                local_log_hazard,
                local_risk_60m,
            ).statistic
        )

        if not np.isfinite(rank_correlation):
            raise ValueError(
                f"第{fold_id}折Landmark {landmark_month}个月的"
                "Cox风险排序相关系数不可计算。"
            )
        if rank_correlation < RISK_RANK_CORRELATION_MIN:
            raise ValueError(
                f"第{fold_id}折Landmark {landmark_month}个月的"
                f"5年累计风险与线性风险评分排序不一致："
                f"Spearman={rank_correlation:.9f}。"
            )
        c_index_audit_difference = abs(
            c_index_from_60m_risk - c_index
        )
        if c_index_audit_difference > C_INDEX_AUDIT_TOLERANCE:
            raise ValueError(
                f"第{fold_id}折Landmark {landmark_month}个月的"
                "C-index在累计风险与线性风险评分之间不一致："
                f"{c_index_from_60m_risk:.9f} vs {c_index:.9f}，"
                f"差值={c_index_audit_difference:.9g}。"
            )

        fold_landmark_rows.append({
            "fold_id": fold_id,
            "landmark_month": int(landmark_month),
            "validation_record_n": int(len(landmark_validation_idx)),
            "validation_event_within_60m_n": int(event.sum()),
            "validation_harrell_c_index": c_index,
            "validation_harrell_c_index_from_60m_risk": (
                c_index_from_60m_risk
            ),
            "risk_log_hazard_spearman": rank_correlation,
            "c_index_audit_absolute_difference": (
                c_index_audit_difference
            ),
            "predicted_risk_12m_mean": float(
                cox_oof_risk_long[
                    landmark_validation_idx,
                    1,
                ].mean()
            ),
            "predicted_risk_36m_mean": float(
                cox_oof_risk_long[
                    landmark_validation_idx,
                    5,
                ].mean()
            ),
            "predicted_risk_60m_mean": float(
                cox_oof_risk_long[
                    landmark_validation_idx,
                    9,
                ].mean()
            ),
        })

    print(
        f"第{fold_id}折完成：固定特征"
        f"{EXPECTED_FIXED_COX_FEATURE_N}项，"
        f"收敛警告{len(convergence_messages)}条，"
        f"求解步长{optimizer_step_size}"
    )


# ============================================================
# 8. 将长格式OOF预测还原为患者×Landmark×未来区间
# ============================================================

if not np.isfinite(cox_oof_survival_long).all():
    missing_n = int(np.sum(~np.isfinite(cox_oof_survival_long)))
    raise ValueError(
        f"开发集OOF生存概率仍有{missing_n}个缺失值。"
    )

if not np.isfinite(cox_oof_risk_long).all():
    raise ValueError("开发集OOF风险概率包含NaN或无穷值。")

if not np.isfinite(cox_oof_log_partial_hazard_long).all():
    raise ValueError("开发集OOF线性风险评分包含NaN或无穷值。")

cox_oof_survival = np.full(
    (
        EXPECTED_DEVELOPMENT_N,
        EXPECTED_LANDMARK_N,
        EXPECTED_FUTURE_INTERVAL_N,
    ),
    np.nan,
    dtype=np.float32,
)

cox_oof_risk = np.full_like(
    cox_oof_survival,
    np.nan,
)

cox_oof_log_partial_hazard = np.full(
    (
        EXPECTED_DEVELOPMENT_N,
        EXPECTED_LANDMARK_N,
    ),
    np.nan,
    dtype=np.float32,
)

valid_patient_landmark_mask = row_index_map >= 0
valid_long_idx_from_map = row_index_map[valid_patient_landmark_mask]

cox_oof_survival[valid_patient_landmark_mask] = (
    cox_oof_survival_long[valid_long_idx_from_map]
)
cox_oof_risk[valid_patient_landmark_mask] = (
    cox_oof_risk_long[valid_long_idx_from_map]
)
cox_oof_log_partial_hazard[valid_patient_landmark_mask] = (
    cox_oof_log_partial_hazard_long[valid_long_idx_from_map]
)

if not np.array_equal(
    np.isfinite(cox_oof_survival).all(axis=2),
    valid_patient_landmark_mask,
):
    raise ValueError("患者级OOF生存概率有效掩码与长行映射不一致。")

if not np.array_equal(
    np.isfinite(cox_oof_risk).all(axis=2),
    valid_patient_landmark_mask,
):
    raise ValueError("患者级OOF风险概率有效掩码与长行映射不一致。")

long_monotonic_violation_n = int(np.sum(
    np.diff(cox_oof_survival_long, axis=1)
    > SURVIVAL_TOLERANCE
))

if long_monotonic_violation_n != 0:
    raise ValueError(
        f"OOF生存概率存在{long_monotonic_violation_n}次随时间上升。"
    )

if not np.allclose(
    cox_oof_risk_long,
    1.0 - cox_oof_survival_long,
    atol=1e-6,
):
    raise ValueError("OOF风险概率不等于1减生存概率。")


# ============================================================
# 9. 保存结果
# ============================================================

np.save(
    OUTPUT_DIR / "cox_oof_survival_long.npy",
    cox_oof_survival_long,
)
np.save(
    OUTPUT_DIR / "cox_oof_risk_long.npy",
    cox_oof_risk_long,
)
np.save(
    OUTPUT_DIR / "cox_oof_log_partial_hazard_long.npy",
    cox_oof_log_partial_hazard_long,
)
np.save(
    OUTPUT_DIR / "cox_oof_survival.npy",
    cox_oof_survival,
)
np.save(
    OUTPUT_DIR / "cox_oof_risk.npy",
    cox_oof_risk,
)
np.save(
    OUTPUT_DIR / "cox_oof_log_partial_hazard.npy",
    cox_oof_log_partial_hazard,
)
np.save(
    OUTPUT_DIR / "valid_patient_landmark_mask.npy",
    valid_patient_landmark_mask,
)
np.save(
    OUTPUT_DIR / "future_end_months.npy",
    FUTURE_END_MONTHS,
)
np.save(
    OUTPUT_DIR / "landmark_months.npy",
    LANDMARK_MONTHS,
)

fold_summary_table = pd.DataFrame(fold_summary_rows)
fold_landmark_table = pd.DataFrame(fold_landmark_rows)

fold_summary_table.to_csv(
    OUTPUT_DIR / "cox_oof_fold_summary.csv",
    index=False,
    encoding="utf-8-sig",
)

fold_landmark_table.to_csv(
    OUTPUT_DIR / "cox_oof_fold_landmark_metrics.csv",
    index=False,
    encoding="utf-8-sig",
)

landmark_summary_table = (
    fold_landmark_table
    .groupby("landmark_month", as_index=False)
    .agg(
        validation_record_n=("validation_record_n", "sum"),
        validation_event_within_60m_n=(
            "validation_event_within_60m_n",
            "sum",
        ),
        mean_harrell_c_index=(
            "validation_harrell_c_index",
            "mean",
        ),
        sd_harrell_c_index=(
            "validation_harrell_c_index",
            "std",
        ),
        predicted_risk_12m_mean=(
            "predicted_risk_12m_mean",
            "mean",
        ),
        predicted_risk_36m_mean=(
            "predicted_risk_36m_mean",
            "mean",
        ),
        predicted_risk_60m_mean=(
            "predicted_risk_60m_mean",
            "mean",
        ),
    )
)

landmark_summary_table.to_csv(
    OUTPUT_DIR / "cox_oof_landmark_summary.csv",
    index=False,
    encoding="utf-8-sig",
)

model_manifest = {
    "step": 7,
    "model_name": "Unpenalized Cox Super Landmark fixed95",
    "model_family": "stratified_cox_proportional_hazards",
    "penalization": "none",
    "hyperparameter_tuning": False,
    "fixed_feature_n": EXPECTED_FIXED_COX_FEATURE_N,
    "laboratory_history_representation": [
        "baseline",
        "current",
        "slope_per_year",
    ],
    "removed_laboratory_representations": [
        "change_from_baseline",
        "history_mean",
        "history_sd",
    ],
    "removed_history_features": [
        "history_observed_step_count",
        "ART_regimen_changed",
    ],
    "reference_baseline_onehot": REFERENCE_BASELINE_ONEHOT,
    "reference_current_art": REFERENCE_CURRENT_ART,
    "reference_cumulative_art": REFERENCE_CUMULATIVE_ART,
    "derived_art_features": [
        "current_ART_Other_at_landmark",
        "ART_Other_cum_month_at_landmark",
    ],
    "coefficient_structure": "shared_across_all_landmarks",
    "baseline_hazard_structure": "landmark_specific_strata",
    "landmark_months": LANDMARK_MONTHS.tolist(),
    "future_end_months": FUTURE_END_MONTHS.astype(int).tolist(),
    "fold_preprocessing": (
        "training_fold_only_standardization_with_fixed_feature_set"
    ),
    "optimizer_step_sizes": OPTIMIZER_STEP_SIZES,
    "test_set_used": False,
    "lifelines_version": lifelines.__version__,
    "python_version": sys.version,
}

with open(
    OUTPUT_DIR / "step7_manifest.json",
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        model_manifest,
        file,
        ensure_ascii=False,
        indent=2,
    )


# ============================================================
# 10. 输出最终核查结果
# ============================================================

print("\n" + "=" * 72)
print("Step 7完成：固定95项未惩罚Super Landmark Cox OOF预测已生成")
print("=" * 72)
print("Cox惩罚：无")
print("超参数调优：无")
print("固定传统Cox特征数：", EXPECTED_FIXED_COX_FEATURE_N)
print("OOF长格式生存概率：", cox_oof_survival_long.shape)
print("OOF患者级生存概率：", cox_oof_survival.shape)
print("有效患者-Landmark起点数：", int(valid_patient_landmark_mask.sum()))
print("生存概率单调性违反次数：", long_monotonic_violation_n)
print("\n五折OOF核查：")
print(fold_summary_table.to_string(index=False))
print("\n各Landmark OOF汇总：")
print(landmark_summary_table.to_string(index=False))
print("\n处理原则：")
print("1. Cox不调参，也不使用任何惩罚项。")
print("2. 六个Landmark共享一套回归系数，并按Landmark分层基线风险。")
print("3. 每项实验室仅保留baseline、current和slope_per_year。")
print("4. 删除实验室change、history_mean和history_sd。")
print("5. 删除history_observed_step_count，保留missing_ratio和最近观测间隔。")
print("6. 删除ART_regimen_changed，仅保留ART_regimen_change_count。")
print("7. ART当前方案和累计暴露均显式加入Other并采用固定参考方案。")
print("8. 五折始终使用同一套95项特征，不自动删除不同变量。")
print("9. 连续变量标准化参数只来自对应训练折。")
print("10. 锁定测试集没有被读取。")
print("\n输出目录：", OUTPUT_DIR)
print("Step 7运行完成。")
