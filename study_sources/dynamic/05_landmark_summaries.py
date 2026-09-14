# ============================================================
# Step 5：构建Cox和RSF共用的Landmark原始历史汇总特征
#
# 代码逻辑：
# 1. 直接读取Step 3未标准化原始张量，避免从标准化数据反推临床值。
# 2. Age按Landmark时间更新：基线Age + Landmark月数/12。
# 3. BMI、Oppinfection、Sex、Marriage、Course、WHOstage只取基线值。
# 4. 15项实验室指标生成基线、当前、变化、均值、标准差和年斜率。
# 5. 持续性疾病状态生成Landmark状态和距首次记录阳性的月数。
# 6. 代谢用药按当前数据编码生成Landmark状态和距首次用药月数。
# 7. ART累计暴露取Landmark当前累计月数；ART方案取Landmark当前方案。
# 8. 加入时间行完整度和ART方案变更信息。
# 9. Step 5只构建原始汇总特征；标准化和常数特征删除在后续训练折内完成。
# 10. 本步骤不训练模型，也不使用锁定测试集结局进行模型选择。
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

STEP3_DIR = Path(
    "__CKD_WORKDIR__/rolling_5y_step3_raw_features"
)

OUTPUT_DIR = Path(
    "__CKD_WORKDIR__/rolling_5y_step5_landmark_summary"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

N_SPLITS = 5
INTERVAL_WIDTH_MONTH = 6
EXPECTED_TOTAL_N = 31911
EXPECTED_DEVELOPMENT_N = 22337
EXPECTED_TEST_N = 9574
EXPECTED_HISTORY_STEP_N = 11
EXPECTED_LANDMARK_N = 6
EXPECTED_CONTINUOUS_N = 25
EXPECTED_BINARY_N = 18
EXPECTED_CATEGORICAL_N = 4
EXPECTED_SUMMARY_FEATURE_N = 146
PADDING_CATEGORY = "__NO_TIME_ROW__"

EXPECTED_LANDMARK_MONTHS = np.array(
    [0, 12, 24, 36, 48, 60],
    dtype=np.int32,
)

EXPECTED_LANDMARK_BINS = np.array(
    [0, 2, 4, 6, 8, 10],
    dtype=np.int32,
)


# ============================================================
# 2. 固定原始变量角色
# ============================================================

AGE_NAME = "Age"
BMI_NAME = "BMI"
STATIC_BINARY_NAMES = ["Oppinfection"]
STATIC_CATEGORICAL_NAMES = [
    "Sex",
    "Marriage",
    "Course",
    "WHOstage",
]

LONGITUDINAL_LAB_NAMES = [
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

PERSISTENT_STATUS_NAMES = [
    "CVD_status",
    "diabetes_status",
    "hypertension_status",
    "hypercholesterolemia_status",
    "HBV_status",
    "HCV_status",
]

METABOLIC_MED_NAMES = [
    "antidiabetic_med",
    "antihypertensive_med",
    "antilipid_med",
]

CURRENT_ART_NAMES = [
    "current_TDF_NNRTI_3TC_FTC",
    "current_TDF_PI_3TC_FTC",
    "current_nonTDF_PI",
    "current_BIC_FTC_TAF",
    "current_EVGc_FTC_TAF",
    "current_TDF_INSTI_3TC_FTC",
    "current_nonTDF_DTG",
    "current_nonTDF_traditional_NNRTI",
]

ART_CUMULATIVE_NAMES = [
    "TDF_NNRTI_3TC_FTC_cum_month",
    "TDF_PI_3TC_FTC_cum_month",
    "nonTDF_PI_cum_month",
    "BIC_FTC_TAF_cum_month",
    "EVGc_FTC_TAF_cum_month",
    "TDF_INSTI_3TC_FTC_cum_month",
    "nonTDF_DTG_cum_month",
    "nonTDF_traditional_NNRTI_cum_month",
]

FIXED_CATEGORY_LEVELS = {
    "Sex": [
        "Female",
        "Male",
    ],
    "Marriage": [
        "Divorced-separated-or-widowed",
        "Married or cohabiting",
        "Never married",
        "Others",
    ],
    "Course": [
        "Drugs",
        "Heterosexual",
        "Male to male",
        "Others",
    ],
    "WHOstage": [
        "1",
        "2",
        "3",
        "4",
    ],
}


# ============================================================
# 3. 检查Step 1～3输入文件
# ============================================================

required_files = [
    STEP1_DIR / "development_idx.npy",
    STEP1_DIR / "test_idx.npy",
    STEP1_DIR / "sequence_row_mask.npy",
    STEP1_DIR / "landmark_months.npy",
    STEP1_DIR / "landmark_bins.npy",
    STEP1_DIR / "landmark_eligible_mask.npy",
    STEP1_DIR / "prediction_origin_mask.npy",
    STEP2_DIR / "fold_id_all.npy",
    STEP3_DIR / "continuous_raw_0_60.npy",
    STEP3_DIR / "binary_raw_0_60.npy",
    STEP3_DIR / "categorical_raw_0_60.npy",
    STEP3_DIR / "sequence_row_mask.npy",
    STEP3_DIR / "patient_info_with_fold.csv",
    STEP3_DIR / "feature_groups.json",
]

missing_files = [
    str(path) for path in required_files if not path.exists()
]

if missing_files:
    raise FileNotFoundError(
        "以下输入文件不存在：\n" + "\n".join(missing_files)
    )


# ============================================================
# 4. 读取患者划分、Landmark和变量定义
# ============================================================

development_idx = np.load(
    STEP1_DIR / "development_idx.npy"
).astype(np.int32)

test_idx = np.load(
    STEP1_DIR / "test_idx.npy"
).astype(np.int32)

sequence_row_mask_step1 = np.load(
    STEP1_DIR / "sequence_row_mask.npy"
).astype(bool)

landmark_months = np.load(
    STEP1_DIR / "landmark_months.npy"
).astype(np.int32)

landmark_bins = np.load(
    STEP1_DIR / "landmark_bins.npy"
).astype(np.int32)

landmark_eligible_mask_all = np.load(
    STEP1_DIR / "landmark_eligible_mask.npy"
).astype(bool)

prediction_origin_mask_all = np.load(
    STEP1_DIR / "prediction_origin_mask.npy"
).astype(bool)

fold_id_all = np.load(
    STEP2_DIR / "fold_id_all.npy"
).astype(np.int8)

patient_info = pd.read_csv(
    STEP3_DIR / "patient_info_with_fold.csv",
    encoding="utf-8-sig",
    dtype={
        "ID": "string",
        "center": "string",
        "analysis_split": "string",
    },
)

with open(
    STEP3_DIR / "feature_groups.json",
    "r",
    encoding="utf-8",
) as file:
    feature_groups = json.load(file)

continuous_names = list(feature_groups["continuous_vars"])
binary_names = list(feature_groups["binary_vars"])
categorical_names = list(feature_groups["categorical_vars"])


# ============================================================
# 5. 读取Step 3原始张量
# ============================================================

continuous_raw = np.load(
    STEP3_DIR / "continuous_raw_0_60.npy",
    mmap_mode="r",
)

binary_raw = np.load(
    STEP3_DIR / "binary_raw_0_60.npy",
    mmap_mode="r",
)

categorical_raw = np.load(
    STEP3_DIR / "categorical_raw_0_60.npy",
    mmap_mode="r",
)

sequence_row_mask_step3 = np.load(
    STEP3_DIR / "sequence_row_mask.npy"
).astype(bool)


# ============================================================
# 6. 核对患者顺序、张量形状和Landmark定义
# ============================================================

expected_shapes = {
    "continuous_raw": (
        EXPECTED_TOTAL_N,
        EXPECTED_HISTORY_STEP_N,
        EXPECTED_CONTINUOUS_N,
    ),
    "binary_raw": (
        EXPECTED_TOTAL_N,
        EXPECTED_HISTORY_STEP_N,
        EXPECTED_BINARY_N,
    ),
    "categorical_raw": (
        EXPECTED_TOTAL_N,
        EXPECTED_HISTORY_STEP_N,
        EXPECTED_CATEGORICAL_N,
    ),
    "sequence_row_mask": (
        EXPECTED_TOTAL_N,
        EXPECTED_HISTORY_STEP_N,
    ),
    "landmark_eligible_mask": (
        EXPECTED_TOTAL_N,
        EXPECTED_LANDMARK_N,
    ),
    "prediction_origin_mask": (
        EXPECTED_TOTAL_N,
        EXPECTED_LANDMARK_N,
    ),
}

actual_shapes = {
    "continuous_raw": continuous_raw.shape,
    "binary_raw": binary_raw.shape,
    "categorical_raw": categorical_raw.shape,
    "sequence_row_mask": sequence_row_mask_step3.shape,
    "landmark_eligible_mask": landmark_eligible_mask_all.shape,
    "prediction_origin_mask": prediction_origin_mask_all.shape,
}

for name, expected_shape in expected_shapes.items():
    if actual_shapes[name] != expected_shape:
        raise ValueError(
            f"{name}形状为{actual_shapes[name]}，"
            f"应为{expected_shape}。"
        )

if not np.array_equal(
    sequence_row_mask_step1,
    sequence_row_mask_step3,
):
    raise ValueError("Step 1和Step 3的时间行掩码不一致。")

if len(patient_info) != EXPECTED_TOTAL_N:
    raise ValueError(
        f"患者数为{len(patient_info)}，"
        f"应为{EXPECTED_TOTAL_N}。"
    )

if not np.array_equal(
    patient_info["patient_index"].to_numpy(dtype=np.int32),
    np.arange(EXPECTED_TOTAL_N, dtype=np.int32),
):
    raise ValueError("patient_info患者顺序与原始张量不一致。")

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

if len(continuous_names) != EXPECTED_CONTINUOUS_N:
    raise ValueError("连续变量数不是25。")

if len(binary_names) != EXPECTED_BINARY_N:
    raise ValueError("二分类变量数不是18。")

if categorical_names != STATIC_CATEGORICAL_NAMES:
    raise ValueError(
        f"多分类变量顺序为{categorical_names}，"
        f"应为{STATIC_CATEGORICAL_NAMES}。"
    )

if not np.array_equal(landmark_months, EXPECTED_LANDMARK_MONTHS):
    raise ValueError(
        f"Landmark月份为{landmark_months.tolist()}，"
        f"应为{EXPECTED_LANDMARK_MONTHS.tolist()}。"
    )

if not np.array_equal(landmark_bins, EXPECTED_LANDMARK_BINS):
    raise ValueError(
        f"Landmark时间点为{landmark_bins.tolist()}，"
        f"应为{EXPECTED_LANDMARK_BINS.tolist()}。"
    )

if not np.all(fold_id_all[development_idx] >= 0):
    raise ValueError("部分开发集患者没有五折编号。")

if not np.all(fold_id_all[test_idx] == -1):
    raise ValueError("锁定测试集患者进入了开发集五折。")


# ============================================================
# 7. 核对变量分组完整性并生成列索引
# ============================================================

expected_continuous = (
    [AGE_NAME, BMI_NAME]
    + LONGITUDINAL_LAB_NAMES
    + ART_CUMULATIVE_NAMES
)

expected_binary = (
    STATIC_BINARY_NAMES
    + PERSISTENT_STATUS_NAMES
    + METABOLIC_MED_NAMES
    + CURRENT_ART_NAMES
)

if set(expected_continuous) != set(continuous_names):
    missing = sorted(set(expected_continuous) - set(continuous_names))
    extra = sorted(set(continuous_names) - set(expected_continuous))
    raise ValueError(
        f"连续变量分组不完整。缺少：{missing}；多出：{extra}。"
    )

if set(expected_binary) != set(binary_names):
    missing = sorted(set(expected_binary) - set(binary_names))
    extra = sorted(set(binary_names) - set(expected_binary))
    raise ValueError(
        f"二分类变量分组不完整。缺少：{missing}；多出：{extra}。"
    )


def get_indices(all_names, selected_names):
    return np.array(
        [all_names.index(name) for name in selected_names],
        dtype=np.int32,
    )


age_index = int(continuous_names.index(AGE_NAME))
bmi_index = int(continuous_names.index(BMI_NAME))
lab_indices = get_indices(
    continuous_names,
    LONGITUDINAL_LAB_NAMES,
)
art_cumulative_indices = get_indices(
    continuous_names,
    ART_CUMULATIVE_NAMES,
)

static_binary_indices = get_indices(
    binary_names,
    STATIC_BINARY_NAMES,
)
persistent_status_indices = get_indices(
    binary_names,
    PERSISTENT_STATUS_NAMES,
)
metabolic_med_indices = get_indices(
    binary_names,
    METABOLIC_MED_NAMES,
)
current_art_indices = get_indices(
    binary_names,
    CURRENT_ART_NAMES,
)


# ============================================================
# 8. 固定14项基线One-Hot特征名称
# ============================================================

static_onehot_names = []
for variable_name in STATIC_CATEGORICAL_NAMES:
    for category in FIXED_CATEGORY_LEVELS[variable_name]:
        static_onehot_names.append(
            f"{variable_name}_{category}_baseline"
        )

if len(static_onehot_names) != 14:
    raise ValueError(
        f"固定One-Hot变量数为{len(static_onehot_names)}，应为14。"
    )


# ============================================================
# 9. 定义最终146项汇总特征名称
# ============================================================

summary_feature_names = []
summary_feature_roles = []
summary_source_variables = []


def add_feature(name, role, source_variable):
    summary_feature_names.append(name)
    summary_feature_roles.append(role)
    summary_source_variables.append(source_variable)


add_feature(
    "Age_at_landmark",
    "deterministic_time_updated",
    "Age",
)
add_feature(
    "BMI_baseline",
    "baseline_static",
    "BMI",
)
add_feature(
    "Oppinfection_baseline",
    "baseline_static",
    "Oppinfection",
)

for name in static_onehot_names:
    source_variable = name.split("_", 1)[0]
    add_feature(
        name,
        "baseline_static_onehot",
        source_variable,
    )

for suffix, role in [
    ("baseline", "longitudinal_baseline"),
    ("current", "longitudinal_current"),
    ("change_from_baseline", "longitudinal_change"),
    ("history_mean", "longitudinal_mean"),
    ("history_sd", "longitudinal_variability"),
    ("slope_per_year", "longitudinal_slope"),
]:
    for name in LONGITUDINAL_LAB_NAMES:
        add_feature(
            f"{name}_{suffix}",
            role,
            name,
        )

# 名称顺序必须与build_landmark_summary中的np.concatenate顺序完全一致。
# 先列出全部持续性状态，再列出全部首次阳性时长。
for name in PERSISTENT_STATUS_NAMES:
    add_feature(
        f"{name}_at_landmark",
        "persistent_status_current",
        name,
    )

for name in PERSISTENT_STATUS_NAMES:
    add_feature(
        f"{name}_months_since_first_positive",
        "persistent_status_duration",
        name,
    )

# 先列出全部代谢用药状态，再列出全部首次用药时长。
for name in METABOLIC_MED_NAMES:
    add_feature(
        f"{name}_at_landmark",
        "medication_status_current",
        name,
    )

for name in METABOLIC_MED_NAMES:
    add_feature(
        f"{name}_months_since_first_positive",
        "medication_exposure_duration",
        name,
    )

for name in ART_CUMULATIVE_NAMES:
    add_feature(
        f"{name}_at_landmark",
        "cumulative_art_exposure",
        name,
    )

for name in CURRENT_ART_NAMES:
    add_feature(
        f"{name}_at_landmark",
        "current_art_regimen",
        name,
    )

for name, role in [
    ("history_observed_step_count", "history_availability"),
    ("history_missing_step_ratio", "history_availability"),
    ("months_since_last_observed_step", "history_recency"),
    ("ART_regimen_changed", "art_regimen_history"),
    ("ART_regimen_change_count", "art_regimen_history"),
]:
    add_feature(
        name,
        role,
        "derived_from_time_history",
    )

if len(summary_feature_names) != EXPECTED_SUMMARY_FEATURE_N:
    raise ValueError(
        f"汇总特征数为{len(summary_feature_names)}，"
        f"应为{EXPECTED_SUMMARY_FEATURE_N}。"
    )

if len(set(summary_feature_names)) != len(summary_feature_names):
    raise ValueError("汇总特征名称存在重复。")


# ============================================================
# 10. 定义基线分类变量One-Hot函数
# ============================================================


def encode_baseline_categories(baseline_categories):
    baseline_categories = np.asarray(baseline_categories).astype(str)

    if baseline_categories.ndim != 2:
        raise ValueError("基线分类变量必须是二维数组。")

    if baseline_categories.shape[1] != EXPECTED_CATEGORICAL_N:
        raise ValueError("基线分类变量列数不是4。")

    blocks = []

    for variable_index, variable_name in enumerate(
        STATIC_CATEGORICAL_NAMES
    ):
        values = baseline_categories[:, variable_index]
        allowed = FIXED_CATEGORY_LEVELS[variable_name]
        unknown = sorted(set(values.tolist()) - set(allowed))

        if unknown:
            raise ValueError(
                f"{variable_name}存在未预设分类：{unknown}。"
            )

        block = np.column_stack([
            (values == category).astype(np.float32)
            for category in allowed
        ])

        if not np.allclose(
            block.sum(axis=1, dtype=np.float64),
            1.0,
        ):
            raise ValueError(
                f"{variable_name}基线One-Hot编码不唯一。"
            )

        blocks.append(block)

    output = np.concatenate(blocks, axis=1).astype(np.float32)

    if output.shape[1] != 14:
        raise ValueError("基线One-Hot结果不是14项。")

    return output


# ============================================================
# 11. 定义时间编码审计函数
#
# 代码逻辑：
# BMI按研究定义应为基线静态指标，因此若随访记录发生变化则报错；
# Age允许随时间增加，只比较其记录值与基线Age+时间的差异；
# Oppinfection和Course等基线变量的后续变化只做审计，正式特征仍取基线；
# 持续性状态和当前编码下的代谢用药不允许1→0；
# ART累计暴露不允许下降；同一真实时间行最多一个当前ART方案为1。
# ============================================================


def audit_sequence_encoding(
    continuous_values,
    binary_values,
    categorical_values,
    sequence_mask,
    dataset_name,
):
    continuous_values = np.asarray(
        continuous_values,
        dtype=np.float32,
    )
    binary_values = np.asarray(
        binary_values,
        dtype=np.float32,
    )
    categorical_values = np.asarray(
        categorical_values,
    ).astype(str)
    sequence_mask = np.asarray(sequence_mask, dtype=bool)

    n_patients, n_steps, _ = continuous_values.shape
    patient_rows = np.arange(n_patients)

    first_index = np.argmax(sequence_mask, axis=1)
    baseline_continuous = continuous_values[
        patient_rows,
        first_index,
        :,
    ]
    baseline_binary = binary_values[
        patient_rows,
        first_index,
        :,
    ]
    baseline_categorical = categorical_values[
        patient_rows,
        first_index,
        :,
    ]

    baseline_age = baseline_continuous[:, age_index]
    baseline_bmi = baseline_continuous[:, bmi_index]
    baseline_opp = baseline_binary[:, static_binary_indices[0]]

    bmi_changed_patient = np.zeros(n_patients, dtype=bool)
    opp_changed_patient = np.zeros(n_patients, dtype=bool)
    category_changed_patient = {
        name: np.zeros(n_patients, dtype=bool)
        for name in STATIC_CATEGORICAL_NAMES
    }

    age_abs_difference = []
    persistent_1_to_0_count = 0
    cumulative_decrease_count = 0
    metabolic_0_to_1_count = 0
    metabolic_1_to_0_count = 0
    multi_art_row_count = 0

    previous_binary = np.zeros(
        (n_patients, EXPECTED_BINARY_N),
        dtype=np.float32,
    )
    previous_cumulative = np.zeros(
        (n_patients, len(ART_CUMULATIVE_NAMES)),
        dtype=np.float32,
    )
    has_previous = np.zeros(n_patients, dtype=bool)

    for step_index in range(n_steps):
        observed = sequence_mask[:, step_index]

        if not np.any(observed):
            continue

        expected_age = (
            baseline_age
            + step_index * INTERVAL_WIDTH_MONTH / 12.0
        )
        observed_age = continuous_values[:, step_index, age_index]
        age_abs_difference.extend(
            np.abs(observed_age[observed] - expected_age[observed]).tolist()
        )

        observed_bmi = continuous_values[:, step_index, bmi_index]
        bmi_changed_patient |= (
            observed
            & (np.abs(observed_bmi - baseline_bmi) > 1e-6)
        )

        observed_opp = binary_values[
            :, step_index, static_binary_indices[0]
        ]
        opp_changed_patient |= (
            observed
            & (observed_opp != baseline_opp)
        )

        for category_index, category_name in enumerate(
            STATIC_CATEGORICAL_NAMES
        ):
            observed_category = categorical_values[
                :, step_index, category_index
            ]
            category_changed_patient[category_name] |= (
                observed
                & (observed_category != baseline_categorical[:, category_index])
            )

        current_binary = binary_values[:, step_index, :]
        current_cumulative = continuous_values[
            :, step_index, art_cumulative_indices
        ]
        comparable = observed & has_previous

        if np.any(comparable):
            previous_persistent = previous_binary[
                :, persistent_status_indices
            ]
            current_persistent = current_binary[
                :, persistent_status_indices
            ]
            persistent_1_to_0_count += int(np.sum(
                comparable[:, None]
                & (previous_persistent == 1.0)
                & (current_persistent == 0.0)
            ))

            cumulative_decrease_count += int(np.sum(
                comparable[:, None]
                & (
                    current_cumulative
                    < previous_cumulative - 1e-6
                )
            ))

            previous_metabolic = previous_binary[
                :, metabolic_med_indices
            ]
            current_metabolic = current_binary[
                :, metabolic_med_indices
            ]
            metabolic_0_to_1_count += int(np.sum(
                comparable[:, None]
                & (previous_metabolic == 0.0)
                & (current_metabolic == 1.0)
            ))
            metabolic_1_to_0_count += int(np.sum(
                comparable[:, None]
                & (previous_metabolic == 1.0)
                & (current_metabolic == 0.0)
            ))

        current_art_sum = current_binary[
            :, current_art_indices
        ].sum(axis=1)
        multi_art_row_count += int(np.sum(
            observed & (current_art_sum > 1.0 + 1e-6)
        ))

        previous_binary[observed] = current_binary[observed]
        previous_cumulative[observed] = current_cumulative[observed]
        has_previous[observed] = True

    age_abs_difference = np.asarray(
        age_abs_difference,
        dtype=np.float64,
    )

    audit = {
        "dataset": dataset_name,
        "patient_n": int(n_patients),
        "observed_time_row_n": int(sequence_mask.sum()),
        "age_record_vs_expected_median_abs_difference": float(
            np.median(age_abs_difference)
        ),
        "age_record_vs_expected_max_abs_difference": float(
            np.max(age_abs_difference)
        ),
        "BMI_changed_patient_n": int(bmi_changed_patient.sum()),
        "Oppinfection_changed_patient_n": int(
            opp_changed_patient.sum()
        ),
        "Sex_changed_patient_n": int(
            category_changed_patient["Sex"].sum()
        ),
        "Marriage_changed_patient_n": int(
            category_changed_patient["Marriage"].sum()
        ),
        "Course_changed_patient_n": int(
            category_changed_patient["Course"].sum()
        ),
        "WHOstage_changed_patient_n": int(
            category_changed_patient["WHOstage"].sum()
        ),
        "persistent_status_1_to_0_transition_n": int(
            persistent_1_to_0_count
        ),
        "art_cumulative_decrease_n": int(
            cumulative_decrease_count
        ),
        "metabolic_med_0_to_1_transition_n": int(
            metabolic_0_to_1_count
        ),
        "metabolic_med_1_to_0_transition_n": int(
            metabolic_1_to_0_count
        ),
        "multi_current_art_row_n": int(multi_art_row_count),
    }

    if audit["BMI_changed_patient_n"] > 0:
        raise ValueError(
            f"{dataset_name}存在{audit['BMI_changed_patient_n']}名患者"
            "的BMI在真实随访时间行中发生变化；"
            "当前研究已将BMI定义为基线静态指标，请先核对原始数据。"
        )

    if audit["persistent_status_1_to_0_transition_n"] > 0:
        raise ValueError(
            f"{dataset_name}持续性疾病状态出现"
            f"{audit['persistent_status_1_to_0_transition_n']}次1→0变化。"
        )

    if audit["art_cumulative_decrease_n"] > 0:
        raise ValueError(
            f"{dataset_name}ART累计暴露出现"
            f"{audit['art_cumulative_decrease_n']}次下降。"
        )

    if audit["metabolic_med_1_to_0_transition_n"] > 0:
        raise ValueError(
            f"{dataset_name}代谢用药状态出现"
            f"{audit['metabolic_med_1_to_0_transition_n']}次1→0变化；"
            "当前汇总规则假定首次用药后持续编码为1。"
        )

    if audit["multi_current_art_row_n"] > 0:
        raise ValueError(
            f"{dataset_name}存在{audit['multi_current_art_row_n']}个"
            "真实时间行同时属于多个当前ART方案。"
        )

    return audit


# ============================================================
# 12. 定义Landmark历史汇总函数
# ============================================================


def build_landmark_summary(
    continuous_values,
    binary_values,
    categorical_values,
    sequence_mask,
):
    continuous_values = np.asarray(
        continuous_values,
        dtype=np.float32,
    )
    binary_values = np.asarray(
        binary_values,
        dtype=np.float32,
    )
    categorical_values = np.asarray(
        categorical_values,
    ).astype(str)
    sequence_mask = np.asarray(sequence_mask, dtype=bool)

    n_patients = continuous_values.shape[0]
    patient_rows = np.arange(n_patients)

    if continuous_values.shape != (
        n_patients,
        EXPECTED_HISTORY_STEP_N,
        EXPECTED_CONTINUOUS_N,
    ):
        raise ValueError("连续变量张量形状不正确。")

    if binary_values.shape != (
        n_patients,
        EXPECTED_HISTORY_STEP_N,
        EXPECTED_BINARY_N,
    ):
        raise ValueError("二分类变量张量形状不正确。")

    if categorical_values.shape != (
        n_patients,
        EXPECTED_HISTORY_STEP_N,
        EXPECTED_CATEGORICAL_N,
    ):
        raise ValueError("分类变量张量形状不正确。")

    if sequence_mask.shape != (
        n_patients,
        EXPECTED_HISTORY_STEP_N,
    ):
        raise ValueError("时间行掩码形状不正确。")

    if not np.isfinite(
        continuous_values[sequence_mask]
    ).all():
        raise ValueError("真实时间行的连续变量存在NaN或无穷值。")

    if not np.isfinite(
        binary_values[sequence_mask]
    ).all():
        raise ValueError("真实时间行的二分类变量存在NaN或无穷值。")

    if not np.isin(
        binary_values[sequence_mask],
        [0.0, 1.0],
    ).all():
        raise ValueError("真实时间行的二分类变量存在非0/1值。")

    summary_values = np.zeros(
        (
            n_patients,
            EXPECTED_LANDMARK_N,
            EXPECTED_SUMMARY_FEATURE_N,
        ),
        dtype=np.float32,
    )

    for landmark_index, landmark_bin in enumerate(landmark_bins):
        landmark_month = int(landmark_months[landmark_index])
        history_end = int(landmark_bin) + 1

        history_continuous = continuous_values[:, :history_end, :]
        history_binary = binary_values[:, :history_end, :]
        history_categorical = categorical_values[:, :history_end, :]
        history_mask = sequence_mask[:, :history_end]

        observed_count = history_mask.sum(axis=1).astype(np.float32)

        if np.any(observed_count < 1):
            raise ValueError(
                f"Landmark {landmark_month}个月存在无历史真实时间行的患者。"
            )

        first_observed_index = np.argmax(history_mask, axis=1)
        last_observed_index = (
            history_end
            - 1
            - np.argmax(history_mask[:, ::-1], axis=1)
        )

        baseline_continuous = history_continuous[
            patient_rows,
            first_observed_index,
            :,
        ]
        current_continuous = history_continuous[
            patient_rows,
            last_observed_index,
            :,
        ]
        baseline_binary = history_binary[
            patient_rows,
            first_observed_index,
            :,
        ]
        current_binary = history_binary[
            patient_rows,
            last_observed_index,
            :,
        ]
        baseline_categorical = history_categorical[
            patient_rows,
            first_observed_index,
            :,
        ]

        age_at_landmark = (
            baseline_continuous[:, age_index]
            + landmark_month / 12.0
        ).astype(np.float32)
        bmi_baseline = baseline_continuous[:, bmi_index].astype(np.float32)
        opp_baseline = baseline_binary[
            :, static_binary_indices[0]
        ].astype(np.float32)
        baseline_onehot = encode_baseline_categories(
            baseline_categorical
        )

        lab_history = history_continuous[:, :, lab_indices]
        observed_3d = history_mask[:, :, None]
        lab_masked = np.where(
            observed_3d,
            lab_history,
            0.0,
        ).astype(np.float64)

        denominator = observed_count.astype(np.float64)[:, None]
        lab_sum = lab_masked.sum(axis=1, dtype=np.float64)
        lab_square_sum = np.square(
            lab_masked
        ).sum(axis=1, dtype=np.float64)

        lab_mean = lab_sum / denominator
        lab_variance = (
            lab_square_sum / denominator
            - np.square(lab_mean)
        )
        lab_variance = np.maximum(lab_variance, 0.0)
        lab_sd = np.sqrt(lab_variance)

        time_year = (
            np.arange(history_end, dtype=np.float64)
            * INTERVAL_WIDTH_MONTH
            / 12.0
        )
        observed_float = history_mask.astype(np.float64)
        time_sum = (
            observed_float * time_year[None, :]
        ).sum(axis=1)
        time_square_sum = (
            observed_float * np.square(time_year)[None, :]
        ).sum(axis=1)
        value_time_sum = (
            lab_masked * time_year[None, :, None]
        ).sum(axis=1)

        slope_denominator = (
            time_square_sum
            - np.square(time_sum)
            / observed_count.astype(np.float64)
        )
        slope_numerator = (
            value_time_sum
            - time_sum[:, None] * lab_sum / denominator
        )

        lab_slope = np.zeros_like(
            slope_numerator,
            dtype=np.float64,
        )
        valid_slope = slope_denominator > 1e-12
        lab_slope[valid_slope] = (
            slope_numerator[valid_slope]
            / slope_denominator[valid_slope, None]
        )

        lab_baseline = baseline_continuous[:, lab_indices]
        lab_current = current_continuous[:, lab_indices]
        lab_change = lab_current - lab_baseline

        persistent_history = history_binary[
            :, :, persistent_status_indices
        ]
        persistent_current = current_binary[
            :, persistent_status_indices
        ].astype(np.float32)

        positive_observed = (
            history_mask[:, :, None]
            & (persistent_history == 1.0)
        )
        ever_positive = positive_observed.any(axis=1)
        first_positive_index = np.argmax(
            positive_observed,
            axis=1,
        )
        months_since_first_positive = np.where(
            ever_positive,
            landmark_month
            - first_positive_index.astype(np.float32)
            * INTERVAL_WIDTH_MONTH,
            0.0,
        ).astype(np.float32)

        if not np.array_equal(
            ever_positive.astype(np.float32),
            persistent_current,
        ):
            raise ValueError(
                f"Landmark {landmark_month}个月持续性状态的"
                "当前值与历史曾阳性不一致。"
            )

        metabolic_history = history_binary[
            :, :, metabolic_med_indices
        ]
        metabolic_current = current_binary[
            :, metabolic_med_indices
        ].astype(np.float32)
        metabolic_positive_observed = (
            history_mask[:, :, None]
            & (metabolic_history == 1.0)
        )
        metabolic_ever_positive = metabolic_positive_observed.any(axis=1)
        metabolic_first_positive_index = np.argmax(
            metabolic_positive_observed,
            axis=1,
        )
        metabolic_months_since_first_positive = np.where(
            metabolic_ever_positive,
            landmark_month
            - metabolic_first_positive_index.astype(np.float32)
            * INTERVAL_WIDTH_MONTH,
            0.0,
        ).astype(np.float32)

        if not np.array_equal(
            metabolic_ever_positive.astype(np.float32),
            metabolic_current,
        ):
            raise ValueError(
                f"Landmark {landmark_month}个月代谢用药当前值与"
                "历史曾用状态不一致；当前编码不再满足首次用药后持续为1。"
            )

        art_cumulative_current = current_continuous[
            :, art_cumulative_indices
        ].astype(np.float32)
        current_art_state = current_binary[
            :, current_art_indices
        ].astype(np.float32)

        art_history = history_binary[:, :, current_art_indices]

        # 同一时间点方案向量发生变化只计1次，不能按8个分量相加。
        art_regimen_change_count = np.zeros(
            n_patients,
            dtype=np.float32,
        )
        previous_art_state = np.zeros(
            (n_patients, len(CURRENT_ART_NAMES)),
            dtype=np.float32,
        )
        has_previous_art = np.zeros(n_patients, dtype=bool)

        for step_index in range(history_end):
            observed = history_mask[:, step_index]
            current_state = art_history[:, step_index, :]
            changed = (
                observed
                & has_previous_art
                & np.any(
                    current_state != previous_art_state,
                    axis=1,
                )
            )
            art_regimen_change_count[changed] += 1.0
            previous_art_state[observed] = current_state[observed]
            has_previous_art[observed] = True

        art_regimen_changed = (
            art_regimen_change_count > 0
        ).astype(np.float32)

        history_missing_ratio = (
            1.0 - observed_count / float(history_end)
        ).astype(np.float32)
        months_since_last_observed = (
            landmark_month
            - last_observed_index.astype(np.float32)
            * INTERVAL_WIDTH_MONTH
        ).astype(np.float32)

        landmark_summary = np.concatenate(
            [
                age_at_landmark[:, None],
                bmi_baseline[:, None],
                opp_baseline[:, None],
                baseline_onehot,
                lab_baseline.astype(np.float32),
                lab_current.astype(np.float32),
                lab_change.astype(np.float32),
                lab_mean.astype(np.float32),
                lab_sd.astype(np.float32),
                lab_slope.astype(np.float32),
                persistent_current,
                months_since_first_positive,
                metabolic_current,
                metabolic_months_since_first_positive,
                art_cumulative_current,
                current_art_state,
                observed_count[:, None],
                history_missing_ratio[:, None],
                months_since_last_observed[:, None],
                art_regimen_changed[:, None],
                art_regimen_change_count[:, None],
            ],
            axis=1,
        ).astype(np.float32)

        if landmark_summary.shape != (
            n_patients,
            EXPECTED_SUMMARY_FEATURE_N,
        ):
            raise ValueError(
                f"Landmark {landmark_month}个月汇总形状为"
                f"{landmark_summary.shape}，"
                f"应为({n_patients}, {EXPECTED_SUMMARY_FEATURE_N})。"
            )

        if not np.isfinite(landmark_summary).all():
            raise ValueError(
                f"Landmark {landmark_month}个月汇总特征"
                "存在NaN或无穷值。"
            )

        summary_values[:, landmark_index, :] = landmark_summary

    return summary_values


# ============================================================
# 14. 定义结果核查函数
# ============================================================


def summarize_output(
    dataset_name,
    summary_values,
    sequence_mask,
    eligible_mask,
    origin_mask,
):
    rows = []

    for landmark_index, landmark_month in enumerate(landmark_months):
        history_end = int(landmark_bins[landmark_index]) + 1
        observed_count = sequence_mask[:, :history_end].sum(axis=1)
        valid_mask = origin_mask[:, landmark_index]
        valid_values = summary_values[
            valid_mask,
            landmark_index,
            :,
        ]

        if valid_values.shape[0] == 0:
            raise ValueError(
                f"{dataset_name}在Landmark {landmark_month}个月"
                "没有有效预测起点。"
            )

        constant_feature_n = int(np.sum(
            np.ptp(
                valid_values.astype(np.float64),
                axis=0,
            ) <= 1e-12
        ))

        rows.append({
            "dataset": dataset_name,
            "landmark_month": int(landmark_month),
            "patient_n": int(summary_values.shape[0]),
            "eligible_n": int(
                eligible_mask[:, landmark_index].sum()
            ),
            "valid_origin_n": int(valid_mask.sum()),
            "median_observed_step_n": float(
                np.median(observed_count)
            ),
            "min_observed_step_n": int(observed_count.min()),
            "max_observed_step_n": int(observed_count.max()),
            "constant_feature_n_in_valid_origins": constant_feature_n,
        })

    return rows


# ============================================================
# 15. 分别构建全部开发集和锁定测试集汇总特征
# ============================================================


def select_raw_tensors(patient_indices):
    return (
        np.asarray(
            continuous_raw[patient_indices],
            dtype=np.float32,
        ),
        np.asarray(
            binary_raw[patient_indices],
            dtype=np.float32,
        ),
        np.asarray(
            categorical_raw[patient_indices],
        ).astype(str),
        sequence_row_mask_step3[patient_indices],
    )


(
    continuous_development,
    binary_development,
    categorical_development,
    mask_development,
) = select_raw_tensors(development_idx)

(
    continuous_test,
    binary_test,
    categorical_test,
    mask_test,
) = select_raw_tensors(test_idx)

encoding_audit_rows = [
    audit_sequence_encoding(
        continuous_development,
        binary_development,
        categorical_development,
        mask_development,
        "development_full",
    ),
    audit_sequence_encoding(
        continuous_test,
        binary_test,
        categorical_test,
        mask_test,
        "test_full",
    ),
]

X_summary_development = build_landmark_summary(
    continuous_development,
    binary_development,
    categorical_development,
    mask_development,
)

X_summary_test = build_landmark_summary(
    continuous_test,
    binary_test,
    categorical_test,
    mask_test,
)

expected_development_shape = (
    EXPECTED_DEVELOPMENT_N,
    EXPECTED_LANDMARK_N,
    EXPECTED_SUMMARY_FEATURE_N,
)
expected_test_shape = (
    EXPECTED_TEST_N,
    EXPECTED_LANDMARK_N,
    EXPECTED_SUMMARY_FEATURE_N,
)

if X_summary_development.shape != expected_development_shape:
    raise ValueError(
        f"开发集汇总张量形状为{X_summary_development.shape}，"
        f"应为{expected_development_shape}。"
    )

if X_summary_test.shape != expected_test_shape:
    raise ValueError(
        f"测试集汇总张量形状为{X_summary_test.shape}，"
        f"应为{expected_test_shape}。"
    )

# 特征名称与矩阵列必须保持严格一一对应。
# 这一核查专门防止状态列和时长列名称错位。
def audit_summary_column_semantics(
    summary_values,
    origin_mask,
    dataset_name,
):
    name_to_index = {
        name: index
        for index, name in enumerate(summary_feature_names)
    }
    valid_values = summary_values[origin_mask]

    if valid_values.shape[0] == 0:
        raise ValueError(f"{dataset_name}没有有效Landmark记录。")

    rows = []
    status_names = (
        [f"{name}_at_landmark" for name in PERSISTENT_STATUS_NAMES]
        + [f"{name}_at_landmark" for name in METABOLIC_MED_NAMES]
    )

    for feature_name in status_names:
        column = valid_values[:, name_to_index[feature_name]]
        invalid_mask = ~(
            np.isclose(column, 0.0, atol=1e-7)
            | np.isclose(column, 1.0, atol=1e-7)
        )
        rows.append({
            "dataset": dataset_name,
            "feature_name": feature_name,
            "expected_type": "binary_0_1",
            "minimum": float(np.min(column)),
            "maximum": float(np.max(column)),
            "invalid_n": int(np.sum(invalid_mask)),
        })
        if np.any(invalid_mask):
            examples = np.unique(column[invalid_mask])[:20].tolist()
            raise ValueError(
                f"{dataset_name}的{feature_name}不是0/1，"
                f"异常值示例：{examples}。"
                "这表示特征名称与矩阵列顺序仍不一致。"
            )

    duration_pairs = []
    for name in PERSISTENT_STATUS_NAMES:
        duration_pairs.append((
            f"{name}_at_landmark",
            f"{name}_months_since_first_positive",
        ))
    for name in METABOLIC_MED_NAMES:
        duration_pairs.append((
            f"{name}_at_landmark",
            f"{name}_months_since_first_positive",
        ))

    for status_name, duration_name in duration_pairs:
        status = valid_values[:, name_to_index[status_name]]
        duration = valid_values[:, name_to_index[duration_name]]
        invalid_mask = (
            (duration < -1e-7)
            | ~np.isclose(
                duration / INTERVAL_WIDTH_MONTH,
                np.round(duration / INTERVAL_WIDTH_MONTH),
                atol=1e-6,
            )
            | (
                (status < 0.5)
                & ~np.isclose(duration, 0.0, atol=1e-7)
            )
        )
        rows.append({
            "dataset": dataset_name,
            "feature_name": duration_name,
            "expected_type": "nonnegative_6_month_grid",
            "minimum": float(np.min(duration)),
            "maximum": float(np.max(duration)),
            "invalid_n": int(np.sum(invalid_mask)),
        })
        if np.any(invalid_mask):
            examples = np.unique(duration[invalid_mask])[:20].tolist()
            raise ValueError(
                f"{dataset_name}的{duration_name}语义核查失败，"
                f"异常值示例：{examples}。"
            )

    return rows


summary_semantic_audit_rows = []
summary_semantic_audit_rows.extend(
    audit_summary_column_semantics(
        X_summary_development,
        prediction_origin_mask_all[development_idx],
        "development_full",
    )
)
summary_semantic_audit_rows.extend(
    audit_summary_column_semantics(
        X_summary_test,
        prediction_origin_mask_all[test_idx],
        "test_full",
    )
)
summary_semantic_audit_df = pd.DataFrame(summary_semantic_audit_rows)

np.save(
    OUTPUT_DIR / "X_summary_development_raw.npy",
    X_summary_development,
)
np.save(
    OUTPUT_DIR / "X_summary_test_raw.npy",
    X_summary_test,
)

# 保存兼容文件名，便于后续代码统一读取。
np.save(
    OUTPUT_DIR / "X_summary_development_full.npy",
    X_summary_development,
)
np.save(
    OUTPUT_DIR / "X_summary_test_full.npy",
    X_summary_test,
)

np.save(
    OUTPUT_DIR / "sequence_row_mask_development.npy",
    mask_development,
)
np.save(
    OUTPUT_DIR / "sequence_row_mask_test.npy",
    mask_test,
)
np.save(
    OUTPUT_DIR / "landmark_eligible_development.npy",
    landmark_eligible_mask_all[development_idx],
)
np.save(
    OUTPUT_DIR / "landmark_eligible_test.npy",
    landmark_eligible_mask_all[test_idx],
)
np.save(
    OUTPUT_DIR / "prediction_origin_development.npy",
    prediction_origin_mask_all[development_idx],
)
np.save(
    OUTPUT_DIR / "prediction_origin_test.npy",
    prediction_origin_mask_all[test_idx],
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
    OUTPUT_DIR / "development_fold_id.npy",
    fold_id_all[development_idx],
)
np.save(
    OUTPUT_DIR / "landmark_months.npy",
    landmark_months,
)
np.save(
    OUTPUT_DIR / "landmark_bins.npy",
    landmark_bins,
)


# ============================================================
# 16. 保存五折患者局部索引
#
# 代码逻辑：
# 汇总特征保持原始单位，五折共用同一份张量；
# 后续每折只用训练患者拟合标准化和模型。
# ============================================================

fold_rows = []
development_fold_id = fold_id_all[development_idx]

for fold_id in range(N_SPLITS):
    fold_dir = OUTPUT_DIR / f"fold_{fold_id}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    validation_local_idx = np.flatnonzero(
        development_fold_id == fold_id
    ).astype(np.int32)
    train_local_idx = np.flatnonzero(
        development_fold_id != fold_id
    ).astype(np.int32)

    if len(np.intersect1d(
        train_local_idx,
        validation_local_idx,
    )) > 0:
        raise ValueError(
            f"第{fold_id}折训练和验证局部索引存在交叉。"
        )

    if len(train_local_idx) + len(validation_local_idx) != (
        EXPECTED_DEVELOPMENT_N
    ):
        raise ValueError(
            f"第{fold_id}折患者数不完整。"
        )

    np.save(
        fold_dir / "train_local_idx.npy",
        train_local_idx,
    )
    np.save(
        fold_dir / "validation_local_idx.npy",
        validation_local_idx,
    )
    np.save(
        fold_dir / "train_global_idx.npy",
        development_idx[train_local_idx],
    )
    np.save(
        fold_dir / "validation_global_idx.npy",
        development_idx[validation_local_idx],
    )

    fold_rows.append({
        "fold_id": fold_id,
        "train_patient_n": int(len(train_local_idx)),
        "validation_patient_n": int(len(validation_local_idx)),
        "summary_feature_n": EXPECTED_SUMMARY_FEATURE_N,
        "shared_raw_summary_tensor": True,
    })


# ============================================================
# 17. 保存特征字典、变量角色和核查结果
# ============================================================

summary_feature_table = pd.DataFrame({
    "summary_feature_index": np.arange(
        EXPECTED_SUMMARY_FEATURE_N,
        dtype=np.int32,
    ),
    "summary_feature_name": summary_feature_names,
    "feature_role": summary_feature_roles,
    "source_variable": summary_source_variables,
})
summary_feature_table.to_csv(
    OUTPUT_DIR / "summary_feature_names.csv",
    index=False,
    encoding="utf-8-sig",
)

summary_semantic_audit_df.to_csv(
    OUTPUT_DIR / "summary_column_semantic_audit.csv",
    index=False,
    encoding="utf-8-sig",
)

source_role_rows = []

for name in [BMI_NAME] + STATIC_BINARY_NAMES + STATIC_CATEGORICAL_NAMES:
    source_role_rows.append({
        "source_variable": name,
        "model_role": "baseline_static",
        "step5_processing": "first_observed_value_only",
    })

source_role_rows.append({
    "source_variable": AGE_NAME,
    "model_role": "deterministic_time_updated",
    "step5_processing": "baseline_age_plus_landmark_month_div_12",
})

for name in LONGITUDINAL_LAB_NAMES:
    source_role_rows.append({
        "source_variable": name,
        "model_role": "longitudinal_laboratory",
        "step5_processing": (
            "baseline_current_change_mean_sd_slope"
        ),
    })

for name in PERSISTENT_STATUS_NAMES:
    source_role_rows.append({
        "source_variable": name,
        "model_role": "persistent_time_updated_status",
        "step5_processing": (
            "current_and_months_since_first_positive"
        ),
    })

for name in METABOLIC_MED_NAMES:
    source_role_rows.append({
        "source_variable": name,
        "model_role": "reversible_time_updated_medication",
        "step5_processing": "current_ever_change_count",
    })

for name in CURRENT_ART_NAMES:
    source_role_rows.append({
        "source_variable": name,
        "model_role": "current_art_regimen",
        "step5_processing": "latest_observed_state",
    })

for name in ART_CUMULATIVE_NAMES:
    source_role_rows.append({
        "source_variable": name,
        "model_role": "cumulative_art_exposure",
        "step5_processing": "latest_observed_cumulative_month",
    })

source_role_table = pd.DataFrame(source_role_rows)
source_role_table.to_csv(
    OUTPUT_DIR / "source_variable_roles.csv",
    index=False,
    encoding="utf-8-sig",
)

encoding_audit_table = pd.DataFrame(encoding_audit_rows)
encoding_audit_table.to_csv(
    OUTPUT_DIR / "time_encoding_audit.csv",
    index=False,
    encoding="utf-8-sig",
)

landmark_audit_rows = []
landmark_audit_rows.extend(
    summarize_output(
        "development_full",
        X_summary_development,
        mask_development,
        landmark_eligible_mask_all[development_idx],
        prediction_origin_mask_all[development_idx],
    )
)
landmark_audit_rows.extend(
    summarize_output(
        "test_full",
        X_summary_test,
        mask_test,
        landmark_eligible_mask_all[test_idx],
        prediction_origin_mask_all[test_idx],
    )
)

landmark_audit_table = pd.DataFrame(landmark_audit_rows)
landmark_audit_table.to_csv(
    OUTPUT_DIR / "landmark_summary_audit.csv",
    index=False,
    encoding="utf-8-sig",
)

fold_table = pd.DataFrame(fold_rows)
fold_table.to_csv(
    OUTPUT_DIR / "fold_summary.csv",
    index=False,
    encoding="utf-8-sig",
)

manifest = {
    "step": 5,
    "summary_tensor_scale": "raw_clinical_units",
    "summary_feature_n": EXPECTED_SUMMARY_FEATURE_N,
    "development_shape": list(X_summary_development.shape),
    "test_shape": list(X_summary_test.shape),
    "landmark_months": landmark_months.tolist(),
    "age_processing": "baseline_age_plus_landmark_month_div_12",
    "BMI_processing": "baseline_only",
    "laboratory_summary": [
        "baseline",
        "current",
        "change_from_baseline",
        "history_mean",
        "history_sd",
        "slope_per_year",
    ],
    "persistent_status_summary": [
        "at_landmark",
        "months_since_first_positive",
    ],
    "metabolic_med_summary": [
        "at_landmark",
        "months_since_first_positive",
    ],
    "art_summary": [
        "cumulative_month_at_landmark",
        "current_regimen",
        "regimen_changed",
        "regimen_change_count",
    ],
    "model_preprocessing_note": (
        "Standardization and constant-feature removal must be fitted "
        "inside each training fold in the next modeling step."
    ),
}

with open(
    OUTPUT_DIR / "step5_manifest.json",
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
# 18. 输出最终核查结果
# ============================================================

print("=" * 40)
print("Step 5完成：Landmark原始历史汇总特征已生成（已校正名称与矩阵列顺序）")
print("=" * 40)
print()
print("全部开发集汇总张量：", X_summary_development.shape)
print("锁定测试集汇总张量：", X_summary_test.shape)
print("汇总特征数：", EXPECTED_SUMMARY_FEATURE_N)
print()
print("汇总特征组成：")
print("Age当前值：1")
print("BMI及基线人口学信息：16")
print("15项实验室历史特征：90")
print("6项持续性疾病状态及首次阳性时长：12")
print("3项代谢用药状态及首次用药时长：6")
print("ART累计暴露：8")
print("当前ART方案：8")
print("历史完整度及ART方案变更：5")
print()
print("时间编码核查：")
print(encoding_audit_table.to_string(index=False))
print()
print("开发集和测试集Landmark核查：")
print(landmark_audit_table.to_string(index=False))
print()
print("五折患者索引核查：")
print(fold_table.to_string(index=False))
print()
print("处理原则：")
print("1. Age = 基线Age + Landmark月数/12，不生成Age历史均值或斜率。")
print("2. BMI及基线人口学变量只取第一次真实时间行。")
print("3. 实验室历史统计只使用Landmark及以前真实时间行。")
print("4. 持续性状态不生成ever，改为当前状态和距首次阳性月数。")
print("5. 代谢用药首次变为1后持续为1，保留Landmark状态和距首次用药月数。")
print("6. Oppinfection和Course后续记录变化只作审计，正式输入固定取基线值。")
print("7. 汇总特征保持原始单位，后续仅在训练折内标准化。")
print()
print("输出目录：", OUTPUT_DIR)
print("Step 5运行完成。")
