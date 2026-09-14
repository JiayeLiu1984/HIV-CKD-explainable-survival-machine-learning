# ============================================================
# Step 4：按固定五折完成无泄漏特征预处理
#
# 代码逻辑：
# 1. 读取Step 3生成的未标准化原始特征张量。
# 2. 每一折只使用该折训练患者拟合连续变量标准化和One-Hot编码。
# 3. 使用训练折参数转换对应的全部开发患者，供该折训练和验证使用。
# 4. 使用全部开发患者拟合最终预处理器，并分别转换开发集和锁定测试集。
# 5. 空缺半年时间行统一保持为全0，并继续使用sequence_row_mask识别。
# 6. 本步骤不训练任何模型，也不读取测试集结局进行参数选择。
# ============================================================

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler


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
    "__CKD_WORKDIR__/rolling_5y_step4_preprocessed"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

N_SPLITS = 5
EXPECTED_TOTAL_N = 31911
EXPECTED_DEVELOPMENT_N = 22337
EXPECTED_TEST_N = 9574
EXPECTED_HISTORY_STEP_N = 11
EXPECTED_CONTINUOUS_N = 25
EXPECTED_BINARY_N = 18
EXPECTED_CATEGORICAL_N = 4
EXPECTED_ONEHOT_N = 14
EXPECTED_FINAL_FEATURE_N = 57
PADDING_CATEGORY = "__NO_TIME_ROW__"


# ============================================================
# 2. 检查Step 1～3所需文件
#
# 代码逻辑：
# 在读取前一次性核对文件名，避免运行到中途才发现路径错误。
# ============================================================

required_files = [
    STEP1_DIR / "development_idx.npy",
    STEP1_DIR / "test_idx.npy",
    STEP2_DIR / "fold_id_all.npy",
    STEP3_DIR / "continuous_raw_0_60.npy",
    STEP3_DIR / "binary_raw_0_60.npy",
    STEP3_DIR / "categorical_raw_0_60.npy",
    STEP3_DIR / "sequence_row_mask.npy",
    STEP3_DIR / "patient_info_with_fold.csv",
    STEP3_DIR / "feature_groups.json",
]

missing_files = [str(path) for path in required_files if not path.exists()]
if missing_files:
    raise FileNotFoundError(
        "以下输入文件不存在：\n" + "\n".join(missing_files)
    )


# ============================================================
# 3. 读取患者索引、五折编号和特征定义
# ============================================================

development_idx = np.load(
    STEP1_DIR / "development_idx.npy"
).astype(np.int32)

test_idx = np.load(
    STEP1_DIR / "test_idx.npy"
).astype(np.int32)

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

continuous_vars = list(feature_groups["continuous_vars"])
binary_vars = list(feature_groups["binary_vars"])
categorical_vars = list(feature_groups["categorical_vars"])


# ============================================================
# 4. 读取Step 3原始张量
#
# 代码逻辑：
# 使用内存映射读取，避免一次复制全部原始张量。
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

sequence_row_mask = np.load(
    STEP3_DIR / "sequence_row_mask.npy"
).astype(bool)


# ============================================================
# 5. 核对患者顺序、数据划分和原始张量形状
# ============================================================

n_patients = len(patient_info)

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
}

actual_shapes = {
    "continuous_raw": continuous_raw.shape,
    "binary_raw": binary_raw.shape,
    "categorical_raw": categorical_raw.shape,
    "sequence_row_mask": sequence_row_mask.shape,
}

for name, expected_shape in expected_shapes.items():
    if actual_shapes[name] != expected_shape:
        raise ValueError(
            f"{name}形状为{actual_shapes[name]}，"
            f"应为{expected_shape}。"
        )

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
        f"测试集人数为{len(test_idx)}，"
        f"应为{EXPECTED_TEST_N}。"
    )

if len(continuous_vars) != EXPECTED_CONTINUOUS_N:
    raise ValueError("连续变量数不是25。")

if len(binary_vars) != EXPECTED_BINARY_N:
    raise ValueError("二分类变量数不是18。")

if len(categorical_vars) != EXPECTED_CATEGORICAL_N:
    raise ValueError("多分类变量数不是4。")

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

if np.intersect1d(development_idx, test_idx).size > 0:
    raise ValueError("开发集和测试集患者存在交叉。")

if not patient_info.loc[
    development_idx, "analysis_split"
].eq("development").all():
    raise ValueError("development_idx中包含非开发集患者。")

if not patient_info.loc[
    test_idx, "analysis_split"
].eq("test").all():
    raise ValueError("test_idx中包含非测试集患者。")


# ============================================================
# 6. 核对真实时间行和空缺时间行
# ============================================================

if not np.isfinite(
    continuous_raw[sequence_row_mask]
).all():
    raise ValueError("真实时间行连续变量存在NaN或无穷值。")

if not np.isfinite(
    binary_raw[sequence_row_mask]
).all():
    raise ValueError("真实时间行二分类变量存在NaN或无穷值。")

if not np.isin(
    binary_raw[sequence_row_mask],
    [0.0, 1.0],
).all():
    raise ValueError("真实时间行二分类变量存在非0/1值。")

if np.any(
    categorical_raw[sequence_row_mask] == PADDING_CATEGORY
):
    raise ValueError("真实时间行多分类变量出现空行填充标记。")

if not np.isnan(
    continuous_raw[~sequence_row_mask]
).all():
    raise ValueError("空缺时间行连续变量不是NaN。")

if not np.isnan(
    binary_raw[~sequence_row_mask]
).all():
    raise ValueError("空缺时间行二分类变量不是NaN。")

if not np.all(
    categorical_raw[~sequence_row_mask] == PADDING_CATEGORY
):
    raise ValueError("空缺时间行多分类变量填充不一致。")


# ============================================================
# 7. 定义One-Hot编码器
#
# 代码逻辑：
# 兼容不同scikit-learn版本；未知类别编码为全0。
# ============================================================

def make_onehot_encoder():
    try:
        return OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False,
            dtype=np.float32,
        )
    except TypeError:
        return OneHotEncoder(
            handle_unknown="ignore",
            sparse=False,
            dtype=np.float32,
        )


# ============================================================
# 8. 定义预处理器拟合函数
#
# 代码逻辑：
# 只提取指定训练患者的真实时间行；
# 连续变量用训练行拟合StandardScaler；
# 多分类变量用训练行拟合OneHotEncoder；
# 二分类变量保持0/1不变。
# ============================================================

def fit_preprocessor(train_patient_idx):
    train_patient_idx = np.asarray(
        train_patient_idx,
        dtype=np.int32,
    )

    train_row_mask = sequence_row_mask[train_patient_idx]

    train_continuous = np.asarray(
        continuous_raw[train_patient_idx]
    )[train_row_mask]

    train_categorical = np.asarray(
        categorical_raw[train_patient_idx]
    )[train_row_mask]

    if train_continuous.shape[0] == 0:
        raise ValueError("训练患者没有可用时间行。")

    if train_categorical.shape[0] != train_continuous.shape[0]:
        raise ValueError("连续变量与多分类变量训练行数不一致。")

    scaler = StandardScaler()
    scaler.fit(train_continuous)

    encoder = make_onehot_encoder()
    encoder.fit(train_categorical)

    onehot_feature_names = encoder.get_feature_names_out(
        categorical_vars
    ).tolist()

    final_feature_names = (
        continuous_vars
        + binary_vars
        + onehot_feature_names
    )

    return {
        "scaler": scaler,
        "encoder": encoder,
        "onehot_feature_names": onehot_feature_names,
        "final_feature_names": final_feature_names,
        "train_observed_row_n": int(train_row_mask.sum()),
    }


# ============================================================
# 9. 定义张量转换函数
#
# 代码逻辑：
# 只转换指定患者的真实时间行；
# 特征顺序固定为连续变量、二分类变量、One-Hot变量；
# 空缺半年时间行保持为全0。
# ============================================================

def transform_patients(patient_idx, preprocessor):
    patient_idx = np.asarray(
        patient_idx,
        dtype=np.int32,
    )

    patient_row_mask = sequence_row_mask[patient_idx]
    n_subset = len(patient_idx)
    n_final_features = len(
        preprocessor["final_feature_names"]
    )

    output = np.zeros(
        (
            n_subset,
            EXPECTED_HISTORY_STEP_N,
            n_final_features,
        ),
        dtype=np.float32,
    )

    subset_continuous = np.asarray(
        continuous_raw[patient_idx]
    )
    subset_binary = np.asarray(
        binary_raw[patient_idx]
    )
    subset_categorical = np.asarray(
        categorical_raw[patient_idx]
    )

    continuous_observed = subset_continuous[
        patient_row_mask
    ]
    binary_observed = subset_binary[
        patient_row_mask
    ].astype(np.float32)
    categorical_observed = subset_categorical[
        patient_row_mask
    ]

    continuous_scaled = preprocessor["scaler"].transform(
        continuous_observed
    ).astype(np.float32)

    categorical_onehot = preprocessor["encoder"].transform(
        categorical_observed
    ).astype(np.float32)

    observed_features = np.concatenate(
        [
            continuous_scaled,
            binary_observed,
            categorical_onehot,
        ],
        axis=1,
    ).astype(np.float32)

    if observed_features.shape[1] != n_final_features:
        raise ValueError(
            f"转换后特征数为{observed_features.shape[1]}，"
            f"应为{n_final_features}。"
        )

    output[patient_row_mask] = observed_features

    if not np.isfinite(output).all():
        raise ValueError("转换后特征张量存在NaN或无穷值。")

    if not np.all(output[~patient_row_mask] == 0.0):
        raise ValueError("空缺时间行转换后不是全0。")

    return output


# ============================================================
# 10. 定义One-Hot分组核查函数
#
# 代码逻辑：
# 每个真实时间行在每个多分类变量对应的One-Hot组内应恰好为1。
# ============================================================

def check_onehot_groups(
    transformed_tensor,
    patient_row_mask,
    encoder,
):
    onehot_start = EXPECTED_CONTINUOUS_N + EXPECTED_BINARY_N
    current_start = onehot_start

    for variable, categories in zip(
        categorical_vars,
        encoder.categories_,
    ):
        current_end = current_start + len(categories)

        group_sum = transformed_tensor[
            :,
            :,
            current_start:current_end,
        ].sum(axis=2)

        if not np.allclose(
            group_sum[patient_row_mask],
            1.0,
            atol=1e-6,
        ):
            raise ValueError(
                f"{variable}的真实时间行One-Hot组和不等于1。"
            )

        if not np.allclose(
            group_sum[~patient_row_mask],
            0.0,
            atol=1e-6,
        ):
            raise ValueError(
                f"{variable}的空缺时间行One-Hot组和不等于0。"
            )

        current_start = current_end

    if current_start != transformed_tensor.shape[2]:
        raise ValueError("One-Hot分组边界与最终特征数不一致。")


# ============================================================
# 11. 使用全部开发集拟合最终预处理器
#
# 代码逻辑：
# 此预处理器用于开发集全量重拟合和锁定测试集转换；
# 测试集不参与任何参数估计。
# ============================================================

full_preprocessor = fit_preprocessor(development_idx)
full_feature_names = full_preprocessor["final_feature_names"]
full_onehot_n = len(full_preprocessor["onehot_feature_names"])

if full_onehot_n != EXPECTED_ONEHOT_N:
    raise ValueError(
        f"全部开发集One-Hot变量数为{full_onehot_n}，"
        f"应为{EXPECTED_ONEHOT_N}。"
    )

if len(full_feature_names) != EXPECTED_FINAL_FEATURE_N:
    raise ValueError(
        f"全部开发集最终特征数为{len(full_feature_names)}，"
        f"应为{EXPECTED_FINAL_FEATURE_N}。"
    )

X_development_full = transform_patients(
    development_idx,
    full_preprocessor,
)

X_test_full = transform_patients(
    test_idx,
    full_preprocessor,
)

check_onehot_groups(
    X_development_full,
    sequence_row_mask[development_idx],
    full_preprocessor["encoder"],
)

check_onehot_groups(
    X_test_full,
    sequence_row_mask[test_idx],
    full_preprocessor["encoder"],
)

np.save(
    OUTPUT_DIR / "X_development_full_preprocessor.npy",
    X_development_full,
)

np.save(
    OUTPUT_DIR / "X_test_full_preprocessor.npy",
    X_test_full,
)

joblib.dump(
    full_preprocessor,
    OUTPUT_DIR / "full_development_preprocessor.joblib",
)

pd.DataFrame({
    "feature_index": np.arange(
        len(full_feature_names),
        dtype=np.int32,
    ),
    "feature_name": full_feature_names,
}).to_csv(
    OUTPUT_DIR / "final_feature_names.csv",
    index=False,
    encoding="utf-8-sig",
)


# ============================================================
# 12. 逐折拟合预处理器并转换开发集
#
# 代码逻辑：
# 第k折预处理参数只来自fold_id不等于k的开发患者；
# 验证折患者从不参与该折的标准化和One-Hot拟合。
# ============================================================

fold_summary_rows = []
category_summary_rows = []

development_fold_id = fold_id_all[development_idx]

for fold_id in range(N_SPLITS):
    fold_dir = OUTPUT_DIR / f"fold_{fold_id}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    train_global_idx = development_idx[
        development_fold_id != fold_id
    ]
    validation_global_idx = development_idx[
        development_fold_id == fold_id
    ]

    if np.intersect1d(
        train_global_idx,
        validation_global_idx,
    ).size > 0:
        raise ValueError(
            f"第{fold_id}折训练和验证患者存在交叉。"
        )

    fold_preprocessor = fit_preprocessor(train_global_idx)
    fold_feature_names = fold_preprocessor[
        "final_feature_names"
    ]

    if fold_feature_names != full_feature_names:
        raise ValueError(
            f"第{fold_id}折的特征名称或顺序与全部开发集不一致。\n"
            f"第{fold_id}折：{fold_feature_names}\n"
            f"全部开发集：{full_feature_names}"
        )

    X_development_fold = transform_patients(
        development_idx,
        fold_preprocessor,
    )

    check_onehot_groups(
        X_development_fold,
        sequence_row_mask[development_idx],
        fold_preprocessor["encoder"],
    )

    train_local_mask = development_fold_id != fold_id
    validation_local_mask = development_fold_id == fold_id

    train_observed_mask = sequence_row_mask[
        development_idx[train_local_mask]
    ]

    train_continuous_transformed = X_development_fold[
        train_local_mask,
        :,
        :EXPECTED_CONTINUOUS_N,
    ][train_observed_mask]

    # 使用float64累计核查均值和标准差，避免float32大样本求和误差
    transformed_mean = train_continuous_transformed.mean(
        axis=0,
        dtype=np.float64,
    )
    transformed_std = train_continuous_transformed.std(
        axis=0,
        dtype=np.float64,
    )

    nonconstant_mask = (
        np.asarray(fold_preprocessor["scaler"].var_)
        > 0
    )

    max_abs_mean = float(
        np.max(np.abs(transformed_mean))
    )

    if nonconstant_mask.any():
        max_abs_sd_difference = float(
            np.max(
                np.abs(
                    transformed_std[nonconstant_mask]
                    - 1.0
                )
            )
        )
    else:
        max_abs_sd_difference = 0.0

    if max_abs_mean > 1e-5:
        raise ValueError(
            f"第{fold_id}折训练连续变量标准化后均值异常："
            f"{max_abs_mean}"
        )

    if max_abs_sd_difference > 1e-5:
        raise ValueError(
            f"第{fold_id}折训练连续变量标准化后标准差异常："
            f"{max_abs_sd_difference}"
        )

    np.save(
        fold_dir / "X_development.npy",
        X_development_fold,
    )

    np.save(
        fold_dir / "train_global_idx.npy",
        train_global_idx.astype(np.int32),
    )

    np.save(
        fold_dir / "validation_global_idx.npy",
        validation_global_idx.astype(np.int32),
    )

    joblib.dump(
        fold_preprocessor,
        fold_dir / "preprocessor.joblib",
    )

    pd.DataFrame({
        "feature_index": np.arange(
            len(fold_feature_names),
            dtype=np.int32,
        ),
        "feature_name": fold_feature_names,
    }).to_csv(
        fold_dir / "feature_names.csv",
        index=False,
        encoding="utf-8-sig",
    )

    fold_summary_rows.append({
        "fold_id": fold_id,
        "train_patient_n": int(len(train_global_idx)),
        "validation_patient_n": int(len(validation_global_idx)),
        "train_event_n": int(
            patient_info.loc[train_global_idx, "event"].sum()
        ),
        "validation_event_n": int(
            patient_info.loc[validation_global_idx, "event"].sum()
        ),
        "train_observed_time_row_n": int(
            sequence_row_mask[train_global_idx].sum()
        ),
        "validation_observed_time_row_n": int(
            sequence_row_mask[validation_global_idx].sum()
        ),
        "onehot_n": int(
            len(fold_preprocessor["onehot_feature_names"])
        ),
        "final_feature_n": int(len(fold_feature_names)),
        "max_abs_train_scaled_mean": max_abs_mean,
        "max_abs_train_scaled_sd_difference": (
            max_abs_sd_difference
        ),
    })

    for variable, categories in zip(
        categorical_vars,
        fold_preprocessor["encoder"].categories_,
    ):
        for category in categories.tolist():
            category_summary_rows.append({
                "preprocessor": f"fold_{fold_id}",
                "variable": variable,
                "category": str(category),
            })

    del X_development_fold
    del train_continuous_transformed


# ============================================================
# 13. 保存公共索引、掩码和核查结果
# ============================================================

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
    development_fold_id.astype(np.int8),
)

np.save(
    OUTPUT_DIR / "sequence_row_mask_development.npy",
    sequence_row_mask[development_idx],
)

np.save(
    OUTPUT_DIR / "sequence_row_mask_test.npy",
    sequence_row_mask[test_idx],
)

patient_info.to_csv(
    OUTPUT_DIR / "patient_info_with_fold.csv",
    index=False,
    encoding="utf-8-sig",
)

fold_preprocessing_summary = pd.DataFrame(
    fold_summary_rows
)

fold_preprocessing_summary.to_csv(
    OUTPUT_DIR / "fold_preprocessing_summary.csv",
    index=False,
    encoding="utf-8-sig",
)

category_summary = pd.DataFrame(
    category_summary_rows
)

category_summary.to_csv(
    OUTPUT_DIR / "fold_category_summary.csv",
    index=False,
    encoding="utf-8-sig",
)

full_category_rows = []
for variable, categories in zip(
    categorical_vars,
    full_preprocessor["encoder"].categories_,
):
    for category in categories.tolist():
        full_category_rows.append({
            "preprocessor": "full_development",
            "variable": variable,
            "category": str(category),
        })

pd.DataFrame(full_category_rows).to_csv(
    OUTPUT_DIR / "full_development_category_summary.csv",
    index=False,
    encoding="utf-8-sig",
)


# ============================================================
# 14. 保存Step 4说明文件
# ============================================================

manifest = {
    "step1_dir": str(STEP1_DIR),
    "step2_dir": str(STEP2_DIR),
    "step3_dir": str(STEP3_DIR),
    "output_dir": str(OUTPUT_DIR),
    "n_patients": int(n_patients),
    "development_n": int(len(development_idx)),
    "test_n": int(len(test_idx)),
    "n_splits": N_SPLITS,
    "history_step_n": EXPECTED_HISTORY_STEP_N,
    "continuous_n": EXPECTED_CONTINUOUS_N,
    "binary_n": EXPECTED_BINARY_N,
    "categorical_n": EXPECTED_CATEGORICAL_N,
    "onehot_n": EXPECTED_ONEHOT_N,
    "final_feature_n": EXPECTED_FINAL_FEATURE_N,
    "feature_order": (
        "continuous + binary + one-hot categorical"
    ),
    "continuous_scaling": (
        "StandardScaler fitted on observed rows of training patients only"
    ),
    "binary_processing": "kept as 0/1",
    "categorical_processing": (
        "OneHotEncoder fitted on observed rows of training patients only"
    ),
    "missing_time_row_processing": (
        "all 57 features set to zero; sequence_row_mask retained"
    ),
    "test_processing": (
        "transformed only with full-development preprocessor"
    ),
}

with open(
    OUTPUT_DIR / "step4_manifest.json",
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
# 15. 输出最终核查结果
# ============================================================

print("\n========================================")
print("Step 4完成：五折无泄漏预处理已生成")
print("========================================")

print("\n全部开发集预处理后的开发张量：", X_development_full.shape)
print("全部开发集预处理后的测试张量：", X_test_full.shape)
print("最终特征数：", len(full_feature_names))
print("One-Hot变量数：", full_onehot_n)

print("\n最终特征顺序：")
print(
    pd.DataFrame({
        "feature_index": np.arange(
            len(full_feature_names),
            dtype=np.int32,
        ),
        "feature_name": full_feature_names,
    }).to_string(index=False)
)

print("\n五折预处理核查：")
print(fold_preprocessing_summary.to_string(index=False))

print("\n处理原则：")
print("1. 每折标准化和One-Hot参数只来自该折训练患者。")
print("2. 验证折不参与对应折预处理器拟合。")
print("3. 测试集只使用全部开发集拟合的预处理器。")
print("4. 空缺半年时间行保持为57项全0，并保留时间行掩码。")

print("\n输出目录：", OUTPUT_DIR)
print("Step 4运行完成。")
