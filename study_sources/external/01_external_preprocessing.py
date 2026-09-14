# -*- coding: utf-8 -*-

# =============================================================================
# 重庆外部验证 Step5 FINAL
#
# 新GLU整理完成后重新进行正式预处理
#
# 输入：
#   重庆_6month_panel_step4_GLU_updated_UTF8.csv
#
# 流程：
#   1. 读取最终Step4
#   2. HIVRNA -> HIVRNA_log10
#   3. 15项实验室患者内LOCF（只向前）
#   4. 静态变量患者内补齐
#   5. ART current/cumulative患者内LOCF，剩余填0
#   6. 其余剩余缺失使用开发队列固定中位数/众数
#   7. 得到正式47个源变量
#   8. 使用冻结full-development scaler + encoder
#   9. 得到0~60月、11时间步、57维输入张量
#
# 不重新拟合任何预处理参数
# =============================================================================

from pathlib import Path
import json
import re

import joblib
import numpy as np
import pandas as pd


# =============================================================================
# 可重复性保护：严格阻止跨scikit-learn版本反序列化
#
# 冻结的StandardScaler/OneHotEncoder必须在保存它们的同一scikit-learn
# 版本中加载；否则外部验证立即停止，不能把兼容性警告当作最终结果。
# =============================================================================
import warnings as _sklearn_version_warnings
from sklearn.exceptions import InconsistentVersionWarning

_sklearn_version_warnings.filterwarnings(
    "error",
    category=InconsistentVersionWarning,
)


# =============================================================================
# 1. 路径
# =============================================================================

BASE = Path("__CKD_WORKDIR__")

INPUT_FILE = (
    BASE /
    "重庆_6month_panel_step4_GLU_updated_UTF8.csv"
)

DEV_STEP3 = (
    BASE /
    "rolling_5y_step3_raw_features"
)

DEV_STEP4 = (
    BASE /
    "rolling_5y_step4_preprocessed"
)

OUT = (
    BASE /
    "重庆外部验证_step5_GLU更新"
)

OUT.mkdir(
    parents=True,
    exist_ok=True
)


# =============================================================================
# 2. 开发模型冻结文件
# =============================================================================

FEATURE_GROUP_FILE = (
    DEV_STEP3 /
    "feature_groups.json"
)

PREPROCESSOR_FILE = (
    DEV_STEP4 /
    "full_development_preprocessor.joblib"
)

DEV_CONT_FILE = (
    DEV_STEP3 /
    "continuous_raw_0_60.npy"
)

DEV_BIN_FILE = (
    DEV_STEP3 /
    "binary_raw_0_60.npy"
)

DEV_CAT_FILE = (
    DEV_STEP3 /
    "categorical_raw_0_60.npy"
)

DEV_MASK_FILE = (
    DEV_STEP3 /
    "sequence_row_mask.npy"
)

DEV_IDX_FILE = (
    DEV_STEP4 /
    "development_idx.npy"
)


# =============================================================================
# 3. 正式15项实验室
# =============================================================================

LAB_VARS = [
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


STATIC_VARS = [
    "BMI",
    "Oppinfection",
    "Sex",
    "Marriage",
    "Course",
    "WHOstage",
]


PERSISTENT_VARS = [
    "CVD_status",
    "diabetes_status",
    "hypertension_status",
    "hypercholesterolemia_status",
    "HBV_status",
    "HCV_status",
]


METABOLIC_MED_VARS = [
    "antidiabetic_med",
    "antihypertensive_med",
    "antilipid_med",
]


CURRENT_ART_VARS = [
    "current_TDF_NNRTI_3TC_FTC",
    "current_TDF_PI_3TC_FTC",
    "current_nonTDF_PI",
    "current_BIC_FTC_TAF",
    "current_EVGc_FTC_TAF",
    "current_TDF_INSTI_3TC_FTC",
    "current_nonTDF_DTG",
    "current_nonTDF_traditional_NNRTI",
]


CUM_ART_VARS = [
    "TDF_NNRTI_3TC_FTC_cum_month",
    "TDF_PI_3TC_FTC_cum_month",
    "nonTDF_PI_cum_month",
    "BIC_FTC_TAF_cum_month",
    "EVGc_FTC_TAF_cum_month",
    "TDF_INSTI_3TC_FTC_cum_month",
    "nonTDF_DTG_cum_month",
    "nonTDF_traditional_NNRTI_cum_month",
]


# =============================================================================
# 4. 读取正式变量定义
# =============================================================================

with open(
    FEATURE_GROUP_FILE,
    "r",
    encoding="utf-8"
) as f:
    groups = json.load(f)


continuous_vars = groups["continuous_vars"]
binary_vars = groups["binary_vars"]
categorical_vars = groups["categorical_vars"]


assert len(continuous_vars) == 25
assert len(binary_vars) == 18
assert len(categorical_vars) == 4


SOURCE47 = (
    continuous_vars
    + binary_vars
    + categorical_vars
)

assert len(SOURCE47) == 47


# =============================================================================
# 5. 读取重庆新Step4
# =============================================================================

def read_csv_safe(path):

    for encoding in [
        "utf-8-sig",
        "utf-8",
        "gb18030",
    ]:

        try:
            df = pd.read_csv(
                path,
                encoding=encoding,
                low_memory=False,
                dtype={"ID": "string"}
            )

            # 防止错误编码“成功读取”但列名乱码
            df.columns = [
                str(x)
                .replace("\ufeff", "")
                .strip()
                for x in df.columns
            ]

            if "ID" in df.columns:
                return df, encoding

        except UnicodeDecodeError:
            pass

    raise ValueError(
        "无法正确读取输入CSV。"
    )


df, encoding = read_csv_safe(
    INPUT_FILE
)


# =============================================================================
# 6. ID和时间
# =============================================================================

df["ID"] = (
    df["ID"]
    .astype("string")
    .str.strip()
    .str.replace(
        r"\.0$",
        "",
        regex=True
    )
)


df["time_bin"] = pd.to_numeric(
    df["time_bin"],
    errors="raise"
).astype(int)


df["month"] = pd.to_numeric(
    df["month"],
    errors="raise"
)


df["CKDstatus"] = pd.to_numeric(
    df["CKDstatus"],
    errors="raise"
).astype(int)


df["interval"] = pd.to_numeric(
    df["interval"],
    errors="raise"
)


df = (
    df
    .sort_values(
        ["ID", "time_bin"]
    )
    .reset_index(drop=True)
)


if df.duplicated(
    ["ID", "time_bin"]
).any():
    raise ValueError(
        "存在重复ID + time_bin。"
    )


if not np.allclose(
    df["month"],
    df["time_bin"] * 6
):
    raise ValueError(
        "month != time_bin × 6。"
    )


N = df["ID"].nunique()


# =============================================================================
# 7. 基线检查
# =============================================================================

baseline = df.loc[
    df["time_bin"] == 0
].copy()


if len(baseline) != N:
    raise ValueError(
        "并非每名患者都有唯一month=0。"
    )


baseline_egfr = pd.to_numeric(
    baseline["eGFR"],
    errors="coerce"
)


if baseline_egfr.isna().any():
    raise ValueError(
        f"仍有{baseline_egfr.isna().sum()}名患者基线eGFR缺失。"
    )


# =============================================================================
# 8. Age重新按基线年龄更新
# =============================================================================

age0 = pd.to_numeric(
    baseline["Age"],
    errors="coerce"
)


if age0.isna().any():
    raise ValueError(
        "存在基线Age缺失。"
    )


age_map = dict(
    zip(
        baseline["ID"],
        age0
    )
)


df["Age"] = (
    df["ID"].map(age_map)
    + df["month"] / 12.0
)


# =============================================================================
# 9. HIVRNA
# =============================================================================

number_pattern = re.compile(
    r"[-+]?"
    r"(?:\d+(?:\.\d*)?|\.\d+)"
    r"(?:[Ee][-+]?\d+)?"
)


def get_number(x):

    if pd.isna(x):
        return np.nan

    text = (
        str(x)
        .strip()
        .replace(",", "")
    )

    m = number_pattern.search(text)

    if m is None:
        return np.nan

    return float(m.group())


def get_lod(x):

    if pd.isna(x):
        return np.nan

    text = (
        str(x)
        .replace("＜", "<")
    )

    if "<" not in text:
        return np.nan

    v = get_number(text)

    if np.isfinite(v) and v > 0:
        return v

    return np.nan


explicit_lod = (
    df["HIVRNA"]
    .map(get_lod)
    .dropna()
)


if len(explicit_lod) == 0:
    raise ValueError(
        "HIVRNA中没有明确<LOD记录。"
    )


DEFAULT_LOD = float(
    explicit_lod.mode().iloc[0]
)


def hivrna_log10(x):

    if pd.isna(x):
        return np.nan

    text = (
        str(x)
        .strip()
        .replace("＜", "<")
        .replace(",", "")
    )

    if text == "":
        return np.nan


    v = get_number(text)


    # <LOD
    if (
        "<" in text
        and np.isfinite(v)
        and v > 0
    ):
        return np.log10(v / 2.0)


    compact = re.sub(
        r"\s+",
        "",
        text.upper()
    )


    undetectable = (
        "TARGETNOTDETECTED" in compact
        or "NOTDETECTED" in compact
        or compact == "TND"
        or "未检出" in text
        or "未检测到" in text
    )


    if undetectable:
        return np.log10(
            DEFAULT_LOD / 2.0
        )


    if np.isfinite(v) and v > 0:
        return np.log10(v)


    if v == 0:
        return np.log10(
            DEFAULT_LOD / 2.0
        )


    return np.nan


df["HIVRNA_log10"] = (
    df["HIVRNA"]
    .map(hivrna_log10)
)


# =============================================================================
# 10. 其他连续变量转数值
# =============================================================================

for col in continuous_vars:

    if col == "HIVRNA_log10":
        continue

    df[col] = pd.to_numeric(
        df[col],
        errors="coerce"
    )


# =============================================================================
# 11. 二分类转数值
# =============================================================================

for col in binary_vars:

    df[col] = pd.to_numeric(
        df[col],
        errors="coerce"
    )


# =============================================================================
# 12. 分类变量
# =============================================================================

for col in categorical_vars:

    df[col] = (
        df[col]
        .astype("string")
        .str.strip()
    )


df["WHOstage"] = (
    pd.to_numeric(
        df["WHOstage"],
        errors="coerce"
    )
    .round()
    .astype("Int64")
    .astype("string")
)


# =============================================================================
# 13. 记录实验室填补前状态
#
# 这一步非常重要：
# 后面可以区分 observed / LOCF / development_median
# =============================================================================

lab_before = (
    df[LAB_VARS]
    .copy()
)

lab_observed = (
    lab_before.notna()
)


# =============================================================================
# 14. 15项实验室LOCF
#
# 只向前填
# 不使用bfill
# 不用未来值
# =============================================================================

df[LAB_VARS] = (
    df
    .groupby(
        "ID",
        sort=False
    )[LAB_VARS]
    .ffill()
)


lab_after_locf = (
    df[LAB_VARS]
    .notna()
)


# =============================================================================
# 15. 静态变量患者内补齐
#
# 静态变量本来就应该患者内一致
# ffill + bfill仅用于复制患者自身的基线/静态值
# =============================================================================

df[STATIC_VARS] = (
    df
    .groupby(
        "ID",
        sort=False
    )[STATIC_VARS]
    .transform(
        lambda x:
        x.ffill().bfill()
    )
)


# =============================================================================
# 16. 持续状态 + 代谢药物
#
# 已构建为纵向0/1状态
# 这里只允许向前延续
# =============================================================================

status_vars = (
    PERSISTENT_VARS
    + METABOLIC_MED_VARS
)


df[status_vars] = (
    df
    .groupby(
        "ID",
        sort=False
    )[status_vars]
    .ffill()
)


# =============================================================================
# 17. ART
#
# current和cumulative均只向前LOCF
# 剩余缺失=尚无暴露，因此填0
# =============================================================================

df[CURRENT_ART_VARS] = (
    df
    .groupby(
        "ID",
        sort=False
    )[CURRENT_ART_VARS]
    .ffill()
    .fillna(0)
)


df[CUM_ART_VARS] = (
    df
    .groupby(
        "ID",
        sort=False
    )[CUM_ART_VARS]
    .ffill()
    .fillna(0)
)


# =============================================================================
# 18. 读取开发集固定数据
#
# 外部队列绝不能自己计算中位数/众数
# =============================================================================

development_idx = np.load(
    DEV_IDX_FILE
).astype(int)


dev_mask_all = np.load(
    DEV_MASK_FILE
).astype(bool)


dev_cont = np.load(
    DEV_CONT_FILE,
    mmap_mode="r"
)


dev_bin = np.load(
    DEV_BIN_FILE,
    mmap_mode="r"
)


dev_cat = np.load(
    DEV_CAT_FILE,
    mmap_mode="r"
)


dev_mask = (
    dev_mask_all[
        development_idx
    ]
)


dev_cont_rows = np.asarray(
    dev_cont[
        development_idx
    ],
    dtype=np.float64
)[
    dev_mask
]


dev_bin_rows = np.asarray(
    dev_bin[
        development_idx
    ],
    dtype=np.float64
)[
    dev_mask
]


dev_cat_rows = np.asarray(
    dev_cat[
        development_idx
    ],
    dtype=object
)[
    dev_mask
]


# =============================================================================
# 19. 开发集固定连续变量中位数
# =============================================================================

continuous_fill = {}


for j, col in enumerate(
    continuous_vars
):

    value = float(
        np.nanmedian(
            dev_cont_rows[:, j]
        )
    )

    if not np.isfinite(value):
        raise ValueError(
            f"{col}开发集中位数异常。"
        )

    continuous_fill[col] = value


# =============================================================================
# 20. 众数函数
# =============================================================================

def get_mode(x):

    s = pd.Series(x).dropna()

    if len(s) == 0:
        return np.nan

    return s.mode().iloc[0]


# =============================================================================
# 21. 开发集二分类众数
# =============================================================================

binary_fill = {}


for j, col in enumerate(
    binary_vars
):

    value = get_mode(
        dev_bin_rows[:, j]
    )

    if pd.isna(value):
        raise ValueError(
            f"{col}开发集众数异常。"
        )

    binary_fill[col] = float(value)


# =============================================================================
# 22. 开发集分类变量众数
# =============================================================================

categorical_fill = {}


for j, col in enumerate(
    categorical_vars
):

    value = get_mode(
        dev_cat_rows[:, j]
    )

    if pd.isna(value):
        raise ValueError(
            f"{col}开发集众数异常。"
        )

    categorical_fill[col] = str(value)


# =============================================================================
# 23. 完成剩余缺失
# =============================================================================

# 连续变量
for col in continuous_vars:

    if col in CUM_ART_VARS:
        # 累计ART剩余缺失已经定义为0
        continue

    df[col] = df[col].fillna(
        continuous_fill[col]
    )


# 二分类变量
for col in binary_vars:

    if col in CURRENT_ART_VARS:
        # current ART剩余缺失已经定义为0
        continue

    df[col] = df[col].fillna(
        binary_fill[col]
    )


# 分类变量
for col in categorical_vars:

    df[col] = (
        df[col]
        .fillna(
            categorical_fill[col]
        )
        .astype(str)
    )


# =============================================================================
# 24. 生成15项实验室填补来源
# =============================================================================

source_audit = []


for col in LAB_VARS:

    source = np.where(
        lab_observed[col],
        "observed",
        np.where(
            lab_after_locf[col],
            "LOCF",
            "development_median"
        )
    )


    source_audit.append(
        pd.DataFrame({
            "ID":
                df["ID"],

            "time_bin":
                df["time_bin"],

            "month":
                df["month"],

            "variable":
                col,

            "source":
                source,

            "value_final":
                df[col],
        })
    )


source_audit = pd.concat(
    source_audit,
    ignore_index=True
)


source_summary = (
    source_audit
    .groupby(
        [
            "variable",
            "source",
        ]
    )
    .size()
    .unstack(
        fill_value=0
    )
)


for col in [
    "observed",
    "LOCF",
    "development_median",
]:
    if col not in source_summary.columns:
        source_summary[col] = 0


source_summary = (
    source_summary[
        [
            "observed",
            "LOCF",
            "development_median",
        ]
    ]
    .reset_index()
)


source_summary["total"] = (
    source_summary[
        [
            "observed",
            "LOCF",
            "development_median",
        ]
    ]
    .sum(axis=1)
)


for col in [
    "observed",
    "LOCF",
    "development_median",
]:

    source_summary[
        f"{col}_pct"
    ] = (
        source_summary[col]
        / source_summary["total"]
        * 100
    )


# =============================================================================
# 25. 47源变量最终QC
# =============================================================================

missing = (
    df[SOURCE47]
    .isna()
    .sum()
)


if missing.sum() != 0:

    print(
        missing[
            missing > 0
        ]
    )

    raise ValueError(
        "47个源变量仍存在缺失。"
    )


# 二分类必须严格0/1
for col in binary_vars:

    values = set(
        df[col]
        .astype(float)
        .unique()
    )

    if not values.issubset(
        {0.0, 1.0}
    ):
        raise ValueError(
            f"{col}存在非0/1值：{values}"
        )


# =============================================================================
# 26. 保存47项填补后长表
# =============================================================================

metadata = [
    "ID",
    "ARTtime",
    "data",
    "time_bin",
    "month",
    "time_date",
    "Diagnosetime",
    "CKDstatus",
    "CKDtime",
    "Lastfollowtime",
    "interval",
]


metadata = [
    x
    for x in metadata
    if x in df.columns
]


source47 = df[
    metadata
    + SOURCE47
].copy()


source47.to_csv(
    OUT /
    "重庆_source47_imputed_GLU_updated.csv",
    index=False,
    encoding="utf-8-sig"
)


source_audit.to_csv(
    OUT /
    "实验室填补来源明细.csv.gz",
    index=False,
    compression="gzip",
    encoding="utf-8"
)


source_summary.to_csv(
    OUT /
    "实验室填补来源汇总.csv",
    index=False,
    encoding="utf-8-sig"
)


# =============================================================================
# 27. 只取0~60月
# =============================================================================

history = df.loc[
    df["time_bin"].between(
        0,
        10
    )
].copy()


# =============================================================================
# 28. 患者顺序
# =============================================================================

patient_ids = np.array(
    sorted(
        df["ID"]
        .astype(str)
        .unique()
    ),
    dtype=object
)


id_map = {
    pid: i
    for i, pid in enumerate(
        patient_ids
    )
}


N = len(patient_ids)


# =============================================================================
# 29. 构建原始25/18/4张量
# =============================================================================

continuous_raw = np.full(
    (
        N,
        11,
        25
    ),
    np.nan,
    dtype=np.float32
)


binary_raw = np.full(
    (
        N,
        11,
        18
    ),
    np.nan,
    dtype=np.float32
)


categorical_raw = np.full(
    (
        N,
        11,
        4
    ),
    "__NO_TIME_ROW__",
    dtype=object
)


sequence_mask = np.zeros(
    (
        N,
        11
    ),
    dtype=bool
)


p = (
    history["ID"]
    .astype(str)
    .map(id_map)
    .to_numpy(int)
)


t = (
    history["time_bin"]
    .to_numpy(int)
)


continuous_raw[
    p,
    t,
    :
] = (
    history[
        continuous_vars
    ]
    .to_numpy(
        np.float32
    )
)


binary_raw[
    p,
    t,
    :
] = (
    history[
        binary_vars
    ]
    .to_numpy(
        np.float32
    )
)


categorical_raw[
    p,
    t,
    :
] = (
    history[
        categorical_vars
    ]
    .to_numpy(
        object
    )
)


sequence_mask[
    p,
    t
] = True


# =============================================================================
# 30. 冻结预处理器
# =============================================================================

preprocessor = joblib.load(
    PREPROCESSOR_FILE
)


scaler = preprocessor[
    "scaler"
]

encoder = preprocessor[
    "encoder"
]

feature_names = list(
    preprocessor[
        "final_feature_names"
    ]
)


if len(feature_names) != 57:
    raise ValueError(
        "冻结模型不是57维。"
    )


# =============================================================================
# 31. 类别空间检查
# =============================================================================

for j, col in enumerate(
    categorical_vars
):

    observed = set(
        categorical_raw[
            :,
            :,
            j
        ][
            sequence_mask
        ]
        .astype(str)
    )


    formal = set(
        map(
            str,
            encoder.categories_[j]
        )
    )


    unknown = (
        observed
        - formal
    )


    if unknown:
        raise ValueError(
            f"{col}存在开发集未知类别："
            f"{unknown}"
        )


# =============================================================================
# 32. 冻结标准化 + One-Hot
# =============================================================================

continuous_observed = (
    continuous_raw[
        sequence_mask
    ]
)


binary_observed = (
    binary_raw[
        sequence_mask
    ]
)


categorical_observed = (
    categorical_raw[
        sequence_mask
    ]
)


continuous_scaled = (
    scaler
    .transform(
        continuous_observed
    )
    .astype(
        np.float32
    )
)


categorical_onehot = (
    encoder
    .transform(
        categorical_observed
    )
)


if hasattr(
    categorical_onehot,
    "toarray"
):
    categorical_onehot = (
        categorical_onehot
        .toarray()
    )


categorical_onehot = np.asarray(
    categorical_onehot,
    dtype=np.float32
)


X_observed = np.concatenate(
    [
        continuous_scaled,
        binary_observed.astype(
            np.float32
        ),
        categorical_onehot,
    ],
    axis=1
)


if X_observed.shape[1] != 57:
    raise ValueError(
        "转换后不是57维。"
    )


# =============================================================================
# 33. 最终11×57张量
# =============================================================================

X = np.zeros(
    (
        N,
        11,
        57
    ),
    dtype=np.float32
)


X[
    sequence_mask
] = X_observed


if not np.isfinite(
    X
).all():
    raise ValueError(
        "最终57维输入存在NaN/Inf。"
    )


if not np.all(
    X[
        ~sequence_mask
    ] == 0
):
    raise ValueError(
        "不存在时间行没有保持全0。"
    )


# =============================================================================
# 34. 患者级结局信息
# =============================================================================

patient_info = (
    df
    .groupby(
        "ID",
        sort=True,
        as_index=False
    )
    .first()
)


patient_info = patient_info[
    [
        x
        for x in [
            "ID",
            "ARTtime",
            "data",
            "CKDstatus",
            "CKDtime",
            "Lastfollowtime",
            "interval",
        ]
        if x in patient_info.columns
    ]
]


patient_info[
    "patient_index"
] = (
    patient_info["ID"]
    .astype(str)
    .map(id_map)
)


patient_info = (
    patient_info
    .sort_values(
        "patient_index"
    )
    .reset_index(drop=True)
)


# =============================================================================
# 35. 保存
# =============================================================================

np.save(
    OUT /
    "continuous_raw_0_60.npy",
    continuous_raw
)


np.save(
    OUT /
    "binary_raw_0_60.npy",
    binary_raw
)


np.save(
    OUT /
    "categorical_raw_0_60.npy",
    categorical_raw
)


np.save(
    OUT /
    "sequence_row_mask.npy",
    sequence_mask
)


np.save(
    OUT /
    "X_full_preprocessor.npy",
    X
)


patient_info.to_csv(
    OUT /
    "patient_info.csv",
    index=False,
    encoding="utf-8-sig"
)


pd.DataFrame({
    "feature_index":
        np.arange(57),

    "feature_name":
        feature_names,
}).to_csv(
    OUT /
    "feature_names.csv",
    index=False,
    encoding="utf-8-sig"
)


# =============================================================================
# 36. 重点GLU审计
# =============================================================================

glu_audit = (
    source_summary.loc[
        source_summary[
            "variable"
        ] == "GLU"
    ]
)


# =============================================================================
# 37. 输出
# =============================================================================

print()
print("=" * 80)
print("重庆外部验证 Step5：GLU更新后重新预处理完成")
print("=" * 80)

print(
    "读取编码：",
    encoding
)

print(
    "患者数：",
    N
)

print(
    "CKD事件数：",
    int(
        patient_info[
            "CKDstatus"
        ].sum()
    )
)

print(
    "完整纵向行数：",
    len(df)
)

print(
    "0~60月真实时间行：",
    int(
        sequence_mask.sum()
    )
)

print()
print(
    "连续原始张量：",
    continuous_raw.shape
)

print(
    "二分类原始张量：",
    binary_raw.shape
)

print(
    "分类原始张量：",
    categorical_raw.shape
)

print(
    "最终57维张量：",
    X.shape
)

print()
print(
    "47项剩余缺失：",
    int(
        df[
            SOURCE47
        ]
        .isna()
        .sum()
        .sum()
    )
)

print(
    "57维非有限值：",
    int(
        (
            ~np.isfinite(X)
        ).sum()
    )
)

print()
print(
    "HIVRNA主要LOD：",
    DEFAULT_LOD
)

print()
print("GLU填补来源：")
print(
    glu_audit
    .round(2)
    .to_string(
        index=False
    )
)

print()
print("15项实验室填补来源：")
print(
    source_summary[
        [
            "variable",
            "observed_pct",
            "LOCF_pct",
            "development_median_pct",
        ]
    ]
    .round(2)
    .to_string(
        index=False
    )
)

print()
print(
    "输出目录：",
    OUT
)

print("=" * 80)