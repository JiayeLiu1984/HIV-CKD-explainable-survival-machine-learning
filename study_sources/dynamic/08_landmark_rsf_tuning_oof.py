# ============================================================
# Step 8：Pooled Landmark RSF适度调参、SQLite可恢复运行版
#
# 运行阶段：
# 1. RUN_STAGE = "tune"
#    使用SQLite持久化Optuna Study；程序中断后保留所有已完成Trial。
# 2. RUN_STAGE = "final"
#    使用当前最佳参数完成最终5折OOF；每折完成后立即保存，支持折级续跑。
# 3. RUN_STAGE = "baseline"
#    可选，仅在希望额外获得固定参数基础模型时运行。
#
# 共同原则：
# - 使用与传统Cox相同的95项固定临床汇总特征。
# - 额外加入landmark_year，共96项RSF输入。
# - 同一患者全部Landmark固定属于同一折。
# - 锁定测试集不参与调参、模型选择或性能评价。
# ============================================================

import gc
import hashlib
import json
import os
import sys
import threading
import time
import warnings
from datetime import datetime
from pathlib import Path

warnings.filterwarnings("ignore", message="IProgress not found.*")
warnings.filterwarnings(
    "ignore",
    message="Argument ``multivariate`` is an experimental feature.*",
)
warnings.filterwarnings(
    "ignore",
    message="RetryFailedTrialCallback is experimental.*",
)

import joblib
import numpy as np
import optuna
import pandas as pd

try:
    import sksurv
    from sksurv.ensemble import RandomSurvivalForest
    from sksurv.metrics import (
        concordance_index_censored,
        integrated_brier_score,
    )
except ImportError as exc:
    raise ImportError(
        "Step 8需要scikit-survival。请先安装：\n"
        "conda install -c conda-forge scikit-survival optuna\n"
        "或：pip install scikit-survival optuna"
    ) from exc


# ============================================================
# 1. 只需要修改这里的运行阶段
# ============================================================

# 可选："baseline"、"tune"、"final"
RUN_STAGE = "final"
# 基础模型：先用于判断RSF是否可行及估算耗时。
BASELINE_TREE_N = 100
BASELINE_PARAMETERS = {
    "max_features": "sqrt",
    "max_depth": 12,
    "min_samples_leaf": 50,
    "min_samples_split": 100,
    "max_samples": 0.70,
}

# 调参以“完成Trial数”为目标。以后增大目标值，SQLite会继续补足。
# 建议先完成12个Trial；确认仍有明显改进空间后可增至20。
TUNING_TOTAL_COMPLETE_TRIAL_TARGET = 12
TUNING_MAX_TOTAL_ATTEMPT_N = 30
TUNING_TREE_N = 100
OPTUNA_STARTUP_TRIAL_N = 4
SEARCH_SPACE_VERSION = "rsf_moderate_v2_20260728"

# 最终OOF模型树数。
FINAL_TREE_N = 500

# 心跳和并行设置。
HEARTBEAT_SECONDS = 60
CPU_COUNT = os.cpu_count() or 1
N_JOBS = min(16, CPU_COUNT)
PREDICTION_CHUNK_SIZE = 4000


# ============================================================
# 2. 路径和固定数据参数
# ============================================================

STEP5_DIR = Path(
    "__CKD_WORKDIR__/rolling_5y_step5_landmark_summary"
)
STEP6_DIR = Path(
    "__CKD_WORKDIR__/rolling_5y_step6_super_landmark_data"
)
STEP7_DIR = Path(
    "__CKD_WORKDIR__/rolling_5y_step7_unpenalized_cox_fixed95_v3"
)
OUTPUT_DIR = Path(
    "__CKD_WORKDIR__/rolling_5y_step8_pooled_landmark_rsf_resume_v2"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BASELINE_DIR = OUTPUT_DIR / "baseline"
TUNING_DIR = OUTPUT_DIR / "tuning"
FINAL_DIR = OUTPUT_DIR / "final_oof"
for path in [BASELINE_DIR, TUNING_DIR, FINAL_DIR]:
    path.mkdir(parents=True, exist_ok=True)

LIVE_PROGRESS_FILE = OUTPUT_DIR / "step8_live_progress.log"
PROGRESS_TABLE_FILE = OUTPUT_DIR / "step8_progress_table.csv"
OPTUNA_DB_FILE = TUNING_DIR / "rsf_optuna.sqlite3"
OPTUNA_STORAGE_URL = f"sqlite:///{OPTUNA_DB_FILE}"
OPTUNA_STUDY_NAME = "pooled_landmark_rsf_ibs_resume_v2"

N_SPLITS = 5
EXPECTED_DEVELOPMENT_N = 22337
EXPECTED_LANDMARK_N = 6
EXPECTED_FUTURE_INTERVAL_N = 10
EXPECTED_SUMMARY_FEATURE_N = 146
EXPECTED_FIXED_FEATURE_N = 95
EXPECTED_RSF_FEATURE_N = 96
EXPECTED_LONG_RECORD_N = 100122

LANDMARK_MONTHS = np.array([0, 12, 24, 36, 48, 60], dtype=np.int32)
FUTURE_END_MONTHS = np.arange(6, 61, 6, dtype=np.float64)
IBS_EVALUATION_TIMES = np.array(
    [6, 12, 18, 24, 30, 36, 42, 48, 54, 59.999],
    dtype=np.float64,
)

BASE_RANDOM_SEED = 20260728
SURVIVAL_TOLERANCE = 1e-6
ART_MONTH_TOLERANCE = 0.25
LANDMARK_FEATURE_NAME = "landmark_year"
DERIVED_CURRENT_ART_OTHER = "current_ART_Other_at_landmark"
DERIVED_CUMULATIVE_ART_OTHER = "ART_Other_cum_month_at_landmark"

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
# 3. 进度与通用辅助函数
# ============================================================

_PRINT_LOCK = threading.Lock()


def format_duration(seconds):
    seconds = max(0, int(round(float(seconds))))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours > 0:
        return f"{hours}小时{minutes:02d}分{seconds:02d}秒"
    if minutes > 0:
        return f"{minutes}分{seconds:02d}秒"
    return f"{seconds}秒"


def progress_print(message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {message}"
    with _PRINT_LOCK:
        print(line, flush=True)
        with open(LIVE_PROGRESS_FILE, "a", encoding="utf-8") as file:
            file.write(line + "\n")
            file.flush()


def append_progress_row(
    phase,
    status,
    trial_number=None,
    fold_id=None,
    metric_name=None,
    metric_value=None,
    elapsed_seconds=None,
    message="",
):
    row = pd.DataFrame([{
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "phase": phase,
        "status": status,
        "trial_number": trial_number,
        "fold_id": fold_id,
        "metric_name": metric_name,
        "metric_value": metric_value,
        "elapsed_seconds": elapsed_seconds,
        "message": message,
    }])
    row.to_csv(
        PROGRESS_TABLE_FILE,
        mode="a",
        header=not PROGRESS_TABLE_FILE.exists(),
        index=False,
        encoding="utf-8-sig",
    )


class ProgressHeartbeat:
    def __init__(self, label, interval_seconds=HEARTBEAT_SECONDS):
        self.label = str(label)
        self.interval_seconds = int(interval_seconds)
        self.start_time = None
        self.stop_event = threading.Event()
        self.thread = None

    def _run(self):
        while not self.stop_event.wait(self.interval_seconds):
            elapsed = time.time() - self.start_time
            progress_print(
                f"{self.label}仍在运行 | 已耗时{format_duration(elapsed)} | "
                f"PID={os.getpid()}"
            )

    def __enter__(self):
        self.start_time = time.time()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=2)
        return False


def build_survival_array(event, time_month):
    event = np.asarray(event, dtype=bool)
    time_month = np.asarray(time_month, dtype=np.float64)
    if event.shape != time_month.shape:
        raise ValueError("事件数组与生存时间数组形状不一致。")
    if not np.isfinite(time_month).all() or np.any(time_month <= 0):
        raise ValueError("生存时间必须为有限正数。")

    y = np.empty(event.shape[0], dtype=[("event", "?"), ("time", "<f8")])
    y["event"] = event
    y["time"] = time_month
    return y


def parameter_signature(parameters, n_estimators):
    payload = {
        "parameters": parameters,
        "n_estimators": int(n_estimators),
        "feature_n": EXPECTED_RSF_FEATURE_N,
        "random_seed": BASE_RANDOM_SEED,
    }
    text = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def create_rsf(parameters, n_estimators, random_state):
    return RandomSurvivalForest(
        n_estimators=int(n_estimators),
        max_depth=parameters["max_depth"],
        min_samples_split=int(parameters["min_samples_split"]),
        min_samples_leaf=int(parameters["min_samples_leaf"]),
        max_features=parameters["max_features"],
        bootstrap=True,
        oob_score=False,
        n_jobs=N_JOBS,
        random_state=int(random_state),
        max_samples=float(parameters["max_samples"]),
        low_memory=False,
        verbose=0,
    )


def predict_survival_at_times(model, X, evaluation_times):
    X = np.asarray(X, dtype=np.float32)
    evaluation_times = np.asarray(evaluation_times, dtype=np.float64)
    unique_times = np.asarray(model.unique_times_, dtype=np.float64)

    if evaluation_times.ndim != 1 or np.any(np.diff(evaluation_times) <= 0):
        raise ValueError("预测时间必须为严格递增的一维数组。")
    if unique_times.ndim != 1 or unique_times.size == 0:
        raise ValueError("RSF模型没有有效的unique_times_。")

    positions = np.searchsorted(unique_times, evaluation_times, side="right") - 1
    result = np.empty((X.shape[0], evaluation_times.size), dtype=np.float32)

    for start in range(0, X.shape[0], PREDICTION_CHUNK_SIZE):
        stop = min(start + PREDICTION_CHUNK_SIZE, X.shape[0])
        full_survival = np.asarray(
            model.predict_survival_function(X[start:stop], return_array=True),
            dtype=np.float64,
        )
        if full_survival.shape != (stop - start, unique_times.size):
            raise ValueError("RSF返回的完整生存概率形状异常。")

        chunk = np.ones((stop - start, evaluation_times.size), dtype=np.float64)
        valid = positions >= 0
        if np.any(valid):
            chunk[:, valid] = full_survival[:, positions[valid]]
        result[start:stop] = np.clip(chunk, 0.0, 1.0).astype(np.float32)

        del full_survival, chunk
        gc.collect()

    return result


def compute_equal_weight_landmark_ibs(
    y_train,
    y_validation,
    train_landmark_month,
    validation_landmark_month,
    validation_survival,
):
    rows = []
    for landmark_month in LANDMARK_MONTHS:
        train_mask = train_landmark_month == landmark_month
        validation_mask = validation_landmark_month == landmark_month
        y_train_landmark = y_train[train_mask]
        y_validation_landmark = y_validation[validation_mask]
        estimate = np.asarray(validation_survival[validation_mask], dtype=np.float64)

        if train_mask.sum() == 0 or validation_mask.sum() == 0:
            raise ValueError(f"Landmark {landmark_month}个月缺少训练或验证记录。")

        max_supported_time = min(
            float(np.max(y_train_landmark["time"])),
            float(np.max(y_validation_landmark["time"])),
        )
        if max_supported_time <= IBS_EVALUATION_TIMES[-1]:
            raise ValueError(
                f"Landmark {landmark_month}个月不足以计算至59.999个月IBS。"
            )

        ibs_value = float(integrated_brier_score(
            y_train_landmark,
            y_validation_landmark,
            estimate,
            IBS_EVALUATION_TIMES,
        ))
        rows.append({
            "landmark_month": int(landmark_month),
            "train_record_n": int(train_mask.sum()),
            "validation_record_n": int(validation_mask.sum()),
            "validation_event_n": int(y_validation_landmark["event"].sum()),
            "ibs_0_5_to_5y": ibs_value,
        })

    table = pd.DataFrame(rows)
    return float(table["ibs_0_5_to_5y"].mean()), table


def compute_landmark_c_index(y_validation, validation_landmark_month, risk_score):
    rows = []
    for landmark_month in LANDMARK_MONTHS:
        mask = validation_landmark_month == landmark_month
        y_landmark = y_validation[mask]
        score = np.asarray(risk_score[mask], dtype=np.float64)
        if int(y_landmark["event"].sum()) == 0:
            c_index = np.nan
        else:
            c_index = float(concordance_index_censored(
                y_landmark["event"],
                y_landmark["time"],
                score,
            )[0])
        rows.append({
            "landmark_month": int(landmark_month),
            "record_n": int(mask.sum()),
            "event_n": int(y_landmark["event"].sum()),
            "harrell_c_index": c_index,
        })
    return pd.DataFrame(rows)


def decode_trial_parameters(trial):
    # 96项输入、约10万条长记录下采用适度搜索空间，避免过深树和过小叶节点。
    max_features_label = trial.suggest_categorical(
        "max_features", ["sqrt", "0.15", "0.25"]
    )
    max_depth_label = trial.suggest_categorical(
        "max_depth", ["8", "12", "16"]
    )
    min_samples_leaf = trial.suggest_categorical(
        "min_samples_leaf", [30, 50, 80, 120]
    )
    split_multiplier = trial.suggest_categorical(
        "split_multiplier", [2, 3]
    )
    max_samples = trial.suggest_categorical(
        "max_samples", [0.60, 0.75, 0.90]
    )

    return {
        "max_features": (
            "sqrt" if max_features_label == "sqrt" else float(max_features_label)
        ),
        "max_depth": int(max_depth_label),
        "min_samples_leaf": int(min_samples_leaf),
        "min_samples_split": int(max(
            2 * min_samples_leaf,
            split_multiplier * min_samples_leaf,
        )),
        "max_samples": float(max_samples),
    }


def normalized_parameter_dictionary(parameters):
    min_samples_leaf = int(parameters["min_samples_leaf"])
    split_multiplier = int(parameters["split_multiplier"])
    return {
        "max_features": (
            "sqrt"
            if parameters["max_features"] == "sqrt"
            else float(parameters["max_features"])
        ),
        "max_depth": int(parameters["max_depth"]),
        "min_samples_leaf": min_samples_leaf,
        "min_samples_split": int(max(
            2 * min_samples_leaf,
            split_multiplier * min_samples_leaf,
        )),
        "max_samples": float(parameters["max_samples"]),
    }


# ============================================================
# 4. 读取并核查输入数据
# ============================================================

required_files = [
    STEP5_DIR / "summary_feature_names.csv",
    STEP5_DIR / "summary_column_semantic_audit.csv",
    STEP6_DIR / "X_super_landmark_development_raw.npy",
    STEP6_DIR / "summary_feature_names.csv",
    STEP6_DIR / "development_long_row_index_map.npy",
    STEP6_DIR / "development_fold_id_long.npy",
    STEP6_DIR / "development_landmark_month_long.npy",
    STEP6_DIR / "development_analysis_time_month.npy",
    STEP6_DIR / "development_event_within_60m.npy",
    STEP7_DIR / "cox_fixed_feature_dictionary_95.csv",
]
for fold_id in range(N_SPLITS):
    required_files.extend([
        STEP6_DIR / f"fold_{fold_id}" / "train_long_idx.npy",
        STEP6_DIR / f"fold_{fold_id}" / "validation_long_idx.npy",
    ])

missing_files = [str(path) for path in required_files if not path.exists()]
if missing_files:
    raise FileNotFoundError("以下输入文件不存在：\n" + "\n".join(missing_files))

step5_feature_table = pd.read_csv(
    STEP5_DIR / "summary_feature_names.csv", encoding="utf-8-sig"
)
step6_feature_table = pd.read_csv(
    STEP6_DIR / "summary_feature_names.csv", encoding="utf-8-sig"
)
semantic_audit = pd.read_csv(
    STEP5_DIR / "summary_column_semantic_audit.csv", encoding="utf-8-sig"
)
fixed_feature_dictionary = pd.read_csv(
    STEP7_DIR / "cox_fixed_feature_dictionary_95.csv", encoding="utf-8-sig"
).sort_values("cox_feature_index").reset_index(drop=True)

X_long_raw = np.load(
    STEP6_DIR / "X_super_landmark_development_raw.npy", mmap_mode="r"
)
row_index_map = np.load(
    STEP6_DIR / "development_long_row_index_map.npy"
).astype(np.int32)
fold_id_long = np.load(
    STEP6_DIR / "development_fold_id_long.npy"
).astype(np.int8)
landmark_month_long = np.load(
    STEP6_DIR / "development_landmark_month_long.npy"
).astype(np.int32)
analysis_time_month = np.load(
    STEP6_DIR / "development_analysis_time_month.npy"
).astype(np.float64)
event_within_60m = np.load(
    STEP6_DIR / "development_event_within_60m.npy"
).astype(bool)

if not step5_feature_table.equals(step6_feature_table):
    raise ValueError(
        "Step 5和Step 6的特征字典不一致，请重新运行最新Step 5和Step 6。"
    )
if int(semantic_audit["invalid_n"].fillna(0).sum()) != 0:
    raise ValueError("Step 5语义审计仍存在异常。")
if X_long_raw.shape != (EXPECTED_LONG_RECORD_N, EXPECTED_SUMMARY_FEATURE_N):
    raise ValueError(f"Step 6长格式特征形状异常：{X_long_raw.shape}。")
if row_index_map.shape != (EXPECTED_DEVELOPMENT_N, EXPECTED_LANDMARK_N):
    raise ValueError(f"长行号映射形状异常：{row_index_map.shape}。")
if fixed_feature_dictionary.shape[0] != EXPECTED_FIXED_FEATURE_N:
    raise ValueError("Step 7固定特征数不是95。")
if np.any(analysis_time_month <= 0) or np.any(analysis_time_month > 60.0001):
    raise ValueError("分析时间不在(0, 60]个月范围。")

feature_name_to_index = dict(zip(
    step6_feature_table["summary_feature_name"].astype(str),
    step6_feature_table["summary_feature_index"].astype(int),
))

current_art_feature_names = [
    f"{name}_at_landmark" for name in CURRENT_ART_SOURCE_NAMES
]
cumulative_art_feature_names = [
    f"{name}_at_landmark" for name in ART_CUMULATIVE_SOURCE_NAMES
]

current_art_matrix = np.asarray(
    X_long_raw[:, [feature_name_to_index[name] for name in current_art_feature_names]],
    dtype=np.float64,
)
if not np.all(
    np.isclose(current_art_matrix, 0.0, atol=1e-7)
    | np.isclose(current_art_matrix, 1.0, atol=1e-7)
):
    raise ValueError("当前ART方案变量不是严格0/1。")
if np.any(current_art_matrix.sum(axis=1) > 1.0 + 1e-7):
    raise ValueError("同一记录同时属于多个当前ART方案。")
current_art_other = np.isclose(
    current_art_matrix.sum(axis=1), 0.0, atol=1e-7
).astype(np.float32)

cumulative_art_matrix = np.asarray(
    X_long_raw[:, [feature_name_to_index[name] for name in cumulative_art_feature_names]],
    dtype=np.float64,
)
if np.any(cumulative_art_matrix < -1e-7):
    raise ValueError("ART累计暴露存在负值。")
other_cumulative_art = (
    landmark_month_long.astype(np.float64) - cumulative_art_matrix.sum(axis=1)
)
if float(np.min(other_cumulative_art)) < -ART_MONTH_TOLERANCE:
    raise ValueError("8类ART累计暴露之和超过Landmark月份。")
other_cumulative_art = np.maximum(other_cumulative_art, 0.0).astype(np.float32)

fixed_feature_names = fixed_feature_dictionary["cox_feature_name"].astype(str).tolist()
fixed_columns = []
for feature_name in fixed_feature_names:
    if feature_name in feature_name_to_index:
        fixed_columns.append(np.asarray(
            X_long_raw[:, feature_name_to_index[feature_name]],
            dtype=np.float32,
        ))
    elif feature_name == DERIVED_CURRENT_ART_OTHER:
        fixed_columns.append(current_art_other)
    elif feature_name == DERIVED_CUMULATIVE_ART_OTHER:
        fixed_columns.append(other_cumulative_art)
    else:
        raise ValueError(f"无法重建固定特征：{feature_name}")

X_fixed95 = np.column_stack(fixed_columns).astype(np.float32)
landmark_year = (landmark_month_long.astype(np.float32) / 12.0).reshape(-1, 1)
X_rsf = np.column_stack([X_fixed95, landmark_year]).astype(np.float32)
if X_rsf.shape != (EXPECTED_LONG_RECORD_N, EXPECTED_RSF_FEATURE_N):
    raise ValueError(f"RSF输入形状异常：{X_rsf.shape}。")
if not np.isfinite(X_rsf).all():
    raise ValueError("RSF输入包含NaN或无穷值。")

y_all = build_survival_array(event_within_60m, analysis_time_month)
np.save(OUTPUT_DIR / "X_pooled_landmark_development_fixed96.npy", X_rsf)

rsf_feature_dictionary = pd.concat([
    fixed_feature_dictionary.rename(columns={
        "cox_feature_index": "rsf_feature_index",
        "cox_feature_name": "rsf_feature_name",
    }),
    pd.DataFrame({
        "rsf_feature_index": [EXPECTED_FIXED_FEATURE_N],
        "rsf_feature_name": [LANDMARK_FEATURE_NAME],
        "feature_role": ["landmark_context"],
        "source_variable": ["landmark_month"],
        "is_binary_unscaled": [False],
    }),
], ignore_index=True)
rsf_feature_dictionary.to_csv(
    OUTPUT_DIR / "rsf_feature_dictionary_96.csv",
    index=False,
    encoding="utf-8-sig",
)

fold_indices = {}
for fold_id in range(N_SPLITS):
    train_idx = np.load(
        STEP6_DIR / f"fold_{fold_id}" / "train_long_idx.npy"
    ).astype(np.int32)
    validation_idx = np.load(
        STEP6_DIR / f"fold_{fold_id}" / "validation_long_idx.npy"
    ).astype(np.int32)
    if np.intersect1d(train_idx, validation_idx).size != 0:
        raise ValueError(f"第{fold_id}折训练和验证记录重叠。")
    if not np.all(fold_id_long[validation_idx] == fold_id):
        raise ValueError(f"第{fold_id}折验证记录fold_id错误。")
    fold_indices[fold_id] = {
        "train_idx": train_idx,
        "validation_idx": validation_idx,
    }


# ============================================================
# 5. 可恢复的五折OOF运行函数
# ============================================================


def run_resumable_oof(stage_name, parameters, n_estimators, stage_dir, prefix):
    stage_dir.mkdir(parents=True, exist_ok=True)
    signature = parameter_signature(parameters, n_estimators)

    with open(stage_dir / "stage_configuration.json", "w", encoding="utf-8") as file:
        json.dump({
            "stage_name": stage_name,
            "parameters": parameters,
            "n_estimators": int(n_estimators),
            "parameter_signature": signature,
            "n_jobs": N_JOBS,
        }, file, ensure_ascii=False, indent=2)

    oof_survival_long = np.full(
        (EXPECTED_LONG_RECORD_N, EXPECTED_FUTURE_INTERVAL_N),
        np.nan,
        dtype=np.float32,
    )
    oof_risk_score_long = np.full(EXPECTED_LONG_RECORD_N, np.nan, dtype=np.float32)
    fold_summary_rows = []
    fold_landmark_tables = []
    stage_start = time.time()

    for fold_id in range(N_SPLITS):
        train_idx = fold_indices[fold_id]["train_idx"]
        validation_idx = fold_indices[fold_id]["validation_idx"]
        fold_dir = stage_dir / f"fold_{fold_id}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        marker_file = fold_dir / "completed.json"

        if marker_file.exists():
            marker = json.loads(marker_file.read_text(encoding="utf-8"))
            if marker.get("parameter_signature") != signature:
                raise ValueError(
                    f"{stage_name}第{fold_id}折已有结果，但参数签名不同。"
                    f"请删除目录：{fold_dir}"
                )

            validation_survival = np.load(
                fold_dir / "validation_survival.npy"
            ).astype(np.float32)
            validation_risk_score = np.load(
                fold_dir / "validation_risk_score.npy"
            ).astype(np.float32)
            if validation_survival.shape != (
                validation_idx.size,
                EXPECTED_FUTURE_INTERVAL_N,
            ):
                raise ValueError(f"{stage_name}第{fold_id}折已保存预测形状异常。")

            oof_survival_long[validation_idx] = validation_survival
            oof_risk_score_long[validation_idx] = validation_risk_score
            fold_summary_rows.append(pd.read_csv(
                fold_dir / "fold_summary.csv", encoding="utf-8-sig"
            ).iloc[0].to_dict())
            fold_landmark_tables.append(pd.read_csv(
                fold_dir / "validation_landmark_metrics.csv",
                encoding="utf-8-sig",
            ))
            progress_print(
                f"{stage_name}第{fold_id + 1}/{N_SPLITS}折已完成，直接读取检查点。"
            )
            continue

        fold_start = time.time()
        progress_print(
            f"{stage_name}第{fold_id + 1}/{N_SPLITS}折开始 | "
            f"训练{train_idx.size}条，验证{validation_idx.size}条，"
            f"树数{n_estimators}"
        )

        model = create_rsf(
            parameters,
            n_estimators=n_estimators,
            random_state=BASE_RANDOM_SEED + 10000 + fold_id,
        )

        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always")
            with ProgressHeartbeat(
                f"{stage_name}第{fold_id + 1}/{N_SPLITS}折RSF拟合"
            ):
                model.fit(X_rsf[train_idx], y_all[train_idx])

        with ProgressHeartbeat(
            f"{stage_name}第{fold_id + 1}/{N_SPLITS}折验证预测"
        ):
            validation_survival = predict_survival_at_times(
                model, X_rsf[validation_idx], FUTURE_END_MONTHS
            )
            validation_survival_ibs = predict_survival_at_times(
                model, X_rsf[validation_idx], IBS_EVALUATION_TIMES
            )
            validation_risk_score = np.asarray(
                model.predict(X_rsf[validation_idx]), dtype=np.float32
            )

        if not np.isfinite(validation_survival).all():
            raise ValueError(f"{stage_name}第{fold_id}折预测含NaN或无穷值。")
        monotonic_violation_n = int(np.sum(
            np.diff(validation_survival.astype(np.float64), axis=1)
            > SURVIVAL_TOLERANCE
        ))
        if monotonic_violation_n != 0:
            raise ValueError(f"{stage_name}第{fold_id}折生存概率不单调。")

        fold_mean_ibs, fold_ibs_table = compute_equal_weight_landmark_ibs(
            y_all[train_idx],
            y_all[validation_idx],
            landmark_month_long[train_idx],
            landmark_month_long[validation_idx],
            validation_survival_ibs,
        )
        fold_c_table = compute_landmark_c_index(
            y_all[validation_idx],
            landmark_month_long[validation_idx],
            validation_risk_score,
        )
        fold_landmark_table = fold_ibs_table.merge(
            fold_c_table,
            on="landmark_month",
            how="inner",
            validate="one_to_one",
        )
        fold_landmark_table.insert(0, "fold_id", fold_id)

        fold_elapsed = float(time.time() - fold_start)
        warning_messages = [
            f"{type(item.message).__name__}: {item.message}"
            for item in caught_warnings
        ]
        fold_summary = {
            "fold_id": fold_id,
            "train_record_n": int(train_idx.size),
            "validation_record_n": int(validation_idx.size),
            "train_event_n": int(event_within_60m[train_idx].sum()),
            "validation_event_n": int(event_within_60m[validation_idx].sum()),
            "feature_n": EXPECTED_RSF_FEATURE_N,
            "n_estimators": int(n_estimators),
            "equal_weight_mean_ibs": float(fold_mean_ibs),
            "mean_harrell_c_index": float(
                fold_c_table["harrell_c_index"].mean()
            ),
            "warning_n": len(warning_messages),
            "survival_monotonic_violation_n": monotonic_violation_n,
            "elapsed_seconds": fold_elapsed,
        }

        np.save(fold_dir / "validation_survival.npy", validation_survival)
        np.save(fold_dir / "validation_risk_score.npy", validation_risk_score)
        np.save(fold_dir / "validation_idx.npy", validation_idx)
        fold_landmark_table.to_csv(
            fold_dir / "validation_landmark_metrics.csv",
            index=False,
            encoding="utf-8-sig",
        )
        pd.DataFrame([fold_summary]).to_csv(
            fold_dir / "fold_summary.csv",
            index=False,
            encoding="utf-8-sig",
        )
        pd.DataFrame({"warning_message": warning_messages}).to_csv(
            fold_dir / "fit_warnings.csv",
            index=False,
            encoding="utf-8-sig",
        )
        joblib.dump(model, fold_dir / "rsf_model.joblib", compress=3)

        marker_file.write_text(json.dumps({
            "completed_at": datetime.now().isoformat(timespec="seconds"),
            "parameter_signature": signature,
            "validation_record_n": int(validation_idx.size),
        }, ensure_ascii=False, indent=2), encoding="utf-8")

        oof_survival_long[validation_idx] = validation_survival
        oof_risk_score_long[validation_idx] = validation_risk_score
        fold_summary_rows.append(fold_summary)
        fold_landmark_tables.append(fold_landmark_table)

        completed_fold_n = fold_id + 1
        elapsed = time.time() - stage_start
        estimated_remaining = elapsed / completed_fold_n * (N_SPLITS - completed_fold_n)
        progress_print(
            f"{stage_name}第{fold_id + 1}/{N_SPLITS}折完成 | "
            f"IBS={fold_mean_ibs:.6f} | "
            f"平均C-index={fold_summary['mean_harrell_c_index']:.6f} | "
            f"折耗时={format_duration(fold_elapsed)} | "
            f"预计剩余={format_duration(estimated_remaining)}"
        )
        append_progress_row(
            phase=stage_name,
            status="fold_completed",
            fold_id=fold_id,
            metric_name="equal_weight_mean_ibs",
            metric_value=fold_mean_ibs,
            elapsed_seconds=fold_elapsed,
        )

        del (
            model,
            validation_survival,
            validation_survival_ibs,
            validation_risk_score,
        )
        gc.collect()

    if not np.isfinite(oof_survival_long).all():
        raise ValueError(f"{stage_name} OOF生存概率未完整覆盖。")
    if not np.isfinite(oof_risk_score_long).all():
        raise ValueError(f"{stage_name} OOF风险分数未完整覆盖。")

    oof_risk_long = 1.0 - oof_survival_long
    oof_survival = np.full(
        (
            EXPECTED_DEVELOPMENT_N,
            EXPECTED_LANDMARK_N,
            EXPECTED_FUTURE_INTERVAL_N,
        ),
        np.nan,
        dtype=np.float32,
    )
    oof_risk = np.full_like(oof_survival, np.nan)
    oof_risk_score = np.full(
        (EXPECTED_DEVELOPMENT_N, EXPECTED_LANDMARK_N),
        np.nan,
        dtype=np.float32,
    )
    valid_origin_mask = row_index_map >= 0
    positions = row_index_map[valid_origin_mask]
    oof_survival[valid_origin_mask] = oof_survival_long[positions]
    oof_risk[valid_origin_mask] = oof_risk_long[positions]
    oof_risk_score[valid_origin_mask] = oof_risk_score_long[positions]

    np.save(stage_dir / f"{prefix}_oof_survival_long.npy", oof_survival_long)
    np.save(stage_dir / f"{prefix}_oof_risk_long.npy", oof_risk_long)
    np.save(stage_dir / f"{prefix}_oof_risk_score_long.npy", oof_risk_score_long)
    np.save(stage_dir / f"{prefix}_oof_survival.npy", oof_survival)
    np.save(stage_dir / f"{prefix}_oof_risk.npy", oof_risk)
    np.save(stage_dir / f"{prefix}_oof_risk_score.npy", oof_risk_score)

    fold_summary_table = pd.DataFrame(fold_summary_rows).sort_values("fold_id")
    fold_landmark_table = pd.concat(
        fold_landmark_tables, ignore_index=True
    ).sort_values(["landmark_month", "fold_id"])
    fold_summary_table.to_csv(
        stage_dir / f"{prefix}_fold_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    fold_landmark_table.to_csv(
        stage_dir / f"{prefix}_fold_landmark_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    total_elapsed = time.time() - stage_start
    overall_summary = {
        "stage_name": stage_name,
        "n_estimators": int(n_estimators),
        "parameters": parameters,
        "mean_fold_ibs": float(fold_summary_table["equal_weight_mean_ibs"].mean()),
        "sd_fold_ibs": float(fold_summary_table["equal_weight_mean_ibs"].std()),
        "mean_fold_c_index": float(fold_summary_table["mean_harrell_c_index"].mean()),
        "total_elapsed_seconds": float(total_elapsed),
        "mean_fold_elapsed_seconds": float(fold_summary_table["elapsed_seconds"].mean()),
        "valid_origin_n": int(valid_origin_mask.sum()),
    }
    with open(
        stage_dir / f"{prefix}_summary.json", "w", encoding="utf-8"
    ) as file:
        json.dump(overall_summary, file, ensure_ascii=False, indent=2)

    progress_print(
        f"{stage_name}完成 | 平均IBS="
        f"{overall_summary['mean_fold_ibs']:.6f} | "
        f"平均C-index={overall_summary['mean_fold_c_index']:.6f} | "
        f"总耗时={format_duration(total_elapsed)}"
    )
    return overall_summary


# ============================================================
# 6. 基础RSF阶段
# ============================================================


def run_baseline_stage():
    print("=" * 72)
    print("Step 8A：基础Pooled Landmark RSF五折OOF")
    print("=" * 72)
    print("基础树数：", BASELINE_TREE_N)
    print("基础参数：", BASELINE_PARAMETERS)
    print("CPU线程：", N_JOBS)
    print("PID：", os.getpid())
    print("支持折级续跑：是")
    print("输出目录：", BASELINE_DIR)

    summary = run_resumable_oof(
        stage_name="baseline",
        parameters=BASELINE_PARAMETERS,
        n_estimators=BASELINE_TREE_N,
        stage_dir=BASELINE_DIR,
        prefix="rsf_baseline",
    )

    print("\n基础RSF运行完成")
    print("五折平均IBS：", f"{summary['mean_fold_ibs']:.6f}")
    print("五折平均C-index：", f"{summary['mean_fold_c_index']:.6f}")
    print("总耗时：", format_duration(summary["total_elapsed_seconds"]))
    print("平均每折耗时：", format_duration(summary["mean_fold_elapsed_seconds"]))
    print("下一步：将RUN_STAGE改成'tune'后运行可恢复调参。")


# ============================================================
# 7. SQLite可恢复Optuna调参阶段
# ============================================================


def create_or_load_study():
    retry_callback = optuna.storages.RetryFailedTrialCallback(
        max_retry=1,
        inherit_intermediate_values=False,
    )
    storage = optuna.storages.RDBStorage(
        url=OPTUNA_STORAGE_URL,
        heartbeat_interval=60,
        grace_period=180,
        failed_trial_callback=retry_callback,
    )
    sampler = optuna.samplers.TPESampler(
        seed=BASE_RANDOM_SEED,
        n_startup_trials=OPTUNA_STARTUP_TRIAL_N,
    )
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=OPTUNA_STARTUP_TRIAL_N,
        n_warmup_steps=2,
    )
    study = optuna.create_study(
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
        study_name=OPTUNA_STUDY_NAME,
        storage=storage,
        load_if_exists=True,
    )
    optuna.storages.fail_stale_trials(study)

    stored_version = study.user_attrs.get("search_space_version")
    if stored_version is None:
        study.set_user_attr("search_space_version", SEARCH_SPACE_VERSION)
        study.set_user_attr("expected_rsf_feature_n", EXPECTED_RSF_FEATURE_N)
        study.set_user_attr("expected_long_record_n", EXPECTED_LONG_RECORD_N)
    elif stored_version != SEARCH_SPACE_VERSION:
        raise ValueError(
            "SQLite中的Study搜索空间版本与当前代码不一致："
            f"数据库={stored_version}，代码={SEARCH_SPACE_VERSION}。"
        )

    return study


def run_tuning_stage():
    print("=" * 72)
    print("Step 8B：SQLite可恢复Pooled Landmark RSF适度调参")
    print("=" * 72)
    print("Study数据库：", OPTUNA_DB_FILE)
    print("目标完成Trial数：", TUNING_TOTAL_COMPLETE_TRIAL_TARGET)
    print("最多终止Trial数：", TUNING_MAX_TOTAL_ATTEMPT_N)
    print("每个模型树数：", TUNING_TREE_N)
    print("搜索空间版本：", SEARCH_SPACE_VERSION)
    print("搜索空间：")
    print("  max_features = sqrt、0.15、0.25")
    print("  max_depth = 8、12、16")
    print("  min_samples_leaf = 30、50、80、120")
    print("  min_samples_split = 2或3 × min_samples_leaf")
    print("  max_samples = 0.60、0.75、0.90")
    print("CPU线程：", N_JOBS)
    print("PID：", os.getpid())

    study = create_or_load_study()
    terminal_states = {
        optuna.trial.TrialState.COMPLETE,
        optuna.trial.TrialState.PRUNED,
        optuna.trial.TrialState.FAIL,
    }

    def count_trials(current_study):
        terminal_n = sum(
            trial.state in terminal_states for trial in current_study.trials
        )
        complete_n = sum(
            trial.state == optuna.trial.TrialState.COMPLETE
            for trial in current_study.trials
        )
        pruned_n = sum(
            trial.state == optuna.trial.TrialState.PRUNED
            for trial in current_study.trials
        )
        failed_n = sum(
            trial.state == optuna.trial.TrialState.FAIL
            for trial in current_study.trials
        )
        return terminal_n, complete_n, pruned_n, failed_n

    terminal_trial_n, complete_trial_n, pruned_trial_n, failed_trial_n = (
        count_trials(study)
    )
    print("数据库已有完成Trial：", complete_trial_n)
    print("数据库已有剪枝Trial：", pruned_trial_n)
    print("数据库已有失败Trial：", failed_trial_n)
    print("中断后再次运行：已完成Trial直接保留，只补足未完成数量。")

    tuning_start = time.time()

    def objective(trial):
        parameters = decode_trial_parameters(trial)
        fold_ibs_values = []
        trial_start = time.time()
        progress_print(
            f"Trial {trial.number}开始 | 参数={parameters}"
        )

        for fold_id in range(N_SPLITS):
            train_idx = fold_indices[fold_id]["train_idx"]
            validation_idx = fold_indices[fold_id]["validation_idx"]
            fold_start = time.time()
            model = create_rsf(
                parameters,
                n_estimators=TUNING_TREE_N,
                random_state=(
                    BASE_RANDOM_SEED + trial.number * 100 + fold_id
                ),
            )
            validation_survival = None
            try:
                with ProgressHeartbeat(
                    f"Trial {trial.number} | 折{fold_id + 1}/{N_SPLITS}拟合"
                ):
                    model.fit(X_rsf[train_idx], y_all[train_idx])
                with ProgressHeartbeat(
                    f"Trial {trial.number} | 折{fold_id + 1}/{N_SPLITS}预测"
                ):
                    validation_survival = predict_survival_at_times(
                        model,
                        X_rsf[validation_idx],
                        IBS_EVALUATION_TIMES,
                    )

                fold_ibs, _ = compute_equal_weight_landmark_ibs(
                    y_all[train_idx],
                    y_all[validation_idx],
                    landmark_month_long[train_idx],
                    landmark_month_long[validation_idx],
                    validation_survival,
                )
                fold_ibs_values.append(fold_ibs)
                running_ibs = float(np.mean(fold_ibs_values))
                trial.report(running_ibs, step=fold_id)
                trial.set_user_attr(f"fold_{fold_id}_ibs", fold_ibs)

                progress_print(
                    f"Trial {trial.number} | 折{fold_id + 1}/{N_SPLITS}完成 | "
                    f"折IBS={fold_ibs:.6f} | "
                    f"当前平均={running_ibs:.6f} | "
                    f"耗时={format_duration(time.time() - fold_start)}"
                )
                if trial.should_prune():
                    progress_print(
                        f"Trial {trial.number}在折{fold_id + 1}后剪枝。"
                    )
                    raise optuna.TrialPruned()
            finally:
                del model
                if validation_survival is not None:
                    del validation_survival
                gc.collect()

        mean_ibs = float(np.mean(fold_ibs_values))
        trial.set_user_attr("five_fold_mean_ibs", mean_ibs)
        trial.set_user_attr("elapsed_seconds", time.time() - trial_start)
        trial.set_user_attr("search_space_version", SEARCH_SPACE_VERSION)
        progress_print(
            f"Trial {trial.number}完成 | 五折平均IBS={mean_ibs:.6f} | "
            f"耗时={format_duration(time.time() - trial_start)}"
        )
        return mean_ibs

    while True:
        study = create_or_load_study()
        terminal_trial_n, complete_trial_n, pruned_trial_n, failed_trial_n = (
            count_trials(study)
        )
        if complete_trial_n >= TUNING_TOTAL_COMPLETE_TRIAL_TARGET:
            break
        if terminal_trial_n >= TUNING_MAX_TOTAL_ATTEMPT_N:
            progress_print(
                "达到最多终止Trial数，停止继续尝试。"
            )
            break

        progress_print(
            f"准备运行下一个Trial | 已完成{complete_trial_n}/"
            f"{TUNING_TOTAL_COMPLETE_TRIAL_TARGET} | "
            f"已剪枝{pruned_trial_n} | 已失败{failed_trial_n}"
        )
        try:
            study.optimize(
                objective,
                n_trials=1,
                gc_after_trial=True,
                show_progress_bar=False,
                catch=(Exception,),
            )
        except KeyboardInterrupt:
            progress_print(
                "检测到手动中断。SQLite已保留此前完成的Trial；下次运行会继续。"
            )
            raise

        study = create_or_load_study()
        trials_table = study.trials_dataframe()
        trials_table.to_csv(
            TUNING_DIR / "rsf_optuna_trials.csv",
            index=False,
            encoding="utf-8-sig",
        )

        completed = [
            trial for trial in study.trials
            if trial.state == optuna.trial.TrialState.COMPLETE
        ]
        if completed:
            terminal_trial_n, complete_trial_n, pruned_trial_n, failed_trial_n = (
                count_trials(study)
            )
            best_parameters = normalized_parameter_dictionary(
                study.best_trial.params
            )
            with open(
                TUNING_DIR / "rsf_best_parameters.json",
                "w",
                encoding="utf-8",
            ) as file:
                json.dump({
                    "best_trial_number": int(study.best_trial.number),
                    "best_equal_weight_mean_ibs": float(study.best_value),
                    "optuna_parameters": study.best_trial.params,
                    "model_parameters": best_parameters,
                    "search_space_version": SEARCH_SPACE_VERSION,
                    "tuning_n_estimators": TUNING_TREE_N,
                    "target_complete_trial_n": (
                        TUNING_TOTAL_COMPLETE_TRIAL_TARGET
                    ),
                    "terminal_trial_n": terminal_trial_n,
                    "complete_trial_n": complete_trial_n,
                    "pruned_trial_n": pruned_trial_n,
                    "failed_trial_n": failed_trial_n,
                }, file, ensure_ascii=False, indent=2)
            progress_print(
                f"当前最佳Trial={study.best_trial.number} | "
                f"最佳IBS={study.best_value:.6f} | "
                f"完成进度={complete_trial_n}/"
                f"{TUNING_TOTAL_COMPLETE_TRIAL_TARGET}"
            )

    study = create_or_load_study()
    completed = [
        trial for trial in study.trials
        if trial.state == optuna.trial.TrialState.COMPLETE
    ]
    if not completed:
        raise RuntimeError("当前Study没有任何完成的Trial。")

    terminal_trial_n, complete_trial_n, pruned_trial_n, failed_trial_n = (
        count_trials(study)
    )
    best_parameters = normalized_parameter_dictionary(study.best_trial.params)
    print("\n调参阶段结束")
    print("已保存数据库：", OPTUNA_DB_FILE)
    print("完成Trial数：", complete_trial_n)
    print("剪枝Trial数：", pruned_trial_n)
    print("失败Trial数：", failed_trial_n)
    print("当前最佳Trial：", study.best_trial.number)
    print("当前最佳IBS：", f"{study.best_value:.6f}")
    print("当前最佳参数：", best_parameters)
    print("本次运行耗时：", format_duration(time.time() - tuning_start))
    print(
        "以后增大TUNING_TOTAL_COMPLETE_TRIAL_TARGET后重跑，"
        "会从同一SQLite数据库继续。"
    )
    print("确认调参结果后，将RUN_STAGE改成'final'。")


# ============================================================
# 8. 最佳参数最终OOF阶段
# ============================================================


def run_final_stage():
    study = create_or_load_study()
    completed = [
        trial for trial in study.trials
        if trial.state == optuna.trial.TrialState.COMPLETE
    ]
    if not completed:
        raise RuntimeError("没有已完成的Optuna Trial，不能运行final阶段。")

    best_parameters = normalized_parameter_dictionary(study.best_trial.params)
    print("=" * 72)
    print("Step 8C：最佳参数最终Pooled Landmark RSF五折OOF")
    print("=" * 72)
    print("最佳Trial：", study.best_trial.number)
    print("最佳调参IBS：", f"{study.best_value:.6f}")
    print("最佳参数：", best_parameters)
    print("最终每折树数：", FINAL_TREE_N)
    print("支持折级续跑：是")
    print("PID：", os.getpid())

    summary = run_resumable_oof(
        stage_name="final_oof",
        parameters=best_parameters,
        n_estimators=FINAL_TREE_N,
        stage_dir=FINAL_DIR,
        prefix="rsf",
    )

    with open(OUTPUT_DIR / "step8_manifest.json", "w", encoding="utf-8") as file:
        json.dump({
            "step": 8,
            "model_name": "Pooled Landmark Random Survival Forest",
            "run_design": "sqlite_resume_moderate_tuning_then_final",
            "feature_n": EXPECTED_RSF_FEATURE_N,
            "best_trial_number": int(study.best_trial.number),
            "best_tuning_ibs": float(study.best_value),
            "best_model_parameters": best_parameters,
            "final_tree_n": FINAL_TREE_N,
            "final_oof_summary": summary,
            "test_set_used": False,
            "scikit_survival_version": sksurv.__version__,
            "optuna_version": optuna.__version__,
            "python_version": sys.version,
        }, file, ensure_ascii=False, indent=2)

    print("\n最终OOF完成")
    print("五折平均IBS：", f"{summary['mean_fold_ibs']:.6f}")
    print("五折平均C-index：", f"{summary['mean_fold_c_index']:.6f}")
    print("总耗时：", format_duration(summary["total_elapsed_seconds"]))
    print("输出目录：", FINAL_DIR)


# ============================================================
# 9. 根据RUN_STAGE执行对应阶段
# ============================================================

if RUN_STAGE not in {"baseline", "tune", "final"}:
    raise ValueError("RUN_STAGE只能是baseline、tune或final。")

print("\n" + "=" * 72)
print("Step 8：Pooled Landmark RSF适度调参与可恢复运行")
print("=" * 72)
print("当前阶段：", RUN_STAGE)
print("scikit-survival版本：", sksurv.__version__)
print("Optuna版本：", optuna.__version__)
print("开发集长记录数：", EXPECTED_LONG_RECORD_N)
print("RSF输入特征数：", EXPECTED_RSF_FEATURE_N)
print("CPU线程数：", N_JOBS)
print("实时日志：", LIVE_PROGRESS_FILE)
print("进程PID：", os.getpid())

if RUN_STAGE == "baseline":
    run_baseline_stage()
elif RUN_STAGE == "tune":
    run_tuning_stage()
else:
    run_final_stage()
