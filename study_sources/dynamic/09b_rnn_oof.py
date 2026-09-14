
from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import random
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
from sksurv.metrics import concordance_index_censored, integrated_brier_score
from torch import nn
from torch.utils.data import DataLoader, Dataset


# =============================================================================
# 1. 固定配置
# =============================================================================

PROJECT_DIR = Path(os.getenv("CKD_LSTM_PROJECT_DIR", "__CKD_WORKDIR__"))

STEP1_DIR = PROJECT_DIR / "rolling_5y_step1_new_split"
STEP2_DIR = PROJECT_DIR / "rolling_5y_step2_folds"
STEP3_DIR = PROJECT_DIR / "rolling_5y_step3_raw_features"
STEP4_DIR = PROJECT_DIR / "rolling_5y_step4_preprocessed"
STEP6_DIR = PROJECT_DIR / "rolling_5y_step6_super_landmark_data"

RUN_ROOT = PROJECT_DIR / "rolling_5y_step9b_rnn_tune_resume_v1"
TUNING_SUMMARY_FILE = RUN_ROOT / "tuning" / "rnn_tuning_summary.json"
OUTPUT_DIR = RUN_ROOT / "final_oof"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = OUTPUT_DIR / "rnn_final_oof.log"

N_SPLITS = 5
SEED = 20260730
MAX_EPOCHS = 40
PATIENCE = 6
MIN_DELTA = 1e-5
GRAD_CLIP = 5.0
NUM_WORKERS = 0
USE_AMP = True
REQUIRE_CUDA = True
LOG_EVERY_BATCHES = 100
EPS = 1e-7

# 由已完成的Step 9B调参锁定，不再修改。
BEST_TRIAL_NUMBER = 5
BEST_TUNING_MEAN_IBS = 0.0253955015845909
BEST_PARAMETERS = {
    "hidden_size": 64,
    "num_layers": 1,
    "dropout": 0.30,
    "learning_rate": 0.002,
    "weight_decay": 1e-6,
    "batch_size": 512,
    "positive_weight": 1.0,
}

EXPECTED_TOTAL_N = 31911
EXPECTED_DEVELOPMENT_N = 22337
EXPECTED_HISTORY_STEPS = 11
EXPECTED_FEATURE_N = 57
EXPECTED_STATIC_N = 16
EXPECTED_DYNAMIC_N = 40
EXPECTED_LANDMARK_N = 6
EXPECTED_FUTURE_N = 10
EXPECTED_LONG_N = 100122

LANDMARK_MONTHS = np.asarray([0, 12, 24, 36, 48, 60], dtype=np.int32)
LANDMARK_BINS = (LANDMARK_MONTHS // 6).astype(np.int64)
FUTURE_END_MONTHS = np.arange(6, 61, 6, dtype=np.float64)
IBS_TIMES = np.asarray([6, 12, 18, 24, 30, 36, 42, 48, 54, 59.999], dtype=np.float64)
TIME_STEP_YEARS = np.arange(EXPECTED_HISTORY_STEPS, dtype=np.float32) * 0.5

LAB_FEATURES = [
    "HIVRNA_log10", "CD4", "CD8", "Urea", "WBC", "PLT", "HB",
    "TC", "TG", "HDL", "LDL", "GLU", "ALT", "AST", "eGFR",
]
STATUS_FEATURES = [
    "CVD_status", "diabetes_status", "hypertension_status",
    "hypercholesterolemia_status", "HBV_status", "HCV_status",
]
METABOLIC_MED_FEATURES = [
    "antidiabetic_med", "antihypertensive_med", "antilipid_med",
]
CURRENT_ART_FEATURES = [
    "current_TDF_NNRTI_3TC_FTC", "current_TDF_PI_3TC_FTC",
    "current_nonTDF_PI", "current_BIC_FTC_TAF", "current_EVGc_FTC_TAF",
    "current_TDF_INSTI_3TC_FTC", "current_nonTDF_DTG",
    "current_nonTDF_traditional_NNRTI",
]
CUMULATIVE_ART_FEATURES = [
    "TDF_NNRTI_3TC_FTC_cum_month", "TDF_PI_3TC_FTC_cum_month",
    "nonTDF_PI_cum_month", "BIC_FTC_TAF_cum_month",
    "EVGc_FTC_TAF_cum_month", "TDF_INSTI_3TC_FTC_cum_month",
    "nonTDF_DTG_cum_month", "nonTDF_traditional_NNRTI_cum_month",
]
DYNAMIC_FEATURES = (
    LAB_FEATURES
    + STATUS_FEATURES
    + METABOLIC_MED_FEATURES
    + CURRENT_ART_FEATURES
    + CUMULATIVE_ART_FEATURES
)


# =============================================================================
# 2. 通用工具
# =============================================================================

def log(message: str) -> None:
    line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}"
    print(line, flush=True)
    with LOG_FILE.open("a", encoding="utf-8") as file:
        file.write(line + "\n")


def format_duration(seconds: float) -> str:
    seconds = int(round(max(0.0, seconds)))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}小时{minutes:02d}分{seconds:02d}秒"
    if minutes:
        return f"{minutes}分{seconds:02d}秒"
    return f"{seconds}秒"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_json(value: Any, path: Path) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


def require_files(paths: list[Path]) -> None:
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError("缺少必要输入文件：\n" + "\n".join(missing))


def build_survival_array(event: np.ndarray, time_month: np.ndarray) -> np.ndarray:
    event = np.asarray(event, dtype=bool)
    time_month = np.asarray(time_month, dtype=np.float64)
    if event.shape != time_month.shape:
        raise ValueError("事件与生存时间数组形状不一致。")
    if not np.isfinite(time_month).all() or np.any(time_month <= 0):
        raise ValueError("生存时间必须为有限正数。")
    result = np.empty(len(event), dtype=[("event", "?"), ("time", "<f8")])
    result["event"] = event
    result["time"] = time_month
    return result


def parameter_signature() -> str:
    payload = {
        "model": "MaskedSimpleRNNSurvival",
        "parameters": BEST_PARAMETERS,
        "max_epochs": MAX_EPOCHS,
        "patience": PATIENCE,
        "landmarks": LANDMARK_MONTHS.tolist(),
        "future_end_months": FUTURE_END_MONTHS.tolist(),
        "version": "step9b_final_oof_clean_v3",
    }
    text = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def autocast_context(device: torch.device, enabled: bool):
    try:
        return torch.amp.autocast(device_type=device.type, enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.autocast(enabled=enabled)


def make_grad_scaler(enabled: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def verify_tuning_summary() -> None:
    """若调参摘要存在，强制核对当前锁定参数。"""
    if not TUNING_SUMMARY_FILE.exists():
        log("未找到调参摘要JSON；使用脚本内已锁定的Trial 5参数。")
        return

    summary = json.loads(TUNING_SUMMARY_FILE.read_text(encoding="utf-8"))
    observed_trial = int(summary["best_trial_number"])
    observed_ibs = float(summary["best_mean_ibs"])
    observed_parameters = summary["best_parameters"]

    normalized = {
        "hidden_size": int(observed_parameters["hidden_size"]),
        "num_layers": int(observed_parameters["num_layers"]),
        "dropout": float(observed_parameters["dropout"]),
        "learning_rate": float(observed_parameters["learning_rate"]),
        "weight_decay": float(observed_parameters["weight_decay"]),
        "batch_size": int(observed_parameters["batch_size"]),
        "positive_weight": float(observed_parameters["positive_weight"]),
    }

    if observed_trial != BEST_TRIAL_NUMBER:
        raise ValueError(f"调参摘要最佳Trial={observed_trial}，脚本锁定为{BEST_TRIAL_NUMBER}。")
    if not math.isclose(observed_ibs, BEST_TUNING_MEAN_IBS, rel_tol=0, abs_tol=1e-10):
        raise ValueError("调参摘要最佳IBS与脚本锁定值不一致。")
    if normalized != BEST_PARAMETERS:
        raise ValueError("调参摘要最佳参数与脚本锁定参数不一致。")

    log("调参摘要核对通过：Trial 5，五折平均IBS=0.025396。")


# =============================================================================
# 3. 共同数据
# =============================================================================

@dataclass
class CommonData:
    development_idx: np.ndarray
    development_fold_id: np.ndarray
    sequence_row_mask: np.ndarray
    prediction_origin_mask: np.ndarray
    future_event_matrix: np.ndarray
    future_at_risk_mask: np.ndarray
    continuous_raw: np.ndarray
    age_continuous_index: int
    long_row_index_map: np.ndarray
    fold_id_long: np.ndarray
    landmark_month_long: np.ndarray
    y_long: np.ndarray


def load_common_data() -> CommonData:
    files = [
        STEP1_DIR / "development_idx.npy",
        STEP1_DIR / "prediction_origin_mask.npy",
        STEP1_DIR / "future_event_matrix.npy",
        STEP1_DIR / "future_at_risk_mask.npy",
        STEP1_DIR / "landmark_months.npy",
        STEP1_DIR / "landmark_bins.npy",
        STEP2_DIR / "development_fold_id.npy",
        STEP3_DIR / "continuous_raw_0_60.npy",
        STEP3_DIR / "feature_groups.json",
        STEP4_DIR / "development_idx.npy",
        STEP4_DIR / "development_fold_id.npy",
        STEP4_DIR / "sequence_row_mask_development.npy",
        STEP6_DIR / "development_long_row_index_map.npy",
        STEP6_DIR / "development_fold_id_long.npy",
        STEP6_DIR / "development_landmark_month_long.npy",
        STEP6_DIR / "development_analysis_time_month.npy",
        STEP6_DIR / "development_event_within_60m.npy",
    ]
    for fold_id in range(N_SPLITS):
        fold_dir = STEP4_DIR / f"fold_{fold_id}"
        files += [
            fold_dir / "X_development.npy",
            fold_dir / "preprocessor.joblib",
            fold_dir / "feature_names.csv",
        ]
    require_files(files)

    development_idx_step1 = np.load(STEP1_DIR / "development_idx.npy").astype(np.int32)
    development_idx = np.load(STEP4_DIR / "development_idx.npy").astype(np.int32)
    fold_step2 = np.load(STEP2_DIR / "development_fold_id.npy").astype(np.int8)
    development_fold_id = np.load(STEP4_DIR / "development_fold_id.npy").astype(np.int8)

    if not np.array_equal(development_idx_step1, development_idx):
        raise ValueError("Step 1和Step 4的development_idx顺序不一致。")
    if not np.array_equal(fold_step2, development_fold_id):
        raise ValueError("Step 2和Step 4的固定五折不一致。")
    if development_idx.shape != (EXPECTED_DEVELOPMENT_N,):
        raise ValueError(f"开发集患者数错误：{development_idx.shape}")
    if not np.array_equal(np.sort(np.unique(development_fold_id)), np.arange(N_SPLITS)):
        raise ValueError("开发集五折编号不是0～4。")

    sequence_row_mask = np.load(
        STEP4_DIR / "sequence_row_mask_development.npy"
    ).astype(bool)
    if sequence_row_mask.shape != (EXPECTED_DEVELOPMENT_N, EXPECTED_HISTORY_STEPS):
        raise ValueError("sequence_row_mask_development形状错误。")
    if (~sequence_row_mask.any(axis=1)).any():
        raise ValueError("部分开发集患者没有任何真实历史时间行。")

    landmark_months = np.load(STEP1_DIR / "landmark_months.npy").astype(np.int32)
    landmark_bins = np.load(STEP1_DIR / "landmark_bins.npy").astype(np.int64)
    if not np.array_equal(landmark_months, LANDMARK_MONTHS):
        raise ValueError("Step 1的Landmark月份不一致。")
    if not np.array_equal(landmark_bins, LANDMARK_BINS):
        raise ValueError("Step 1的Landmark时间行号不一致。")

    prediction_origin_all = np.load(
        STEP1_DIR / "prediction_origin_mask.npy", mmap_mode="r"
    )
    future_event_all = np.load(
        STEP1_DIR / "future_event_matrix.npy", mmap_mode="r"
    )
    future_at_risk_all = np.load(
        STEP1_DIR / "future_at_risk_mask.npy", mmap_mode="r"
    )

    if prediction_origin_all.shape != (EXPECTED_TOTAL_N, EXPECTED_LANDMARK_N):
        raise ValueError("prediction_origin_mask形状错误。")
    if future_event_all.shape != (EXPECTED_TOTAL_N, EXPECTED_LANDMARK_N, EXPECTED_FUTURE_N):
        raise ValueError("future_event_matrix形状错误。")
    if future_at_risk_all.shape != (EXPECTED_TOTAL_N, EXPECTED_LANDMARK_N, EXPECTED_FUTURE_N):
        raise ValueError("future_at_risk_mask形状错误。")

    prediction_origin_mask = np.asarray(prediction_origin_all[development_idx], dtype=bool)
    future_event_matrix = np.asarray(future_event_all[development_idx], dtype=np.float32)
    future_at_risk_mask = np.asarray(future_at_risk_all[development_idx], dtype=bool)

    if int(prediction_origin_mask.sum()) != EXPECTED_LONG_N:
        raise ValueError("开发集有效患者-Landmark记录数不是100122。")
    if np.any(future_event_matrix.astype(bool) & ~future_at_risk_mask):
        raise ValueError("存在事件标签为1但风险掩码为0的区间。")
    if np.any(future_event_matrix.sum(axis=2) > 1):
        raise ValueError("部分患者-Landmark存在多个事件区间。")

    long_row_index_map = np.load(
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

    if long_row_index_map.shape != (EXPECTED_DEVELOPMENT_N, EXPECTED_LANDMARK_N):
        raise ValueError("development_long_row_index_map形状错误。")
    if not np.array_equal(prediction_origin_mask, long_row_index_map >= 0):
        raise ValueError("Step 1有效预测起点与Step 6长记录映射不一致。")
    valid_long_idx = long_row_index_map[long_row_index_map >= 0]
    if not np.array_equal(np.sort(valid_long_idx), np.arange(EXPECTED_LONG_N)):
        raise ValueError("Step 6长行号不是0～100121的完整排列。")
    if len(analysis_time_month) != EXPECTED_LONG_N:
        raise ValueError("Step 6分析时间记录数错误。")
    if np.any((analysis_time_month <= 0) | (analysis_time_month > 60.0001)):
        raise ValueError("Step 6分析时间不在(0,60]个月。")

    feature_groups = json.loads(
        (STEP3_DIR / "feature_groups.json").read_text(encoding="utf-8")
    )
    continuous_vars = list(feature_groups["continuous_vars"])
    if "Age" not in continuous_vars:
        raise ValueError("Step 3连续变量中缺少Age。")

    continuous_raw = np.load(
        STEP3_DIR / "continuous_raw_0_60.npy", mmap_mode="r"
    )

    return CommonData(
        development_idx=development_idx,
        development_fold_id=development_fold_id,
        sequence_row_mask=sequence_row_mask,
        prediction_origin_mask=prediction_origin_mask,
        future_event_matrix=future_event_matrix,
        future_at_risk_mask=future_at_risk_mask,
        continuous_raw=continuous_raw,
        age_continuous_index=continuous_vars.index("Age"),
        long_row_index_map=long_row_index_map,
        fold_id_long=fold_id_long,
        landmark_month_long=landmark_month_long,
        y_long=build_survival_array(event_within_60m, analysis_time_month),
    )


# =============================================================================
# 4. 数据集与模型
# =============================================================================

class PatientLandmarkDataset(Dataset):
    def __init__(
        self,
        sample_pairs: np.ndarray,
        x_development: np.ndarray,
        row_mask: np.ndarray,
        static_baseline: np.ndarray,
        baseline_age: np.ndarray,
        event_matrix: np.ndarray,
        at_risk_mask: np.ndarray,
        dynamic_indices: np.ndarray,
        age_mean: float,
        age_scale: float,
    ):
        self.sample_pairs = np.asarray(sample_pairs, dtype=np.int32)
        self.x_development = x_development
        self.row_mask = row_mask
        self.static_baseline = static_baseline
        self.baseline_age = baseline_age
        self.event_matrix = event_matrix
        self.at_risk_mask = at_risk_mask
        self.dynamic_indices = np.asarray(dynamic_indices, dtype=np.int64)
        self.age_mean = float(age_mean)
        self.age_scale = float(age_scale)

    def __len__(self) -> int:
        return len(self.sample_pairs)

    def __getitem__(self, index: int) -> dict[str, np.ndarray]:
        patient_local, landmark_index = self.sample_pairs[index]
        landmark_month = float(LANDMARK_MONTHS[landmark_index])

        return {
            "dynamic_sequence": np.asarray(
                self.x_development[patient_local][:, self.dynamic_indices],
                dtype=np.float32,
            ),
            "row_mask": np.asarray(self.row_mask[patient_local], dtype=np.bool_),
            "static_baseline": np.asarray(
                self.static_baseline[patient_local], dtype=np.float32
            ),
            "age_at_landmark": np.float32(
                (float(self.baseline_age[patient_local]) + landmark_month / 12.0 - self.age_mean)
                / self.age_scale
            ),
            "landmark_normalized": np.float32(landmark_month / 60.0),
            "landmark_bin": np.int64(LANDMARK_BINS[landmark_index]),
            "event_target": np.asarray(
                self.event_matrix[patient_local, landmark_index], dtype=np.float32
            ),
            "at_risk_mask": np.asarray(
                self.at_risk_mask[patient_local, landmark_index], dtype=np.float32
            ),
            "patient_local": np.int64(patient_local),
            "landmark_index": np.int64(landmark_index),
        }


class MaskedSimpleRNNSurvival(nn.Module):
    def __init__(
        self,
        dynamic_n: int,
        static_n: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        future_n: int,
    ):
        super().__init__()
        self.dynamic_n = int(dynamic_n)
        self.hidden_size = int(hidden_size)
        self.num_layers = int(num_layers)

        self.rnn_cells = nn.ModuleList(
            [
                nn.RNNCell(
                    input_size=(self.dynamic_n + 1 if layer == 0 else self.hidden_size),
                    hidden_size=self.hidden_size,
                    nonlinearity="tanh",
                )
                for layer in range(self.num_layers)
            ]
        )
        self.dropout = nn.Dropout(float(dropout))
        head_input_n = self.hidden_size + int(static_n) + 2
        self.head = nn.Sequential(
            nn.LayerNorm(head_input_n),
            nn.Linear(head_input_n, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.hidden_size, int(future_n)),
        )

    def forward(
        self,
        dynamic_sequence: torch.Tensor,
        row_mask: torch.Tensor,
        static_baseline: torch.Tensor,
        age_at_landmark: torch.Tensor,
        landmark_normalized: torch.Tensor,
        landmark_bin: torch.Tensor,
    ) -> torch.Tensor:
        batch_n, time_n, dynamic_n = dynamic_sequence.shape
        if dynamic_n != self.dynamic_n:
            raise ValueError("模型收到的动态特征数不正确。")

        hidden = [
            torch.zeros(
                batch_n,
                self.hidden_size,
                dtype=dynamic_sequence.dtype,
                device=dynamic_sequence.device,
            )
            for _ in range(self.num_layers)
        ]
        time_values = torch.arange(time_n, device=dynamic_sequence.device).float()
        time_values = time_values / float(max(time_n - 1, 1))

        for step in range(time_n):
            active = (row_mask[:, step] & (step <= landmark_bin)).unsqueeze(1)
            layer_input = torch.cat(
                [dynamic_sequence[:, step, :], time_values[step].expand(batch_n, 1)],
                dim=1,
            )
            for layer, cell in enumerate(self.rnn_cells):
                candidate = cell(layer_input, hidden[layer])
                hidden[layer] = torch.where(active, candidate, hidden[layer])
                layer_input = hidden[layer]
                if layer < self.num_layers - 1:
                    layer_input = self.dropout(layer_input)

        context = torch.cat(
            [
                hidden[-1],
                static_baseline,
                age_at_landmark.unsqueeze(1),
                landmark_normalized.unsqueeze(1),
            ],
            dim=1,
        )
        return self.head(context)


class MaskedDiscreteTimeNLL(nn.Module):
    def __init__(self, positive_weight: float):
        super().__init__()
        self.register_buffer(
            "positive_weight", torch.tensor(float(positive_weight), dtype=torch.float32)
        )

    def forward(
        self,
        logits: torch.Tensor,
        event_target: torch.Tensor,
        at_risk_mask: torch.Tensor,
    ) -> torch.Tensor:
        loss = nn.functional.binary_cross_entropy_with_logits(
            logits,
            event_target,
            reduction="none",
            pos_weight=self.positive_weight,
        )
        return (loss * at_risk_mask).sum() / at_risk_mask.sum().clamp_min(1.0)


# =============================================================================
# 5. 每折数据准备
# =============================================================================

@dataclass
class FoldData:
    fold_id: int
    x_development: np.ndarray
    train_dataset: PatientLandmarkDataset
    validation_dataset: PatientLandmarkDataset
    train_long_idx: np.ndarray
    validation_long_idx: np.ndarray
    static_features: list[str]


def prepare_fold_data(common: CommonData, fold_id: int) -> FoldData:
    fold_dir = STEP4_DIR / f"fold_{fold_id}"
    x_development = np.load(fold_dir / "X_development.npy", mmap_mode="r")
    feature_names = pd.read_csv(
        fold_dir / "feature_names.csv", encoding="utf-8-sig"
    )["feature_name"].astype(str).tolist()
    preprocessor = joblib.load(fold_dir / "preprocessor.joblib")

    expected_shape = (
        EXPECTED_DEVELOPMENT_N,
        EXPECTED_HISTORY_STEPS,
        EXPECTED_FEATURE_N,
    )
    if x_development.shape != expected_shape:
        raise ValueError(f"第{fold_id}折X_development形状错误：{x_development.shape}")
    if len(feature_names) != EXPECTED_FEATURE_N or len(set(feature_names)) != EXPECTED_FEATURE_N:
        raise ValueError(f"第{fold_id}折特征名称数量或唯一性错误。")

    # 只审计空缺时间行；不把整个mmap张量复制到内存。
    if not np.all(np.asarray(x_development[~common.sequence_row_mask]) == 0.0):
        raise ValueError(f"第{fold_id}折空缺时间行不是全0。")

    onehot_static = [
        name
        for name in feature_names
        if name.startswith(("Sex_", "Marriage_", "Course_", "WHOstage_"))
    ]
    static_features = ["BMI", "Oppinfection", *onehot_static]
    if len(static_features) != EXPECTED_STATIC_N:
        raise ValueError(f"第{fold_id}折静态特征数为{len(static_features)}，应为16。")
    if len(DYNAMIC_FEATURES) != EXPECTED_DYNAMIC_N:
        raise ValueError("固定动态特征列表不是40项。")

    required = {"Age", *static_features, *DYNAMIC_FEATURES}
    missing = sorted(required - set(feature_names))
    if missing:
        raise ValueError(f"第{fold_id}折缺少特征：{missing}")
    if required != set(feature_names):
        extra = sorted(set(feature_names) - required)
        raise ValueError(f"第{fold_id}折存在未分组特征：{extra}")

    feature_index = {name: index for index, name in enumerate(feature_names)}
    static_indices = np.asarray([feature_index[name] for name in static_features], dtype=np.int64)
    dynamic_indices = np.asarray([feature_index[name] for name in DYNAMIC_FEATURES], dtype=np.int64)

    first_step = np.argmax(common.sequence_row_mask, axis=1).astype(np.int64)
    patient_position = np.arange(EXPECTED_DEVELOPMENT_N, dtype=np.int64)
    static_baseline = np.asarray(
        x_development[patient_position, first_step][:, static_indices], dtype=np.float32
    )

    age_first = np.asarray(
        common.continuous_raw[
            common.development_idx,
            first_step,
            common.age_continuous_index,
        ],
        dtype=np.float32,
    )
    baseline_age = age_first - TIME_STEP_YEARS[first_step]

    age_mean = float(preprocessor["scaler"].mean_[common.age_continuous_index])
    age_scale = float(preprocessor["scaler"].scale_[common.age_continuous_index])
    if not np.isfinite(age_mean) or not np.isfinite(age_scale) or age_scale <= 0:
        raise ValueError(f"第{fold_id}折Age标准化参数无效。")

    train_patient = common.development_fold_id != fold_id
    validation_patient = common.development_fold_id == fold_id
    train_pairs = np.argwhere(common.prediction_origin_mask & train_patient[:, None]).astype(np.int32)
    validation_pairs = np.argwhere(
        common.prediction_origin_mask & validation_patient[:, None]
    ).astype(np.int32)

    train_long_idx = common.long_row_index_map[train_pairs[:, 0], train_pairs[:, 1]]
    validation_long_idx = common.long_row_index_map[
        validation_pairs[:, 0], validation_pairs[:, 1]
    ]
    if np.any(train_long_idx < 0) or np.any(validation_long_idx < 0):
        raise ValueError(f"第{fold_id}折存在无效长记录索引。")
    if not np.all(common.fold_id_long[validation_long_idx] == fold_id):
        raise ValueError(f"第{fold_id}折验证长记录的折编号错误。")

    dataset_args = dict(
        x_development=x_development,
        row_mask=common.sequence_row_mask,
        static_baseline=static_baseline,
        baseline_age=baseline_age,
        event_matrix=common.future_event_matrix,
        at_risk_mask=common.future_at_risk_mask,
        dynamic_indices=dynamic_indices,
        age_mean=age_mean,
        age_scale=age_scale,
    )
    return FoldData(
        fold_id=fold_id,
        x_development=x_development,
        train_dataset=PatientLandmarkDataset(train_pairs, **dataset_args),
        validation_dataset=PatientLandmarkDataset(validation_pairs, **dataset_args),
        train_long_idx=train_long_idx.astype(np.int32),
        validation_long_idx=validation_long_idx.astype(np.int32),
        static_features=static_features,
    )


# =============================================================================
# 6. 训练与预测
# =============================================================================

def move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "dynamic_sequence": batch["dynamic_sequence"].to(device, torch.float32, non_blocking=True),
        "row_mask": batch["row_mask"].to(device, torch.bool, non_blocking=True),
        "static_baseline": batch["static_baseline"].to(device, torch.float32, non_blocking=True),
        "age_at_landmark": batch["age_at_landmark"].to(device, torch.float32, non_blocking=True),
        "landmark_normalized": batch["landmark_normalized"].to(device, torch.float32, non_blocking=True),
        "landmark_bin": batch["landmark_bin"].to(device, torch.long, non_blocking=True),
        "event_target": batch["event_target"].to(device, torch.float32, non_blocking=True),
        "at_risk_mask": batch["at_risk_mask"].to(device, torch.float32, non_blocking=True),
        "patient_local": batch["patient_local"],
        "landmark_index": batch["landmark_index"],
    }


def forward_model(model: nn.Module, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    return model(
        dynamic_sequence=batch["dynamic_sequence"],
        row_mask=batch["row_mask"],
        static_baseline=batch["static_baseline"],
        age_at_landmark=batch["age_at_landmark"],
        landmark_normalized=batch["landmark_normalized"],
        landmark_bin=batch["landmark_bin"],
    )


def make_loaders(
    fold_data: FoldData,
    device: torch.device,
    seed: int,
) -> tuple[DataLoader, DataLoader]:
    generator = torch.Generator().manual_seed(seed)
    common_args = dict(
        batch_size=BEST_PARAMETERS["batch_size"],
        num_workers=NUM_WORKERS,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )
    train_loader = DataLoader(
        fold_data.train_dataset,
        shuffle=True,
        generator=generator,
        **common_args,
    )
    validation_loader = DataLoader(
        fold_data.validation_dataset,
        shuffle=False,
        **common_args,
    )
    return train_loader, validation_loader


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    scaler: Any,
    device: torch.device,
    amp_enabled: bool,
    epoch: int,
) -> float:
    model.train()
    weighted_loss = 0.0
    valid_intervals = 0.0

    for batch_index, raw_batch in enumerate(loader, start=1):
        batch = move_batch(raw_batch, device)
        optimizer.zero_grad(set_to_none=True)

        with autocast_context(device, amp_enabled):
            logits = forward_model(model, batch)
            loss = loss_fn(logits, batch["event_target"], batch["at_risk_mask"])
        if not torch.isfinite(loss):
            raise FloatingPointError("训练损失不是有限数。")

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        gradient_norm = nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        if not torch.isfinite(gradient_norm):
            raise FloatingPointError("梯度范数不是有限数。")
        scaler.step(optimizer)
        scaler.update()

        interval_n = float(batch["at_risk_mask"].sum().item())
        weighted_loss += float(loss.item()) * interval_n
        valid_intervals += interval_n

        if batch_index == 1 or batch_index % LOG_EVERY_BATCHES == 0 or batch_index == len(loader):
            log(
                f"Epoch {epoch}/{MAX_EPOCHS} | 批次{batch_index}/{len(loader)} | "
                f"训练NLL={weighted_loss / max(valid_intervals, 1.0):.6f}"
            )

    return weighted_loss / max(valid_intervals, 1.0)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    amp_enabled: bool,
    return_predictions: bool,
) -> tuple[float, dict[str, np.ndarray] | None]:
    model.eval()
    weighted_loss = 0.0
    valid_intervals = 0.0
    hazard_batches: list[np.ndarray] = []
    patient_batches: list[np.ndarray] = []
    landmark_batches: list[np.ndarray] = []

    for raw_batch in loader:
        batch = move_batch(raw_batch, device)
        with autocast_context(device, amp_enabled):
            logits = forward_model(model, batch)
            loss = loss_fn(logits, batch["event_target"], batch["at_risk_mask"])
        if not torch.isfinite(loss):
            raise FloatingPointError("验证损失不是有限数。")

        interval_n = float(batch["at_risk_mask"].sum().item())
        weighted_loss += float(loss.item()) * interval_n
        valid_intervals += interval_n

        if return_predictions:
            hazard_batches.append(torch.sigmoid(logits.float()).cpu().numpy().astype(np.float32))
            patient_batches.append(batch["patient_local"].numpy().astype(np.int32))
            landmark_batches.append(batch["landmark_index"].numpy().astype(np.int8))

    nll = weighted_loss / max(valid_intervals, 1.0)
    if not return_predictions:
        return nll, None

    return nll, {
        "hazard": np.concatenate(hazard_batches),
        "patient_local": np.concatenate(patient_batches),
        "landmark_index": np.concatenate(landmark_batches),
    }


@dataclass
class FitResult:
    state_dict: dict[str, torch.Tensor]
    optimizer_state_dict: dict[str, Any]
    history: pd.DataFrame
    best_epoch: int
    best_validation_nll: float
    reloaded_validation_nll: float
    hazard: np.ndarray
    survival: np.ndarray
    risk: np.ndarray
    patient_local: np.ndarray
    landmark_index: np.ndarray
    parameter_n: int
    elapsed_seconds: float


def fit_fold(
    fold_data: FoldData,
    device: torch.device,
    amp_enabled: bool,
    seed: int,
) -> FitResult:
    set_seed(seed)
    train_loader, validation_loader = make_loaders(fold_data, device, seed)

    model = MaskedSimpleRNNSurvival(
        dynamic_n=EXPECTED_DYNAMIC_N,
        static_n=EXPECTED_STATIC_N,
        hidden_size=BEST_PARAMETERS["hidden_size"],
        num_layers=BEST_PARAMETERS["num_layers"],
        dropout=BEST_PARAMETERS["dropout"],
        future_n=EXPECTED_FUTURE_N,
    ).to(device)
    loss_fn = MaskedDiscreteTimeNLL(BEST_PARAMETERS["positive_weight"]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=BEST_PARAMETERS["learning_rate"],
        weight_decay=BEST_PARAMETERS["weight_decay"],
    )
    scaler = make_grad_scaler(amp_enabled)

    best_nll = math.inf
    best_epoch = -1
    best_state: dict[str, torch.Tensor] | None = None
    no_improvement = 0
    history: list[dict[str, float | int]] = []
    start = time.time()

    for epoch in range(1, MAX_EPOCHS + 1):
        train_nll = train_epoch(
            model, train_loader, optimizer, loss_fn, scaler, device, amp_enabled, epoch
        )
        validation_nll, _ = evaluate(
            model, validation_loader, loss_fn, device, amp_enabled, False
        )
        history.append(
            {
                "epoch": epoch,
                "train_nll": train_nll,
                "validation_nll": validation_nll,
                "elapsed_seconds": time.time() - start,
            }
        )
        log(
            f"Epoch {epoch}/{MAX_EPOCHS}完成 | 训练NLL={train_nll:.6f} | "
            f"验证NLL={validation_nll:.6f}"
        )

        if validation_nll < best_nll - MIN_DELTA:
            best_nll = float(validation_nll)
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
            no_improvement = 0
        else:
            no_improvement += 1
        if no_improvement >= PATIENCE:
            log(f"早停：连续{PATIENCE}个Epoch未改善。")
            break

    if best_state is None:
        raise RuntimeError("未获得有效最佳模型状态。")

    model.load_state_dict(best_state)
    reloaded_nll, prediction = evaluate(
        model, validation_loader, loss_fn, device, amp_enabled, True
    )
    assert prediction is not None

    hazard = prediction["hazard"]
    survival = np.cumprod(
        1.0 - np.clip(hazard.astype(np.float64), EPS, 1.0 - EPS), axis=1
    ).astype(np.float32)
    risk = (1.0 - survival).astype(np.float32)

    if hazard.shape != (len(fold_data.validation_dataset), EXPECTED_FUTURE_N):
        raise ValueError("验证条件风险形状错误。")
    if not np.isfinite(hazard).all() or np.any((hazard < 0) | (hazard > 1)):
        raise ValueError("验证条件风险存在无效值。")
    if np.any(np.diff(survival.astype(np.float64), axis=1) > 1e-7):
        raise ValueError("验证生存概率不单调下降。")
    if np.any(np.diff(risk.astype(np.float64), axis=1) < -1e-7):
        raise ValueError("验证累积风险不单调上升。")

    return FitResult(
        state_dict=best_state,
        optimizer_state_dict=optimizer.state_dict(),
        history=pd.DataFrame(history),
        best_epoch=best_epoch,
        best_validation_nll=best_nll,
        reloaded_validation_nll=float(reloaded_nll),
        hazard=hazard,
        survival=survival,
        risk=risk,
        patient_local=prediction["patient_local"],
        landmark_index=prediction["landmark_index"],
        parameter_n=sum(parameter.numel() for parameter in model.parameters()),
        elapsed_seconds=time.time() - start,
    )


# =============================================================================
# 7. 折内评价
# =============================================================================

def landmark_metrics(
    common: CommonData,
    fold_data: FoldData,
    survival: np.ndarray,
    risk_60m: np.ndarray,
) -> tuple[float, pd.DataFrame]:
    rows: list[dict[str, float | int]] = []
    train_landmark = common.landmark_month_long[fold_data.train_long_idx]
    validation_landmark = common.landmark_month_long[fold_data.validation_long_idx]

    for landmark_month in LANDMARK_MONTHS:
        train_mask = train_landmark == landmark_month
        validation_mask = validation_landmark == landmark_month
        y_train = common.y_long[fold_data.train_long_idx[train_mask]]
        y_validation = common.y_long[fold_data.validation_long_idx[validation_mask]]
        estimate = np.asarray(survival[validation_mask], dtype=np.float64)

        if np.max(y_train["time"]) <= IBS_TIMES[-1]:
            raise ValueError(f"Landmark {landmark_month}个月的训练参考不足以计算5年IBS。")
        if np.max(y_validation["time"]) <= IBS_TIMES[-1]:
            raise ValueError(f"Landmark {landmark_month}个月的验证集不足以计算5年IBS。")

        ibs = float(integrated_brier_score(y_train, y_validation, estimate, IBS_TIMES))
        c_index = float(
            concordance_index_censored(
                y_validation["event"],
                y_validation["time"],
                np.asarray(risk_60m[validation_mask], dtype=np.float64),
            )[0]
        )
        rows.append(
            {
                "landmark_month": int(landmark_month),
                "train_record_n": int(train_mask.sum()),
                "validation_record_n": int(validation_mask.sum()),
                "validation_event_n": int(y_validation["event"].sum()),
                "ibs_0_5_to_5y": ibs,
                "harrell_c_index": c_index,
            }
        )

    table = pd.DataFrame(rows)
    return float(table["ibs_0_5_to_5y"].mean()), table


# =============================================================================
# 8. 折级保存与续跑
# =============================================================================

def save_fold_result(
    common: CommonData,
    fold_data: FoldData,
    result: FitResult,
    metrics: pd.DataFrame,
    mean_ibs: float,
    signature: str,
) -> dict[str, Any]:
    fold_id = fold_data.fold_id
    fold_dir = OUTPUT_DIR / f"fold_{fold_id}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    expected_long = common.long_row_index_map[result.patient_local, result.landmark_index]
    if not np.array_equal(expected_long, fold_data.validation_long_idx):
        raise ValueError(f"第{fold_id}折预测顺序与Step 6长记录顺序不一致。")

    prediction_map = pd.DataFrame(
        {
            "prediction_row": np.arange(len(fold_data.validation_long_idx), dtype=np.int32),
            "long_row_index": fold_data.validation_long_idx,
            "patient_local_index": result.patient_local,
            "patient_global_index": common.development_idx[result.patient_local],
            "landmark_index": result.landmark_index,
            "landmark_month": LANDMARK_MONTHS[result.landmark_index],
        }
    )

    summary = {
        "fold_id": fold_id,
        "train_patient_n": int(np.sum(common.development_fold_id != fold_id)),
        "validation_patient_n": int(np.sum(common.development_fold_id == fold_id)),
        "train_origin_n": len(fold_data.train_dataset),
        "validation_origin_n": len(fold_data.validation_dataset),
        "best_epoch": result.best_epoch,
        "best_validation_nll": result.best_validation_nll,
        "reloaded_validation_nll": result.reloaded_validation_nll,
        "mean_ibs": mean_ibs,
        "mean_harrell_c_index": float(metrics["harrell_c_index"].mean()),
        "model_parameter_n": result.parameter_n,
        "elapsed_seconds": result.elapsed_seconds,
    }

    torch.save(
        {
            "model_state_dict": result.state_dict,
            "optimizer_state_dict": result.optimizer_state_dict,
            "fold_id": fold_id,
            "parameters": BEST_PARAMETERS,
            "best_epoch": result.best_epoch,
            "best_validation_nll": result.best_validation_nll,
            "static_features": fold_data.static_features,
            "dynamic_features": DYNAMIC_FEATURES,
        },
        fold_dir / "best_model.pt",
    )
    result.history.to_csv(fold_dir / "training_history.csv", index=False, encoding="utf-8-sig")
    np.save(fold_dir / "validation_hazard.npy", result.hazard)
    np.save(fold_dir / "validation_survival.npy", result.survival)
    np.save(fold_dir / "validation_risk.npy", result.risk)
    prediction_map.to_csv(
        fold_dir / "validation_prediction_map.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame([summary]).to_csv(
        fold_dir / "fold_summary.csv", index=False, encoding="utf-8-sig"
    )
    metrics.assign(fold_id=fold_id).loc[
        :, ["fold_id", *metrics.columns]
    ].to_csv(
        fold_dir / "validation_landmark_metrics.csv", index=False, encoding="utf-8-sig"
    )
    save_json(
        {
            "fold_id": fold_id,
            "parameter_signature": signature,
            "completed_at": datetime.now().isoformat(timespec="seconds"),
        },
        fold_dir / "completed.json",
    )
    return summary


def load_completed_fold(
    fold_data: FoldData,
    signature: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any], pd.DataFrame] | None:
    fold_dir = OUTPUT_DIR / f"fold_{fold_data.fold_id}"
    completed_file = fold_dir / "completed.json"
    if not completed_file.exists():
        return None

    completed = json.loads(completed_file.read_text(encoding="utf-8"))
    if completed.get("parameter_signature") != signature:
        raise ValueError(
            f"第{fold_data.fold_id}折已有结果，但参数签名不同。请删除目录：{fold_dir}"
        )

    hazard = np.load(fold_dir / "validation_hazard.npy").astype(np.float32)
    survival = np.load(fold_dir / "validation_survival.npy").astype(np.float32)
    risk = np.load(fold_dir / "validation_risk.npy").astype(np.float32)
    expected_shape = (len(fold_data.validation_dataset), EXPECTED_FUTURE_N)
    if hazard.shape != expected_shape or survival.shape != expected_shape or risk.shape != expected_shape:
        raise ValueError(f"第{fold_data.fold_id}折已保存预测形状错误。")

    summary = pd.read_csv(
        fold_dir / "fold_summary.csv", encoding="utf-8-sig"
    ).iloc[0].to_dict()
    metrics = pd.read_csv(
        fold_dir / "validation_landmark_metrics.csv", encoding="utf-8-sig"
    )
    if "fold_id" not in metrics.columns:
        metrics.insert(0, "fold_id", int(fold_data.fold_id))
    elif not (
        metrics["fold_id"].astype(int) == int(fold_data.fold_id)
    ).all():
        raise ValueError(
            f"第{fold_data.fold_id}折已保存指标表中的fold_id不一致。"
        )
    return hazard, survival, risk, summary, metrics


# =============================================================================
# 9. 汇总OOF
# =============================================================================

def build_patient_arrays(
    common: CommonData,
    hazard_long: np.ndarray,
    survival_long: np.ndarray,
    risk_long: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    shape = (EXPECTED_DEVELOPMENT_N, EXPECTED_LANDMARK_N, EXPECTED_FUTURE_N)
    hazard = np.full(shape, np.nan, dtype=np.float32)
    survival = np.full(shape, np.nan, dtype=np.float32)
    risk = np.full(shape, np.nan, dtype=np.float32)

    patient, landmark = np.where(common.long_row_index_map >= 0)
    long_index = common.long_row_index_map[patient, landmark]
    hazard[patient, landmark] = hazard_long[long_index]
    survival[patient, landmark] = survival_long[long_index]
    risk[patient, landmark] = risk_long[long_index]

    if np.isnan(risk[common.prediction_origin_mask]).any():
        raise ValueError("患者级有效预测起点存在缺失。")
    if not np.isnan(risk[~common.prediction_origin_mask]).all():
        raise ValueError("患者级无效预测起点没有保持NaN。")
    return hazard, survival, risk


def run() -> None:
    LOG_FILE.write_text("", encoding="utf-8")
    verify_tuning_summary()
    common = load_common_data()

    torch.set_num_threads(min(8, os.cpu_count() or 1))
    set_seed(SEED)
    if REQUIRE_CUDA and not torch.cuda.is_available():
        raise RuntimeError("未检测到CUDA；本步骤要求GPU运行。")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = bool(USE_AMP and device.type == "cuda")
    if device.type == "cuda":
        torch.cuda.empty_cache()

    print("\n" + "=" * 88)
    print("Step 9B v4：普通RNN最佳参数正式五折OOF")
    print("=" * 88)
    print("PyTorch版本：", torch.__version__)
    print("运行设备：", device)
    if device.type == "cuda":
        print("GPU：", torch.cuda.get_device_name(0))
        print(
            "GPU显存：",
            f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GiB",
        )
    print("AMP：", amp_enabled)
    print("最佳Trial：", BEST_TRIAL_NUMBER)
    print("最佳调参IBS：", f"{BEST_TUNING_MEAN_IBS:.6f}")
    print("最佳参数：", BEST_PARAMETERS)
    print("开发集患者数：", EXPECTED_DEVELOPMENT_N)
    print("有效患者-Landmark记录数：", EXPECTED_LONG_N)
    print("固定五折：是")
    print("折级续跑：是")
    print("锁定测试集：不参与")
    print("输出目录：", OUTPUT_DIR)
    print("=" * 88)

    signature = parameter_signature()
    save_json(
        {
            "stage": "Step9B_RNN_final_5fold_OOF_clean_v4",
            "best_trial_number": BEST_TRIAL_NUMBER,
            "best_tuning_mean_ibs": BEST_TUNING_MEAN_IBS,
            "parameters": BEST_PARAMETERS,
            "parameter_signature": signature,
            "max_epochs": MAX_EPOCHS,
            "patience": PATIENCE,
            "locked_test_used": False,
        },
        OUTPUT_DIR / "stage_configuration.json",
    )

    hazard_long = np.full((EXPECTED_LONG_N, EXPECTED_FUTURE_N), np.nan, dtype=np.float32)
    survival_long = np.full_like(hazard_long, np.nan)
    risk_long = np.full_like(hazard_long, np.nan)
    fold_summaries: list[dict[str, Any]] = []
    fold_metrics: list[pd.DataFrame] = []
    static_features_reference: list[str] | None = None
    total_start = time.time()

    for fold_id in range(N_SPLITS):
        fold_data = prepare_fold_data(common, fold_id)
        if static_features_reference is None:
            static_features_reference = fold_data.static_features
        elif static_features_reference != fold_data.static_features:
            raise ValueError("不同折的静态特征名称或顺序不一致。")

        cached = load_completed_fold(fold_data, signature)
        if cached is not None:
            hazard, survival, risk, summary, metrics = cached
            log(f"第{fold_id + 1}/{N_SPLITS}折已完成，直接读取检查点。")
        else:
            log(
                f"第{fold_id + 1}/{N_SPLITS}折开始 | "
                f"训练样本={len(fold_data.train_dataset)} | "
                f"验证样本={len(fold_data.validation_dataset)}"
            )
            result = fit_fold(
                fold_data,
                device=device,
                amp_enabled=amp_enabled,
                seed=SEED + 10000 + fold_id,
            )
            mean_ibs, metrics = landmark_metrics(
                common,
                fold_data,
                survival=result.survival,
                risk_60m=result.risk[:, -1],
            )
            summary = save_fold_result(
                common, fold_data, result, metrics, mean_ibs, signature
            )
            hazard, survival, risk = result.hazard, result.survival, result.risk
            log(
                f"第{fold_id + 1}/{N_SPLITS}折完成 | IBS={mean_ibs:.6f} | "
                f"平均C-index={metrics['harrell_c_index'].mean():.6f} | "
                f"最佳Epoch={result.best_epoch} | "
                f"耗时={format_duration(result.elapsed_seconds)}"
            )
            del result

        hazard_long[fold_data.validation_long_idx] = hazard
        survival_long[fold_data.validation_long_idx] = survival
        risk_long[fold_data.validation_long_idx] = risk
        if "fold_id" not in metrics.columns:
            metrics = metrics.copy()
            metrics.insert(0, "fold_id", int(fold_id))
        elif not (
            metrics["fold_id"].astype(int) == int(fold_id)
        ).all():
            raise ValueError(
                f"第{fold_id}折指标表中的fold_id不一致。"
            )

        fold_summaries.append(summary)
        fold_metrics.append(metrics)

        del fold_data, hazard, survival, risk
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if np.isnan(hazard_long).any() or np.isnan(survival_long).any() or np.isnan(risk_long).any():
        raise ValueError("最终长格式OOF预测存在缺失。")

    survival_violations = int(
        np.sum(np.diff(survival_long.astype(np.float64), axis=1) > 1e-7)
    )
    risk_violations = int(
        np.sum(np.diff(risk_long.astype(np.float64), axis=1) < -1e-7)
    )
    identity_difference = float(np.max(np.abs(risk_long - (1.0 - survival_long))))
    if survival_violations:
        raise ValueError("最终OOF生存概率不单调下降。")
    if risk_violations:
        raise ValueError("最终OOF累积风险不单调上升。")
    if identity_difference > 1e-6:
        raise ValueError("最终OOF风险不等于1-survival。")

    hazard_patient, survival_patient, risk_patient = build_patient_arrays(
        common, hazard_long, survival_long, risk_long
    )

    np.save(OUTPUT_DIR / "rnn_oof_hazard_long.npy", hazard_long)
    np.save(OUTPUT_DIR / "rnn_oof_survival_long.npy", survival_long)
    np.save(OUTPUT_DIR / "rnn_oof_risk_long.npy", risk_long)
    np.save(OUTPUT_DIR / "rnn_oof_risk_score_long.npy", risk_long[:, -1])
    np.save(OUTPUT_DIR / "rnn_oof_hazard.npy", hazard_patient)
    np.save(OUTPUT_DIR / "rnn_oof_survival.npy", survival_patient)
    np.save(OUTPUT_DIR / "rnn_oof_risk.npy", risk_patient)
    np.save(OUTPUT_DIR / "rnn_oof_risk_score.npy", risk_patient[:, :, -1])

    fold_summary_table = (
        pd.DataFrame(fold_summaries)
        .sort_values("fold_id")
        .reset_index(drop=True)
    )
    fold_metric_table = (
        pd.concat(fold_metrics, ignore_index=True)
        .sort_values(["fold_id", "landmark_month"])
        .reset_index(drop=True)
    )
    fold_summary_table.to_csv(
        OUTPUT_DIR / "rnn_fold_summary.csv", index=False, encoding="utf-8-sig"
    )
    fold_metric_table.to_csv(
        OUTPUT_DIR / "rnn_fold_landmark_metrics.csv", index=False, encoding="utf-8-sig"
    )

    if static_features_reference is None:
        raise RuntimeError("未获得静态特征列表。")
    feature_partition = pd.DataFrame(
        [("Age", "age_context")]
        + [(name, "baseline_static") for name in static_features_reference]
        + [(name, "dynamic_sequence") for name in DYNAMIC_FEATURES],
        columns=["feature_name", "feature_group"],
    )
    feature_partition.to_csv(
        OUTPUT_DIR / "rnn_feature_partition.csv", index=False, encoding="utf-8-sig"
    )

    total_elapsed = time.time() - total_start
    summary = {
        "stage": "Step9B_RNN_final_5fold_OOF_clean_v4",
        "best_trial_number": BEST_TRIAL_NUMBER,
        "best_tuning_mean_ibs": BEST_TUNING_MEAN_IBS,
        "parameters": BEST_PARAMETERS,
        "development_patient_n": EXPECTED_DEVELOPMENT_N,
        "development_valid_origin_n": EXPECTED_LONG_N,
        "landmark_months": LANDMARK_MONTHS.tolist(),
        "future_end_months": FUTURE_END_MONTHS.tolist(),
        "oof_long_shape": list(risk_long.shape),
        "oof_patient_shape": list(risk_patient.shape),
        "mean_fold_ibs": float(fold_summary_table["mean_ibs"].mean()),
        "mean_fold_harrell_c_index": float(
            fold_summary_table["mean_harrell_c_index"].mean()
        ),
        "survival_monotonic_violation_n": survival_violations,
        "risk_monotonic_violation_n": risk_violations,
        "risk_identity_max_difference": identity_difference,
        "total_elapsed_seconds": total_elapsed,
        "locked_test_used": False,
    }
    save_json(summary, OUTPUT_DIR / "rnn_summary.json")

    print("\n" + "=" * 88)
    print("最终RNN五折OOF完成")
    print("=" * 88)
    print("五折平均IBS：", f"{summary['mean_fold_ibs']:.6f}")
    print("五折平均C-index：", f"{summary['mean_fold_harrell_c_index']:.6f}")
    print("OOF长格式形状：", risk_long.shape)
    print("OOF患者级形状：", risk_patient.shape)
    print("生存概率单调性违反数：", survival_violations)
    print("累积风险单调性违反数：", risk_violations)
    print("总耗时：", format_duration(total_elapsed))
    print("输出目录：", OUTPUT_DIR)


if __name__ == "__main__":
    run()
