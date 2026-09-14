#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Step 12C-2：LSTM-v2 Grouped Clinical-Domain Occlusion 患者级 paired bootstrap 95% CI

目的
----
1. 不重新训练模型，不重新运行LSTM前向，不读取锁定测试集。
2. 直接读取Step12C已经保存的baseline与10个临床域group occlusion累计风险预测。
3. 完全沿用Step11/Step12B-2患者级paired bootstrap框架：
   - 每个replicate从全部22,337名开发患者中有放回抽样22,337次；
   - 同一批bootstrap患者同时用于6个正式Landmark与全部10个group；
   - 某Landmark无有效prediction origin的患者自动排除；
   - IPCW survival_train/reference始终使用该Landmark完整原始OOF风险集。
4. 对每个Landmark × group计算：
   - Uno C-index loss = baseline - occluded；
   - iAUC loss = baseline - occluded；
   - IBS increase = occluded - baseline；
   - mean absolute change in 5-year cumulative CKD risk；
   并给出患者级paired bootstrap percentile 95% CI。
5. 正值统一表示“遮挡该临床域后预测表现变差”。
6. 支持断点续跑。正式分析默认1000次；测试时可临时：
      CKD_GROUP_BOOTSTRAP_REPS=50 python Step12C2_....py

解释边界
--------
- 这是冻结LSTM的post-hoc grouped input reference occlusion，不是重新训练后的ablation。
- 不同group包含的输入通道数不同，结果表示“整个临床域被替换为训练风险集reference后的性能变化”，
  不应除以通道数解释成单通道平均重要性。
- group效应不假定可加。
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sksurv.metrics import (
    concordance_index_ipcw,
    cumulative_dynamic_auc,
    integrated_brier_score,
)
from sksurv.util import Surv


# =============================================================================
# 1. 路径与固定配置
# =============================================================================

PROJECT_DIR = Path(
    os.getenv(
        "CKD_LSTM_PROJECT_DIR",
        "__CKD_WORKDIR__",
    )
)



INTERPRETATION_ROOT = Path(
    os.getenv(
        "CKD_MODEL_INTERPRETATION_ROOT",
        str(PROJECT_DIR / "rolling_5y_model_interpretation_final_6landmarks"),
    )
)
COMMON_FIGURE_DIR = INTERPRETATION_ROOT / "FIGURES_ALL"
COMMON_LOG_DIR = INTERPRETATION_ROOT / "LOGS"
for directory in [INTERPRETATION_ROOT, COMMON_FIGURE_DIR, COMMON_LOG_DIR]:
    directory.mkdir(parents=True, exist_ok=True)
STEP4_DIR = (
    PROJECT_DIR
    / "rolling_5y_step4_preprocessed"
)

STEP6_DIR = (
    PROJECT_DIR
    / "rolling_5y_step6_super_landmark_data"
)

STEP12C_DIR = (
    INTERPRETATION_ROOT
    / "05_GROUPED_OCCLUSION"
)

STEP12C_PREDICTION_DIR = (
    STEP12C_DIR
    / "prediction_checkpoints"
)

OUTPUT_DIR = INTERPRETATION_ROOT / "06_GROUPED_BOOTSTRAP"

BOOTSTRAP_DIR = (
    OUTPUT_DIR
    / "bootstrap_checkpoints"
)

FIGURE_DIR = COMMON_FIGURE_DIR

for directory in [
    OUTPUT_DIR,
    BOOTSTRAP_DIR,
    FIGURE_DIR,
]:
    directory.mkdir(
        parents=True,
        exist_ok=True,
    )

EXPECTED_DEVELOPMENT_N = 22337
EXPECTED_LANDMARK_N = 6
EXPECTED_INTERVAL_N = 10
EXPECTED_LONG_N = 100122
EXPECTED_GROUP_N = 10
N_SPLITS = 5

LANDMARK_MONTHS = np.asarray(
    [0, 12, 24, 36, 48, 60],
    dtype=np.int32,
)

PRIMARY_LANDMARK_INDICES = np.arange(EXPECTED_LANDMARK_N, dtype=np.int64)

PRIMARY_LANDMARK_MONTHS = (
    LANDMARK_MONTHS[
        PRIMARY_LANDMARK_INDICES
    ]
)

FUTURE_END_MONTHS = np.arange(
    6,
    61,
    6,
    dtype=np.float64,
)

# 与Step11、Step12B-2完全一致。
# 第10个风险仍表示未来60月风险；数值评价时用59.999月满足sksurv要求。
METRIC_TIMES = np.asarray(
    [
        6,
        12,
        18,
        24,
        30,
        36,
        42,
        48,
        54,
        59.999,
    ],
    dtype=np.float64,
)

BOOTSTRAP_REPS = int(
    os.getenv(
        "CKD_GROUP_BOOTSTRAP_REPS",
        "1000",
    )
)

RANDOM_SEED = int(
    os.getenv(
        "CKD_GROUP_BOOTSTRAP_SEED",
        "20260812",
    )
)

BOOTSTRAP_SAVE_EVERY = int(
    os.getenv(
        "CKD_GROUP_BOOTSTRAP_SAVE_EVERY",
        "10",
    )
)

PROGRESS_EVERY = int(
    os.getenv(
        "CKD_GROUP_BOOTSTRAP_PROGRESS_EVERY",
        "25",
    )
)

BOOTSTRAP_SHARD_COUNT = int(
    os.getenv("CKD_BOOTSTRAP_SHARD_COUNT", "1")
)
BOOTSTRAP_SHARD_INDEX = int(
    os.getenv("CKD_BOOTSTRAP_SHARD_INDEX", "0")
)

MINIMUM_VALID_BOOTSTRAP_RATE = 0.80
POINT_AUDIT_TOL = 5e-8

PIPELINE_VERSION = (
    "step12c2_grouped_occlusion_"
    "patient_paired_bootstrap_v1"
)

BOOTSTRAP_METRICS = [
    "uno_c_loss",
    "iAUC_loss",
    "IBS_increase",
    "mean_abs_delta_risk5",
]

METRIC_INDEX = {
    metric_name: metric_index
    for metric_index, metric_name
    in enumerate(
        BOOTSTRAP_METRICS
    )
}

STEP12C_METRICS_FILE = (
    STEP12C_DIR
    / "grouped_occlusion_metrics.csv"
)

STEP12C_GROUP_DEFINITION_FILE = (
    STEP12C_DIR
    / "grouped_occlusion_group_definition.csv"
)

STEP12C_CHANNEL_MAPPING_FILE = (
    STEP12C_DIR
    / "grouped_occlusion_channel_mapping.csv"
)

STEP12C_BASELINE_AUDIT_FILE = (
    STEP12C_DIR
    / "grouped_occlusion_baseline_metric_audit.csv"
)

STEP12C_CHECKPOINT_METADATA_FILE = (
    STEP12C_PREDICTION_DIR
    / "checkpoint_metadata.json"
)


# =============================================================================
# 2. 通用工具
# =============================================================================

def require_files(
    paths: list[Path],
) -> None:
    missing = [
        str(path)
        for path in paths
        if not path.exists()
    ]

    if missing:
        raise FileNotFoundError(
            "以下必要输入文件不存在：\n"
            + "\n".join(
                missing
            )
        )


def save_json(
    value: Any,
    path: Path,
) -> None:
    path.write_text(
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )


def sha256_file(
    path: Path,
) -> str:
    digest = hashlib.sha256()

    with path.open(
        "rb"
    ) as file:
        while True:
            block = file.read(
                1024 * 1024
            )
            if not block:
                break
            digest.update(
                block
            )

    return digest.hexdigest()


def format_duration(
    seconds: float,
) -> str:
    seconds = max(
        0,
        int(
            round(
                float(
                    seconds
                )
            )
        ),
    )

    hours, remainder = divmod(
        seconds,
        3600,
    )

    minutes, seconds = divmod(
        remainder,
        60,
    )

    if hours:
        return (
            f"{hours}小时"
            f"{minutes:02d}分"
            f"{seconds:02d}秒"
        )

    if minutes:
        return (
            f"{minutes}分"
            f"{seconds:02d}秒"
        )

    return f"{seconds}秒"


def percentile_interval(
    values: np.ndarray,
    expected_n: int,
) -> tuple[
    float,
    float,
    float,
    int,
]:
    values = np.asarray(
        values,
        dtype=np.float64,
    )

    valid = values[
        np.isfinite(
            values
        )
    ]

    required_n = int(
        np.ceil(
            expected_n
            * MINIMUM_VALID_BOOTSTRAP_RATE
        )
    )

    if len(valid) < required_n:
        raise RuntimeError(
            f"仅获得{len(valid)}/{expected_n}"
            "个有效bootstrap估计。"
        )

    lower = float(
        np.percentile(
            valid,
            2.5,
        )
    )

    upper = float(
        np.percentile(
            valid,
            97.5,
        )
    )

    median = float(
        np.median(
            valid
        )
    )

    return (
        lower,
        upper,
        median,
        int(
            len(
                valid
            )
        ),
    )


def positive_fraction(
    values: np.ndarray,
) -> float:
    values = np.asarray(
        values,
        dtype=np.float64,
    )

    valid = values[
        np.isfinite(
            values
        )
    ]

    if len(valid) == 0:
        return np.nan

    return float(
        np.mean(
            valid > 0
        )
    )


def display_table(
    frame: pd.DataFrame,
    title: str,
) -> None:
    print(
        "\n"
        + "=" * 120
    )
    print(
        title
    )
    print(
        "=" * 120
    )

    try:
        from IPython.display import display
        display(
            frame
        )
    except ImportError:
        print(
            frame.to_string(
                index=False
            )
        )


# =============================================================================
# 3. 生存评价——与Step11/Step12B-2一致
# =============================================================================

def build_survival_array(
    event: np.ndarray,
    time_month: np.ndarray,
) -> np.ndarray:
    event = np.asarray(
        event,
        dtype=bool,
    )

    time_month = np.asarray(
        time_month,
        dtype=np.float64,
    )

    if event.shape != time_month.shape:
        raise ValueError(
            "事件和随访时间形状不一致。"
        )

    if not np.isfinite(
        time_month
    ).all():
        raise ValueError(
            "随访时间存在NaN或无穷值。"
        )

    if np.any(
        time_month <= 0
    ):
        raise ValueError(
            "Landmark后随访时间必须>0。"
        )

    return Surv.from_arrays(
        event=event,
        time=time_month,
    )


def evaluate_survival_predictions(
    survival_train: np.ndarray,
    survival_test: np.ndarray,
    risk_matrix: np.ndarray,
) -> dict[str, float]:
    risk_matrix = np.asarray(
        risk_matrix,
        dtype=np.float64,
    )

    if (
        risk_matrix.ndim != 2
        or risk_matrix.shape[1]
        != EXPECTED_INTERVAL_N
    ):
        raise ValueError(
            "risk_matrix形状非法。"
        )

    survival_matrix = (
        1.0
        - risk_matrix
    )

    c_index = float(
        concordance_index_ipcw(
            survival_train,
            survival_test,
            risk_matrix[
                :,
                -1,
            ],
            tau=float(
                METRIC_TIMES[
                    -1
                ]
            ),
        )[0]
    )

    _, mean_auc = (
        cumulative_dynamic_auc(
            survival_train,
            survival_test,
            risk_matrix,
            METRIC_TIMES,
        )
    )

    ibs = float(
        integrated_brier_score(
            survival_train,
            survival_test,
            survival_matrix,
            METRIC_TIMES,
        )
    )

    return {
        "uno_c_index_5y": (
            c_index
        ),
        "integrated_dynamic_auc": float(
            mean_auc
        ),
        "integrated_brier": (
            ibs
        ),
    }


def safe_evaluate_survival_predictions(
    survival_train: np.ndarray,
    survival_test: np.ndarray,
    risk_matrix: np.ndarray,
) -> dict[str, float] | None:
    try:
        return (
            evaluate_survival_predictions(
                survival_train,
                survival_test,
                risk_matrix,
            )
        )

    except (
        ValueError,
        ArithmeticError,
        ZeroDivisionError,
        FloatingPointError,
    ):
        return None


# =============================================================================
# 4. Step4 + Step6共同数组
# =============================================================================

def load_common_arrays() -> dict[str, np.ndarray]:
    required = [
        STEP4_DIR
        / "development_fold_id.npy",

        STEP6_DIR
        / "development_long_row_index_map.npy",

        STEP6_DIR
        / "development_local_patient_idx_long.npy",

        STEP6_DIR
        / "development_landmark_index_long.npy",

        STEP6_DIR
        / "development_analysis_time_month.npy",

        STEP6_DIR
        / "development_event_within_60m.npy",
    ]

    require_files(
        required
    )

    development_fold_id = np.load(
        STEP4_DIR
        / "development_fold_id.npy"
    ).astype(
        np.int8
    )

    row_index_map = np.load(
        STEP6_DIR
        / "development_long_row_index_map.npy"
    ).astype(
        np.int32
    )

    local_patient_idx_long = np.load(
        STEP6_DIR
        / "development_local_patient_idx_long.npy"
    ).astype(
        np.int32
    )

    landmark_index_long = np.load(
        STEP6_DIR
        / "development_landmark_index_long.npy"
    ).astype(
        np.int8
    )

    analysis_time_month = np.load(
        STEP6_DIR
        / "development_analysis_time_month.npy"
    ).astype(
        np.float64
    )

    event_within_60m = np.load(
        STEP6_DIR
        / "development_event_within_60m.npy"
    ).astype(
        bool
    )

    arrays = {
        "development_fold_id": (
            development_fold_id
        ),
        "row_index_map": (
            row_index_map
        ),
        "local_patient_idx_long": (
            local_patient_idx_long
        ),
        "landmark_index_long": (
            landmark_index_long
        ),
        "analysis_time_month": (
            analysis_time_month
        ),
        "event_within_60m": (
            event_within_60m
        ),
    }

    expected_shapes = {
        "development_fold_id": (
            EXPECTED_DEVELOPMENT_N,
        ),
        "row_index_map": (
            EXPECTED_DEVELOPMENT_N,
            EXPECTED_LANDMARK_N,
        ),
        "local_patient_idx_long": (
            EXPECTED_LONG_N,
        ),
        "landmark_index_long": (
            EXPECTED_LONG_N,
        ),
        "analysis_time_month": (
            EXPECTED_LONG_N,
        ),
        "event_within_60m": (
            EXPECTED_LONG_N,
        ),
    }

    for name, expected_shape in (
        expected_shapes.items()
    ):
        if (
            arrays[
                name
            ].shape
            != expected_shape
        ):
            raise ValueError(
                f"{name}形状="
                f"{arrays[name].shape}，"
                f"预期={expected_shape}。"
            )

    valid_map = (
        row_index_map
        >= 0
    )

    expected_valid_map = np.column_stack(
        [
            np.bincount(
                local_patient_idx_long[
                    landmark_index_long
                    == landmark_index
                ],
                minlength=(
                    EXPECTED_DEVELOPMENT_N
                ),
            )
            > 0
            for landmark_index
            in range(
                EXPECTED_LANDMARK_N
            )
        ]
    )

    if not np.array_equal(
        valid_map,
        expected_valid_map,
    ):
        raise ValueError(
            "Step6 row_index_map与"
            "long格式患者-Landmark映射不一致。"
        )

    return arrays


# =============================================================================
# 5. Step12C group定义/point-estimate结果
# =============================================================================

def load_group_tables() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    require_files(
        [
            STEP12C_METRICS_FILE,
            STEP12C_GROUP_DEFINITION_FILE,
            STEP12C_CHANNEL_MAPPING_FILE,
            STEP12C_BASELINE_AUDIT_FILE,
            STEP12C_CHECKPOINT_METADATA_FILE,
        ]
    )

    metrics_df = pd.read_csv(
        STEP12C_METRICS_FILE,
        encoding="utf-8-sig",
    )

    group_definition_df = pd.read_csv(
        STEP12C_GROUP_DEFINITION_FILE,
        encoding="utf-8-sig",
    )

    required_metric_columns = {
        "group_key",
        "group_label",
        "landmark_index",
        "landmark_month",
        "landmark_year",
        "risk_set_n",
        "future_5y_event_n",
        "occluded_dynamic_channel_n",
        "occluded_static_channel_n",
        "age_occluded",
        "total_occluded_input_channel_n",
        "clinical_features",
        "fp32_baseline_uno_c",
        "fp32_baseline_iAUC",
        "fp32_baseline_IBS",
        "occluded_uno_c",
        "occluded_iAUC",
        "occluded_IBS",
        "uno_c_loss",
        "iAUC_loss",
        "IBS_increase",
        "mean_abs_delta_risk5",
    }

    missing = sorted(
        required_metric_columns
        - set(
            metrics_df.columns
        )
    )

    if missing:
        raise ValueError(
            "Step12C grouped_occlusion_metrics.csv缺少列："
            + ", ".join(
                missing
            )
        )

    required_definition_columns = {
        "group_key",
        "group_label",
        "clinical_features",
        "dynamic_channel_n",
        "static_channel_n",
        "age_included",
        "total_input_channel_n",
    }

    missing_definition = sorted(
        required_definition_columns
        - set(
            group_definition_df.columns
        )
    )

    if missing_definition:
        raise ValueError(
            "Step12C group definition缺少列："
            + ", ".join(
                missing_definition
            )
        )

    metrics_df = metrics_df.loc[
        metrics_df[
            "landmark_index"
        ].isin(
            PRIMARY_LANDMARK_INDICES
        )
    ].copy()

    metrics_df[
        "landmark_index"
    ] = metrics_df[
        "landmark_index"
    ].astype(
        int
    )

    metrics_df[
        "landmark_month"
    ] = metrics_df[
        "landmark_month"
    ].astype(
        int
    )

    group_definition_df = (
        group_definition_df.copy()
    )

    if len(
        group_definition_df
    ) != EXPECTED_GROUP_N:
        raise ValueError(
            "Step12C group definition"
            f"应为{EXPECTED_GROUP_N}组，"
            f"实际={len(group_definition_df)}。"
        )

    if group_definition_df[
        "group_key"
    ].duplicated().any():
        raise ValueError(
            "Step12C group_key存在重复。"
        )

    group_keys = (
        group_definition_df[
            "group_key"
        ]
        .astype(
            str
        )
        .tolist()
    )

    expected_row_n = (
        len(
            PRIMARY_LANDMARK_INDICES
        )
        * EXPECTED_GROUP_N
    )

    if len(
        metrics_df
    ) != expected_row_n:
        raise ValueError(
            "Step12C主Landmark metric行数异常："
            f"{len(metrics_df)}，预期={expected_row_n}。"
        )

    for landmark_index in (
        PRIMARY_LANDMARK_INDICES
    ):
        landmark_index = int(
            landmark_index
        )

        sub = metrics_df.loc[
            metrics_df[
                "landmark_index"
            ]
            == landmark_index
        ]

        observed_keys = set(
            sub[
                "group_key"
            ].astype(
                str
            )
        )

        if observed_keys != set(
            group_keys
        ):
            raise ValueError(
                f"Landmark {LANDMARK_MONTHS[landmark_index]}月"
                "group集合与定义文件不一致。"
            )

        if sub[
            "group_key"
        ].duplicated().any():
            raise ValueError(
                f"Landmark {LANDMARK_MONTHS[landmark_index]}月"
                "存在重复group。"
            )

    # 以定义文件顺序固定所有Landmark的group顺序，防止按结果排名后顺序改变。
    group_order = {
        group_key: position
        for position, group_key
        in enumerate(
            group_keys
        )
    }

    metrics_df[
        "group_position"
    ] = (
        metrics_df[
            "group_key"
        ]
        .map(
            group_order
        )
        .astype(
            int
        )
    )

    metrics_df = metrics_df.sort_values(
        [
            "landmark_index",
            "group_position",
        ]
    ).reset_index(
        drop=True
    )

    metrics_df[
        "condition_position"
    ] = np.arange(
        len(
            metrics_df
        ),
        dtype=np.int32,
    )

    return (
        metrics_df,
        group_definition_df,
    )


# =============================================================================
# 6. 读取Step12C prediction checkpoints并拼回完整OOF风险集
# =============================================================================

class PredictionStore:
    def __init__(
        self,
        common: dict[str, np.ndarray],
        metrics_df: pd.DataFrame,
        group_definition_df: pd.DataFrame,
    ) -> None:
        self.common = common
        self.metrics_df = metrics_df
        self.group_definition_df = (
            group_definition_df
        )

        self.group_keys = (
            group_definition_df[
                "group_key"
            ]
            .astype(
                str
            )
            .tolist()
        )

        self.rows_by_landmark: dict[
            int,
            np.ndarray,
        ] = {}

        self.position_lookup_by_landmark: dict[
            int,
            np.ndarray,
        ] = {}

        self.survival_reference_by_landmark: dict[
            int,
            np.ndarray,
        ] = {}

        self.risk_matrix_by_key: dict[
            tuple[
                int,
                str | None,
            ],
            np.ndarray,
        ] = {}

        self._checkpoint_patient_position: dict[
            tuple[
                int,
                int,
            ],
            np.ndarray,
        ] = {}

        self._prepare_landmark_maps()
        self._load_all_predictions()

    def _prepare_landmark_maps(
        self,
    ) -> None:
        landmark_index_long = (
            self.common[
                "landmark_index_long"
            ]
        )

        event = self.common[
            "event_within_60m"
        ]

        time_month = self.common[
            "analysis_time_month"
        ]

        for landmark_index in (
            PRIMARY_LANDMARK_INDICES
        ):
            landmark_index = int(
                landmark_index
            )

            rows = np.where(
                landmark_index_long
                == landmark_index
            )[0].astype(
                np.int64
            )

            if len(rows) == 0:
                raise ValueError(
                    f"Landmark {LANDMARK_MONTHS[landmark_index]}月"
                    "没有风险集。"
                )

            self.rows_by_landmark[
                landmark_index
            ] = rows

            position_lookup = np.full(
                EXPECTED_LONG_N,
                -1,
                dtype=np.int32,
            )

            position_lookup[
                rows
            ] = np.arange(
                len(rows),
                dtype=np.int32,
            )

            self.position_lookup_by_landmark[
                landmark_index
            ] = position_lookup

            self.survival_reference_by_landmark[
                landmark_index
            ] = build_survival_array(
                event[
                    rows
                ],
                time_month[
                    rows
                ],
            )

    def _checkpoint_patient_lookup(
        self,
        fold_id: int,
        landmark_month: int,
    ) -> np.ndarray:
        key = (
            int(
                fold_id
            ),
            int(
                landmark_month
            ),
        )

        if key in (
            self._checkpoint_patient_position
        ):
            return (
                self._checkpoint_patient_position[
                    key
                ]
            )

        checkpoint_dir = (
            STEP12C_PREDICTION_DIR
            / f"fold_{int(fold_id)}"
            / f"landmark_{int(landmark_month)}m"
        )

        patient_file = (
            checkpoint_dir
            / "patient_local.npy"
        )

        require_files(
            [
                patient_file
            ]
        )

        patient_local = np.load(
            patient_file
        ).astype(
            np.int32
        )

        if (
            len(
                np.unique(
                    patient_local
                )
            )
            != len(
                patient_local
            )
        ):
            raise ValueError(
                f"{patient_file}存在重复患者。"
            )

        if np.any(
            patient_local < 0
        ) or np.any(
            patient_local
            >= EXPECTED_DEVELOPMENT_N
        ):
            raise ValueError(
                f"{patient_file}存在非法患者索引。"
            )

        lookup = np.full(
            EXPECTED_DEVELOPMENT_N,
            -1,
            dtype=np.int32,
        )

        lookup[
            patient_local
        ] = np.arange(
            len(
                patient_local
            ),
            dtype=np.int32,
        )

        self._checkpoint_patient_position[
            key
        ] = lookup

        return lookup

    @staticmethod
    def _prediction_filename(
        group_key: str | None,
    ) -> str:
        if group_key is None:
            return (
                "baseline_calibrated_risk.npy"
            )

        return (
            f"group_{group_key}"
            "_calibrated_risk.npy"
        )

    def _assemble_landmark_prediction(
        self,
        landmark_index: int,
        group_key: str | None,
    ) -> np.ndarray:
        landmark_index = int(
            landmark_index
        )

        landmark_month = int(
            LANDMARK_MONTHS[
                landmark_index
            ]
        )

        rows = self.rows_by_landmark[
            landmark_index
        ]

        patient_for_rows = (
            self.common[
                "local_patient_idx_long"
            ][
                rows
            ]
        )

        fold_for_rows = (
            self.common[
                "development_fold_id"
            ][
                patient_for_rows
            ]
        )

        risk = np.full(
            (
                len(
                    rows
                ),
                EXPECTED_INTERVAL_N,
            ),
            np.nan,
            dtype=np.float32,
        )

        filename = (
            self._prediction_filename(
                group_key
            )
        )

        for fold_id in range(
            N_SPLITS
        ):
            target_position = np.where(
                fold_for_rows
                == fold_id
            )[0]

            if len(
                target_position
            ) == 0:
                continue

            checkpoint_dir = (
                STEP12C_PREDICTION_DIR
                / f"fold_{fold_id}"
                / f"landmark_{landmark_month}m"
            )

            prediction_file = (
                checkpoint_dir
                / filename
            )

            require_files(
                [
                    prediction_file
                ]
            )

            prediction = np.load(
                prediction_file,
                mmap_mode="r",
            )

            if (
                prediction.ndim != 2
                or prediction.shape[1]
                != EXPECTED_INTERVAL_N
            ):
                raise ValueError(
                    f"{prediction_file}形状非法："
                    f"{prediction.shape}"
                )

            lookup = (
                self._checkpoint_patient_lookup(
                    fold_id=fold_id,
                    landmark_month=landmark_month,
                )
            )

            target_patients = (
                patient_for_rows[
                    target_position
                ]
            )

            source_position = (
                lookup[
                    target_patients
                ]
            )

            if np.any(
                source_position
                < 0
            ):
                bad = target_patients[
                    source_position
                    < 0
                ][
                    :20
                ]

                raise ValueError(
                    f"Fold {fold_id}, "
                    f"Landmark {landmark_month}月"
                    "checkpoint缺少风险集患者："
                    f"{bad.tolist()}"
                )

            risk[
                target_position,
                :,
            ] = np.asarray(
                prediction[
                    source_position,
                    :,
                ],
                dtype=np.float32,
            )

        if not np.isfinite(
            risk
        ).all():
            raise ValueError(
                f"Landmark {landmark_month}月, "
                f"group={group_key}拼接后存在未填充预测。"
            )

        if np.any(
            risk < 0
        ) or np.any(
            risk > 1
        ):
            raise ValueError(
                "风险预测超出[0,1]。"
            )

        if np.any(
            np.diff(
                risk.astype(
                    np.float64
                ),
                axis=1,
            )
            < -1e-6
        ):
            raise ValueError(
                f"Landmark {landmark_month}月, "
                f"group={group_key}累计风险非单调。"
            )

        return risk

    def _load_all_predictions(
        self,
    ) -> None:
        for landmark_index in (
            PRIMARY_LANDMARK_INDICES
        ):
            landmark_index = int(
                landmark_index
            )

            self.risk_matrix_by_key[
                (
                    landmark_index,
                    None,
                )
            ] = (
                self._assemble_landmark_prediction(
                    landmark_index=landmark_index,
                    group_key=None,
                )
            )

            for group_key in (
                self.group_keys
            ):
                self.risk_matrix_by_key[
                    (
                        landmark_index,
                        group_key,
                    )
                ] = (
                    self._assemble_landmark_prediction(
                        landmark_index=landmark_index,
                        group_key=group_key,
                    )
                )

    def get(
        self,
        landmark_index: int,
        group_key: str | None,
    ) -> np.ndarray:
        key = (
            int(
                landmark_index
            ),
            group_key,
        )

        if key not in (
            self.risk_matrix_by_key
        ):
            raise KeyError(
                f"没有prediction matrix：{key}"
            )

        return (
            self.risk_matrix_by_key[
                key
            ]
        )


# =============================================================================
# 7. Step12C point-estimate重建审计
# =============================================================================

def audit_point_estimates(
    store: PredictionStore,
    metrics_df: pd.DataFrame,
) -> pd.DataFrame:
    baseline_audit = pd.read_csv(
        STEP12C_BASELINE_AUDIT_FILE,
        encoding="utf-8-sig",
    )

    audit_rows: list[
        dict[str, Any]
    ] = []

    for landmark_index in (
        PRIMARY_LANDMARK_INDICES
    ):
        landmark_index = int(
            landmark_index
        )

        landmark_month = int(
            LANDMARK_MONTHS[
                landmark_index
            ]
        )

        survival_reference = (
            store.survival_reference_by_landmark[
                landmark_index
            ]
        )

        baseline_risk = store.get(
            landmark_index,
            None,
        )

        baseline_metrics = (
            evaluate_survival_predictions(
                survival_reference,
                survival_reference,
                baseline_risk,
            )
        )

        saved_baseline = baseline_audit.loc[
            baseline_audit[
                "landmark_index"
            ].astype(
                int
            )
            == landmark_index
        ]

        if len(
            saved_baseline
        ) != 1:
            raise ValueError(
                f"Landmark {landmark_month}月"
                "baseline audit不是1行。"
            )

        saved_baseline = (
            saved_baseline.iloc[
                0
            ]
        )

        baseline_diffs = {
            "uno_c": abs(
                float(
                    baseline_metrics[
                        "uno_c_index_5y"
                    ]
                )
                - float(
                    saved_baseline[
                        "fp32_baseline_uno_c"
                    ]
                )
            ),
            "iAUC": abs(
                float(
                    baseline_metrics[
                        "integrated_dynamic_auc"
                    ]
                )
                - float(
                    saved_baseline[
                        "fp32_baseline_iAUC"
                    ]
                )
            ),
            "IBS": abs(
                float(
                    baseline_metrics[
                        "integrated_brier"
                    ]
                )
                - float(
                    saved_baseline[
                        "fp32_baseline_IBS"
                    ]
                )
            ),
        }

        if max(
            baseline_diffs.values()
        ) > POINT_AUDIT_TOL:
            raise ValueError(
                f"Landmark {landmark_month}月"
                "baseline point estimate重建不一致："
                f"{baseline_diffs}"
            )

        landmark_groups = metrics_df.loc[
            metrics_df[
                "landmark_index"
            ]
            == landmark_index
        ]

        for row in landmark_groups.itertuples(
            index=False
        ):
            group_key = str(
                row.group_key
            )

            occluded_risk = store.get(
                landmark_index,
                group_key,
            )

            occluded_metrics = (
                evaluate_survival_predictions(
                    survival_reference,
                    survival_reference,
                    occluded_risk,
                )
            )

            mean_abs_delta = float(
                np.mean(
                    np.abs(
                        occluded_risk[
                            :,
                            -1,
                        ].astype(
                            np.float64
                        )
                        - baseline_risk[
                            :,
                            -1,
                        ].astype(
                            np.float64
                        )
                    )
                )
            )

            differences = {
                "occluded_uno_c": abs(
                    float(
                        occluded_metrics[
                            "uno_c_index_5y"
                        ]
                    )
                    - float(
                        row.occluded_uno_c
                    )
                ),
                "occluded_iAUC": abs(
                    float(
                        occluded_metrics[
                            "integrated_dynamic_auc"
                        ]
                    )
                    - float(
                        row.occluded_iAUC
                    )
                ),
                "occluded_IBS": abs(
                    float(
                        occluded_metrics[
                            "integrated_brier"
                        ]
                    )
                    - float(
                        row.occluded_IBS
                    )
                ),
                "mean_abs_delta_risk5": abs(
                    mean_abs_delta
                    - float(
                        row.mean_abs_delta_risk5
                    )
                ),
            }

            max_difference = max(
                differences.values()
            )

            if (
                max_difference
                > POINT_AUDIT_TOL
            ):
                raise ValueError(
                    "Point estimate重建失败："
                    f"Landmark={landmark_month}月, "
                    f"group={group_key}, "
                    f"diff={differences}"
                )

            audit_rows.append(
                {
                    "landmark_index": (
                        landmark_index
                    ),
                    "landmark_month": (
                        landmark_month
                    ),
                    "group_key": (
                        group_key
                    ),
                    "group_label": (
                        row.group_label
                    ),
                    "max_abs_metric_difference": (
                        max_difference
                    ),
                }
            )

    audit_df = pd.DataFrame(
        audit_rows
    )

    audit_df.to_csv(
        OUTPUT_DIR
        / "step12c_point_estimate_reconstruction_audit.csv",
        index=False,
        encoding="utf-8-sig",
    )

    return audit_df


# =============================================================================
# 8. Bootstrap checkpoint与版本签名
# =============================================================================

def build_source_signature(
    metrics_df: pd.DataFrame,
    group_definition_df: pd.DataFrame,
) -> dict[str, Any]:
    return {
        "pipeline_version": (
            PIPELINE_VERSION
        ),
        "step12c_checkpoint_metadata_sha256": (
            sha256_file(
                STEP12C_CHECKPOINT_METADATA_FILE
            )
        ),
        "step12c_metrics_sha256": (
            sha256_file(
                STEP12C_METRICS_FILE
            )
        ),
        "step12c_group_definition_sha256": (
            sha256_file(
                STEP12C_GROUP_DEFINITION_FILE
            )
        ),
        "step12c_channel_mapping_sha256": (
            sha256_file(
                STEP12C_CHANNEL_MAPPING_FILE
            )
        ),
        "bootstrap_reps": int(
            BOOTSTRAP_REPS
        ),
        "random_seed": int(
            RANDOM_SEED
        ),
        "bootstrap_unit": (
            "development patient"
        ),
        "bootstrap_pairing": (
            "same sampled patients across all primary landmarks and all groups"
        ),
        "ipcw_reference": (
            "complete original OOF risk set within each landmark"
        ),
        "primary_landmark_indices": (
            PRIMARY_LANDMARK_INDICES.tolist()
        ),
        "primary_landmark_months": (
            PRIMARY_LANDMARK_MONTHS.tolist()
        ),
        "metric_times": (
            METRIC_TIMES.tolist()
        ),
        "group_keys": (
            group_definition_df[
                "group_key"
            ].astype(
                str
            ).tolist()
        ),
        "condition_order": (
            metrics_df[
                [
                    "landmark_index",
                    "group_key",
                ]
            ].to_dict(
                orient="records"
            )
        ),
        "bootstrap_metrics": (
            BOOTSTRAP_METRICS
        ),
        "locked_test_used": False,
    }


def initialize_bootstrap_storage(
    metrics_df: pd.DataFrame,
    group_definition_df: pd.DataFrame,
) -> tuple[
    np.ndarray,
    np.ndarray,
    Path,
]:
    source_signature = (
        build_source_signature(
            metrics_df=metrics_df,
            group_definition_df=(
                group_definition_df
            ),
        )
    )

    metadata_file = (
        BOOTSTRAP_DIR
        / (
            f"paired_bootstrap_"
            f"{BOOTSTRAP_REPS}_metadata.json"
        )
    )

    base_checkpoint_file = (
        BOOTSTRAP_DIR
        / (
            f"paired_bootstrap_"
            f"{BOOTSTRAP_REPS}_replicates.npz"
        )
    )

    checkpoint_file = (
        BOOTSTRAP_DIR
        / (
            f"paired_bootstrap_{BOOTSTRAP_REPS}_"
            f"shard_{BOOTSTRAP_SHARD_INDEX:02d}_of_"
            f"{BOOTSTRAP_SHARD_COUNT:02d}.npz"
        )
        if BOOTSTRAP_SHARD_COUNT > 1
        else base_checkpoint_file
    )
    load_checkpoint_file = (
        checkpoint_file
        if checkpoint_file.exists()
        else base_checkpoint_file
    )

    if metadata_file.exists():
        observed = json.loads(
            metadata_file.read_text(
                encoding="utf-8"
            )
        )

        if observed != source_signature:
            raise ValueError(
                "已有bootstrap metadata"
                "与当前Step12C结果/配置不一致。"
                "请不要混用不同版本结果。"
            )

    else:
        if base_checkpoint_file.exists():
            raise ValueError(
                "检测到bootstrap checkpoint"
                "但缺少metadata。"
            )

        save_json(
            source_signature,
            metadata_file,
        )

    expected_shape = (
        BOOTSTRAP_REPS,
        len(
            metrics_df
        ),
        len(
            BOOTSTRAP_METRICS
        ),
    )

    if load_checkpoint_file.exists():
        checkpoint = np.load(
            load_checkpoint_file,
            allow_pickle=False,
        )

        values = checkpoint[
            "bootstrap_values"
        ].astype(
            np.float64
        )

        completed = checkpoint[
            "completed_replicates"
        ].astype(
            bool
        )

        if values.shape != expected_shape:
            raise ValueError(
                "已有bootstrap values"
                "形状与当前配置不一致。"
            )

        if completed.shape != (
            BOOTSTRAP_REPS,
        ):
            raise ValueError(
                "已有completed_replicates"
                "形状不一致。"
            )

        print(
            "\n检测到bootstrap断点："
            f"{int(completed.sum())}/"
            f"{BOOTSTRAP_REPS}已完成。"
        )

    else:
        values = np.full(
            expected_shape,
            np.nan,
            dtype=np.float64,
        )

        completed = np.zeros(
            BOOTSTRAP_REPS,
            dtype=bool,
        )

    return (
        values,
        completed,
        checkpoint_file,
    )


# =============================================================================
# 9. 患者级 paired bootstrap
# =============================================================================

def run_bootstrap(
    common: dict[str, np.ndarray],
    store: PredictionStore,
    metrics_df: pd.DataFrame,
    bootstrap_values: np.ndarray,
    completed_replicates: np.ndarray,
    checkpoint_file: Path,
) -> None:
    row_index_map = (
        common[
            "row_index_map"
        ]
    )

    event_within_60m = (
        common[
            "event_within_60m"
        ]
    )

    analysis_time_month = (
        common[
            "analysis_time_month"
        ]
    )

    bootstrap_start = time.time()

    for bootstrap_index in range(
        BOOTSTRAP_REPS
    ):
        if (
            BOOTSTRAP_SHARD_COUNT > 1
            and bootstrap_index
            % BOOTSTRAP_SHARD_COUNT
            != BOOTSTRAP_SHARD_INDEX
        ):
            continue
        if completed_replicates[
            bootstrap_index
        ]:
            continue

        # 与Step11/Step12B-2一致：每个replicate有独立、可复现seed。
        rng = np.random.default_rng(
            RANDOM_SEED
            + bootstrap_index
            * 1009
        )

        sampled_patients = rng.integers(
            0,
            EXPECTED_DEVELOPMENT_N,
            size=EXPECTED_DEVELOPMENT_N,
            endpoint=False,
        )

        for landmark_index in (
            PRIMARY_LANDMARK_INDICES
        ):
            landmark_index = int(
                landmark_index
            )

            # Allow exact reuse of previously verified four-landmark
            # bootstrap values while computing only newly added landmarks.
            landmark_positions = (
                metrics_df.loc[
                    metrics_df[
                        "landmark_index"
                    ]
                    == landmark_index,
                    "condition_position",
                ]
                .to_numpy(dtype=np.int64)
            )
            if (
                len(landmark_positions) > 0
                and np.isfinite(
                    bootstrap_values[
                        bootstrap_index,
                        landmark_positions,
                        :,
                    ]
                ).all()
            ):
                continue

            sampled_long_rows = (
                row_index_map[
                    sampled_patients,
                    landmark_index,
                ]
            )

            sampled_long_rows = sampled_long_rows[
                sampled_long_rows
                >= 0
            ].astype(
                np.int64
            )

            if len(
                sampled_long_rows
            ) < 50:
                continue

            sampled_event = (
                event_within_60m[
                    sampled_long_rows
                ]
            )

            if int(
                sampled_event.sum()
            ) < 5:
                continue

            survival_test = (
                build_survival_array(
                    sampled_event,
                    analysis_time_month[
                        sampled_long_rows
                    ],
                )
            )

            survival_train = (
                store.survival_reference_by_landmark[
                    landmark_index
                ]
            )

            position_lookup = (
                store.position_lookup_by_landmark[
                    landmark_index
                ]
            )

            sampled_position = (
                position_lookup[
                    sampled_long_rows
                ]
            )

            if np.any(
                sampled_position
                < 0
            ):
                raise RuntimeError(
                    "bootstrap sampled row"
                    "无法映射回Landmark位置。"
                )

            baseline_risk_full = store.get(
                landmark_index,
                None,
            )

            baseline_risk = (
                baseline_risk_full[
                    sampled_position,
                    :,
                ]
            )

            baseline_metrics = (
                safe_evaluate_survival_predictions(
                    survival_train,
                    survival_test,
                    baseline_risk,
                )
            )

            if baseline_metrics is None:
                continue

            landmark_groups = metrics_df.loc[
                metrics_df[
                    "landmark_index"
                ]
                == landmark_index
            ]

            for row in landmark_groups.itertuples(
                index=False
            ):
                group_key = str(
                    row.group_key
                )

                occluded_risk_full = store.get(
                    landmark_index,
                    group_key,
                )

                occluded_risk = (
                    occluded_risk_full[
                        sampled_position,
                        :,
                    ]
                )

                occluded_metrics = (
                    safe_evaluate_survival_predictions(
                        survival_train,
                        survival_test,
                        occluded_risk,
                    )
                )

                if occluded_metrics is None:
                    continue

                condition_position = int(
                    row.condition_position
                )

                bootstrap_values[
                    bootstrap_index,
                    condition_position,
                    METRIC_INDEX[
                        "uno_c_loss"
                    ],
                ] = (
                    float(
                        baseline_metrics[
                            "uno_c_index_5y"
                        ]
                    )
                    - float(
                        occluded_metrics[
                            "uno_c_index_5y"
                        ]
                    )
                )

                bootstrap_values[
                    bootstrap_index,
                    condition_position,
                    METRIC_INDEX[
                        "iAUC_loss"
                    ],
                ] = (
                    float(
                        baseline_metrics[
                            "integrated_dynamic_auc"
                        ]
                    )
                    - float(
                        occluded_metrics[
                            "integrated_dynamic_auc"
                        ]
                    )
                )

                bootstrap_values[
                    bootstrap_index,
                    condition_position,
                    METRIC_INDEX[
                        "IBS_increase"
                    ],
                ] = (
                    float(
                        occluded_metrics[
                            "integrated_brier"
                        ]
                    )
                    - float(
                        baseline_metrics[
                            "integrated_brier"
                        ]
                    )
                )

                bootstrap_values[
                    bootstrap_index,
                    condition_position,
                    METRIC_INDEX[
                        "mean_abs_delta_risk5"
                    ],
                ] = float(
                    np.mean(
                        np.abs(
                            occluded_risk[
                                :,
                                -1,
                            ].astype(
                                np.float64
                            )
                            - baseline_risk[
                                :,
                                -1,
                            ].astype(
                                np.float64
                            )
                        )
                    )
                )

        completed_replicates[
            bootstrap_index
        ] = True

        completed_n = int(
            completed_replicates.sum()
        )

        if (
            completed_n
            % BOOTSTRAP_SAVE_EVERY
            == 0
            or completed_n
            == BOOTSTRAP_REPS
        ):
            np.savez_compressed(
                checkpoint_file,
                bootstrap_values=(
                    bootstrap_values
                ),
                completed_replicates=(
                    completed_replicates
                ),
            )

        if (
            completed_n
            % PROGRESS_EVERY
            == 0
            or completed_n
            == BOOTSTRAP_REPS
        ):
            print(
                "Paired bootstrap："
                f"{completed_n}/"
                f"{BOOTSTRAP_REPS} | "
                "本次运行 "
                f"{format_duration(time.time() - bootstrap_start)}",
                flush=True,
            )

    if BOOTSTRAP_SHARD_COUNT > 1:
        np.savez_compressed(
            checkpoint_file,
            bootstrap_values=bootstrap_values,
            completed_replicates=completed_replicates,
        )


# =============================================================================
# 10. 汇总95% CI
# =============================================================================

def summarize_bootstrap(
    metrics_df: pd.DataFrame,
    bootstrap_values: np.ndarray,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    wide_rows: list[
        dict[str, Any]
    ] = []

    long_rows: list[
        dict[str, Any]
    ] = []

    point_column = {
        "uno_c_loss": (
            "uno_c_loss"
        ),
        "iAUC_loss": (
            "iAUC_loss"
        ),
        "IBS_increase": (
            "IBS_increase"
        ),
        "mean_abs_delta_risk5": (
            "mean_abs_delta_risk5"
        ),
    }

    for row in metrics_df.itertuples(
        index=False
    ):
        condition_position = int(
            row.condition_position
        )

        base = {
            "condition_position": (
                condition_position
            ),
            "group_position": int(
                row.group_position
            ),
            "group_key": (
                row.group_key
            ),
            "group_label": (
                row.group_label
            ),
            "landmark_index": int(
                row.landmark_index
            ),
            "landmark_month": int(
                row.landmark_month
            ),
            "landmark_year": float(
                row.landmark_year
            ),
            "risk_set_n": int(
                row.risk_set_n
            ),
            "future_5y_event_n": int(
                row.future_5y_event_n
            ),
            "occluded_dynamic_channel_n": int(
                row.occluded_dynamic_channel_n
            ),
            "occluded_static_channel_n": int(
                row.occluded_static_channel_n
            ),
            "age_occluded": bool(
                row.age_occluded
            ),
            "total_occluded_input_channel_n": int(
                row.total_occluded_input_channel_n
            ),
            "clinical_features": (
                row.clinical_features
            ),
        }

        wide = dict(
            base
        )

        for metric_name in (
            BOOTSTRAP_METRICS
        ):
            values = (
                bootstrap_values[
                    :,
                    condition_position,
                    METRIC_INDEX[
                        metric_name
                    ],
                ]
            )

            (
                lower,
                upper,
                median,
                valid_n,
            ) = percentile_interval(
                values,
                expected_n=(
                    BOOTSTRAP_REPS
                ),
            )

            point = float(
                getattr(
                    row,
                    point_column[
                        metric_name
                    ],
                )
            )

            favorable = (
                positive_fraction(
                    values
                )
                if metric_name
                != "mean_abs_delta_risk5"
                else np.nan
            )

            ci_excludes_zero = bool(
                lower > 0
                or upper < 0
            )

            wide[
                metric_name
            ] = point
            wide[
                f"{metric_name}_ci_lower"
            ] = lower
            wide[
                f"{metric_name}_ci_upper"
            ] = upper
            wide[
                f"{metric_name}_bootstrap_median"
            ] = median
            wide[
                f"{metric_name}_valid_n"
            ] = valid_n

            if metric_name != (
                "mean_abs_delta_risk5"
            ):
                wide[
                    f"{metric_name}_positive_fraction"
                ] = favorable
                wide[
                    f"{metric_name}_ci_excludes_zero"
                ] = (
                    ci_excludes_zero
                )

            long_rows.append(
                {
                    **base,
                    "metric": (
                        metric_name
                    ),
                    "point_estimate": (
                        point
                    ),
                    "ci_lower": (
                        lower
                    ),
                    "ci_upper": (
                        upper
                    ),
                    "bootstrap_median": (
                        median
                    ),
                    "valid_bootstrap_n": (
                        valid_n
                    ),
                    "positive_fraction": (
                        favorable
                    ),
                    "ci_excludes_zero": (
                        ci_excludes_zero
                        if metric_name
                        != "mean_abs_delta_risk5"
                        else np.nan
                    ),
                }
            )

        wide_rows.append(
            wide
        )

    wide_df = pd.DataFrame(
        wide_rows
    )

    long_df = pd.DataFrame(
        long_rows
    )

    # 在每个Landmark内按point estimate生成便于展示的rank。
    for metric_name, rank_name in [
        (
            "iAUC_loss",
            "iAUC_loss_rank",
        ),
        (
            "IBS_increase",
            "IBS_increase_rank",
        ),
        (
            "uno_c_loss",
            "uno_c_loss_rank",
        ),
        (
            "mean_abs_delta_risk5",
            "risk_change_rank",
        ),
    ]:
        wide_df[
            rank_name
        ] = (
            wide_df.groupby(
                "landmark_index"
            )[
                metric_name
            ]
            .rank(
                method="first",
                ascending=False,
            )
            .astype(
                int
            )
        )

    return (
        wide_df,
        long_df,
    )


# =============================================================================
# 11. 图形
# =============================================================================

def configure_plot_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "text.color": "black",
            "axes.labelcolor": "black",
            "axes.edgecolor": "black",
            "axes.titlecolor": "black",
            "xtick.color": "black",
            "ytick.color": "black",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save_figure(
    fig: plt.Figure,
    stem: str,
) -> None:
    for extension in [
        "png",
        "pdf",
        "svg",
    ]:
        kwargs = {
            "bbox_inches": "tight",
            "facecolor": "white",
        }

        if extension == "png":
            kwargs[
                "dpi"
            ] = 600

        fig.savefig(
            FIGURE_DIR
            / f"Step12C2_{stem}.{extension}",
            **kwargs,
        )

    plt.close(
        fig
    )


def plot_group_ci_by_landmark(
    wide_df: pd.DataFrame,
    metric_name: str,
    x_label: str,
    stem: str,
) -> None:
    group_order = (
        wide_df.loc[
            wide_df[
                "landmark_index"
            ]
            == 5
        ]
        .sort_values(
            metric_name,
            ascending=True,
        )[
            "group_label"
        ]
        .tolist()
    )

    if len(
        group_order
    ) != EXPECTED_GROUP_N:
        group_order = (
            wide_df[
                [
                    "group_position",
                    "group_label",
                ]
            ]
            .drop_duplicates()
            .sort_values(
                "group_position"
            )[
                "group_label"
            ]
            .tolist()
        )

    fig, axes = plt.subplots(
        1,
        4,
        figsize=(
            18.0,
            6.5,
        ),
        sharey=True,
    )

    for ax, landmark_index in zip(
        axes,
        PRIMARY_LANDMARK_INDICES,
    ):
        landmark_index = int(
            landmark_index
        )

        sub = wide_df.loc[
            wide_df[
                "landmark_index"
            ]
            == landmark_index
        ].copy()

        sub[
            "_group_order"
        ] = sub[
            "group_label"
        ].apply(
            group_order.index
        )

        sub = sub.sort_values(
            "_group_order"
        )

        y = np.arange(
            len(
                sub
            )
        )

        point = sub[
            metric_name
        ].to_numpy(
            dtype=float
        )

        lower = sub[
            f"{metric_name}_ci_lower"
        ].to_numpy(
            dtype=float
        )

        upper = sub[
            f"{metric_name}_ci_upper"
        ].to_numpy(
            dtype=float
        )

        xerr = np.vstack(
            [
                point - lower,
                upper - point,
            ]
        )

        ax.errorbar(
            point,
            y,
            xerr=xerr,
            fmt="o",
            capsize=2.5,
            linewidth=1.2,
        )

        ax.axvline(
            0.0,
            linestyle="--",
            linewidth=1.0,
        )

        ax.set_yticks(
            y
        )

        ax.set_yticklabels(
            sub[
                "group_label"
            ]
        )

        landmark_year = (
            LANDMARK_MONTHS[
                landmark_index
            ]
            / 12.0
        )

        ax.set_title(
            "Baseline"
            if landmark_year == 0
            else f"ART year {landmark_year:.0f}"
        )

        ax.set_xlabel(
            x_label
        )

        ax.grid(
            axis="x",
            linestyle="--",
            alpha=0.20,
        )

    fig.suptitle(
        "Grouped clinical-domain occlusion | patient-level paired bootstrap 95% CI",
        y=1.02,
    )

    fig.tight_layout()

    save_figure(
        fig,
        stem,
    )


def plot_5y_group_ci(
    wide_df: pd.DataFrame,
    metric_name: str,
    x_label: str,
    stem: str,
) -> None:
    sub = wide_df.loc[
        wide_df[
            "landmark_index"
        ]
        == 5
    ].copy()

    sub = sub.sort_values(
        metric_name,
        ascending=True,
    )

    y = np.arange(
        len(
            sub
        )
    )

    point = sub[
        metric_name
    ].to_numpy(
        dtype=float
    )

    lower = sub[
        f"{metric_name}_ci_lower"
    ].to_numpy(
        dtype=float
    )

    upper = sub[
        f"{metric_name}_ci_upper"
    ].to_numpy(
        dtype=float
    )

    xerr = np.vstack(
        [
            point - lower,
            upper - point,
        ]
    )

    fig, ax = plt.subplots(
        figsize=(
            8.8,
            6.2,
        )
    )

    ax.errorbar(
        point,
        y,
        xerr=xerr,
        fmt="o",
        capsize=3,
        linewidth=1.3,
    )

    ax.axvline(
        0.0,
        linestyle="--",
        linewidth=1.0,
    )

    ax.set_yticks(
        y
    )

    ax.set_yticklabels(
        sub[
            "group_label"
        ]
    )

    ax.set_xlabel(
        x_label
    )

    ax.set_title(
        "ART year 5 grouped clinical-domain occlusion | paired bootstrap 95% CI"
    )

    ax.grid(
        axis="x",
        linestyle="--",
        alpha=0.20,
    )

    fig.tight_layout()

    save_figure(
        fig,
        stem,
    )


# =============================================================================
# 12. 主流程
# =============================================================================

def run() -> None:
    total_start = time.time()

    if BOOTSTRAP_REPS < 1:
        raise ValueError(
            "BOOTSTRAP_REPS必须为正整数。"
        )

    if BOOTSTRAP_SAVE_EVERY < 1:
        raise ValueError(
            "BOOTSTRAP_SAVE_EVERY必须>=1。"
        )

    configure_plot_style()

    print(
        "=" * 122
    )
    print(
        "Step 12C-2：Grouped Clinical-Domain Occlusion 患者级paired bootstrap 95% CI"
    )
    print(
        "=" * 122
    )
    print(
        "Bootstrap次数：",
        BOOTSTRAP_REPS,
    )
    print(
        "Bootstrap单位：development patient"
    )
    print(
        "同一bootstrap患者样本：同时用于全部6个正式Landmark和全部10个临床域"
    )
    print(
        "IPCW reference：各Landmark完整原始OOF风险集"
    )
    print(
        "模型前向：不重新运行，直接读取Step12C保存预测"
    )
    print(
        "锁定测试集：未读取"
    )
    print(
        "输出目录：",
        OUTPUT_DIR,
    )
    print(
        "=" * 122
    )

    common = load_common_arrays()

    (
        metrics_df,
        group_definition_df,
    ) = load_group_tables()

    print(
        "Step12C group数：",
        len(
            group_definition_df
        ),
    )

    print(
        "主Landmark condition数：",
        len(
            metrics_df
        ),
    )

    print(
        "\n固定group顺序："
    )

    print(
        group_definition_df[
            [
                "group_key",
                "group_label",
                "dynamic_channel_n",
                "static_channel_n",
                "age_included",
            ]
        ].to_string(
            index=False
        )
    )

    print(
        "\n开始读取Step12C prediction checkpoints..."
    )

    store = PredictionStore(
        common=common,
        metrics_df=metrics_df,
        group_definition_df=(
            group_definition_df
        ),
    )

    print(
        "Prediction checkpoints读取完成。"
    )

    print(
        "\n开始point-estimate重建审计..."
    )

    audit_df = audit_point_estimates(
        store=store,
        metrics_df=metrics_df,
    )

    print(
        "Point-estimate审计通过 | "
        "worst abs difference="
        f"{audit_df['max_abs_metric_difference'].max():.3e}"
    )

    (
        bootstrap_values,
        completed_replicates,
        checkpoint_file,
    ) = initialize_bootstrap_storage(
        metrics_df=metrics_df,
        group_definition_df=(
            group_definition_df
        ),
    )

    if int(
        completed_replicates.sum()
    ) < BOOTSTRAP_REPS:
        run_bootstrap(
            common=common,
            store=store,
            metrics_df=metrics_df,
            bootstrap_values=(
                bootstrap_values
            ),
            completed_replicates=(
                completed_replicates
            ),
            checkpoint_file=(
                checkpoint_file
            ),
        )

    if BOOTSTRAP_SHARD_COUNT > 1:
        print(
            "Bootstrap shard completed: "
            f"{BOOTSTRAP_SHARD_INDEX + 1}/"
            f"{BOOTSTRAP_SHARD_COUNT}"
        )
        return

    if int(
        completed_replicates.sum()
    ) != BOOTSTRAP_REPS:
        raise RuntimeError(
            "Bootstrap未全部完成。"
        )

    wide_df, long_df = summarize_bootstrap(
        metrics_df=metrics_df,
        bootstrap_values=(
            bootstrap_values
        ),
    )

    wide_df.to_csv(
        OUTPUT_DIR
        / "grouped_occlusion_paired_bootstrap_CI_all.csv",
        index=False,
        encoding="utf-8-sig",
    )

    long_df.to_csv(
        OUTPUT_DIR
        / "grouped_occlusion_paired_bootstrap_CI_long.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # 便于论文主结果直接读取：按Landmark、iAUC loss排序。
    ranked_df = wide_df.sort_values(
        [
            "landmark_index",
            "iAUC_loss",
        ],
        ascending=[
            True,
            False,
        ],
    ).copy()

    ranked_df.to_csv(
        OUTPUT_DIR
        / "grouped_occlusion_paired_bootstrap_CI_ranked_by_iAUC.csv",
        index=False,
        encoding="utf-8-sig",
    )

    plot_group_ci_by_landmark(
        wide_df=wide_df,
        metric_name=(
            "iAUC_loss"
        ),
        x_label=(
            "iAUC loss after group occlusion"
        ),
        stem=(
            "Figure12C2A_group_iAUC_loss_bootstrap_CI"
        ),
    )

    plot_group_ci_by_landmark(
        wide_df=wide_df,
        metric_name=(
            "IBS_increase"
        ),
        x_label=(
            "IBS increase after group occlusion"
        ),
        stem=(
            "Figure12C2B_group_IBS_increase_bootstrap_CI"
        ),
    )

    plot_group_ci_by_landmark(
        wide_df=wide_df,
        metric_name=(
            "uno_c_loss"
        ),
        x_label=(
            "Uno C-index loss after group occlusion"
        ),
        stem=(
            "Figure12C2C_group_UnoC_loss_bootstrap_CI"
        ),
    )

    plot_5y_group_ci(
        wide_df=wide_df,
        metric_name=(
            "iAUC_loss"
        ),
        x_label=(
            "iAUC loss after group occlusion"
        ),
        stem=(
            "Figure12C2D_ARTyear5_group_iAUC_loss_bootstrap_CI"
        ),
    )

    metadata = {
        "pipeline_version": (
            PIPELINE_VERSION
        ),
        "bootstrap_reps": int(
            BOOTSTRAP_REPS
        ),
        "bootstrap_unit": (
            "development patient"
        ),
        "bootstrap_pairing": (
            "same sampled patients across all primary landmarks and all groups"
        ),
        "ipcw_reference": (
            "complete original OOF risk set within each landmark"
        ),
        "primary_landmark_months": (
            PRIMARY_LANDMARK_MONTHS.tolist()
        ),
        "group_keys": (
            group_definition_df[
                "group_key"
            ].astype(
                str
            ).tolist()
        ),
        "metrics": (
            BOOTSTRAP_METRICS
        ),
        "ci_method": (
            "2.5th and 97.5th percentile of patient-level paired bootstrap distribution"
        ),
        "positive_direction": {
            "uno_c_loss": (
                "positive = group occlusion worsens performance"
            ),
            "iAUC_loss": (
                "positive = group occlusion worsens performance"
            ),
            "IBS_increase": (
                "positive = group occlusion worsens performance"
            ),
        },
        "interpretation_scope": (
            "post-hoc grouped input reference occlusion of frozen LSTM; not retrained ablation; group effects not assumed additive"
        ),
        "locked_test_used": False,
        "elapsed_seconds": float(
            time.time()
            - total_start
        ),
        "output_dir": str(
            OUTPUT_DIR
        ),
    }

    save_json(
        metadata,
        OUTPUT_DIR
        / "step12c2_bootstrap_summary.json",
    )

    print(
        "\n"
        + "=" * 122
    )
    print(
        "Step 12C-2 paired bootstrap完成"
    )
    print(
        "=" * 122
    )
    print(
        "Bootstrap次数：",
        BOOTSTRAP_REPS,
    )
    print(
        "锁定测试集：未读取"
    )
    print(
        "总耗时：",
        format_duration(
            time.time()
            - total_start
        ),
    )

    print(
        "\n各Landmark按iAUC loss排序的临床域（含95%CI）："
    )

    for landmark_index in (
        PRIMARY_LANDMARK_INDICES
    ):
        landmark_index = int(
            landmark_index
        )

        sub = wide_df.loc[
            wide_df[
                "landmark_index"
            ]
            == landmark_index
        ].sort_values(
            "iAUC_loss",
            ascending=False,
        )

        print(
            "\n"
            + "-" * 112
        )
        print(
            "Landmark "
            f"{LANDMARK_MONTHS[landmark_index] / 12:.0f}年"
        )
        print(
            sub[
                [
                    "iAUC_loss_rank",
                    "group_label",
                    "iAUC_loss",
                    "iAUC_loss_ci_lower",
                    "iAUC_loss_ci_upper",
                    "iAUC_loss_positive_fraction",
                    "IBS_increase",
                    "IBS_increase_ci_lower",
                    "IBS_increase_ci_upper",
                    "uno_c_loss",
                    "uno_c_loss_ci_lower",
                    "uno_c_loss_ci_upper",
                    "mean_abs_delta_risk5",
                ]
            ].to_string(
                index=False
            )
        )

    # 单独打印ART相关组，便于核查研究重点。
    art_sub = wide_df.loc[
        wide_df[
            "group_key"
        ].isin(
            [
                "current_art_regimen",
                "cumulative_art_exposure",
            ]
        )
    ].sort_values(
        [
            "landmark_index",
            "group_position",
        ]
    )

    print(
        "\nCurrent ART vs Cumulative ART（含95%CI）："
    )
    print(
        art_sub[
            [
                "landmark_year",
                "group_label",
                "iAUC_loss",
                "iAUC_loss_ci_lower",
                "iAUC_loss_ci_upper",
                "IBS_increase",
                "IBS_increase_ci_lower",
                "IBS_increase_ci_upper",
                "uno_c_loss",
                "uno_c_loss_ci_lower",
                "uno_c_loss_ci_upper",
                "mean_abs_delta_risk5",
            ]
        ].to_string(
            index=False
        )
    )

    print(
        "\n最重要结果文件："
    )

    for filename in [
        "grouped_occlusion_paired_bootstrap_CI_all.csv",
        "grouped_occlusion_paired_bootstrap_CI_ranked_by_iAUC.csv",
        "grouped_occlusion_paired_bootstrap_CI_long.csv",
        "step12c_point_estimate_reconstruction_audit.csv",
        "step12c2_bootstrap_summary.json",
    ]:
        print(
            " -",
            OUTPUT_DIR
            / filename,
        )

    print(
        "=" * 122
    )


if __name__ == "__main__":
    run()
