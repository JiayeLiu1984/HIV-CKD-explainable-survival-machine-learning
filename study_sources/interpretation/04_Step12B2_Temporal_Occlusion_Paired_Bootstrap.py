#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Step 12B-2：LSTM-v2 Temporal Occlusion 患者级 paired bootstrap 95% CI

研究目的
--------
1. 不重新训练模型，不重新运行LSTM前向，不读取锁定内部测试集。
2. 直接读取Step12B已经保存的：
   - baseline calibrated risk；
   - 每个temporal occlusion condition的10个未来半年累计风险；
   - Step12B point estimates。
3. 完全沿用Step11的患者级paired bootstrap逻辑：
   - 每个bootstrap replicate从整个开发集22,337名患者中有放回抽样22,337次；
   - 同一bootstrap患者样本同时用于所有Landmark和所有occlusion condition；
   - 对某Landmark无有效prediction origin的患者自动剔除；
   - IPCW censoring/reference distribution始终来自该Landmark完整原始OOF风险集，
     与Step11正式paired bootstrap一致。
4. 对每个temporal occlusion condition计算：
   - Uno C-index loss = baseline - occluded；
   - iAUC loss = baseline - occluded；
   - IBS increase = occluded - baseline；
   - mean absolute change in 5-year risk；
   并给出患者级paired bootstrap 95% percentile CI。
5. 正值统一表示“遮挡后预测表现变差”：
   - Uno C loss > 0：遮挡后C-index下降；
   - iAUC loss > 0：遮挡后iAUC下降；
   - IBS increase > 0：遮挡后IBS变差。
6. 支持断点续跑。
7. 默认1000次bootstrap；测试时可临时：
      CKD_OCCLUSION_BOOTSTRAP_REPS=50 python Step12B2_....py
   正式论文结果保留1000次。

重要方法学边界
--------------
- 本步骤是对Step12B post-hoc reference occlusion结果做统计不确定性量化。
- 它不是重新训练后的ablation，也不替代后续Baseline-only / Current-only /
  Full-history longitudinal incremental value分析。
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sksurv.metrics import (
    brier_score,
    concordance_index_ipcw,
    cumulative_dynamic_auc,
    integrated_brier_score,
)
from sksurv.util import Surv


# =============================================================================
# 1. 固定路径与配置
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

STEP12B_DIR = (
    INTERPRETATION_ROOT
    / "03_TEMPORAL_OCCLUSION"
)

PREDICTION_CHECKPOINT_DIR = (
    STEP12B_DIR
    / "prediction_checkpoints"
)

OUTPUT_DIR = INTERPRETATION_ROOT / "04_TEMPORAL_BOOTSTRAP"

BOOTSTRAP_DIR = OUTPUT_DIR / "bootstrap_checkpoints"
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

# 与Step11、Step12B完全一致。
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
        "CKD_OCCLUSION_BOOTSTRAP_REPS",
        "1000",
    )
)

RANDOM_SEED = int(
    os.getenv(
        "CKD_OCCLUSION_BOOTSTRAP_SEED",
        "20260812",
    )
)

BOOTSTRAP_SAVE_EVERY = int(
    os.getenv(
        "CKD_OCCLUSION_BOOTSTRAP_SAVE_EVERY",
        "10",
    )
)

PROGRESS_EVERY = int(
    os.getenv(
        "CKD_OCCLUSION_BOOTSTRAP_PROGRESS_EVERY",
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
    "step12b2_temporal_occlusion_"
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
    in enumerate(BOOTSTRAP_METRICS)
}

STEP12B_ALL_METRICS_FILE = (
    STEP12B_DIR
    / "temporal_occlusion_all_conditions_metrics.csv"
)

STEP12B_BASELINE_AUDIT_FILE = (
    STEP12B_DIR
    / "temporal_occlusion_baseline_metric_audit.csv"
)

STEP12B_METADATA_FILE = (
    PREDICTION_CHECKPOINT_DIR
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
            + "\n".join(missing)
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

    with path.open("rb") as file:
        while True:
            block = file.read(
                1024 * 1024
            )
            if not block:
                break
            digest.update(block)

    return digest.hexdigest()


def format_duration(
    seconds: float,
) -> str:
    seconds = max(
        0,
        int(
            round(
                float(seconds)
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
            len(valid)
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


def parse_occluded_steps(
    value: Any,
) -> tuple[int, ...]:
    if pd.isna(
        value
    ):
        return tuple()

    text = str(
        value
    ).strip()

    if not text:
        return tuple()

    return tuple(
        int(
            part.strip()
        )
        for part in text.split(",")
        if part.strip()
    )


def condition_key(
    steps: tuple[int, ...],
) -> str:
    if not steps:
        return "baseline"

    return (
        "steps_"
        + "_".join(
            str(
                int(step)
            )
            for step in steps
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
# 3. 生存评价——严格沿用Step11 paired bootstrap定义
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
    """
    与Step11 evaluate_survival_predictions一致：
      - 5年Uno C；
      - 6~60月 cumulative/dynamic AUC的integrated mean；
      - 6~60月 IBS。
    """

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
        "uno_c_index_5y": c_index,
        "integrated_dynamic_auc": float(
            mean_auc
        ),
        "integrated_brier": ibs,
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
# 4. 读取Step6 + Step4共同数据
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

    actual = {
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

    for name, expected_shape in (
        expected_shapes.items()
    ):
        if (
            actual[
                name
            ].shape
            != expected_shape
        ):
            raise ValueError(
                f"{name}形状="
                f"{actual[name].shape}，"
                f"预期={expected_shape}。"
            )

    if not np.array_equal(
        row_index_map
        >= 0,
        np.column_stack(
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
        ),
    ):
        raise ValueError(
            "Step6 row_index_map与"
            "long格式患者-Landmark映射不一致。"
        )

    return {
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


# =============================================================================
# 5. 读取并标准化Step12B condition定义
# =============================================================================

def load_condition_table() -> pd.DataFrame:
    require_files(
        [
            STEP12B_ALL_METRICS_FILE,
            STEP12B_BASELINE_AUDIT_FILE,
            STEP12B_METADATA_FILE,
        ]
    )

    table = pd.read_csv(
        STEP12B_ALL_METRICS_FILE,
        encoding="utf-8-sig",
    )

    required_columns = {
        "condition_type",
        "condition_label",
        "landmark_index",
        "landmark_month",
        "landmark_year",
        "occluded_steps",
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
        required_columns
        - set(
            table.columns
        )
    )

    if missing:
        raise ValueError(
            "Step12B all-condition结果缺少列："
            + ", ".join(
                missing
            )
        )

    table = table.loc[
        table[
            "landmark_index"
        ].isin(
            PRIMARY_LANDMARK_INDICES
        )
    ].copy()

    table[
        "landmark_index"
    ] = table[
        "landmark_index"
    ].astype(
        int
    )

    table[
        "landmark_month"
    ] = table[
        "landmark_month"
    ].astype(
        int
    )

    table[
        "occluded_steps_tuple"
    ] = [
        parse_occluded_steps(
            value
        )
        for value
        in table[
            "occluded_steps"
        ].tolist()
    ]

    table[
        "condition_id"
    ] = [
        (
            f"L{int(row.landmark_index)}"
            f"|{row.condition_type}"
            f"|{row.condition_label}"
            f"|steps="
            + "_".join(
                str(
                    int(step)
                )
                for step
                in row.occluded_steps_tuple
            )
        )
        for row
        in table.itertuples(
            index=False
        )
    ]

    if table[
        "condition_id"
    ].duplicated().any():
        duplicates = table.loc[
            table[
                "condition_id"
            ].duplicated(
                keep=False
            ),
            "condition_id",
        ].tolist()

        raise ValueError(
            "Step12B condition_id存在重复："
            + str(
                duplicates[
                    :20
                ]
            )
        )

    # 保持主结果自然顺序。
    type_order = {
        "single_step": 0,
        "lag_band": 1,
    }

    table[
        "_type_order"
    ] = table[
        "condition_type"
    ].map(
        type_order
    ).fillna(
        99
    )

    table = table.sort_values(
        [
            "landmark_index",
            "_type_order",
            "lag_month"
            if "lag_month"
            in table.columns
            else "landmark_index",
            "condition_label",
        ],
        na_position="last",
    ).reset_index(
        drop=True
    )

    table[
        "condition_position"
    ] = np.arange(
        len(
            table
        ),
        dtype=np.int32,
    )

    return table


# =============================================================================
# 6. 从Step12B prediction checkpoints重建Landmark风险矩阵
# =============================================================================

class PredictionStore:
    """
    把Step12B各fold保存的prediction checkpoint，
    重新拼回Landmark完整OOF风险集顺序。

    只读取已保存风险，不运行神经网络。
    """

    def __init__(
        self,
        common: dict[str, np.ndarray],
        condition_table: pd.DataFrame,
    ) -> None:
        self.common = common
        self.condition_table = (
            condition_table
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
                tuple[int, ...],
            ],
            np.ndarray,
        ] = {}

        self._checkpoint_patient_position: dict[
            tuple[int, int],
            np.ndarray,
        ] = {}

        self._prepare_landmark_maps()
        self._load_all_needed_predictions()

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

            if len(
                rows
            ) == 0:
                raise ValueError(
                    f"Landmark {landmark_index}"
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
                len(
                    rows
                ),
                dtype=np.int32,
            )

            self.position_lookup_by_landmark[
                landmark_index
            ] = (
                position_lookup
            )

            self.survival_reference_by_landmark[
                landmark_index
            ] = (
                build_survival_array(
                    event[
                        rows
                    ],
                    time_month[
                        rows
                    ],
                )
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
            PREDICTION_CHECKPOINT_DIR
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

    def _assemble_landmark_condition(
        self,
        landmark_index: int,
        steps: tuple[int, ...],
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
            f"{condition_key(steps)}"
            "_calibrated_risk.npy"
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
                PREDICTION_CHECKPOINT_DIR
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
                or prediction.shape[
                    1
                ]
                != EXPECTED_INTERVAL_N
            ):
                raise ValueError(
                    f"{prediction_file}"
                    "形状非法。"
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

            source_position = lookup[
                target_patients
            ]

            if np.any(
                source_position
                < 0
            ):
                bad = target_patients[
                    source_position
                    < 0
                ][:20]

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
                f"Landmark {landmark_month}月 "
                f"steps={steps}"
                "拼接后存在未填充预测。"
            )

        if np.any(
            risk < 0
        ) or np.any(
            risk > 1
        ):
            raise ValueError(
                "风险预测超出[0,1]。"
            )

        # 累计风险应非递减。
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
                f"Landmark {landmark_month}月 "
                f"steps={steps}"
                "出现累计风险非单调。"
            )

        return risk

    def _load_all_needed_predictions(
        self,
    ) -> None:
        for landmark_index in (
            PRIMARY_LANDMARK_INDICES
        ):
            landmark_index = int(
                landmark_index
            )

            # baseline
            baseline_key = (
                landmark_index,
                tuple(),
            )

            self.risk_matrix_by_key[
                baseline_key
            ] = (
                self._assemble_landmark_condition(
                    landmark_index=landmark_index,
                    steps=tuple(),
                )
            )

            landmark_conditions = (
                self.condition_table.loc[
                    self.condition_table[
                        "landmark_index"
                    ]
                    == landmark_index
                ]
            )

            unique_steps = sorted(
                set(
                    landmark_conditions[
                        "occluded_steps_tuple"
                    ].tolist()
                )
            )

            for steps in (
                unique_steps
            ):
                key = (
                    landmark_index,
                    tuple(
                        steps
                    ),
                )

                if key in (
                    self.risk_matrix_by_key
                ):
                    continue

                self.risk_matrix_by_key[
                    key
                ] = (
                    self._assemble_landmark_condition(
                        landmark_index=landmark_index,
                        steps=tuple(
                            steps
                        ),
                    )
                )

    def get(
        self,
        landmark_index: int,
        steps: tuple[int, ...],
    ) -> np.ndarray:
        key = (
            int(
                landmark_index
            ),
            tuple(
                steps
            ),
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
# 7. 读取完成后做point-estimate重建审计
# =============================================================================

def audit_point_estimates(
    store: PredictionStore,
    condition_table: pd.DataFrame,
) -> pd.DataFrame:
    baseline_audit = pd.read_csv(
        STEP12B_BASELINE_AUDIT_FILE,
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
            tuple(),
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

        landmark_conditions = (
            condition_table.loc[
                condition_table[
                    "landmark_index"
                ]
                == landmark_index
            ]
        )

        for row in (
            landmark_conditions.itertuples(
                index=False
            )
        ):
            steps = tuple(
                row.occluded_steps_tuple
            )

            occluded_risk = (
                store.get(
                    landmark_index,
                    steps,
                )
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
                    f"Point estimate重建失败："
                    f"Landmark={landmark_month}月, "
                    f"condition={row.condition_label}, "
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
                    "condition_type": (
                        row.condition_type
                    ),
                    "condition_label": (
                        row.condition_label
                    ),
                    "occluded_steps": (
                        row.occluded_steps
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
        / "step12b_point_estimate_reconstruction_audit.csv",
        index=False,
        encoding="utf-8-sig",
    )

    return audit_df


# =============================================================================
# 8. Bootstrap metadata / checkpoint
# =============================================================================

def build_source_signature(
    condition_table: pd.DataFrame,
) -> dict[str, Any]:
    return {
        "pipeline_version": (
            PIPELINE_VERSION
        ),
        "step12b_pipeline_metadata_sha256": (
            sha256_file(
                STEP12B_METADATA_FILE
            )
        ),
        "step12b_all_metrics_sha256": (
            sha256_file(
                STEP12B_ALL_METRICS_FILE
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
            "same sampled patients across "
            "all primary landmarks and "
            "all occlusion conditions"
        ),
        "ipcw_reference": (
            "complete original OOF risk set "
            "within each landmark"
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
        "condition_ids": (
            condition_table[
                "condition_id"
            ].tolist()
        ),
        "bootstrap_metrics": (
            BOOTSTRAP_METRICS
        ),
        "locked_test_used": False,
    }


def initialize_bootstrap_storage(
    condition_table: pd.DataFrame,
) -> tuple[
    np.ndarray,
    np.ndarray,
    Path,
]:
    source_signature = (
        build_source_signature(
            condition_table
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

        if (
            observed
            != source_signature
        ):
            raise ValueError(
                "已有bootstrap metadata"
                "与当前Step12B结果/配置不一致。"
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
            condition_table
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

        if (
            values.shape
            != expected_shape
        ):
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
    condition_table: pd.DataFrame,
    bootstrap_values: np.ndarray,
    completed_replicates: np.ndarray,
    checkpoint_file: Path,
) -> None:
    """
    完全沿用Step11 paired bootstrap抽样框架：
    每个replicate在22,337开发患者中有放回抽样22,337次，
    同一个sampled_patients同时用于所有Landmark和所有condition。
    """

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

            # A six-landmark rerun may be seeded with rigorously verified
            # bootstrap values from the earlier four-landmark analysis.
            # Skip a landmark only when every condition/metric for this
            # replicate is already present in the checkpoint.
            landmark_positions = (
                condition_table.loc[
                    condition_table[
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

            sampled_long_rows = (
                sampled_long_rows[
                    sampled_long_rows
                    >= 0
                ].astype(
                    np.int64
                )
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

            baseline_risk_full = (
                store.get(
                    landmark_index,
                    tuple(),
                )
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

            if (
                baseline_metrics
                is None
            ):
                continue

            landmark_conditions = (
                condition_table.loc[
                    condition_table[
                        "landmark_index"
                    ]
                    == landmark_index
                ]
            )

            # 同一个occluded_steps可能同时对应：
            # single-step current 和 lag-band current。
            # 每个bootstrap replicate只评价一次，随后复用结果。
            unique_cache: dict[
                tuple[int, ...],
                tuple[
                    dict[str, float] | None,
                    float,
                ],
            ] = {}

            for row in (
                landmark_conditions.itertuples(
                    index=False
                )
            ):
                steps = tuple(
                    row.occluded_steps_tuple
                )

                if steps not in (
                    unique_cache
                ):
                    occluded_risk_full = (
                        store.get(
                            landmark_index,
                            steps,
                        )
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

                    mean_abs_delta_risk5 = float(
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

                    unique_cache[
                        steps
                    ] = (
                        occluded_metrics,
                        mean_abs_delta_risk5,
                    )

                (
                    occluded_metrics,
                    mean_abs_delta_risk5,
                ) = unique_cache[
                    steps
                ]

                if (
                    occluded_metrics
                    is None
                ):
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
                ] = (
                    mean_abs_delta_risk5
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
    condition_table: pd.DataFrame,
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

    for row in (
        condition_table.itertuples(
            index=False
        )
    ):
        condition_position = int(
            row.condition_position
        )

        base = {
            "condition_position": (
                condition_position
            ),
            "condition_id": (
                row.condition_id
            ),
            "condition_type": (
                row.condition_type
            ),
            "condition_label": (
                row.condition_label
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
            "history_month": (
                row.history_month
                if hasattr(
                    row,
                    "history_month"
                )
                else np.nan
            ),
            "lag_month": (
                row.lag_month
                if hasattr(
                    row,
                    "lag_month"
                )
                else np.nan
            ),
            "lag_band_start_month": (
                row.lag_band_start_month
                if hasattr(
                    row,
                    "lag_band_start_month"
                )
                else np.nan
            ),
            "lag_band_end_month": (
                row.lag_band_end_month
                if hasattr(
                    row,
                    "lag_band_end_month"
                )
                else np.nan
            ),
            "occluded_steps": (
                row.occluded_steps
            ),
            "risk_set_n": int(
                row.risk_set_n
            )
            if hasattr(
                row,
                "risk_set_n"
            )
            else np.nan,
            "future_5y_event_n": int(
                row.future_5y_event_n
            )
            if hasattr(
                row,
                "future_5y_event_n"
            )
            else np.nan,
            "active_origin_fraction": float(
                row.active_origin_fraction
            )
            if hasattr(
                row,
                "active_origin_fraction"
            )
            else np.nan,
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
            "figure.facecolor": (
                "white"
            ),
            "axes.facecolor": (
                "white"
            ),
            "savefig.facecolor": (
                "white"
            ),
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
    fig.savefig(
        FIGURE_DIR
        / f"Step12B2_{stem}.png",
        dpi=600,
        bbox_inches="tight",
        facecolor="white",
    )

    fig.savefig(
        FIGURE_DIR
        / f"Step12B2_{stem}.pdf",
        bbox_inches="tight",
        facecolor="white",
    )

    fig.savefig(
        FIGURE_DIR
        / f"Step12B2_{stem}.svg",
        bbox_inches="tight",
        facecolor="white",
    )

    plt.close(
        fig
    )


def plot_single_step_ci(
    wide_df: pd.DataFrame,
    metric_name: str,
    y_label: str,
    stem: str,
) -> None:
    sub = wide_df.loc[
        wide_df[
            "condition_type"
        ]
        == "single_step"
    ].copy()

    fig, ax = plt.subplots(
        figsize=(
            9.5,
            6.2,
        )
    )

    for landmark_index in (
        PRIMARY_LANDMARK_INDICES
    ):
        landmark_index = int(
            landmark_index
        )

        s = sub.loc[
            sub[
                "landmark_index"
            ]
            == landmark_index
        ].sort_values(
            "lag_month"
        )

        if s.empty:
            continue

        x = s[
            "lag_month"
        ].to_numpy(
            dtype=float
        )

        y = s[
            metric_name
        ].to_numpy(
            dtype=float
        )

        lower = s[
            f"{metric_name}_ci_lower"
        ].to_numpy(
            dtype=float
        )

        upper = s[
            f"{metric_name}_ci_upper"
        ].to_numpy(
            dtype=float
        )

        yerr = np.vstack(
            [
                y - lower,
                upper - y,
            ]
        )

        landmark_year = (
            LANDMARK_MONTHS[
                landmark_index
            ]
            / 12.0
        )

        label = (
            "Baseline"
            if landmark_year
            == 0
            else (
                f"ART year "
                f"{landmark_year:.0f}"
            )
        )

        ax.errorbar(
            x,
            y,
            yerr=yerr,
            marker="o",
            linewidth=1.5,
            capsize=2.5,
            label=label,
        )

    ax.axhline(
        0.0,
        linewidth=1.0,
        linestyle="--",
    )

    ax.set_xlabel(
        "Months before prediction landmark"
    )

    ax.set_ylabel(
        y_label
    )

    ax.set_title(
        "Temporal occlusion with patient-level paired bootstrap 95% CI"
    )

    ax.legend(
        frameon=False
    )

    ax.grid(
        axis="y",
        linestyle="--",
        alpha=0.25,
    )

    fig.tight_layout()

    save_figure(
        fig,
        stem,
    )


def plot_5y_lag_band_ci(
    wide_df: pd.DataFrame,
    metric_name: str,
    y_label: str,
    stem: str,
) -> None:
    sub = wide_df.loc[
        (
            wide_df[
                "condition_type"
            ]
            == "lag_band"
        )
        & (
            wide_df[
                "landmark_index"
            ]
            == 5
        )
    ].copy()

    desired_order = [
        "current_0m",
        "lag_6_12m",
        "lag_18_24m",
        "lag_30_36m",
        "lag_42_48m",
        "lag_54_60m",
    ]

    sub[
        "_order"
    ] = sub[
        "condition_label"
    ].apply(
        lambda x: (
            desired_order.index(
                x
            )
            if x
            in desired_order
            else 999
        )
    )

    sub = sub.sort_values(
        "_order"
    )

    if sub.empty:
        return

    x = np.arange(
        len(
            sub
        )
    )

    y = sub[
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

    yerr = np.vstack(
        [
            y - lower,
            upper - y,
        ]
    )

    labels = [
        str(
            label
        )
        .replace(
            "current_0m",
            "Current"
        )
        .replace(
            "lag_",
            ""
        )
        .replace(
            "m",
            " mo"
        )
        .replace(
            "_",
            "–"
        )
        for label
        in sub[
            "condition_label"
        ].tolist()
    ]

    fig, ax = plt.subplots(
        figsize=(
            9.0,
            5.8,
        )
    )

    ax.errorbar(
        x,
        y,
        yerr=yerr,
        marker="o",
        linewidth=1.5,
        capsize=3,
    )

    ax.axhline(
        0.0,
        linewidth=1.0,
        linestyle="--",
    )

    ax.set_xticks(
        x
    )

    ax.set_xticklabels(
        labels,
        rotation=25,
        ha="right",
    )

    ax.set_xlabel(
        "Historical lag band before the 5-year landmark"
    )

    ax.set_ylabel(
        y_label
    )

    ax.set_title(
        "5-year landmark temporal occlusion | paired bootstrap 95% CI"
    )

    ax.grid(
        axis="y",
        linestyle="--",
        alpha=0.25,
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
        "=" * 120
    )
    print(
        "Step 12B-2：Temporal Occlusion 患者级paired bootstrap 95% CI"
    )
    print(
        "=" * 120
    )
    print(
        "Bootstrap次数：",
        BOOTSTRAP_REPS,
    )
    print(
        "Bootstrap单位：development patient"
    )
    print(
        "同一bootstrap患者样本：同时用于所有Landmark和所有occlusion condition"
    )
    print(
        "IPCW reference：各Landmark完整原始OOF风险集"
    )
    print(
        "模型前向：不重新运行，直接读取Step12B保存预测"
    )
    print(
        "锁定测试集：未读取"
    )
    print(
        "输出目录：",
        OUTPUT_DIR,
    )
    print(
        "=" * 120
    )

    common = load_common_arrays()

    condition_table = (
        load_condition_table()
    )

    print(
        "Step12B condition数：",
        len(
            condition_table
        ),
    )

    print(
        "\n各Landmark condition数："
    )
    print(
        condition_table.groupby(
            [
                "landmark_year",
                "condition_type",
            ]
        ).size()
    )

    print(
        "\n开始读取Step12B prediction checkpoints..."
    )

    store = PredictionStore(
        common=common,
        condition_table=condition_table,
    )

    print(
        "Prediction checkpoints读取完成。"
    )

    print(
        "\n开始point-estimate重建审计..."
    )

    audit_df = audit_point_estimates(
        store=store,
        condition_table=condition_table,
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
        condition_table
    )

    if int(
        completed_replicates.sum()
    ) < BOOTSTRAP_REPS:
        run_bootstrap(
            common=common,
            store=store,
            condition_table=condition_table,
            bootstrap_values=bootstrap_values,
            completed_replicates=completed_replicates,
            checkpoint_file=checkpoint_file,
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

    wide_df, long_df = (
        summarize_bootstrap(
            condition_table=condition_table,
            bootstrap_values=bootstrap_values,
        )
    )

    wide_df.to_csv(
        OUTPUT_DIR
        / "temporal_occlusion_paired_bootstrap_CI_all_conditions.csv",
        index=False,
        encoding="utf-8-sig",
    )

    long_df.to_csv(
        OUTPUT_DIR
        / "temporal_occlusion_paired_bootstrap_CI_long.csv",
        index=False,
        encoding="utf-8-sig",
    )

    single_df = wide_df.loc[
        wide_df[
            "condition_type"
        ]
        == "single_step"
    ].copy()

    band_df = wide_df.loc[
        wide_df[
            "condition_type"
        ]
        == "lag_band"
    ].copy()

    single_df.to_csv(
        OUTPUT_DIR
        / "temporal_occlusion_single_step_paired_bootstrap_CI.csv",
        index=False,
        encoding="utf-8-sig",
    )

    band_df.to_csv(
        OUTPUT_DIR
        / "temporal_occlusion_lag_band_paired_bootstrap_CI.csv",
        index=False,
        encoding="utf-8-sig",
    )

    plot_single_step_ci(
        wide_df=wide_df,
        metric_name="iAUC_loss",
        y_label=(
            "iAUC loss after occlusion"
        ),
        stem=(
            "Figure12B2A_single_step_iAUC_loss_bootstrap_CI"
        ),
    )

    plot_single_step_ci(
        wide_df=wide_df,
        metric_name="IBS_increase",
        y_label=(
            "IBS increase after occlusion"
        ),
        stem=(
            "Figure12B2B_single_step_IBS_increase_bootstrap_CI"
        ),
    )

    plot_5y_lag_band_ci(
        wide_df=wide_df,
        metric_name="iAUC_loss",
        y_label=(
            "iAUC loss after occlusion"
        ),
        stem=(
            "Figure12B2C_5y_lag_band_iAUC_loss_bootstrap_CI"
        ),
    )

    plot_5y_lag_band_ci(
        wide_df=wide_df,
        metric_name="IBS_increase",
        y_label=(
            "IBS increase after occlusion"
        ),
        stem=(
            "Figure12B2D_5y_lag_band_IBS_increase_bootstrap_CI"
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
            "same sampled patients across "
            "all primary landmarks and "
            "all temporal occlusion conditions"
        ),
        "ipcw_reference": (
            "complete original OOF risk set "
            "within each landmark"
        ),
        "primary_landmark_months": (
            PRIMARY_LANDMARK_MONTHS.tolist()
        ),
        "metrics": (
            BOOTSTRAP_METRICS
        ),
        "ci_method": (
            "2.5th and 97.5th percentile "
            "of patient-level paired bootstrap distribution"
        ),
        "positive_direction": {
            "uno_c_loss": (
                "positive = occlusion worsens performance"
            ),
            "iAUC_loss": (
                "positive = occlusion worsens performance"
            ),
            "IBS_increase": (
                "positive = occlusion worsens performance"
            ),
        },
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
        / "step12b2_bootstrap_summary.json",
    )

    # -------------------------------------------------------------------------
    # 终端摘要
    # -------------------------------------------------------------------------
    print(
        "\n"
        + "=" * 120
    )
    print(
        "Step 12B-2 paired bootstrap完成"
    )
    print(
        "=" * 120
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
        "\n各Landmark单时间点："
        "按iAUC loss排序的前5项（含95%CI）"
    )

    for landmark_index in (
        PRIMARY_LANDMARK_INDICES
    ):
        landmark_index = int(
            landmark_index
        )

        sub = single_df.loc[
            single_df[
                "landmark_index"
            ]
            == landmark_index
        ].sort_values(
            "iAUC_loss",
            ascending=False,
        ).head(
            5
        )

        print(
            "\n"
            + "-" * 100
        )
        print(
            "Landmark "
            f"{LANDMARK_MONTHS[landmark_index] / 12:.0f}年"
        )
        print(
            sub[
                [
                    "history_month",
                    "lag_month",
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
                ]
            ].to_string(
                index=False
            )
        )

    print(
        "\nLag-band结果（含95%CI）："
    )

    print(
        band_df[
            [
                "landmark_year",
                "condition_label",
                "iAUC_loss",
                "iAUC_loss_ci_lower",
                "iAUC_loss_ci_upper",
                "iAUC_loss_positive_fraction",
                "IBS_increase",
                "IBS_increase_ci_lower",
                "IBS_increase_ci_upper",
                "IBS_increase_positive_fraction",
                "uno_c_loss",
                "uno_c_loss_ci_lower",
                "uno_c_loss_ci_upper",
            ]
        ].to_string(
            index=False
        )
    )

    print(
        "\n最重要结果文件："
    )

    for filename in [
        "temporal_occlusion_paired_bootstrap_CI_all_conditions.csv",
        "temporal_occlusion_single_step_paired_bootstrap_CI.csv",
        "temporal_occlusion_lag_band_paired_bootstrap_CI.csv",
        "step12b_point_estimate_reconstruction_audit.csv",
        "step12b2_bootstrap_summary.json",
    ]:
        print(
            " -",
            OUTPUT_DIR
            / filename,
        )

    print(
        "=" * 120
    )


if __name__ == "__main__":
    run()
