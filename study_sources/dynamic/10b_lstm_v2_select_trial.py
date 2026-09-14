#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Step 10D-FINALIZE v2
====================
直接读取已经完成的 LSTM-v2 Optuna Study，
不新增Trial、不重新训练、不读取锁定测试集。

本脚本专门对应实际21小时运行的Study：
输出目录：
  __CKD_WORKDIR__/rolling_5y_step10d_lstm_v2_tune_resume

Study：
  lstm_v2_hybrid_attention_equal_weight_ibs

选择规则保持原方案：
1. 仅使用 COMPLETE Trial；
2. 最低五折平均IBS作为基准；
3. IBS <= 最低IBS + 0.00010 的Trial进入候选；
4. 候选中优先平均iAUC最高；
5. 再平均Uno C最高；
6. 再IBS更低。

运行时间：通常几秒。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import optuna
import pandas as pd


PROJECT_DIR = Path("__CKD_WORKDIR__")
OUTPUT_DIR = PROJECT_DIR / "rolling_5y_step10d_lstm_v2_tune_resume"
TUNING_DIR = OUTPUT_DIR / "tuning"

EXPECTED_DB = TUNING_DIR / "lstm_v2_optuna.sqlite3"
EXPECTED_STUDY_NAME = "lstm_v2_hybrid_attention_equal_weight_ibs"

IBS_TOLERANCE = 0.00010
REFERENCE_LSTM_V1_MEAN_IBS = 0.02557902661806264

SUMMARY_FILE = TUNING_DIR / "lstm_v2_tuning_summary.json"
RANKING_FILE = TUNING_DIR / "lstm_v2_existing_complete_trial_ranking.csv"
ALL_TRIALS_FILE = TUNING_DIR / "lstm_v2_optuna_trials_finalized.csv"


def save_json(value: dict, path: Path) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


def normalize_parameters(parameters: dict) -> dict:
    required = [
        "hidden_size",
        "num_layers",
        "dropout",
        "projection_size",
        "bidirectional",
        "pooling_mode",
        "static_hidden",
        "summary_hidden",
        "horizon_embed_dim",
        "learning_rate",
        "weight_decay",
        "batch_size",
        "positive_weight",
        "focal_gamma",
        "auxiliary_5y_weight",
        "ranking_weight",
        "smoothness_weight",
    ]
    missing = [name for name in required if name not in parameters]
    if missing:
        raise ValueError(
            "选中Trial缺少必要参数："
            + ", ".join(missing)
        )

    return {
        "hidden_size": int(parameters["hidden_size"]),
        "num_layers": int(parameters["num_layers"]),
        "dropout": float(parameters["dropout"]),
        "projection_size": int(parameters["projection_size"]),
        "bidirectional": bool(parameters["bidirectional"]),
        "pooling_mode": str(parameters["pooling_mode"]),
        "static_hidden": int(parameters["static_hidden"]),
        "summary_hidden": int(parameters["summary_hidden"]),
        "horizon_embed_dim": int(parameters["horizon_embed_dim"]),
        "learning_rate": float(parameters["learning_rate"]),
        "weight_decay": float(parameters["weight_decay"]),
        "batch_size": int(parameters["batch_size"]),
        "positive_weight": float(parameters["positive_weight"]),
        "focal_gamma": float(parameters["focal_gamma"]),
        "auxiliary_5y_weight": float(parameters["auxiliary_5y_weight"]),
        "ranking_weight": float(parameters["ranking_weight"]),
        "smoothness_weight": float(parameters["smoothness_weight"]),
    }


def find_database() -> Path:
    if EXPECTED_DB.exists():
        return EXPECTED_DB

    candidates = sorted(
        PROJECT_DIR.glob(
            "rolling_5y_step10d_lstm_v2_tune_resume*/tuning/lstm_v2_optuna.sqlite3"
        )
    )
    if len(candidates) == 1:
        print(
            "警告：标准路径不存在，但自动找到唯一数据库：",
            candidates[0],
        )
        return candidates[0]

    if not candidates:
        raise FileNotFoundError(
            "没有找到LSTM-v2 Optuna数据库。\n"
            "预期路径：\n"
            f"{EXPECTED_DB}\n\n"
            "请先在服务器执行：\n"
            "find __CKD_WORKDIR__ -name "
            "'lstm_v2_optuna.sqlite3' -type f -print"
        )

    raise RuntimeError(
        "找到多个LSTM-v2 Optuna数据库，无法自动判断：\n"
        + "\n".join(str(path) for path in candidates)
    )


def choose_study(storage_url: str) -> optuna.Study:
    summaries = optuna.study.get_all_study_summaries(
        storage=storage_url
    )
    if not summaries:
        raise RuntimeError("数据库中没有任何Optuna Study。")

    names = [summary.study_name for summary in summaries]

    if EXPECTED_STUDY_NAME in names:
        return optuna.load_study(
            study_name=EXPECTED_STUDY_NAME,
            storage=storage_url,
        )

    if len(names) == 1:
        print(
            "警告：Study名称与日志不完全一致，"
            "但数据库中只有一个Study，自动使用：",
            names[0],
        )
        return optuna.load_study(
            study_name=names[0],
            storage=storage_url,
        )

    raise RuntimeError(
        "数据库中存在多个Study，且没有找到预期Study：\n"
        + "\n".join(names)
    )


def run() -> None:
    db_file = find_database()
    tuning_dir = db_file.parent
    storage_url = f"sqlite:///{db_file}"

    study = choose_study(storage_url)
    all_trials = list(study.trials)

    complete_trials = []
    for trial in all_trials:
        if trial.state != optuna.trial.TrialState.COMPLETE:
            continue
        if trial.value is None or not np.isfinite(float(trial.value)):
            continue

        mean_iauc = trial.user_attrs.get("mean_iauc", np.nan)
        mean_uno_c = trial.user_attrs.get("mean_uno_c", np.nan)

        if not np.isfinite(float(mean_iauc)):
            continue
        if not np.isfinite(float(mean_uno_c)):
            continue

        complete_trials.append(trial)

    if len(complete_trials) < 5:
        raise RuntimeError(
            f"仅找到{len(complete_trials)}个具有完整指标的COMPLETE Trial，"
            "不足以进行可靠选择。"
        )

    minimum_ibs = min(
        float(trial.value)
        for trial in complete_trials
    )

    eligible_trials = [
        trial
        for trial in complete_trials
        if float(trial.value)
        <= minimum_ibs + IBS_TOLERANCE
    ]

    eligible_trials.sort(
        key=lambda trial: (
            -float(trial.user_attrs["mean_iauc"]),
            -float(trial.user_attrs["mean_uno_c"]),
            float(trial.value),
            int(trial.number),
        )
    )

    selected_trial = eligible_trials[0]
    selected_parameters = normalize_parameters(
        selected_trial.params
    )

    ranking_rows = []
    for trial in complete_trials:
        ranking_rows.append(
            {
                "trial_number": int(trial.number),
                "mean_ibs": float(trial.value),
                "mean_iauc": float(
                    trial.user_attrs["mean_iauc"]
                ),
                "mean_uno_c": float(
                    trial.user_attrs["mean_uno_c"]
                ),
                "within_ibs_tolerance": bool(
                    float(trial.value)
                    <= minimum_ibs + IBS_TOLERANCE
                ),
                "selected": bool(
                    trial.number == selected_trial.number
                ),
                "fold_ibs": json.dumps(
                    trial.user_attrs.get("fold_ibs", []),
                    ensure_ascii=False,
                ),
                "fold_iauc": json.dumps(
                    trial.user_attrs.get("fold_iauc", []),
                    ensure_ascii=False,
                ),
                "fold_uno_c": json.dumps(
                    trial.user_attrs.get("fold_uno_c", []),
                    ensure_ascii=False,
                ),
                "fold_selected_snapshot_n": json.dumps(
                    trial.user_attrs.get(
                        "fold_selected_snapshot_n",
                        [],
                    ),
                    ensure_ascii=False,
                ),
            }
        )

    ranking_table = pd.DataFrame(ranking_rows)
    ranking_table = ranking_table.sort_values(
        by=[
            "within_ibs_tolerance",
            "mean_ibs",
            "mean_iauc",
            "mean_uno_c",
        ],
        ascending=[False, True, False, False],
    )
    ranking_table.to_csv(
        tuning_dir / RANKING_FILE.name,
        index=False,
        encoding="utf-8-sig",
    )

    study.trials_dataframe().to_csv(
        tuning_dir / ALL_TRIALS_FILE.name,
        index=False,
        encoding="utf-8-sig",
    )

    selected_mean_ibs = float(selected_trial.value)
    selected_mean_iauc = float(
        selected_trial.user_attrs["mean_iauc"]
    )
    selected_mean_uno_c = float(
        selected_trial.user_attrs["mean_uno_c"]
    )

    summary = {
        "study_name": study.study_name,
        "database_file": str(db_file),
        "planned_complete_trial_n": 30,
        "actual_complete_trial_n_used": len(complete_trials),
        "tuning_stopped_before_planned_target": True,
        "stop_reason": (
            "The original run reached its maximum trial-attempt limit "
            "before 30 complete trials. Existing complete trials were "
            "finalized without additional optimization."
        ),
        "objective": (
            "Five-fold mean of equal-weight six-landmark IBS."
        ),
        "selection_rule": {
            "minimum_observed_mean_ibs": minimum_ibs,
            "ibs_tolerance": IBS_TOLERANCE,
            "eligible_trial_n": len(eligible_trials),
            "rule": (
                "Among COMPLETE trials with mean IBS <= minimum IBS "
                "+ 0.00010, choose highest mean iAUC; "
                "then highest mean Uno C; then lower IBS."
            ),
        },
        "selected_trial_number": int(selected_trial.number),
        "selected_mean_ibs": selected_mean_ibs,
        "selected_mean_iauc": selected_mean_iauc,
        "selected_mean_uno_c": selected_mean_uno_c,
        "selected_parameters": selected_parameters,
        "fold_ibs": selected_trial.user_attrs.get(
            "fold_ibs",
            [],
        ),
        "fold_iauc": selected_trial.user_attrs.get(
            "fold_iauc",
            [],
        ),
        "fold_uno_c": selected_trial.user_attrs.get(
            "fold_uno_c",
            [],
        ),
        "fold_selected_snapshot_n": (
            selected_trial.user_attrs.get(
                "fold_selected_snapshot_n",
                [],
            )
        ),
        "reference_lstm_v1_mean_ibs": (
            REFERENCE_LSTM_V1_MEAN_IBS
        ),
        "selected_minus_reference_ibs": (
            selected_mean_ibs
            - REFERENCE_LSTM_V1_MEAN_IBS
        ),
        "locked_test_read": False,
        "database_modified": False,
    }

    save_json(
        summary,
        tuning_dir / SUMMARY_FILE.name,
    )

    print("\n" + "=" * 100)
    print("LSTM-v2现有Trial最终选择完成")
    print("=" * 100)
    print("数据库：", db_file)
    print("Study：", study.study_name)
    print("全部Trial数：", len(all_trials))
    print(
        "可用于最终选择的完整Trial数：",
        len(complete_trials),
    )
    print(
        "最低平均IBS：",
        f"{minimum_ibs:.9f}",
    )
    print(
        "IBS容忍范围内候选Trial数：",
        len(eligible_trials),
    )
    print(
        "最终选中Trial：",
        selected_trial.number,
    )
    print(
        "选中平均IBS：",
        f"{selected_mean_ibs:.9f}",
    )
    print(
        "选中平均iAUC：",
        f"{selected_mean_iauc:.9f}",
    )
    print(
        "选中平均Uno C：",
        f"{selected_mean_uno_c:.9f}",
    )
    print(
        "相对LSTM-v1的IBS差值：",
        f"{selected_mean_ibs - REFERENCE_LSTM_V1_MEAN_IBS:+.9f}",
    )
    print("\n选中参数：")
    print(
        json.dumps(
            selected_parameters,
            ensure_ascii=False,
            indent=2,
        )
    )
    print("\n已写入摘要：")
    print(tuning_dir / SUMMARY_FILE.name)
    print("\n已写入完整Trial排序：")
    print(tuning_dir / RANKING_FILE.name)
    print("\n锁定测试集：未读取")
    print("=" * 100)

    display_columns = [
        "trial_number",
        "mean_ibs",
        "mean_iauc",
        "mean_uno_c",
        "within_ibs_tolerance",
        "selected",
    ]
    print("\n按IBS排序的前10个完整Trial：")
    print(
        ranking_table.sort_values(
            "mean_ibs",
            ascending=True,
        )[display_columns]
        .head(10)
        .to_string(index=False)
    )


if __name__ == "__main__":
    run()