"""
Module 2: Feature selection
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sksurv.util import Surv
from sksurv.ensemble import RandomSurvivalForest
from eli5.sklearn import PermutationImportance

warnings.filterwarnings("ignore")
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["axes.unicode_minus"] = False


# =========================
# Configuration
# =========================
INPUT_DIR = os.path.join("data", "split_data")
OUTPUT_DIR = os.path.join("results", "feature_selection")

TIME_COL = "interval"
EVENT_COL = "CKDstatus"

RANDOM_STATE = 42
N_ESTIMATORS = 200
CORR_THRESHOLD = 0.70
TOP_N_FEATURES = 10
N_JOBS_FULL = 4
N_JOBS_SEQ = -1
PERMUTATION_ITER = 20


# =========================
# Utility functions
# =========================
def load_split_data(input_dir: str):
    """Load training and testing datasets."""
    X_train = pd.read_csv(os.path.join(input_dir, "X_train.csv"), encoding="utf-8-sig")
    X_test = pd.read_csv(os.path.join(input_dir, "X_test.csv"), encoding="utf-8-sig")
    y_train = pd.read_csv(os.path.join(input_dir, "y_train.csv"), encoding="utf-8-sig")
    y_test = pd.read_csv(os.path.join(input_dir, "y_test.csv"), encoding="utf-8-sig")

    return X_train, X_test, y_train, y_test


def validate_survival_data(y_df: pd.DataFrame, time_col: str, event_col: str) -> None:
    """Validate survival outcome columns."""
    required_cols = [time_col, event_col]
    missing_cols = [col for col in required_cols if col not in y_df.columns]
    if missing_cols:
        raise ValueError(f"Missing required outcome columns: {missing_cols}")

    if y_df[time_col].isna().any():
        raise ValueError(f"Column '{time_col}' contains missing values.")

    if y_df[event_col].isna().any():
        raise ValueError(f"Column '{event_col}' contains missing values.")

    if not set(y_df[event_col].dropna().unique()).issubset({0, 1}):
        raise ValueError(f"Column '{event_col}' must contain only 0 and 1.")

    if (pd.to_numeric(y_df[time_col], errors='raise') <= 0).any():
        raise ValueError(f"Column '{time_col}' must contain positive values.")


def build_survival_outcome(y_df: pd.DataFrame, time_col: str, event_col: str):
    """Construct structured survival outcome."""
    return Surv.from_arrays(
        event=y_df[event_col].astype(bool),
        time=y_df[time_col].astype(float)
    )


def save_dataframe(df: pd.DataFrame, output_dir: str, filename: str) -> None:
    """Save dataframe to CSV."""
    df.to_csv(
        os.path.join(output_dir, filename),
        index=False,
        encoding="utf-8-sig"
    )


def fit_full_rsf(X_train: pd.DataFrame, y_train_surv):
    """Fit a full RSF model using all predictors."""
    model = RandomSurvivalForest(
        n_estimators=N_ESTIMATORS,
        random_state=RANDOM_STATE,
        n_jobs=N_JOBS_FULL
    )
    model.fit(X_train, y_train_surv)
    return model


def compute_permutation_importance(model, X_train: pd.DataFrame, y_train_surv) -> pd.DataFrame:
    """
    Compute permutation importance on the training set.

    Note:
    To avoid information leakage, the testing set is not used in feature selection.
    """
    perm = PermutationImportance(
        model,
        n_iter=PERMUTATION_ITER,
        random_state=RANDOM_STATE
    )
    perm.fit(X_train, y_train_surv)

    importance_df = pd.DataFrame({
        "Feature": X_train.columns,
        "Weight": perm.feature_importances_,
        "Std": perm.feature_importances_std_
    }).sort_values(by="Weight", ascending=False).reset_index(drop=True)

    return importance_df


def filter_correlated_features(X_train: pd.DataFrame, ranked_features: list, corr_threshold: float):
    """Filter correlated predictors using absolute Spearman correlation."""
    corr_abs = X_train[ranked_features].corr(method="spearman").abs()

    selected_features = []
    removed_records = []

    for feature in ranked_features:
        if len(selected_features) == 0:
            selected_features.append(feature)
        else:
            max_corr = corr_abs.loc[feature, selected_features].max()
            if max_corr >= corr_threshold:
                correlated_feature = corr_abs.loc[feature, selected_features].idxmax()
                removed_records.append({
                    "removed_feature": feature,
                    "correlated_with": correlated_feature,
                    "abs_spearman_rho": max_corr
                })
            else:
                selected_features.append(feature)

    selected_df = pd.DataFrame({"selected_feature": selected_features})
    removed_df = pd.DataFrame(removed_records)

    return selected_features, selected_df, removed_df


def run_sequential_feature_addition(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train_surv,
    y_test_surv,
    selected_features: list
) -> pd.DataFrame:
    """
    Evaluate performance as predictors are added sequentially.

    The testing set is used here only for reporting performance trends,
    not for re-ranking predictors or redefining the final feature list.
    """
    sequential_results = []

    for i in range(1, len(selected_features) + 1):
        current_features = selected_features[:i]
        last_added = current_features[-1]

        rsf_seq = RandomSurvivalForest(
            n_estimators=N_ESTIMATORS,
            random_state=RANDOM_STATE,
            n_jobs=N_JOBS_SEQ
        )
        rsf_seq.fit(X_train[current_features], y_train_surv)

        train_cindex = rsf_seq.score(X_train[current_features], y_train_surv)
        test_cindex = rsf_seq.score(X_test[current_features], y_test_surv)

        sequential_results.append({
            "Feature_Count": i,
            "Last_Added": last_added,
            "Train_C_Index": train_cindex,
            "Test_C_Index": test_cindex
        })

        print(
            f"[{i:02d}/{len(selected_features)}] "
            f"+ {last_added} | "
            f"Train C-index: {train_cindex:.4f} | "
            f"Test C-index: {test_cindex:.4f}"
        )

    return pd.DataFrame(sequential_results)


def plot_sequential_cindex(df_seq: pd.DataFrame, top_n_features: int, output_dir: str) -> None:
    """Plot C-index across sequentially added predictors."""
    selected_n = min(top_n_features, len(df_seq))

    plt.figure(figsize=(10, 6), facecolor="white")

    plt.plot(
        df_seq["Feature_Count"],
        df_seq["Train_C_Index"],
        marker="o",
        linewidth=2,
        label="Training C-index"
    )

    plt.plot(
        df_seq["Feature_Count"],
        df_seq["Test_C_Index"],
        marker="x",
        linestyle="--",
        linewidth=2,
        label="Testing C-index"
    )

    plt.axvline(
        x=selected_n,
        linestyle=":",
        linewidth=2,
        label=f"Selected features = {selected_n}"
    )

    plt.xticks(
        df_seq["Feature_Count"],
        df_seq["Last_Added"],
        rotation=90,
        fontsize=9
    )

    ax = plt.gca()
    for tick_value, tick_label in zip(df_seq["Feature_Count"], ax.get_xticklabels()):
        if tick_value <= selected_n:
            tick_label.set_color("red")
            tick_label.set_fontweight("bold")

    plt.xlabel("Sequentially added predictors", fontsize=12)
    plt.ylabel("C-index", fontsize=12)
    plt.title("C-index vs. Sequentially Added Predictors", fontsize=14)
    plt.legend(frameon=False)
    plt.grid(False)
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)
    plt.subplots_adjust(bottom=0.30)

    plt.savefig(
        os.path.join(output_dir, "cindex_vs_sequential_features.pdf"),
        bbox_inches="tight",
        dpi=600
    )
    plt.savefig(
        os.path.join(output_dir, "cindex_vs_sequential_features.png"),
        bbox_inches="tight",
        dpi=600
    )
    plt.close()


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load data
    X_train, X_test, y_train, y_test = load_split_data(INPUT_DIR)

    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)
    print("y_train shape:", y_train.shape)
    print("y_test shape:", y_test.shape)

    # Validate outcome data
    validate_survival_data(y_train, TIME_COL, EVENT_COL)
    validate_survival_data(y_test, TIME_COL, EVENT_COL)

    # Build survival outcomes
    y_train_surv = build_survival_outcome(y_train, TIME_COL, EVENT_COL)
    y_test_surv = build_survival_outcome(y_test, TIME_COL, EVENT_COL)

    # Convert predictors to float32
    X_train = X_train.apply(pd.to_numeric, errors="raise").astype(np.float32)
    X_test = X_test.apply(pd.to_numeric, errors="raise").astype(np.float32)

    # Fit full RSF
    rsf_full = fit_full_rsf(X_train, y_train_surv)

    train_cindex_full = rsf_full.score(X_train, y_train_surv)
    test_cindex_full = rsf_full.score(X_test, y_test_surv)

    print("\nFull-model performance")
    print(f"Training C-index: {train_cindex_full:.4f}")
    print(f"Testing  C-index: {test_cindex_full:.4f}")

    # Permutation importance
    feature_importance = compute_permutation_importance(rsf_full, X_train, y_train_surv)
    save_dataframe(feature_importance, OUTPUT_DIR, "feature_importance.csv")

    print("\nTop features by permutation importance:")
    print(feature_importance.head(10))

    # Correlation filtering
    ranked_features = feature_importance["Feature"].tolist()
    selected_after_corr, selected_after_corr_df, removed_due_to_corr_df = filter_correlated_features(
        X_train, ranked_features, CORR_THRESHOLD
    )

    save_dataframe(
        selected_after_corr_df,
        OUTPUT_DIR,
        "selected_features_after_corr_filter.csv"
    )
    save_dataframe(
        removed_due_to_corr_df,
        OUTPUT_DIR,
        "removed_features_due_to_corr.csv"
    )

    print(f"\nNumber of features before correlation filtering: {len(ranked_features)}")
    print(f"Number of features after correlation filtering: {len(selected_after_corr)}")

    # Sequential feature addition
    df_seq = run_sequential_feature_addition(
        X_train, X_test, y_train_surv, y_test_surv, selected_after_corr
    )
    save_dataframe(df_seq, OUTPUT_DIR, "sequential_feature_selection_results.csv")

    # Select final top N features
    final_features = selected_after_corr[:TOP_N_FEATURES]
    final_features_df = pd.DataFrame({"Feature": final_features})
    save_dataframe(final_features_df, OUTPUT_DIR, "final_top_features.csv")

    print(f"\nFinal top {TOP_N_FEATURES} features:")
    print(final_features)

    # Plot
    plot_sequential_cindex(df_seq, TOP_N_FEATURES, OUTPUT_DIR)

    print("\nFeature selection module completed successfully.")
    print(f"Results saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()