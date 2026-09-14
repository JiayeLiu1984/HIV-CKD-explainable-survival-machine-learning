"""
Module 03: Feature selection using RSF importance, correlation filtering,
and sequential feature addition
"""

import os
import warnings
import numpy as np
import pandas as pd

from sksurv.util import Surv
from sksurv.ensemble import RandomSurvivalForest
from sksurv.metrics import concordance_index_censored
from eli5.sklearn import PermutationImportance

warnings.filterwarnings("ignore")

# =========================
# Configuration
# =========================
INPUT_DIR = "__CKD_LEGACY_WORKDIR__/data/split_data"
OUTPUT_DIR = "__CKD_LEGACY_WORKDIR__/feature_selection"

X_TRAIN_PATH = os.path.join(INPUT_DIR, "X_train.csv")
X_TEST_PATH = os.path.join(INPUT_DIR, "X_test.csv")
Y_TRAIN_PATH = os.path.join(INPUT_DIR, "y_train.csv")
Y_TEST_PATH = os.path.join(INPUT_DIR, "y_test.csv")

TIME_COL = "interval"
EVENT_COL = "CKDstatus"

RANDOM_STATE = 42
N_ESTIMATORS = 200
CORR_THRESHOLD = 0.70
TOP_N_FEATURES = 10

os.makedirs(OUTPUT_DIR, exist_ok=True)


# =========================
# Utility functions
# =========================
def validate_inputs(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.DataFrame,
    y_test: pd.DataFrame
) -> None:
    """Validate input datasets."""
    if X_train.empty or X_test.empty:
        raise ValueError("X_train or X_test is empty.")

    if y_train.empty or y_test.empty:
        raise ValueError("y_train or y_test is empty.")

    for df_name, df in [("y_train", y_train), ("y_test", y_test)]:
        missing_cols = [col for col in [TIME_COL, EVENT_COL] if col not in df.columns]
        if missing_cols:
            raise ValueError(f"{df_name} is missing required columns: {missing_cols}")

    if X_train.shape[1] != X_test.shape[1]:
        raise ValueError("X_train and X_test have different numbers of features.")

    if list(X_train.columns) != list(X_test.columns):
        raise ValueError("X_train and X_test do not have identical feature columns.")

    if X_train.isna().any().any() or X_test.isna().any().any():
        raise ValueError("Missing values detected in X_train or X_test.")

    if y_train[[TIME_COL, EVENT_COL]].isna().any().any():
        raise ValueError("Missing values detected in y_train.")

    if y_test[[TIME_COL, EVENT_COL]].isna().any().any():
        raise ValueError("Missing values detected in y_test.")


def make_survival_object(y: pd.DataFrame):
    """Convert outcome dataframe to sksurv Surv object."""
    event = y[EVENT_COL].astype(bool).values
    time = y[TIME_COL].astype(float).values
    return Surv.from_arrays(event=event, time=time)


def compute_cindex(y_true, risk_scores):
    """Compute Harrell's C-index."""
    result = concordance_index_censored(
        y_true[EVENT_COL].astype(bool).values,
        y_true[TIME_COL].astype(float).values,
        risk_scores
    )
    return float(result[0])


def save_csv(df: pd.DataFrame, filename: str) -> None:
    """Save dataframe to CSV."""
    df.to_csv(
        os.path.join(OUTPUT_DIR, filename),
        index=False,
        encoding="utf-8-sig"
    )


# =========================
# Main function
# =========================
def main() -> None:
    # -------------------------
    # Load data
    # -------------------------
    X_train = pd.read_csv(X_TRAIN_PATH, encoding="utf-8-sig")
    X_test = pd.read_csv(X_TEST_PATH, encoding="utf-8-sig")
    y_train = pd.read_csv(Y_TRAIN_PATH, encoding="utf-8-sig")
    y_test = pd.read_csv(Y_TEST_PATH, encoding="utf-8-sig")

    # Ensure numeric predictors
    X_train = X_train.apply(pd.to_numeric, errors="raise")
    X_test = X_test.apply(pd.to_numeric, errors="raise")

    y_train[TIME_COL] = pd.to_numeric(y_train[TIME_COL], errors="raise")
    y_train[EVENT_COL] = pd.to_numeric(y_train[EVENT_COL], errors="raise").astype(int)

    y_test[TIME_COL] = pd.to_numeric(y_test[TIME_COL], errors="raise")
    y_test[EVENT_COL] = pd.to_numeric(y_test[EVENT_COL], errors="raise").astype(int)

    validate_inputs(X_train, X_test, y_train, y_test)

    y_train_surv = make_survival_object(y_train)

    # -------------------------
    # Step 1: Fit RSF using all features
    # -------------------------
    rsf = RandomSurvivalForest(
        n_estimators=N_ESTIMATORS,
        min_samples_split=10,
        min_samples_leaf=5,
        max_features="sqrt",
        n_jobs=-1,
        random_state=RANDOM_STATE
    )
    rsf.fit(X_train, y_train_surv)

    # -------------------------
    # Step 2: Permutation importance
    # -------------------------
    perm = PermutationImportance(
        rsf,
        random_state=RANDOM_STATE,
        n_iter=10
    )
    perm.fit(X_train, y_train_surv)

    importance_df = pd.DataFrame({
        "Feature": X_train.columns,
        "Importance": perm.feature_importances_
    }).sort_values(by="Importance", ascending=False).reset_index(drop=True)

    save_csv(importance_df, "permutation_importance.csv")

    ranked_features = importance_df["Feature"].tolist()

    # -------------------------
    # Step 3: Spearman correlation filtering
    # Retain more important feature when |rho| > threshold
    # -------------------------
    corr_matrix = X_train[ranked_features].corr(method="spearman")
    corr_df = corr_matrix.reset_index()
    save_csv(corr_df, "spearman_correlation.csv")

    selected_features = []
    removed_features = []

    for feature in ranked_features:
        keep_feature = True
        for kept in selected_features:
            rho = corr_matrix.loc[feature, kept]
            if abs(rho) > CORR_THRESHOLD:
                keep_feature = False
                removed_features.append({
                    "Removed_Feature": feature,
                    "Kept_Feature": kept,
                    "Spearman_rho": rho
                })
                break
        if keep_feature:
            selected_features.append(feature)

    removed_df = pd.DataFrame(removed_features)
    if not removed_df.empty:
        save_csv(removed_df, "removed_correlated_features.csv")

    # -------------------------
    # Step 4: Sequential feature addition
    # -------------------------
    sequential_results = []

    for i in range(1, len(selected_features) + 1):
        current_features = selected_features[:i]

        rsf_seq = RandomSurvivalForest(
            n_estimators=N_ESTIMATORS,
            min_samples_split=10,
            min_samples_leaf=5,
            max_features="sqrt",
            n_jobs=-1,
            random_state=RANDOM_STATE
        )
        rsf_seq.fit(X_train[current_features], y_train_surv)

        train_risk = rsf_seq.predict(X_train[current_features])
        test_risk = rsf_seq.predict(X_test[current_features])

        train_cindex = compute_cindex(y_train, train_risk)
        test_cindex = compute_cindex(y_test, test_risk)

        sequential_results.append({
            "n_features": i,
            "features": ", ".join(current_features),
            "train_cindex": train_cindex,
            "test_cindex": test_cindex
        })

    sequential_df = pd.DataFrame(sequential_results)
    save_csv(sequential_df, "sequential_cindex_results.csv")

    # -------------------------
    # Step 5: Select final top features
    # Here we use the predefined TOP_N_FEATURES
    # -------------------------
    final_top_features = selected_features[:TOP_N_FEATURES]
    final_features_df = pd.DataFrame({"Feature": final_top_features})
    save_csv(final_features_df, "final_top_features.csv")

    # -------------------------
    # Print summary
    # -------------------------
    print("Feature selection completed successfully.")
    print(f"Input directory: {INPUT_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Original number of features: {X_train.shape[1]}")
    print(f"Number of ranked features: {len(ranked_features)}")
    print(f"Number of features after correlation filtering: {len(selected_features)}")
    print(f"Final selected top features ({TOP_N_FEATURES}):")
    print(final_top_features)


if __name__ == "__main__":
    main()