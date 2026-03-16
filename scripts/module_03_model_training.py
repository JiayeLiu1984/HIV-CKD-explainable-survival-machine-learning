"""
Module 3: Final RSF model training and hyperparameter tuning
"""

import os
import json
import warnings
import joblib
import numpy as np
import pandas as pd

from sksurv.util import Surv
from sksurv.ensemble import RandomSurvivalForest
from sklearn.model_selection import GridSearchCV, StratifiedKFold

warnings.filterwarnings("ignore")


# =========================
# Configuration
# =========================
INPUT_DIR = os.path.join("data", "split_data")
FEATURE_DIR = os.path.join("results", "feature_selection")
OUTPUT_DIR = os.path.join("results", "rsf_model")

TIME_COL = "interval"
EVENT_COL = "CKDstatus"

RANDOM_STATE = 42
N_FOLDS = 5
GRID_SEARCH_N_JOBS = 4
MODEL_N_JOBS = 1


# =========================
# Utility functions
# =========================
def load_training_data(input_dir: str):
    """Load training predictors and outcomes."""
    X_train = pd.read_csv(os.path.join(input_dir, "X_train.csv"), encoding="utf-8-sig")
    y_train = pd.read_csv(os.path.join(input_dir, "y_train.csv"), encoding="utf-8-sig")
    return X_train, y_train


def load_selected_features(feature_dir: str) -> list:
    """Load selected feature list from Module 2."""
    feature_path = os.path.join(feature_dir, "final_top_features.csv")
    if not os.path.exists(feature_path):
        raise FileNotFoundError(f"Feature file not found: {feature_path}")

    selected_features = pd.read_csv(feature_path, encoding="utf-8-sig")["Feature"].tolist()

    if len(selected_features) == 0:
        raise ValueError("No selected features were found in final_top_features.csv.")

    return selected_features


def validate_training_data(X_train: pd.DataFrame, y_train: pd.DataFrame, selected_features: list) -> None:
    """Validate training data and selected feature list."""
    required_outcome_cols = [TIME_COL, EVENT_COL]
    missing_outcome_cols = [col for col in required_outcome_cols if col not in y_train.columns]
    if missing_outcome_cols:
        raise ValueError(f"Missing required outcome columns: {missing_outcome_cols}")

    missing_feature_cols = [col for col in selected_features if col not in X_train.columns]
    if missing_feature_cols:
        raise ValueError(f"Selected features not found in X_train: {missing_feature_cols}")

    if y_train[TIME_COL].isna().any():
        raise ValueError(f"Column '{TIME_COL}' contains missing values.")

    if y_train[EVENT_COL].isna().any():
        raise ValueError(f"Column '{EVENT_COL}' contains missing values.")

    if not set(y_train[EVENT_COL].dropna().unique()).issubset({0, 1}):
        raise ValueError(f"Column '{EVENT_COL}' must contain only 0 and 1.")

    if (pd.to_numeric(y_train[TIME_COL], errors="raise") <= 0).any():
        raise ValueError(f"Column '{TIME_COL}' must contain positive values.")


def build_survival_outcome(y_train: pd.DataFrame):
    """Construct structured survival outcome."""
    return Surv.from_arrays(
        event=y_train[EVENT_COL].astype(bool),
        time=y_train[TIME_COL].astype(float)
    )


def save_dataframe(df: pd.DataFrame, output_dir: str, filename: str) -> None:
    """Save dataframe to CSV."""
    df.to_csv(
        os.path.join(output_dir, filename),
        index=False,
        encoding="utf-8-sig"
    )


def save_json(data: dict, output_dir: str, filename: str) -> None:
    """Save dictionary to JSON."""
    with open(os.path.join(output_dir, filename), "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load data
    X_train, y_train = load_training_data(INPUT_DIR)
    selected_features = load_selected_features(FEATURE_DIR)

    # Validate data
    validate_training_data(X_train, y_train, selected_features)

    # Subset selected predictors
    X_train_sel = X_train[selected_features].apply(pd.to_numeric, errors="raise").astype(np.float32)

    # Build survival outcome
    y_train_surv = build_survival_outcome(y_train)

    # Parameter grid
    # Consistent with Supplementary Table S3
    param_grid = {
        "n_estimators": [100, 300, 500],
        "max_depth": [10, 12, 14],
        "min_samples_split": [4, 6, 10],
        "min_samples_leaf": [10, 20, 30],
        "max_features": [None, "sqrt", "log2"],
        "max_samples": [None, 0.7],
        "bootstrap": [True]
    }

    # Stratified CV based on event indicator
    cv = StratifiedKFold(
        n_splits=N_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    # Base model
    rsf = RandomSurvivalForest(
        random_state=RANDOM_STATE,
        n_jobs=MODEL_N_JOBS
    )

    # Grid search
    # Default scoring in sksurv estimator is concordance index
    grid_search = GridSearchCV(
        estimator=rsf,
        param_grid=param_grid,
        cv=cv.split(X_train_sel, y_train[EVENT_COL]),
        n_jobs=GRID_SEARCH_N_JOBS,
        refit=True,
        verbose=1
    )

    grid_search.fit(X_train_sel, y_train_surv)

    best_rsf = grid_search.best_estimator_
    best_params = grid_search.best_params_
    best_cv_score = grid_search.best_score_

    # Save model
    joblib.dump(best_rsf, os.path.join(OUTPUT_DIR, "best_rsf_model.joblib"))

    # Save selected features actually used
    selected_features_df = pd.DataFrame({"Feature": selected_features})
    save_dataframe(
        selected_features_df,
        OUTPUT_DIR,
        "selected_features_used_in_final_model.csv"
    )

    # Save best parameters
    save_json(best_params, OUTPUT_DIR, "best_params.json")

    # Save CV results
    cv_results_df = pd.DataFrame(grid_search.cv_results_)
    save_dataframe(cv_results_df, OUTPUT_DIR, "rsf_grid_search_results.csv")

    # Save training summary
    summary_df = pd.DataFrame({
        "Metric": [
            "Number of selected features",
            "Best 5-fold CV C-index"
        ],
        "Value": [
            len(selected_features),
            best_cv_score
        ]
    })
    save_dataframe(summary_df, OUTPUT_DIR, "rsf_training_summary.csv")

    # Print summary
    print("RSF model training completed successfully.")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Number of selected features: {len(selected_features)}")
    print(f"Best 5-fold CV C-index: {best_cv_score:.4f}")
    print("Best parameters:")
    print(best_params)


if __name__ == "__main__":
    main()
