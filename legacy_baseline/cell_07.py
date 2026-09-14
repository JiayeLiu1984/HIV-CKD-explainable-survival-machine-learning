"""
Module 02: Load analysis-ready dataset and perform train-test split
"""

import os
import warnings
import pandas as pd
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

# =========================
# Configuration
# =========================
DATA_PATH = "__CKD_LEGACY_WORKDIR__/example_dataset.csv"
OUTPUT_DIR = "__CKD_LEGACY_WORKDIR__/split_data"

ID_COL = "ID"
TIME_COL = "interval"
EVENT_COL = "CKDstatus"

TEST_SIZE = 0.30
RANDOM_STATE = 42


# =========================
# Utility functions
# =========================
def validate_dataset(df: pd.DataFrame) -> None:
    """Validate required columns and basic data integrity."""
    required_columns = [ID_COL, TIME_COL, EVENT_COL]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    if df[ID_COL].isna().any():
        raise ValueError(f"Column '{ID_COL}' contains missing values.")

    if df[ID_COL].duplicated().any():
        raise ValueError(f"Column '{ID_COL}' contains duplicated IDs.")

    if df[TIME_COL].isna().any():
        raise ValueError(f"Column '{TIME_COL}' contains missing values.")

    if df[EVENT_COL].isna().any():
        raise ValueError(f"Column '{EVENT_COL}' contains missing values.")

    if not set(df[EVENT_COL].dropna().unique()).issubset({0, 1}):
        raise ValueError(f"Column '{EVENT_COL}' must contain only 0 and 1.")

    df[TIME_COL] = pd.to_numeric(df[TIME_COL], errors="raise")
    if (df[TIME_COL] <= 0).any():
        raise ValueError(f"Column '{TIME_COL}' must contain positive follow-up time.")

    df[EVENT_COL] = pd.to_numeric(df[EVENT_COL], errors="raise").astype(int)


def save_dataframe(df: pd.DataFrame, output_dir: str, filename: str) -> None:
    """Save dataframe to CSV."""
    df.to_csv(
        os.path.join(output_dir, filename),
        index=False,
        encoding="utf-8-sig"
    )


# =========================
# Main function
# =========================
def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")

    validate_dataset(df)

    drop_columns = [ID_COL, TIME_COL, EVENT_COL]
    predictor_columns = [col for col in df.columns if col not in drop_columns]

    X = df[predictor_columns].copy()
    y = df[[TIME_COL, EVENT_COL]].copy()

    X = X.apply(pd.to_numeric, errors="raise")
    y[TIME_COL] = pd.to_numeric(y[TIME_COL], errors="raise")
    y[EVENT_COL] = pd.to_numeric(y[EVENT_COL], errors="raise").astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y[EVENT_COL]
    )

    save_dataframe(X_train, OUTPUT_DIR, "X_train.csv")
    save_dataframe(X_test, OUTPUT_DIR, "X_test.csv")
    save_dataframe(y_train, OUTPUT_DIR, "y_train.csv")
    save_dataframe(y_test, OUTPUT_DIR, "y_test.csv")

    train_survival = pd.DataFrame({
        "event": y_train[EVENT_COL].astype(int),
        "time": y_train[TIME_COL].astype(float)
    })
    test_survival = pd.DataFrame({
        "event": y_test[EVENT_COL].astype(int),
        "time": y_test[TIME_COL].astype(float)
    })

    save_dataframe(train_survival, OUTPUT_DIR, "y_train_survival.csv")
    save_dataframe(test_survival, OUTPUT_DIR, "y_test_survival.csv")

    print("Train-test split completed successfully.")
    print(f"Input dataset: {DATA_PATH}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Total samples: {len(df)}")
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    print(f"Number of predictors: {X.shape[1]}")
    print(f"Training events: {int(y_train[EVENT_COL].sum())}")
    print(f"Testing events: {int(y_test[EVENT_COL].sum())}")
    print("\nPredictor columns:")
    print(X.columns.tolist())


if __name__ == "__main__":
    main()