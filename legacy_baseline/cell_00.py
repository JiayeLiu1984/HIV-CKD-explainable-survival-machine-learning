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

TIME_COL = "interval"
EVENT_COL = "CKDstatus"
ID_COL = "ID"

TEST_SIZE = 0.30
RANDOM_STATE = 42

# =========================
# Create output directory
# =========================
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================
# Load dataset
# =========================
df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")

# =========================
# Check required columns
# =========================
required_columns = [ID_COL, TIME_COL, EVENT_COL]
missing_columns = [col for col in required_columns if col not in df.columns]

if missing_columns:
    raise ValueError(f"Missing required columns: {missing_columns}")

# =========================
# Basic validation
# =========================
if df[TIME_COL].isna().any():
    raise ValueError(f"Column '{TIME_COL}' contains missing values.")

if df[EVENT_COL].isna().any():
    raise ValueError(f"Column '{EVENT_COL}' contains missing values.")

if not set(df[EVENT_COL].dropna().unique()).issubset({0, 1}):
    raise ValueError(f"Column '{EVENT_COL}' must contain only 0 and 1.")

# =========================
# Define predictors and outcomes
# =========================
drop_columns = [ID_COL, TIME_COL, EVENT_COL]
X = df.drop(columns=drop_columns).copy()
y = df[[TIME_COL, EVENT_COL]].copy()

# Convert columns to numeric format
X = X.apply(pd.to_numeric, errors="raise")
y[TIME_COL] = pd.to_numeric(y[TIME_COL], errors="raise")
y[EVENT_COL] = pd.to_numeric(y[EVENT_COL], errors="raise").astype(int)

# =========================
# Split dataset
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y[EVENT_COL]
)

# =========================
# Save split datasets
# =========================
X_train.to_csv(os.path.join(OUTPUT_DIR, "X_train.csv"), index=False, encoding="utf-8-sig")
X_test.to_csv(os.path.join(OUTPUT_DIR, "X_test.csv"), index=False, encoding="utf-8-sig")
y_train.to_csv(os.path.join(OUTPUT_DIR, "y_train.csv"), index=False, encoding="utf-8-sig")
y_test.to_csv(os.path.join(OUTPUT_DIR, "y_test.csv"), index=False, encoding="utf-8-sig")

# =========================
# Save survival labels
# =========================
train_surv_df = pd.DataFrame({
    "event": y_train[EVENT_COL].astype(int),
    "time": y_train[TIME_COL].astype(float)
})

test_surv_df = pd.DataFrame({
    "event": y_test[EVENT_COL].astype(int),
    "time": y_test[TIME_COL].astype(float)
})

train_surv_df.to_csv(os.path.join(OUTPUT_DIR, "y_train_survival.csv"), index=False, encoding="utf-8-sig")
test_surv_df.to_csv(os.path.join(OUTPUT_DIR, "y_test_survival.csv"), index=False, encoding="utf-8-sig")

# =========================
# Print summary
# =========================
print("Dataset split completed successfully.")
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