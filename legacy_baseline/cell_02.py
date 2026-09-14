import os
import json
import warnings
import joblib
import numpy as np
import pandas as pd

from sksurv.util import Surv
from sksurv.ensemble import RandomSurvivalForest
from sklearn.model_selection import GridSearchCV, KFold

warnings.filterwarnings("ignore")

# =========================
# Configuration
# =========================
INPUT_DIR = "__CKD_LEGACY_WORKDIR__/split_data"
FEATURE_DIR = "__CKD_LEGACY_WORKDIR__/feature_selection"
OUTPUT_DIR = "__CKD_LEGACY_WORKDIR__/rsf_model"

random_state = 42
N_FOLDS = 5

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================
# Load datasets
# =========================
X_train = pd.read_csv(os.path.join(INPUT_DIR, "X_train.csv"), encoding="utf-8-sig")
y_train = pd.read_csv(os.path.join(INPUT_DIR, "y_train.csv"), encoding="utf-8-sig")

# =========================
# Load selected features
# =========================
selected_features = pd.read_csv(
    os.path.join(FEATURE_DIR, "final_top_features.csv"),
    encoding="utf-8-sig"
)["Feature"].tolist()

X_train_sel = X_train[selected_features].astype(np.float32)

# =========================
# Build survival outcome
# =========================
y_train_surv = Surv.from_arrays(
    event=y_train["CKDstatus"].astype(bool),
    time=y_train["interval"].astype(float)
)

# =========================
# Define parameter grid
# Consistent with Supplementary Table S3
# =========================
param_grid = {
    "n_estimators": [100,300,500],
    "max_depth": [10, 12, 14],
    "min_samples_split": [4, 6, 10],
    "min_samples_leaf": [10, 20, 30],
    "max_features": [None,"sqrt","log2"],
    "max_samples": [None, 0.7],
    "bootstrap": [True]
}

# =========================
# Cross-validation strategy
# =========================
cv = KFold(
    n_splits=N_FOLDS,
    shuffle=True,
    random_state=random_state
)

# =========================
# Initialize model
# =========================
rsf = RandomSurvivalForest(
    random_state=random_state,
    n_jobs=1
)

# =========================
# Grid search
# Default score = concordance index
# =========================
grid_search = GridSearchCV(
    estimator=rsf,
    param_grid=param_grid,
    cv=cv,
    n_jobs=4,
    refit=True,
    verbose=1
)

grid_search.fit(X_train_sel, y_train_surv)

best_rsf = grid_search.best_estimator_
best_params = grid_search.best_params_
best_cv_score = grid_search.best_score_

# =========================
# Save model
# =========================
joblib.dump(best_rsf, os.path.join(OUTPUT_DIR, "best_rsf_model.joblib"))

# =========================
# Save selected features
# =========================
pd.DataFrame({
    "Feature": selected_features
}).to_csv(
    os.path.join(OUTPUT_DIR, "selected_features_used_in_final_model.csv"),
    index=False,
    encoding="utf-8-sig"
)

# =========================
# Save best parameters
# =========================
with open(os.path.join(OUTPUT_DIR, "best_params.json"), "w", encoding="utf-8") as f:
    json.dump(best_params, f, indent=4, ensure_ascii=False)

# =========================
# Save CV results
# =========================
pd.DataFrame(grid_search.cv_results_).to_csv(
    os.path.join(OUTPUT_DIR, "rsf_grid_search_results.csv"),
    index=False,
    encoding="utf-8-sig"
)

# =========================
# Save training summary
# =========================
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

summary_df.to_csv(
    os.path.join(OUTPUT_DIR, "rsf_training_summary.csv"),
    index=False,
    encoding="utf-8-sig"
)

# =========================
# Print summary
# =========================
print("RSF model training completed successfully.")
print(f"Output directory: {OUTPUT_DIR}")
print(f"Number of selected features: {len(selected_features)}")
print(f"Best 5-fold CV C-index: {best_cv_score:.4f}")
print("Best parameters:")
print(best_params)