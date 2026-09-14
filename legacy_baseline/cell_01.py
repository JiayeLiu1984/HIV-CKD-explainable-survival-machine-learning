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
INPUT_DIR = "__CKD_LEGACY_WORKDIR__/split_data"
OUTPUT_DIR = "__CKD_LEGACY_WORKDIR__/feature_selection"

RANDOM_STATE = 42
N_ESTIMATORS = 200
CORR_THRESHOLD = 0.70
TOP_N_FEATURES = 10

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================
# Load split datasets
# =========================
X_train = pd.read_csv(os.path.join(INPUT_DIR, "X_train.csv"), encoding="utf-8-sig")
X_test = pd.read_csv(os.path.join(INPUT_DIR, "X_test.csv"), encoding="utf-8-sig")
y_train = pd.read_csv(os.path.join(INPUT_DIR, "y_train.csv"), encoding="utf-8-sig")
y_test = pd.read_csv(os.path.join(INPUT_DIR, "y_test.csv"), encoding="utf-8-sig")

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

# =========================
# Build survival outcomes
# event: 1 = event occurred, 0 = censored
# =========================
y_train_surv = Surv.from_arrays(
    event=y_train["CKDstatus"].astype(bool),
    time=y_train["interval"].astype(float)
)

y_test_surv = Surv.from_arrays(
    event=y_test["CKDstatus"].astype(bool),
    time=y_test["interval"].astype(float)
)

# Convert predictors to float32 for compatibility
X_train = X_train.astype(np.float32)
X_test = X_test.astype(np.float32)

# =========================
# Fit RSF using all predictors
# =========================
rsf_full = RandomSurvivalForest(
    n_estimators=N_ESTIMATORS,
    random_state=RANDOM_STATE,
    n_jobs=4
)

rsf_full.fit(X_train, y_train_surv)

train_cindex_full = rsf_full.score(X_train, y_train_surv)
test_cindex_full = rsf_full.score(X_test, y_test_surv)

print("\nFull-model performance")
print(f"Training C-index: {train_cindex_full:.4f}")
print(f"Testing  C-index: {test_cindex_full:.4f}")

# =========================
# Permutation importance
# =========================
perm = PermutationImportance(
    rsf_full,
    n_iter=20,
    random_state=RANDOM_STATE
)

perm.fit(X_test, y_test_surv)

feature_importance = pd.DataFrame({
    "Feature": X_train.columns,
    "Weight": perm.feature_importances_,
    "Std": perm.feature_importances_std_
}).sort_values(by="Weight", ascending=False).reset_index(drop=True)

feature_importance.to_csv(
    os.path.join(OUTPUT_DIR, "feature_importance.csv"),
    index=False,
    encoding="utf-8-sig"
)

print("\nTop features by permutation importance:")
print(feature_importance.head(10))

# =========================
# Remove collinearity using Spearman correlation
# =========================
ranked_features = feature_importance["Feature"].tolist()
corr_abs = X_train[ranked_features].corr(method="spearman").abs()

selected_after_corr = []
removed_due_to_corr = []

for feature in ranked_features:
    if len(selected_after_corr) == 0:
        selected_after_corr.append(feature)
    else:
        max_corr = corr_abs.loc[feature, selected_after_corr].max()
        if max_corr >= CORR_THRESHOLD:
            correlated_feature = corr_abs.loc[feature, selected_after_corr].idxmax()
            removed_due_to_corr.append({
                "removed_feature": feature,
                "correlated_with": correlated_feature,
                "abs_spearman_rho": max_corr
            })
        else:
            selected_after_corr.append(feature)

selected_after_corr_df = pd.DataFrame({
    "selected_feature": selected_after_corr
})

removed_due_to_corr_df = pd.DataFrame(removed_due_to_corr)

selected_after_corr_df.to_csv(
    os.path.join(OUTPUT_DIR, "selected_features_after_corr_filter.csv"),
    index=False,
    encoding="utf-8-sig"
)

removed_due_to_corr_df.to_csv(
    os.path.join(OUTPUT_DIR, "removed_features_due_to_corr.csv"),
    index=False,
    encoding="utf-8-sig"
)

print(f"\nNumber of features before correlation filtering: {len(ranked_features)}")
print(f"Number of features after correlation filtering: {len(selected_after_corr)}")

# =========================
# Sequential feature addition
# =========================
sequential_results = []

for i in range(1, len(selected_after_corr) + 1):
    current_features = selected_after_corr[:i]
    last_added = current_features[-1]

    rsf_seq = RandomSurvivalForest(
        n_estimators=N_ESTIMATORS,
        random_state=RANDOM_STATE,
        n_jobs=-1
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
        f"[{i:02d}/{len(selected_after_corr)}] "
        f"+ {last_added} | "
        f"Train C-index: {train_cindex:.4f} | "
        f"Test C-index: {test_cindex:.4f}"
    )

df_seq = pd.DataFrame(sequential_results)

df_seq.to_csv(
    os.path.join(OUTPUT_DIR, "sequential_feature_selection_results.csv"),
    index=False,
    encoding="utf-8-sig"
)

# =========================
# Select top N features
# =========================
final_features = selected_after_corr[:TOP_N_FEATURES]

final_features_df = pd.DataFrame({
    "Feature": final_features
})

final_features_df.to_csv(
    os.path.join(OUTPUT_DIR, "final_top_features.csv"),
    index=False,
    encoding="utf-8-sig"
)

print(f"\nFinal top {TOP_N_FEATURES} features:")
print(final_features)

# =========================
# Plot sequential C-index curve
# =========================
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
    x=min(TOP_N_FEATURES, len(df_seq)),
    linestyle=":",
    linewidth=2,
    label=f"Selected features = {min(TOP_N_FEATURES, len(df_seq))}"
)

plt.xticks(
    df_seq["Feature_Count"],
    df_seq["Last_Added"],
    rotation=90,
    fontsize=9
)

ax = plt.gca()
for tick_value, tick_label in zip(df_seq["Feature_Count"], ax.get_xticklabels()):
    if tick_value <= min(TOP_N_FEATURES, len(df_seq)):
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
    os.path.join(OUTPUT_DIR, "cindex_vs_sequential_features.pdf"),
    bbox_inches="tight",
    dpi=600
)

plt.savefig(
    os.path.join(OUTPUT_DIR, "cindex_vs_sequential_features.png"),
    bbox_inches="tight",
    dpi=600
)

plt.show()

print("\nFeature selection module completed successfully.")
print(f"Results saved to: {OUTPUT_DIR}")