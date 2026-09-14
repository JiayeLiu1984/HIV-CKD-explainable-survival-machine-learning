import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

warnings.filterwarnings("ignore")

# =========================
# Configuration
# =========================
INPUT_DIR = "__CKD_LEGACY_WORKDIR__/split_data"
FEATURE_DIR = "__CKD_LEGACY_WORKDIR__/feature_selection"
SHAP_DIR = "__CKD_LEGACY_WORKDIR__/model_evaluation"
OUTPUT_DIR = "__CKD_LEGACY_WORKDIR__/model_evaluation/shap_figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Local explanation settings
HORIZONS = [12, 36, 60, 120]
SAMPLE_IDX = 0
TOP_K_GLOBAL = 10
TOP_K_LOCAL = 6

plt.rcParams.update({
    "font.family": "DejaVu Sans",   # 如有 Times New Roman 可改
    "font.size": 11,
    "axes.titlesize": 11,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 300,
    "savefig.dpi": 600
})

# =========================
# Load original data
# =========================
X_test = pd.read_csv(os.path.join(INPUT_DIR, "X_test.csv"))
features = pd.read_csv(os.path.join(FEATURE_DIR, "final_top_features.csv"))["Feature"].tolist()
X_test_sel = X_test[features].astype(np.float32)

# =========================
# 1. Global SHAP visualization
# =========================
global_shap_path = os.path.join(SHAP_DIR, "global_shap_values.csv")
global_importance_path = os.path.join(SHAP_DIR, "global_shap_importance.csv")

if not os.path.exists(global_shap_path):
    raise FileNotFoundError(f"File not found: {global_shap_path}")
if not os.path.exists(global_importance_path):
    raise FileNotFoundError(f"File not found: {global_importance_path}")

global_shap_df = pd.read_csv(global_shap_path)
global_importance_df = pd.read_csv(global_importance_path)

# 对齐解释样本数
n_explained = len(global_shap_df)
X_explain_global = X_test_sel.iloc[:n_explained].copy()

if global_shap_df.shape[1] != len(features):
    raise ValueError("The number of columns in global_shap_values.csv does not match the number of selected features.")

shap_values_global = global_shap_df[features].values

# -------------------------
# Global summary plot (bar)
# -------------------------
plt.figure(figsize=(10, 6))
shap.summary_plot(
    shap_values_global,
    X_explain_global,
    feature_names=features,
    plot_type="bar",
    show=False
)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "global_shap_summary_bar.png"), bbox_inches="tight")
plt.savefig(os.path.join(OUTPUT_DIR, "global_shap_summary_bar.pdf"), bbox_inches="tight")
plt.close()

# -------------------------
# Global summary plot (dot)
# -------------------------
plt.figure(figsize=(10, 7))
shap.summary_plot(
    shap_values_global,
    X_explain_global,
    feature_names=features,
    plot_type="dot",
    show=False
)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "global_shap_summary_dot.png"), bbox_inches="tight")
plt.savefig(os.path.join(OUTPUT_DIR, "global_shap_summary_dot.pdf"), bbox_inches="tight")
plt.close()

# -------------------------
# Global feature importance bar chart
# -------------------------
plot_df = global_importance_df.sort_values("MeanAbsSHAP", ascending=False).head(TOP_K_GLOBAL)

plt.figure(figsize=(8, 6))
plt.barh(plot_df["Feature"][::-1], plot_df["MeanAbsSHAP"][::-1])
plt.xlabel("Mean |SHAP value|")
plt.ylabel("Feature")
plt.title("Top global feature importance")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "global_feature_importance_top10.png"), bbox_inches="tight")
plt.savefig(os.path.join(OUTPUT_DIR, "global_feature_importance_top10.pdf"), bbox_inches="tight")
plt.close()

# =========================
# 2. Local SHAP visualization
# =========================
local_summary_path = os.path.join(SHAP_DIR, "local_shap_summary.csv")
if not os.path.exists(local_summary_path):
    raise FileNotFoundError(f"File not found: {local_summary_path}")

local_summary_df = pd.read_csv(local_summary_path)

# -------------------------
# Local predicted event probability over horizons
# -------------------------
plot_summary = local_summary_df.sort_values("time_months")

plt.figure(figsize=(8, 5))
plt.plot(
    plot_summary["time_months"],
    plot_summary["predicted_event_probability"],
    marker="o",
    linewidth=2
)
plt.xlabel("Time (months)")
plt.ylabel("Predicted event probability")
plt.title(f"Predicted event probability over time (Sample {SAMPLE_IDX})")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, f"sample{SAMPLE_IDX}_predicted_probability_curve.png"), bbox_inches="tight")
plt.savefig(os.path.join(OUTPUT_DIR, f"sample{SAMPLE_IDX}_predicted_probability_curve.pdf"), bbox_inches="tight")
plt.close()

# -------------------------
# Local top feature trajectories across horizons
# -------------------------
local_shap_tables = []
for t in HORIZONS:
    file_path = os.path.join(SHAP_DIR, f"local_shap_values_{t}m_sample{SAMPLE_IDX}.csv")
    if os.path.exists(file_path):
        df_t = pd.read_csv(file_path)
        df_t["time_months"] = t
        local_shap_tables.append(df_t)

if len(local_shap_tables) == 0:
    raise FileNotFoundError("No local SHAP result files were found.")

local_shap_all = pd.concat(local_shap_tables, axis=0, ignore_index=True)

# 根据跨时间平均绝对SHAP选出 top 特征
local_top_df = (
    local_shap_all.groupby("Feature")["SHAP"]
    .apply(lambda x: np.mean(np.abs(x)))
    .reset_index(name="MeanAbsSHAP")
    .sort_values("MeanAbsSHAP", ascending=False)
)

top_local_features = local_top_df["Feature"].head(TOP_K_LOCAL).tolist()

plt.figure(figsize=(10, 6))
for feat in top_local_features:
    df_feat = local_shap_all[local_shap_all["Feature"] == feat].sort_values("time_months")
    plt.plot(df_feat["time_months"], df_feat["SHAP"], marker="o", linewidth=2, label=feat)

plt.axhline(0, linestyle="--", linewidth=1)
plt.xlabel("Time (months)")
plt.ylabel("SHAP value")
plt.title(f"Top local feature contributions over time (Sample {SAMPLE_IDX})")
plt.legend(frameon=False, fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, f"sample{SAMPLE_IDX}_top_local_feature_trajectories.png"), bbox_inches="tight")
plt.savefig(os.path.join(OUTPUT_DIR, f"sample{SAMPLE_IDX}_top_local_feature_trajectories.pdf"), bbox_inches="tight")
plt.close()

# -------------------------
# Waterfall plots for each horizon
# -------------------------
for t in HORIZONS:
    file_path = os.path.join(SHAP_DIR, f"local_shap_values_{t}m_sample{SAMPLE_IDX}.csv")
    if not os.path.exists(file_path):
        continue

    df_local = pd.read_csv(file_path)
    df_local = df_local.set_index("Feature").loc[features].reset_index()

    # base value and predicted probability
    row_summary = local_summary_df[local_summary_df["time_months"] == t]
    if row_summary.empty:
        continue

    base_value = float(row_summary["base_value"].iloc[0])

    exp = shap.Explanation(
        values=df_local["SHAP"].values,
        base_values=base_value,
        data=df_local["Value"].values,
        feature_names=df_local["Feature"].tolist()
    )

    plt.figure(figsize=(10, 6))
    shap.plots.waterfall(exp, show=False, max_display=15)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"sample{SAMPLE_IDX}_waterfall_{t}m.png"), bbox_inches="tight")
    plt.savefig(os.path.join(OUTPUT_DIR, f"sample{SAMPLE_IDX}_waterfall_{t}m.pdf"), bbox_inches="tight")
    plt.close()

# =========================
# 3. Save plotting metadata
# =========================
meta = pd.DataFrame({
    "item": ["sample_idx", "top_k_global", "top_k_local", "n_global_samples"],
    "value": [SAMPLE_IDX, TOP_K_GLOBAL, TOP_K_LOCAL, n_explained]
})
meta.to_csv(os.path.join(OUTPUT_DIR, "shap_visualization_metadata.csv"), index=False)

print("SHAP visualization completed successfully.")
print(f"Figures saved to: {OUTPUT_DIR}")