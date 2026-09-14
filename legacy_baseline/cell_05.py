import os
import joblib
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
MODEL_DIR = "__CKD_LEGACY_WORKDIR__/rsf_model"
OUTPUT_DIR = "__CKD_LEGACY_WORKDIR__/model_evaluation"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Local explanation horizons (months)
HORIZONS = [12, 36, 60, 120]

# Speed-control parameters
GLOBAL_BACKGROUND_SIZE = 50
GLOBAL_EXPLAIN_SIZE = 200     # 不建议直接解释整个测试集
LOCAL_BACKGROUND_K = 10
LOCAL_NSAMPLES = 100          # 可调大到 200，速度会更慢

# Sample index for local explanation
SAMPLE_IDX = 0

# =========================
# Load Data and Model
# =========================
X_train = pd.read_csv(os.path.join(INPUT_DIR, "X_train.csv"))
X_test = pd.read_csv(os.path.join(INPUT_DIR, "X_test.csv"))
features = pd.read_csv(os.path.join(FEATURE_DIR, "final_top_features.csv"))["Feature"].tolist()

# 检查特征是否存在
missing_train = [f for f in features if f not in X_train.columns]
missing_test = [f for f in features if f not in X_test.columns]
if missing_train:
    raise ValueError(f"Missing features in X_train: {missing_train}")
if missing_test:
    raise ValueError(f"Missing features in X_test: {missing_test}")

X_train_sel = X_train[features].astype(np.float32).copy()
X_test_sel = X_test[features].astype(np.float32).copy()

model = joblib.load(os.path.join(MODEL_DIR, "best_rsf_model.joblib"))

# 如果模型支持 unique_times_，可用于 horizon 检查
model_max_time = None
if hasattr(model, "unique_times_"):
    valid_times = np.asarray(model.unique_times_, dtype=float)
    valid_times = valid_times[np.isfinite(valid_times)]
    if len(valid_times) > 0:
        model_max_time = float(valid_times.max())

print("Data loaded successfully.")
print("X_train_sel:", X_train_sel.shape)
print("X_test_sel :", X_test_sel.shape)
print("Number of features:", len(features))
if model_max_time is not None:
    print(f"Maximum supported follow-up time in model: {model_max_time:.2f} months")


# =========================
# Wrapper functions
# =========================
def model_predict_risk(data_to_predict):
    """
    Return RSF risk score for global SHAP explanation.

    Parameters
    ----------
    data_to_predict : np.ndarray or pd.DataFrame

    Returns
    -------
    np.ndarray
        Predicted risk scores.
    """
    if isinstance(data_to_predict, np.ndarray):
        data_df = pd.DataFrame(data_to_predict, columns=features)
    else:
        data_df = data_to_predict.copy()
        data_df = data_df[features]

    return model.predict(data_df)


def predict_event_prob_at_t(data_to_predict, target_time):
    """
    Return predicted event probability at a specific time point:
    Risk(t) = 1 - S(t)

    Parameters
    ----------
    data_to_predict : np.ndarray or pd.DataFrame
    target_time : float

    Returns
    -------
    np.ndarray
    """
    if isinstance(data_to_predict, np.ndarray):
        data_df = pd.DataFrame(data_to_predict, columns=features)
    else:
        data_df = data_to_predict.copy()
        data_df = data_df[features]

    surv_fns = model.predict_survival_function(data_df)
    return np.array([1.0 - fn(target_time) for fn in surv_fns], dtype=float)


# =========================
# 1. Global SHAP (Risk Score)
# =========================
print("\n[1/3] Running global SHAP for RSF risk score...")

# 背景集建议来自训练集
background_risk = shap.sample(X_train_sel, min(GLOBAL_BACKGROUND_SIZE, len(X_train_sel)), random_state=42)

# 解释集不建议用全量测试集，取固定子集即可
X_explain_global = shap.sample(X_test_sel, min(GLOBAL_EXPLAIN_SIZE, len(X_test_sel)), random_state=42)

# 显式用 KernelExplainer，避免 shap.Explainer 自动猜测导致不稳
explainer_risk = shap.KernelExplainer(model_predict_risk, background_risk)
shap_values_global = explainer_risk.shap_values(X_explain_global, nsamples=LOCAL_NSAMPLES)

# 保证输出为二维数组
shap_values_global = np.asarray(shap_values_global)
if shap_values_global.ndim != 2:
    raise ValueError(f"Unexpected global SHAP shape: {shap_values_global.shape}")

# Save global SHAP values
shap_global_df = pd.DataFrame(shap_values_global, columns=features)
shap_global_df.to_csv(os.path.join(OUTPUT_DIR, "global_shap_values.csv"), index=False)

# Save feature importance
global_importance_df = pd.DataFrame({
    "Feature": features,
    "MeanAbsSHAP": np.abs(shap_values_global).mean(axis=0)
}).sort_values("MeanAbsSHAP", ascending=False)
global_importance_df.to_csv(os.path.join(OUTPUT_DIR, "global_shap_importance.csv"), index=False)

# Summary plot (bar)
plt.figure(figsize=(10, 6))
shap.summary_plot(
    shap_values_global,
    X_explain_global,
    feature_names=features,
    plot_type="bar",
    show=False
)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "shap_summary_bar.png"), dpi=600, bbox_inches="tight")
plt.close()

# Summary plot (dot)
plt.figure(figsize=(10, 7))
shap.summary_plot(
    shap_values_global,
    X_explain_global,
    feature_names=features,
    plot_type="dot",
    show=False
)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "shap_summary_dot.png"), dpi=600, bbox_inches="tight")
plt.close()

print("Global SHAP completed.")


# =========================
# 2. Local SHAP (Dynamic slices at multiple horizons)
# =========================
print("\n[2/3] Running local SHAP for multiple horizons...")

# 用训练集做背景压缩更合理
background_prob = shap.kmeans(X_train_sel, LOCAL_BACKGROUND_K)

if SAMPLE_IDX < 0 or SAMPLE_IDX >= len(X_test_sel):
    raise IndexError(f"SAMPLE_IDX={SAMPLE_IDX} is out of range for X_test_sel with {len(X_test_sel)} rows.")

sample_data = X_test_sel.iloc[[SAMPLE_IDX]].copy()

local_summary_records = []

for t in HORIZONS:
    print(f"  Explaining local SHAP at t = {t} months...")

    # 跳过超出模型时间范围的 horizon
    if model_max_time is not None and t > model_max_time:
        print(f"  Skipped t = {t}: exceeds model max time {model_max_time:.2f}")
        continue

    def f_t(x):
        return predict_event_prob_at_t(x, t)

    try:
        explainer_local = shap.KernelExplainer(f_t, background_prob)
        shap_values_local = explainer_local.shap_values(sample_data, nsamples=LOCAL_NSAMPLES)
        shap_values_local = np.asarray(shap_values_local)

        if shap_values_local.ndim == 2:
            local_values = shap_values_local[0]
        elif shap_values_local.ndim == 1:
            local_values = shap_values_local
        else:
            raise ValueError(f"Unexpected local SHAP shape at t={t}: {shap_values_local.shape}")

        base_value = explainer_local.expected_value
        if isinstance(base_value, (list, np.ndarray)):
            base_value = float(np.asarray(base_value).reshape(-1)[0])
        else:
            base_value = float(base_value)

        pred_prob = float(f_t(sample_data)[0])

        # Save local SHAP values
        local_df = pd.DataFrame({
            "Feature": features,
            "Value": sample_data.iloc[0].values,
            "SHAP": local_values
        }).sort_values("SHAP", key=lambda x: np.abs(x), ascending=False)

        local_df.to_csv(
            os.path.join(OUTPUT_DIR, f"local_shap_values_{t}m_sample{SAMPLE_IDX}.csv"),
            index=False
        )

        # Waterfall plot
        exp = shap.Explanation(
            values=local_values,
            base_values=base_value,
            data=sample_data.iloc[0].values,
            feature_names=features
        )

        plt.figure(figsize=(10, 6))
        shap.plots.waterfall(exp, show=False, max_display=15)
        plt.tight_layout()
        plt.savefig(
            os.path.join(OUTPUT_DIR, f"shap_waterfall_{t}m_sample{SAMPLE_IDX}.png"),
            dpi=600,
            bbox_inches="tight"
        )
        plt.close()

        local_summary_records.append({
            "sample_idx": SAMPLE_IDX,
            "time_months": t,
            "base_value": base_value,
            "predicted_event_probability": pred_prob
        })

    except Exception as e:
        print(f"  Failed at t = {t} months: {e}")

local_summary_df = pd.DataFrame(local_summary_records)
local_summary_df.to_csv(os.path.join(OUTPUT_DIR, "local_shap_summary.csv"), index=False)

print("Local SHAP completed.")


# =========================
# 3. Save metadata
# =========================
meta = {
    "global_background_size": min(GLOBAL_BACKGROUND_SIZE, len(X_train_sel)),
    "global_explain_size": min(GLOBAL_EXPLAIN_SIZE, len(X_test_sel)),
    "local_background_k": LOCAL_BACKGROUND_K,
    "local_nsamples": LOCAL_NSAMPLES,
    "sample_idx": SAMPLE_IDX,
    "horizons": HORIZONS,
    "n_features": len(features),
}

pd.Series(meta).to_json(os.path.join(OUTPUT_DIR, "shap_metadata.json"), indent=2)

print("\n[3/3] SHAP analysis completed successfully.")
print(f"Results saved to: {OUTPUT_DIR}")