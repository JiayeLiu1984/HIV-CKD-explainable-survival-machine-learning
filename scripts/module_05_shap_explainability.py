"""
Module 5: SHAP explainability
"""

import os
import json
import joblib
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

warnings.filterwarnings("ignore")
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["axes.unicode_minus"] = False


# =========================
# Configuration
# =========================
INPUT_DIR = os.path.join("data", "split_data")
FEATURE_FILE = os.path.join("results", "rsf_model", "selected_features_used_in_final_model.csv")
MODEL_FILE = os.path.join("results", "rsf_model", "best_rsf_model.joblib")
OUTPUT_DIR = os.path.join("results", "shap")

HORIZONS = [12, 36, 60, 120]

GLOBAL_BACKGROUND_SIZE = 50
GLOBAL_EXPLAIN_SIZE = 200
LOCAL_BACKGROUND_K = 10
LOCAL_NSAMPLES = 100
SAMPLE_IDX = 0

TOP_K_GLOBAL = 10
TOP_K_LOCAL = 6
RANDOM_STATE = 42


# =========================
# Utility functions
# =========================
def load_split_data(input_dir: str):
    """Load training and testing predictors."""
    X_train = pd.read_csv(os.path.join(input_dir, "X_train.csv"), encoding="utf-8-sig")
    X_test = pd.read_csv(os.path.join(input_dir, "X_test.csv"), encoding="utf-8-sig")
    return X_train, X_test


def load_selected_features(feature_file: str) -> list:
    """Load selected features used in the final model."""
    if not os.path.exists(feature_file):
        raise FileNotFoundError(f"Feature file not found: {feature_file}")

    feature_df = pd.read_csv(feature_file, encoding="utf-8-sig")
    if "Feature" not in feature_df.columns:
        raise ValueError("Feature file must contain a column named 'Feature'.")

    features = feature_df["Feature"].tolist()
    if len(features) == 0:
        raise ValueError("No features found in selected feature file.")

    return features


def validate_predictors(X_train: pd.DataFrame, X_test: pd.DataFrame, features: list) -> None:
    """Validate that all required features exist."""
    missing_train = [f for f in features if f not in X_train.columns]
    missing_test = [f for f in features if f not in X_test.columns]

    if missing_train:
        raise ValueError(f"Missing features in X_train: {missing_train}")
    if missing_test:
        raise ValueError(f"Missing features in X_test: {missing_test}")


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


def get_model_max_time(model):
    """Get maximum supported follow-up time from fitted RSF model if available."""
    if hasattr(model, "unique_times_"):
        valid_times = np.asarray(model.unique_times_, dtype=float)
        valid_times = valid_times[np.isfinite(valid_times)]
        if len(valid_times) > 0:
            return float(valid_times.max())
    return None


def make_model_predict_risk(model, features):
    """Create wrapper for RSF risk score prediction."""
    def model_predict_risk(data_to_predict):
        if isinstance(data_to_predict, np.ndarray):
            data_df = pd.DataFrame(data_to_predict, columns=features)
        else:
            data_df = data_to_predict.copy()[features]
        return model.predict(data_df)
    return model_predict_risk


def make_predict_event_prob_at_t(model, features, target_time):
    """Create wrapper for event probability prediction at a given horizon."""
    def predict_event_prob(data_to_predict):
        if isinstance(data_to_predict, np.ndarray):
            data_df = pd.DataFrame(data_to_predict, columns=features)
        else:
            data_df = data_to_predict.copy()[features]

        surv_fns = model.predict_survival_function(data_df)
        return np.array([1.0 - fn(target_time) for fn in surv_fns], dtype=float)

    return predict_event_prob


def plot_global_shap(shap_values_global, X_explain_global, features, output_dir):
    """Plot global SHAP summary figures."""
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
    plt.savefig(os.path.join(output_dir, "shap_summary_bar.png"), dpi=600, bbox_inches="tight")
    plt.savefig(os.path.join(output_dir, "shap_summary_bar.pdf"), bbox_inches="tight")
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
    plt.savefig(os.path.join(output_dir, "shap_summary_dot.png"), dpi=600, bbox_inches="tight")
    plt.savefig(os.path.join(output_dir, "shap_summary_dot.pdf"), bbox_inches="tight")
    plt.close()


def plot_global_importance(global_importance_df, output_dir, top_k=10):
    """Plot top global feature importance."""
    plot_df = global_importance_df.sort_values("MeanAbsSHAP", ascending=False).head(top_k)

    plt.figure(figsize=(8, 6))
    plt.barh(plot_df["Feature"][::-1], plot_df["MeanAbsSHAP"][::-1])
    plt.xlabel("Mean |SHAP value|")
    plt.ylabel("Feature")
    plt.title("Top global feature importance")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "global_feature_importance_top10.png"), bbox_inches="tight")
    plt.savefig(os.path.join(output_dir, "global_feature_importance_top10.pdf"), bbox_inches="tight")
    plt.close()


def plot_local_probability_curve(local_summary_df, sample_idx, output_dir):
    """Plot predicted event probability across horizons for one sample."""
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
    plt.title(f"Predicted event probability over time (Sample {sample_idx})")
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, f"sample{sample_idx}_predicted_probability_curve.png"),
        bbox_inches="tight"
    )
    plt.savefig(
        os.path.join(output_dir, f"sample{sample_idx}_predicted_probability_curve.pdf"),
        bbox_inches="tight"
    )
    plt.close()


def plot_local_feature_trajectories(local_shap_all, sample_idx, output_dir, top_k=6):
    """Plot top local feature trajectories across horizons."""
    local_top_df = (
        local_shap_all.groupby("Feature")["SHAP"]
        .apply(lambda x: np.mean(np.abs(x)))
        .reset_index(name="MeanAbsSHAP")
        .sort_values("MeanAbsSHAP", ascending=False)
    )

    top_local_features = local_top_df["Feature"].head(top_k).tolist()

    plt.figure(figsize=(10, 6))
    for feat in top_local_features:
        df_feat = local_shap_all[local_shap_all["Feature"] == feat].sort_values("time_months")
        plt.plot(df_feat["time_months"], df_feat["SHAP"], marker="o", linewidth=2, label=feat)

    plt.axhline(0, linestyle="--", linewidth=1)
    plt.xlabel("Time (months)")
    plt.ylabel("SHAP value")
    plt.title(f"Top local feature contributions over time (Sample {sample_idx})")
    plt.legend(frameon=False, fontsize=9)
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, f"sample{sample_idx}_top_local_feature_trajectories.png"),
        bbox_inches="tight"
    )
    plt.savefig(
        os.path.join(output_dir, f"sample{sample_idx}_top_local_feature_trajectories.pdf"),
        bbox_inches="tight"
    )
    plt.close()


def plot_local_waterfalls(local_summary_df, features, sample_idx, output_dir):
    """Plot local SHAP waterfall plots for each available horizon."""
    for _, row in local_summary_df.iterrows():
        t = int(row["time_months"])
        local_file = os.path.join(output_dir, f"local_shap_values_{t}m_sample{sample_idx}.csv")
        if not os.path.exists(local_file):
            continue

        df_local = pd.read_csv(local_file, encoding="utf-8-sig")
        df_local = df_local.set_index("Feature").loc[features].reset_index()

        exp = shap.Explanation(
            values=df_local["SHAP"].values,
            base_values=float(row["base_value"]),
            data=df_local["Value"].values,
            feature_names=df_local["Feature"].tolist()
        )

        plt.figure(figsize=(10, 6))
        shap.plots.waterfall(exp, show=False, max_display=15)
        plt.tight_layout()
        plt.savefig(
            os.path.join(output_dir, f"sample{sample_idx}_waterfall_{t}m.png"),
            bbox_inches="tight"
        )
        plt.savefig(
            os.path.join(output_dir, f"sample{sample_idx}_waterfall_{t}m.pdf"),
            bbox_inches="tight"
        )
        plt.close()


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load data and model
    X_train, X_test = load_split_data(INPUT_DIR)
    features = load_selected_features(FEATURE_FILE)
    validate_predictors(X_train, X_test, features)

    X_train_sel = X_train[features].apply(pd.to_numeric, errors="raise").astype(np.float32).copy()
    X_test_sel = X_test[features].apply(pd.to_numeric, errors="raise").astype(np.float32).copy()

    if not os.path.exists(MODEL_FILE):
        raise FileNotFoundError(f"Model file not found: {MODEL_FILE}")
    model = joblib.load(MODEL_FILE)

    model_max_time = get_model_max_time(model)

    print("Data loaded successfully.")
    print("X_train_sel:", X_train_sel.shape)
    print("X_test_sel :", X_test_sel.shape)
    print("Number of features:", len(features))
    if model_max_time is not None:
        print(f"Maximum supported follow-up time in model: {model_max_time:.2f} months")

    if SAMPLE_IDX < 0 or SAMPLE_IDX >= len(X_test_sel):
        raise IndexError(f"SAMPLE_IDX={SAMPLE_IDX} is out of range for X_test_sel with {len(X_test_sel)} rows.")

    # =========================
    # 1. Global SHAP
    # =========================
    print("\n[1/3] Running global SHAP for RSF risk score...")

    background_risk = shap.sample(
        X_train_sel,
        min(GLOBAL_BACKGROUND_SIZE, len(X_train_sel)),
        random_state=RANDOM_STATE
    )

    X_explain_global = shap.sample(
        X_test_sel,
        min(GLOBAL_EXPLAIN_SIZE, len(X_test_sel)),
        random_state=RANDOM_STATE
    ).copy()

    model_predict_risk = make_model_predict_risk(model, features)

    explainer_risk = shap.KernelExplainer(model_predict_risk, background_risk)
    shap_values_global = explainer_risk.shap_values(X_explain_global, nsamples=LOCAL_NSAMPLES)
    shap_values_global = np.asarray(shap_values_global)

    if shap_values_global.ndim != 2:
        raise ValueError(f"Unexpected global SHAP shape: {shap_values_global.shape}")

    # Save explained dataset to avoid sample mismatch in later plotting/use
    save_dataframe(X_explain_global.reset_index(drop=True), OUTPUT_DIR, "global_explain_dataset.csv")

    # Save global SHAP values
    shap_global_df = pd.DataFrame(shap_values_global, columns=features)
    save_dataframe(shap_global_df, OUTPUT_DIR, "global_shap_values.csv")

    # Save global feature importance
    global_importance_df = pd.DataFrame({
        "Feature": features,
        "MeanAbsSHAP": np.abs(shap_values_global).mean(axis=0)
    }).sort_values("MeanAbsSHAP", ascending=False)
    save_dataframe(global_importance_df, OUTPUT_DIR, "global_shap_importance.csv")

    # Plot global figures
    plot_global_shap(shap_values_global, X_explain_global, features, OUTPUT_DIR)
    plot_global_importance(global_importance_df, OUTPUT_DIR, top_k=TOP_K_GLOBAL)

    print("Global SHAP completed.")

    # =========================
    # 2. Local SHAP
    # =========================
    print("\n[2/3] Running local SHAP for multiple horizons...")

    background_prob = shap.kmeans(X_train_sel, LOCAL_BACKGROUND_K)
    sample_data = X_test_sel.iloc[[SAMPLE_IDX]].copy()

    local_summary_records = []
    local_shap_tables = []

    for t in HORIZONS:
        print(f"  Explaining local SHAP at t = {t} months...")

        if model_max_time is not None and t > model_max_time:
            print(f"  Skipped t = {t}: exceeds model max time {model_max_time:.2f}")
            continue

        predict_event_prob_t = make_predict_event_prob_at_t(model, features, t)

        try:
            explainer_local = shap.KernelExplainer(predict_event_prob_t, background_prob)
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

            pred_prob = float(predict_event_prob_t(sample_data)[0])

            local_df = pd.DataFrame({
                "Feature": features,
                "Value": sample_data.iloc[0].values,
                "SHAP": local_values
            }).sort_values("SHAP", key=lambda x: np.abs(x), ascending=False)

            save_dataframe(local_df, OUTPUT_DIR, f"local_shap_values_{t}m_sample{SAMPLE_IDX}.csv")

            local_summary_records.append({
                "sample_idx": SAMPLE_IDX,
                "time_months": t,
                "base_value": base_value,
                "predicted_event_probability": pred_prob
            })

            local_df_for_traj = local_df.copy()
            local_df_for_traj["time_months"] = t
            local_shap_tables.append(local_df_for_traj)

        except Exception as e:
            print(f"  Failed at t = {t} months: {e}")

    local_summary_df = pd.DataFrame(local_summary_records)
    save_dataframe(local_summary_df, OUTPUT_DIR, "local_shap_summary.csv")

    if len(local_shap_tables) > 0:
        local_shap_all = pd.concat(local_shap_tables, axis=0, ignore_index=True)
        save_dataframe(local_shap_all, OUTPUT_DIR, "local_shap_all_horizons.csv")

        plot_local_probability_curve(local_summary_df, SAMPLE_IDX, OUTPUT_DIR)
        plot_local_feature_trajectories(local_shap_all, SAMPLE_IDX, OUTPUT_DIR, top_k=TOP_K_LOCAL)
        plot_local_waterfalls(local_summary_df, features, SAMPLE_IDX, OUTPUT_DIR)

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
        "horizons_requested": HORIZONS,
        "n_features": len(features),
        "top_k_global": TOP_K_GLOBAL,
        "top_k_local": TOP_K_LOCAL,
        "model_max_time": model_max_time
    }
    save_json(meta, OUTPUT_DIR, "shap_metadata.json")

    print("\n[3/3] SHAP analysis completed successfully.")
    print(f"Results saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
