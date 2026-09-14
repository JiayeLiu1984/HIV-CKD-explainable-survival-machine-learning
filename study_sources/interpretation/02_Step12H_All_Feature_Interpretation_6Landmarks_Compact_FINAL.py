from pathlib import Path
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from IPython.display import display


# ============================================================
# 1. Project paths
# ============================================================
BASE = Path(
    os.getenv("CKD_LSTM_PROJECT_DIR", "__CKD_WORKDIR__")
)

INTERPRETATION_ROOT = Path(
    os.getenv(
        "CKD_MODEL_INTERPRETATION_ROOT",
        str(BASE / "rolling_5y_model_interpretation_final_6landmarks"),
    )
)
COMMON_FIGURE_DIR = INTERPRETATION_ROOT / "FIGURES_ALL"
OUTPUT_DIR = INTERPRETATION_ROOT / "02_ALL_FEATURE_INTERPRETATION"

for directory in [
    INTERPRETATION_ROOT,
    COMMON_FIGURE_DIR,
    OUTPUT_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)

STEP12A = INTERPRETATION_ROOT / "01_GLOBAL_IG"

if not (STEP12A / "ig_global_clinical_importance.csv").exists():
    raise FileNotFoundError(
        "未找到统一结果目录中的六Landmark Step12A正式IG结果。"
        "请先运行 01_Step12A_Formal_IG_6Landmarks_FINAL.py。"
    )

IG_SAMPLE_FILE = STEP12A / "ig_sample_level_clinical_attribution.csv"
IG_GLOBAL_FILE = STEP12A / "ig_global_clinical_importance.csv"

HEATMAP_PNG = COMMON_FIGURE_DIR / "Step12H_Figure_All_Features_Heatmap_Compact_6Landmarks.png"
HEATMAP_PDF = COMMON_FIGURE_DIR / "Step12H_Figure_All_Features_Heatmap_Compact_6Landmarks.pdf"
BEESWARM_PNG = COMMON_FIGURE_DIR / "Step12H_Figure_All_Features_5y_Beeswarm.png"
BEESWARM_PDF = COMMON_FIGURE_DIR / "Step12H_Figure_All_Features_5y_Beeswarm.pdf"

FEATURE_TYPE_FILE = OUTPUT_DIR / "all_feature_type_audit.csv"
GLOBAL_RANKING_FILE = OUTPUT_DIR / "all_feature_global_ranking_mean_0to5y.csv"
CONTINUOUS_RANKING_FILE = OUTPUT_DIR / "continuous_feature_ranking_5y.csv"
BINARY_RANKING_FILE = OUTPUT_DIR / "binary_feature_ranking_5y.csv"
GLOBAL_IMPORTANCE_FILE = OUTPUT_DIR / "all_feature_global_importance_with_types.csv"


# ============================================================
# 2. Validate and read frozen results
# ============================================================
required_files = {
    "ig_sample": IG_SAMPLE_FILE,
    "ig_global": IG_GLOBAL_FILE,
}

missing_files = [
    str(path)
    for path in required_files.values()
    if not path.exists()
]

if missing_files:
    raise FileNotFoundError(
        "以下结果文件不存在，请检查路径：\n"
        + "\n".join(missing_files)
    )

ig_sample = pd.read_csv(IG_SAMPLE_FILE, encoding="utf-8-sig")
ig_global = pd.read_csv(IG_GLOBAL_FILE, encoding="utf-8-sig")


def require_columns(table, required_columns, table_name):
    missing_columns = sorted(set(required_columns) - set(table.columns))
    if missing_columns:
        raise ValueError(
            f"{table_name}缺少字段：{missing_columns}"
        )


require_columns(
    ig_sample,
    [
        "patient_local",
        "landmark_year",
        "clinical_variable",
        "signed_ig",
        "absolute_ig",
        "current_model_value",
        "current_value_percentile",
    ],
    "ig_sample",
)

require_columns(
    ig_global,
    [
        "landmark_year",
        "clinical_variable",
        "mean_abs_ig",
    ],
    "ig_global",
)


# ============================================================
# 3. Convert data types
# ============================================================
ig_sample["clinical_variable"] = ig_sample["clinical_variable"].astype(str)
ig_global["clinical_variable"] = ig_global["clinical_variable"].astype(str)

for column in [
    "landmark_year",
    "signed_ig",
    "absolute_ig",
    "current_model_value",
    "current_value_percentile",
]:
    ig_sample[column] = pd.to_numeric(ig_sample[column], errors="coerce")

for column in ["landmark_year", "mean_abs_ig"]:
    ig_global[column] = pd.to_numeric(ig_global[column], errors="coerce")

ig_sample = ig_sample.dropna(subset=["clinical_variable", "landmark_year"])
ig_global = ig_global.dropna(subset=["clinical_variable", "landmark_year", "mean_abs_ig"])


# ============================================================
# 4. Display settings
# ============================================================
plt.rcParams.update(
    {
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "font.size": 9,
        "axes.titlesize": 10.5,
        "axes.labelsize": 9.5,
        "legend.fontsize": 8,
        "xtick.labelsize": 8.2,
        "ytick.labelsize": 7.4,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

COLORS = {
    "blue": "#0072B2",
    "green": "#009E73",
    "orange": "#D55E00",
    "purple": "#7B3294",
    "grey": "#6B6B6B",
    "light_grey": "#B0B0B0",
    "binary_absent": "#56B4E9",
    "binary_present": "#D55E00",
}

LANDMARKS = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
available_landmarks = sorted(
    ig_global["landmark_year"].dropna().unique().tolist()
)

missing_landmarks = [
    landmark
    for landmark in LANDMARKS
    if not any(
        np.isclose(
            landmark,
            observed,
        )
        for observed in available_landmarks
    )
]

unexpected_landmarks = [
    observed
    for observed in available_landmarks
    if not any(
        np.isclose(
            observed,
            landmark,
        )
        for landmark in LANDMARKS
    )
]

if missing_landmarks:
    raise ValueError(
        "六Landmark正式IG结果不完整，缺少："
        f"{missing_landmarks}。禁止插值，请先补算正式IG。"
    )

if unexpected_landmarks:
    print(
        "提示：IG文件还包含非主Landmark："
        f"{unexpected_landmarks}；本图仅使用0/1/2/3/4/5年。"
    )


# ============================================================
# 5. Pretty variable names
# ============================================================
DISPLAY_NAME_MAP = {
    "Age": "Age",
    "Sex": "Sex",
    "BMI": "BMI",
    "eGFR": "eGFR",
    "HIVRNA_log10": "HIV RNA (log10)",
    "CD4": "CD4",
    "CD8": "CD8",
    "CD4_CD8": "CD4/CD8",
    "WBC": "White blood cell count",
    "PLT": "Platelet count",
    "HB": "Hemoglobin",
    "GLU": "Glucose",
    "TC": "Total cholesterol",
    "TG": "Triglycerides",
    "HDL": "HDL",
    "LDL": "LDL",
    "ALT": "ALT",
    "AST": "AST",
    "Urea": "Urea",
    "ACR": "Urine albumin-to-creatinine ratio",
    "Marriage": "Marriage",
    "Course": "Disease course",
    "Oppinfection": "Opportunistic infection",
    "WHOstage": "WHO stage",
    "CVD_status": "Cardiovascular disease",
    "diabetes_status": "Diabetes",
    "hypertension_status": "Hypertension",
    "hypercholesterolemia_status": "Hypercholesterolemia",
    "antidiabetic_med": "Antidiabetic medication",
    "antihypertensive_med": "Antihypertensive medication",
    "antilipid_med": "Lipid-lowering medication",
    "HBV_status": "HBV coinfection",
    "HCV_status": "HCV coinfection",
}


def prettify_variable_name(name):
    if name in DISPLAY_NAME_MAP:
        return DISPLAY_NAME_MAP[name]

    cleaned = name
    cleaned = cleaned.replace("_status", "")
    cleaned = cleaned.replace("_med", " medication")
    cleaned = cleaned.replace("_cum_month", " cumulative exposure (months)")
    cleaned = cleaned.replace("HIVRNA", "HIV RNA")
    cleaned = cleaned.replace("NNRTI", "NNRTI")
    cleaned = cleaned.replace("INSTI", "INSTI")
    cleaned = cleaned.replace("FTC", "FTC")
    cleaned = cleaned.replace("TDF", "TDF")
    cleaned = cleaned.replace("TAF", "TAF")
    cleaned = cleaned.replace("BIC", "BIC")
    cleaned = cleaned.replace("PI", "PI")
    cleaned = cleaned.replace("HBV", "HBV")
    cleaned = cleaned.replace("HCV", "HCV")
    cleaned = cleaned.replace("_", " ")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    parts = []
    for token in cleaned.split(" "):
        if token.isupper() or any(char.isdigit() for char in token):
            parts.append(token)
        else:
            parts.append(token.capitalize())
    return " ".join(parts)


# ============================================================
# 6. Infer feature type from actual model inputs
# ============================================================
def infer_feature_type(variable_df, tolerance=1e-6):
    model_values = pd.to_numeric(
        variable_df["current_model_value"],
        errors="coerce",
    )
    percentile_values = pd.to_numeric(
        variable_df["current_value_percentile"],
        errors="coerce",
    )

    finite_model = model_values.dropna()
    finite_percentile = percentile_values.dropna()

    binary_flag = False
    if len(finite_model) > 0:
        rounded = np.rint(finite_model.to_numpy(dtype=float))
        if np.all(
            np.abs(finite_model.to_numpy(dtype=float) - rounded) <= tolerance
        ):
            unique_states = set(rounded.astype(int).tolist())
            if unique_states.issubset({0, 1}):
                binary_flag = True

    if binary_flag:
        feature_type = "binary"
    elif len(finite_percentile) > 0:
        feature_type = "continuous"
    elif len(finite_model) > 0:
        feature_type = "continuous"
    else:
        feature_type = "unknown"

    return pd.Series(
        {
            "feature_type": feature_type,
            "n_rows": int(len(variable_df)),
            "finite_model_value_n": int(model_values.notna().sum()),
            "finite_percentile_n": int(percentile_values.notna().sum()),
            "min_model_value": float(finite_model.min()) if len(finite_model) else np.nan,
            "max_model_value": float(finite_model.max()) if len(finite_model) else np.nan,
            "n_unique_model_value": int(finite_model.nunique(dropna=True)),
        }
    )


feature_type_rows = []
for variable_name, variable_df in ig_sample.groupby("clinical_variable", dropna=False):
    stats = infer_feature_type(variable_df)
    stats["clinical_variable"] = variable_name
    feature_type_rows.append(stats)

feature_type_table = pd.DataFrame(feature_type_rows)
feature_type_table["clinical_display"] = feature_type_table["clinical_variable"].map(prettify_variable_name)

feature_type_table = feature_type_table.sort_values(
    ["feature_type", "clinical_variable"],
    ascending=[True, True],
).reset_index(drop=True)


# ============================================================
# 7. Recalculate all-feature global importance
# ============================================================
ig_global = ig_global.merge(
    feature_type_table[
        ["clinical_variable", "clinical_display", "feature_type"]
    ],
    on="clinical_variable",
    how="left",
    validate="many_to_one",
)

ig_global["feature_type"] = ig_global["feature_type"].fillna("unknown")
ig_global["clinical_display"] = ig_global["clinical_display"].fillna(ig_global["clinical_variable"])

all_ig_denominator = (
    ig_global.groupby("landmark_year", as_index=False)
    .agg(all_variable_total_ig=("mean_abs_ig", "sum"))
)

ig_global = ig_global.merge(
    all_ig_denominator,
    on="landmark_year",
    how="left",
    validate="many_to_one",
)

if (ig_global["all_variable_total_ig"] <= 0).any():
    raise ValueError("存在landmark的all_variable_total_ig <= 0。")

ig_global["share_of_total_ig_pct"] = (
    100.0 * ig_global["mean_abs_ig"] / ig_global["all_variable_total_ig"]
)

ig_global["importance_rank"] = (
    ig_global.groupby("landmark_year")["mean_abs_ig"]
    .rank(method="min", ascending=False)
    .astype(int)
)

ranking_overall = (
    ig_global.loc[ig_global["landmark_year"].isin(LANDMARKS)]
    .groupby(["clinical_variable", "clinical_display", "feature_type"], as_index=False)
    .agg(
        mean_share_across_landmarks=("share_of_total_ig_pct", "mean"),
        mean_abs_ig_across_landmarks=("mean_abs_ig", "mean"),
    )
    .sort_values(
        ["mean_share_across_landmarks", "mean_abs_ig_across_landmarks", "clinical_display"],
        ascending=[False, False, True],
    )
    .reset_index(drop=True)
)
ranking_overall.insert(0, "display_rank", np.arange(1, len(ranking_overall) + 1))
variable_order = ranking_overall["clinical_display"].tolist()
order_lookup = dict(zip(ranking_overall["clinical_display"], ranking_overall["display_rank"]))

# Keep 5-y ranking for the 5-y beeswarm only.
ranking_5y = (
    ig_global.loc[np.isclose(ig_global["landmark_year"], 5.0)]
    .sort_values(["mean_abs_ig", "clinical_display"], ascending=[False, True])
    .reset_index(drop=True)
)
ranking_5y.insert(0, "display_rank", np.arange(1, len(ranking_5y) + 1))

feature_type_lookup = dict(
    zip(
        feature_type_table["clinical_variable"],
        feature_type_table["feature_type"],
    )
)

display_lookup = dict(
    zip(
        feature_type_table["clinical_variable"],
        feature_type_table["clinical_display"],
    )
)


# ============================================================
# 8. Save audit tables
# ============================================================
feature_type_table.to_csv(FEATURE_TYPE_FILE, index=False, encoding="utf-8-sig")
ranking_overall.to_csv(GLOBAL_RANKING_FILE, index=False, encoding="utf-8-sig")
ig_global.to_csv(GLOBAL_IMPORTANCE_FILE, index=False, encoding="utf-8-sig")

continuous_ranking_5y = ranking_5y.loc[
    ranking_5y["feature_type"].eq("continuous")
].copy()
continuous_ranking_5y.to_csv(CONTINUOUS_RANKING_FILE, index=False, encoding="utf-8-sig")

binary_ranking_5y = ranking_5y.loc[
    ranking_5y["feature_type"].eq("binary")
].copy()
binary_ranking_5y.to_csv(BINARY_RANKING_FILE, index=False, encoding="utf-8-sig")


# ============================================================
# 9. Figure 1: all-feature global importance heatmap
# ============================================================
importance_matrix = np.full((len(variable_order), len(LANDMARKS)), np.nan, dtype=float)

for row_index, variable in enumerate(variable_order):
    for col_index, landmark in enumerate(LANDMARKS):
        subset = ig_global.loc[
            ig_global["clinical_display"].eq(variable)
            & np.isclose(ig_global["landmark_year"], landmark)
        ]

        if len(subset) == 0:
            continue
        if len(subset) > 1:
            raise ValueError(
                f"{variable} 在 {landmark:g} 年landmark存在多条全局重要性记录。"
            )

        importance_matrix[row_index, col_index] = subset["share_of_total_ig_pct"].iloc[0]

finite_values = importance_matrix[np.isfinite(importance_matrix)]
if len(finite_values) == 0:
    raise ValueError("Heatmap没有有效数值。")

color_max = float(np.nanmax(finite_values))
if color_max <= 0:
    color_max = 1.0

heatmap_height = max(7.2, 0.155 * len(variable_order) + 1.35)
fig_heatmap, ax_heatmap = plt.subplots(figsize=(9.8, heatmap_height))

heatmap = ax_heatmap.imshow(
    importance_matrix,
    aspect="auto",
    cmap="YlGnBu",
    vmin=0,
    vmax=color_max,
    interpolation="nearest",
)

ax_heatmap.set_xticks(
    range(len(LANDMARKS)),
    [f"{landmark:g} y" for landmark in LANDMARKS],
)
ax_heatmap.set_yticks(range(len(variable_order)), variable_order)
ax_heatmap.set_xlabel("Prediction landmark after ART initiation")
ax_heatmap.set_ylabel("All model input features")
ax_heatmap.set_title(
    "Temporal evolution of global LSTM feature importance across all six prediction landmarks\n"
    "(features ordered by mean attribution magnitude across 0–5 years)"
)

for row_index in range(len(variable_order)):
    for col_index in range(len(LANDMARKS)):
        value = importance_matrix[row_index, col_index]
        if np.isfinite(value) and value >= 1.0:
            ax_heatmap.text(
                col_index,
                row_index,
                f"{value:.1f}",
                ha="center",
                va="center",
                fontsize=5.8,
                color=("white" if value > color_max * 0.55 else "black"),
            )

ax_heatmap.set_xticks(np.arange(-0.5, len(LANDMARKS), 1), minor=True)
ax_heatmap.set_yticks(np.arange(-0.5, len(variable_order), 1), minor=True)
ax_heatmap.grid(which="minor", color="white", linestyle="-", linewidth=0.25)
ax_heatmap.tick_params(which="minor", bottom=False, left=False)

colorbar_heatmap = fig_heatmap.colorbar(heatmap, ax=ax_heatmap, fraction=0.035, pad=0.02)
colorbar_heatmap.set_label("Share of total mean absolute IG (%)")

fig_heatmap.tight_layout()
fig_heatmap.savefig(HEATMAP_PNG, dpi=600, bbox_inches="tight", facecolor="white")
fig_heatmap.savefig(HEATMAP_PDF, bbox_inches="tight", facecolor="white")


# ============================================================
# 10. Prepare 5-year sample-level data for split beeswarm figure
# ============================================================
beeswarm_data = ig_sample.loc[
    np.isclose(ig_sample["landmark_year"], 5.0)
].copy()

beeswarm_data["feature_type"] = beeswarm_data["clinical_variable"].map(feature_type_lookup)
beeswarm_data["clinical_display"] = beeswarm_data["clinical_variable"].map(display_lookup)
beeswarm_data["feature_order"] = beeswarm_data["clinical_display"].map(order_lookup)

beeswarm_data = beeswarm_data.dropna(subset=["clinical_display", "feature_order"]).copy()
beeswarm_data["feature_order"] = beeswarm_data["feature_order"].astype(int)

continuous_order = (
    ranking_5y.loc[ranking_5y["feature_type"].eq("continuous"), "clinical_display"].tolist()
)
binary_order = (
    ranking_5y.loc[ranking_5y["feature_type"].eq("binary"), "clinical_display"].tolist()
)

continuous_data = beeswarm_data.loc[
    beeswarm_data["feature_type"].eq("continuous")
].copy()

binary_data = beeswarm_data.loc[
    beeswarm_data["feature_type"].eq("binary")
].copy()


def binary_state_from_model_value(values, label, tolerance=1e-6):
    numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    result = np.full(len(numeric), np.nan, dtype=float)

    finite = np.isfinite(numeric)
    if finite.sum() == 0:
        return result

    rounded = np.rint(numeric[finite])
    if np.max(np.abs(numeric[finite] - rounded)) > tolerance:
        raise ValueError(
            f"{label} 的current_model_value并非0/1二分类。"
        )

    unique_states = set(rounded.astype(int).tolist())
    if not unique_states.issubset({0, 1}):
        raise ValueError(
            f"{label} 出现0/1以外的状态：{sorted(unique_states)}"
        )

    result[finite] = rounded
    return result


if len(binary_data) > 0:
    binary_data["binary_state"] = np.nan
    for variable in binary_order:
        mask = binary_data["clinical_display"].eq(variable)
        binary_data.loc[mask, "binary_state"] = binary_state_from_model_value(
            binary_data.loc[mask, "current_model_value"],
            variable,
        )


# ============================================================
# 11. Figure 2: split all-feature beeswarm at 5-year landmark
# ============================================================
n_cont = len(continuous_order)
n_bin = len(binary_order)
max_rows = max(n_cont, n_bin, 1)
beeswarm_height = max(8.5, 0.28 * max_rows + 1.8)

fig_bee = plt.figure(figsize=(15.5, beeswarm_height), constrained_layout=True)
grid = GridSpec(
    1,
    2,
    figure=fig_bee,
    width_ratios=[1.35, 1.05],
)

ax_cont = fig_bee.add_subplot(grid[0, 0])
ax_bin = fig_bee.add_subplot(grid[0, 1])

# ---- continuous features panel ----
rng_cont = np.random.default_rng(20260816)
colour_mappable = None

for row_index, variable in enumerate(continuous_order):
    subset = continuous_data.loc[
        continuous_data["clinical_display"].eq(variable)
    ].copy()

    signed_ig = pd.to_numeric(subset["signed_ig"], errors="coerce").to_numpy(dtype=float)
    value_percentile = pd.to_numeric(subset["current_value_percentile"], errors="coerce").to_numpy(dtype=float)

    finite_signed = np.isfinite(signed_ig)
    jitter = np.clip(rng_cont.normal(0, 0.10, size=len(subset)), -0.28, 0.28)

    ordered = finite_signed & np.isfinite(value_percentile)
    missing_color = finite_signed & ~np.isfinite(value_percentile)

    if ordered.any():
        colour_mappable = ax_cont.scatter(
            signed_ig[ordered],
            row_index + jitter[ordered],
            c=value_percentile[ordered],
            cmap="viridis_r",
            vmin=0,
            vmax=1,
            s=10,
            alpha=0.58,
            linewidths=0,
            rasterized=True,
        )

    if missing_color.any():
        ax_cont.scatter(
            signed_ig[missing_color],
            row_index + jitter[missing_color],
            color=COLORS["light_grey"],
            s=10,
            alpha=0.45,
            linewidths=0,
            rasterized=True,
        )

ax_cont.axvline(0, color=COLORS["grey"], linestyle="--", linewidth=1)
ax_cont.set_yticks(range(n_cont), continuous_order)
if n_cont > 0:
    ax_cont.invert_yaxis()
ax_cont.set_xlabel("Signed IG contribution to subsequent 5-year CKD risk")
ax_cont.set_ylabel("Continuous model input features")
ax_cont.set_title("A  Continuous features")
ax_cont.grid(axis="x", linestyle="--", alpha=0.25)

if colour_mappable is not None:
    colorbar_cont = fig_bee.colorbar(
        colour_mappable,
        ax=ax_cont,
        fraction=0.035,
        pad=0.02,
    )
    colorbar_cont.set_ticks([0, 0.5, 1.0])
    colorbar_cont.set_ticklabels(["Low", "Median", "High"])
    colorbar_cont.set_label("Within-feature value percentile")

# ---- binary features panel ----
rng_bin = np.random.default_rng(20260817)

for row_index, variable in enumerate(binary_order):
    subset = binary_data.loc[
        binary_data["clinical_display"].eq(variable)
    ].copy()

    signed_ig = pd.to_numeric(subset["signed_ig"], errors="coerce").to_numpy(dtype=float)
    binary_state = pd.to_numeric(subset["binary_state"], errors="coerce").to_numpy(dtype=float)

    finite_signed = np.isfinite(signed_ig)
    jitter = np.clip(rng_bin.normal(0, 0.10, size=len(subset)), -0.28, 0.28)

    absent = finite_signed & np.isclose(binary_state, 0.0, equal_nan=False)
    present = finite_signed & np.isclose(binary_state, 1.0, equal_nan=False)
    unknown = finite_signed & ~(absent | present)

    if absent.any():
        ax_bin.scatter(
            signed_ig[absent],
            row_index + jitter[absent],
            color=COLORS["binary_absent"],
            s=11,
            alpha=0.55,
            linewidths=0,
            rasterized=True,
            label="0 / absent" if row_index == 0 else None,
        )

    if present.any():
        ax_bin.scatter(
            signed_ig[present],
            row_index + jitter[present],
            color=COLORS["binary_present"],
            s=11,
            alpha=0.60,
            linewidths=0,
            rasterized=True,
            label="1 / present" if row_index == 0 else None,
        )

    if unknown.any():
        ax_bin.scatter(
            signed_ig[unknown],
            row_index + jitter[unknown],
            color=COLORS["light_grey"],
            s=10,
            alpha=0.45,
            linewidths=0,
            rasterized=True,
            label="Unknown" if row_index == 0 else None,
        )

ax_bin.axvline(0, color=COLORS["grey"], linestyle="--", linewidth=1)
ax_bin.set_yticks(range(n_bin), binary_order)
if n_bin > 0:
    ax_bin.invert_yaxis()
ax_bin.set_xlabel("Signed IG contribution to subsequent 5-year CKD risk")
ax_bin.set_ylabel("Binary model input features")
ax_bin.set_title("B  Binary features")
ax_bin.grid(axis="x", linestyle="--", alpha=0.25)
ax_bin.legend(frameon=False, loc="lower right")

fig_bee.suptitle(
    "All-feature explanation of the dynamic LSTM at the 5-year landmark",
    fontsize=13,
    fontweight="bold",
)

fig_bee.savefig(BEESWARM_PNG, dpi=600, bbox_inches="tight", facecolor="white")
fig_bee.savefig(BEESWARM_PDF, bbox_inches="tight", facecolor="white")


# ============================================================
# 12. Save checks
# ============================================================
for output_file in [
    HEATMAP_PNG,
    HEATMAP_PDF,
    BEESWARM_PNG,
    BEESWARM_PDF,
    FEATURE_TYPE_FILE,
    GLOBAL_RANKING_FILE,
    CONTINUOUS_RANKING_FILE,
    BINARY_RANKING_FILE,
    GLOBAL_IMPORTANCE_FILE,
]:
    if (not output_file.exists()) or output_file.stat().st_size <= 0:
        raise RuntimeError(f"输出文件保存失败或为空：{output_file}")


# ============================================================
# 13. Display concise numerical results
# ============================================================
print("\nAll-feature interpretation outputs saved successfully:\n")
for output_file in [
    HEATMAP_PNG,
    HEATMAP_PDF,
    BEESWARM_PNG,
    BEESWARM_PDF,
    FEATURE_TYPE_FILE,
    GLOBAL_RANKING_FILE,
    CONTINUOUS_RANKING_FILE,
    BINARY_RANKING_FILE,
    GLOBAL_IMPORTANCE_FILE,
]:
    print(f"  {output_file} ({output_file.stat().st_size / 1024:.1f} KB)")

print("\nFeature-type audit:\n")
display(feature_type_table)

print("\nTop 20 all-feature ranking at the 5-year landmark:\n")
display(
    ranking_5y[
        [
            "display_rank",
            "clinical_display",
            "feature_type",
            "mean_abs_ig",
            "share_of_total_ig_pct",
            "importance_rank",
        ]
    ].head(20).round(
        {
            "mean_abs_ig": 6,
            "share_of_total_ig_pct": 2,
        }
    )
)

print("\nInterpretation note:\n")
print(
    "This script does not preselect 12 clinically preferred variables. "
    "Instead, it explains all model input features found in the frozen IG export. "
    "Global attribution is shown for all features across all six formal landmarks, and the 5-year "
    "sample-level direction plot is split into continuous and binary features. "
    "Feature type is inferred from the actual current_model_value/current_value_percentile "
    "content in ig_sample. If you later want to aggregate one-hot features or ART-related "
    "features back to broader clinical domains, please do that as a second-step summary "
    "on top of this complete feature-level explanation."
)

plt.show()
plt.close(fig_heatmap)
plt.close(fig_bee)
