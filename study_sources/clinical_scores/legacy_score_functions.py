import numpy as np
import pandas as pd

def compute_dad_full(df):
    s = np.zeros(len(df))

    # IDU
    s += np.where(df["idu"] == 1, 2, 0)

    # HCV
    s += np.where(df["hcv"] == 1, 1, 0)

    # Age
    age = df["Age"]
    s += np.select(
        [age <= 35,
         (age > 35) & (age <= 50),
         (age > 50) & (age <= 60),
         age > 60],
        [0, 4, 7, 10]
    )

    # baseline eGFR
    egfr = df["eGFR"]
    s += np.select(
        [egfr > 90,
         (egfr > 70) & (egfr <= 90),
         (egfr > 60) & (egfr <= 70)],
        [-6, 0, 6],
        default=0
    )

    # Female
    s += np.where(df["female"] == 1, 1, 0)

    # CD4 >200
    s += np.where(df["cd4_nadir_proxy"] > 200, -1, 0)

    # HTN / CVD / DM
    s += np.where(df["htn"] == 1, 1, 0)
    s += np.where(df["cvd"] == 1, 1, 0)
    s += np.where(df["dm"] == 1, 2, 0)

    return s

def scherzer_ckd_points_no_nan(
    df: pd.DataFrame,
    col_age: str = "Age",
    col_glu: str = "GLU",
    col_tg: str = "TG",
    col_htn: str = "Hypertension",
    col_cd4: str = "CD4",
    glu_unit: str = "mmol/L",
    tg_unit: str = "mmol/L",
) -> pd.Series:
    """
    Scherzer score (points) — modified to NEVER output NaN:
    - Age missing/outside: assume no-risk (0 points)
    - GLU/TG/CD4 missing: assume no-risk for threshold components
    - HTN missing: assume no-risk (0)
    SBP & proteinuria are unavailable in your dataset → assumed no-risk (0)
    """
    n = len(df)
    score = np.zeros(n, dtype=float)

    # thresholds
    glu_thr = 7.8 if glu_unit.lower() in ["mmol/l", "mmol"] else 140.0
    tg_thr  = 2.26 if tg_unit.lower() in ["mmol/l", "mmol"] else 200.0

    # ---- Age points: 19–39=0; 40–49=2; 50–59=4; 60–90=6 ----
    age = pd.to_numeric(df[col_age], errors="coerce").values
    # 对缺失/超界：按0分（等价于低风险）
    age_pts = np.zeros(n, dtype=float)
    age_pts += np.where((age >= 40) & (age <= 49), 2, 0)
    age_pts += np.where((age >= 50) & (age <= 59), 4, 0)
    age_pts += np.where((age >= 60) & (age <= 90), 6, 0)
    # 注意：19–39默认0；age为NaN或<19或>90也保持0
    score += age_pts

    # ---- GLU > 140 mg/dL (≈7.8 mmol/L): yes=2 ----
    glu = pd.to_numeric(df[col_glu], errors="coerce").values
    score += np.where(glu > glu_thr, 2, 0)  # NaN 会自动走0

    # ---- SBP > 140: 你没有SBP，按0分 ----
    # score += 0

    # ---- Hypertension: yes=2 ----
    htn = pd.to_numeric(df[col_htn], errors="coerce").fillna(0).astype(int).values
    score += np.where(htn == 1, 2, 0)

    # ---- TG > 200 mg/dL (≈2.26 mmol/L): yes=1 ----
    tg = pd.to_numeric(df[col_tg], errors="coerce").values
    score += np.where(tg > tg_thr, 1, 0)

    # ---- Proteinuria: 你没有，按0分 ----
    # score += 0

    # ---- CD4 < 200: yes=1 ----
    cd4 = pd.to_numeric(df[col_cd4], errors="coerce").values
    score += np.where(cd4 < 200, 1, 0)  # NaN 会自动走0

    return pd.Series(score, index=df.index, name="Scherzer_score")