"""
explain.py
----------
SHAP-based explainability for the saved best model.

Generates two kinds of explanations:

1. Global (model-level):
   - Summary plot      -- each feature's impact distribution across all samples
   - Feature importance bar -- mean |SHAP| per feature, ranked

2. Local (single-prediction):
   - Waterfall plot    -- shows which features pushed this prediction up/down
   - Force plot (HTML) -- interactive push/pull view

All plots saved under model/reports/shap/.
"""

import logging
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from ml.config import (
    ALL_FEATURES,
    BEST_MODEL_PATH,
    REPORT_DIR,
    SCALER_PATH,
    TARGET,
)
from ml.feature_engineering import load_and_engineer

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

# SHAP output dir
SHAP_DIR = REPORT_DIR / "shap"
SHAP_DIR.mkdir(parents=True, exist_ok=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_artefacts():
    """Load model + scaler, return (model, scaler)."""
    model  = joblib.load(BEST_MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    logger.info("Artefacts loaded: %s", type(model).__name__)
    return model, scaler


def _get_background_sample(n: int = 500, random_state: int = 42) -> pd.DataFrame:
    """
    Load the engineered feature matrix and return a stratified sample of n rows
    to use as the SHAP background / reference dataset.
    Sampling keeps the computation fast without losing coverage.
    """
    logger.info("Loading background sample (%d rows) for SHAP ...", n)
    df, _ = load_and_engineer()
    X = df[ALL_FEATURES]
    return X.sample(n=min(n, len(X)), random_state=random_state)


# ── 1. Global Explanations ─────────────────────────────────────────────────────

def global_explanation(sample_size: int = 500) -> None:
    """
    Compute SHAP values over a background sample and produce:
      - shap_summary.png          (beeswarm — shows direction + magnitude)
      - shap_feature_importance.png  (bar chart — mean |SHAP| ranked)
    """
    model, scaler = _load_artefacts()
    X_bg = _get_background_sample(sample_size)

    # Scale the background using the saved scaler
    X_bg_sc = pd.DataFrame(
        scaler.transform(X_bg), columns=ALL_FEATURES, index=X_bg.index
    )

    logger.info("Computing SHAP values (TreeExplainer) ...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_bg_sc)

    # For binary classifiers sklearn returns a list [class0_vals, class1_vals]
    # We care about class 1 (Present)
    if isinstance(shap_values, list):
        sv = shap_values[1]
    else:
        sv = shap_values

    feature_names = ALL_FEATURES

    # ── Beeswarm summary plot ──────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 7))
    shap.summary_plot(
        sv, X_bg_sc,
        feature_names=feature_names,
        show=False,
        plot_size=None,
    )
    plt.title("SHAP Summary — Feature Impact on Attendance Prediction", fontsize=13, pad=12)
    plt.tight_layout()
    out = SHAP_DIR / "shap_summary.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close("all")
    logger.info("Saved: %s", out)

    # ── Bar chart (mean |SHAP|) ────────────────────────────────────────────
    mean_abs = np.abs(sv).mean(axis=0)
    order = np.argsort(mean_abs)[::-1]
    ranked_features = [feature_names[i] for i in order]
    ranked_vals     = mean_abs[order]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(ranked_features[::-1], ranked_vals[::-1],
                   color="#4C72B0", edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Mean |SHAP Value|  (average impact on model output)", fontsize=11)
    ax.set_title("Feature Importance (SHAP)", fontsize=13, pad=12)
    ax.bar_label(bars, fmt="%.4f", padding=3, fontsize=9)
    ax.set_xlim(0, ranked_vals.max() * 1.18)
    plt.tight_layout()
    out = SHAP_DIR / "shap_feature_importance.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close("all")
    logger.info("Saved: %s", out)

    # Print ranked table to console
    print("\n" + "=" * 55)
    print("  FEATURE IMPORTANCE (SHAP)  |  Mean |SHAP Value|")
    print("=" * 55)
    for feat, val in zip(ranked_features, ranked_vals):
        bar = "#" * int(val / ranked_vals.max() * 30)
        print(f"  {feat:<28s} {val:.5f}  {bar}")
    print("=" * 55 + "\n")

    return sv, X_bg_sc, explainer


# ── 2. Local (per-prediction) Explanation ─────────────────────────────────────

def local_explanation(row_dict: dict, label: str = "sample") -> None:
    """
    Explain a single prediction with a SHAP waterfall plot.

    Parameters
    ----------
    row_dict : the same dict you'd pass to predict_attendance()
    label    : short name used in the saved filename (e.g. student_id)
    """
    from ml.predict import predict_attendance
    from ml.config import CAT_FEATURES, LABEL_ENCODERS_PATH

    model, scaler = _load_artefacts()
    encoders = joblib.load(LABEL_ENCODERS_PATH)

    # Build a one-row DataFrame, mirror the same logic as predict.py
    df = pd.DataFrame([row_dict])

    # Derive time features
    if "class_start_time" in df.columns:
        t = df["class_start_time"].iloc[0]
        if isinstance(t, str) and ":" in t:
            h, m = t.split(":")
            df["class_hour"] = int(h)
    if "class_end_time" in df.columns and "class_start_time" in df.columns:
        _s = df["class_start_time"].iloc[0]
        _e = df["class_end_time"].iloc[0]
        if isinstance(_s, str) and isinstance(_e, str):
            s_min = int(_s.split(":")[0]) * 60 + int(_s.split(":")[1])
            e_min = int(_e.split(":")[0]) * 60 + int(_e.split(":")[1])
            df["class_duration_min"] = max(0, e_min - s_min)

    for col in ["late_entry"]:
        if col not in df.columns:
            df[col] = 0

    for col in CAT_FEATURES:
        le = encoders[col]
        val = str(df[col].iloc[0])
        df[col] = le.transform([val])[0] if val in le.classes_ else -1

    for col in ALL_FEATURES:
        if col not in df.columns:
            df[col] = 0

    X = df[ALL_FEATURES].fillna(0)
    X_sc = pd.DataFrame(scaler.transform(X), columns=ALL_FEATURES)

    # Run prediction for display
    pred_result = predict_attendance(row_dict)
    pred_label  = pred_result["label"]
    pred_prob   = pred_result["probability_present"]

    # SHAP values for this row
    explainer   = shap.TreeExplainer(model)
    shap_vals   = explainer.shap_values(X_sc)

    if isinstance(shap_vals, list):
        sv_row = shap_vals[1][0]
    else:
        sv_row = shap_vals[0]

    # expected_value can be: float, np.ndarray size 1, or list/array of 2 values
    ev = explainer.expected_value
    if isinstance(ev, (list, np.ndarray)):
        ev = np.asarray(ev).ravel()
        expected_val = float(ev[1]) if len(ev) >= 2 else float(ev[0])
    else:
        expected_val = float(ev)

    expl_obj = shap.Explanation(
        values=sv_row,
        base_values=expected_val,
        data=X_sc.values[0],
        feature_names=ALL_FEATURES,
    )
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.waterfall_plot(expl_obj, show=False, max_display=14)
    plt.title(
        f"SHAP Waterfall — Student: {row_dict.get('student_id', label)}"
        f"  |  Predicted: {pred_label}  (P={pred_prob})",
        fontsize=12, pad=14,
    )
    plt.tight_layout()
    safe_label = str(label).replace("/", "_")
    out = SHAP_DIR / f"shap_waterfall_{safe_label}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close("all")
    logger.info("Saved: %s", out)

    # Console summary
    print("\n" + "=" * 55)
    print(f"  LOCAL EXPLANATION  |  {row_dict.get('student_id', label)}")
    print(f"  Predicted : {pred_label}  |  P(Present) = {pred_prob}")
    print("=" * 55)
    order = np.argsort(np.abs(sv_row))[::-1]
    for i in order[:8]:
        direction = "+" if sv_row[i] > 0 else "-"
        print(f"  {direction}  {ALL_FEATURES[i]:<28s}  SHAP={sv_row[i]:+.4f}  value={X_sc.values[0][i]:.3f}")
    print("=" * 55 + "\n")


# ── 3. Run both from main pipeline ───────────────────────────────────────────

def run_full_explanation(sample_size: int = 500) -> None:
    """
    Called by main.py --mode explain.
    Runs global explanation + 3 illustrative local explanations.
    """
    logger.info("=== SHAP Explanation Pipeline Started ===")

    # Global
    global_explanation(sample_size=sample_size)

    # Local examples: high / borderline / at-risk
    sample_cases = [
        {
            "label": "ST1007_BD403_high",
            "row": {
                "student_id": "ST1007", "semester": "4",
                "subject_code": "BD403", "faculty_id": "F105",
                "class_start_time": "09:00", "class_end_time": "10:00",
                "day_of_week": 1, "is_exam_week": 0,
                "rolling_attendance_pct": 95.5, "late_entry": 0,
                "day_of_month": 21, "month": 2,
                "prev_class_attended": 1, "consecutive_absences": 0,
                "weekly_attendance_pct": 100.0,
            },
        },
        {
            "label": "ST1004_DS301_borderline",
            "row": {
                "student_id": "ST1004", "semester": "4",
                "subject_code": "DS301", "faculty_id": "F101",
                "class_start_time": "11:00", "class_end_time": "12:00",
                "day_of_week": 3, "is_exam_week": 0,
                "rolling_attendance_pct": 74.4, "late_entry": 0,
                "day_of_month": 21, "month": 2,
                "prev_class_attended": 0, "consecutive_absences": 1,
                "weekly_attendance_pct": 57.1,
            },
        },
        {
            "label": "ST1121_AI401_atrisk",
            "row": {
                "student_id": "ST1121", "semester": "6",
                "subject_code": "AI401", "faculty_id": "F103",
                "class_start_time": "14:00", "class_end_time": "15:00",
                "day_of_week": 2, "is_exam_week": 0,
                "rolling_attendance_pct": 57.8, "late_entry": 0,
                "day_of_month": 21, "month": 2,
                "prev_class_attended": 0, "consecutive_absences": 4,
                "weekly_attendance_pct": 0.0,
            },
        },
    ]

    for case in sample_cases:
        logger.info("Local explanation for %s ...", case["label"])
        local_explanation(case["row"], label=case["label"])

    logger.info("=== SHAP Explanation Pipeline Done ===")
    print(f"\nAll SHAP plots saved to: {SHAP_DIR}\n")
