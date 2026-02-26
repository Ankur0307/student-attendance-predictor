"""
predict.py
----------
Two public utilities:

1. predict_attendance(row_dict)
   → Uses the saved best model to predict Present/Absent for a future class.

2. attendance_gap_report(student_id, subject_code)
   → Calculates how many MORE classes a student must attend to reach the
     minimum 75% attendance threshold (or confirms they are safe / detained).
"""

import logging
import math
from pathlib import Path

import joblib
import pandas as pd

from ml.config import (
    ALL_FEATURES,
    BEST_MODEL_PATH,
    CAT_FEATURES,
    DATASET_PATH,
    LABEL_ENCODERS_PATH,
    MIN_ATTENDANCE_PCT,
    SCALER_PATH,
)
from ml.feature_engineering import load_raw_for_attendance_check

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)


# ─── Load artefacts (lazy) ────────────────────────────────────────────────────

_model    = None
_scaler   = None
_encoders = None


def _load_artefacts() -> None:
    global _model, _scaler, _encoders
    if _model is None:
        _model    = joblib.load(BEST_MODEL_PATH)
        _scaler   = joblib.load(SCALER_PATH)
        _encoders = joblib.load(LABEL_ENCODERS_PATH)
        logger.info("Artefacts loaded.")


# ─── 1. Predict attendance for a single future class ─────────────────────────

def predict_attendance(row_dict: dict) -> dict:
    """
    Predict whether a student will be Present (1) or Absent (0) for an
    upcoming class.

    Parameters
    ----------
    row_dict : dict with the following keys (all required unless noted):
        student_id               (str)
        semester                 (int/str)
        subject_code             (str)
        faculty_id               (str)
        class_start_time         (str "HH:MM")
        class_end_time           (str "HH:MM")
        day_of_week              (int 0=Mon…6=Sun)
        is_exam_week             (int 0/1)
        rolling_attendance_pct   (float 0-100)  — student's current % for subject
        late_entry               (int 0/1)       — historically late? (optional)
        day_of_month             (int)
        month                    (int)

    Returns
    -------
    dict with keys: prediction (int), label (str), probability_present (float)
    """
    _load_artefacts()
    df = pd.DataFrame([row_dict])

    # Derive time features
    if "class_start_time" in df.columns:
        t = df["class_start_time"].iloc[0]
        if isinstance(t, str) and ":" in t:
            h, m  = t.split(":")
            df["class_hour"] = int(h)
    if "class_end_time" in df.columns and "class_start_time" in df.columns:
        _start = df["class_start_time"].iloc[0]
        _end   = df["class_end_time"].iloc[0]
        if isinstance(_start, str) and isinstance(_end, str):
            s_min = int(_start.split(":")[0]) * 60 + int(_start.split(":")[1])
            e_min = int(_end.split(":")[0])   * 60 + int(_end.split(":")[1])
            df["class_duration_min"] = max(0, e_min - s_min)

    # Fill defaults for optional cols
    for col in ["late_entry"]:
        if col not in df.columns:
            df[col] = 0

    # Encode categoricals
    for col in CAT_FEATURES:
        le = _encoders[col]
        val = str(df[col].iloc[0])
        df[col] = le.transform([val])[0] if val in le.classes_ else -1

    # Ensure all features present
    for col in ALL_FEATURES:
        if col not in df.columns:
            df[col] = 0

    X = df[ALL_FEATURES].fillna(0)
    X_sc = pd.DataFrame(
        _scaler.transform(X),
        columns=ALL_FEATURES,   # preserve feature names -> silence sklearn warning
    )

    pred     = int(_model.predict(X_sc)[0])
    proba    = float(_model.predict_proba(X_sc)[0][1]) if hasattr(_model, "predict_proba") else None
    label    = "Present" if pred == 1 else "Absent"

    return {
        "prediction":        pred,
        "label":             label,
        "probability_present": round(proba, 4) if proba is not None else "N/A",
    }


# ─── 2. Attendance gap report ─────────────────────────────────────────────────

def attendance_gap_report(
    student_id: str,
    subject_code: str | None = None,
    dataset_path: Path = DATASET_PATH,
    total_classes_remaining: int = 0,
    min_pct: float = MIN_ATTENDANCE_PCT,
) -> pd.DataFrame:
    """
    For a given student (and optionally a specific subject), compute:
    - Classes attended so far
    - Classes held so far
    - Current attendance %
    - Minimum classes still required to reach `min_pct` threshold
    - Classes still needed (0 if already safe, None if impossible)

    Parameters
    ----------
    student_id              : e.g. "ST1001"
    subject_code            : e.g. "ML302" (None = all subjects)
    total_classes_remaining : number of future classes planned for the semester
    min_pct                 : minimum attendance % to avoid detention (default 75)
    """
    df = load_raw_for_attendance_check(dataset_path)

    df_stu = df[df["student_id"] == student_id]
    if df_stu.empty:
        logger.warning("Student %s not found in dataset.", student_id)
        return pd.DataFrame()

    if subject_code:
        df_stu = df_stu[df_stu["subject_code"] == subject_code]

    groups = df_stu.groupby(["subject_code", "subject_name"])
    records = []
    for (sub_code, sub_name), grp in groups:
        held    = len(grp)
        attended = int(grp["status"].sum())
        current_pct = 100.0 * attended / held if held > 0 else 0.0

        # Total classes if semester ends after `total_classes_remaining` more
        total_at_end = held + total_classes_remaining

        # Classes needed at minimum to reach min_pct
        # attended + x >= min_pct/100 * (held + total_remaining + y)
        # where y = total_remaining – x (classes skipped)
        # Simplification: we need attended + x >= min_pct/100 * (held + total_classes_remaining)
        required_present = (min_pct / 100) * total_at_end
        # Use proper ceiling: how many more present classes are needed?
        classes_needed   = max(0, math.ceil(required_present - attended))

        if total_classes_remaining == 0:
            # No future classes — can't improve
            feasible = classes_needed == 0
        else:
            feasible = classes_needed <= total_classes_remaining

        if classes_needed == 0:
            # Can skip every remaining class and still stay above threshold
            status_flag = "✅ Safe"
        elif current_pct >= min_pct:
            # Currently above threshold but must attend some future classes to maintain it
            status_flag = "💛 Caution (maintain attendance)"
        elif feasible:
            # Currently below threshold but mathematically recoverable
            status_flag = "🔴 At Risk (needs recovery)"
        else:
            status_flag = "❌ Detained (irrecoverable)"

        records.append({
            "student_id":            student_id,
            "subject_code":          sub_code,
            "subject_name":          sub_name,
            "classes_held":          held,
            "classes_attended":      attended,
            "current_pct":           round(current_pct, 2),
            "remaining_classes":     total_classes_remaining,
            "classes_needed_more":   classes_needed if feasible else "N/A (detained)",
            "status":                status_flag,
        })

    report_df = pd.DataFrame(records)

    # Print nicely
    print(f"\n{'═'*72}")
    print(f"  ATTENDANCE GAP REPORT  |  Student: {student_id}  |  Min: {min_pct}%")
    print(f"{'═'*72}")
    print(report_df.to_string(index=False))
    print(f"{'═'*72}\n")

    return report_df
