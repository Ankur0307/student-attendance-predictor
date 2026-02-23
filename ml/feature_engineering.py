"""
feature_engineering.py
-----------------------
Loads the raw attendance CSV and transforms it into an ML-ready feature matrix.

Features engineered
-------------------
Base features:
  day_of_week          -- Mon=0 ... Sun=6  (from date)
  class_hour           -- integer hour of class_start_time
  class_duration_min   -- class length in minutes
  day_of_month         -- 1-31
  month                -- 1-12
  rolling_attendance_pct -- per (student x subject) expanding-window attendance %
  Categorical encoding -- semester, subject_code, faculty_id, day_of_week

Lag / History features (all shift(1) -- zero leakage):
  prev_class_attended  -- 1 if student attended the immediately previous class, 0 if absent
  consecutive_absences -- number of classes missed in a row up to this point
  weekly_attendance_pct -- rolling 7-class window attendance % (recent form)
"""

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from ml.config import (
    ALL_FEATURES,
    CAT_FEATURES,
    DATASET_PATH,
    LABEL_ENCODERS_PATH,
    TARGET,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _parse_time_to_minutes(series: pd.Series) -> pd.Series:
    """Convert 'HH:MM' string column to total minutes since midnight."""
    return series.apply(
        lambda t: int(t.split(":")[0]) * 60 + int(t.split(":")[1])
        if isinstance(t, str) and ":" in t
        else np.nan
    )


def _rolling_attendance(df: pd.DataFrame) -> pd.Series:
    """
    For each row, compute the % of *previous* classes the student attended
    for the same subject. Returns a float in [0, 100].
    Uses expanding window (no leakage).
    """
    df = df.copy()
    df["_rolling_pct"] = np.nan
    groups = df.groupby(["student_id", "subject_code"])
    results = []
    for (_, _), grp in groups:
        # expanding mean on status (1=present, 0=absent) then shift by 1
        rolling = grp["status"].expanding().mean().shift(1) * 100
        results.append(rolling)
    result_series = pd.concat(results).reindex(df.index)
    # Fill NaN (first appearance) with 0 -- unknown history -> assume 0
    return result_series.fillna(0.0)


def _prev_class_attended(df: pd.DataFrame) -> pd.Series:
    """
    Lag-1 feature: was the student present at their immediately previous
    class for the same subject?  1 = Present, 0 = Absent, 0 for first row.
    Shift by 1 within (student_id, subject_code) -- zero leakage.
    """
    return (
        df.groupby(["student_id", "subject_code"])["status"]
        .shift(1)
        .fillna(0)
        .astype(int)
    )


def _consecutive_absences(df: pd.DataFrame) -> pd.Series:
    """
    Count how many consecutive classes the student has missed *up to the
    current row* (i.e., the streak of 0s ending at the previous class).
    Resets to 0 whenever a Present row is encountered.
    Zero leakage: built on status values shifted by 1.
    """
    result = pd.Series(0, index=df.index, dtype=int)
    for (sid, sub), grp in df.groupby(["student_id", "subject_code"]):
        shifted = grp["status"].shift(1).fillna(1)  # assume present before first row
        streak = []
        count = 0
        for val in shifted:
            if val == 0:        # previous class was absent -> increment streak
                count += 1
            else:               # previous class was present -> reset streak
                count = 0
            streak.append(count)
        result.loc[grp.index] = streak
    return result


def _weekly_attendance_pct(df: pd.DataFrame) -> pd.Series:
    """
    Rolling 7-class window attendance % per (student x subject), shifted by 1
    so only past classes are used.  Gives a 'recent form' signal more responsive
    than the full expanding average.
    Returns a float in [0, 100]; NaN filled with rolling_attendance_pct (fallback).
    """
    results = []
    for (_, _), grp in df.groupby(["student_id", "subject_code"]):
        rolling = (
            grp["status"]
            .shift(1)                    # shift so current row not included
            .rolling(window=7, min_periods=1)
            .mean() * 100
        )
        results.append(rolling)
    result_series = pd.concat(results).reindex(df.index)
    return result_series.fillna(0.0)


# ─── Main pipeline ────────────────────────────────────────────────────────────

def load_and_engineer(
    path: Path = DATASET_PATH,
    fit_encoders: bool = True,
    encoders: dict | None = None,
) -> tuple[pd.DataFrame, dict]:
    """
    Load raw CSV, engineer features, and return:
      (feature_df, label_encoders_dict)

    Parameters
    ----------
    path         : path to the CSV file
    fit_encoders : True for training pass; False for inference (pass `encoders`)
    encoders     : pre-fitted {col: LabelEncoder} dict (required when fit_encoders=False)
    """
    logger.info("Loading dataset from %s …", path)
    df = pd.read_csv(path)
    logger.info("  Rows: %d  |  Columns: %s", len(df), list(df.columns))

    # ── 1. Parse date ──────────────────────────────────────────────────────────
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df.dropna(subset=["date"], inplace=True)

    df["day_of_week"]  = df["date"].dt.dayofweek        # 0=Mon
    df["day_of_month"] = df["date"].dt.day
    df["month"]        = df["date"].dt.month

    # ── 2. Parse times ─────────────────────────────────────────────────────────
    start_min = _parse_time_to_minutes(df["class_start_time"])
    end_min   = _parse_time_to_minutes(df["class_end_time"])

    df["class_hour"]         = (start_min // 60).astype("Int64")
    df["class_duration_min"] = (end_min - start_min).clip(lower=0)

    # -- 3. Rolling & lag attendance features (no leakage) ---------------------
    # Sort chronologically first so all windowed ops are meaningful
    df.sort_values(["student_id", "subject_code", "date"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    df["rolling_attendance_pct"] = _rolling_attendance(df)
    df["prev_class_attended"]    = _prev_class_attended(df)
    df["consecutive_absences"]   = _consecutive_absences(df)
    df["weekly_attendance_pct"]  = _weekly_attendance_pct(df)

    # ── 4. Ensure binary flag columns are int ──────────────────────────────────
    for col in ["late_entry", "is_exam_week"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    # ── 5. Target ──────────────────────────────────────────────────────────────
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    df.dropna(subset=[TARGET], inplace=True)
    df[TARGET] = df[TARGET].astype(int)

    # ── 6. Categorical encoding ────────────────────────────────────────────────
    if fit_encoders:
        encoders = {}
        for col in CAT_FEATURES:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
        joblib.dump(encoders, LABEL_ENCODERS_PATH)
        logger.info("Label encoders saved to %s", LABEL_ENCODERS_PATH)
    else:
        if encoders is None:
            raise ValueError("encoders dict must be provided when fit_encoders=False")
        for col in CAT_FEATURES:
            le = encoders[col]
            # Handle unseen labels gracefully
            df[col] = df[col].astype(str).apply(
                lambda v: le.transform([v])[0] if v in le.classes_ else -1
            )

    # ── 7. Fill remaining NaNs ────────────────────────────────────────────────
    df[ALL_FEATURES] = df[ALL_FEATURES].fillna(df[ALL_FEATURES].median(numeric_only=True))

    logger.info("Feature engineering complete. Shape: %s", df[ALL_FEATURES].shape)
    return df[ALL_FEATURES + [TARGET]], encoders


def load_raw_for_attendance_check(path: Path = DATASET_PATH) -> pd.DataFrame:
    """Return minimal raw DF used only by the attendance-gap calculator."""
    df = pd.read_csv(
        path,
        usecols=["student_id", "student_name", "subject_code", "subject_name", "date", "status"],
    )
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df.dropna(subset=["date", "status"], inplace=True)
    df["status"] = pd.to_numeric(df["status"], errors="coerce").astype(int)
    return df
