"""
config.py
---------
Central configuration for the Student Attendance Predictive System.
Modify paths and hyperparameters here; no changes needed in other modules.
"""

from pathlib import Path

# ─── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parent.parent
DATA_DIR   = BASE_DIR                          # CSV lives in project root
MODEL_DIR  = BASE_DIR / "model"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

DATASET_PATH = DATA_DIR / "extra_large_student_attendance_dataset.csv"

# Saved artefacts
BEST_MODEL_PATH      = MODEL_DIR / "best_model.joblib"
LABEL_ENCODERS_PATH  = MODEL_DIR / "label_encoders.joblib"
SCALER_PATH          = MODEL_DIR / "scaler.joblib"
METRICS_CSV_PATH     = MODEL_DIR / "metrics_comparison.csv"
REPORT_DIR           = MODEL_DIR / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# ─── Attendance / Detention threshold ─────────────────────────────────────────
MIN_ATTENDANCE_PCT = 75.0   # % below which student is detained

# ─── Feature columns ──────────────────────────────────────────────────────────
# Categorical columns to be label-encoded
CAT_FEATURES = ["semester", "subject_code", "faculty_id", "day_of_week"]

# Numeric columns (after engineering)
NUM_FEATURES = [
    "class_hour",              # hour extracted from class_start_time
    "class_duration_min",      # class length in minutes
    "is_exam_week",            # already binary
    "rolling_attendance_pct",  # per-student-subject expanding presence rate (no leakage)
    "late_entry",              # already binary
    "day_of_month",
    "month",
    # --- Lag / history features (shift-1, zero leakage) ---
    "prev_class_attended",     # 1 if student attended their last class, else 0
    "consecutive_absences",    # how many classes missed in a row before this one
    "weekly_attendance_pct",   # rolling 7-class window attendance % (recent form)
]

ALL_FEATURES = CAT_FEATURES + NUM_FEATURES

TARGET = "status"   # 1 = Present, 0 = Absent

# ─── Train / test split ────────────────────────────────────────────────────────
TEST_SIZE    = 0.20
RANDOM_STATE = 42

# ─── Model hyperparameters ────────────────────────────────────────────────────
MODEL_PARAMS = {
    "LogisticRegression": {
        "max_iter": 1000,
        "random_state": RANDOM_STATE,
        "class_weight": "balanced",
    },
    "DecisionTree": {
        "max_depth": 8,
        "min_samples_leaf": 20,
        "random_state": RANDOM_STATE,
        "class_weight": "balanced",
    },
    "RandomForest": {
        "n_estimators": 200,
        "max_depth": 12,
        "min_samples_leaf": 10,
        "random_state": RANDOM_STATE,
        "class_weight": "balanced",
        "n_jobs": -1,
    },
    "GradientBoosting": {
        "n_estimators": 200,
        "learning_rate": 0.05,
        "max_depth": 5,
        "random_state": RANDOM_STATE,
        "subsample": 0.8,
    },
    "XGBoost": {
        "n_estimators": 400,         # more trees to compensate for shallower depth
        "learning_rate": 0.04,       # slightly lower for smoother convergence
        "max_depth": 4,              # reduced from 6 -> key overfit fix
        "min_child_weight": 15,      # prevents over-specialised leaf nodes
        "subsample": 0.75,           # more aggressive sampling -> less overfit
        "colsample_bytree": 0.75,    # more aggressive col sampling
        "reg_alpha": 0.1,            # L1 regularisation (sparsity)
        "reg_lambda": 2.0,           # L2 regularisation (default=1, doubled)
        "scale_pos_weight": 5,       # Present/Absent ratio before SMOTE
        "eval_metric": "logloss",
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
    },
}
