"""
main.py
-------
Orchestrator entry-point for the Student Attendance Predictive System.

Usage
─────
# Full training pipeline
python main.py --mode train

# Attendance gap report for one student
python main.py --mode gap --student ST1001 --remaining 20

# Predict a single future class
python main.py --mode predict --student ST1001 --subject ML302 \
               --faculty F102 --semester 6 --start 09:00 --end 10:00 \
               --exam_week 0 --rolling_pct 78.5 --dow 0 --dom 25 --month 9
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to PYTHONPATH so `ml.*` imports resolve correctly
sys.path.insert(0, str(Path(__file__).resolve().parent))

from ml.feature_engineering import load_and_engineer
from ml.predict import attendance_gap_report, predict_attendance
from ml.train_evaluate import train_and_compare
from ml.explain import run_full_explanation

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)


def run_training() -> None:
    """Load data, engineer features, train & compare models, save best."""
    logger.info("═══ TRAINING PIPELINE STARTED ═══")
    df, _ = load_and_engineer()   # fit encoders, save to disk
    best_name, _, df_metrics = train_and_compare(df)
    logger.info("═══ DONE — Best model: %s ═══", best_name)


def run_gap(student_id: str, subject_code: str | None, remaining: int) -> None:
    """Print attendance gap report for a student."""
    attendance_gap_report(
        student_id=student_id,
        subject_code=subject_code,
        total_classes_remaining=remaining,
    )


def run_predict(args: argparse.Namespace) -> None:
    """Run single-record inference."""
    result = predict_attendance({
        "student_id":             args.student,
        "semester":               args.semester,
        "subject_code":           args.subject,
        "faculty_id":             args.faculty,
        "class_start_time":       args.start,
        "class_end_time":         args.end,
        "day_of_week":            args.dow,
        "is_exam_week":           args.exam_week,
        "rolling_attendance_pct": args.rolling_pct,
        "late_entry":             0,
        "day_of_month":           args.dom,
        "month":                  args.month,
    })
    print("\n" + "-" * 43)
    print(f"  Prediction : {result['label']}")
    print(f"  P(Present) : {result['probability_present']}")
    print("-" * 43 + "\n")


def run_explain(sample_size: int = 500) -> None:
    """Generate SHAP global + local explanation plots."""
    run_full_explanation(sample_size=sample_size)


# --- CLI ---

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Student Attendance Predictive System",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--mode", choices=["train", "gap", "predict", "explain"],
                        default="train", help="Pipeline mode")

    # Gap report args
    parser.add_argument("--student",   default="ST1001", help="Student ID")
    parser.add_argument("--remaining", type=int, default=0,
                        help="Future classes remaining in semester")
    parser.add_argument("--subject",   default=None, help="Subject code filter (gap/predict)")

    # Predict args
    parser.add_argument("--faculty",     default="F101")
    parser.add_argument("--semester",    default="6")
    parser.add_argument("--start",       default="09:00", help="Class start HH:MM")
    parser.add_argument("--end",         default="10:00", help="Class end HH:MM")
    parser.add_argument("--dow",         type=int, default=0, help="Day of week 0=Mon")
    parser.add_argument("--exam_week",   type=int, default=0, choices=[0, 1])
    parser.add_argument("--rolling_pct", type=float, default=75.0,
                        help="Student's current attendance % for subject")
    parser.add_argument("--dom",         type=int, default=15, help="Day of month")
    parser.add_argument("--month",       type=int, default=6, help="Month 1-12")

    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    if args.mode == "train":
        run_training()
    elif args.mode == "gap":
        run_gap(args.student, args.subject, args.remaining)
    elif args.mode == "predict":
        run_predict(args)
    elif args.mode == "explain":
        run_explain()
