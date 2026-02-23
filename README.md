# Student Attendance Predictive System

A machine learning pipeline that predicts whether a student will be **Present** or **Absent** for an upcoming class, and generates an **attendance gap report** showing how many more classes they need to attend to avoid detention.

---

## Features

- **5 ML models** trained and compared: Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, XGBoost
- **SMOTE** class balancing to handle the 84% Present / 16% Absent imbalance
- **14 engineered features** including 3 lag features (no data leakage):
  - `prev_class_attended` — did the student attend the last class?
  - `consecutive_absences` — streak of missed classes
  - `weekly_attendance_pct` — last 7-class rolling attendance %
- **SHAP explainability** — see exactly *why* the model predicted Present or Absent
- **Attendance gap report** — tells each student how many lectures they must still attend to stay above 75%

---

## Model Performance

| Model | F1-Score | Overfit Gap |
|---|---|---|
| **GradientBoosting** ⭐ | **0.7810** | +0.038 |
| XGBoost (tuned) | 0.7632 | +0.015 ✅ |
| DecisionTree | 0.6418 | +0.006 ✅ |
| RandomForest | 0.6214 | +0.099 |
| LogisticRegression | 0.3913 | baseline |

---

## Project Structure

```
student-attendance-predictive-system/
├── ml/
│   ├── config.py            # Paths, feature lists, hyperparameters
│   ├── feature_engineering.py  # Data loading + 14-feature pipeline
│   ├── train_evaluate.py    # Model training, SMOTE, evaluation
│   ├── predict.py           # Single-class prediction + gap report
│   └── explain.py           # SHAP global + local explainability
├── main.py                  # CLI entry point
├── requirements.txt
└── .gitignore
```

---

## Setup

```bash
# 1. Create virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Place dataset in project root
#    (extra_large_student_attendance_dataset.csv)
```

---

## Usage

### Train all models
```bash
python main.py --mode train
```

### Attendance gap report
```bash
python main.py --mode gap --student ST1001 --remaining 20
python main.py --mode gap --student ST1004 --subject DS301 --remaining 10
```

### Predict a single class
```bash
python main.py --mode predict --student ST1001 --subject ML302 \
               --faculty F102 --semester 6 --start 09:00 --end 10:00 \
               --rolling_pct 89.0 --dow 0 --dom 23 --month 2
```

### SHAP explainability
```bash
python main.py --mode explain
# Saves plots to model/reports/shap/
```

---

## Dataset

The dataset is not included in this repo due to its size (~107,000 rows).  
Columns: `student_id`, `subject_code`, `faculty_id`, `semester`, `date`, `class_start_time`, `class_end_time`, `status` (target), `late_entry`, `is_exam_week`, and more.

---

## Tech Stack

`Python 3.11` · `scikit-learn` · `XGBoost` · `imbalanced-learn (SMOTE)` · `SHAP` · `pandas` · `matplotlib` · `seaborn` · `joblib`
