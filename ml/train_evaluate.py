"""
train_evaluate.py
-----------------
Trains four binary-classification models, evaluates them, and automatically
selects the best model based on F1-score (weighted). Saves the best model,
scaler, and a metrics comparison CSV.

Class Imbalance Strategy
------------------------
The dataset is ~83.7% Present / 16.3% Absent.  SMOTE (Synthetic Minority
Over-sampling Technique) is applied ONLY to the training set AFTER
standardisation so the test set remains a faithful reflection of reality.

Models compared
---------------
1. Logistic Regression   -- linear, interpretable, fast
2. Decision Tree         -- rule-based, highly interpretable
3. Random Forest         -- ensemble, robust, low overfitting
4. Gradient Boosting     -- powerful ensemble, usually top performer
"""

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — must be set before pyplot import
import matplotlib.pyplot as plt
import joblib
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

from ml.config import (
    ALL_FEATURES,
    BEST_MODEL_PATH,
    METRICS_CSV_PATH,
    MODEL_PARAMS,
    RANDOM_STATE,
    REPORT_DIR,
    SCALER_PATH,
    TARGET,
    TEST_SIZE,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)


# ─── Model factory ────────────────────────────────────────────────────────────

def _build_models() -> dict:
    """Return a dict of {name: sklearn_estimator} with configured params."""
    p = MODEL_PARAMS
    return {
        "LogisticRegression": LogisticRegression(**p["LogisticRegression"]),
        "DecisionTree":       DecisionTreeClassifier(**p["DecisionTree"]),
        "RandomForest":       RandomForestClassifier(**p["RandomForest"]),
        "GradientBoosting":   GradientBoostingClassifier(**p["GradientBoosting"]),
        "XGBoost":            XGBClassifier(**p["XGBoost"]),
    }


# ─── Split helpers ────────────────────────────────────────────────────────────

def split_dataset(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Stratified train/test split respecting class balance."""
    X = df[ALL_FEATURES]
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    logger.info(
        "Split → Train: %d rows | Test: %d rows | "
        "Present: %.1f%%",
        len(X_train), len(X_test),
        100 * y.mean(),
    )
    return X_train, X_test, y_train, y_test


def scale_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    """Fit scaler on train, apply to both. Persists scaler to disk."""
    scaler = StandardScaler()
    X_train_sc = pd.DataFrame(
        scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index
    )
    X_test_sc = pd.DataFrame(
        scaler.transform(X_test), columns=X_test.columns, index=X_test.index
    )
    joblib.dump(scaler, SCALER_PATH)
    logger.info("Scaler saved to %s", SCALER_PATH)
    return X_train_sc, X_test_sc, scaler


# --- SMOTE: Fix Class Imbalance ---

def smote_train_data(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Apply SMOTE to the training set to balance Present/Absent classes.

    Rules:
    - Called AFTER scaling (SMOTE works on continuous feature space).
    - Applied ONLY to training data — test set stays untouched.
    - Creates synthetic Absent samples until both classes are 50/50.
    """
    counts = y_train.value_counts()
    logger.info(
        "Before SMOTE -- Present: %d (%.1f%%)  |  Absent: %d (%.1f%%)",
        counts.get(1, 0), 100 * counts.get(1, 0) / len(y_train),
        counts.get(0, 0), 100 * counts.get(0, 0) / len(y_train),
    )
    sm = SMOTE(random_state=random_state)
    X_res, y_res = sm.fit_resample(X_train, y_train)
    counts_after = pd.Series(y_res).value_counts()
    logger.info(
        "After  SMOTE -- Present: %d (%.1f%%)  |  Absent: %d (%.1f%%)",
        counts_after.get(1, 0), 100 * counts_after.get(1, 0) / len(y_res),
        counts_after.get(0, 0), 100 * counts_after.get(0, 0) / len(y_res),
    )
    X_res = pd.DataFrame(X_res, columns=X_train.columns)
    y_res = pd.Series(y_res, name=y_train.name)
    return X_res, y_res


# --- Train & Evaluate ---


def evaluate_model(
    name: str,
    model,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> dict:
    """Fit the model and return a metrics dict."""
    logger.info("  Training %-20s …", name)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec  = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1   = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    # 5-fold CV F1 on training data (overfitting check)
    cv_f1 = cross_val_score(
        model, X_train, y_train,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
        scoring="f1_weighted",
        n_jobs=-1,
    ).mean()

    logger.info(
        "    Acc=%.4f  Prec=%.4f  Rec=%.4f  F1=%.4f  CV_F1=%.4f",
        acc, prec, rec, f1, cv_f1,
    )

    # Confusion matrix figure
    _save_confusion_matrix(name, y_test, y_pred)

    # Full classification report
    report_path = REPORT_DIR / f"{name}_classification_report.txt"
    report_path.write_text(
        classification_report(y_test, y_pred, target_names=["Absent", "Present"])
    )

    return {
        "Model":          name,
        "Accuracy":       round(acc,  4),
        "Precision":      round(prec, 4),
        "Recall":         round(rec,  4),
        "F1_Score":       round(f1,   4),
        "CV_F1_Train":    round(cv_f1, 4),
        "Overfit_Gap":    round(cv_f1 - f1, 4),   # negative = slight underfit; positive = overfit
    }


def _save_confusion_matrix(name: str, y_true, y_pred) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Absent", "Present"])
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"Confusion Matrix — {name}", fontsize=12, pad=10)
    plt.tight_layout()
    fig.savefig(REPORT_DIR / f"{name}_confusion_matrix.png", dpi=120)
    plt.close(fig)


# ─── Comparison & Selection ───────────────────────────────────────────────────

def compare_models(metrics: list[dict]) -> pd.DataFrame:
    """Build and save a comparison DataFrame, print a formatted table."""
    df_metrics = pd.DataFrame(metrics).sort_values("F1_Score", ascending=False).reset_index(drop=True)
    df_metrics.to_csv(METRICS_CSV_PATH, index=False)

    print("\n" + "=" * 70)
    print("   MODEL COMPARISON (sorted by F1-Score, desc)")
    print("=" * 70)
    print(df_metrics.to_string(index=False))
    print("=" * 70 + "\n")

    _save_comparison_chart(df_metrics)
    return df_metrics


def _save_comparison_chart(df_metrics: pd.DataFrame) -> None:
    metrics_to_plot = ["Accuracy", "Precision", "Recall", "F1_Score", "CV_F1_Train"]
    x  = np.arange(len(df_metrics))
    w  = 0.15
    fig, ax = plt.subplots(figsize=(12, 6))
    for i, m in enumerate(metrics_to_plot):
        ax.bar(x + i * w, df_metrics[m], w, label=m)
    ax.set_xticks(x + w * (len(metrics_to_plot) - 1) / 2)
    ax.set_xticklabels(df_metrics["Model"], rotation=15, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison — Student Attendance Prediction")
    ax.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    fig.savefig(REPORT_DIR / "model_comparison.png", dpi=120)
    plt.close(fig)
    logger.info("Comparison chart saved.")


def select_best_model(trained_models: dict, df_metrics: pd.DataFrame) -> tuple[str, object]:
    """Return (best_name, best_model) based on highest F1_Score."""
    best_name  = df_metrics.iloc[0]["Model"]
    best_model = trained_models[best_name]
    joblib.dump(best_model, BEST_MODEL_PATH)
    logger.info("Best model → %s  (F1=%.4f)  saved to %s",
                best_name, df_metrics.iloc[0]["F1_Score"], BEST_MODEL_PATH)
    return best_name, best_model


# ─── Public entry point ───────────────────────────────────────────────────────

def train_and_compare(df: pd.DataFrame) -> tuple[str, object, pd.DataFrame]:
    """
    End-to-end: split → scale → train all models → compare → pick best.

    Returns
    -------
    best_name   : str
    best_model  : fitted sklearn estimator
    df_metrics  : comparison DataFrame
    """
    X_train, X_test, y_train, y_test = split_dataset(df)
    X_train_sc, X_test_sc, _         = scale_features(X_train, X_test)

    # Apply SMOTE to training split only (test set untouched)
    X_train_sc, y_train = smote_train_data(X_train_sc, y_train)

    models  = _build_models()
    metrics = []

    print("\n" + "-" * 50)
    print("  Training 4 models ...")
    print("-" * 50)

    for name, mdl in models.items():
        m = evaluate_model(name, mdl, X_train_sc, X_test_sc, y_train, y_test)
        metrics.append(m)

    df_metrics = compare_models(metrics)
    best_name, best_model = select_best_model(models, df_metrics)

    _print_model_analysis(df_metrics)

    return best_name, best_model, df_metrics


# ─── Analysis summary (printed to console) ────────────────────────────────────

_ANALYSIS = {
    "LogisticRegression": (
        "Linear boundary — fast, interpretable, low overfit risk. "
        "May underfit if classes are non-linearly separable."
    ),
    "DecisionTree": (
        "Rule-based, highly interpretable. "
        "Prone to overfitting (controlled via max_depth/min_samples_leaf)."
    ),
    "RandomForest": (
        "Ensemble of trees — robust, low variance, handles non-linearities well. "
        "Good balance of accuracy & overfit resistance."
    ),
    "GradientBoosting": (
        "Sequential boosting -- typically highest accuracy on tabular data. "
        "Slower to train; deployment needs serialisation."
    ),
    "XGBoost": (
        "Extreme Gradient Boosting -- faster & usually more accurate than sklearn GB. "
        "Built-in scale_pos_weight handles imbalance; excellent for production deployment."
    ),
}


def _print_model_analysis(df_metrics: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("  MODEL ANALYSIS & SUITABILITY")
    print("=" * 70)
    for _, row in df_metrics.iterrows():
        print(f"\n>>  {row['Model']}")
        print(f"   {_ANALYSIS[row['Model']]}")
        overfit_label = (
            "[!] overfit"   if row["Overfit_Gap"] >  0.02 else
            "[?] underfit?" if row["Overfit_Gap"] < -0.02 else
            "[OK] balanced"
        )
        print(f"   Overfit gap (CV_F1 - Test_F1): {row['Overfit_Gap']:+.4f}  {overfit_label}")
    best = df_metrics.iloc[0]["Model"]
    print(f"\n{'-'*70}")
    print(f"  [BEST] RECOMMENDED: {best}")
    print(f"     Highest weighted F1 on the held-out test set.")
    print("=" * 70 + "\n")
