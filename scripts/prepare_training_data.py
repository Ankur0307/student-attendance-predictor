import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INPUT_CSV = ROOT / "phase3_cleaned_dataset.csv"
OUTPUT_CSV = ROOT / "data" / "student_attendance.csv"


def main():
    df = pd.read_csv(INPUT_CSV)

    if "student_id" not in df.columns:
        raise ValueError("Expected column 'student_id' in phase3_cleaned_dataset.csv")
    if "status" not in df.columns:
        raise ValueError("Expected column 'status' in phase3_cleaned_dataset.csv")

    status = df["status"].astype(str).str.strip().str.lower()
    df["risk_level"] = (status == "a").astype(int)

    drop_cols = [
        "date",
        "class_start_time",
        "class_end_time",
        "status",
        "remarks",
        "attendance_binary",
    ]
    for col in drop_cols:
        if col in df.columns:
            df = df.drop(columns=[col])

    # Ensure numeric-only features (train_model.py assumes everything besides risk_level/student_id is numeric)
    for col in df.columns:
        if df[col].dtype == bool:
            df[col] = df[col].astype(int)
        elif df[col].dtype == object:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.fillna(0)

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)

    print(f"✅ Wrote: {OUTPUT_CSV}")
    print(f"Rows: {len(df)} | Columns: {len(df.columns)}")
    print("risk_level counts:")
    print(df["risk_level"].value_counts(dropna=False))


if __name__ == "__main__":
    main()
