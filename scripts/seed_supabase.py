"""
scripts/seed_supabase.py
------------------------
Uploads sample_dataset.csv to the Supabase `attendance` table in batches.

Run ONCE after applying supabase/migration.sql:
    python scripts/seed_supabase.py
"""

from pathlib import Path
import sys, math
import pandas as pd
from supabase import create_client

# ── Credentials ───────────────────────────────────────────────────────────────
SUPABASE_URL = "https://tosolythxignxqheborz.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InRvc29seXRoeGlnbnhxaGVib3J6Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3Njk2MTU5MzYsImV4cCI6MjA4NTE5MTkzNn0.8zoxEXG0umbGe68TokWWABJGqpumxZGvWXxD4Wj6t3w"

# ── Load CSV ──────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
CSV  = ROOT / "sample_dataset.csv"

if not CSV.exists():
    sys.exit(f"ERROR: {CSV} not found. Run training first to generate sample_dataset.csv.")

df = pd.read_csv(CSV)
print(f"Loaded {len(df)} rows from {CSV.name}")

# Keep only columns the attendance table expects
KEEP = [
    "student_id", "date", "subject_code", "subject_name",
    "faculty_id", "class_start_time", "class_end_time",
    "semester", "status", "is_exam_week", "late_entry",
]
df = df[[c for c in KEEP if c in df.columns]]
df["date"]   = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
df["status"] = df["status"].fillna(0).astype(int)
for col in ["is_exam_week", "late_entry"]:
    if col in df.columns:
        df[col] = df[col].fillna(0).astype(int)

records = df.where(pd.notnull(df), None).to_dict(orient="records")

# ── Upload in batches ─────────────────────────────────────────────────────────
sb         = create_client(SUPABASE_URL, SUPABASE_KEY)
BATCH_SIZE = 500
n_batches  = math.ceil(len(records) / BATCH_SIZE)

print(f"Uploading {len(records)} rows in {n_batches} batches of {BATCH_SIZE}...")

for i in range(n_batches):
    batch = records[i * BATCH_SIZE : (i + 1) * BATCH_SIZE]
    sb.table("attendance").insert(batch).execute()
    print(f"  Batch {i+1}/{n_batches} done ({len(batch)} rows)")

print(f"\n✅ Seed complete — {len(records)} rows uploaded to Supabase 'attendance' table.")
