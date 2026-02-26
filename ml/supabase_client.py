"""
ml/supabase_client.py
---------------------
Loads attendance data from Supabase via the REST API.
Falls back to the local CSV automatically if Supabase is empty or unreachable.
"""

from __future__ import annotations

import os
import pandas as pd
import streamlit as st


# ── Credentials from Streamlit secrets or environment ────────────────────────

def _get_credentials() -> tuple[str, str]:
    """Return (url, key) — tries Streamlit secrets, env vars, then hardcoded defaults.
    The anon key is intentionally public-safe (protected by Supabase RLS).
    """
    # 1. Try top-level secrets (recommended Streamlit Cloud format)
    try:
        url = st.secrets["SUPABASE_URL"]
        key = st.secrets["SUPABASE_KEY"]
        if url and key:
            return url, key
    except Exception:
        pass

    # 2. Try nested [supabase] section
    try:
        url = st.secrets["supabase"]["SUPABASE_URL"]
        key = st.secrets["supabase"]["SUPABASE_KEY"]
        if url and key:
            return url, key
    except Exception:
        pass

    # 3. Try environment variables
    url = os.getenv("SUPABASE_URL", "")
    key = os.getenv("SUPABASE_KEY", "")
    if url and key:
        return url, key

    # 4. Hardcoded defaults (anon key is public-safe, protected by RLS)
    return (
        "https://tosolythxignxqheborz.supabase.co",
        "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InRvc29seXRoeGlnbnhxaGVib3J6Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3Njk2MTU5MzYsImV4cCI6MjA4NTE5MTkzNn0.8zoxEXG0umbGe68TokWWABJGqpumxZGvWXxD4Wj6t3w",
    )


# ── Main loader ───────────────────────────────────────────────────────────────

@st.cache_data(ttl=300, show_spinner=False)   # refresh every 5 min
def load_attendance_from_supabase() -> tuple[pd.DataFrame, str]:
    """
    Returns (df, source) where source is 'supabase' or 'csv'.

    The DataFrame has the same columns as the raw CSV so the rest of the
    pipeline (feature_engineering, predict, gap_report) works unchanged.
    """
    url, key = _get_credentials()

    if url and key:
        try:
            from supabase import create_client
            sb  = create_client(url, key)

            # Paginate: Supabase returns max 1000 rows per call
            rows: list[dict] = []
            page_size = 1000
            offset    = 0

            while True:
                resp = (
                    sb.table("attendance")
                    .select("*")
                    .range(offset, offset + page_size - 1)
                    .execute()
                )
                batch = resp.data or []
                rows.extend(batch)
                if len(batch) < page_size:
                    break
                offset += page_size

            if rows:
                df = pd.DataFrame(rows)
                # Normalise column names to match CSV expectations
                if "date" in df.columns:
                    df["date"] = pd.to_datetime(df["date"], errors="coerce")
                df["status"] = pd.to_numeric(df.get("status", 0), errors="coerce").fillna(0).astype(int)
                # drop supabase-only cols
                df.drop(columns=["id", "created_at"], errors="ignore", inplace=True)
                return df, "supabase"
            else:
                # Table is empty — fall through to CSV
                pass
        except Exception as e:
            print(f"[supabase_client] Supabase error, falling back to CSV: {e}")

    # ── CSV fallback ──────────────────────────────────────────────────────────
    from ml.config import DATASET_PATH
    df = pd.read_csv(DATASET_PATH, parse_dates=["date"])
    df["status"] = pd.to_numeric(df["status"], errors="coerce").fillna(0).astype(int)
    return df, "csv"
