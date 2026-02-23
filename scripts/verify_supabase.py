from __future__ import annotations

import os
from getpass import getpass
from pathlib import Path

from dotenv import load_dotenv
import psycopg2


def _load_env() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    env_file = repo_root / ".env"
    example_file = repo_root / ".env.example"

    if env_file.exists():
        load_dotenv(env_file, override=False)
    elif example_file.exists():
        load_dotenv(example_file, override=False)


def _get_password() -> str:
    password = os.getenv("SUPABASE_PASSWORD")
    if password:
        return password
    return getpass("Supabase DB password (input hidden): ")


_load_env()

host = os.getenv("SUPABASE_HOST", "db.tosolythxignxqheborz.supabase.co")
dbname = os.getenv("SUPABASE_DB", "postgres")
user = os.getenv("SUPABASE_USER", "postgres")
port = int(os.getenv("SUPABASE_PORT", "5432"))
sslmode = os.getenv("SUPABASE_SSLMODE", "require")

print(f"Connecting to {host}:{port}/{dbname} as {user} (sslmode={sslmode})")

with psycopg2.connect(
    host=host,
    dbname=dbname,
    user=user,
    password=_get_password(),
    port=port,
    sslmode=sslmode,
    connect_timeout=10,
) as conn:
    with conn.cursor() as cur:
        cur.execute("SELECT NOW();")
        print(cur.fetchone()[0])
