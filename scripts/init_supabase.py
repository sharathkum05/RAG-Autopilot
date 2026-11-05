"""Checks the Supabase project: tables from supabase/schema.sql exist, and creates the public page-image bucket.
Run after pasting supabase/schema.sql into the Supabase SQL editor:  uv run python scripts/init_supabase.py
"""
from pipeline.config import settings
from pipeline.stores import supabase

TABLES = ["documents", "pages", "chunks", "jobs", "conversations", "turns", "eval_runs", "answer_eval_runs"]


def main() -> None:
    s = settings()
    sb = supabase()
    missing = []
    for t in TABLES:
        try:
            sb.table(t).select("*", count="exact").limit(1).execute()
            print(f"table ok   {t}")
        except Exception as e:  # noqa: BLE001
            missing.append(t); print(f"table MISSING {t}: {str(e)[:80]}")
    if missing:
        print("\nRun supabase/schema.sql in the Supabase SQL editor, then re-run this script.")
    buckets = {b.name for b in sb.storage.list_buckets()}
    if s.supabase_bucket in buckets:
        print(f"bucket ok  {s.supabase_bucket}")
    else:
        sb.storage.create_bucket(s.supabase_bucket, options={"public": True})
        print(f"bucket created (public): {s.supabase_bucket}")


if __name__ == "__main__":
    main()
