"""
SQLite storage for actual OpenAI API usage records.

There is no existing ATS database. This file is the usage ledger.
"""

import sqlite3
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path


DB_PATH = Path(__file__).resolve().parent.parent / "data" / "openai_usage.sqlite"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def get_connection(db_path: str | Path | None = None) -> sqlite3.Connection:
    path = Path(db_path) if db_path else DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)

    connection = sqlite3.connect(str(path))
    connection.row_factory = sqlite3.Row
    _ensure_schema(connection)
    return connection


def _ensure_schema(connection: sqlite3.Connection) -> None:
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS openai_usage (
            id TEXT PRIMARY KEY,
            user_id TEXT,
            job_id TEXT,
            resume_id TEXT,
            run_id TEXT,
            operation TEXT,
            model TEXT,
            input_tokens INTEGER,
            output_tokens INTEGER,
            total_tokens INTEGER,
            input_cost REAL,
            output_cost REAL,
            total_cost REAL,
            request_id TEXT,
            created_at TEXT NOT NULL
        )
        """
    )
    connection.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS idx_openai_usage_request_id
        ON openai_usage(request_id)
        WHERE request_id IS NOT NULL AND request_id != ''
        """
    )
    connection.commit()


def insert_usage_record(record: dict, db_path: str | Path | None = None) -> str | None:
    """
    Insert one actual API-usage row.

    Returns the row id, or None if this request_id was already stored
    (prevents double-counting the same OpenAI response).
    """

    row_id = record.get("id") or str(uuid.uuid4())
    created_at = record.get("created_at") or _utc_now()
    request_id = record.get("request_id") or None

    connection = get_connection(db_path)

    try:
        connection.execute(
            """
            INSERT INTO openai_usage (
                id, user_id, job_id, resume_id, run_id, operation,
                model, input_tokens, output_tokens, total_tokens,
                input_cost, output_cost, total_cost, request_id, created_at
            ) VALUES (
                ?, ?, ?, ?, ?, ?,
                ?, ?, ?, ?,
                ?, ?, ?, ?, ?
            )
            """,
            (
                row_id,
                record.get("user_id"),
                record.get("job_id"),
                record.get("resume_id"),
                record.get("run_id"),
                record.get("operation"),
                record.get("model"),
                record.get("input_tokens"),
                record.get("output_tokens"),
                record.get("total_tokens"),
                record.get("input_cost"),
                record.get("output_cost"),
                record.get("total_cost"),
                request_id,
                created_at,
            ),
        )
        connection.commit()
        return row_id
    except sqlite3.IntegrityError:
        # Same OpenAI request_id already recorded.
        return None
    finally:
        connection.close()


def _period_start(period: str) -> str | None:
    now = datetime.now(timezone.utc)

    if period == "today":
        start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    elif period == "week":
        start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        start = start - timedelta(days=start.weekday())
    elif period == "month":
        start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    else:
        return None

    return start.isoformat()


def query_usage_records(
    job_id: str | None = None,
    resume_id: str | None = None,
    user_id: str | None = None,
    model: str | None = None,
    run_id: str | None = None,
    period: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    db_path: str | Path | None = None,
) -> list[dict]:
    clauses = []
    params = []

    if job_id:
        clauses.append("job_id = ?")
        params.append(job_id)

    if resume_id:
        clauses.append("resume_id = ?")
        params.append(resume_id)

    if user_id:
        clauses.append("user_id = ?")
        params.append(user_id)

    if model:
        clauses.append("model = ?")
        params.append(model)

    if run_id:
        clauses.append("run_id = ?")
        params.append(run_id)

    period_from = _period_start(period) if period else None
    if period_from:
        clauses.append("created_at >= ?")
        params.append(period_from)

    if start_date:
        clauses.append("created_at >= ?")
        params.append(start_date)

    if end_date:
        clauses.append("created_at <= ?")
        params.append(end_date)

    sql = "SELECT * FROM openai_usage"
    if clauses:
        sql += " WHERE " + " AND ".join(clauses)
    sql += " ORDER BY created_at ASC"

    connection = get_connection(db_path)
    try:
        rows = connection.execute(sql, params).fetchall()
        return [dict(row) for row in rows]
    finally:
        connection.close()


def summarize_records(records: list[dict]) -> dict:
    resume_ids = {
        row["resume_id"]
        for row in records
        if row.get("resume_id")
    }

    def _sum_int(key):
        values = [
            row[key]
            for row in records
            if row.get(key) is not None
        ]
        return int(sum(values)) if values else 0

    def _sum_cost(key):
        values = [
            row[key]
            for row in records
            if row.get(key) is not None
        ]
        return float(sum(values)) if values else 0.0

    total_resumes = len(resume_ids)
    total_cost = _sum_cost("total_cost")
    average_cost = (
        total_cost / total_resumes
        if total_resumes
        else 0.0
    )

    return {
        "total_resumes": total_resumes,
        "total_calls": len(records),
        "input_tokens": _sum_int("input_tokens"),
        "output_tokens": _sum_int("output_tokens"),
        "total_tokens": _sum_int("total_tokens"),
        "total_cost": total_cost,
        "average_cost_per_resume": average_cost,
    }


def summarize_by_resume(
    records: list[dict],
) -> dict[str, dict]:
    grouped = {}

    for row in records:
        resume_id = row.get("resume_id") or "Unassigned"
        bucket = grouped.setdefault(
            resume_id,
            {
                "resume_id": resume_id,
                "calls": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "total_cost": 0.0,
            },
        )
        bucket["calls"] += 1
        bucket["input_tokens"] += int(row.get("input_tokens") or 0)
        bucket["output_tokens"] += int(row.get("output_tokens") or 0)
        bucket["total_tokens"] += int(row.get("total_tokens") or 0)
        bucket["total_cost"] += float(row.get("total_cost") or 0.0)

    return grouped


def list_filter_values(column: str, db_path: str | Path | None = None) -> list[str]:
    allowed = {"job_id", "resume_id", "user_id", "model"}
    if column not in allowed:
        return []

    connection = get_connection(db_path)
    try:
        rows = connection.execute(
            f"""
            SELECT DISTINCT {column}
            FROM openai_usage
            WHERE {column} IS NOT NULL AND {column} != ''
            ORDER BY {column}
            """
        ).fetchall()
        return [row[0] for row in rows]
    finally:
        connection.close()
