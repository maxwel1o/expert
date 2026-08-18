import json
import sqlite3
from pathlib import Path


def create_hermes_db(path: Path, tasks: list[dict]) -> None:
    conn = sqlite3.connect(path)
    try:
        conn.executescript(
            """
            CREATE TABLE tasks (
              id TEXT PRIMARY KEY,
              assignee TEXT,
              status TEXT,
              result TEXT,
              started_at TEXT,
              completed_at TEXT,
              current_run_id INTEGER
            );
            CREATE TABLE task_runs (
              id INTEGER PRIMARY KEY,
              task_id TEXT,
              status TEXT,
              outcome TEXT,
              summary TEXT,
              ended_at TEXT
            );
            CREATE TABLE task_events (
              id INTEGER PRIMARY KEY,
              task_id TEXT,
              kind TEXT,
              payload TEXT,
              created_at TEXT
            );
            """
        )
        for row in tasks:
            conn.execute(
                "INSERT INTO tasks VALUES(?,?,?,?,?,?,?)",
                (
                    row["id"],
                    row["assignee"],
                    row["status"],
                    row.get("result"),
                    row.get("started_at"),
                    row.get("completed_at"),
                    row.get("run_id"),
                ),
            )
            if row.get("run_id") is not None:
                conn.execute(
                    "INSERT INTO task_runs VALUES(?,?,?,?,?,?)",
                    (
                        row["run_id"],
                        row["id"],
                        row.get("run_status"),
                        row.get("outcome"),
                        row.get("summary"),
                        row.get("ended_at"),
                    ),
                )
            if row.get("event_id") is not None:
                conn.execute(
                    "INSERT INTO task_events VALUES(?,?,?,?,?)",
                    (
                        row["event_id"],
                        row["id"],
                        row.get("event_kind", "completed"),
                        json.dumps(row.get("payload", {})),
                        row.get("event_at"),
                    ),
                )
        conn.commit()
    finally:
        conn.close()
