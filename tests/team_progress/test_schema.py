import tempfile
import unittest
from pathlib import Path

from team_progress.schema import SCHEMA_VERSION, connect, initialize


class SchemaTests(unittest.TestCase):
    def test_initialize_creates_required_tables_and_wal(self):
        with tempfile.TemporaryDirectory() as tmp:
            conn = connect(Path(tmp) / "progress.db")
            initialize(conn)
            tables = {
                row[0]
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                )
            }
            self.assertTrue(
                {"meta", "jobs", "tasks", "events", "consumers", "locks"} <= tables
            )
            self.assertEqual("wal", conn.execute("PRAGMA journal_mode").fetchone()[0])
            self.assertEqual(
                str(SCHEMA_VERSION),
                conn.execute(
                    "SELECT value FROM meta WHERE key='schema_version'"
                ).fetchone()[0],
            )
            conn.close()

    def test_initialize_is_idempotent(self):
        with tempfile.TemporaryDirectory() as tmp:
            conn = connect(Path(tmp) / "progress.db")
            initialize(conn)
            initialize(conn)
            count = conn.execute(
                "SELECT COUNT(*) FROM meta WHERE key='schema_version'"
            ).fetchone()[0]
            self.assertEqual(1, count)
            conn.close()


if __name__ == "__main__":
    unittest.main()
