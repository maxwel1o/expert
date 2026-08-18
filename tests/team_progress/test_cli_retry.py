import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class CliRetryTests(unittest.TestCase):
    def run_cli(self, db, *args):
        env = os.environ.copy()
        env["TEAM_PROGRESS_DB"] = str(db)
        return subprocess.run(
            [sys.executable, "-m", "team_progress.cli", *args],
            text=True,
            capture_output=True,
            env=env,
        )

    def test_retry_command_reopens_task_as_queued(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "progress.db"
            self.run_cli(
                db, "job", "create", "--job-id", "job-a", "--title", "job"
            )
            self.run_cli(
                db,
                "task",
                "add",
                "--job-id",
                "job-a",
                "--task-id",
                "task-a",
                "--role",
                "tester",
            )
            self.run_cli(
                db,
                "fail",
                "--job-id",
                "job-a",
                "--task-id",
                "task-a",
                "--role",
                "tester",
                "--message",
                "first attempt failed",
            )
            result = self.run_cli(
                db,
                "retry",
                "--job-id",
                "job-a",
                "--task-id",
                "task-a",
                "--role",
                "tester",
                "--message",
                "retry approved",
            )
            self.assertEqual(0, result.returncode, result.stderr)
            status = json.loads(
                self.run_cli(db, "status", "job-a", "--json").stdout
            )
            self.assertEqual("queued", status["tasks"][0]["status"])


if __name__ == "__main__":
    unittest.main()
