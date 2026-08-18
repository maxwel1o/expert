import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class CliTests(unittest.TestCase):
    def run_cli(self, db, *args):
        env = os.environ.copy()
        env["TEAM_PROGRESS_DB"] = str(db)
        return subprocess.run(
            [sys.executable, "-m", "team_progress.cli", *args],
            text=True,
            capture_output=True,
            env=env,
        )

    def test_two_jobs_do_not_mix_events(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "progress.db"
            for job in ("job-a", "job-b"):
                self.assertEqual(
                    0,
                    self.run_cli(
                        db, "job", "create", "--job-id", job, "--title", job
                    ).returncode,
                )
                self.assertEqual(
                    0,
                    self.run_cli(
                        db,
                        "task",
                        "add",
                        "--job-id",
                        job,
                        "--task-id",
                        f"task-{job[-1]}",
                        "--role",
                        "tester",
                    ).returncode,
                )
                self.assertEqual(
                    0,
                    self.run_cli(
                        db,
                        "update",
                        "--job-id",
                        job,
                        "--task-id",
                        f"task-{job[-1]}",
                        "--role",
                        "tester",
                        "--phase",
                        "check",
                        "--message",
                        f"{job} milestone",
                    ).returncode,
                )
            result = self.run_cli(db, "events", "--job-id", "job-a", "--json")
            rows = json.loads(result.stdout)
            self.assertTrue(rows)
            self.assertTrue(all(row["job_id"] == "job-a" for row in rows))

    def test_missing_identity_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = self.run_cli(Path(tmp) / "progress.db", "start")
            self.assertNotEqual(0, result.returncode)
            self.assertIn("job-id", result.stderr)

    def test_archive_hides_job_without_deleting_events(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "progress.db"
            self.run_cli(
                db, "job", "create", "--job-id", "job-a", "--title", "job-a"
            )
            self.assertEqual(
                0, self.run_cli(db, "job", "archive", "job-a").returncode
            )
            visible = self.run_cli(db, "status", "--all", "--json")
            all_rows = self.run_cli(
                db, "status", "--all", "--include-archived", "--json"
            )
            self.assertNotIn("job-a", visible.stdout)
            self.assertIn("job-a", all_rows.stdout)

    def test_complete_emits_final_summary_and_consumer_reads_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "progress.db"
            self.run_cli(
                db, "job", "create", "--job-id", "job-a", "--title", "job-a"
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
            result = self.run_cli(
                db,
                "complete",
                "--job-id",
                "job-a",
                "--task-id",
                "task-a",
                "--role",
                "tester",
                "--phase",
                "done",
                "--message",
                "test complete",
            )
            self.assertEqual(0, result.returncode, result.stderr)
            first = self.run_cli(
                db,
                "consume",
                "--consumer",
                "leader",
                "--job-id",
                "job-a",
                "--important-only",
                "--json",
            )
            second = self.run_cli(
                db,
                "consume",
                "--consumer",
                "leader",
                "--job-id",
                "job-a",
                "--important-only",
                "--json",
            )
            self.assertEqual(
                ["completed", "final_summary"],
                [row["kind"] for row in json.loads(first.stdout)],
            )
            self.assertEqual([], json.loads(second.stdout))


if __name__ == "__main__":
    unittest.main()
