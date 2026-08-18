import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from team_progress.adapters.hermes import extract_safe_tasks, register_active_tasks
from team_progress.store import ProgressStore


class HermesAdapterTests(unittest.TestCase):
    def payload(self):
        return {
            "tasks": [
                {
                    "id": "t-1",
                    "status": "running",
                    "assignee": "tester",
                    "title": "performance test",
                    "body": "password=must-never-be-copied",
                },
                {
                    "id": "t-2",
                    "status": "done",
                    "assignee": "deployer",
                    "title": "old done task",
                    "body": "irrelevant",
                },
                {
                    "id": "t-3",
                    "status": "todo",
                    "assignee": "leader",
                    "body": "not a worker",
                },
            ]
        }

    def test_ignores_body_and_registers_only_active_worker_fields(self):
        refs = extract_safe_tasks(self.payload())
        self.assertEqual(["t-1"], [ref.task_id for ref in refs])
        self.assertFalse(hasattr(refs[0], "body"))
        with tempfile.TemporaryDirectory() as tmp:
            store = ProgressStore(Path(tmp) / "progress.db")
            store.create_job("legacy active tasks", "legacy-20260725")
            self.assertEqual(
                1,
                register_active_tasks(
                    store, self.payload(), "legacy-20260725"
                ),
            )
            serialized = repr(store.status("legacy-20260725"))
            self.assertNotIn("must-never-be-copied", serialized)

    def test_cli_import_prints_only_count_and_job_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db = root / "progress.db"
            payload_path = root / "kanban.json"
            payload_path.write_text(json.dumps(self.payload()), encoding="utf-8")
            env = os.environ.copy()
            env["TEAM_PROGRESS_DB"] = str(db)
            create = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "team_progress.cli",
                    "job",
                    "create",
                    "--job-id",
                    "legacy-20260725",
                    "--title",
                    "legacy",
                ],
                text=True,
                capture_output=True,
                env=env,
            )
            self.assertEqual(0, create.returncode, create.stderr)
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "team_progress.cli",
                    "hermes",
                    "import-active",
                    "--input-json",
                    str(payload_path),
                    "--job-id",
                    "legacy-20260725",
                ],
                text=True,
                capture_output=True,
                env=env,
            )
            self.assertEqual(0, result.returncode, result.stderr)
            self.assertEqual(
                "registered=1 job_id=legacy-20260725", result.stdout.strip()
            )
            self.assertNotIn("must-never-be-copied", result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
