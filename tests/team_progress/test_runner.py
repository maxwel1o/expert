import sys
import tempfile
import unittest
from pathlib import Path

from team_progress.model import EventKind, Role
from team_progress.runner import Identity, run_command
from team_progress.store import ProgressStore


class RunnerTests(unittest.TestCase):
    def test_success_records_start_heartbeat_and_complete(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = ProgressStore(Path(tmp) / "progress.db", heartbeat_seconds=0.1)
            store.create_job("job", "job-a")
            store.add_task("job-a", "task-a", Role.TESTER)
            identity = Identity("job-a", "task-a", Role.TESTER, "smoke")
            code = run_command(
                store,
                identity,
                [sys.executable, "-c", "import time; time.sleep(0.2)"],
                (),
                heartbeat_seconds=0.1,
            )
            self.assertEqual(0, code)
            kinds = [event.kind for event in store.read_events("job-a")]
            self.assertIn(EventKind.STARTED, kinds)
            self.assertIn(EventKind.HEARTBEAT, kinds)
            self.assertIn(EventKind.COMPLETED, kinds)
            self.assertEqual(EventKind.FINAL_SUMMARY, kinds[-1])

    def test_failure_records_exit_code_and_final_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = ProgressStore(Path(tmp) / "progress.db")
            store.create_job("job", "job-a")
            store.add_task("job-a", "task-a", Role.TESTER)
            code = run_command(
                store,
                Identity("job-a", "task-a", Role.TESTER, "smoke"),
                [sys.executable, "-c", "raise SystemExit(7)"],
                (),
            )
            self.assertEqual(7, code)
            events = store.read_events("job-a")
            self.assertEqual(EventKind.FAILED, events[-2].kind)
            self.assertEqual(EventKind.FINAL_SUMMARY, events[-1].kind)

    def test_command_arguments_are_not_saved(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = ProgressStore(Path(tmp) / "progress.db")
            store.create_job("job", "job-a")
            store.add_task("job-a", "task-a", Role.TESTER)
            run_command(
                store,
                Identity("job-a", "task-a", Role.TESTER, "smoke"),
                [
                    sys.executable,
                    "-c",
                    "raise SystemExit(0)",
                    "--token",
                    "not-recorded-value",
                ],
                (),
            )
            text = " ".join(event.message for event in store.read_events("job-a"))
            self.assertNotIn("not-recorded-value", text)

    def test_nonblocking_lock_conflict_returns_75(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = ProgressStore(Path(tmp) / "progress.db")
            store.create_job("job", "job-a")
            store.add_task("job-a", "task-a", Role.TESTER)
            store.add_task("job-a", "task-b", Role.TESTER)
            self.assertEqual((True, None), store.acquire("task-a", ()))
            code = run_command(
                store,
                Identity("job-a", "task-b", Role.TESTER, "smoke"),
                [sys.executable, "-c", "raise SystemExit(0)"],
                (),
                lock_timeout_seconds=0,
            )
            self.assertEqual(75, code)


if __name__ == "__main__":
    unittest.main()
