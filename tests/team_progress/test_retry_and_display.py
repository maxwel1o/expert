import tempfile
import unittest
from pathlib import Path

from team_progress.cli import should_display
from team_progress.model import (
    EventKind,
    ProgressEvent,
    Role,
    TaskStatus,
)
from team_progress.store import ProgressStore


class RetryAndDisplayTests(unittest.TestCase):
    def test_retry_reopens_finalized_job_and_creates_new_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = ProgressStore(Path(tmp) / "progress.db")
            store.create_job("job", "job-a")
            store.add_task("job-a", "task-a", Role.TESTER)
            store.record(
                "job-a",
                "task-a",
                Role.TESTER,
                EventKind.FAILED,
                TaskStatus.FAILED,
                "first",
                None,
                "first attempt failed",
            )
            self.assertIsNotNone(store.maybe_finalize("job-a"))
            store.record(
                "job-a",
                "task-a",
                Role.TESTER,
                EventKind.RETRYING,
                TaskStatus.QUEUED,
                "retry",
                None,
                "retry requested",
            )
            self.assertIsNone(store.maybe_finalize("job-a"))
            store.record(
                "job-a",
                "task-a",
                Role.TESTER,
                EventKind.COMPLETED,
                TaskStatus.COMPLETED,
                "done",
                100,
                "retry completed",
            )
            self.assertIsNotNone(store.maybe_finalize("job-a"))
            summaries = [
                event
                for event in store.read_events("job-a")
                if event.kind is EventKind.FINAL_SUMMARY
            ]
            self.assertEqual(2, len(summaries))

    def test_background_heartbeat_hidden_but_completion_shown(self):
        base = dict(
            seq=1,
            event_id="e",
            job_id="background-job",
            task_id="task-a",
            parent_id=None,
            role=Role.TESTER,
            status=TaskStatus.RUNNING,
            phase="test",
            percent=None,
            message="message",
            artifact_refs=(),
            important=False,
            created_at="2026-07-25T00:00:00Z",
        )
        heartbeat = ProgressEvent(kind=EventKind.HEARTBEAT, **base)
        completed = ProgressEvent(
            kind=EventKind.COMPLETED,
            **{**base, "status": TaskStatus.COMPLETED, "important": True},
        )
        self.assertFalse(should_display(heartbeat, "foreground-job"))
        self.assertTrue(should_display(completed, "foreground-job"))
        self.assertTrue(should_display(heartbeat, "background-job"))


if __name__ == "__main__":
    unittest.main()
