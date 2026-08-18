from dataclasses import dataclass

from .model import ProgressEvent, TaskStatus


@dataclass(frozen=True)
class ReconciliationResult:
    job_id: str
    task_id: str
    previous_status: TaskStatus
    current_status: TaskStatus
    changed: bool
    conflict: bool
    event: ProgressEvent | None
