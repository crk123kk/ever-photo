import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from app.services.pipeline import RestoreParams

logger = logging.getLogger(__name__)


class TaskStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    DONE = "done"
    FAILED = "failed"


@dataclass
class Task:
    task_id: str
    status: TaskStatus = TaskStatus.PENDING
    progress: float = 0.0
    step: str = ""
    input_path: Optional[Path] = None
    output_path: Optional[Path] = None
    params: Optional["RestoreParams"] = None
    error: Optional[str] = None


class TaskManager:
    def __init__(self):
        self._tasks: dict[str, Task] = {}
        self._queue: list[str] = []
        self._lock = threading.Lock()
        self._worker = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker.start()

    def create_task(
        self,
        input_path: Path,
        output_path: Path,
        params: Optional["RestoreParams"] = None,
    ) -> str:
        task_id = uuid.uuid4().hex[:12]
        task = Task(
            task_id=task_id,
            input_path=input_path,
            output_path=output_path,
            params=params,
        )
        with self._lock:
            self._tasks[task_id] = task
            self._queue.append(task_id)
        logger.info("Task created: %s", task_id)
        return task_id

    def get_task(self, task_id: str) -> Optional[Task]:
        return self._tasks.get(task_id)

    def _update_task(self, task_id: str, **kwargs):
        with self._lock:
            task = self._tasks.get(task_id)
            if task:
                for k, v in kwargs.items():
                    setattr(task, k, v)

    def _worker_loop(self):
        while True:
            task_id = None
            with self._lock:
                if self._queue:
                    task_id = self._queue.pop(0)

            if task_id is None:
                time.sleep(0.5)
                continue

            task = self._tasks.get(task_id)
            if not task or not task.input_path:
                continue

            from app.services.pipeline import pipeline

            self._update_task(
                task_id, status=TaskStatus.PROCESSING, progress=0.0, step="Starting"
            )

            try:
                pipeline.restore(
                    str(task.input_path),
                    str(task.output_path),
                    params=task.params,
                    progress_cb=lambda step, pct, name: self._update_task(
                        task_id, progress=pct, step=name
                    ),
                )
                self._update_task(
                    task_id, status=TaskStatus.DONE, progress=1.0, step="Complete"
                )
                logger.info("Task completed: %s", task_id)
            except Exception as e:
                self._update_task(
                    task_id,
                    status=TaskStatus.FAILED,
                    error=str(e),
                    step="Failed",
                )
                logger.exception("Task failed: %s", task_id)


task_manager = TaskManager()
