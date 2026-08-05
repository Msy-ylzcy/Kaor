from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from threading import Event, RLock
from typing import Callable, Literal
from uuid import uuid4


JobStatus = Literal["queued", "running", "failed", "cancelled", "completed"]
Progress = Callable[[float, str, dict[str, object] | None], None]
Runner = Callable[[Progress, Event], dict[str, object] | None]
logger = logging.getLogger("kaor.jobs")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class JobRecord:
    id: str
    project_id: str
    kind: str
    status: JobStatus = "queued"
    stage: str = "queued"
    progress: float = 0.0
    message: str = "Waiting"
    error: dict[str, str] | None = None
    result: dict[str, object] | None = None
    snapshot: dict[str, object] | None = None
    created_at: str = field(default_factory=_now)
    updated_at: str = field(default_factory=_now)
    cancel_event: Event = field(default_factory=Event, repr=False)
    future: Future | None = field(default=None, repr=False)

    def public(self) -> dict[str, object]:
        return {
            "id": self.id,
            "project_id": self.project_id,
            "kind": self.kind,
            "status": self.status,
            "stage": self.stage,
            "progress": self.progress,
            "message": self.message,
            "error": self.error,
            "result": self.result,
            "snapshot": self.snapshot,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


class JobManager:
    def __init__(self, max_workers: int = 2) -> None:
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="kaor"
        )
        self._jobs: dict[str, JobRecord] = {}
        self._lock = RLock()
        self._closed = False

    def submit(self, project_id: str, kind: str, runner: Runner) -> dict[str, object]:
        with self._lock:
            return self._submit_locked(project_id, kind, runner)

    def submit_unique(
        self, project_id: str, kind: str, runner: Runner
    ) -> dict[str, object]:
        with self._lock:
            if self._has_active_locked(project_id):
                raise RuntimeError(f"an active job already exists for project {project_id}")
            return self._submit_locked(project_id, kind, runner)

    def _submit_locked(
        self, project_id: str, kind: str, runner: Runner
    ) -> dict[str, object]:
        if self._closed:
            raise RuntimeError("job manager has been shut down")
        job = JobRecord(id=str(uuid4()), project_id=project_id, kind=kind)
        self._jobs[job.id] = job
        job.future = self._executor.submit(self._run, job, runner)
        logger.info("job queued id=%s project=%s kind=%s", job.id, project_id, kind)
        return job.public()

    def _run(self, job: JobRecord, runner: Runner) -> None:
        logger.info("job started id=%s project=%s kind=%s", job.id, job.project_id, job.kind)
        self._update(job, status="running", stage="starting", message="Starting")

        def progress(
            value: float,
            message: str,
            snapshot: dict[str, object] | None = None,
        ) -> None:
            if job.cancel_event.is_set():
                raise InterruptedError("job cancelled")
            values: dict[str, object] = {
                "progress": max(0.0, min(1.0, value)),
                "stage": job.kind,
                "message": message,
            }
            if snapshot is not None:
                values["snapshot"] = snapshot
            self._update(job, **values)
            logger.info(
                "job progress id=%s kind=%s progress=%d%% message=%s",
                job.id,
                job.kind,
                round(float(values["progress"]) * 100),
                message,
            )

        try:
            result = runner(progress, job.cancel_event)
            if job.cancel_event.is_set():
                self._update(
                    job, status="cancelled", stage="cancelled", message="Cancelled"
                )
            else:
                self._update(
                    job,
                    status="completed",
                    stage="completed",
                    progress=1.0,
                    message="Completed",
                    result=result or {},
                )
                logger.info("job completed id=%s kind=%s", job.id, job.kind)
        except InterruptedError:
            self._update(job, status="cancelled", stage="cancelled", message="Cancelled")
            logger.warning("job cancelled id=%s kind=%s", job.id, job.kind)
        except Exception as exc:  # Worker failures are surfaced through the job API.
            self._update(
                job,
                status="failed",
                stage="failed",
                message=str(exc),
                error={"code": type(exc).__name__, "detail": str(exc)},
            )
            logger.exception("job failed id=%s kind=%s: %s", job.id, job.kind, exc)

    def _update(self, job: JobRecord, **values: object) -> None:
        with self._lock:
            for key, value in values.items():
                setattr(job, key, value)
            job.updated_at = _now()

    def list(self, project_id: str | None = None) -> list[dict[str, object]]:
        with self._lock:
            jobs = list(self._jobs.values())
            if project_id:
                jobs = [job for job in jobs if job.project_id == project_id]
            jobs.sort(key=lambda item: item.created_at, reverse=True)
            return [job.public() for job in jobs]

    def get(self, job_id: str) -> dict[str, object]:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise KeyError(job_id)
            return job.public()

    def cancel(self, job_id: str) -> dict[str, object]:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise KeyError(job_id)
            job.cancel_event.set()
            if job.status == "queued" and job.future and job.future.cancel():
                job.status = "cancelled"
                job.stage = "cancelled"
                job.message = "Cancelled"
                job.updated_at = _now()
            return job.public()

    def has_active(self, project_id: str) -> bool:
        with self._lock:
            return self._has_active_locked(project_id)

    def _has_active_locked(self, project_id: str) -> bool:
        return any(
            job.project_id == project_id and job.status in {"queued", "running"}
            for job in self._jobs.values()
        )

    def shutdown(self, *, cancel: bool = True, wait: bool = True) -> None:
        with self._lock:
            self._closed = True
            if cancel:
                for job in self._jobs.values():
                    if job.status not in {"queued", "running"}:
                        continue
                    job.cancel_event.set()
                    if job.status == "queued" and job.future and job.future.cancel():
                        job.status = "cancelled"
                        job.stage = "cancelled"
                        job.message = "Cancelled"
                        job.updated_at = _now()
        self._executor.shutdown(wait=wait, cancel_futures=cancel)

    def clear_project(self, project_id: str) -> None:
        with self._lock:
            if any(
                job.project_id == project_id and job.status in {"queued", "running"}
                for job in self._jobs.values()
            ):
                raise RuntimeError("project still has active jobs")
            self._jobs = {
                job_id: job
                for job_id, job in self._jobs.items()
                if job.project_id != project_id
            }
