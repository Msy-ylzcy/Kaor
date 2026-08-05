import time
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier, Event

import pytest

from backend.jobs import JobManager


def wait_for(manager, job_id, timeout=2):
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = manager.get(job_id)
        if job["status"] in {"completed", "failed", "cancelled"}:
            return job
        time.sleep(0.01)
    raise AssertionError("job did not finish")


def test_job_reports_progress_and_result():
    manager = JobManager(max_workers=1)

    def runner(progress, cancel):
        progress(0.5, "Halfway")
        return {"cue_count": 2}

    submitted = manager.submit("project", "ocr", runner)
    result = wait_for(manager, submitted["id"])

    assert result["status"] == "completed"
    assert result["progress"] == 1
    assert result["result"] == {"cue_count": 2}


def test_job_preserves_actionable_exception_detail():
    manager = JobManager(max_workers=1)

    def runner(progress, cancel):
        raise PermissionError(r"denied: C:\Users\User\.cache\paddle")

    submitted = manager.submit("project", "ocr", runner)
    result = wait_for(manager, submitted["id"])

    assert result["status"] == "failed"
    assert result["error"]["code"] == "PermissionError"
    assert r".cache\paddle" in result["error"]["detail"]
    assert r".cache\paddle" in result["message"]


def test_job_exposes_latest_live_snapshot():
    manager = JobManager(max_workers=1)

    def runner(progress, cancel):
        progress(
            0.5,
            "OCR 1.0s",
            {"timestamp_ms": 1000, "cues": [{"source_text": "hello"}]},
        )
        return {"cue_count": 1}

    submitted = manager.submit("project", "ocr", runner)
    result = wait_for(manager, submitted["id"])

    assert result["snapshot"]["timestamp_ms"] == 1000
    assert result["snapshot"]["cues"][0]["source_text"] == "hello"


def test_clear_project_removes_finished_jobs_only():
    manager = JobManager(max_workers=1)
    first = manager.submit("first", "ocr", lambda _progress, _cancel: {})
    second = manager.submit("second", "ocr", lambda _progress, _cancel: {})
    wait_for(manager, first["id"])
    wait_for(manager, second["id"])

    assert manager.has_active("first") is False
    manager.clear_project("first")

    assert manager.list("first") == []
    assert len(manager.list("second")) == 1


def test_submit_unique_rejects_a_second_active_job():
    manager = JobManager(max_workers=1)
    started = Event()
    release = Event()

    def runner(_progress, cancel):
        started.set()
        while not release.wait(0.01):
            if cancel.is_set():
                raise InterruptedError("job cancelled")
        return {}

    first = manager.submit_unique("project", "local-model-deploy", runner)
    assert started.wait(1)

    with pytest.raises(RuntimeError, match="active job"):
        manager.submit_unique("project", "local-model-deploy", runner)

    release.set()
    assert wait_for(manager, first["id"])["status"] == "completed"
    manager.shutdown()


def test_submit_unique_is_atomic_for_concurrent_callers():
    manager = JobManager(max_workers=1)
    callers_ready = Barrier(3)
    release = Event()

    def runner(_progress, cancel):
        while not release.wait(0.01):
            if cancel.is_set():
                raise InterruptedError("job cancelled")
        return {}

    def submit():
        callers_ready.wait()
        try:
            return manager.submit_unique("project", "local-model-deploy", runner)
        except RuntimeError:
            return None

    with ThreadPoolExecutor(max_workers=2) as callers:
        futures = [callers.submit(submit) for _ in range(2)]
        callers_ready.wait()
        results = [future.result(timeout=1) for future in futures]

    submitted = [result for result in results if result is not None]
    assert len(submitted) == 1
    release.set()
    assert wait_for(manager, submitted[0]["id"])["status"] == "completed"
    manager.shutdown()


def test_shutdown_cancels_running_jobs_and_rejects_new_submissions():
    manager = JobManager(max_workers=1)
    started = Event()

    def runner(_progress, cancel):
        started.set()
        cancel.wait(2)
        if cancel.is_set():
            raise InterruptedError("job cancelled")
        return {}

    submitted = manager.submit("project", "local-model-deploy", runner)
    assert started.wait(1)

    manager.shutdown(cancel=True, wait=True)

    assert manager.get(submitted["id"])["status"] == "cancelled"
    with pytest.raises(RuntimeError, match="shut down"):
        manager.submit("project", "ocr", runner)
