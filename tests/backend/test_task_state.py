from __future__ import annotations

from backend.task_state import TaskStateStore, task_signature


def test_task_state_requires_matching_source_and_options(tmp_path):
    source = tmp_path / "source.bin"
    source.write_bytes(b"first")
    first = task_signature(sources=[source], options={"batch": 4})
    state = TaskStateStore(tmp_path / "project")
    state.update("ocr", first, status="complete", artifact="ocr.csv")

    assert state.matches("ocr", first, "complete")
    assert not state.matches(
        "ocr",
        task_signature(sources=[source], options={"batch": 8}),
        "complete",
    )

    source.write_bytes(b"changed-source")
    assert not state.matches(
        "ocr",
        task_signature(sources=[source], options={"batch": 4}),
        "complete",
    )


def test_task_state_recovers_from_truncated_metadata(tmp_path):
    state = TaskStateStore(tmp_path / "project")
    state.path.parent.mkdir(parents=True)
    state.path.write_text("{truncated", encoding="utf-8")
    assert state.get("asr") is None
    state.update("asr", "signature", status="running", next_slice=12)
    assert state.get("asr")["next_slice"] == 12
