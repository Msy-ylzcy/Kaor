from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from threading import RLock
from typing import Any


_LOCKS: dict[Path, RLock] = {}
_LOCKS_GUARD = RLock()


def file_identity(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    try:
        stat = resolved.stat()
    except OSError:
        return {"path": str(resolved), "missing": True}
    return {
        "path": str(resolved),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def task_signature(*, sources: list[Path], options: Any) -> str:
    payload = {
        "sources": [file_identity(path) for path in sources],
        "options": options,
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class TaskStateStore:
    """Atomic per-project task metadata used to validate reuse and checkpoints."""

    def __init__(self, project_dir: Path) -> None:
        self.path = project_dir.resolve() / "cache" / "task-state.json"
        with _LOCKS_GUARD:
            self._lock = _LOCKS.setdefault(self.path, RLock())

    def _read(self) -> dict[str, Any]:
        if not self.path.is_file():
            return {"version": 1, "tasks": {}}
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {"version": 1, "tasks": {}}
        if not isinstance(payload, dict) or not isinstance(payload.get("tasks"), dict):
            return {"version": 1, "tasks": {}}
        return payload

    def _write(self, payload: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, default=str)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(self.path)

    def get(self, kind: str) -> dict[str, Any] | None:
        with self._lock:
            value = self._read()["tasks"].get(kind)
            return dict(value) if isinstance(value, dict) else None

    def matches(self, kind: str, signature: str, status: str | None = None) -> bool:
        record = self.get(kind)
        if record is None or record.get("signature") != signature:
            return False
        return status is None or record.get("status") == status

    def update(self, kind: str, signature: str, **values: Any) -> dict[str, Any]:
        with self._lock:
            payload = self._read()
            record = payload["tasks"].get(kind)
            if not isinstance(record, dict) or record.get("signature") != signature:
                record = {"signature": signature}
            record.update(values)
            payload["tasks"][kind] = record
            self._write(payload)
            return dict(record)

    def remove(self, kind: str) -> None:
        with self._lock:
            payload = self._read()
            if payload["tasks"].pop(kind, None) is not None:
                self._write(payload)


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, default=str)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def read_json(path: Path) -> Any | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
