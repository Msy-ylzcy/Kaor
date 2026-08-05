from __future__ import annotations

from collections import deque
from copy import deepcopy
from typing import Any


_EVENT_LABELS = {
    "provider_request_started": "Waiting for the upstream model",
    "provider_delta": "Receiving the upstream model stream",
    "provider_retry": "Retrying the upstream request",
    "batch_split": "Splitting a slow output subset",
    "subset_completed": "Saved an AI output subset",
}


class AiTraceAccumulator:
    """Bounded, serializable live trace for fusion and translation jobs."""

    def __init__(self, operation: str) -> None:
        self.operation = operation
        self.phase = "starting"
        self.status = "Preparing the AI request"
        self.reasoning = ""
        self.output_preview = ""
        self.subset = ""
        self.attempt = 0
        self.completed = 0
        self.total = 0
        self.events: deque[dict[str, Any]] = deque(maxlen=24)

    def consume(self, event: dict[str, Any]) -> None:
        event_type = str(event.get("type") or "provider_event")
        self.phase = event_type
        self.status = _EVENT_LABELS.get(event_type, event_type.replace("_", " "))
        if event.get("subset") is not None:
            self.subset = str(event["subset"])
        if event.get("attempt") is not None:
            self.attempt = int(event["attempt"])
        if event.get("completed") is not None:
            self.completed = int(event["completed"])
        if event.get("total") is not None:
            self.total = int(event["total"])
        if event_type == "provider_delta":
            delta = str(event.get("delta") or "")
            if event.get("field") == "reasoning_content":
                self.reasoning = (self.reasoning + delta)[-100_000:]
            elif event.get("field") == "content":
                self.output_preview = (self.output_preview + delta)[-30_000:]
        compact = {
            key: value
            for key, value in event.items()
            if key not in {"delta", "reasoning_content", "content"}
        }
        self.events.append(compact)

    def snapshot(self) -> dict[str, Any]:
        return deepcopy(
            {
                "operation": self.operation,
                "phase": self.phase,
                "status": self.status,
                "reasoning": self.reasoning,
                "reasoning_content": self.reasoning,
                "content_preview": self.output_preview,
                "output_preview": self.output_preview,
                "subset": self.subset,
                "batch": self.subset,
                "attempt": self.attempt,
                "completed": self.completed,
                "total": self.total,
                "events": list(self.events),
            }
        )
