"""PyInstaller entry point shared by the UI and the isolated audio worker."""

from __future__ import annotations

import sys
from pathlib import Path


def main() -> int:
    executable = Path(sys.executable).stem.casefold()
    if executable == "kaoraudioworker":
        from backend.audio_worker import main as worker_main

        return worker_main()

    from backend.startup_trace import trace_startup

    trace_startup("entry point started")
    from backend.main import main as application_main

    trace_startup("backend.main imported")
    trace_startup("calling backend.main.main")
    application_main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
