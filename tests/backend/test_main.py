from __future__ import annotations

import sys

import uvicorn

import backend.main as main_module


async def _asgi_app(scope, receive, send):
    return None


def _remove_standard_streams() -> tuple[object, object, object, object]:
    original_streams = (sys.stdout, sys.stderr, sys.__stdout__, sys.__stderr__)
    sys.stdout = None
    sys.stderr = None
    sys.__stdout__ = None
    sys.__stderr__ = None
    return original_streams


def _restore_standard_streams(original_streams) -> None:
    replacement_streams = (sys.stdout, sys.stderr, sys.__stdout__, sys.__stderr__)
    sys.stdout, sys.stderr, sys.__stdout__, sys.__stderr__ = original_streams

    unique_replacements = {id(stream): stream for stream in replacement_streams}
    for stream in unique_replacements.values():
        if (
            stream is not None
            and all(stream is not original for original in original_streams)
            and not stream.closed
        ):
            stream.close()


def test_ensure_standard_streams_supports_uvicorn_without_console():
    original_streams = _remove_standard_streams()
    try:
        main_module.ensure_standard_streams()

        assert sys.stdout is not None
        assert sys.stderr is not None
        assert isinstance(sys.stdout.isatty(), bool)
        assert isinstance(sys.stderr.isatty(), bool)
        assert sys.stdout.writable()
        assert sys.stderr.writable()

        # Uvicorn configures its DefaultFormatter during Config construction.
        uvicorn.Config(_asgi_app, log_level="warning")
    finally:
        _restore_standard_streams(original_streams)


def test_main_repairs_streams_before_runtime_and_uvicorn(monkeypatch, tmp_path):
    events: list[str] = []

    def configure_runtime() -> None:
        assert sys.stdout is not None
        assert sys.stderr is not None
        events.append("runtime")

    def run_uvicorn(app, **kwargs) -> None:
        assert sys.stdout is not None
        assert sys.stderr is not None
        events.append("uvicorn")
        uvicorn.Config(app, **kwargs)

    monkeypatch.setenv("KAOR_NO_BROWSER", "1")
    monkeypatch.setattr(main_module, "configure_runtime_directories", configure_runtime)
    monkeypatch.setattr(main_module, "find_free_port", lambda preferred: 8876)
    monkeypatch.setattr(main_module, "portable_data_directory", lambda: tmp_path)
    monkeypatch.setattr(main_module, "create_app", lambda data_dir: _asgi_app)
    monkeypatch.setattr(main_module.uvicorn, "run", run_uvicorn)

    original_streams = _remove_standard_streams()
    try:
        main_module.main()
    finally:
        _restore_standard_streams(original_streams)

    assert events == ["runtime", "uvicorn"]


def test_ensure_standard_streams_preserves_existing_console_streams():
    original_stdout = sys.stdout
    original_stderr = sys.stderr

    main_module.ensure_standard_streams()

    assert sys.stdout is original_stdout
    assert sys.stderr is original_stderr
