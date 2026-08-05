from __future__ import annotations

import builtins
import json
from pathlib import Path
from types import SimpleNamespace

from backend import audio_worker


def test_uvr_reports_runtime_loading_before_import(tmp_path, monkeypatch):
    request_path = tmp_path / "request.json"
    result_path = tmp_path / "result.json"
    progress_path = tmp_path / "progress.json"
    output_dir = tmp_path / "output"
    request_path.write_text(
        json.dumps(
            {
                "input_wav": str(tmp_path / "input.wav"),
                "output_dir": str(output_dir),
                "device": "cpu",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(audio_worker, "_configure", lambda: None)
    original_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "speech_pipeline" and level == 1:
            payload = json.loads(progress_path.read_text(encoding="utf-8"))
            assert payload == {
                "progress": 0.005,
                "message": "Loading packaged audio runtime",
            }

            def separate_vocals(_input, destination, **_kwargs):
                destination.mkdir(parents=True, exist_ok=True)
                vocals = destination / "vocals.wav"
                vocals.write_bytes(b"RIFF" + b"\0" * 44)
                return vocals

            return SimpleNamespace(separate_vocals=separate_vocals)
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    assert audio_worker._run_uvr(request_path, result_path, progress_path) == 0
    assert json.loads(result_path.read_text(encoding="utf-8"))["status"] == "ok"
    assert json.loads(progress_path.read_text(encoding="utf-8")) == {
        "progress": 1.0,
        "message": "Vocal separation completed",
    }
