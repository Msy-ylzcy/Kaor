from __future__ import annotations

import json
import zipfile

from fastapi.testclient import TestClient

import backend.diagnostics as diagnostics_module
from backend.app import create_app
from backend.diagnostics import (
    REPAIR_GUIDES,
    DiagnosticLogService,
    matching_guides,
    redact_log_text,
)


def test_real_error_lines_are_linked_to_repair_guides():
    assert "translation-timeout" in matching_guides(
        "translation request failed: HTTP 524 origin_response_timeout"
    )
    assert "uvr-model-damaged" in matching_guides(
        "BS-Roformer checkpoint download failed from the fixed upstream"
    )
    assert "gpu-out-of-memory" in matching_guides("CUDA out of memory")


def test_log_snapshot_parses_sources_levels_and_guides(tmp_path):
    log_dir = tmp_path / "logs"
    log_dir.mkdir(parents=True)
    (log_dir / "kaor.log").write_text(
        "2026-08-04 10:00:00,000Z | INFO | kaor.main | started\n"
        "2026-08-04 10:00:01,000Z | ERROR | kaor.jobs | HTTP 524 origin_response_timeout\n",
        encoding="utf-8",
    )

    payload = DiagnosticLogService(tmp_path).entries(levels=["ERROR"])

    assert len(payload["sources"]) == 1
    assert len(payload["entries"]) == 1
    entry = payload["entries"][0]
    assert entry["level"] == "ERROR"
    assert entry["guide_ids"] == ["translation-timeout"]
    assert all("path" not in source for source in payload["sources"])


def test_log_redaction_covers_common_provider_credentials():
    value = (
        'Authorization: Bearer sk-secret-token-123456 '
        'api_key="plain-provider-key" access_token=token-value'
    )
    redacted = redact_log_text(value)
    assert "secret-token" not in redacted
    assert "plain-provider-key" not in redacted
    assert "token-value" not in redacted
    assert redacted.count("[REDACTED]") == 3


def test_log_redaction_covers_json_basic_proxy_and_url_credentials():
    value = (
        '{"Authorization": "Basic dXNlcjpwYXNz", '
        '"Proxy-Authorization": "Token proxy-secret", '
        '"X-API-Key": "custom-secret", '
        '"url": "https://relay-user:relay-pass@example.test/v1"}'
    )

    redacted = redact_log_text(value)

    for secret in ("dXNlcjpwYXNz", "proxy-secret", "custom-secret", "relay-pass"):
        assert secret not in redacted
    assert redacted.count("[REDACTED]") == 4


def test_log_snapshot_uses_a_global_read_budget(tmp_path, monkeypatch):
    log_dir = tmp_path / "logs"
    log_dir.mkdir(parents=True)
    for index in range(20):
        (log_dir / f"worker-{index:02d}.log").write_text(
            "\n".join(f"line {line}" for line in range(100)) + "\n",
            encoding="utf-8",
        )
    read_counts: list[int] = []
    original = diagnostics_module._tail_lines

    def tracked(path, limit, max_bytes=2 * 1024 * 1024):
        lines = original(path, limit, max_bytes)
        read_counts.append(len(lines))
        return lines

    monkeypatch.setattr(diagnostics_module, "_tail_lines", tracked)

    payload = DiagnosticLogService(tmp_path).entries(tail=120)

    assert len(payload["entries"]) == 120
    assert sum(read_counts) <= 240


def test_diagnostic_exports_have_unique_names(tmp_path):
    service = DiagnosticLogService(tmp_path)

    first = service.export_bundle()
    second = service.export_bundle()

    assert first != second
    assert first.is_file()
    assert second.is_file()


def test_diagnostic_api_filters_and_exports_redacted_bundle(tmp_path):
    log_dir = tmp_path / "logs"
    log_dir.mkdir(parents=True)
    (log_dir / "kaor.log").write_text(
        "2026-08-04 10:00:00,000Z | ERROR | kaor.jobs | "
        "PaddleOCR is not installed Authorization: Bearer sk-do-not-export-123456\n",
        encoding="utf-8",
    )
    with TestClient(create_app(tmp_path)) as client:
        response = client.get("/api/diagnostics/logs?query=PaddleOCR")
        guides = client.get("/api/diagnostics/guides")
        troubleshooting = client.get("/api/diagnostics/troubleshooting")
        exported = client.get("/api/diagnostics/export")

    assert response.status_code == 200
    assert response.json()["entries"][0]["guide_ids"] == ["ocr-runtime-missing"]
    assert any(guide["id"] == "ocr-runtime-missing" for guide in guides.json())
    assert troubleshooting.status_code == 200
    assert troubleshooting.headers["content-type"].startswith("text/html")
    assert all(
        f'id="{guide.anchor}"' in troubleshooting.text for guide in REPAIR_GUIDES
    )
    assert exported.status_code == 200
    archive = tmp_path / "diagnostics.zip"
    archive.write_bytes(exported.content)
    with zipfile.ZipFile(archive) as bundle:
        system = json.loads(bundle.read("system.json"))
        assert "platform" in system
        assert not any("config" in name.casefold() for name in bundle.namelist())
        exported_log = bundle.read(next(name for name in bundle.namelist() if name.startswith("logs/")))
        assert b"do-not-export" not in exported_log
        assert b"[REDACTED]" in exported_log
