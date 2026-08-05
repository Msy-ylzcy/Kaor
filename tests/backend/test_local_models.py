from __future__ import annotations

import time
import zipfile
from pathlib import Path
from threading import Event

import httpx
import pytest
from fastapi.testclient import TestClient

from backend.app import create_app
from backend.local_models import (
    GpuAdapter,
    HardwareProfile,
    LocalModelManager,
    recommend_model,
    recommend_runtime,
    select_runtime_assets,
)


def hardware(
    *,
    vendor: str = "unknown",
    gpu_memory: int = 0,
    system_memory: int = 16 * 1024**3,
    build_profile: str = "",
) -> HardwareProfile:
    gpus = (
        (GpuAdapter(f"{vendor} test adapter", vendor, gpu_memory),)
        if vendor != "unknown"
        else ()
    )
    return HardwareProfile(
        system="Windows",
        architecture="AMD64",
        cpu_name="Test CPU",
        logical_cpus=16,
        memory_bytes=system_memory,
        gpus=gpus,
        build_profile=build_profile,
    )


def wait_for_job(client: TestClient, job_id: str, timeout: float = 2) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        payload = client.get(f"/api/jobs/{job_id}").json()
        if payload["status"] in {"completed", "failed", "cancelled"}:
            return payload
        time.sleep(0.01)
    raise AssertionError("job did not finish")


def test_hardware_recommendation_selects_gpu_runtime_and_memory_tier():
    amd = hardware(vendor="amd", gpu_memory=6 * 1024**3)
    nvidia = hardware(vendor="nvidia", gpu_memory=16 * 1024**3)
    forced_cpu = hardware(vendor="nvidia", gpu_memory=16 * 1024**3, build_profile="cpu")

    assert recommend_runtime(amd) == "vulkan"
    assert recommend_model(amd, "vulkan").id == "qwen3-4b-q4-k-m"
    assert recommend_runtime(nvidia) == "cuda"
    assert recommend_model(nvidia, "cuda").id == "qwen3-14b-q4-k-m"
    assert recommend_runtime(forced_cpu) == "cpu"


def test_cuda_runtime_selects_matching_cudart_companion():
    release = {
        "tag_name": "b9999",
        "assets": [
            {
                "name": "llama-b9999-bin-win-cuda-12.4-x64.zip",
                "browser_download_url": "https://example.test/llama.zip",
                "size": 10,
            },
            {
                "name": "cudart-llama-bin-win-cuda-12.4-x64.zip",
                "browser_download_url": "https://example.test/cudart.zip",
                "size": 20,
            },
            {
                "name": "llama-b9999-bin-win-cpu-x64.zip",
                "browser_download_url": "https://example.test/cpu.zip",
                "size": 5,
            },
        ],
    }

    tag, assets = select_runtime_assets(release, "cuda")

    assert tag == "b9999"
    assert [asset["name"] for asset in assets] == [
        "llama-b9999-bin-win-cuda-12.4-x64.zip",
        "cudart-llama-bin-win-cuda-12.4-x64.zip",
    ]


def test_download_resumes_an_existing_partial_file(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["range"] == "bytes=5-"
        return httpx.Response(206, content=b" world", headers={"content-length": "6"})

    manager = LocalModelManager(
        tmp_path / "local-models",
        transport=httpx.MockTransport(handler),
        hardware_detector=lambda: hardware(),
    )
    destination = tmp_path / "artifact.bin"
    destination.with_suffix(".bin.part").write_bytes(b"hello")
    samples: list[tuple[int, int | None]] = []

    manager._download(
        "https://example.test/artifact.bin",
        destination,
        progress=lambda done, total: samples.append((done, total)),
    )

    assert destination.read_bytes() == b"hello world"
    assert samples[-1] == (11, 11)


def test_runtime_archive_rejects_path_traversal(tmp_path):
    archive = tmp_path / "runtime.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("../outside.txt", "bad")

    with pytest.raises(RuntimeError, match="unsafe path"):
        LocalModelManager._extract_archives([archive], tmp_path / "runtime")

    assert not (tmp_path / "outside.txt").exists()


def test_deploy_reuses_complete_managed_artifacts_without_release_request(
    tmp_path, monkeypatch
):
    manager = LocalModelManager(
        tmp_path / "managed",
        hardware_detector=lambda: hardware(build_profile="cpu"),
    )
    executable = tmp_path / "llama-server.exe"
    model = manager.models_dir / "Qwen3-8B-Q4_K_M.gguf"
    executable.write_bytes(b"fixture")
    model.write_bytes(b"fixture")
    manager.configure(
        {
            "mode": "managed",
            "model": "qwen3-8b-q4-k-m",
            "executable_path": str(executable),
            "model_path": str(model),
            "runtime_variant": "cpu",
        }
    )
    monkeypatch.setattr(manager, "start", lambda: {})
    monkeypatch.setattr(manager, "_probe", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(manager, "_verify_download", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        manager,
        "_release",
        lambda: (_ for _ in ()).throw(AssertionError("network should not be used")),
    )
    updates = []

    result = manager.deploy(
        {
            "model_id": "qwen3-8b-q4-k-m",
            "runtime_variant": "cpu",
            "port": 18081,
            "context_size": 32768,
            "gpu_layers": 0,
            "threads": 3,
            "auto_start": False,
        },
        lambda value, message, snapshot: updates.append((value, message, snapshot)),
        Event(),
    )

    assert result["reused"] is True
    assert result["configuration"]["port"] == 18081
    assert result["configuration"]["context_size"] == 32768
    assert result["configuration"]["threads"] == 3
    assert result["configuration"]["auto_start"] is False
    assert updates[-1][1] == "Local model is ready"


def test_deploy_never_deletes_a_user_model_outside_the_managed_directory(
    tmp_path, monkeypatch
):
    manager = LocalModelManager(
        tmp_path / "managed",
        hardware_detector=lambda: hardware(build_profile="cpu"),
    )
    executable = tmp_path / "llama-server.exe"
    user_model = tmp_path / "user-owned.gguf"
    executable.write_bytes(b"fixture")
    user_model.write_bytes(b"GGUF-user-owned")
    manager.configure(
        {
            "mode": "managed",
            "model": "qwen3-1.7b-q8-0",
            "executable_path": str(executable),
            "model_path": str(user_model),
            "runtime_variant": "cpu",
        }
    )
    monkeypatch.setattr(
        manager,
        "_release",
        lambda: (_ for _ in ()).throw(RuntimeError("stop after validation")),
    )

    with pytest.raises(RuntimeError, match="stop after validation"):
        manager.deploy(
            {"model_id": "qwen3-1.7b-q8-0", "runtime_variant": "cpu"},
            lambda *_args: None,
            Event(),
        )

    assert user_model.read_bytes() == b"GGUF-user-owned"


def test_cancel_after_reconfiguration_restores_the_previous_managed_settings(
    tmp_path, monkeypatch
):
    manager = LocalModelManager(
        tmp_path / "managed",
        hardware_detector=lambda: hardware(build_profile="cpu"),
    )
    executable = tmp_path / "llama-server.exe"
    model = manager.models_dir / "Qwen3-1.7B-Q8_0.gguf"
    executable.write_bytes(b"fixture")
    model.write_bytes(b"fixture")
    manager.configure(
        {
            "mode": "managed",
            "model": "qwen3-1.7b-q8-0",
            "executable_path": str(executable),
            "model_path": str(model),
            "runtime_variant": "cpu",
            "context_size": 8192,
        }
    )
    monkeypatch.setattr(manager, "_verify_download", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        manager,
        "_release",
        lambda: (_ for _ in ()).throw(AssertionError("runtime should be reused")),
    )

    def cancel_before_start(value, _message, _snapshot):
        if value >= 0.95:
            raise InterruptedError("job cancelled")

    with pytest.raises(InterruptedError, match="job cancelled"):
        manager.deploy(
            {
                "model_id": "qwen3-1.7b-q8-0",
                "runtime_variant": "cpu",
                "context_size": 32768,
            },
            cancel_before_start,
            Event(),
        )

    assert manager.status(include_hardware=False)["configuration"]["context_size"] == 8192


def test_external_local_model_configuration_becomes_translation_default(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == "http://127.0.0.1:18080/v1/models"
        return httpx.Response(200, json={"data": [{"id": "local-qwen"}]})

    manager = LocalModelManager(
        tmp_path / "managed",
        transport=httpx.MockTransport(handler),
        hardware_detector=lambda: hardware(),
    )
    app = create_app(tmp_path / "data", local_model_manager=manager)
    with TestClient(app) as client:
        remote_config = client.get("/api/config").json()
        remote_config.update(
            {
                "api_base_url": "https://relay.example/v1",
                "api_model": "remote-model",
            }
        )
        assert client.put("/api/config", json=remote_config).status_code == 200
        response = client.put(
            "/api/local-models/configuration",
            json={
                "mode": "external",
                "base_url": "http://127.0.0.1:18080/v1/",
                "model": "local-qwen",
                "make_default": True,
            },
        )
        provider = client.get("/api/local-models/provider")
        status = client.get("/api/local-models/status")
        config = client.get("/api/config")
        restored = client.post("/api/local-models/deactivate")
        restored_config = client.get("/api/config")

    assert response.status_code == 200
    assert provider.json()["base_url"] == "http://127.0.0.1:18080/v1"
    assert status.json()["state"] == "ready"
    assert config.json()["api_base_url"] == "http://127.0.0.1:18080/v1"
    assert config.json()["api_model"] == "local-qwen"
    assert config.json()["request_timeout_seconds"] == 600
    assert restored.status_code == 200
    assert restored_config.json()["api_base_url"] == "https://relay.example/v1"
    assert restored_config.json()["api_model"] == "remote-model"
    assert manager.remote_profile() is None
    assert manager.local_active_provider() is None


def test_switching_between_local_endpoints_preserves_the_original_remote_profile(
    tmp_path,
):
    def handler(request: httpx.Request) -> httpx.Response:
        model = "local-one" if request.url.port == 18081 else "local-two"
        return httpx.Response(200, json={"data": [{"id": model}]})

    manager = LocalModelManager(
        tmp_path / "managed",
        transport=httpx.MockTransport(handler),
        hardware_detector=lambda: hardware(),
    )
    app = create_app(tmp_path / "data", local_model_manager=manager)
    with TestClient(app) as client:
        project_id = client.get("/api/workspace").json()["project"]["id"]
        remote = client.get(f"/api/projects/{project_id}/translation-profile").json()
        remote.update(
            {
                "base_url": "https://relay.example/v1",
                "api_key": "remote-secret",
                "model": "remote-model",
            }
        )
        assert (
            client.put(
                f"/api/projects/{project_id}/translation-profile", json=remote
            ).status_code
            == 200
        )
        for port, model in ((18081, "local-one"), (18082, "local-two")):
            response = client.put(
                "/api/local-models/configuration",
                json={
                    "mode": "external",
                    "base_url": f"http://127.0.0.1:{port}/v1",
                    "model": model,
                    "make_default": True,
                },
            )
            assert response.status_code == 200

        assert client.post("/api/local-models/deactivate").status_code == 200
        restored = client.get(
            f"/api/projects/{project_id}/translation-profile"
        ).json()

    assert restored["base_url"] == "https://relay.example/v1"
    assert restored["model"] == "remote-model"
    assert restored["api_key"] == "remote-secret"
    assert manager.remote_profile() is None
    assert manager.local_active_provider() is None


def test_same_base_url_with_a_different_model_still_preserves_remote_credentials(
    tmp_path,
):
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": [{"id": "local-model"}]})

    manager = LocalModelManager(
        tmp_path / "managed",
        transport=httpx.MockTransport(handler),
        hardware_detector=lambda: hardware(),
    )
    app = create_app(tmp_path / "data", local_model_manager=manager)
    with TestClient(app) as client:
        project_id = client.get("/api/workspace").json()["project"]["id"]
        profile = client.get(
            f"/api/projects/{project_id}/translation-profile"
        ).json()
        profile.update(
            {
                "base_url": "http://127.0.0.1:18081/v1",
                "api_key": "remote-secret",
                "model": "remote-model",
            }
        )
        client.put(f"/api/projects/{project_id}/translation-profile", json=profile)
        activated = client.put(
            "/api/local-models/configuration",
            json={
                "mode": "external",
                "base_url": "http://127.0.0.1:18081/v1",
                "model": "local-model",
                "make_default": True,
            },
        )
        restored_response = client.post("/api/local-models/deactivate")
        restored = client.get(
            f"/api/projects/{project_id}/translation-profile"
        ).json()

    assert activated.status_code == 200
    assert restored_response.status_code == 200
    assert restored["model"] == "remote-model"
    assert restored["api_key"] == "remote-secret"


def test_startup_migrates_an_existing_active_local_profile_before_switching(
    tmp_path,
):
    def handler(request: httpx.Request) -> httpx.Response:
        model = "local-one" if request.url.port == 18081 else "local-two"
        return httpx.Response(200, json={"data": [{"id": model}]})

    manager = LocalModelManager(
        tmp_path / "managed",
        transport=httpx.MockTransport(handler),
        hardware_detector=lambda: hardware(),
    )
    manager.configure(
        {
            "mode": "external",
            "base_url": "http://127.0.0.1:18081/v1",
            "model": "local-one",
        }
    )
    manager.save_remote_profile(
        {
            "api_base_url": "https://relay.example/v1",
            "api_model": "remote-model",
            "api_path": "/chat/completions",
        }
    )
    app = create_app(tmp_path / "data", local_model_manager=manager)
    current = app.state.storage.get_config()
    app.state.storage.set_config(
        current.model_copy(
            update={
                "api_base_url": "http://127.0.0.1:18081/v1",
                "api_model": "local-one",
            }
        )
    )
    with TestClient(app) as client:
        assert manager.local_active_provider()["model"] == "local-one"
        switched = client.put(
            "/api/local-models/configuration",
            json={
                "mode": "external",
                "base_url": "http://127.0.0.1:18082/v1",
                "model": "local-two",
                "make_default": True,
            },
        )
        restored = client.post("/api/local-models/deactivate")
        restored_config = client.get("/api/config").json()

    assert switched.status_code == 200
    assert restored.status_code == 200
    assert restored_config["api_base_url"] == "https://relay.example/v1"
    assert restored_config["api_model"] == "remote-model"


def test_external_configuration_is_probed_before_replacing_the_current_config(
    tmp_path,
):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.port == 18081:
            return httpx.Response(200, json={"data": [{"id": "current-model"}]})
        return httpx.Response(200, json={"data": [{"id": "different-model"}]})

    manager = LocalModelManager(
        tmp_path / "managed",
        transport=httpx.MockTransport(handler),
        hardware_detector=lambda: hardware(),
    )
    manager.configure(
        {
            "mode": "external",
            "base_url": "http://127.0.0.1:18081/v1",
            "model": "current-model",
        }
    )
    app = create_app(tmp_path / "data", local_model_manager=manager)
    with TestClient(app) as client:
        response = client.put(
            "/api/local-models/configuration",
            json={
                "mode": "external",
                "base_url": "http://127.0.0.1:18082/v1",
                "model": "missing-model",
            },
        )
        provider = client.get("/api/local-models/provider").json()

    assert response.status_code == 422
    assert provider["base_url"] == "http://127.0.0.1:18081/v1"
    assert provider["model"] == "current-model"


def test_managed_start_failure_rolls_back_the_previous_configuration(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": [{"id": "remote-model"}]})

    def fail_to_start(*_args, **_kwargs):
        raise OSError("process failed to start")

    manager = LocalModelManager(
        tmp_path / "managed",
        transport=httpx.MockTransport(handler),
        hardware_detector=lambda: hardware(),
        process_factory=fail_to_start,
    )
    manager.configure(
        {
            "mode": "external",
            "base_url": "http://127.0.0.1:18081/v1",
            "model": "remote-model",
        }
    )
    executable = tmp_path / "llama-server.exe"
    model = tmp_path / "candidate.gguf"
    executable.write_bytes(b"fixture")
    model.write_bytes(b"GGUFfixture")
    app = create_app(tmp_path / "data", local_model_manager=manager)
    with TestClient(app) as client:
        app_config = client.get("/api/config").json()
        app_config.update(
            {
                "api_base_url": "http://127.0.0.1:18081/v1",
                "api_model": "remote-model",
            }
        )
        assert client.put("/api/config", json=app_config).status_code == 200
        configured = client.put(
            "/api/local-models/configuration",
            json={
                "mode": "managed",
                "executable_path": str(executable),
                "model_path": str(model),
                "model": "candidate",
                "runtime_variant": "cpu",
            },
        )
        started = client.post("/api/local-models/start")
        provider = client.get("/api/local-models/provider").json()
        restored_app_config = client.get("/api/config").json()

    assert configured.status_code == 200
    assert started.status_code == 409
    assert provider["base_url"] == "http://127.0.0.1:18081/v1"
    assert provider["model"] == "remote-model"
    assert restored_app_config["api_base_url"] == "http://127.0.0.1:18081/v1"
    assert restored_app_config["api_model"] == "remote-model"


def test_managed_process_failure_during_polling_returns_failure_then_rolls_back(
    tmp_path,
):
    class Process:
        exited = False

        def poll(self):
            return 1 if self.exited else None

        def terminate(self):
            self.exited = True

        def wait(self, timeout=None):
            return 1

        def kill(self):
            self.exited = True

    process = Process()

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.port == 18081:
            return httpx.Response(200, json={"data": [{"id": "remote-model"}]})
        if request.url.path == "/health":
            return httpx.Response(503, json={"status": "loading"})
        return httpx.Response(200, json={"data": [{"id": "candidate"}]})

    manager = LocalModelManager(
        tmp_path / "managed",
        transport=httpx.MockTransport(handler),
        hardware_detector=lambda: hardware(),
        process_factory=lambda *_args, **_kwargs: process,
    )
    manager.configure(
        {
            "mode": "external",
            "base_url": "http://127.0.0.1:18081/v1",
            "model": "remote-model",
        }
    )
    executable = tmp_path / "llama-server.exe"
    model = tmp_path / "candidate.gguf"
    executable.write_bytes(b"fixture")
    model.write_bytes(b"GGUFfixture")
    app = create_app(tmp_path / "data", local_model_manager=manager)
    with TestClient(app) as client:
        configured = client.put(
            "/api/local-models/configuration",
            json={
                "mode": "managed",
                "executable_path": str(executable),
                "model_path": str(model),
                "model": "candidate",
                "runtime_variant": "cpu",
                "make_default": False,
            },
        )
        started = client.post("/api/local-models/start")
        process.exited = True
        failed = client.get("/api/local-models/status")
        provider = client.get("/api/local-models/provider").json()

    assert configured.status_code == 200
    assert started.json()["state"] == "starting"
    assert failed.json()["state"] == "failed"
    assert failed.json()["configuration"]["model"] == "candidate"
    assert provider["base_url"] == "http://127.0.0.1:18081/v1"
    assert provider["model"] == "remote-model"


def test_managed_probe_requires_a_process_owned_by_the_manager(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "ok"})
        return httpx.Response(200, json={"data": [{"id": "candidate"}]})

    executable = tmp_path / "llama-server.exe"
    model = tmp_path / "candidate.gguf"
    executable.write_bytes(b"fixture")
    model.write_bytes(b"GGUFfixture")
    manager = LocalModelManager(
        tmp_path / "managed",
        transport=httpx.MockTransport(handler),
        hardware_detector=lambda: hardware(),
    )
    manager.configure(
        {
            "mode": "managed",
            "executable_path": str(executable),
            "model_path": str(model),
            "model": "candidate",
            "runtime_variant": "cpu",
        }
    )

    status = manager.status(include_hardware=False)

    assert status["ready"] is False
    assert status["state"] == "stopped"


def test_active_deployment_rejects_duplicate_and_configuration_mutations(
    tmp_path, monkeypatch
):
    entered = Event()
    release = Event()
    manager = LocalModelManager(
        tmp_path / "managed",
        hardware_detector=lambda: hardware(build_profile="cpu"),
    )

    def blocked_release():
        entered.set()
        assert release.wait(2)
        raise RuntimeError("deployment released")

    monkeypatch.setattr(manager, "_release", blocked_release)
    app = create_app(tmp_path / "data", local_model_manager=manager)
    with TestClient(app) as client:
        submitted = client.post("/api/local-models/deploy-jobs", json={})
        assert submitted.status_code == 200
        assert entered.wait(1)

        duplicate = client.post("/api/local-models/deploy-jobs", json={})
        configuration = client.put(
            "/api/local-models/configuration",
            json={
                "mode": "external",
                "base_url": "http://127.0.0.1:18081/v1",
                "model": "other",
            },
        )
        stopped = client.post("/api/local-models/stop")
        release.set()
        wait_for_job(client, submitted.json()["id"])

    assert duplicate.status_code == 409
    assert configuration.status_code == 409
    assert stopped.status_code == 409


def test_app_shutdown_cancels_jobs_before_closing_the_local_model(tmp_path, monkeypatch):
    started = Event()
    finished = Event()
    closed = Event()
    manager = LocalModelManager(
        tmp_path / "managed", hardware_detector=lambda: hardware()
    )

    def blocked_deploy(_values, _progress, cancel):
        started.set()
        assert cancel.wait(2)
        finished.set()
        raise InterruptedError("job cancelled")

    def close_after_job():
        assert finished.is_set()
        closed.set()

    monkeypatch.setattr(manager, "deploy", blocked_deploy)
    monkeypatch.setattr(manager, "close", close_after_job)
    app = create_app(tmp_path / "data", local_model_manager=manager)
    with TestClient(app) as client:
        submitted = client.post("/api/local-models/deploy-jobs", json={})
        assert submitted.status_code == 200
        assert started.wait(1)

    assert closed.is_set()
    assert app.state.jobs.get(submitted.json()["id"])["status"] == "cancelled"


def test_deploy_endpoint_uses_background_job_and_activates_provider(
    tmp_path, monkeypatch
):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": [{"id": "deployed-model"}]})

    manager = LocalModelManager(
        tmp_path / "managed",
        transport=httpx.MockTransport(handler),
        hardware_detector=lambda: hardware(build_profile="cpu"),
    )

    def fake_deploy(values, progress, _cancel):
        assert values["runtime_variant"] == "auto"
        progress(0.5, "Downloading translation model", {"downloaded_bytes": 50})
        manager.configure(
            {
                "mode": "external",
                "base_url": "http://127.0.0.1:18081/v1",
                "model": "deployed-model",
            }
        )
        return {"model_id": "deployed-model"}

    monkeypatch.setattr(manager, "deploy", fake_deploy)
    app = create_app(tmp_path / "data", local_model_manager=manager)
    with TestClient(app) as client:
        submitted = client.post("/api/local-models/deploy-jobs", json={})
        assert submitted.status_code == 200
        completed = wait_for_job(client, submitted.json()["id"])
        config = client.get("/api/config").json()

    assert completed["status"] == "completed"
    assert completed["result"]["model_id"] == "deployed-model"
    assert config["api_model"] == "deployed-model"
    assert config["api_base_url"] == "http://127.0.0.1:18081/v1"
