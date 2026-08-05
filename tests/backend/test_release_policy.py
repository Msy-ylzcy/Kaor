from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_release_workflow_does_not_redistribute_bs_roformer_checkpoint():
    workflow = (ROOT / ".github" / "workflows" / "release.yml").read_text(
        encoding="utf-8"
    )

    assert "model_repo/releases/download/all_public_uvr_models" not in workflow
    assert "KAOR_BS_ROFORMER_REDISTRIBUTION_CONFIRMED" not in workflow
    assert "binary-license-gate" not in workflow
    assert "Download and verify the bundled BS-Roformer configuration" not in workflow
    assert "model_bs_roformer_ep_317_sdr_12.9755.yaml" not in workflow


def test_portable_build_keeps_all_uvr_model_assets_on_demand():
    build_script = (ROOT / "scripts" / "build-portable.ps1").read_text(
        encoding="utf-8"
    )

    assert "Copy-Item -LiteralPath $uvrConfigSource" not in build_script
    assert "Copy-Item -LiteralPath $uvrModelSource" not in build_script
    assert '$onDemandModels = @("uvr-bs-roformer"' in build_script
    assert '"models\\uvr\\$uvrModelName",' not in build_script
    assert 'foreach ($assetName in @($uvrModelName, $uvrConfigName))' in build_script
    assert "assets must be downloaded on first use, not redistributed" in build_script


def test_nvidia_build_installs_paddle_wheel_without_conflicting_metadata():
    build_script = (ROOT / "scripts" / "build-portable.ps1").read_text(
        encoding="utf-8"
    )
    requirements = (ROOT / "requirements-nvidia-cu126.txt").read_text(
        encoding="utf-8"
    )

    assert '"--no-deps",' in build_script
    assert "$paddleGpuWheel" in build_script
    assert "paddlepaddle-gpu @" not in requirements
    assert "nvidia-cudnn-cu12==9.9.0.52" in requirements
