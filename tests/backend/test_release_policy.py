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
    assert "Download and verify the bundled BS-Roformer configuration" in workflow


def test_portable_build_packages_uvr_config_and_marks_checkpoint_on_demand():
    build_script = (ROOT / "scripts" / "build-portable.ps1").read_text(
        encoding="utf-8"
    )

    assert "Copy-Item -LiteralPath $uvrConfigSource" in build_script
    assert "Copy-Item -LiteralPath $uvrModelSource" not in build_script
    assert '$onDemandModels = @("uvr-bs-roformer"' in build_script
    assert '"models\\uvr\\$uvrModelName",' not in build_script
    assert "must be downloaded on first use, not redistributed" in build_script
