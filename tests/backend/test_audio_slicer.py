from __future__ import annotations

import math
import sys
import types
import wave
from contextlib import nullcontext
from pathlib import Path
from threading import Event

import numpy as np
import pytest

from backend import audio_process, speech_pipeline
from backend.audio_pipeline import AudioPipelineError, recommended_asr_model
from backend.audio_slicer import (
    AudioSlice,
    AudioSliceManifest,
    Slicer,
    SlicerSettings,
    load_slice_manifest,
    slice_audio,
)
from backend.models import Cue


def write_pcm16(path: Path, samples: np.ndarray, sample_rate: int = 16_000) -> None:
    values = np.rint(np.clip(samples, -1.0, 1.0) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(values.tobytes())


def tone(duration_seconds: float, sample_rate: int = 16_000) -> np.ndarray:
    positions = np.arange(round(duration_seconds * sample_rate), dtype=np.float32)
    return 0.2 * np.sin(2 * math.pi * 220 * positions / sample_rate)


def test_slicer2_writes_absolute_manifest_around_middle_silence(tmp_path: Path):
    source = tmp_path / "speech.wav"
    samples = np.concatenate(
        (tone(1.0), np.zeros(8_000, dtype=np.float32), tone(1.0))
    )
    write_pcm16(source, samples)
    messages: list[str] = []

    manifest = slice_audio(
        source,
        tmp_path / "slices",
        threshold_db=-40,
        min_length_ms=500,
        min_interval_ms=200,
        hop_size_ms=10,
        max_sil_kept_ms=50,
        progress=lambda _value, message: messages.append(message),
    )

    assert len(manifest.slices) == 2
    first, second = manifest.slices
    assert first.start_ms == 0
    assert 900 <= first.end_ms <= 1_100
    assert 1_350 <= second.start_ms <= 1_550
    assert second.end_ms == 2_500
    assert all(item.path.is_file() for item in manifest.slices)
    assert manifest.settings.min_interval_ms == 200
    loaded = load_slice_manifest(manifest.manifest_path)
    assert loaded.to_dict() == manifest.to_dict()
    assert messages[-1] == "Created 2 speech slices"


def test_slicer_hard_caps_non_silent_ranges_at_thirty_seconds():
    sample_rate = 1_000
    waveform = np.ones((1, 65 * sample_rate), dtype=np.float32) * 0.1
    settings = SlicerSettings(
        min_length_ms=5_000,
        min_interval_ms=200,
        hop_size_ms=10,
        max_sil_kept_ms=500,
        max_length_ms=30_000,
    )

    ranges = Slicer(sample_rate, settings).slice_ranges(waveform)

    assert len(ranges) == 3
    assert ranges[0][0] == 0
    assert ranges[-1][1] == waveform.shape[1]
    assert all(end - start <= 30 * sample_rate for start, end in ranges)
    assert all(following[0] == current[1] for current, following in zip(ranges, ranges[1:]))


def _install_fake_nemo(monkeypatch: pytest.MonkeyPatch, outputs_by_name: dict[str, object]):
    calls: list[tuple[list[str], int, bool]] = []

    class FakeModel:
        cfg = types.SimpleNamespace(decoding={})

        def to(self, _device):
            return self

        def eval(self):
            return self

        def transcribe(self, *, audio, batch_size, timestamps):
            calls.append((list(audio), batch_size, timestamps))
            return [outputs_by_name[Path(path).stem] for path in audio]

    class FakeASRModel:
        @staticmethod
        def restore_from(**_kwargs):
            return FakeModel()

    nemo = types.ModuleType("nemo")
    collections = types.ModuleType("nemo.collections")
    asr = types.ModuleType("nemo.collections.asr")
    asr.models = types.SimpleNamespace(ASRModel=FakeASRModel)
    nemo.collections = collections
    collections.asr = asr
    torch = types.ModuleType("torch")
    torch.device = lambda value: value
    torch.inference_mode = nullcontext
    monkeypatch.setitem(sys.modules, "nemo", nemo)
    monkeypatch.setitem(sys.modules, "nemo.collections", collections)
    monkeypatch.setitem(sys.modules, "nemo.collections.asr", asr)
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setattr(speech_pipeline, "_enable_nemo_confidence", lambda _model: True)
    return calls


def test_nemo_batches_slice_paths_and_offsets_real_hypothesis_timestamps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "fixture.nemo").write_bytes(b"checkpoint")
    slices: list[AudioSlice] = []
    hypotheses: dict[str, object] = {}
    for index, start_ms in enumerate((1_000, 5_000, 9_000), start=1):
        path = tmp_path / f"part{index}.wav"
        path.write_bytes(b"wav")
        slices.append(
            AudioSlice(index=index, path=path, start_ms=start_ms, end_ms=start_ms + 2_000)
        )
        hypotheses[path.stem] = types.SimpleNamespace(
            text=f"line {index}",
            timestep={
                "segment": [
                    {"segment": f"line {index}", "start": 0.2, "end": 1.2}
                ],
                "word": [
                    {
                        "word": "line",
                        "start": 0.2,
                        "end": 0.7,
                        "confidence": 0.9,
                    }
                ],
            },
            word_confidence=[0.9],
        )
    calls = _install_fake_nemo(monkeypatch, hypotheses)
    messages: list[str] = []

    cues = speech_pipeline._recognize_nemo_slices(
        slices,
        model_dir,
        "cuda:0",
        lambda _value, message: messages.append(message),
        batch_size=2,
    )

    assert [len(call[0]) for call in calls] == [2, 1]
    assert [call[1] for call in calls] == [2, 1]
    assert all(call[2] is True for call in calls)
    assert [(cue.start_ms, cue.end_ms) for cue in cues] == [
        (1_200, 2_200),
        (5_200, 6_200),
        (9_200, 10_200),
    ]
    assert [cue.cue_id for cue in cues] == ["ASR000001", "ASR000002", "ASR000003"]
    assert any("3/3" in message for message in messages)


def test_nemo_zero_cues_is_a_failed_asr_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "fixture.nemo").write_bytes(b"checkpoint")
    path = tmp_path / "silent.wav"
    path.write_bytes(b"wav")
    _install_fake_nemo(
        monkeypatch,
        {path.stem: types.SimpleNamespace(text="", timestep={})},
    )

    with pytest.raises(AudioPipelineError, match="0 speech cues"):
        speech_pipeline._recognize_nemo_slices(
            [AudioSlice(index=1, path=path, start_ms=0, end_ms=1_000)],
            model_dir,
            "cuda:0",
            None,
            batch_size=1,
        )


def test_nemo_resume_starts_at_next_unfinished_slice(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "fixture.nemo").write_bytes(b"checkpoint")
    slices: list[AudioSlice] = []
    hypotheses: dict[str, object] = {}
    for index, start_ms in enumerate((0, 2_000, 4_000), start=1):
        path = tmp_path / f"resume-{index}.wav"
        path.write_bytes(b"wav")
        slices.append(
            AudioSlice(index=index, path=path, start_ms=start_ms, end_ms=start_ms + 1_500)
        )
        hypotheses[path.stem] = types.SimpleNamespace(
            text=f"line {index}",
            timestep={"segment": [{"segment": f"line {index}", "start": 0, "end": 1}]},
        )
    calls = _install_fake_nemo(monkeypatch, hypotheses)
    initial = [
        Cue(
            cue_id=f"ASR{index:06d}",
            start_ms=(index - 1) * 2_000,
            end_ms=(index - 1) * 2_000 + 1_000,
            source_kind="speech",
            source_text=f"line {index}",
        )
        for index in (1, 2)
    ]
    checkpoints: list[tuple[int, list[Cue]]] = []

    cues = speech_pipeline._recognize_nemo_slices(
        slices,
        model_dir,
        "cuda:0",
        None,
        batch_size=2,
        start_slice=2,
        initial_cues=initial,
        checkpoint=lambda index, rows: checkpoints.append((index, rows)),
    )

    assert [[Path(path).name for path in call[0]] for call in calls] == [["resume-3.wav"]]
    assert [cue.source_text for cue in cues] == ["line 1", "line 2", "line 3"]
    assert checkpoints[-1][0] == 3


def test_one_click_audio_pipeline_orders_isolated_uvr_slice_and_asr_stages(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    events: list[str] = []
    mix = tmp_path / "mix.wav"
    vocals = tmp_path / "vocals.wav"
    asr_audio = tmp_path / "speech.wav"
    for path in (mix, vocals):
        path.write_bytes(b"W" * 96)
    model = recommended_asr_model("ja")
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    manifest_path = tmp_path / "slices" / "slices.json"
    manifest = AudioSliceManifest(
        manifest_path=manifest_path,
        source_path=asr_audio,
        sample_rate=16_000,
        channels=1,
        duration_ms=1_000,
        settings=SlicerSettings(),
        slices=(),
    )
    cue = Cue(
        cue_id="ASR000001",
        start_ms=0,
        end_ms=1_000,
        track_id="speech",
        source_kind="speech",
        source_text="fixture",
    )

    def fake_uvr(*_args, **_kwargs):
        events.append("uvr-process")
        return vocals

    def fake_transcode(_source, destination):
        events.append("transcode")
        Path(destination).write_bytes(b"A" * 96)
        return Path(destination)

    def fake_slice(*_args, **_kwargs):
        events.append("slicer")
        return manifest

    def fake_asr(*_args, **_kwargs):
        events.append("asr-process")
        return [cue]

    monkeypatch.setattr(audio_process, "run_uvr_worker", fake_uvr)
    monkeypatch.setattr(audio_process, "transcode_audio_for_asr", fake_transcode)
    monkeypatch.setattr(audio_process, "run_slicer_stage", fake_slice)
    monkeypatch.setattr(audio_process, "run_asr_worker", fake_asr)

    result = audio_process.run_speech_worker(
        mix,
        tmp_path / "work",
        asr_audio,
        model,
        model_dir,
        device="cuda:0",
        separate_vocals=True,
        progress=lambda _value, _message: None,
        cancel_event=Event(),
    )

    assert result == [cue]
    assert events == ["uvr-process", "transcode", "slicer", "asr-process"]


def test_stage_wrappers_dispatch_uvr_and_asr_as_distinct_worker_commands(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    vocals = tmp_path / "vocals.wav"
    vocals.write_bytes(b"V" * 96)
    audio = tmp_path / "speech.wav"
    audio.write_bytes(b"A" * 96)
    manifest = tmp_path / "slices.json"
    manifest.write_text("{}", encoding="utf-8")
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    model = recommended_asr_model("ja")
    cue = Cue(
        cue_id="ASR000001",
        start_ms=0,
        end_ms=1_000,
        track_id="speech",
        source_kind="speech",
        source_text="fixture",
    )
    observed: list[tuple[str, dict[str, object]]] = []

    def fake_stage(stage, request, _output_dir, **_kwargs):
        observed.append((stage, request))
        if stage == "uvr":
            return {"status": "ok", "vocals_path": str(vocals)}
        return {"status": "ok", "cues": [cue.model_dump(mode="json")]}

    monkeypatch.setattr(audio_process, "_run_stage_worker", fake_stage)
    event = Event()
    assert audio_process.run_uvr_worker(
        audio,
        tmp_path,
        device="cuda:0",
        progress=lambda _value, _message: None,
        cancel_event=event,
    ) == vocals.resolve()
    assert audio_process.run_asr_worker(
        audio,
        manifest,
        tmp_path,
        model,
        model_dir,
        device="cuda:0",
        batch_size=3,
        progress=lambda _value, _message: None,
        cancel_event=event,
    ) == [cue]

    assert [stage for stage, _request in observed] == ["uvr", "asr"]
    assert observed[1][1]["batch_size"] == 3
    assert observed[1][1]["slice_manifest"] == str(manifest.resolve())
