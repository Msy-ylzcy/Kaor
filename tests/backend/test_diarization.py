from pathlib import Path

from backend.diarization import (
    DiarizationSegment,
    assign_speakers,
    default_model_dir,
    diarize_cues,
    parse_rttm,
)
from backend.models import Cue


def cue(cue_id: str, start_ms: int, end_ms: int, **updates: object) -> Cue:
    values: dict[str, object] = {
        "cue_id": cue_id,
        "start_ms": start_ms,
        "end_ms": end_ms,
        "source_kind": "speech",
        "source_text": cue_id,
    }
    values.update(updates)
    return Cue(**values)


def test_default_model_dir_uses_portable_application_root(
    tmp_path: Path, monkeypatch,
):
    monkeypatch.setattr("backend.diarization.application_root", lambda: tmp_path)

    assert default_model_dir() == tmp_path / "models" / "diarization"


def test_parse_rttm_reads_speaker_intervals(tmp_path: Path):
    path = tmp_path / "sample.rttm"
    path.write_text(
        "# ignored\n"
        "SPEAKER sample 1 0.460 0.590 <NA> <NA> speaker_0 <NA> <NA>\n"
        "SPEAKER sample 1 1.260 1.390 <NA> <NA> speaker_1 <NA> <NA>\n",
        encoding="utf-8",
    )

    assert parse_rttm(path) == [
        DiarizationSegment("sample", 460, 1050, "speaker_0"),
        DiarizationSegment("sample", 1260, 2650, "speaker_1"),
    ]


def test_assign_speakers_uses_greatest_total_overlap_and_stable_order():
    segments = [
        DiarizationSegment("sample", 0, 600, "raw_b"),
        DiarizationSegment("sample", 600, 1000, "raw_a"),
        DiarizationSegment("sample", 1000, 1600, "raw_a"),
        DiarizationSegment("sample", 1400, 1900, "raw_b"),
    ]
    output, stats = assign_speakers(
        [cue("one", 100, 900), cue("two", 900, 1800), cue("none", 2200, 2500)],
        segments,
    )

    assert output[0].speaker_id == "SPK_01"
    assert output[0].speaker_name == "Speaker 1"
    assert output[0].speaker_color == "#F4D35E"
    assert output[1].speaker_id == "SPK_02"
    assert output[1].speaker_color == "#76C7C0"
    assert output[2].speaker_id == ""
    assert stats.assigned_cues == 2
    assert stats.unmatched_cues == 1
    assert stats.speaker_count == 2


def test_assign_speakers_preserves_existing_speaker_fields():
    original = cue(
        "existing",
        0,
        1000,
        speaker_id="CAST_01",
        speaker_name="Existing",
        speaker_color="#123456",
    )
    output, stats = assign_speakers(
        [original], [DiarizationSegment("sample", 0, 1000, "speaker_0")]
    )

    assert output == [original]
    assert stats.preserved_cues == 1
    assert stats.assigned_cues == 0


def test_diarize_cues_reports_missing_models_without_changing_cues(tmp_path: Path):
    original = cue("one", 0, 1000)
    result = diarize_cues(
        tmp_path / "missing.wav",
        [original],
        tmp_path / "output",
        model_dir=tmp_path / "models",
    )

    assert result.success is False
    assert result.cues == (original,)
    assert "audio file not found" in result.error
