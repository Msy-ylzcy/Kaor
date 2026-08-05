from backend.subtitle_import import parse_srt


def test_import_preserves_staggered_overlapping_subtitles():
    cues = parse_srt(
        """1
00:00:01,000 --> 00:00:05,000
First line

2
00:00:02,800 --> 00:00:05,000
Second line
"""
    )

    assert [(cue.start_ms, cue.end_ms) for cue in cues] == [
        (1000, 5000),
        (2800, 5000),
    ]
    assert cues[0].group_id == cues[1].group_id
    assert [cue.layer for cue in cues] == [0, 1]
    assert all(cue.source_kind == "imported" for cue in cues)
