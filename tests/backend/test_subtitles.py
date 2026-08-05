from backend.subtitles import SubtitleEvent, build_ass


def test_ass_keeps_overlapping_events_on_stable_layers():
    content = build_ass(
        [
            SubtitleEvent(1000, 5000, "First", "SPK_01", "#F4D35E", 0),
            SubtitleEvent(2800, 5000, "Second", "SPK_02", "#76C7C0", 1),
        ],
        width=1920,
        height=1080,
    )

    assert "Dialogue: 0,0:00:01.00,0:00:05.00" in content
    assert "Dialogue: 1,0:00:02.80,0:00:05.00" in content
    assert "&H005ED3F4" in content
    assert "&H00C0C776" in content


def test_ass_uses_custom_target_region_and_preserves_line_breaks():
    content = build_ass(
        [SubtitleEvent(0, 1000, r"Line one\NLine two")],
        width=1000,
        height=500,
        target_region=(0.1, 0.1, 0.8, 0.2),
    )

    assert r"{\an8\pos(500,76)}Line one\NLine two" in content


def test_ass_fits_concurrent_layers_inside_small_target_region():
    content = build_ass(
        [
            SubtitleEvent(0, 1000, "First", "SPK_01", "#F4D35E", 0),
            SubtitleEvent(0, 1000, "Second", "SPK_02", "#76C7C0", 1),
        ],
        width=640,
        height=360,
        font_size=48,
        target_region=(0.08, 0.08, 0.84, 0.22),
    )

    assert "Noto Sans SC,33," in content
    assert r"{\an8\pos(320,29)}First" in content
    assert r"{\an8\pos(320,74)}Second" in content
