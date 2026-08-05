from backend.recognition import (
    Box,
    FrameDetections,
    TemporalTextTracker,
    TextDetection,
)


def detection(text, y, confidence=0.95, color=(255, 210, 80)):
    return TextDetection(text, Box(0.15, y, 0.7, 0.08), confidence, color)


def test_second_subtitle_can_appear_while_first_track_continues():
    tracker = TemporalTextTracker(max_missed_frames=0)
    tracker.update(FrameDetections(1000, (detection("First line", 0.72),)))
    tracker.update(
        FrameDetections(
            2800,
            (
                detection("First line", 0.72),
                detection("Second line", 0.82, color=(90, 210, 230)),
            ),
        )
    )
    tracker.update(
        FrameDetections(
            5000,
            (
                detection("First line", 0.72),
                detection("Second line", 0.82, color=(90, 210, 230)),
            ),
        )
    )

    cues = tracker.finalize(end_ms=5100)

    assert [(cue.text, cue.start_ms, cue.end_ms) for cue in cues] == [
        ("First line", 1000, 5100),
        ("Second line", 2800, 5100),
    ]
    assert cues[0].group_id == cues[1].group_id
    assert [cue.layer for cue in cues] == [0, 1]


def test_progressive_text_is_one_track_and_uses_longest_consensus():
    tracker = TemporalTextTracker(max_missed_frames=0)
    tracker.update(FrameDetections(0, (detection("Hel", 0.75),)))
    tracker.update(FrameDetections(200, (detection("Hello", 0.75),)))
    tracker.update(FrameDetections(400, (detection("Hello!", 0.75),)))

    cues = tracker.finalize(end_ms=600)

    assert len(cues) == 1
    assert cues[0].text == "Hello!"
    assert cues[0].start_ms == 0
    assert cues[0].end_ms == 600


def test_low_confidence_or_single_frame_cues_require_review():
    tracker = TemporalTextTracker(max_missed_frames=0, filter_noise=False)
    tracker.update(FrameDetections(100, (detection("unclear", 0.7, confidence=0.51),)))
    tracker.update(FrameDetections(200, ()))

    cue = tracker.finalize()[0]

    assert cue.review_required is True


def test_noise_filter_drops_weak_single_observation_tracks():
    tracker = TemporalTextTracker(max_missed_frames=0)
    tracker.update(
        FrameDetections(
            100,
            (
                detection("4C", 0.6, confidence=0.45),
                detection("_", 0.8, confidence=0.99),
            ),
        )
    )
    tracker.update(FrameDetections(300, ()))

    assert tracker.finalize() == []


def test_noise_filter_keeps_high_confidence_flash_dialogue():
    tracker = TemporalTextTracker(max_missed_frames=0)
    tracker.update(FrameDetections(100, (detection("一瞬", 0.7, confidence=0.99),)))
    tracker.update(FrameDetections(300, ()))

    cues = tracker.finalize()

    assert [(cue.text, cue.start_ms, cue.end_ms) for cue in cues] == [
        ("一瞬", 100, 101)
    ]


def test_noise_filter_keeps_200ms_short_dialogue():
    tracker = TemporalTextTracker(max_missed_frames=0)
    tracker.update(FrameDetections(100, (detection("は？", 0.7, confidence=0.9995),)))
    tracker.update(FrameDetections(300, (detection("は？", 0.7, confidence=0.9995),)))
    tracker.update(FrameDetections(500, ()))

    cues = tracker.finalize()

    assert [(cue.text, cue.start_ms, cue.end_ms) for cue in cues] == [
        ("は？", 100, 300)
    ]


def test_noise_filter_drops_persistent_low_confidence_ascii_counter():
    tracker = TemporalTextTracker(max_missed_frames=0)
    tracker.update(FrameDetections(100, (detection("00", 0.7, confidence=0.66),)))
    tracker.update(FrameDetections(300, (detection("00", 0.7, confidence=0.67),)))
    tracker.update(FrameDetections(500, ()))

    assert tracker.finalize() == []


def test_noise_filter_drops_longer_low_confidence_ascii_garbage():
    tracker = TemporalTextTracker(max_missed_frames=0)
    tracker.update(FrameDetections(100, (detection("00PPP80S", 0.7, confidence=0.46),)))
    tracker.update(FrameDetections(300, (detection("00PPP80S", 0.7, confidence=0.47),)))

    assert tracker.finalize() == []


def test_noise_filter_keeps_persistent_short_ascii_dialogue_when_clear():
    tracker = TemporalTextTracker(max_missed_frames=0)
    tracker.update(FrameDetections(100, (detection("OK", 0.7, confidence=0.96),)))
    tracker.update(FrameDetections(300, (detection("OK", 0.7, confidence=0.97),)))
    tracker.update(FrameDetections(500, ()))

    assert [cue.text for cue in tracker.finalize()] == ["OK"]


def test_noise_filter_drops_unclear_persistent_single_cjk_fragment():
    tracker = TemporalTextTracker(max_missed_frames=0)
    tracker.update(FrameDetections(100, (detection("ツ", 0.7, confidence=0.90),)))
    tracker.update(FrameDetections(300, (detection("ツ", 0.7, confidence=0.91),)))

    assert tracker.finalize() == []


def test_noise_filter_drops_high_confidence_transition_fragment():
    tracker = TemporalTextTracker(max_missed_frames=0)
    tracker.update(
        FrameDetections(
            100,
            (detection("今日は連れてきてくれてありがとう、魔理沙", 0.7, confidence=0.99),),
        )
    )
    tracker.update(
        FrameDetections(
            300,
            (
                detection("今日は連れてきてくれてありがとう、魔理沙", 0.7, confidence=0.99),
                detection("がとう、魔理沙", 0.5, confidence=0.995),
            ),
        )
    )
    tracker.update(FrameDetections(500, ()))

    cues = tracker.finalize()

    assert [cue.text for cue in cues] == ["今日は連れてきてくれてありがとう、魔理沙"]


def test_noise_filter_deduplicates_same_text_tracks_in_same_region():
    tracker = TemporalTextTracker(max_missed_frames=0)
    duplicate = (
        detection("Long subtitle", 0.7, confidence=0.98),
        detection("Long subtitle", 0.7, confidence=0.97),
    )
    tracker.update(FrameDetections(100, duplicate))
    tracker.update(FrameDetections(300, duplicate))

    assert [cue.text for cue in tracker.finalize()] == ["Long subtitle"]


def test_noise_filter_merges_repeated_long_subtitle_after_short_tracking_gap():
    tracker = TemporalTextTracker(max_missed_frames=0)
    tracker.update(FrameDetections(100, (detection("Long subtitle", 0.7, confidence=0.98),)))
    tracker.update(FrameDetections(300, ()))
    tracker.update(FrameDetections(900, (detection("Long subtitle", 0.7, confidence=0.97),)))
    tracker.update(FrameDetections(1100, (detection("Long subtitle", 0.7, confidence=0.98),)))

    cues = tracker.finalize(end_ms=1200)

    assert [(cue.text, cue.start_ms, cue.end_ms) for cue in cues] == [
        ("Long subtitle", 100, 1200)
    ]


def test_noise_filter_drops_persistent_substring_track_next_to_full_line():
    tracker = TemporalTextTracker(max_missed_frames=0)
    detections = (
        detection("今天一起去温泉吧", 0.7, confidence=0.98),
        detection("去温泉吧", 0.7, confidence=0.96),
    )
    tracker.update(FrameDetections(100, detections))
    tracker.update(FrameDetections(300, detections))

    assert [cue.text for cue in tracker.finalize()] == ["今天一起去温泉吧"]


def test_noise_filter_prefers_complete_simultaneous_text_over_clearer_substring():
    tracker = TemporalTextTracker(max_missed_frames=0)
    detections = (
        detection("ちょっと待ちなさいよ", 0.7, confidence=0.89),
        detection("待ちなさいよ", 0.7, confidence=0.999),
    )
    tracker.update(FrameDetections(100, detections))
    tracker.update(FrameDetections(300, detections))

    assert [cue.text for cue in tracker.finalize()] == ["ちょっと待ちなさいよ"]


def test_noise_filter_can_be_disabled_for_raw_single_frame_output():
    tracker = TemporalTextTracker(max_missed_frames=0, filter_noise=False)
    tracker.update(FrameDetections(100, (detection("7", 0.7, confidence=0.2),)))
    tracker.update(FrameDetections(300, ()))

    assert [cue.text for cue in tracker.finalize()] == ["7"]


def test_snapshot_is_non_destructive_and_includes_active_tracks():
    tracker = TemporalTextTracker(max_missed_frames=1)
    tracker.update(FrameDetections(100, (detection("live text", 0.7),)))

    snapshot = tracker.snapshot(end_ms=200)
    tracker.update(FrameDetections(300, (detection("live text", 0.7),)))
    finalized = tracker.finalize(end_ms=400)

    assert [(cue.text, cue.start_ms, cue.end_ms) for cue in snapshot] == [
        ("live text", 100, 200)
    ]
    assert [(cue.text, cue.start_ms, cue.end_ms) for cue in finalized] == [
        ("live text", 100, 400)
    ]


def test_snapshot_new_track_has_positive_duration_at_current_timestamp():
    tracker = TemporalTextTracker(max_missed_frames=1)
    tracker.update(FrameDetections(100, (detection("just appeared", 0.7),)))

    cue = tracker.snapshot(end_ms=100)[0]

    assert cue.start_ms == 100
    assert cue.end_ms == 101
