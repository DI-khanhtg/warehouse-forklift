from forklift_phone_detection.logic.temporal_filter import TemporalFilter


def test_single_frame_never_triggers():
    temporal = TemporalFilter(fps=10, window_seconds=1, min_window_fill_ratio=0.5)
    result = temporal.update(True, "PHONE_CALL")
    assert result.state == "NORMAL"
    assert not result.ready


def test_window_and_hysteresis():
    temporal = TemporalFilter(
        fps=10,
        window_seconds=1,
        alert_on_ratio=0.6,
        alert_off_ratio=0.3,
        min_window_fill_ratio=0.5,
    )
    for _ in range(5):
        result = temporal.update(True, "PHONE_CALL")
    assert result.state == "USING_PHONE"
    assert result.behavior == "PHONE_CALL"
    for _ in range(4):
        result = temporal.update(False)
    assert result.state == "USING_PHONE"
    for _ in range(3):
        result = temporal.update(False)
    assert result.state == "NORMAL"
    assert result.usage_ratio <= 0.3


def test_reset():
    temporal = TemporalFilter(fps=2, window_seconds=1, min_window_fill_ratio=1)
    temporal.update(True, "PHONE_CALL")
    result = temporal.update(True, "PHONE_CALL")
    assert result.state == "USING_PHONE"
    temporal.reset()
    assert not temporal.history
    assert not temporal.active


def test_explicit_timestamps_control_window_instead_of_reported_fps():
    temporal = TemporalFilter(
        fps=30,
        window_seconds=1.0,
        min_window_fill_ratio=0.5,
    )
    first = temporal.update(True, "PHONE_CALL", timestamp=0.0)
    second = temporal.update(True, "PHONE_CALL", timestamp=0.2)
    third = temporal.update(True, "PHONE_CALL", timestamp=0.6)
    assert not first.ready
    assert not second.ready
    assert third.ready
    assert third.state == "USING_PHONE"


def test_long_timestamp_gap_expires_an_active_state():
    temporal = TemporalFilter(
        fps=30,
        window_seconds=1.0,
        min_window_fill_ratio=0.5,
    )
    temporal.update(True, "PHONE_CALL", timestamp=0.0)
    active = temporal.update(True, "PHONE_CALL", timestamp=0.6)
    expired = temporal.update(True, "PHONE_CALL", timestamp=2.0)
    assert active.state == "USING_PHONE"
    assert expired.state == "NORMAL"
    assert not expired.ready
    assert expired.valid_frames == 1
