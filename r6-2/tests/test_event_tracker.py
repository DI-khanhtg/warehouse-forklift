import csv

from forklift_phone_detection.logic.event_tracker import EventTracker


def test_event_start_end_and_merge(tmp_path):
    tracker = EventTracker()
    tracker.update(False, 0.0, "NORMAL")
    tracker.update(True, 1.0, "PHONE_CALL", 0.7)
    tracker.update(True, 2.0, "PHONE_CALL", 0.9)
    tracker.update(False, 3.0, "NORMAL")
    assert len(tracker.events) == 1
    event = tracker.events[0]
    assert event["start_time"] == 1.0
    assert event["end_time"] == 3.0
    assert event["duration"] == 2.0
    assert event["behavior"] == "PHONE_CALL"
    assert event["max_phone_confidence"] == 0.9

    output = tmp_path / "events.csv"
    tracker.write(output)
    with output.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert rows[0]["behavior"] == "PHONE_CALL"


def test_finalize_open_event():
    tracker = EventTracker()
    tracker.update(True, 5.0, "TEXTING_OR_HOLDING_PHONE", 0.8)
    tracker.finalize(7.5)
    assert tracker.events[0]["duration"] == 2.5
