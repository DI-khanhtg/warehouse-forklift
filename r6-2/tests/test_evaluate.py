from evaluate import evaluate_events


def test_event_metrics_and_delay():
    predictions = [
        {"start": 11.0, "end": 15.0},
        {"start": 30.0, "end": 32.0},
    ]
    truth = [
        {"start": 10.0, "end": 16.0},
        {"start": 40.0, "end": 45.0},
    ]
    report = evaluate_events(predictions, truth, duration_seconds=3600)
    assert report["precision"] == 0.5
    assert report["recall"] == 0.5
    assert report["f1"] == 0.5
    assert report["false_positive_events"] == 1
    assert report["false_negative_events"] == 1
    assert report["false_alarms_per_hour"] == 1.0
    assert report["average_detection_delay_seconds"] == 1.0
