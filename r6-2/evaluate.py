"""Event-level evaluation for R6.2 behavior detection."""

import argparse
import csv
import json
from pathlib import Path


def load_ground_truth(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return [
        {"start": float(item["start"]), "end": float(item["end"]), "label": item.get("label", "USING_PHONE")}
        for item in data
        if item.get("label", "USING_PHONE") == "USING_PHONE"
    ]


def load_predictions(path):
    path = Path(path)
    if path.suffix.lower() == ".json":
        rows = json.loads(path.read_text(encoding="utf-8"))
    else:
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    return [{"start": float(row["start_time"]), "end": float(row["end_time"]), **row} for row in rows]


def interval_overlap(a, b) -> float:
    return max(0.0, min(a["end"], b["end"]) - max(a["start"], b["start"]))


def evaluate_events(predictions, ground_truth, duration_seconds=None):
    """Greedily match events by greatest positive temporal overlap."""
    possible = []
    for prediction_index, prediction in enumerate(predictions):
        for truth_index, truth in enumerate(ground_truth):
            overlap = interval_overlap(prediction, truth)
            if overlap > 0:
                possible.append((overlap, prediction_index, truth_index))
    matched_predictions, matched_truth, matches = set(), set(), []
    for overlap, prediction_index, truth_index in sorted(possible, reverse=True):
        if prediction_index in matched_predictions or truth_index in matched_truth:
            continue
        matched_predictions.add(prediction_index)
        matched_truth.add(truth_index)
        matches.append((prediction_index, truth_index, overlap))

    true_positives = len(matches)
    false_positives = len(predictions) - true_positives
    false_negatives = len(ground_truth) - true_positives
    precision = true_positives / len(predictions) if predictions else (1.0 if not ground_truth else 0.0)
    recall = true_positives / len(ground_truth) if ground_truth else (1.0 if not predictions else 0.0)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    delays = [
        max(0.0, predictions[prediction_index]["start"] - ground_truth[truth_index]["start"])
        for prediction_index, truth_index, _ in matches
    ]
    if duration_seconds is None:
        endpoints = [item["end"] for item in predictions] + [item["end"] for item in ground_truth]
        duration_seconds = max(endpoints, default=0.0)
    false_alarms_per_hour = false_positives / (duration_seconds / 3600.0) if duration_seconds > 0 else 0.0
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "true_positive_events": true_positives,
        "false_positive_events": false_positives,
        "false_negative_events": false_negatives,
        "false_alarms_per_hour": round(false_alarms_per_hour, 4),
        "average_detection_delay_seconds": round(sum(delays) / len(delays), 4) if delays else None,
        "duration_seconds": round(float(duration_seconds), 3),
        "matched_events": len(matches),
    }


def build_parser():
    parser = argparse.ArgumentParser(description="Evaluate R6.2 detected events against manual intervals")
    parser.add_argument("--predictions", required=True, help="events.csv or events.json")
    parser.add_argument("--ground-truth", required=True, help="Manual interval JSON")
    parser.add_argument("--duration", type=float, default=None, help="Full video duration in seconds")
    parser.add_argument("--output", default=None, help="Optional JSON report path")
    return parser


def main():
    args = build_parser().parse_args()
    try:
        predictions = load_predictions(args.predictions)
        ground_truth = load_ground_truth(args.ground_truth)
        report = evaluate_events(predictions, ground_truth, args.duration)
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"Evaluation failed: {exc}")
        return 1
    print("R6.2 Event Evaluation")
    print(f"Precision: {report['precision']:.4f}")
    print(f"Recall: {report['recall']:.4f}")
    print(f"F1: {report['f1']:.4f}")
    print(f"False positive events: {report['false_positive_events']}")
    print(f"False negative events: {report['false_negative_events']}")
    print(f"False alarms / hour: {report['false_alarms_per_hour']:.4f}")
    delay = report["average_detection_delay_seconds"]
    print(f"Average detection delay: {'n/a' if delay is None else f'{delay:.3f} s'}")
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Report: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
