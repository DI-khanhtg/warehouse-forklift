"""Merge consecutive alert frames into violation events."""

import csv
import json
from collections import Counter
from pathlib import Path


EVENT_FIELDS = (
    "event_id", "start_time", "end_time", "duration", "behavior", "max_phone_confidence"
)


class EventTracker:
    def __init__(self):
        self.events = []
        self.current = None

    def update(self, active: bool, timestamp: float, behavior: str, phone_confidence: float = 0.0):
        timestamp = max(0.0, float(timestamp))
        if active and self.current is None:
            self.current = {
                "event_id": len(self.events) + 1,
                "start_time": timestamp,
                "last_time": timestamp,
                "behaviors": Counter(),
                "max_phone_confidence": float(phone_confidence),
            }
        if active and self.current is not None:
            self.current["last_time"] = timestamp
            if behavior and behavior != "NORMAL":
                self.current["behaviors"][behavior] += 1
            self.current["max_phone_confidence"] = max(
                self.current["max_phone_confidence"], float(phone_confidence)
            )
        elif not active and self.current is not None:
            self._close(timestamp)
        return self.current

    def _close(self, end_time: float):
        current = self.current
        end_time = max(float(end_time), current["last_time"], current["start_time"])
        behavior = current["behaviors"].most_common(1)[0][0] if current["behaviors"] else "USING_PHONE"
        event = {
            "event_id": current["event_id"],
            "start_time": round(current["start_time"], 3),
            "end_time": round(end_time, 3),
            "duration": round(end_time - current["start_time"], 3),
            "behavior": behavior,
            "max_phone_confidence": round(current["max_phone_confidence"], 4),
        }
        self.events.append(event)
        self.current = None
        return event

    def finalize(self, timestamp: float):
        if self.current is not None:
            return self._close(timestamp)
        return None

    @property
    def current_duration(self) -> float:
        if self.current is None:
            return 0.0
        return max(0.0, self.current["last_time"] - self.current["start_time"])

    def write(self, csv_path, json_path=None):
        csv_path = Path(csv_path)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=EVENT_FIELDS)
            writer.writeheader()
            writer.writerows(self.events)
        if json_path:
            json_path = Path(json_path)
            json_path.parent.mkdir(parents=True, exist_ok=True)
            json_path.write_text(json.dumps(self.events, indent=2), encoding="utf-8")
