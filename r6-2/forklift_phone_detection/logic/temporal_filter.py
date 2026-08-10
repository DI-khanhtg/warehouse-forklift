"""Time-aware sliding-window filtering with hysteresis."""

from collections import Counter, deque
from dataclasses import dataclass
from math import ceil


@dataclass(frozen=True)
class TemporalResult:
    state: str
    behavior: str
    usage_ratio: float
    valid_frames: int
    window_frames: int
    ready: bool
    covered_seconds: float
    window_seconds: float


class TemporalFilter:
    def __init__(
        self,
        fps: float,
        window_seconds: float = 1.5,
        alert_on_ratio: float = 0.6,
        alert_off_ratio: float = 0.3,
        min_window_fill_ratio: float = 0.5,
    ):
        self.fps = max(1.0, float(fps))
        self.window_seconds = max(1e-6, float(window_seconds))
        self.min_window_fill_ratio = float(min_window_fill_ratio)
        self.window_frames = max(2, int(ceil(self.fps * window_seconds)))
        self.minimum_frames = max(2, int(ceil(self.window_frames * min_window_fill_ratio)))
        self.alert_on_ratio = float(alert_on_ratio)
        self.alert_off_ratio = float(alert_off_ratio)
        if self.alert_off_ratio > self.alert_on_ratio:
            raise ValueError("alert_off_ratio must not exceed alert_on_ratio")
        self.history = deque()
        self.active = False
        self._next_synthetic_timestamp = 0.0
        self._last_timestamp = None
        self._last_interval = 1.0 / self.fps

    def update(
        self,
        using_phone: bool,
        behavior: str = "NORMAL",
        timestamp=None,
    ) -> TemporalResult:
        explicit_timestamp = timestamp is not None
        if not explicit_timestamp:
            timestamp = self._next_synthetic_timestamp
            self._next_synthetic_timestamp += 1.0 / self.fps
        timestamp = float(timestamp)
        if self._last_timestamp is not None and timestamp < self._last_timestamp:
            self.reset()
        if (
            explicit_timestamp
            and self._last_timestamp is not None
            and timestamp - self._last_timestamp >= self.window_seconds
        ):
            self.history.clear()
            self.active = False
        if self._last_timestamp is not None:
            interval = timestamp - self._last_timestamp
            if interval > 0:
                self._last_interval = interval
        self._last_timestamp = timestamp
        self.history.append(
            (timestamp, bool(using_phone), behavior if using_phone else "NORMAL")
        )
        if explicit_timestamp:
            cutoff = timestamp - self.window_seconds
            while len(self.history) > 1 and self.history[0][0] < cutoff:
                self.history.popleft()
        else:
            while len(self.history) > self.window_frames:
                self.history.popleft()

        positives = sum(1 for _, value, _ in self.history if value)
        ratio = positives / len(self.history)
        covered_seconds = max(0.0, timestamp - self.history[0][0])
        if not explicit_timestamp:
            covered_seconds += self._last_interval
        covered_seconds = min(self.window_seconds, covered_seconds)
        ready = (
            len(self.history) >= 2
            and covered_seconds
            >= self.window_seconds * self.min_window_fill_ratio
        )
        if not self.active and ready and ratio >= self.alert_on_ratio:
            self.active = True
        elif self.active and ratio <= self.alert_off_ratio:
            self.active = False

        behavior_counts = Counter(
            behavior_name for _, positive, behavior_name in self.history
            if positive and behavior_name != "NORMAL"
        )
        dominant = behavior_counts.most_common(1)[0][0] if self.active and behavior_counts else "NORMAL"
        return TemporalResult(
            state="USING_PHONE" if self.active else "NORMAL",
            behavior=dominant,
            usage_ratio=ratio,
            valid_frames=len(self.history),
            window_frames=self.window_frames,
            ready=ready,
            covered_seconds=covered_seconds,
            window_seconds=self.window_seconds,
        )

    def reset(self) -> None:
        self.history.clear()
        self.active = False
        self._next_synthetic_timestamp = 0.0
        self._last_timestamp = None
        self._last_interval = 1.0 / self.fps
