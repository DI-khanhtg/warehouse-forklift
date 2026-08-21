"""Timestamp-driven phone behavior state machine with short occlusion memory."""

import logging
from dataclasses import dataclass
from typing import Optional

from .behaviors import (
    HANDHELD_PHONE_USE,
    NORMAL,
    PHONE_CALL,
    PHONE_PRESENT,
    WATCHING_PHONE,
    canonical_behavior,
)


STATE_NORMAL = "NORMAL"
STATE_PHONE_PRESENT = "PHONE_PRESENT"
STATE_HANDHELD = "HANDHELD"
STATE_CALLING = "CALLING"
STATE_WATCHING = "WATCHING"

STATE_TO_BEHAVIOR = {
    STATE_NORMAL: NORMAL,
    STATE_PHONE_PRESENT: NORMAL,
    STATE_HANDHELD: HANDHELD_PHONE_USE,
    STATE_CALLING: PHONE_CALL,
    STATE_WATCHING: WATCHING_PHONE,
}
USAGE_STATES = frozenset({STATE_HANDHELD, STATE_CALLING, STATE_WATCHING})


@dataclass
class PhoneTrack:
    last_bbox: Optional[list] = None
    last_phone_confidence: float = 0.0
    last_seen_time: Optional[float] = None
    missing_duration: float = 0.0
    previous_behavior: str = NORMAL
    behavior_start_time: Optional[float] = None
    last_near_head: bool = False
    associated_wrist: Optional[str] = None

    def update(self, instant: dict, timestamp: float) -> None:
        phone = instant.get("phone")
        debug = instant.get("debug", {})
        if phone is not None:
            self.last_bbox = list(phone["bbox"])
            self.last_phone_confidence = float(phone.get("confidence", 0.0))
            self.last_seen_time = timestamp
            self.missing_duration = 0.0
            self.last_near_head = bool(debug.get("phone_near_head", False))
            wrist = debug.get("nearest_hand_name")
            self.associated_wrist = wrist if debug.get("phone_near_hand") else None
        elif self.last_seen_time is not None:
            self.missing_duration = max(0.0, timestamp - self.last_seen_time)

    def reset(self) -> None:
        self.last_bbox = None
        self.last_phone_confidence = 0.0
        self.last_seen_time = None
        self.missing_duration = 0.0
        self.previous_behavior = NORMAL
        self.behavior_start_time = None
        self.last_near_head = False
        self.associated_wrist = None


@dataclass(frozen=True)
class BehaviorStateResult:
    state: str
    behavior: str
    behavior_state: str
    phone_confidence: float
    near_head: bool
    near_wrist: bool
    call_evidence: float
    call_duration: float
    phone_missing_duration: float
    call_persisted: bool
    associated_wrist: Optional[str]
    evidence_duration: float
    release_duration: float
    # Compatibility fields retained for existing callers/reports. The final
    # decision no longer uses a positive-frame ratio.
    usage_ratio: float
    valid_frames: int
    window_frames: int
    ready: bool
    covered_seconds: float
    window_seconds: float


class BehaviorStateMachine:
    """Fuse independent behavior pathways using source timestamps."""

    def __init__(
        self,
        call_trigger_time: float = 0.6,
        handheld_trigger_time: float = 1.0,
        watching_trigger_time: float = 1.5,
        usage_release_time: float = 0.7,
        call_release_time: float = 0.7,
        logger=None,
    ):
        self.trigger_times = {
            PHONE_CALL: max(0.0, float(call_trigger_time)),
            HANDHELD_PHONE_USE: max(0.0, float(handheld_trigger_time)),
            WATCHING_PHONE: max(0.0, float(watching_trigger_time)),
        }
        self.usage_release_time = max(0.0, float(usage_release_time))
        self.call_release_time = max(0.0, float(call_release_time))
        self.log = logger or logging.getLogger("r6_phone_detection")
        self.phone_track = PhoneTrack()
        self.behavior_state = STATE_NORMAL
        self._pathway_started = {name: None for name in self.trigger_times}
        self._insufficient_started = None
        self._last_timestamp = None
        self._observations = 0

    @property
    def active(self) -> bool:
        return self.behavior_state in USAGE_STATES

    def _pathway_duration(self, behavior: str, timestamp: float) -> float:
        started = self._pathway_started[behavior]
        return 0.0 if started is None else max(0.0, timestamp - started)

    def _update_pathway_timer(self, behavior: str, positive: bool, timestamp: float) -> None:
        if positive:
            if self._pathway_started[behavior] is None:
                self._pathway_started[behavior] = timestamp
        else:
            self._pathway_started[behavior] = None

    def _log_transition(self, old_state: str, new_state: str, timestamp: float) -> None:
        if old_state == new_state:
            return
        self.log.info(
            "Behavior transition at source %.3fs: %s -> %s",
            timestamp,
            old_state,
            new_state,
        )
        old_active = old_state in USAGE_STATES
        new_active = new_state in USAGE_STATES
        if not old_active and new_active:
            self.log.info(
                "Final transition at source %.3fs: %s -> USING_PHONE",
                timestamp,
                new_state,
            )
        elif old_active and not new_active:
            self.log.info(
                "Final transition at source %.3fs: USING_PHONE -> NORMAL",
                timestamp,
            )

    def update(self, instant: dict, timestamp: float) -> BehaviorStateResult:
        timestamp = max(0.0, float(timestamp))
        if self._last_timestamp is not None and timestamp < self._last_timestamp:
            self.reset()
        self._last_timestamp = timestamp
        self._observations += 1
        self.phone_track.update(instant, timestamp)

        debug = instant.get("debug", {})
        pathways = instant.get("pathways", {})
        instant_behavior = canonical_behavior(instant.get("behavior"))
        phone_visible = instant.get("phone") is not None
        direct_call = bool(
            phone_visible
            and (pathways.get(PHONE_CALL, False) or instant_behavior == PHONE_CALL)
        )
        direct_handheld = bool(
            phone_visible
            and (
                pathways.get(HANDHELD_PHONE_USE, False)
                or instant_behavior == HANDHELD_PHONE_USE
            )
        )
        direct_watching = bool(
            phone_visible
            and (
                pathways.get(WATCHING_PHONE, False)
                or instant_behavior == WATCHING_PHONE
            )
        )
        call_persisted = bool(
            not phone_visible
            and self.phone_track.previous_behavior == PHONE_CALL
            and self.phone_track.last_near_head
            and self.phone_track.missing_duration <= self.call_release_time
            and debug.get("wrist_near_head", False)
        )
        call_positive = direct_call or call_persisted

        self._update_pathway_timer(PHONE_CALL, call_positive, timestamp)
        self._update_pathway_timer(HANDHELD_PHONE_USE, direct_handheld, timestamp)
        self._update_pathway_timer(WATCHING_PHONE, direct_watching, timestamp)
        durations = {
            name: self._pathway_duration(name, timestamp)
            for name in self.trigger_times
        }
        eligible = {
            name: durations[name] >= self.trigger_times[name]
            and self._pathway_started[name] is not None
            for name in self.trigger_times
        }

        requested_state = None
        # Calls are strongest. Handheld may activate before the more cautious
        # watching timer if both independent pathways are present.
        if eligible[PHONE_CALL]:
            requested_state = STATE_CALLING
        elif eligible[WATCHING_PHONE]:
            requested_state = STATE_WATCHING
        elif eligible[HANDHELD_PHONE_USE]:
            requested_state = STATE_HANDHELD

        old_state = self.behavior_state
        if requested_state is not None:
            new_state = requested_state
            self._insufficient_started = None
        elif old_state in USAGE_STATES:
            if self._insufficient_started is None:
                self._insufficient_started = timestamp
            insufficient_duration = max(0.0, timestamp - self._insufficient_started)
            if insufficient_duration < self.usage_release_time:
                new_state = old_state
            else:
                new_state = STATE_PHONE_PRESENT if phone_visible else STATE_NORMAL
        else:
            self._insufficient_started = None
            new_state = STATE_PHONE_PRESENT if phone_visible else STATE_NORMAL

        self.behavior_state = new_state
        behavior = STATE_TO_BEHAVIOR[new_state]
        if new_state in USAGE_STATES:
            if behavior != self.phone_track.previous_behavior:
                self.phone_track.behavior_start_time = timestamp
            self.phone_track.previous_behavior = behavior
        elif old_state in USAGE_STATES and new_state not in USAGE_STATES:
            self.phone_track.previous_behavior = NORMAL
            self.phone_track.behavior_start_time = None
        self._log_transition(old_state, new_state, timestamp)

        release_duration = (
            0.0
            if self._insufficient_started is None
            else max(0.0, timestamp - self._insufficient_started)
        )
        evidence_duration = durations.get(behavior, 0.0)
        direct_call_score = float(debug.get("call_evidence", 0.0))
        if call_persisted:
            remaining = max(
                0.0,
                1.0 - self.phone_track.missing_duration / max(self.call_release_time, 1e-6),
            )
            call_score = self.phone_track.last_phone_confidence * remaining
        else:
            call_score = direct_call_score
        near_head = bool(debug.get("phone_near_head", False))
        if call_persisted:
            near_head = self.phone_track.last_near_head

        return BehaviorStateResult(
            state="USING_PHONE" if new_state in USAGE_STATES else NORMAL,
            behavior=behavior,
            behavior_state=new_state,
            phone_confidence=self.phone_track.last_phone_confidence,
            near_head=near_head,
            near_wrist=bool(debug.get("phone_near_hand", False)),
            call_evidence=call_score,
            call_duration=durations[PHONE_CALL],
            phone_missing_duration=self.phone_track.missing_duration,
            call_persisted=call_persisted,
            associated_wrist=self.phone_track.associated_wrist,
            evidence_duration=evidence_duration,
            release_duration=release_duration,
            usage_ratio=1.0 if new_state in USAGE_STATES else 0.0,
            valid_frames=self._observations,
            window_frames=0,
            ready=new_state in USAGE_STATES,
            covered_seconds=evidence_duration,
            window_seconds=max(self.trigger_times.values()),
        )

    def reset(self) -> None:
        self.phone_track.reset()
        self.behavior_state = STATE_NORMAL
        self._pathway_started = {name: None for name in self.trigger_times}
        self._insufficient_started = None
        self._last_timestamp = None
        self._observations = 0
