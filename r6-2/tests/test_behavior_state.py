import logging

import pytest

from forklift_phone_detection.logic.behavior_state import BehaviorStateMachine
from forklift_phone_detection.logic.behaviors import (
    HANDHELD_PHONE_USE,
    PHONE_CALL,
    WATCHING_PHONE,
    canonical_behavior,
    is_using_phone_behavior,
)


def instant(
    behavior="NORMAL",
    *,
    phone=True,
    call=False,
    handheld=False,
    watching=False,
    wrist_near_head=False,
    confidence=0.86,
):
    detection = (
        {
            "bbox": [100, 60, 125, 105],
            "center": [112.5, 82.5],
            "confidence": confidence,
        }
        if phone
        else None
    )
    return {
        "using_phone": call or handheld or watching,
        "behavior": behavior,
        "phone": detection,
        "phone_confidence": confidence if phone else 0.0,
        "pathways": {
            PHONE_CALL: call,
            HANDHELD_PHONE_USE: handheld,
            WATCHING_PHONE: watching,
        },
        "debug": {
            "phone_near_head": call,
            "phone_near_hand": handheld,
            "wrist_near_head": wrist_near_head,
            "nearest_hand_name": "right_wrist" if handheld else None,
            "call_evidence": 0.91 if call else 0.0,
        },
    }


def test_legacy_behavior_names_are_normalized_before_fusion():
    assert canonical_behavior("PHONE_NEAR_HEAD") == PHONE_CALL
    assert canonical_behavior("CALLING") == PHONE_CALL
    assert canonical_behavior("TEXTING_OR_HOLDING_PHONE") == HANDHELD_PHONE_USE
    assert is_using_phone_behavior("PHONE_NEAR_HEAD")
    assert is_using_phone_behavior("CALLING")


def test_phone_near_head_label_cannot_be_fused_as_negative():
    machine = BehaviorStateMachine(call_trigger_time=0.0)
    evidence = instant("PHONE_NEAR_HEAD", call=False)
    result = machine.update(evidence, 4.2)
    assert result.behavior_state == "CALLING"
    assert result.behavior == PHONE_CALL
    assert result.state == "USING_PHONE"


def test_direct_call_still_requires_a_valid_phone_candidate():
    machine = BehaviorStateMachine(call_trigger_time=0.0)
    result = machine.update(instant("PHONE_NEAR_HEAD", phone=False), 4.2)
    assert result.behavior_state == "NORMAL"
    assert result.state == "NORMAL"


def test_phone_call_uses_short_independent_timestamp_trigger():
    machine = BehaviorStateMachine(call_trigger_time=0.6)
    first = machine.update(instant("PHONE_CALL", call=True), 0.0)
    middle = machine.update(instant("PHONE_CALL", call=True), 0.3)
    active = machine.update(instant("PHONE_CALL", call=True), 0.6)

    assert first.behavior_state == "PHONE_PRESENT"
    assert middle.state == "NORMAL"
    assert active.behavior_state == "CALLING"
    assert active.behavior == "PHONE_CALL"
    assert active.state == "USING_PHONE"
    assert active.call_duration == 0.6
    assert active.call_evidence == 0.91
    assert machine.phone_track.last_bbox == [100, 60, 125, 105]


def test_call_survives_short_phone_miss_when_wrist_remains_near_head():
    machine = BehaviorStateMachine(call_trigger_time=0.6, call_release_time=0.7)
    machine.update(instant("PHONE_CALL", call=True), 0.0)
    active = machine.update(instant("PHONE_CALL", call=True), 0.6)
    missing = machine.update(
        instant(phone=False, wrist_near_head=True),
        1.1,
    )

    assert active.state == "USING_PHONE"
    assert missing.state == "USING_PHONE"
    assert missing.behavior == "PHONE_CALL"
    assert missing.call_persisted
    assert missing.phone_missing_duration == pytest.approx(0.5)
    assert missing.call_duration == 1.1


def test_call_miss_then_usage_hysteresis_requires_continuous_release_time():
    machine = BehaviorStateMachine(
        call_trigger_time=0.6,
        call_release_time=0.7,
        usage_release_time=0.7,
    )
    machine.update(instant("PHONE_CALL", call=True), 0.0)
    machine.update(instant("PHONE_CALL", call=True), 0.6)
    machine.update(instant(phone=False, wrist_near_head=True), 1.31)
    still_active = machine.update(instant(phone=False), 1.9)
    released = machine.update(instant(phone=False), 2.02)

    assert still_active.state == "USING_PHONE"
    assert released.state == "NORMAL"


def test_handheld_and_watching_have_independent_activation_times():
    handheld_machine = BehaviorStateMachine(handheld_trigger_time=1.0)
    handheld_machine.update(
        instant(HANDHELD_PHONE_USE, handheld=True),
        0.0,
    )
    handheld = handheld_machine.update(
        instant(HANDHELD_PHONE_USE, handheld=True),
        1.0,
    )
    assert handheld.behavior_state == "HANDHELD"
    assert handheld.behavior == HANDHELD_PHONE_USE

    watching_machine = BehaviorStateMachine(watching_trigger_time=1.5)
    watching_machine.update(instant(WATCHING_PHONE, watching=True), 0.0)
    early = watching_machine.update(instant(WATCHING_PHONE, watching=True), 1.49)
    watching = watching_machine.update(instant(WATCHING_PHONE, watching=True), 1.5)
    assert early.state == "NORMAL"
    assert watching.behavior_state == "WATCHING"
    assert watching.behavior == WATCHING_PHONE


def test_transition_logs_include_source_timestamps(caplog):
    machine = BehaviorStateMachine(call_trigger_time=0.6, usage_release_time=0.0)
    with caplog.at_level(logging.INFO, logger="r6_phone_detection"):
        machine.update(instant("PHONE_CALL", call=True), 0.0)
        machine.update(instant("PHONE_CALL", call=True), 0.6)
        machine.update(instant(phone=False), 0.7)

    messages = [record.getMessage() for record in caplog.records]
    assert any("0.000s: NORMAL -> PHONE_PRESENT" in message for message in messages)
    assert any("0.600s: PHONE_PRESENT -> CALLING" in message for message in messages)
    assert any("0.600s: CALLING -> USING_PHONE" in message for message in messages)
    assert any("0.700s: USING_PHONE -> NORMAL" in message for message in messages)
