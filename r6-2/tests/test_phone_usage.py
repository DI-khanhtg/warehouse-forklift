from forklift_phone_detection.config import Settings
from forklift_phone_detection.logic.phone_usage import classify_phone_usage


def pose():
    return {
        "bbox": [100, 50, 500, 470],
        "keypoints": {
            "left_ear": (210, 110, 0.9),
            "right_ear": (290, 110, 0.9),
            "left_shoulder": (180, 190, 0.9),
            "right_shoulder": (320, 190, 0.9),
            "left_wrist": (190, 300, 0.9),
            "right_wrist": (330, 300, 0.9),
        },
    }


def phone_at(x, y, confidence=0.8):
    return {"bbox": [x - 10, y - 20, x + 10, y + 20], "center": [x, y], "confidence": confidence}


def test_phone_near_ear_is_phone_call():
    result = classify_phone_usage([phone_at(215, 115)], pose(), (480, 640, 3))
    assert result["using_phone"]
    assert result["behavior"] == "PHONE_CALL"
    assert result["debug"]["phone_near_head"]


def test_phone_near_wrist_is_holding_phone():
    result = classify_phone_usage([phone_at(195, 305)], pose(), (480, 640, 3))
    assert result["using_phone"]
    assert result["behavior"] == "TEXTING_OR_HOLDING_PHONE"


def test_visible_phone_without_interaction_is_normal():
    result = classify_phone_usage([phone_at(480, 450)], pose(), (480, 640, 3))
    assert not result["using_phone"]
    assert result["behavior"] == "NORMAL"


def test_no_pose_fails_gracefully_to_normal():
    result = classify_phone_usage([phone_at(215, 115)], None, (480, 640, 3), Settings())
    assert not result["using_phone"]
