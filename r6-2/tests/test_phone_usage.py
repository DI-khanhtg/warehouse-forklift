from forklift_phone_detection.config import Settings
from forklift_phone_detection.logic.phone_usage import build_head_roi, classify_phone_usage


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
    assert result["behavior"] == "HANDHELD_PHONE_USE"


def test_visible_phone_without_interaction_is_normal():
    result = classify_phone_usage([phone_at(480, 450)], pose(), (480, 640, 3))
    assert not result["using_phone"]
    assert result["behavior"] == "PHONE_PRESENT"


def test_phone_in_front_of_face_has_independent_watching_pathway():
    result = classify_phone_usage([phone_at(250, 240)], pose(), (480, 640, 3))
    assert result["using_phone"]
    assert result["behavior"] == "WATCHING_PHONE"
    assert result["pathways"]["WATCHING_PHONE"]
    assert not result["pathways"]["HANDHELD_PHONE_USE"]


def test_no_pose_fails_gracefully_to_normal():
    result = classify_phone_usage([phone_at(215, 115)], None, (480, 640, 3), Settings())
    assert not result["using_phone"]


def test_head_roi_recovers_call_when_ear_keypoints_are_missing():
    unstable_pose = pose()
    unstable_pose["keypoints"] = {
        name: value
        for name, value in unstable_pose["keypoints"].items()
        if "ear" not in name
    }
    unstable_pose["keypoints"].update({
        "nose": (250, 112, 0.9),
        "left_eye": (230, 100, 0.9),
        "right_eye": (270, 100, 0.9),
    })
    head_roi = build_head_roi(unstable_pose, (480, 640, 3))
    result = classify_phone_usage(
        [phone_at(head_roi[0] + 5, (head_roi[1] + head_roi[3]) / 2)],
        unstable_pose,
        (480, 640, 3),
    )
    assert result["behavior"] == "PHONE_CALL"
    assert result["using_phone"]
    assert result["debug"]["phone_near_head_roi"]
    assert result["debug"]["head_roi"] == head_roi
