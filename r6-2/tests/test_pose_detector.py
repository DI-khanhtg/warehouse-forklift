from forklift_phone_detection.models.pose_detector import RELEVANT_KEYPOINTS


def test_pose_tracker_keeps_arm_and_face_evidence_points():
    assert {
        "nose",
        "left_eye",
        "right_eye",
        "left_ear",
        "right_ear",
        "left_shoulder",
        "right_shoulder",
        "left_elbow",
        "right_elbow",
        "left_wrist",
        "right_wrist",
    }.issubset(RELEVANT_KEYPOINTS)
