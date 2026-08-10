from forklift_phone_detection.logic.pose_tracker import DriverPoseTracker
from forklift_phone_detection.models.pose_detector import select_pose_index


def make_pose(x=0, wrist_x=25):
    return {
        "bbox": [x, 0, x + 100, 200],
        "confidence": 0.9,
        "keypoints": {
            "left_shoulder": (x + 25, 60, 0.9),
            "right_shoulder": (x + 75, 60, 0.9),
            "left_elbow": (x + 25, 100, 0.9),
            "right_elbow": (x + 75, 100, 0.9),
            "left_wrist": (x + wrist_x, 140, 0.9),
            "right_wrist": (x + 75, 140, 0.9),
        },
    }


def test_tracker_smooths_small_driver_motion():
    tracker = DriverPoseTracker(smoothing_alpha=0.5, max_center_jump=0.35)
    first = tracker.update(make_pose(0), (240, 640, 3))
    second = tracker.update(make_pose(10), (240, 640, 3))
    assert first["tracking_status"] == "acquired"
    assert second["tracking_status"] == "tracked"
    assert second["tracking_fresh"]
    assert second["bbox"][0] == 5.0
    assert second["keypoints"]["left_shoulder"][0] == 30.0


def test_far_candidate_is_held_for_drawing_but_not_fresh():
    tracker = DriverPoseTracker(max_center_jump=0.25, minimum_iou=0.1)
    first = tracker.update(make_pose(0), (240, 640, 3))
    held = tracker.update(make_pose(350), (240, 640, 3))
    assert held["track_id"] == first["track_id"]
    assert held["tracking_status"] == "held"
    assert held["tracking_stale"]
    assert not held["tracking_fresh"]
    assert held["tracking_reason"] == "driver_jump_rejected"


def test_repeated_far_candidate_eventually_reacquires_new_track():
    tracker = DriverPoseTracker(max_center_jump=0.25, max_missed_frames=2)
    original = tracker.update(make_pose(0), (240, 640, 3))
    tracker.update(make_pose(350), (240, 640, 3))
    tracker.update(make_pose(350), (240, 640, 3))
    reacquired = tracker.update(make_pose(350), (240, 640, 3))
    assert reacquired["tracking_status"] == "reacquired"
    assert reacquired["track_id"] != original["track_id"]


def test_isolated_wrist_outlier_is_removed_from_fresh_evidence():
    tracker = DriverPoseTracker(keypoint_max_jump=0.5)
    tracker.update(make_pose(0), (240, 640, 3))
    outlier = make_pose(0, wrist_x=95)
    tracked = tracker.update(outlier, (240, 640, 3))
    assert "left_wrist" in tracked["tracking_rejected_keypoints"]
    assert "left_wrist" not in tracked["keypoints"]


def test_repeated_wrist_outlier_is_compared_with_last_trusted_joint():
    tracker = DriverPoseTracker(keypoint_max_jump=0.5)
    tracker.update(make_pose(0), (240, 640, 3))
    second = tracker.update(make_pose(0, wrist_x=95), (240, 640, 3))
    third = tracker.update(make_pose(0, wrist_x=100), (240, 640, 3))
    assert "left_wrist" in second["tracking_rejected_keypoints"]
    assert "left_wrist" in third["tracking_rejected_keypoints"]
    assert "left_wrist" not in third["keypoints"]


def test_large_overlapping_bystander_cannot_bypass_center_jump_gate():
    tracker = DriverPoseTracker(max_center_jump=0.35, minimum_iou=0.1)
    tracker.update(make_pose(0), (360, 640, 3))
    bystander = make_pose(60)
    bystander["bbox"] = [60, 0, 260, 300]
    held = tracker.update(bystander, (360, 640, 3))
    assert held["tracking_status"] == "held"
    assert held["tracking_reason"] == "driver_jump_rejected"


def test_pose_selection_prefers_previous_driver_over_larger_bystander():
    previous = [10, 10, 110, 210]
    boxes = [
        [12, 12, 112, 212],
        [160, 0, 360, 300],
    ]
    assert select_pose_index(boxes, [0.8, 0.95]) == 1
    assert select_pose_index(boxes, [0.8, 0.95], target_bbox=previous) == 0
