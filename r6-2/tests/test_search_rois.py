from forklift_phone_detection.logic.search_rois import generate_phone_search_rois


def test_pose_search_rois_are_scale_aware():
    pose = {
        "bbox": [40, 20, 280, 460],
        "keypoints": {
            "left_shoulder": (100, 150, 0.9),
            "right_shoulder": (200, 150, 0.9),
            "left_wrist": (100, 250, 0.9),
            "right_wrist": (200, 250, 0.9),
            "left_ear": (125, 80, 0.9),
            "right_ear": (175, 80, 0.9),
            "nose": (150, 95, 0.9),
        },
    }
    rois = generate_phone_search_rois(pose, (480, 640, 3), roi_scale=0.5)
    by_source = {item["source"]: item["bbox"] for item in rois}
    assert set(by_source) == {
        "left_wrist_roi", "right_wrist_roi", "both_hands_roi", "head_roi"
    }
    assert by_source["left_wrist_roi"] == (50, 200, 150, 300)
    assert by_source["right_wrist_roi"] == (150, 200, 250, 300)
    assert by_source["both_hands_roi"] == (50, 200, 250, 300)


def test_search_rois_clip_to_frame_and_handle_missing_keypoints():
    pose = {
        "bbox": [0, 0, 100, 200],
        "keypoints": {"left_wrist": (2, 3, 0.9)},
    }
    rois = generate_phone_search_rois(pose, (100, 100, 3), roi_scale=1.0)
    assert rois == [{"source": "left_wrist_roi", "bbox": (0, 0, 47, 48)}]
    assert generate_phone_search_rois(None, (100, 100, 3)) == []
