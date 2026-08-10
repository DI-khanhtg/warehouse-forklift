import math

from forklift_phone_detection.logic.geometry import (
    bbox_area,
    bbox_center,
    bbox_iou,
    euclidean_distance,
    is_point_inside_bbox,
    normalized_distance,
    nms_detections,
    operator_scale,
    point_to_bbox_distance,
)


def test_basic_geometry():
    bbox = [10, 20, 30, 50]
    assert bbox_center(bbox) == (20.0, 35.0)
    assert bbox_area(bbox) == 600.0
    assert euclidean_distance((0, 0), (3, 4)) == 5.0
    assert is_point_inside_bbox((10, 35), bbox)
    assert not is_point_inside_bbox((9, 35), bbox)


def test_point_to_bbox_distance_is_zero_inside_and_correct_outside():
    bbox = [10, 10, 20, 20]
    assert point_to_bbox_distance((15, 15), bbox) == 0.0
    assert math.isclose(point_to_bbox_distance((7, 6), bbox), 5.0)


def test_normalized_distance_and_operator_scale():
    assert normalized_distance((0, 0), (6, 8), 20) == 0.5
    assert normalized_distance((0, 0), (1, 1), 0) is None
    pose = {
        "keypoints": {
            "left_shoulder": (10, 20, 0.9),
            "right_shoulder": (110, 20, 0.9),
        },
        "bbox": [0, 0, 200, 400],
    }
    assert operator_scale(pose, (480, 640, 3)) == 100.0


def test_operator_scale_fallbacks():
    assert operator_scale({"keypoints": {}, "bbox": [10, 0, 210, 400]}, (480, 640, 3)) == 90.0
    assert operator_scale(None, (480, 800, 3)) == 200.0


def test_iou_nms_keeps_highest_confidence_and_sources():
    detections = [
        {"bbox": [10, 10, 30, 30], "confidence": 0.4, "source": "full_frame"},
        {"bbox": [11, 11, 31, 31], "confidence": 0.8, "source": "left_wrist_roi"},
        {"bbox": [100, 100, 120, 120], "confidence": 0.3, "source": "head_roi"},
    ]
    assert bbox_iou(detections[0]["bbox"], detections[1]["bbox"]) > 0.5
    kept = nms_detections(detections, 0.5)
    assert len(kept) == 2
    assert kept[0]["confidence"] == 0.8
    assert kept[0]["source"] == "left_wrist_roi"
    assert kept[0]["sources"] == ["full_frame", "left_wrist_roi"]
