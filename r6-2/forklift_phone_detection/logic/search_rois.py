"""Generate scale-aware phone search regions from a selected driver pose."""

from typing import Dict, List, Optional, Sequence, Tuple

from .geometry import operator_scale


def _xy(keypoints: dict, name: str):
    point = keypoints.get(name)
    return None if point is None else (float(point[0]), float(point[1]))


def _clip_box(box: Sequence[float], frame_shape) -> Optional[Tuple[int, int, int, int]]:
    height, width = frame_shape[:2]
    x1 = max(0, min(width, int(round(box[0]))))
    y1 = max(0, min(height, int(round(box[1]))))
    x2 = max(0, min(width, int(round(box[2]))))
    y2 = max(0, min(height, int(round(box[3]))))
    if x2 - x1 < 8 or y2 - y1 < 8:
        return None
    return (x1, y1, x2, y2)


def _square(center, radius: float, frame_shape):
    return _clip_box(
        (center[0] - radius, center[1] - radius, center[0] + radius, center[1] + radius),
        frame_shape,
    )


def generate_phone_search_rois(pose: Optional[Dict], frame_shape, roi_scale: float = 0.75) -> List[Dict]:
    if not pose:
        return []
    keypoints = pose.get("keypoints", pose)
    scale = operator_scale(pose, frame_shape)
    radius = max(8.0, scale * float(roi_scale))
    left_wrist = _xy(keypoints, "left_wrist")
    right_wrist = _xy(keypoints, "right_wrist")
    rois = []

    for source, point in (
        ("left_wrist_roi", left_wrist),
        ("right_wrist_roi", right_wrist),
    ):
        if point is not None:
            box = _square(point, radius, frame_shape)
            if box:
                rois.append({"source": source, "bbox": box})

    if left_wrist is not None and right_wrist is not None:
        box = _clip_box(
            (
                min(left_wrist[0], right_wrist[0]) - radius,
                min(left_wrist[1], right_wrist[1]) - radius,
                max(left_wrist[0], right_wrist[0]) + radius,
                max(left_wrist[1], right_wrist[1]) + radius,
            ),
            frame_shape,
        )
        if box:
            rois.append({"source": "both_hands_roi", "bbox": box})

    head_points = [
        point for point in (
            _xy(keypoints, "left_ear"),
            _xy(keypoints, "right_ear"),
            _xy(keypoints, "nose"),
            _xy(keypoints, "left_eye"),
            _xy(keypoints, "right_eye"),
        )
        if point is not None
    ]
    if head_points:
        center = (
            sum(point[0] for point in head_points) / len(head_points),
            sum(point[1] for point in head_points) / len(head_points),
        )
        box = _square(center, radius, frame_shape)
        if box:
            rois.append({"source": "head_roi", "bbox": box})
    return rois
