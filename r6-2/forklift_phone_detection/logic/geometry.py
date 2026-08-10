"""Reusable geometry helpers for phone/pose relationships."""

import math
from typing import Optional, Sequence, Tuple


Point = Sequence[float]
BBox = Sequence[float]


def bbox_center(bbox: BBox) -> Tuple[float, float]:
    return ((float(bbox[0]) + float(bbox[2])) / 2.0, (float(bbox[1]) + float(bbox[3])) / 2.0)


def bbox_area(bbox: BBox) -> float:
    return max(0.0, float(bbox[2]) - float(bbox[0])) * max(0.0, float(bbox[3]) - float(bbox[1]))


def bbox_iou(bbox_a: BBox, bbox_b: BBox) -> float:
    x1 = max(float(bbox_a[0]), float(bbox_b[0]))
    y1 = max(float(bbox_a[1]), float(bbox_b[1]))
    x2 = min(float(bbox_a[2]), float(bbox_b[2]))
    y2 = min(float(bbox_a[3]), float(bbox_b[3]))
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = bbox_area(bbox_a) + bbox_area(bbox_b) - intersection
    return intersection / union if union > 0 else 0.0


def nms_detections(detections, iou_threshold: float = 0.5):
    """Class-agnostic NMS for already filtered phone detections."""
    ordered = sorted(detections, key=lambda item: float(item["confidence"]), reverse=True)
    kept = []
    for detection in ordered:
        overlaps = [item for item in kept if bbox_iou(detection["bbox"], item["bbox"]) >= iou_threshold]
        if overlaps:
            existing = overlaps[0]
            sources = set(existing.get("sources", [existing.get("source", "unknown")]))
            sources.update(detection.get("sources", [detection.get("source", "unknown")]))
            existing["sources"] = sorted(sources)
            continue
        item = dict(detection)
        item["sources"] = sorted(set(item.get("sources", [item.get("source", "unknown")])))
        kept.append(item)
    return kept


def euclidean_distance(point_a: Point, point_b: Point) -> float:
    return math.hypot(float(point_a[0]) - float(point_b[0]), float(point_a[1]) - float(point_b[1]))


def is_point_inside_bbox(point: Point, bbox: BBox, margin: float = 0.0) -> bool:
    return (
        float(bbox[0]) - margin <= float(point[0]) <= float(bbox[2]) + margin
        and float(bbox[1]) - margin <= float(point[1]) <= float(bbox[3]) + margin
    )


def point_to_bbox_distance(point: Point, bbox: BBox) -> float:
    x = max(float(bbox[0]), min(float(point[0]), float(bbox[2])))
    y = max(float(bbox[1]), min(float(point[1]), float(bbox[3])))
    return euclidean_distance(point, (x, y))


def normalized_distance(point_a: Point, point_b: Point, scale: float) -> Optional[float]:
    if scale is None or scale <= 0:
        return None
    return euclidean_distance(point_a, point_b) / float(scale)


def operator_scale(pose: Optional[dict], frame_shape) -> float:
    """Use shoulder width, then person width, then a conservative frame fallback."""
    if pose:
        keypoints = pose.get("keypoints", pose)
        left = keypoints.get("left_shoulder")
        right = keypoints.get("right_shoulder")
        if left is not None and right is not None:
            width = euclidean_distance(left, right)
            if width >= 5.0:
                return width
        bbox = pose.get("bbox")
        if bbox is not None:
            width = float(bbox[2]) - float(bbox[0])
            if width >= 5.0:
                return width * 0.45
    frame_width = float(frame_shape[1]) if frame_shape is not None else 640.0
    return max(1.0, frame_width * 0.25)
