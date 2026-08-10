"""Explainable instantaneous phone-use rules."""

from typing import Dict, Iterable, Optional, Tuple

from ..config import Settings
from .geometry import (
    euclidean_distance,
    is_point_inside_bbox,
    normalized_distance,
    operator_scale,
)


def _point(keypoints: dict, name: str):
    value = keypoints.get(name)
    return None if value is None else (float(value[0]), float(value[1]))


def _nearest(center, named_points: Iterable[Tuple[str, Tuple[float, float]]], scale: float):
    distances = []
    for name, point in named_points:
        if point is not None:
            distances.append((normalized_distance(center, point, scale), name, point))
    return min(distances, key=lambda value: value[0]) if distances else (None, None, None)


def _empty_result(phone_count: int = 0) -> Dict:
    return {
        "using_phone": False,
        "behavior": "NORMAL",
        "score": 0.0,
        "phone_confidence": 0.0,
        "phone": None,
        "debug": {
            "phone_count": phone_count,
            "phone_near_head": False,
            "phone_near_hand": False,
            "phone_to_head_distance": None,
            "phone_to_hand_distance": None,
            "nearest_head_point": None,
            "nearest_hand_point": None,
            "operator_scale": None,
            "inside_body_region": False,
            "below_shoulders": None,
        },
    }


def classify_phone_usage(
    phone_detections,
    pose: Optional[dict],
    frame_shape,
    settings: Optional[Settings] = None,
) -> Dict:
    """Classify a single frame; phone visibility alone is never a violation."""
    settings = settings or Settings()
    if not phone_detections:
        return _empty_result()
    if not pose:
        return _empty_result(len(phone_detections))

    keypoints = pose.get("keypoints", pose)
    scale = operator_scale(pose, frame_shape)
    hand_points = [(name, _point(keypoints, name)) for name in ("left_wrist", "right_wrist")]
    ear_points = [(name, _point(keypoints, name)) for name in ("left_ear", "right_ear")]
    if not any(point for _, point in ear_points):
        ear_points = [
            (name, _point(keypoints, name))
            for name in ("nose", "left_eye", "right_eye")
        ]

    candidates = []
    person_bbox = pose.get("bbox")
    shoulders = [
        _point(keypoints, name) for name in ("left_shoulder", "right_shoulder")
        if _point(keypoints, name) is not None
    ]
    shoulder_y = sum(point[1] for point in shoulders) / len(shoulders) if shoulders else None

    for phone in phone_detections:
        center = tuple(phone.get("center") or (
            (phone["bbox"][0] + phone["bbox"][2]) / 2.0,
            (phone["bbox"][1] + phone["bbox"][3]) / 2.0,
        ))
        hand_distance, hand_name, hand_point = _nearest(center, hand_points, scale)
        head_distance, head_name, head_point = _nearest(center, ear_points, scale)
        wrist_overlap = any(
            point is not None and is_point_inside_bbox(point, phone["bbox"], margin=2.0)
            for _, point in hand_points
        )
        near_hand = wrist_overlap or (
            hand_distance is not None
            and hand_distance < settings.hand_phone_distance_threshold
        )
        near_head = (
            head_distance is not None
            and head_distance < settings.head_phone_distance_threshold
        )
        inside_body = person_bbox is not None and is_point_inside_bbox(center, person_bbox)
        below_shoulders = shoulder_y is None or center[1] > shoulder_y
        body_ok = inside_body or not settings.require_phone_in_body_region
        vertical_ok = below_shoulders or not settings.require_phone_below_shoulders

        if near_head:
            behavior = "PHONE_CALL"
        elif near_hand and body_ok and vertical_ok:
            behavior = "TEXTING_OR_HOLDING_PHONE"
        else:
            behavior = "NORMAL"
        proximity = 0.0
        if behavior == "PHONE_CALL" and head_distance is not None:
            proximity = max(0.0, 1.0 - head_distance / settings.head_phone_distance_threshold)
        elif behavior == "TEXTING_OR_HOLDING_PHONE" and hand_distance is not None:
            proximity = max(0.0, 1.0 - hand_distance / settings.hand_phone_distance_threshold)
        score = float(phone["confidence"]) * (0.5 + 0.5 * proximity) if behavior != "NORMAL" else 0.0
        candidates.append(
            {
                "using_phone": behavior != "NORMAL",
                "behavior": behavior,
                "score": score,
                "phone_confidence": float(phone["confidence"]),
                "phone": phone,
                "debug": {
                    "phone_count": len(phone_detections),
                    "phone_near_head": near_head,
                    "phone_near_hand": near_hand,
                    "phone_to_head_distance": head_distance,
                    "phone_to_hand_distance": hand_distance,
                    "nearest_head_point": head_point,
                    "nearest_head_name": head_name,
                    "nearest_hand_point": hand_point,
                    "nearest_hand_name": hand_name,
                    "operator_scale": scale,
                    "inside_body_region": inside_body,
                    "below_shoulders": below_shoulders if shoulder_y is not None else None,
                    "wrist_overlaps_phone": wrist_overlap,
                },
            }
        )

    priority = {"NORMAL": 0, "TEXTING_OR_HOLDING_PHONE": 1, "PHONE_CALL": 2}
    return max(candidates, key=lambda item: (priority[item["behavior"]], item["score"], item["phone_confidence"]))
