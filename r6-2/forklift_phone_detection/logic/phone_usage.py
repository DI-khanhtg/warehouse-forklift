"""Explainable instantaneous evidence for independent phone-use pathways."""

from typing import Dict, Iterable, Optional, Tuple

from ..config import Settings
from .behaviors import (
    HANDHELD_PHONE_USE,
    NORMAL,
    PHONE_CALL,
    PHONE_PRESENT,
    WATCHING_PHONE,
    is_using_phone_behavior,
)
from .geometry import (
    bbox_center,
    bbox_iou,
    is_point_inside_bbox,
    normalized_distance,
    operator_scale,
    point_to_bbox_distance,
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


def _boxes_overlap(box_a, box_b) -> bool:
    return not (
        float(box_a[2]) < float(box_b[0])
        or float(box_b[2]) < float(box_a[0])
        or float(box_a[3]) < float(box_b[1])
        or float(box_b[3]) < float(box_a[1])
    )


def build_head_roi(pose: Optional[dict], frame_shape):
    """Build a scale-aware head box from face points with a person-box fallback."""
    if not pose:
        return None
    height, width = frame_shape[:2]
    keypoints = pose.get("keypoints", pose)
    scale = operator_scale(pose, frame_shape)
    face_points = [
        _point(keypoints, name)
        for name in ("nose", "left_eye", "right_eye", "left_ear", "right_ear")
    ]
    face_points = [point for point in face_points if point is not None]
    shoulders = [
        _point(keypoints, name) for name in ("left_shoulder", "right_shoulder")
    ]
    shoulders = [point for point in shoulders if point is not None]

    if face_points:
        xs = [point[0] for point in face_points]
        ys = [point[1] for point in face_points]
        center_x = sum(xs) / len(xs)
        half_width = max(scale * 0.42, (max(xs) - min(xs)) / 2.0 + scale * 0.18)
        top = min(ys) - scale * 0.35
        bottom = max(ys) + scale * 0.45
        if shoulders:
            shoulder_y = sum(point[1] for point in shoulders) / len(shoulders)
            bottom = max(bottom, shoulder_y - scale * 0.10)
            bottom = min(bottom, shoulder_y + scale * 0.08)
        box = (center_x - half_width, top, center_x + half_width, bottom)
    else:
        person_bbox = pose.get("bbox")
        if person_bbox is None:
            return None
        x1, y1, x2, y2 = (float(value) for value in person_bbox)
        person_width = max(1.0, x2 - x1)
        person_height = max(1.0, y2 - y1)
        head_width = min(person_width * 0.55, max(scale, person_width * 0.30))
        center_x = (x1 + x2) / 2.0
        head_height = min(person_height * 0.32, head_width * 1.25)
        box = (
            center_x - head_width / 2.0,
            y1,
            center_x + head_width / 2.0,
            y1 + head_height,
        )

    x1 = max(0.0, min(float(width), box[0]))
    y1 = max(0.0, min(float(height), box[1]))
    x2 = max(0.0, min(float(width), box[2]))
    y2 = max(0.0, min(float(height), box[3]))
    return [x1, y1, x2, y2] if x2 > x1 and y2 > y1 else None


def _pose_context(pose: Optional[dict], frame_shape) -> dict:
    if not pose:
        return {
            "scale": operator_scale(None, frame_shape),
            "keypoints": {},
            "hand_points": [],
            "head_points": [],
            "head_roi": None,
            "wrist_near_head": False,
            "wrist_near_head_name": None,
            "person_bbox": None,
            "shoulder_y": None,
        }
    keypoints = pose.get("keypoints", pose)
    scale = operator_scale(pose, frame_shape)
    hand_points = [
        (name, _point(keypoints, name))
        for name in ("left_wrist", "right_wrist")
    ]
    head_points = [
        (name, _point(keypoints, name))
        for name in ("left_ear", "right_ear", "nose", "left_eye", "right_eye")
    ]
    head_roi = build_head_roi(pose, frame_shape)
    wrist_near_head = False
    wrist_near_head_name = None
    if head_roi is not None:
        ranked_wrists = []
        for name, point in hand_points:
            if point is None:
                continue
            distance = point_to_bbox_distance(point, head_roi) / scale
            ranked_wrists.append((distance, name))
        if ranked_wrists:
            wrist_distance, wrist_name = min(ranked_wrists)
            wrist_near_head = wrist_distance <= 0.18
            wrist_near_head_name = wrist_name if wrist_near_head else None
    shoulders = [
        _point(keypoints, name) for name in ("left_shoulder", "right_shoulder")
    ]
    shoulders = [point for point in shoulders if point is not None]
    shoulder_y = (
        sum(point[1] for point in shoulders) / len(shoulders)
        if shoulders
        else None
    )
    return {
        "scale": scale,
        "keypoints": keypoints,
        "hand_points": hand_points,
        "head_points": head_points,
        "head_roi": head_roi,
        "wrist_near_head": wrist_near_head,
        "wrist_near_head_name": wrist_near_head_name,
        "person_bbox": pose.get("bbox"),
        "shoulder_y": shoulder_y,
    }


def _debug_defaults(phone_count: int, context: dict) -> dict:
    return {
        "phone_count": phone_count,
        "phone_near_head": False,
        "phone_near_head_roi": False,
        "phone_near_hand": False,
        "phone_in_viewing_region": False,
        "phone_to_head_distance": None,
        "phone_to_head_roi_distance": None,
        "phone_to_hand_distance": None,
        "nearest_head_point": None,
        "nearest_head_name": None,
        "nearest_hand_point": None,
        "nearest_hand_name": None,
        "operator_scale": context["scale"],
        "head_roi": context["head_roi"],
        "inside_body_region": False,
        "below_shoulders": None,
        "wrist_overlaps_phone": False,
        "wrist_near_head": context["wrist_near_head"],
        "wrist_near_head_name": context["wrist_near_head_name"],
        "call_evidence": 0.0,
    }


def _empty_result(phone_detections, context: dict) -> Dict:
    selected = (
        max(phone_detections, key=lambda item: float(item.get("confidence", 0.0)))
        if phone_detections
        else None
    )
    return {
        "using_phone": False,
        "behavior": PHONE_PRESENT if selected is not None else NORMAL,
        "score": 0.0,
        "phone_confidence": float(selected.get("confidence", 0.0)) if selected else 0.0,
        "phone": selected,
        "pathways": {
            PHONE_CALL: False,
            HANDHELD_PHONE_USE: False,
            WATCHING_PHONE: False,
        },
        "debug": _debug_defaults(len(phone_detections), context),
    }


def _inside_viewing_region(center, head_roi, scale: float, person_bbox) -> bool:
    if head_roi is None:
        return False
    viewing_box = (
        head_roi[0] - scale * 0.32,
        head_roi[3] - scale * 0.05,
        head_roi[2] + scale * 0.32,
        head_roi[3] + scale * 1.45,
    )
    if not is_point_inside_bbox(center, viewing_box):
        return False
    return person_bbox is None or is_point_inside_bbox(
        center, person_bbox, margin=scale * 0.15
    )


def classify_phone_usage(
    phone_detections,
    pose: Optional[dict],
    frame_shape,
    settings: Optional[Settings] = None,
    tracked_phone_bbox=None,
) -> Dict:
    """Classify independent call, handheld, and watching evidence for one frame."""
    settings = settings or Settings()
    context = _pose_context(pose, frame_shape)
    if not phone_detections or not pose:
        return _empty_result(phone_detections, context)

    scale = context["scale"]
    hand_points = context["hand_points"]
    head_points = context["head_points"]
    head_roi = context["head_roi"]
    person_bbox = context["person_bbox"]
    shoulder_y = context["shoulder_y"]
    candidates = []

    for phone in phone_detections:
        center = tuple(phone.get("center") or bbox_center(phone["bbox"]))
        hand_distance, hand_name, hand_point = _nearest(center, hand_points, scale)
        head_distance, head_name, head_point = _nearest(center, head_points, scale)
        wrist_overlap = any(
            point is not None
            and is_point_inside_bbox(point, phone["bbox"], margin=2.0)
            for _, point in hand_points
        )
        near_hand = wrist_overlap or (
            hand_distance is not None
            and hand_distance < settings.hand_phone_distance_threshold
        )

        head_roi_distance = (
            point_to_bbox_distance(center, head_roi) / scale
            if head_roi is not None
            else None
        )
        overlaps_head_roi = bool(
            head_roi is not None and _boxes_overlap(phone["bbox"], head_roi)
        )
        near_head_roi = overlaps_head_roi or (
            head_roi_distance is not None
            and head_roi_distance <= settings.head_phone_distance_threshold * 0.35
        )
        near_head_keypoint = bool(
            head_distance is not None
            and head_distance < settings.head_phone_distance_threshold
        )
        near_head = near_head_roi or near_head_keypoint

        inside_body = person_bbox is not None and is_point_inside_bbox(center, person_bbox)
        below_shoulders = shoulder_y is None or center[1] > shoulder_y
        body_ok = inside_body or not settings.require_phone_in_body_region
        vertical_ok = below_shoulders or not settings.require_phone_below_shoulders
        handheld = near_hand and body_ok and vertical_ok and not near_head
        watching = (
            not near_head
            and body_ok
            and _inside_viewing_region(center, head_roi, scale, person_bbox)
        )

        if near_head:
            behavior = PHONE_CALL
        elif handheld:
            behavior = HANDHELD_PHONE_USE
        elif watching:
            behavior = WATCHING_PHONE
        else:
            behavior = PHONE_PRESENT

        head_proximity = 0.0
        if near_head:
            roi_proximity = (
                1.0
                if overlaps_head_roi
                else max(
                    0.0,
                    1.0
                    - float(head_roi_distance or 0.0)
                    / max(settings.head_phone_distance_threshold * 0.35, 1e-6),
                )
            )
            point_proximity = (
                max(
                    0.0,
                    1.0
                    - float(head_distance)
                    / max(settings.head_phone_distance_threshold, 1e-6),
                )
                if head_distance is not None
                else 0.0
            )
            head_proximity = max(roi_proximity, point_proximity)
        hand_proximity = (
            max(
                0.0,
                1.0
                - float(hand_distance)
                / max(settings.hand_phone_distance_threshold, 1e-6),
            )
            if hand_distance is not None
            else float(wrist_overlap)
        )
        confidence = float(phone["confidence"])
        call_evidence = (
            min(1.0, 0.70 * confidence + 0.30 * head_proximity)
            if near_head
            else 0.0
        )
        pathway_score = {
            PHONE_CALL: call_evidence,
            WATCHING_PHONE: confidence * 0.85,
            HANDHELD_PHONE_USE: confidence * (0.65 + 0.35 * hand_proximity),
            PHONE_PRESENT: 0.0,
        }[behavior]
        tracking_iou = (
            bbox_iou(phone["bbox"], tracked_phone_bbox)
            if tracked_phone_bbox is not None
            else 0.0
        )
        candidates.append(
            {
                "using_phone": is_using_phone_behavior(behavior),
                "behavior": behavior,
                "score": pathway_score,
                "phone_confidence": confidence,
                "phone": phone,
                "pathways": {
                    PHONE_CALL: near_head,
                    HANDHELD_PHONE_USE: handheld,
                    WATCHING_PHONE: watching,
                },
                "debug": {
                    "phone_count": len(phone_detections),
                    "phone_near_head": near_head,
                    "phone_near_head_roi": near_head_roi,
                    "phone_near_hand": near_hand,
                    "phone_in_viewing_region": watching,
                    "phone_to_head_distance": head_distance,
                    "phone_to_head_roi_distance": head_roi_distance,
                    "phone_to_hand_distance": hand_distance,
                    "nearest_head_point": head_point,
                    "nearest_head_name": head_name,
                    "nearest_hand_point": hand_point,
                    "nearest_hand_name": hand_name,
                    "operator_scale": scale,
                    "head_roi": head_roi,
                    "inside_body_region": inside_body,
                    "below_shoulders": below_shoulders if shoulder_y is not None else None,
                    "wrist_overlaps_phone": wrist_overlap,
                    "wrist_near_head": context["wrist_near_head"],
                    "wrist_near_head_name": context["wrist_near_head_name"],
                    "call_evidence": call_evidence,
                    "tracking_iou": tracking_iou,
                },
            }
        )

    priority = {
        PHONE_PRESENT: 0,
        WATCHING_PHONE: 1,
        HANDHELD_PHONE_USE: 2,
        PHONE_CALL: 3,
    }
    return max(
        candidates,
        key=lambda item: (
            priority[item["behavior"]],
            item["debug"]["tracking_iou"],
            item["score"],
            item["phone_confidence"],
        ),
    )
