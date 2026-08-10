"""Readable, evidence-focused OpenCV visualization."""

from typing import Optional, Tuple


POSE_COLORS = {
    "left_wrist": (255, 180, 0),
    "right_wrist": (255, 180, 0),
    "left_elbow": (255, 210, 80),
    "right_elbow": (255, 210, 80),
    "left_ear": (200, 80, 255),
    "right_ear": (200, 80, 255),
}

POSE_EDGES = (
    ("left_eye", "right_eye"),
    ("nose", "left_eye"),
    ("nose", "right_eye"),
    ("left_eye", "left_ear"),
    ("right_eye", "right_ear"),
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
)


def _draw_pose_tracker(cv2, frame, pose):
    if not pose:
        return
    keypoints = pose.get("keypoints", pose)
    stale = bool(pose.get("tracking_stale", False))
    edge_color = (130, 130, 130) if stale else (80, 220, 255)
    for start_name, end_name in POSE_EDGES:
        start = keypoints.get(start_name)
        end = keypoints.get(end_name)
        if start is None or end is None:
            continue
        cv2.line(
            frame,
            (int(round(start[0])), int(round(start[1]))),
            (int(round(end[0])), int(round(end[1]))),
            edge_color,
            2,
            cv2.LINE_AA,
        )
    for name, value in keypoints.items():
        x, y = int(round(value[0])), int(round(value[1]))
        color = (150, 150, 150) if stale else POSE_COLORS.get(name, (80, 220, 255))
        radius = 5 if name in POSE_COLORS else 4
        cv2.circle(frame, (x, y), radius, color, -1, cv2.LINE_AA)


def _format_distance(value) -> str:
    return "n/a" if value is None else f"{value:.2f}"


def draw_annotations(
    frame,
    phone_detections,
    pose,
    instant,
    temporal,
    processing_fps: float,
    event_duration: float,
    roi: Optional[Tuple[int, int, int, int]] = None,
    phone_search_rois=None,
    raw_phone_candidates=None,
    debug_phone_detection: bool = False,
    show_phone_search_rois: bool = False,
    display_mode: str = "debug",
    show_pose: bool = True,
    show_debug_lines: bool = True,
):
    import cv2
    state_color = (0, 50, 230) if temporal.state == "USING_PHONE" else (50, 190, 60)
    if display_mode == "phone_only":
        if show_pose:
            _draw_pose_tracker(cv2, frame, pose)
        for phone in phone_detections:
            x1, y1, x2, y2 = (int(round(value)) for value in phone["bbox"])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 210, 255), 3)
            cv2.putText(
                frame,
                f"cell phone {phone['confidence']:.2f}",
                (x1, max(22, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 210, 255),
                2,
                cv2.LINE_AA,
            )
        selected_phone = instant.get("phone")
        debug = instant.get("debug", {})
        evidence_text = None
        evidence_point = None
        evidence_color = (60, 230, 80)
        if selected_phone is not None and debug.get("phone_near_head"):
            evidence_text = "Evidence: phone near head"
            evidence_point = debug.get("nearest_head_point")
            evidence_color = (200, 80, 255)
        elif selected_phone is not None and debug.get("phone_near_hand"):
            evidence_text = "Evidence: phone near wrist"
            evidence_point = debug.get("nearest_hand_point")
        if evidence_point is not None:
            phone_center = tuple(int(round(value)) for value in selected_phone["center"])
            target = tuple(int(round(value)) for value in evidence_point)
            cv2.line(frame, phone_center, target, evidence_color, 3, cv2.LINE_AA)
        height, width = frame.shape[:2]
        overlay = frame.copy()
        bar_height = max(72, min(105, height // 7))
        cv2.rectangle(overlay, (0, 0), (width, bar_height), (12, 12, 12), -1)
        frame[:] = cv2.addWeighted(overlay, 0.72, frame, 0.28, 0)
        cv2.putText(
            frame,
            temporal.state,
            (24, int(bar_height * 0.68)),
            cv2.FONT_HERSHEY_SIMPLEX,
            max(0.8, min(1.4, width / 900.0)),
            state_color,
            3,
            cv2.LINE_AA,
        )
        if pose is not None and pose.get("track_id") is not None:
            tracking_status = str(pose.get("tracking_status", "tracked")).upper()
            missed_frames = int(pose.get("tracking_missed_frames", 0))
            tracking_text = f"Driver #{pose['track_id']}  {tracking_status}"
            if pose.get("tracking_stale", False) and missed_frames:
                tracking_text += f" ({missed_frames})"
            tracking_color = (
                (150, 150, 150)
                if pose.get("tracking_stale", False)
                else (80, 220, 255)
            )
            cv2.putText(
                frame,
                tracking_text,
                (25, max(18, bar_height - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                tracking_color,
                1,
                cv2.LINE_AA,
            )
        if evidence_text:
            cv2.putText(
                frame,
                evidence_text,
                (24, min(height - 15, bar_height + 30)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                evidence_color,
                2,
                cv2.LINE_AA,
            )
        fps_text = f"FPS {processing_fps:.1f}"
        (text_width, _), _ = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
        cv2.putText(
            frame,
            fps_text,
            (max(10, width - text_width - 24), int(bar_height * 0.62)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (235, 235, 235),
            2,
            cv2.LINE_AA,
        )
        return frame
    if roi is not None:
        x1, y1, x2, y2 = roi
        cv2.rectangle(frame, (x1, y1), (x2, y2), (160, 160, 160), 1)
        cv2.putText(frame, "DRIVER ROI", (x1 + 5, max(15, y1 + 18)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

    if show_phone_search_rois:
        roi_colors = {
            "left_wrist_roi": (255, 150, 20),
            "right_wrist_roi": (255, 210, 20),
            "both_hands_roi": (40, 210, 255),
            "head_roi": (210, 80, 255),
        }
        for search_roi in phone_search_rois or []:
            x1, y1, x2, y2 = search_roi["bbox"]
            color = roi_colors.get(search_roi["source"], (180, 180, 180))
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
            cv2.putText(
                frame,
                search_roi["source"],
                (x1 + 3, min(frame.shape[0] - 5, y1 + 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                color,
                1,
                cv2.LINE_AA,
            )

    if debug_phone_detection:
        for phone in raw_phone_candidates or []:
            x1, y1, x2, y2 = (int(round(value)) for value in phone["bbox"])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (130, 130, 255), 1)
            cv2.putText(
                frame,
                f"LOW {phone['confidence']:.3f} [{phone.get('source', 'unknown')}]",
                (x1, max(18, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (130, 130, 255),
                1,
                cv2.LINE_AA,
            )

    selected_phone = instant.get("phone")
    for phone in phone_detections:
        x1, y1, x2, y2 = (int(round(value)) for value in phone["bbox"])
        is_selected = phone is selected_phone
        color = state_color if is_selected and instant["using_phone"] else (0, 210, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        source_text = f" [{phone.get('source', 'unknown')}]" if debug_phone_detection else ""
        cv2.putText(
            frame,
            f"cell phone {phone['confidence']:.2f}{source_text}",
            (x1, max(18, y1 - 7)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
            cv2.LINE_AA,
        )

    if show_pose:
        _draw_pose_tracker(cv2, frame, pose)

    debug = instant.get("debug", {})
    if show_debug_lines and selected_phone is not None:
        phone_center = tuple(int(round(value)) for value in selected_phone["center"])
        hand_point = debug.get("nearest_hand_point")
        head_point = debug.get("nearest_head_point")
        if hand_point is not None:
            target = tuple(int(round(value)) for value in hand_point)
            cv2.line(frame, phone_center, target, (255, 180, 0), 2, cv2.LINE_AA)
        if head_point is not None:
            target = tuple(int(round(value)) for value in head_point)
            cv2.line(frame, phone_center, target, (200, 80, 255), 2, cv2.LINE_AA)

    height, width = frame.shape[:2]
    panel_width = min(500, max(350, width - 20))
    panel_height = min(270, max(200, height - 20))
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (10 + panel_width, 10 + panel_height), (12, 12, 12), -1)
    frame[:] = cv2.addWeighted(overlay, 0.72, frame, 0.28, 0)

    cv2.putText(frame, "R6.2 Forklift Phone Detection", (25, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.67, (245, 245, 245), 2, cv2.LINE_AA)
    cv2.putText(frame, temporal.state, (25, 85), cv2.FONT_HERSHEY_SIMPLEX, 1.15, state_color, 3, cv2.LINE_AA)
    lines = [
        f"Behavior: {temporal.behavior}",
        f"Phone conf: {instant.get('phone_confidence', 0.0):.2f}",
        f"Phone-Hand: {_format_distance(debug.get('phone_to_hand_distance'))}",
        f"Phone-Head: {_format_distance(debug.get('phone_to_head_distance'))}",
        f"Usage ratio: {temporal.usage_ratio:.2f} "
        f"({temporal.covered_seconds:.1f}/{temporal.window_seconds:.1f} s)",
        f"Duration: {event_duration:.1f} s    FPS: {processing_fps:.1f}",
    ]
    y = 116
    for line in lines:
        cv2.putText(frame, line, (25, y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (235, 235, 235), 1, cv2.LINE_AA)
        y += 25
    return frame
