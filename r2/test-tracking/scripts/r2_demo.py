"""R2 forklift slowdown PoC for test1.mp4.

Pipeline:
  best_fresh.pt -> YOLO one-to-many/NMS -> tuned ByteTrack -> temporal
  confirmation + geometry/dedup filters -> ROI -> smoothed trajectory/speed
  -> Approach Zone -> Intersection Zone -> R2 decision

Speed is image-plane px/s (PoC), not km/h.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Deque, Dict, List, Optional, Sequence, Tuple
import csv
import json
import math
import tempfile
import time

import cv2
import numpy as np
import torch
from ultralytics import YOLO

# =============================================================================
# PATHS
# =============================================================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "best_fresh.pt"
VIDEO_PATH = PROJECT_ROOT / "video" / "test2.mp4"
CONFIG_DIR = PROJECT_ROOT / "configs"
ZONE_CONFIG_PATH = CONFIG_DIR / "test2_r2_zones.json"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_VIDEO_PATH = OUTPUT_DIR / "r2_test2_demo.mp4"
OUTPUT_CSV_PATH = OUTPUT_DIR / "r2_test2_events.csv"

# =============================================================================
# DETECTOR / TRACKER -- TUNED FOR SMALL FORKLIFTS AND TEMPORARY OCCLUSION
# =============================================================================
CONF_THRESHOLD = 0.10
IOU_THRESHOLD = 0.55
IMAGE_SIZE = 1920
TRACK_LOW_THRESHOLD = 0.10
TRACK_HIGH_THRESHOLD = 0.25
NEW_TRACK_THRESHOLD = 0.30
TRACK_BUFFER_SECONDS = 2.0
MATCH_THRESHOLD = 0.85
FUSE_SCORE = True
DEVICE = 0
FP16_PRECISION = 16
PROGRESS_INTERVAL = 100
MAX_FRAMES: Optional[int] = None

# Post-tracker safeguards.  The model occasionally returns a large rack/wall
# box or two overlapping IDs for the same forklift.
MAX_BOX_AREA_RATIO = 0.12
TOP_CORNER_REJECTION_AREA_RATIO = 0.08
TRACK_DEDUPLICATION_IOU = 0.65
TRACK_DEDUPLICATION_IOS = 0.80
TRACK_CONTINUITY_IOU = 0.30
BOX_SMOOTHING_ALPHA = 0.65

# =============================================================================
# ZONES
# =============================================================================
FORCE_REDRAW_ZONES = False
NUM_APPROACH_ZONES = 2
USE_TRACKING_ROI = True

# =============================================================================
# TRAJECTORY / SPEED
# =============================================================================
TRAJECTORY_LENGTH = 45
TRACK_HISTORY_LENGTH = 180
SPEED_WINDOW_SECONDS = 0.40
MIN_SPEED_OBSERVATIONS = 4
SPEED_EMA_ALPHA = 0.30
MIN_VALID_SPEED_PX_S = 5.0

# Static false positives may be tracked. They are not allowed to trigger R2
# until they have demonstrated meaningful motion. Once confirmed, a track stays
# confirmed even if it later slows/stops.
ENABLE_MOTION_CONFIRMATION = True
CONFIRMATION_EVIDENCE_SECONDS = 0.50
MOTION_WINDOW_SECONDS = 2.0
MIN_MOTION_OBSERVATIONS = 6
MIN_NET_MOTION_PX = 30.0
MIN_MOTION_FRAME_DIAGONAL_RATIO = 0.02
VISUAL_MOTION_INTERVAL_SECONDS = 0.15
VISUAL_DIFFERENCE_THRESHOLD = 20
MIN_VISUAL_MOTION_RATIO = 0.10
STATIC_CONFIRMATION_SECONDS = 0.50
STATIC_CONFIDENCE_THRESHOLD = 0.65
SHOW_CANDIDATES = False

# A recovered ByteTrack ID may bridge a short occlusion.  A longer gap is
# treated as a fresh traversal so stale speed/zone history is never reused.
STATE_RESET_GAP_SECONDS = 2.25
STALE_STATE_SECONDS = 5.0

# =============================================================================
# R2 RULE
# =============================================================================
MIN_APPROACH_SPEED_SAMPLES = 5
MIN_INTERSECTION_SPEED_SAMPLES = 3
APPROACH_BASELINE_WINDOW = 15
MIN_BASELINE_SPEED_PX_S = 10.0
MAX_APPROACH_TO_INTERSECTION_SECONDS = 3.0
# Speed history is restarted exactly at intersection entry, so no additional
# warmup is required before collecting zone-only speed samples.
INTERSECTION_WARMUP_SECONDS = 0.0
INTERSECTION_EXIT_GRACE_SECONDS = 0.50
MAX_INTERSECTION_EVALUATION_SECONDS = 3.0
TRAVERSAL_RESET_SECONDS = 0.75
ZONE_DWELL_FRAMES = 2
MIN_SLOWDOWN_RATIO = 0.20
EVENT_BANNER_SECONDS = 3.0

# =============================================================================
# DISPLAY
# =============================================================================
DRAW_ZONES = True
DRAW_TRAJECTORY = True
DRAW_SPEED = True
DRAW_PANEL = True
ZONE_LINE_THICKNESS = 1
ZONE_LABEL_FONT_SCALE = 0.42
ZONE_LABEL_THICKNESS = 1
BOX_THICKNESS = 2
TRAJECTORY_THICKNESS = 1
FONT_SCALE = 0.43
FONT_THICKNESS = 1
PANEL_FONT_SCALE = 0.38
PANEL_FONT_THICKNESS = 1
PANEL_LINE_HEIGHT = 19
PANEL_PADDING_X = 10
PANEL_PADDING_Y = 8
PANEL_MARGIN = 10

# OpenCV BGR
COLOR_ROI = (255, 255, 0)
COLOR_APPROACH = (0, 255, 255)
COLOR_INTERSECTION = (0, 128, 255)
COLOR_CANDIDATE = (180, 180, 180)
COLOR_NORMAL = (0, 220, 0)
COLOR_EVALUATING = (0, 215, 255)
COLOR_SLOWED = (0, 220, 0)
COLOR_NO_SLOWDOWN = (0, 0, 255)

Point = Tuple[float, float]
IntPoint = Tuple[int, int]
Polygon = List[IntPoint]


@dataclass
class TrackState:
    track_id: int
    positions: Deque[Tuple[int, float, float]] = field(
        default_factory=lambda: deque(maxlen=TRACK_HISTORY_LENGTH)
    )
    speed_history: Deque[float] = field(
        default_factory=lambda: deque(maxlen=TRAJECTORY_LENGTH)
    )
    display_positions: Deque[Tuple[int, float, float]] = field(
        default_factory=lambda: deque(maxlen=TRAJECTORY_LENGTH)
    )
    confidences: Deque[float] = field(
        default_factory=lambda: deque(maxlen=TRACK_HISTORY_LENGTH)
    )
    visual_motion: Deque[float] = field(
        default_factory=lambda: deque(maxlen=TRACK_HISTORY_LENGTH)
    )
    approach_speeds: Deque[float] = field(default_factory=lambda: deque(maxlen=100))
    intersection_speeds: Deque[float] = field(default_factory=lambda: deque(maxlen=100))
    smoothed_speed_px_s: Optional[float] = None
    smoothed_box: Optional[np.ndarray] = None
    total_hits: int = 0
    last_seen_frame: int = 0
    detection_confirmed: bool = False
    confirmation_reason: Optional[str] = None
    motion_confirmed: bool = False
    approach_zone: Optional[int] = None
    has_visited_approach: bool = False
    entered_intersection: bool = False
    intersection_entry_frame: Optional[int] = None
    baseline_speed_px_s: Optional[float] = None
    intersection_speed_px_s: Optional[float] = None
    slowdown_ratio: Optional[float] = None
    decision: Optional[str] = None
    event_logged: bool = False
    last_approach_frame: Optional[int] = None
    last_intersection_frame: Optional[int] = None
    intersection_exit_frame: Optional[int] = None
    outside_zones_since: Optional[int] = None
    observed_zone: Optional[Tuple[str, int]] = None
    observed_zone_streak: int = 0
    last_observed_zone_frame: Optional[int] = None


class ZoneSelectionAborted(RuntimeError):
    pass


def bottom_center(box: Sequence[float]) -> Point:
    x1, y1, x2, y2 = box
    return float((x1 + x2) / 2.0), float(y2)


def point_in_polygon(point: Point, polygon: Polygon) -> bool:
    if len(polygon) < 3:
        return False
    contour = np.asarray(polygon, dtype=np.int32).reshape((-1, 1, 2))
    return cv2.pointPolygonTest(contour, point, False) >= 0


def point_in_tracking_area(point: Point, zones: Dict) -> bool:
    """Accept the ROI plus every R2 zone so configured events stay reachable."""
    if not USE_TRACKING_ROI:
        return True
    polygons = [
        zones["tracking_roi"],
        zones["intersection_zone"],
        *zones["approach_zones"],
    ]
    return any(point_in_polygon(point, polygon) for polygon in polygons)


def normalize_polygon(points: Sequence[Sequence[int]]) -> Polygon:
    """Remove duplicate closing/consecutive vertices and normalize to ints."""
    normalized: Polygon = []
    for raw_point in points:
        if len(raw_point) != 2:
            raise ValueError(f"Invalid polygon point: {raw_point!r}")
        point = (int(raw_point[0]), int(raw_point[1]))
        if not normalized or normalized[-1] != point:
            normalized.append(point)
    if (
        len(normalized) >= 2
        and math.hypot(
            normalized[0][0] - normalized[-1][0],
            normalized[0][1] - normalized[-1][1],
        )
        <= 4.0
    ):
        normalized.pop()
    return normalized


def polygon_self_intersects(polygon: Polygon) -> bool:
    """Return True when non-neighbouring polygon edges cross or touch."""

    def orientation(a: IntPoint, b: IntPoint, c: IntPoint) -> int:
        cross = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
        return 0 if cross == 0 else (1 if cross > 0 else -1)

    def on_segment(a: IntPoint, b: IntPoint, point: IntPoint) -> bool:
        return min(a[0], b[0]) <= point[0] <= max(a[0], b[0]) and min(
            a[1], b[1]
        ) <= point[1] <= max(a[1], b[1])

    def intersects(a: IntPoint, b: IntPoint, c: IntPoint, d: IntPoint) -> bool:
        first = orientation(a, b, c)
        second = orientation(a, b, d)
        third = orientation(c, d, a)
        fourth = orientation(c, d, b)
        if first != second and third != fourth:
            return True
        return (
            (first == 0 and on_segment(a, b, c))
            or (second == 0 and on_segment(a, b, d))
            or (third == 0 and on_segment(c, d, a))
            or (fourth == 0 and on_segment(c, d, b))
        )

    count = len(polygon)
    for first_index in range(count):
        first_start = polygon[first_index]
        first_end = polygon[(first_index + 1) % count]
        for second_index in range(first_index + 1, count):
            if second_index in {
                first_index,
                (first_index + 1) % count,
                (first_index - 1) % count,
            }:
                continue
            # Edge 0 and the closing edge are neighbours too.
            if first_index == 0 and second_index == count - 1:
                continue
            second_start = polygon[second_index]
            second_end = polygon[(second_index + 1) % count]
            if intersects(first_start, first_end, second_start, second_end):
                return True
    return False


def validate_polygon(polygon: Polygon, frame_shape: Sequence[int], name: str) -> None:
    if len(polygon) < 3 or len(set(polygon)) < 3:
        raise ValueError(f"{name} must contain at least 3 distinct points.")
    height, width = frame_shape[:2]
    outside = [
        point
        for point in polygon
        if not (0 <= point[0] < width and 0 <= point[1] < height)
    ]
    if outside:
        raise ValueError(f"{name} contains points outside {width}x{height}: {outside}")
    if polygon_self_intersects(polygon):
        raise ValueError(f"{name} self-intersects; redraw a simple polygon.")
    contour = np.asarray(polygon, dtype=np.float32).reshape((-1, 1, 2))
    if abs(float(cv2.contourArea(contour))) < 4.0:
        raise ValueError(f"{name} has near-zero area; redraw the polygon.")


def validate_zones(zones: Dict, frame_shape: Sequence[int]) -> None:
    validate_polygon(zones["tracking_roi"], frame_shape, "tracking_roi")
    approach_zones = zones["approach_zones"]
    if len(approach_zones) != NUM_APPROACH_ZONES:
        raise ValueError(
            f"Expected {NUM_APPROACH_ZONES} approach zones, got {len(approach_zones)}."
        )
    for idx, polygon in enumerate(approach_zones, start=1):
        validate_polygon(polygon, frame_shape, f"approach_zone_{idx}")
    validate_polygon(zones["intersection_zone"], frame_shape, "intersection_zone")


def print_zone_coverage_warnings(zones: Dict, frame_shape: Sequence[int]) -> None:
    """Warn about unreachable/overlapping pixels without changing user zones."""
    height, width = frame_shape[:2]

    def polygon_mask(polygon: Polygon) -> np.ndarray:
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(mask, [np.asarray(polygon, dtype=np.int32)], 1)
        return mask

    roi_mask = polygon_mask(zones["tracking_roi"])
    intersection_mask = polygon_mask(zones["intersection_zone"])
    warnings: List[str] = []
    for idx, polygon in enumerate(zones["approach_zones"], start=1):
        mask = polygon_mask(polygon)
        area = max(1, int(mask.sum()))
        outside_ratio = float(np.logical_and(mask, roi_mask == 0).sum()) / area
        overlap_ratio = float(np.logical_and(mask, intersection_mask).sum()) / area
        if outside_ratio > 0.01:
            warnings.append(
                f"approach {idx}: {outside_ratio * 100.0:.1f}% outside tracking ROI"
            )
        if overlap_ratio > 0.01:
            warnings.append(
                f"approach {idx}: {overlap_ratio * 100.0:.1f}% overlaps intersection"
            )
    intersection_area = max(1, int(intersection_mask.sum()))
    intersection_outside = (
        float(np.logical_and(intersection_mask, roi_mask == 0).sum())
        / intersection_area
    )
    if intersection_outside > 0.01:
        warnings.append(
            f"intersection: {intersection_outside * 100.0:.1f}% outside tracking ROI"
        )
    if warnings:
        print(
            "WARNING: R2 zones extend beyond the tracking ROI; "
            "the demo will track their union:"
        )
        for warning in warnings:
            print(f"  - {warning}")


def select_polygon(frame: np.ndarray, title: str, instruction: str, color) -> Polygon:
    """Left click add; Right/Enter/Space finish; U undo; R reset; Q/Esc cancel."""
    points: Polygon = []
    done = False
    status = ""
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)

    def mouse(event, x, y, flags, param):
        nonlocal done, status
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
            status = ""
        elif event == cv2.EVENT_RBUTTONDOWN:
            candidate = normalize_polygon(points)
            try:
                validate_polygon(candidate, frame.shape, title)
            except ValueError as error:
                status = str(error)
            else:
                done = True

    cv2.setMouseCallback(title, mouse)

    while not done:
        canvas = frame.copy()
        for p in points:
            cv2.circle(canvas, p, 5, color, -1)
        if len(points) >= 2:
            cv2.polylines(
                canvas,
                [np.asarray(points, dtype=np.int32).reshape((-1, 1, 2))],
                False,
                color,
                ZONE_LINE_THICKNESS,
                cv2.LINE_AA,
            )

        overlay = canvas.copy()
        cv2.rectangle(overlay, (0, 0), (min(1100, canvas.shape[1]), 115), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.68, canvas, 0.32, 0, canvas)
        cv2.putText(
            canvas,
            instruction,
            (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            "Left:add | Right/ENTER/SPACE:finish | U:undo | R:reset | Q/ESC:cancel",
            (15, 62),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.49,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            status or f"Points: {len(points)}",
            (15, 92),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.49,
            (80, 80, 255) if status else (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.imshow(title, canvas)
        key = cv2.waitKey(20) & 0xFF
        try:
            window_visible = cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE)
        except cv2.error as error:
            raise ZoneSelectionAborted(title) from error
        if window_visible < 1:
            cv2.destroyAllWindows()
            raise ZoneSelectionAborted(title)

        if key in (10, 13, 32):
            candidate = normalize_polygon(points)
            try:
                validate_polygon(candidate, frame.shape, title)
            except ValueError as error:
                status = str(error)
            else:
                done = True
        elif key in (8, 127, ord("u"), ord("U")):
            if points:
                points.pop()
            status = ""
        elif key in (ord("r"), ord("R")):
            points.clear()
            status = "Polygon reset."
        elif key in (ord("q"), ord("Q"), 27):
            cv2.destroyAllWindows()
            raise ZoneSelectionAborted(title)

    cv2.destroyWindow(title)
    return normalize_polygon(points)


def create_zones(first_frame: np.ndarray) -> Dict:
    print("\nDraw R2 zones on the first frame.")
    tracking_roi = select_polygon(
        first_frame,
        "1 - Tracking ROI",
        "Draw the forklift road/operating region. Exclude pallet/rack areas where possible.",
        COLOR_ROI,
    )

    approach_zones = []
    for idx in range(NUM_APPROACH_ZONES):
        approach_zones.append(
            select_polygon(
                first_frame,
                f"2.{idx + 1} - Approach Zone {idx + 1}",
                f"Draw approach zone {idx + 1}: road segment immediately before the intersection.",
                COLOR_APPROACH,
            )
        )

    intersection_zone = select_polygon(
        first_frame,
        "3 - Intersection Zone",
        "Draw the blind-corner intersection area where slowdown is evaluated.",
        COLOR_INTERSECTION,
    )

    zones = {
        "tracking_roi": tracking_roi,
        "approach_zones": approach_zones,
        "intersection_zone": intersection_zone,
    }
    validate_zones(zones, first_frame.shape)
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "frame_size": [int(first_frame.shape[1]), int(first_frame.shape[0])],
        **zones,
    }
    ZONE_CONFIG_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved zones: {ZONE_CONFIG_PATH}")
    return zones


def load_or_create_zones(first_frame: np.ndarray) -> Dict:
    if ZONE_CONFIG_PATH.exists() and not FORCE_REDRAW_ZONES:
        raw = json.loads(ZONE_CONFIG_PATH.read_text(encoding="utf-8"))
        saved_size = raw.get("frame_size")
        current_size = [int(first_frame.shape[1]), int(first_frame.shape[0])]
        if saved_size is None:
            print(
                "WARNING: legacy zone config has no frame_size metadata; "
                f"validating coordinates against current {current_size[0]}x"
                f"{current_size[1]} frame. Redraw if the source video changed."
            )
        elif list(saved_size) != current_size:
            raise ValueError(
                f"Zone config was created for {saved_size}, but video is "
                f"{current_size}. Set FORCE_REDRAW_ZONES=True and redraw."
            )
        zones = {
            "tracking_roi": normalize_polygon(raw["tracking_roi"]),
            "approach_zones": [
                normalize_polygon(poly) for poly in raw["approach_zones"]
            ],
            "intersection_zone": normalize_polygon(raw["intersection_zone"]),
        }
        validate_zones(zones, first_frame.shape)
        print(f"Loaded zones: {ZONE_CONFIG_PATH}")
        print_zone_coverage_warnings(zones, first_frame.shape)
        return zones
    zones = create_zones(first_frame)
    print_zone_coverage_warnings(zones, first_frame.shape)
    return zones


def get_forklift_class_id(model: YOLO) -> int:
    names = model.names
    class_map = (
        {int(k): str(v) for k, v in names.items()}
        if isinstance(names, dict)
        else {i: str(v) for i, v in enumerate(names)}
    )
    matches = [
        i for i, name in class_map.items() if name.strip().casefold() == "forklift"
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one 'forklift' class. Classes: {class_map}"
        )
    return matches[0]


def write_tracker_config(path: Path, fps: float) -> int:
    """Create the FPS-aware ByteTrack configuration used by this demo."""
    buffer_frames = max(1, int(round(TRACK_BUFFER_SECONDS * fps)))
    path.write_text(
        "\n".join(
            (
                "tracker_type: bytetrack",
                f"track_high_thresh: {TRACK_HIGH_THRESHOLD}",
                f"track_low_thresh: {TRACK_LOW_THRESHOLD}",
                f"new_track_thresh: {NEW_TRACK_THRESHOLD}",
                f"track_buffer: {buffer_frames}",
                f"match_thresh: {MATCH_THRESHOLD}",
                f"fuse_score: {str(FUSE_SCORE).lower()}",
                "",
            )
        ),
        encoding="utf-8",
    )
    return buffer_frames


def box_iou(first: Sequence[float], second: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = first
    bx1, by1, bx2, by2 = second
    intersection = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(
        0.0, min(ay2, by2) - max(ay1, by1)
    )
    first_area = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    second_area = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = first_area + second_area - intersection
    return intersection / union if union > 0 else 0.0


def box_intersection_over_smaller(
    first: Sequence[float], second: Sequence[float]
) -> float:
    ax1, ay1, ax2, ay2 = first
    bx1, by1, bx2, by2 = second
    intersection = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(
        0.0, min(ay2, by2) - max(ay1, by1)
    )
    first_area = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    second_area = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    smaller = min(first_area, second_area)
    return intersection / smaller if smaller > 0 else 0.0


def select_track_detections(
    track_data: np.ndarray,
    states: Dict[int, TrackState],
    frame_shape: Sequence[int],
    forklift_class_id: int,
    zones: Dict,
) -> Tuple[List[np.ndarray], int, int]:
    """Filter implausible boxes and suppress duplicate tracked IDs."""
    frame_height, frame_width = frame_shape[:2]
    frame_area = float(frame_height * frame_width)
    candidates: List[np.ndarray] = []
    geometry_rejected = 0

    for detection in track_data:
        box = detection[:4]
        if int(detection[5]) != forklift_class_id:
            continue
        box_width = max(0.0, float(box[2] - box[0]))
        box_height = max(0.0, float(box[3] - box[1]))
        area_ratio = box_width * box_height / frame_area
        touches_top = float(box[1]) <= 0.02 * frame_height
        touches_side = (
            float(box[0]) <= 0.02 * frame_width or float(box[2]) >= 0.98 * frame_width
        )
        implausible = area_ratio > MAX_BOX_AREA_RATIO or (
            area_ratio >= TOP_CORNER_REJECTION_AREA_RATIO
            and touches_top
            and touches_side
        )
        point = bottom_center(box)
        outside_tracking_area = not point_in_tracking_area(point, zones)
        if implausible or outside_tracking_area:
            geometry_rejected += 1
            continue
        candidates.append(detection)

    def priority(detection: np.ndarray) -> Tuple[int, int, int, int, int, float]:
        state = states.get(int(detection[6]))
        return (
            int(state.motion_confirmed) if state is not None else 0,
            int(float(detection[4]) >= TRACK_HIGH_THRESHOLD),
            int(state.detection_confirmed) if state is not None else 0,
            state.last_seen_frame if state is not None else 0,
            state.total_hits if state is not None else 0,
            float(detection[4]),
        )

    selected: List[np.ndarray] = []
    duplicates_rejected = 0
    for detection in sorted(candidates, key=priority, reverse=True):
        if any(
            box_iou(detection[:4], kept[:4]) >= TRACK_DEDUPLICATION_IOU
            or box_intersection_over_smaller(detection[:4], kept[:4])
            >= TRACK_DEDUPLICATION_IOS
            for kept in selected
        ):
            duplicates_rejected += 1
            continue
        selected.append(detection)
    return selected, geometry_rejected, duplicates_rejected


def visual_motion_ratio(
    change_mask: Optional[np.ndarray], box: Sequence[float]
) -> float:
    if change_mask is None:
        return 0.0
    height, width = change_mask.shape[:2]
    x1, y1, x2, y2 = [int(round(value)) for value in box]
    x1, x2 = max(0, min(x1, width)), max(0, min(x2, width))
    y1, y2 = max(0, min(y1, height)), max(0, min(y2, height))
    if x2 <= x1 or y2 <= y1:
        return 0.0
    return float(np.mean(change_mask[y1:y2, x1:x2]))


def robust_net_motion(
    positions: Sequence[Tuple[int, float, float]], window_frames: int
) -> float:
    recent = list(positions)[-window_frames:]
    if len(recent) < 2:
        return 0.0
    edge = min(3, max(1, len(recent) // 3))
    sx = float(np.median([p[1] for p in recent[:edge]]))
    sy = float(np.median([p[2] for p in recent[:edge]]))
    ex = float(np.median([p[1] for p in recent[-edge:]]))
    ey = float(np.median([p[2] for p in recent[-edge:]]))
    return math.hypot(ex - sx, ey - sy)


def reset_r2_cycle(state: TrackState) -> None:
    state.approach_speeds.clear()
    state.intersection_speeds.clear()
    state.approach_zone = None
    state.has_visited_approach = False
    state.entered_intersection = False
    state.intersection_entry_frame = None
    state.baseline_speed_px_s = None
    state.intersection_speed_px_s = None
    state.slowdown_ratio = None
    state.decision = None
    state.event_logged = False
    state.last_approach_frame = None
    state.last_intersection_frame = None
    state.intersection_exit_frame = None
    state.outside_zones_since = None


def reset_track_history(state: TrackState) -> None:
    state.positions.clear()
    state.display_positions.clear()
    state.speed_history.clear()
    state.confidences.clear()
    state.visual_motion.clear()
    state.smoothed_speed_px_s = None
    state.smoothed_box = None
    state.total_hits = 0
    state.last_seen_frame = 0
    state.detection_confirmed = False
    state.confirmation_reason = None
    state.motion_confirmed = False
    state.observed_zone = None
    state.observed_zone_streak = 0
    state.last_observed_zone_frame = None
    reset_r2_cycle(state)


def reset_short_gap_history(state: TrackState, frame_idx: int) -> None:
    """Keep identity/R2 phase, but never bridge speed through an occlusion."""
    state.positions.clear()
    state.speed_history.clear()
    state.confidences.clear()
    state.visual_motion.clear()
    state.smoothed_speed_px_s = None
    if state.entered_intersection:
        state.intersection_speeds.clear()
        state.intersection_entry_frame = frame_idx


def restart_speed_at_intersection_entry(state: TrackState) -> None:
    """Keep the entry point but remove approach-speed memory from the EMA."""
    entry_observation = state.positions[-1] if state.positions else None
    state.positions.clear()
    if entry_observation is not None:
        state.positions.append(entry_observation)
    state.speed_history.clear()
    state.smoothed_speed_px_s = None


def update_motion_confirmation(
    state: TrackState, fps: float, frame_diagonal: float
) -> None:
    if not ENABLE_MOTION_CONFIRMATION:
        state.detection_confirmed = True
        state.motion_confirmed = True
        state.confirmation_reason = "immediate"
        return

    evidence_hits = max(
        MIN_MOTION_OBSERVATIONS,
        int(math.ceil(CONFIRMATION_EVIDENCE_SECONDS * fps)),
    )
    motion_window = max(
        evidence_hits,
        int(math.ceil(MOTION_WINDOW_SECONDS * fps)),
    )
    if not state.motion_confirmed and len(state.positions) >= evidence_hits:
        motion_threshold = max(
            MIN_NET_MOTION_PX,
            frame_diagonal * MIN_MOTION_FRAME_DIAGONAL_RATIO,
        )
        visual_evidence = list(state.visual_motion)[-evidence_hits:]
        median_visual_motion = (
            float(np.median(visual_evidence)) if visual_evidence else 0.0
        )
        if (
            robust_net_motion(state.positions, motion_window) >= motion_threshold
            and median_visual_motion >= MIN_VISUAL_MOTION_RATIO
        ):
            state.motion_confirmed = True
            state.detection_confirmed = True
            state.confirmation_reason = "motion"

    static_hits = max(1, int(math.ceil(STATIC_CONFIRMATION_SECONDS * fps)))
    if not state.detection_confirmed and len(state.confidences) >= static_hits:
        recent_confidences = list(state.confidences)[-static_hits:]
        if float(np.median(recent_confidences)) >= STATIC_CONFIDENCE_THRESHOLD:
            state.detection_confirmed = True
            state.confirmation_reason = "strong-static"


def update_track_observation(
    state: TrackState,
    box: Sequence[float],
    confidence: float,
    current_visual_motion: float,
    frame_idx: int,
    fps: float,
    frame_diagonal: float,
) -> Optional[Point]:
    reset_gap = max(1, int(round(STATE_RESET_GAP_SECONDS * fps)))
    speed_gap = max(
        1,
        int(round(SPEED_WINDOW_SECONDS * fps)),
    )
    accepted_gap = frame_idx - state.last_seen_frame if state.last_seen_frame else 0
    if accepted_gap > reset_gap:
        reset_track_history(state)
        accepted_gap = 0

    raw_box = np.asarray(box, dtype=np.float32)
    overlaps_previous = (
        state.smoothed_box is not None
        and box_iou(raw_box, state.smoothed_box) >= TRACK_CONTINUITY_IOU
    )
    healthy = (
        not state.detection_confirmed
        or confidence >= TRACK_HIGH_THRESHOLD
        or current_visual_motion >= MIN_VISUAL_MOTION_RATIO
        or overlaps_previous
    )
    if state.smoothed_box is None:
        state.smoothed_box = raw_box.copy()
    elif healthy:
        state.smoothed_box = (
            BOX_SMOOTHING_ALPHA * raw_box
            + (1.0 - BOX_SMOOTHING_ALPHA) * state.smoothed_box
        )

    # Once confirmed, ignore a weak/background-drift observation instead of
    # converting it into a false stop or an artificial trajectory jump.
    if not healthy or state.smoothed_box is None:
        return None

    if accepted_gap > speed_gap:
        reset_short_gap_history(state, frame_idx)
    state.last_seen_frame = frame_idx
    state.total_hits += 1
    state.confidences.append(confidence)
    state.visual_motion.append(current_visual_motion)
    point = bottom_center(state.smoothed_box)
    state.positions.append((frame_idx, point[0], point[1]))
    update_motion_confirmation(state, fps, frame_diagonal)
    return point


def update_speed(state: TrackState, fps: float) -> Optional[float]:
    if len(state.positions) < MIN_SPEED_OBSERVATIONS:
        return None

    current_frame = state.positions[-1][0]
    window_frames = max(
        MIN_SPEED_OBSERVATIONS - 1,
        int(round(SPEED_WINDOW_SECONDS * fps)),
    )
    recent = [
        observation
        for observation in state.positions
        if observation[0] >= current_frame - window_frames
    ]
    if len(recent) < MIN_SPEED_OBSERVATIONS:
        return None

    edge = min(3, max(1, len(recent) // 3))
    start = recent[:edge]
    end = recent[-edge:]
    start_frame = float(np.median([item[0] for item in start]))
    end_frame = float(np.median([item[0] for item in end]))
    frame_delta = end_frame - start_frame
    if frame_delta <= 0:
        return None
    start_x = float(np.median([item[1] for item in start]))
    start_y = float(np.median([item[2] for item in start]))
    end_x = float(np.median([item[1] for item in end]))
    end_y = float(np.median([item[2] for item in end]))
    dt = float(frame_delta) / fps
    distance = math.hypot(end_x - start_x, end_y - start_y)
    raw = distance / dt
    if raw < MIN_VALID_SPEED_PX_S:
        raw = 0.0
    if state.smoothed_speed_px_s is None:
        smoothed = raw
    else:
        smoothed = (
            SPEED_EMA_ALPHA * raw + (1.0 - SPEED_EMA_ALPHA) * state.smoothed_speed_px_s
        )
    state.smoothed_speed_px_s = smoothed
    state.speed_history.append(smoothed)
    return smoothed


def find_approach_zone(point: Point, zones: Dict) -> Optional[int]:
    for idx, poly in enumerate(zones["approach_zones"]):
        if point_in_polygon(point, poly):
            return idx
    return None


def stable_zone_observation(
    state: TrackState, point: Point, frame_idx: int, zones: Dict
) -> Optional[Tuple[str, int]]:
    """Use intersection precedence and require a short dwell at boundaries."""
    if point_in_polygon(point, zones["intersection_zone"]):
        observed = ("intersection", -1)
    else:
        approach_idx = find_approach_zone(point, zones)
        observed = (
            ("approach", approach_idx) if approach_idx is not None else ("outside", -1)
        )

    consecutive = state.last_observed_zone_frame == frame_idx - 1
    if observed == state.observed_zone and consecutive:
        state.observed_zone_streak += 1
    else:
        state.observed_zone = observed
        state.observed_zone_streak = 1
    state.last_observed_zone_frame = frame_idx
    return observed if state.observed_zone_streak >= ZONE_DWELL_FRAMES else None


def update_r2(
    state: TrackState,
    point: Point,
    speed: Optional[float],
    frame_idx: int,
    fps: float,
    zones: Dict,
) -> Optional[Dict]:
    zone = stable_zone_observation(state, point, frame_idx, zones)
    if zone is None:
        return None
    zone_kind, zone_index = zone

    approach_max_age = max(1, int(round(MAX_APPROACH_TO_INTERSECTION_SECONDS * fps)))
    exit_grace = max(1, int(round(INTERSECTION_EXIT_GRACE_SECONDS * fps)))
    traversal_reset = max(1, int(round(TRAVERSAL_RESET_SECONDS * fps)))
    intersection_warmup = max(0, int(round(INTERSECTION_WARMUP_SECONDS * fps)))
    intersection_timeout = max(1, int(round(MAX_INTERSECTION_EVALUATION_SECONDS * fps)))

    # Keep the result visible briefly, then allow the same ByteTrack ID to
    # start another valid Approach -> Intersection traversal.
    if state.decision is not None:
        if zone_kind == "approach":
            reset_r2_cycle(state)
        elif zone_kind == "outside":
            if state.outside_zones_since is None:
                state.outside_zones_since = frame_idx
            elif frame_idx - state.outside_zones_since >= traversal_reset:
                reset_r2_cycle(state)
            return None
        else:
            state.outside_zones_since = None
            return None

    if state.entered_intersection:
        if (
            state.intersection_entry_frame is not None
            and frame_idx - state.intersection_entry_frame > intersection_timeout
        ):
            reset_r2_cycle(state)
            return None
        if zone_kind == "intersection":
            state.last_intersection_frame = frame_idx
            state.intersection_exit_frame = None
            if (
                speed is not None
                and state.intersection_entry_frame is not None
                and frame_idx - state.intersection_entry_frame >= intersection_warmup
            ):
                # Zero is meaningful here: the forklift may have stopped.
                state.intersection_speeds.append(speed)
        else:
            if state.intersection_exit_frame is None:
                state.intersection_exit_frame = frame_idx
            elif frame_idx - state.intersection_exit_frame > exit_grace:
                reset_r2_cycle(state)
            return None

        if len(state.intersection_speeds) < MIN_INTERSECTION_SPEED_SAMPLES:
            return None
        samples = list(state.intersection_speeds)[:MIN_INTERSECTION_SPEED_SAMPLES]
        intersection_speed = float(np.median(samples))
        baseline = state.baseline_speed_px_s
        if baseline is None or baseline < MIN_BASELINE_SPEED_PX_S:
            # Never manufacture a 100% slowdown from a zero/near-zero baseline.
            reset_r2_cycle(state)
            return None

        state.intersection_speed_px_s = intersection_speed
        ratio = (baseline - intersection_speed) / baseline
        state.slowdown_ratio = ratio
        state.decision = "SLOWED" if ratio >= MIN_SLOWDOWN_RATIO else "NO_SLOWDOWN"
        return {
            "track_id": state.track_id,
            "approach_zone": (
                state.approach_zone + 1 if state.approach_zone is not None else ""
            ),
            "intersection_entry_frame": state.intersection_entry_frame,
            "approach_speed_px_s": round(baseline, 3),
            "intersection_speed_px_s": round(intersection_speed, 3),
            "slowdown_percent": round(ratio * 100.0, 2),
            "result": state.decision,
        }

    if zone_kind == "approach":
        if state.has_visited_approach and state.approach_zone != zone_index:
            reset_r2_cycle(state)
        state.approach_zone = zone_index
        state.has_visited_approach = True
        state.last_approach_frame = frame_idx
        state.outside_zones_since = None
        # Approach baseline represents actual travel, not stopped-box jitter.
        if speed is not None and speed >= MIN_VALID_SPEED_PX_S:
            state.approach_speeds.append(speed)
        return None

    if zone_kind == "outside":
        if (
            state.last_approach_frame is not None
            and frame_idx - state.last_approach_frame > approach_max_age
        ):
            reset_r2_cycle(state)
        return None

    # R2 is valid only for a recent, motion-confirmed Approach -> Intersection
    # transition with enough positive-speed baseline evidence.
    if not state.motion_confirmed or not state.has_visited_approach:
        return None
    if state.last_approach_frame is None:
        return None
    if frame_idx - state.last_approach_frame > approach_max_age:
        reset_r2_cycle(state)
        return None
    if len(state.approach_speeds) < MIN_APPROACH_SPEED_SAMPLES:
        return None

    recent = list(state.approach_speeds)[-APPROACH_BASELINE_WINDOW:]
    baseline = float(np.median(recent))
    if baseline < MIN_BASELINE_SPEED_PX_S:
        reset_r2_cycle(state)
        return None
    state.baseline_speed_px_s = baseline
    state.entered_intersection = True
    state.intersection_entry_frame = frame_idx
    state.last_intersection_frame = frame_idx
    state.intersection_exit_frame = None
    state.intersection_speeds.clear()
    restart_speed_at_intersection_entry(state)
    return None


def draw_polygon(
    frame,
    polygon: Polygon,
    color,
    label: str,
    thickness: int = ZONE_LINE_THICKNESS,
):
    if len(polygon) < 3:
        return
    pts = np.asarray(polygon, dtype=np.int32).reshape((-1, 1, 2))
    cv2.polylines(frame, [pts], True, color, thickness, cv2.LINE_AA)
    x, y = polygon[0]
    cv2.putText(
        frame,
        label,
        (x + 4, max(20, y - 7)),
        cv2.FONT_HERSHEY_SIMPLEX,
        ZONE_LABEL_FONT_SCALE,
        color,
        ZONE_LABEL_THICKNESS,
        cv2.LINE_AA,
    )


def draw_zones(frame, zones: Dict):
    if not DRAW_ZONES:
        return
    if USE_TRACKING_ROI:
        draw_polygon(frame, zones["tracking_roi"], COLOR_ROI, "TRACKING ROI")
    for idx, poly in enumerate(zones["approach_zones"]):
        draw_polygon(frame, poly, COLOR_APPROACH, f"APPROACH {idx + 1}")
    draw_polygon(frame, zones["intersection_zone"], COLOR_INTERSECTION, "INTERSECTION")


def state_color(state: TrackState):
    if not state.detection_confirmed:
        return COLOR_CANDIDATE
    if state.decision == "NO_SLOWDOWN":
        return COLOR_NO_SLOWDOWN
    if state.decision == "SLOWED":
        return COLOR_SLOWED
    if state.entered_intersection:
        return COLOR_EVALUATING
    return COLOR_NORMAL


def state_text(state: TrackState) -> str:
    if not state.detection_confirmed:
        return "CANDIDATE"
    if state.decision == "NO_SLOWDOWN":
        return "R2: NO SLOWDOWN"
    if state.decision == "SLOWED":
        return "R2: SLOWED"
    if state.entered_intersection:
        return "R2: EVALUATING"
    if state.has_visited_approach:
        return "APPROACH"
    if state.confirmation_reason == "strong-static":
        return "DETECTED"
    return "TRACKED"


def draw_track(
    frame,
    confidence: float,
    state: TrackState,
    *,
    candidate: bool = False,
    box_override: Optional[Sequence[float]] = None,
):
    box = box_override if box_override is not None else state.smoothed_box
    if box is None:
        return
    color = COLOR_CANDIDATE if candidate else state_color(state)
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    x1, x2 = max(0, min(x1, w - 1)), max(0, min(x2, w - 1))
    y1, y2 = max(0, min(y1, h - 1)), max(0, min(y2, h - 1))
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, BOX_THICKNESS)

    gx, gy = bottom_center(box)
    cv2.circle(frame, (int(gx), int(gy)), 3, color, -1)

    if DRAW_TRAJECTORY and not candidate and len(state.display_positions) >= 2:
        recent = list(state.display_positions)[-TRAJECTORY_LENGTH:]
        for first, second in zip(recent, recent[1:]):
            if second[0] - first[0] > 2:
                continue
            cv2.line(
                frame,
                (int(round(first[1])), int(round(first[2]))),
                (int(round(second[1])), int(round(second[2]))),
                color,
                TRAJECTORY_THICKNESS,
                cv2.LINE_AA,
            )

    speed = (
        "-- px/s"
        if state.smoothed_speed_px_s is None
        else f"{state.smoothed_speed_px_s:.1f} px/s"
    )
    prefix = "Candidate" if candidate else "Forklift"
    parts = [f"{prefix} #{state.track_id}", f"{confidence:.2f}"]
    if DRAW_SPEED:
        parts.append(speed)
    parts.append(state_text(state))
    label = " | ".join(parts)

    (tw, th), base = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, FONT_THICKNESS
    )
    tx = max(0, min(x1, w - tw - 10))
    ty = y1 - 8
    if ty - th - base < 0:
        ty = min(h - base - 2, y1 + th + base + 8)
    top = max(0, ty - th - base - 5)
    bottom = min(h - 1, ty + base + 4)
    cv2.rectangle(frame, (tx, top), (min(w - 1, tx + tw + 10), bottom), color, -1)
    cv2.putText(
        frame,
        label,
        (tx + 5, ty),
        cv2.FONT_HERSHEY_SIMPLEX,
        FONT_SCALE,
        (0, 0, 0),
        FONT_THICKNESS,
        cv2.LINE_AA,
    )


def draw_panel(
    frame,
    frame_idx,
    total_frames,
    roi_tracks,
    detected,
    motion_confirmed,
    events,
    fps_proc,
):
    if not DRAW_PANEL:
        return
    lines = [
        "R2 Forklift Slowdown PoC",
        f"Frame: {frame_idx} / {total_frames}",
        f"Tracks in ROI: {roi_tracks}",
        f"Detected: {detected} | Moving: {motion_confirmed}",
        f"R2 events: {events}",
        f"Processing FPS: {fps_proc:.1f}",
        "Speed: relative image-plane px/s",
    ]

    font = cv2.FONT_HERSHEY_SIMPLEX
    text_sizes = [
        cv2.getTextSize(line, font, PANEL_FONT_SCALE, PANEL_FONT_THICKNESS)[0]
        for line in lines
    ]
    panel_width = max(width for width, _ in text_sizes) + 2 * PANEL_PADDING_X
    panel_height = len(lines) * PANEL_LINE_HEIGHT + 2 * PANEL_PADDING_Y
    panel_width = min(panel_width, frame.shape[1] - 2 * PANEL_MARGIN)
    panel_height = min(panel_height, frame.shape[0] - 2 * PANEL_MARGIN)
    x1 = max(0, frame.shape[1] - PANEL_MARGIN - panel_width)
    y1 = PANEL_MARGIN
    x2 = min(frame.shape[1] - 1, x1 + panel_width)
    y2 = min(frame.shape[0] - 1, y1 + panel_height)

    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)
    for i, line in enumerate(lines):
        text_y = y1 + PANEL_PADDING_Y + 15 + i * PANEL_LINE_HEIGHT
        cv2.putText(
            frame,
            line,
            (x1 + PANEL_PADDING_X, text_y),
            font,
            PANEL_FONT_SCALE,
            (255, 255, 255),
            PANEL_FONT_THICKNESS,
            cv2.LINE_AA,
        )


def draw_event_banner(frame, event):
    if event is None:
        return
    color = COLOR_NO_SLOWDOWN if event["result"] == "NO_SLOWDOWN" else COLOR_SLOWED
    slowdown_percent = float(event["slowdown_percent"])
    change_text = (
        f"Slowdown: {slowdown_percent:.1f}%"
        if slowdown_percent >= 0
        else f"Speed increase: {-slowdown_percent:.1f}%"
    )
    text = (
        f"R2 RESULT | Forklift #{event['track_id']} | "
        f"{event['result'].replace('_', ' ')} | {change_text}"
    )
    banner_font_scale = 0.55
    banner_thickness = 1
    (tw, th), base = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, banner_font_scale, banner_thickness
    )
    x1, y2 = 20, frame.shape[0] - 20
    y1 = max(0, y2 - th - base - 24)
    x2 = min(frame.shape[1] - 20, x1 + tw + 30)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, -1)
    cv2.putText(
        frame,
        text,
        (x1 + 15, y2 - base - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        banner_font_scale,
        (255, 255, 255),
        banner_thickness,
        cv2.LINE_AA,
    )


def write_events_csv(events: List[Dict]):
    fields = [
        "track_id",
        "approach_zone",
        "intersection_entry_frame",
        "approach_speed_px_s",
        "intersection_speed_px_s",
        "slowdown_percent",
        "result",
    ]
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with OUTPUT_CSV_PATH.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(events)


def main():
    print("=" * 80)
    print("R2 FORKLIFT SLOWDOWN DEMO - test1.mp4")
    print("=" * 80)

    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    if not VIDEO_PATH.is_file():
        raise FileNotFoundError(f"Video not found: {VIDEO_PATH}")

    use_cuda = torch.cuda.is_available()
    device = DEVICE if use_cuda else "cpu"
    print(f"CUDA: {use_cuda}")
    if use_cuda:
        torch.backends.cudnn.benchmark = True
        print(f"GPU : {torch.cuda.get_device_name(DEVICE)}")
    print(
        f"Config: conf={CONF_THRESHOLD}, iou={IOU_THRESHOLD}, "
        f"imgsz={IMAGE_SIZE}, tuned ByteTrack"
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(VIDEO_PATH))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open: {VIDEO_PATH}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if width <= 0 or height <= 0:
        cap.release()
        raise RuntimeError(f"Invalid resolution: {width}x{height}")
    if fps <= 0:
        cap.release()
        raise RuntimeError(f"Invalid FPS: {fps}")

    ok, first_frame = cap.read()
    if not ok:
        cap.release()
        raise RuntimeError("Cannot read first frame")

    zones = load_or_create_zones(first_frame)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    model = YOLO(str(MODEL_PATH))
    if model.task != "detect":
        cap.release()
        raise RuntimeError(f"Expected a detection model, got task={model.task!r}")
    forklift_class_id = get_forklift_class_id(model)
    native_end2end = bool(getattr(model.model, "end2end", False))
    if native_end2end:
        # YOLO26's one-to-many head produces better-calibrated boxes for
        # association than the native NMS-free inference head.
        model.model.end2end = False
        print("Model head: one-to-many + NMS (tracking mode)")
    else:
        print("Model head: standard NMS")

    tracker_temp = tempfile.TemporaryDirectory(prefix="r2_bytetrack_")
    tracker_path = Path(tracker_temp.name) / "bytetrack_r2.yaml"
    tracker_buffer = write_tracker_config(tracker_path, fps)
    print(
        "ByteTrack: "
        f"low/high/new={TRACK_LOW_THRESHOLD:.2f}/"
        f"{TRACK_HIGH_THRESHOLD:.2f}/{NEW_TRACK_THRESHOLD:.2f}, "
        f"buffer={tracker_buffer} frames, match={MATCH_THRESHOLD:.2f}"
    )

    writer = cv2.VideoWriter(
        str(OUTPUT_VIDEO_PATH),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        cap.release()
        tracker_temp.cleanup()
        raise RuntimeError(f"Cannot create: {OUTPUT_VIDEO_PATH}")

    states: Dict[int, TrackState] = {}
    events: List[Dict] = []
    recent_event = None
    recent_event_until = 0
    frame_idx = 0
    written = 0
    geometry_rejected_total = 0
    duplicates_rejected_total = 0
    frame_diagonal = math.hypot(width, height)
    visual_interval_frames = max(1, int(round(VISUAL_MOTION_INTERVAL_SECONDS * fps)))
    grayscale_history: Deque[np.ndarray] = deque(maxlen=visual_interval_frames + 1)
    start = time.perf_counter()

    try:
        with torch.inference_mode():
            while MAX_FRAMES is None or frame_idx < MAX_FRAMES:
                ok, frame = cap.read()
                if not ok:
                    break
                frame_idx += 1
                t0 = time.perf_counter()

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray = cv2.GaussianBlur(gray, (5, 5), 0)
                grayscale_history.append(gray)
                change_mask: Optional[np.ndarray] = None
                if len(grayscale_history) == grayscale_history.maxlen:
                    difference = cv2.absdiff(
                        grayscale_history[0], grayscale_history[-1]
                    )
                    change_mask = difference >= VISUAL_DIFFERENCE_THRESHOLD

                results = model.track(
                    frame,
                    persist=True,
                    tracker=str(tracker_path),
                    classes=[forklift_class_id],
                    conf=CONF_THRESHOLD,
                    iou=IOU_THRESHOLD,
                    imgsz=IMAGE_SIZE,
                    device=device,
                    quantize=FP16_PRECISION if use_cuda else None,
                    verbose=False,
                )

                # Zone lines are deliberately thin and stay behind boxes/text.
                draw_zones(frame, zones)
                roi_tracks = 0
                detected_visible = 0
                motion_visible = 0

                if results:
                    boxes = results[0].boxes
                    if boxes is not None and len(boxes) > 0 and boxes.id is not None:
                        data = (
                            torch.cat(
                                (
                                    boxes.xyxy,
                                    boxes.conf.unsqueeze(1),
                                    boxes.cls.unsqueeze(1),
                                    boxes.id.unsqueeze(1),
                                ),
                                dim=1,
                            )
                            .detach()
                            .cpu()
                            .numpy()
                        )
                        selected, geometry_rejected, duplicates_rejected = (
                            select_track_detections(
                                data,
                                states,
                                frame.shape,
                                forklift_class_id,
                                zones,
                            )
                        )
                        geometry_rejected_total += geometry_rejected
                        duplicates_rejected_total += duplicates_rejected
                        roi_tracks = len(selected)

                        for detection in selected:
                            box = detection[:4]
                            confidence = float(detection[4])
                            track_id = int(detection[6])
                            state = states.setdefault(
                                track_id, TrackState(track_id=track_id)
                            )
                            current_visual_motion = visual_motion_ratio(
                                change_mask, box
                            )
                            point = update_track_observation(
                                state,
                                box,
                                confidence,
                                current_visual_motion,
                                frame_idx,
                                fps,
                                frame_diagonal,
                            )
                            if point is None:
                                continue

                            speed = update_speed(state, fps)
                            event = update_r2(
                                state, point, speed, frame_idx, fps, zones
                            )
                            if event is not None and not state.event_logged:
                                state.event_logged = True
                                events.append(event)
                                recent_event = event
                                recent_event_until = frame_idx + int(
                                    EVENT_BANNER_SECONDS * fps
                                )
                                print("R2 EVENT:", event)

                            if state.detection_confirmed:
                                state.display_positions.append(
                                    (frame_idx, point[0], point[1])
                                )
                                detected_visible += 1
                                if state.motion_confirmed:
                                    motion_visible += 1
                                draw_track(frame, confidence, state)
                            elif SHOW_CANDIDATES:
                                draw_track(
                                    frame,
                                    confidence,
                                    state,
                                    candidate=True,
                                    box_override=box,
                                )

                stale_frames = max(1, int(round(STALE_STATE_SECONDS * fps)))
                stale_ids = [
                    track_id
                    for track_id, state in states.items()
                    if frame_idx - state.last_seen_frame > stale_frames
                ]
                for track_id in stale_ids:
                    del states[track_id]

                elapsed_frame = time.perf_counter() - t0
                fps_proc = 1.0 / elapsed_frame if elapsed_frame > 0 else 0.0
                draw_panel(
                    frame,
                    frame_idx,
                    total_frames,
                    roi_tracks,
                    detected_visible,
                    motion_visible,
                    len(events),
                    fps_proc,
                )

                if recent_event is not None and frame_idx <= recent_event_until:
                    draw_event_banner(frame, recent_event)

                writer.write(frame)
                written += 1

                if frame_idx == 1 or frame_idx % PROGRESS_INTERVAL == 0:
                    elapsed = time.perf_counter() - start
                    avg = frame_idx / elapsed if elapsed > 0 else 0.0
                    print(
                        f"[{frame_idx}/{total_frames}] ROI={roi_tracks} | "
                        f"detected={detected_visible} | moving={motion_visible} | "
                        f"events={len(events)} | avg_fps={avg:.2f}"
                    )
    finally:
        cap.release()
        writer.release()
        tracker_temp.cleanup()
        cv2.destroyAllWindows()

    write_events_csv(events)
    elapsed = time.perf_counter() - start
    avg = written / elapsed if elapsed > 0 else 0.0

    print("\n" + "=" * 80)
    print("R2 DEMO COMPLETE")
    print("=" * 80)
    print(f"Frames written : {written}")
    print(f"R2 events      : {len(events)}")
    print(f"Average FPS    : {avg:.2f}")
    print(f"Geometry rejects: {geometry_rejected_total}")
    print(f"Duplicate rejects: {duplicates_rejected_total}")
    print(f"Video output   : {OUTPUT_VIDEO_PATH}")
    print(f"CSV output     : {OUTPUT_CSV_PATH}")
    print("NOTE: speed is px/s, not calibrated km/h.")


if __name__ == "__main__":
    try:
        main()
    except ZoneSelectionAborted as exc:
        cv2.destroyAllWindows()
        print(f"Zone setup cancelled: {exc}")
