from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Deque, Dict, List, Optional, Tuple
import csv
import json
import math
import time

import cv2
import numpy as np
import torch
from ultralytics import YOLO
import supervision as sv


# =============================================================================
# PROJECT PATHS
# =============================================================================

# Expected project layout:
#
# R2/
# ├── scripts/
# │   └── r2_forklift_slowdown_demo.py
# ├── results/
# │   └── forklift_fresh_model/
# │       └── best_fresh.pt
# ├── video/
# │   └── Forklift Accident_ The Blind Corner.mp4
# ├── configs/
# └── outputs/
#
PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "best_fresh.pt"
)

VIDEO_PATH = (
    PROJECT_ROOT
    / "video"
    / "test.mp4"
)

CONFIG_DIR = PROJECT_ROOT / "configs"

ZONES_PATH = (
    CONFIG_DIR
    / "r2_blind_corner_zones.json"
)

OUTPUT_DIR = PROJECT_ROOT / "outputs"

OUTPUT_VIDEO_PATH = (
    OUTPUT_DIR
    / "r2_blind_corner_demo.mp4"
)

OUTPUT_EVENTS_CSV = (
    OUTPUT_DIR
    / "r2_events.csv"
)


# =============================================================================
# DETECTOR CONFIGURATION
# =============================================================================

DEVICE = 0
USE_FP16 = True

# best_fresh.pt worked best in previous experiments.
CONF_THRESHOLD = 0.08
IOU_THRESHOLD = 0.50

# Full-frame inference only.
IMAGE_SIZE = 1280


# =============================================================================
# BYTETRACK CONFIGURATION
# =============================================================================

# Keep this aligned with the YOLO threshold so low-confidence but useful
# forklift detections can still initialize a track.
TRACK_ACTIVATION_THRESHOLD = 0.08

# Keep a temporarily lost ID alive for longer because the current detector
# occasionally misses forklifts for several frames.
LOST_TRACK_BUFFER = 60

MINIMUM_MATCHING_THRESHOLD = 0.80


# =============================================================================
# R2 ZONE CONFIGURATION
# =============================================================================

# On the first run, zones are drawn interactively and saved to ZONES_PATH.
# On subsequent runs, the saved zones are loaded automatically.
FORCE_REDRAW_ZONES = False

# Blind-corner scenarios often have two possible approach directions.
# Change to 1 if only one approach zone is needed.
NUM_APPROACH_ZONES = 2

# A detection is considered relevant only when its bottom-center ground point
# lies inside the tracking ROI.
USE_TRACKING_ROI = True


# =============================================================================
# MOTION CONFIRMATION
# =============================================================================

# Static pallets/racks can be falsely detected as forklifts. We therefore do
# not immediately trust every ByteTrack ID.
#
# A candidate becomes a confirmed vehicle only after its ground point has
# moved a meaningful distance across several frames.
MOTION_HISTORY_LENGTH = 20

# Require at least this many observations before checking motion.
MIN_MOTION_OBSERVATIONS = 8

# Net image-plane displacement required to confirm a moving forklift.
#
# For 1280x720 Blind Corner footage, 20-30 px is a reasonable starting range.
MIN_NET_MOTION_PX = 24.0

# Once a track has been confirmed as moving, it remains a valid forklift even
# if it later slows down or stops. This is essential for R2.
KEEP_CONFIRMED_TRACKS = True


# =============================================================================
# RELATIVE SPEED ESTIMATION
# =============================================================================

# This PoC estimates image-plane speed in pixels/second.
#
# It is NOT a calibrated km/h measurement.
#
# Physical speed requires camera calibration / homography and real-world
# distances. Relative image-plane speed is sufficient for a first R2 demo
# showing whether the forklift slowed down before entering the intersection.
SPEED_WINDOW_FRAMES = 5

# Exponential moving average for speed stabilization.
SPEED_EMA_ALPHA = 0.30

# Ignore tiny apparent speeds caused by bounding-box jitter.
MIN_VALID_SPEED_PX_S = 5.0


# =============================================================================
# R2 SLOWDOWN DECISION
# =============================================================================

# Number of valid speed samples required in an approach zone before an
# intersection event can be evaluated.
MIN_APPROACH_SPEED_SAMPLES = 5

# Number of speed samples collected after entering the intersection before
# making the slowdown decision.
MIN_INTERSECTION_SPEED_SAMPLES = 5

# Use only the most recent approach samples before intersection entry.
APPROACH_BASELINE_WINDOW = 15

# R2 PoC rule:
#
# reduction >= 20% => SLOWED
# reduction < 20%  => NO SLOWDOWN
#
# This threshold is configurable and should not be treated as a production
# safety standard.
MIN_SLOWDOWN_RATIO = 0.20


# =============================================================================
# TRAJECTORY / STATE CONFIGURATION
# =============================================================================

TRAJECTORY_LENGTH = 80

# Remove stale candidate state after this many unseen frames.
STALE_TRACK_FRAMES = 120


# =============================================================================
# DISPLAY CONFIGURATION
# =============================================================================

PROGRESS_INTERVAL = 50

BOX_THICKNESS = 3
FONT_SCALE = 0.60
FONT_THICKNESS = 2

# OpenCV uses BGR.
COLOR_TRACKING_ROI = (255, 255, 0)
COLOR_APPROACH = (0, 255, 255)
COLOR_INTERSECTION = (0, 128, 255)

COLOR_CANDIDATE = (180, 180, 180)
COLOR_CONFIRMED = (0, 220, 0)
COLOR_SLOWED = (0, 220, 0)
COLOR_RISK = (0, 0, 255)
COLOR_UNKNOWN = (255, 180, 0)


# =============================================================================
# DATA STRUCTURES
# =============================================================================


@dataclass
class TrackState:
    track_id: int

    confirmed: bool = False

    positions: Deque[Tuple[int, float, float]] = field(
        default_factory=lambda: deque(maxlen=TRAJECTORY_LENGTH)
    )

    box_heights: Deque[float] = field(
        default_factory=lambda: deque(maxlen=TRAJECTORY_LENGTH)
    )

    speed_history: Deque[float] = field(
        default_factory=lambda: deque(maxlen=TRAJECTORY_LENGTH)
    )

    approach_speeds: Deque[float] = field(
        default_factory=lambda: deque(maxlen=80)
    )

    intersection_speeds: Deque[float] = field(
        default_factory=lambda: deque(maxlen=80)
    )

    smoothed_speed_px_s: Optional[float] = None

    active_approach_zone: Optional[int] = None

    entered_intersection: bool = False
    intersection_entry_frame: Optional[int] = None

    baseline_speed_px_s: Optional[float] = None
    intersection_speed_px_s: Optional[float] = None

    slowdown_ratio: Optional[float] = None
    decision: Optional[str] = None

    event_logged: bool = False

    last_seen_frame: int = 0


# =============================================================================
# GEOMETRY HELPERS
# =============================================================================


def point_in_polygon(
    point: Tuple[float, float],
    polygon: List[Tuple[int, int]],
) -> bool:
    if len(polygon) < 3:
        return False

    contour = np.asarray(
        polygon,
        dtype=np.int32,
    ).reshape((-1, 1, 2))

    return (
        cv2.pointPolygonTest(
            contour,
            point,
            False,
        )
        >= 0
    )


def bottom_center(
    xyxy: np.ndarray,
) -> Tuple[float, float]:
    x1, y1, x2, y2 = xyxy

    return (
        float((x1 + x2) / 2.0),
        float(y2),
    )


def draw_polygon(
    frame: np.ndarray,
    polygon: List[Tuple[int, int]],
    color: Tuple[int, int, int],
    label: str,
    thickness: int = 2,
) -> None:
    if len(polygon) < 3:
        return

    points = np.asarray(
        polygon,
        dtype=np.int32,
    ).reshape((-1, 1, 2))

    cv2.polylines(
        frame,
        [points],
        isClosed=True,
        color=color,
        thickness=thickness,
        lineType=cv2.LINE_AA,
    )

    x, y = polygon[0]

    cv2.putText(
        frame,
        label,
        (x + 5, max(20, y - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.60,
        color,
        2,
        cv2.LINE_AA,
    )


# =============================================================================
# INTERACTIVE ZONE SELECTION
# =============================================================================


class ZoneSelectionAborted(RuntimeError):
    """Raised when the user explicitly quits interactive zone setup."""


def select_polygon(
    frame: np.ndarray,
    title: str,
    instruction: str,
    color: Tuple[int, int, int],
) -> List[Tuple[int, int]]:
    """
    Interactive polygon drawing.

    Controls:
      Left click : add a point
      Right click, Enter, or Space : finish polygon
      U or Backspace : undo the last point
      R or Esc   : reset the current polygon
      Q          : quit zone setup
    """
    points: List[Tuple[int, int]] = []
    selection_complete = False
    status_message = ""

    window_name = title

    cv2.namedWindow(
        window_name,
        cv2.WINDOW_NORMAL,
    )

    def mouse_callback(
        event,
        x,
        y,
        flags,
        param,
    ):
        nonlocal selection_complete, status_message

        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))

            status_message = ""

        elif event == cv2.EVENT_RBUTTONDOWN:
            if len(points) >= 3:
                selection_complete = True

            else:
                status_message = (
                    "Add at least 3 points before finishing."
                )

    cv2.setMouseCallback(
        window_name,
        mouse_callback,
    )

    while True:
        if selection_complete:
            break

        canvas = frame.copy()

        if points:
            for point in points:
                cv2.circle(
                    canvas,
                    point,
                    5,
                    color,
                    -1,
                )

            if len(points) >= 2:
                cv2.polylines(
                    canvas,
                    [
                        np.asarray(
                            points,
                            dtype=np.int32,
                        ).reshape((-1, 1, 2))
                    ],
                isClosed=len(points) >= 3,
                    color=color,
                    thickness=2,
                    lineType=cv2.LINE_AA,
                )

        overlay = canvas.copy()

        cv2.rectangle(
            overlay,
            (0, 0),
            (
                min(canvas.shape[1], 1000),
                120,
            ),
            (0, 0, 0),
            -1,
        )

        cv2.addWeighted(
            overlay,
            0.65,
            canvas,
            0.35,
            0,
            canvas,
        )

        cv2.putText(
            canvas,
            instruction,
            (15, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.putText(
            canvas,
            "Left: add | Right / ENTER / SPACE: finish",
            (15, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        cv2.putText(
            canvas,
            (
                status_message
                or "U / Backspace: undo | R / ESC: reset | Q: quit"
            ),
            (15, 92),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (
                (80, 80, 255)
                if status_message
                else (255, 255, 255)
            ),
            1,
            cv2.LINE_AA,
        )

        cv2.imshow(
            window_name,
            canvas,
        )

        try:
            window_visible = cv2.getWindowProperty(
                window_name,
                cv2.WND_PROP_VISIBLE,
            )
        except cv2.error:
            window_visible = 0

        if window_visible < 1:
            raise ZoneSelectionAborted(
                f"Zone-selection window was closed: {title}"
            )

        wait_key = getattr(
            cv2,
            "waitKeyEx",
            cv2.waitKey,
        )
        key = wait_key(20)
        key_code = key & 0xFF if key >= 0 else -1

        if key_code in (10, 13, 32):
            if len(points) >= 3:
                break

            status_message = (
                "Add at least 3 points before finishing."
            )

        elif key_code in (
            8,
            127,
            ord("u"),
            ord("U"),
        ):
            if points:
                points.pop()

            status_message = ""

        elif key_code in (
            ord("r"),
            ord("R"),
            27,
        ):
            points.clear()

            status_message = (
                "Current polygon reset. Add new points."
            )

        elif key_code in (
            ord("q"),
            ord("Q"),
        ):
            raise ZoneSelectionAborted(
                f"Zone selection aborted: {title}"
            )

    try:
        cv2.destroyWindow(
            window_name
        )
    except cv2.error:
        pass

    return points


def create_zones_interactively(
    first_frame: np.ndarray,
) -> Dict:
    print()
    print("=" * 80)
    print("INTERACTIVE R2 ZONE SETUP")
    print("=" * 80)
    print(
        "The first video frame will be displayed several times."
    )
    print(
        "Draw polygons using LEFT CLICK. Press ENTER when each polygon is complete."
    )
    print()

    tracking_roi = select_polygon(
        first_frame,
        "1 - Tracking ROI",
        (
            "Draw a BROAD forklift operating/road region. "
            "Exclude irrelevant pallet/rack areas when possible."
        ),
        COLOR_TRACKING_ROI,
    )

    intersection_zone = select_polygon(
        first_frame,
        "2 - Intersection Zone",
        (
            "Draw the actual blind-corner INTERSECTION area where R2 is evaluated."
        ),
        COLOR_INTERSECTION,
    )

    approach_zones = []

    for index in range(
        NUM_APPROACH_ZONES
    ):
        zone = select_polygon(
            first_frame,
            f"3.{index + 1} - Approach Zone {index + 1}",
            (
                f"Draw APPROACH ZONE {index + 1}: "
                "the road segment immediately before the intersection."
            ),
            COLOR_APPROACH,
        )

        approach_zones.append(
            zone
        )

    zones = {
        "tracking_roi": tracking_roi,
        "intersection_zone": intersection_zone,
        "approach_zones": approach_zones,
    }

    CONFIG_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        ZONES_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            zones,
            file,
            indent=2,
        )

    print(
        f"Zone configuration saved to: {ZONES_PATH}"
    )

    return zones


def load_or_create_zones(
    first_frame: np.ndarray,
) -> Dict:
    if (
        ZONES_PATH.exists()
        and not FORCE_REDRAW_ZONES
    ):
        with open(
            ZONES_PATH,
            "r",
            encoding="utf-8",
        ) as file:
            raw = json.load(file)

        zones = {
            "tracking_roi": [
                tuple(point)
                for point in raw["tracking_roi"]
            ],
            "intersection_zone": [
                tuple(point)
                for point in raw["intersection_zone"]
            ],
            "approach_zones": [
                [
                    tuple(point)
                    for point in polygon
                ]
                for polygon in raw["approach_zones"]
            ],
        }

        print(
            f"Loaded zones from: {ZONES_PATH}"
        )

        return zones

    return create_zones_interactively(
        first_frame
    )


# =============================================================================
# MODEL / TRACKER HELPERS
# =============================================================================


def find_forklift_class_id(
    model: YOLO,
) -> int:
    names = model.names

    if isinstance(names, list):
        class_map = {
            index: name
            for index, name in enumerate(names)
        }

    else:
        class_map = {
            int(index): name
            for index, name in names.items()
        }

    matches = [
        class_id
        for class_id, class_name in class_map.items()
        if str(class_name).strip().casefold()
        == "forklift"
    ]

    if len(matches) != 1:
        raise RuntimeError(
            "Expected exactly one class named 'forklift'. "
            f"Available classes: {class_map}"
        )

    return matches[0]


def create_tracker(
    frame_rate: float,
):
    tracker_fps = max(
        1,
        int(round(frame_rate)),
    )

    try:
        return sv.ByteTrack(
            track_activation_threshold=TRACK_ACTIVATION_THRESHOLD,
            lost_track_buffer=LOST_TRACK_BUFFER,
            minimum_matching_threshold=MINIMUM_MATCHING_THRESHOLD,
            frame_rate=tracker_fps,
        )

    except TypeError:
        print(
            "WARNING: This Supervision version does not support all "
            "custom ByteTrack parameters. Falling back to defaults."
        )

        return sv.ByteTrack(
            frame_rate=tracker_fps,
        )


def yolo_detect(
    model: YOLO,
    frame: np.ndarray,
    forklift_class_id: int,
) -> Tuple[np.ndarray, np.ndarray]:
    results = model.predict(
        source=frame,
        conf=CONF_THRESHOLD,
        iou=IOU_THRESHOLD,
        imgsz=IMAGE_SIZE,
        device=DEVICE,
        half=(
            USE_FP16
            and torch.cuda.is_available()
        ),
        classes=[
            forklift_class_id
        ],
        verbose=False,
    )

    result = results[0]

    if (
        result.boxes is None
        or len(result.boxes) == 0
    ):
        return (
            np.empty(
                (0, 4),
                dtype=np.float32,
            ),
            np.empty(
                (0,),
                dtype=np.float32,
            ),
        )

    boxes = (
        result.boxes.xyxy
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32)
    )

    confidences = (
        result.boxes.conf
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32)
    )

    return (
        boxes,
        confidences,
    )


def filter_detections_by_tracking_roi(
    boxes: np.ndarray,
    confidences: np.ndarray,
    tracking_roi: List[Tuple[int, int]],
) -> Tuple[np.ndarray, np.ndarray]:
    if (
        not USE_TRACKING_ROI
        or len(boxes) == 0
    ):
        return (
            boxes,
            confidences,
        )

    keep_indices = []

    for index, box in enumerate(
        boxes
    ):
        point = bottom_center(
            box
        )

        if point_in_polygon(
            point,
            tracking_roi,
        ):
            keep_indices.append(
                index
            )

    if not keep_indices:
        return (
            np.empty(
                (0, 4),
                dtype=np.float32,
            ),
            np.empty(
                (0,),
                dtype=np.float32,
            ),
        )

    keep = np.asarray(
        keep_indices,
        dtype=np.int64,
    )

    return (
        boxes[keep],
        confidences[keep],
    )


def make_detections(
    boxes: np.ndarray,
    confidences: np.ndarray,
) -> sv.Detections:
    if len(boxes) == 0:
        return sv.Detections.empty()

    return sv.Detections(
        xyxy=boxes,
        confidence=confidences,
        class_id=np.zeros(
            len(boxes),
            dtype=np.int32,
        ),
    )


# =============================================================================
# TRACK STATE / MOTION / SPEED
# =============================================================================


def get_or_create_state(
    states: Dict[int, TrackState],
    track_id: int,
) -> TrackState:
    if track_id not in states:
        states[track_id] = TrackState(
            track_id=track_id
        )

    return states[track_id]


def robust_net_displacement(
    positions: Deque[
        Tuple[int, float, float]
    ],
) -> float:
    if len(positions) < 2:
        return 0.0

    recent = list(
        positions
    )[-MOTION_HISTORY_LENGTH:]

    if len(recent) < 2:
        return 0.0

    edge_count = min(
        3,
        max(
            1,
            len(recent) // 3,
        ),
    )

    first_x = float(
        np.median(
            [
                item[1]
                for item in recent[:edge_count]
            ]
        )
    )

    first_y = float(
        np.median(
            [
                item[2]
                for item in recent[:edge_count]
            ]
        )
    )

    last_x = float(
        np.median(
            [
                item[1]
                for item in recent[-edge_count:]
            ]
        )
    )

    last_y = float(
        np.median(
            [
                item[2]
                for item in recent[-edge_count:]
            ]
        )
    )

    return math.hypot(
        last_x - first_x,
        last_y - first_y,
    )


def update_motion_confirmation(
    state: TrackState,
) -> None:
    if (
        state.confirmed
        and KEEP_CONFIRMED_TRACKS
    ):
        return

    if (
        len(state.positions)
        < MIN_MOTION_OBSERVATIONS
    ):
        return

    displacement = robust_net_displacement(
        state.positions
    )

    if displacement >= MIN_NET_MOTION_PX:
        state.confirmed = True


def estimate_speed_px_s(
    state: TrackState,
    fps: float,
) -> Optional[float]:
    if (
        len(state.positions)
        <= SPEED_WINDOW_FRAMES
    ):
        return None

    recent_positions = list(
        state.positions
    )

    current = recent_positions[-1]
    previous = recent_positions[
        -1 - SPEED_WINDOW_FRAMES
    ]

    frame_delta = (
        current[0]
        - previous[0]
    )

    if frame_delta <= 0:
        return None

    distance = math.hypot(
        current[1] - previous[1],
        current[2] - previous[2],
    )

    time_delta = (
        frame_delta
        / fps
    )

    if time_delta <= 0:
        return None

    speed = (
        distance
        / time_delta
    )

    if speed < MIN_VALID_SPEED_PX_S:
        speed = 0.0

    if (
        state.smoothed_speed_px_s
        is None
    ):
        state.smoothed_speed_px_s = speed

    else:
        alpha = SPEED_EMA_ALPHA

        state.smoothed_speed_px_s = (
            alpha * speed
            + (1.0 - alpha)
            * state.smoothed_speed_px_s
        )

    state.speed_history.append(
        state.smoothed_speed_px_s
    )

    return state.smoothed_speed_px_s


# =============================================================================
# R2 LOGIC
# =============================================================================


def find_approach_zone(
    point: Tuple[float, float],
    approach_zones: List[
        List[Tuple[int, int]]
    ],
) -> Optional[int]:
    for zone_index, polygon in enumerate(
        approach_zones
    ):
        if point_in_polygon(
            point,
            polygon,
        ):
            return zone_index

    return None


def update_r2_state(
    state: TrackState,
    ground_point: Tuple[float, float],
    speed_px_s: Optional[float],
    frame_index: int,
    zones: Dict,
) -> Optional[Dict]:
    """
    Update one confirmed forklift's R2 state.

    Returns an event dictionary when a new slowdown decision is made.
    """
    if not state.confirmed:
        return None

    approach_zone_index = find_approach_zone(
        ground_point,
        zones["approach_zones"],
    )

    in_intersection = point_in_polygon(
        ground_point,
        zones["intersection_zone"],
    )

    if (
        approach_zone_index is not None
        and not state.entered_intersection
        and speed_px_s is not None
        and speed_px_s >= MIN_VALID_SPEED_PX_S
    ):
        state.active_approach_zone = (
            approach_zone_index
        )

        state.approach_speeds.append(
            speed_px_s
        )

    # Freeze the approach baseline exactly when the forklift first enters
    # the intersection.
    if (
        in_intersection
        and not state.entered_intersection
    ):
        state.entered_intersection = True

        state.intersection_entry_frame = (
            frame_index
        )

        if (
            len(state.approach_speeds)
            >= MIN_APPROACH_SPEED_SAMPLES
        ):
            recent_approach = list(
                state.approach_speeds
            )[
                -APPROACH_BASELINE_WINDOW:
            ]

            state.baseline_speed_px_s = float(
                np.median(
                    recent_approach
                )
            )

    if (
        in_intersection
        and speed_px_s is not None
    ):
        state.intersection_speeds.append(
            speed_px_s
        )

    if state.decision is not None:
        return None

    if not state.entered_intersection:
        return None

    if state.baseline_speed_px_s is None:
        return None

    if (
        len(state.intersection_speeds)
        < MIN_INTERSECTION_SPEED_SAMPLES
    ):
        return None

    intersection_speed = float(
        np.median(
            list(
                state.intersection_speeds
            )[
                :MIN_INTERSECTION_SPEED_SAMPLES
            ]
        )
    )

    state.intersection_speed_px_s = (
        intersection_speed
    )

    baseline = max(
        state.baseline_speed_px_s,
        1e-6,
    )

    slowdown_ratio = (
        baseline
        - intersection_speed
    ) / baseline

    state.slowdown_ratio = (
        slowdown_ratio
    )

    if slowdown_ratio >= MIN_SLOWDOWN_RATIO:
        state.decision = "SLOWED"

    else:
        state.decision = "NO_SLOWDOWN"

    event = {
        "track_id": state.track_id,

        "approach_zone": (
            state.active_approach_zone + 1
            if state.active_approach_zone
            is not None
            else ""
        ),

        "intersection_entry_frame": (
            state.intersection_entry_frame
        ),

        "baseline_speed_px_s": round(
            state.baseline_speed_px_s,
            3,
        ),

        "intersection_speed_px_s": round(
            state.intersection_speed_px_s,
            3,
        ),

        "slowdown_percent": round(
            state.slowdown_ratio
            * 100.0,
            2,
        ),

        "decision": state.decision,
    }

    return event


# =============================================================================
# VISUALIZATION
# =============================================================================


def decision_color(
    state: TrackState,
) -> Tuple[int, int, int]:
    if not state.confirmed:
        return COLOR_CANDIDATE

    if state.decision == "SLOWED":
        return COLOR_SLOWED

    if state.decision == "NO_SLOWDOWN":
        return COLOR_RISK

    return COLOR_CONFIRMED


def draw_zones(
    frame: np.ndarray,
    zones: Dict,
) -> None:
    if USE_TRACKING_ROI:
        draw_polygon(
            frame,
            zones["tracking_roi"],
            COLOR_TRACKING_ROI,
            "TRACKING ROI",
            2,
        )

    for index, polygon in enumerate(
        zones["approach_zones"]
    ):
        draw_polygon(
            frame,
            polygon,
            COLOR_APPROACH,
            f"APPROACH {index + 1}",
            2,
        )

    draw_polygon(
        frame,
        zones["intersection_zone"],
        COLOR_INTERSECTION,
        "INTERSECTION",
        3,
    )


def draw_trajectory(
    frame: np.ndarray,
    state: TrackState,
    color: Tuple[int, int, int],
) -> None:
    if len(state.positions) < 2:
        return

    points = np.asarray(
        [
            (
                int(position[1]),
                int(position[2]),
            )
            for position in state.positions
        ],
        dtype=np.int32,
    ).reshape((-1, 1, 2))

    cv2.polylines(
        frame,
        [points],
        isClosed=False,
        color=color,
        thickness=2,
        lineType=cv2.LINE_AA,
    )


def draw_track(
    frame: np.ndarray,
    box: np.ndarray,
    confidence: float,
    state: TrackState,
) -> None:
    color = decision_color(
        state
    )

    x1, y1, x2, y2 = (
        box.astype(int)
    )

    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        color,
        BOX_THICKNESS,
    )

    ground_point = bottom_center(
        box
    )

    cv2.circle(
        frame,
        (
            int(ground_point[0]),
            int(ground_point[1]),
        ),
        5,
        color,
        -1,
    )

    if not state.confirmed:
        status = "CANDIDATE"

    elif state.decision == "SLOWED":
        status = "R2: SLOWED"

    elif state.decision == "NO_SLOWDOWN":
        status = "R2: NO SLOWDOWN"

    elif state.entered_intersection:
        status = "R2: EVALUATING"

    elif state.active_approach_zone is not None:
        status = "APPROACH"

    else:
        status = "CONFIRMED"

    speed_text = (
        f"{state.smoothed_speed_px_s:.1f} px/s"
        if state.smoothed_speed_px_s
        is not None
        else "-- px/s"
    )

    label = (
        f"Forklift #{state.track_id} "
        f"| {confidence:.2f} "
        f"| {speed_text} "
        f"| {status}"
    )

    text_size, baseline = cv2.getTextSize(
        label,
        cv2.FONT_HERSHEY_SIMPLEX,
        FONT_SCALE,
        FONT_THICKNESS,
    )

    text_width, text_height = (
        text_size
    )

    top = max(
        0,
        y1
        - text_height
        - baseline
        - 10,
    )

    cv2.rectangle(
        frame,
        (x1, top),
        (
            min(
                frame.shape[1] - 1,
                x1 + text_width + 10,
            ),
            y1,
        ),
        color,
        -1,
    )

    cv2.putText(
        frame,
        label,
        (
            x1 + 5,
            y1 - baseline - 5,
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        FONT_SCALE,
        (0, 0, 0),
        FONT_THICKNESS,
        cv2.LINE_AA,
    )

    draw_trajectory(
        frame,
        state,
        color,
    )


def draw_panel(
    frame: np.ndarray,
    frame_index: int,
    total_frames: int,
    raw_detection_count: int,
    roi_detection_count: int,
    active_track_count: int,
    confirmed_count: int,
    processing_fps: float,
    event_count: int,
) -> None:
    lines = [
        "R2 Forklift Slowdown Demo",
        f"Frame: {frame_index} / {total_frames}",
        (
            f"YOLO: {raw_detection_count} "
            f"| In ROI: {roi_detection_count}"
        ),
        (
            f"Active tracks: {active_track_count} "
            f"| Confirmed: {confirmed_count}"
        ),
        f"R2 events: {event_count}",
        f"Processing FPS: {processing_fps:.1f}",
        "Speed unit: image-plane px/s",
    ]

    panel_width = 560
    panel_height = (
        15
        + len(lines) * 31
    )

    overlay = frame.copy()

    cv2.rectangle(
        overlay,
        (0, 0),
        (
            min(
                panel_width,
                frame.shape[1],
            ),
            panel_height,
        ),
        (0, 0, 0),
        -1,
    )

    cv2.addWeighted(
        overlay,
        0.62,
        frame,
        0.38,
        0,
        frame,
    )

    for line_index, line in enumerate(
        lines
    ):
        cv2.putText(
            frame,
            line,
            (
                15,
                30 + 31 * line_index,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )


def draw_recent_event_banner(
    frame: np.ndarray,
    event: Optional[Dict],
) -> None:
    if event is None:
        return

    decision = event["decision"]

    if decision == "NO_SLOWDOWN":
        color = COLOR_RISK

    else:
        color = COLOR_SLOWED

    text = (
        f"R2 RESULT | Forklift #{event['track_id']} | "
        f"{decision} | "
        f"Reduction: {event['slowdown_percent']:.1f}%"
    )

    text_size, baseline = cv2.getTextSize(
        text,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        2,
    )

    text_width, text_height = (
        text_size
    )

    x1 = 20
    y1 = frame.shape[0] - 75

    x2 = min(
        frame.shape[1] - 20,
        x1 + text_width + 30,
    )

    y2 = frame.shape[0] - 20

    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        color,
        -1,
    )

    cv2.putText(
        frame,
        text,
        (
            x1 + 15,
            y2 - 16,
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


# =============================================================================
# CSV OUTPUT
# =============================================================================


def write_events_csv(
    events: List[Dict],
) -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = [
        "track_id",
        "approach_zone",
        "intersection_entry_frame",
        "baseline_speed_px_s",
        "intersection_speed_px_s",
        "slowdown_percent",
        "decision",
    ]

    with open(
        OUTPUT_EVENTS_CSV,
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for event in events:
            writer.writerow(
                event
            )


# =============================================================================
# MAIN PIPELINE
# =============================================================================


def main() -> None:
    print("=" * 80)
    print("R2 FORKLIFT SLOWDOWN DEMO")
    print("=" * 80)

    print(
        "PyTorch:",
        torch.__version__,
    )

    print(
        "CUDA available:",
        torch.cuda.is_available(),
    )

    if torch.cuda.is_available():
        print(
            "GPU:",
            torch.cuda.get_device_name(
                DEVICE
            ),
        )

    else:
        print(
            "WARNING: CUDA is unavailable."
        )

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    if not VIDEO_PATH.exists():
        raise FileNotFoundError(
            f"Video not found: {VIDEO_PATH}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    CONFIG_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    capture = cv2.VideoCapture(
        str(VIDEO_PATH)
    )

    if not capture.isOpened():
        raise RuntimeError(
            f"Cannot open video: {VIDEO_PATH}"
        )

    width = int(
        capture.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    height = int(
        capture.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )

    fps = float(
        capture.get(
            cv2.CAP_PROP_FPS
        )
    )

    total_frames = int(
        capture.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    if fps <= 0:
        capture.release()

        raise RuntimeError(
            f"Invalid video FPS: {fps}"
        )

    success, first_frame = (
        capture.read()
    )

    if not success:
        capture.release()

        raise RuntimeError(
            "Unable to read the first video frame."
        )

    # Zone setup is performed before processing.
    try:
        zones = load_or_create_zones(
            first_frame
        )
    except Exception:
        capture.release()
        cv2.destroyAllWindows()
        raise

    # Restart from frame zero.
    capture.set(
        cv2.CAP_PROP_POS_FRAMES,
        0,
    )

    model = YOLO(
        str(MODEL_PATH)
    )

    forklift_class_id = (
        find_forklift_class_id(
            model
        )
    )

    tracker = create_tracker(
        fps
    )

    writer = cv2.VideoWriter(
        str(
            OUTPUT_VIDEO_PATH
        ),
        cv2.VideoWriter_fourcc(
            *"mp4v"
        ),
        fps,
        (
            width,
            height,
        ),
    )

    if not writer.isOpened():
        capture.release()

        raise RuntimeError(
            "Unable to create output video: "
            f"{OUTPUT_VIDEO_PATH}"
        )

    print()
    print("Video")
    print("-" * 80)

    print(
        f"Resolution : {width}x{height}"
    )

    print(
        f"FPS        : {fps:.3f}"
    )

    print(
        f"Frames     : {total_frames}"
    )

    print()
    print("R2 PoC settings")
    print("-" * 80)

    print(
        f"Detector confidence     : "
        f"{CONF_THRESHOLD}"
    )

    print(
        f"YOLO imgsz              : "
        f"{IMAGE_SIZE}"
    )

    print(
        f"Motion confirmation     : "
        f"{MIN_NET_MOTION_PX:.1f}px"
    )

    print(
        f"Slowdown threshold      : "
        f"{MIN_SLOWDOWN_RATIO * 100:.1f}%"
    )

    print(
        "Speed measurement       : "
        "relative image-plane px/s"
    )

    states: Dict[
        int,
        TrackState,
    ] = {}

    events: List[Dict] = []

    frame_index = 0
    output_frame_count = 0

    recent_event: Optional[
        Dict
    ] = None

    recent_event_until_frame = 0

    start_time = (
        time.perf_counter()
    )

    try:
        while True:
            success, frame = (
                capture.read()
            )

            if not success:
                break

            frame_index += 1

            frame_start = (
                time.perf_counter()
            )

            raw_boxes, raw_confidences = (
                yolo_detect(
                    model,
                    frame,
                    forklift_class_id,
                )
            )

            boxes, confidences = (
                filter_detections_by_tracking_roi(
                    raw_boxes,
                    raw_confidences,
                    zones["tracking_roi"],
                )
            )

            detections = make_detections(
                boxes,
                confidences,
            )

            tracked = (
                tracker.update_with_detections(
                    detections
                )
            )

            visible_track_ids = set()

            if (
                len(tracked) > 0
                and tracked.tracker_id
                is not None
            ):
                for detection_index in range(
                    len(tracked)
                ):
                    raw_track_id = (
                        tracked.tracker_id[
                            detection_index
                        ]
                    )

                    if raw_track_id is None:
                        continue

                    track_id = int(
                        raw_track_id
                    )

                    visible_track_ids.add(
                        track_id
                    )

                    box = tracked.xyxy[
                        detection_index
                    ]

                    confidence = (
                        float(
                            tracked.confidence[
                                detection_index
                            ]
                        )
                        if tracked.confidence
                        is not None
                        else 0.0
                    )

                    ground_point = bottom_center(
                        box
                    )

                    state = get_or_create_state(
                        states,
                        track_id,
                    )

                    state.last_seen_frame = (
                        frame_index
                    )

                    state.positions.append(
                        (
                            frame_index,
                            ground_point[0],
                            ground_point[1],
                        )
                    )

                    state.box_heights.append(
                        float(
                            box[3] - box[1]
                        )
                    )

                    update_motion_confirmation(
                        state
                    )

                    speed = estimate_speed_px_s(
                        state,
                        fps,
                    )

                    if state.confirmed:
                        event = update_r2_state(
                            state=state,
                            ground_point=ground_point,
                            speed_px_s=speed,
                            frame_index=frame_index,
                            zones=zones,
                        )

                        if (
                            event is not None
                            and not state.event_logged
                        ):
                            events.append(
                                event
                            )

                            state.event_logged = True

                            recent_event = (
                                event
                            )

                            recent_event_until_frame = (
                                frame_index
                                + int(
                                    fps * 3.0
                                )
                            )

                            print()
                            print(
                                "R2 EVENT:",
                                event,
                            )

                    draw_track(
                        frame=frame,
                        box=box,
                        confidence=confidence,
                        state=state,
                    )

            # Purge stale, never-confirmed states.
            stale_track_ids = []

            for track_id, state in states.items():
                if (
                    frame_index
                    - state.last_seen_frame
                    > STALE_TRACK_FRAMES
                ):
                    if not state.confirmed:
                        stale_track_ids.append(
                            track_id
                        )

            for track_id in stale_track_ids:
                del states[
                    track_id
                ]

            confirmed_count = sum(
                state.confirmed
                for state in states.values()
                if (
                    frame_index
                    - state.last_seen_frame
                    <= LOST_TRACK_BUFFER
                )
            )

            processing_elapsed = (
                time.perf_counter()
                - frame_start
            )

            processing_fps = (
                1.0
                / processing_elapsed
                if processing_elapsed > 0
                else 0.0
            )

            draw_zones(
                frame,
                zones,
            )

            draw_panel(
                frame=frame,
                frame_index=frame_index,
                total_frames=total_frames,
                raw_detection_count=len(
                    raw_boxes
                ),
                roi_detection_count=len(
                    boxes
                ),
                active_track_count=len(
                    visible_track_ids
                ),
                confirmed_count=confirmed_count,
                processing_fps=processing_fps,
                event_count=len(
                    events
                ),
            )

            if (
                recent_event is not None
                and frame_index
                <= recent_event_until_frame
            ):
                draw_recent_event_banner(
                    frame,
                    recent_event,
                )

            writer.write(
                frame
            )

            output_frame_count += 1

            if (
                frame_index == 1
                or frame_index
                % PROGRESS_INTERVAL
                == 0
            ):
                total_elapsed = (
                    time.perf_counter()
                    - start_time
                )

                average_fps = (
                    frame_index
                    / total_elapsed
                    if total_elapsed > 0
                    else 0.0
                )

                print(
                    f"[{frame_index}/{total_frames}] "
                    f"raw={len(raw_boxes)} | "
                    f"roi={len(boxes)} | "
                    f"tracks={len(visible_track_ids)} | "
                    f"confirmed={confirmed_count} | "
                    f"events={len(events)} | "
                    f"avg_fps={average_fps:.2f}"
                )

    finally:
        capture.release()
        writer.release()

    write_events_csv(
        events
    )

    elapsed = (
        time.perf_counter()
        - start_time
    )

    average_fps = (
        output_frame_count
        / elapsed
        if elapsed > 0
        else 0.0
    )

    print()
    print("=" * 80)
    print("R2 DEMO SUMMARY")
    print("=" * 80)

    print(
        f"Decoded frames        : "
        f"{frame_index}"
    )

    print(
        f"Output frames         : "
        f"{output_frame_count}"
    )

    print(
        f"R2 events             : "
        f"{len(events)}"
    )

    print(
        f"Elapsed time          : "
        f"{elapsed:.2f}s"
    )

    print(
        f"Average processing FPS: "
        f"{average_fps:.2f}"
    )

    print(
        f"Output video          : "
        f"{OUTPUT_VIDEO_PATH}"
    )

    print(
        f"Event CSV             : "
        f"{OUTPUT_EVENTS_CSV}"
    )

    print()
    print(
        "IMPORTANT: R2 speed in this PoC is relative image-plane speed "
        "(pixels/second), not calibrated physical speed."
    )

    print(
        "For production deployment, add camera calibration/homography "
        "before interpreting speed in km/h."
    )


if __name__ == "__main__":
    try:
        main()
    except ZoneSelectionAborted as error:
        cv2.destroyAllWindows()
        print()
        print(f"Zone setup cancelled: {error}")
        print(
            "No zone configuration was saved. Run the script again to retry."
        )
