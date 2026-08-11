"""R7 forklift demo: tracking, fork/mast segmentation, and temporal alerts.

The script intentionally returns UNKNOWN whenever the current evidence is
missing or ambiguous.  It never carries a stale fork/direction estimate into a
new R7 candidate.  This is important because R7 is a compound temporal rule,
not a single-frame classifier.

Run with the configured defaults from any working directory::

    uv run python r7/test-tracking/scripts/r7_demo.py

CLI arguments remain available to override any default for one-off experiments.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from collections import Counter, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.utils.torch_utils import select_device


# =============================================================================
# PATHS -- EDIT THESE DEFAULTS, THEN RUN THIS FILE WITHOUT ARGUMENTS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "model"
VIDEO_DIR = PROJECT_ROOT / "videos"
CONFIG_DIR = PROJECT_ROOT / "configs"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

FORKLIFT_MODEL_PATH = MODEL_DIR / "best_fresh.pt"
R7_SEGMENTATION_MODEL_PATH = MODEL_DIR / "r7_yolo26s_seg_forks_mast_best.pt"
VIDEO_PATH = VIDEO_DIR / "test3.mp4"
TRACKER_CONFIG_PATH = CONFIG_DIR / "r7_bytetrack.yaml"
OUTPUT_VIDEO_PATH = OUTPUT_DIR / "test3_r7.mp4"
OUTPUT_CSV_PATH = OUTPUT_DIR / "test3_r7_events.csv"

# =============================================================================
# RUN DEFAULTS
# =============================================================================

# "auto" selects CUDA when Ultralytics can access it and otherwise uses CPU.
DEVICE = "auto"
FP16_PRECISION = 16
PROGRESS_INTERVAL = 100
MAX_FRAMES: int | None = None
OVERWRITE_OUTPUT = True
SHOW_PREVIEW = False
DEBUG_OVERLAY = False
ENABLE_CAMERA_COMPENSATION = True
FALLBACK_FPS = 30.0

# =============================================================================
# DETECTOR / TRACKER -- TUNED FOR WEAK DETECTIONS AND TEMPORARY OCCLUSION
# =============================================================================

FORKLIFT_CONF_THRESHOLD = 0.10
IMAGE_SIZE = 1920

# YOLO26 is end-to-end/NMS-free, so a legacy detector IOU_THRESHOLD is not a
# meaningful tuning control here. ByteTrack association uses MATCH_THRESHOLD.
TRACK_LOW_THRESHOLD = 0.05
TRACK_HIGH_THRESHOLD = 0.12
NEW_TRACK_THRESHOLD = 0.12
TRACK_BUFFER_SECONDS = 2.0
MATCH_THRESHOLD = 0.80
FUSE_SCORE = True
TRACK_STATE_TTL_SECONDS = 2.0

# =============================================================================
# CROP / SEGMENTATION
# =============================================================================

SEGMENTATION_CONF_THRESHOLD = 0.10
SEGMENTATION_IMAGE_SIZE = 640

# Keep symmetric padding modest, but extend farther above the detector box: the
# detector can cover only the forklift body while the mast continues upward.
CROP_PADDING = 0.15
CROP_TOP_PADDING = 0.75

# Conservative background-camera compensation.  Only a well-supported affine
# transform is accepted; otherwise the trajectory history is re-warmed.
CAMERA_MAX_CORNERS = 400
CAMERA_MIN_TRACKED_POINTS = 20
CAMERA_MIN_INLIERS = 15
CAMERA_MIN_INLIER_RATIO = 0.50
CAMERA_MAX_FRAME_SCALE_CHANGE = 0.12
CAMERA_MAX_FRAME_ROTATION_DEGREES = 8.0

# Trajectory is expressed in forklift-bbox diagonals, making the thresholds less
# sensitive to resolution and distance from the camera.
MOTION_WINDOW_SECONDS = 0.60
MIN_MOTION_SAMPLES = 6
MIN_MOTION_SPAN_SECONDS = 0.35
MAX_MOTION_GAP_SECONDS = 0.20
MOVING_ENTER_SPEED = 0.060  # bbox diagonals / second
MOVING_EXIT_SPEED = 0.035  # bbox diagonals / second
MIN_NET_DISPLACEMENT = 0.035  # bbox diagonals over the motion window

TURN_ENTER_DEGREES = 20.0
TURN_EXIT_DEGREES = 12.0
TURN_REVERSAL_CUTOFF_DEGREES = 150.0
MIN_TURN_LEG_DISPLACEMENT = 0.040  # bbox diagonals per trajectory leg
TURN_CONFIRM_SECONDS = 0.25
TURN_RELEASE_SECONDS = 0.15

# Relative fork height is projected along the mast axis and normalized by the
# forklift bbox height.  Therefore the regression slope unit is bbox-height/s.
FORK_WINDOW_SECONDS = 0.70
FORK_MAX_GAP_SECONDS = 0.25
MIN_FORK_SAMPLES = 8
MIN_FORK_SPAN_SECONDS = 0.40
MIN_FORK_DYNAMIC_CHANGE = 0.015  # bbox heights over the fitted time span
FORK_LOWER_ENTER_SLOPE = 0.060
FORK_LOWER_EXIT_SLOPE = 0.030
FORK_RAISE_ENTER_SLOPE = -0.060
FORK_RAISE_EXIT_SLOPE = -0.030

MIN_MAST_ELONGATION = 1.80
MAX_FORK_MERGE_DISTANCE = 0.25  # crop diagonals
MAX_PAIR_PERPENDICULAR_DISTANCE = 0.65  # forklift bbox heights
MAX_PAIR_AXIS_DISTANCE = 1.75  # forklift bbox heights

MIN_FRONT_VECTOR = 0.080  # bbox diagonals
DIRECTION_FORWARD_COSINE = 0.45
DIRECTION_REVERSE_COSINE = -0.45
DIRECTION_CONFIRM_SECONDS = 0.30
DIRECTION_REQUIRED_RATIO = 0.70

R7_CONFIRM_SECONDS = 0.50
R7_RELEASE_SECONDS = 0.22

FORK_COLOR = (255, 60, 210)  # BGR, magenta
MAST_COLOR = (40, 220, 255)  # BGR, yellow/cyan
NORMAL_COLOR = (70, 220, 90)
CAUTION_COLOR = (0, 180, 255)
ALERT_COLOR = (20, 20, 240)
UNKNOWN_COLOR = (165, 165, 165)

UNKNOWN = "UNKNOWN"
STATIC = "STATIC"
RAISING = "RAISING"
LOWERING = "LOWERING"
FORWARD = "FORWARD"
REVERSE = "REVERSE"


@dataclass(frozen=True)
class MotionSample:
    """A normalized-trajectory source observation for one tracked forklift."""

    timestamp: float
    cx: float
    cy: float
    width: float
    height: float


@dataclass(frozen=True)
class HeightSample:
    """Fork position along the mast, in normalized forklift-height units."""

    timestamp: float
    value: float


@dataclass
class MaskObservation:
    """One selected/merged segmentation class inside a forklift crop."""

    class_id: int
    confidence: float
    mask: np.ndarray
    centroid: tuple[float, float]
    area: int
    score: float
    principal_axis: np.ndarray | None = None
    axis_extent: float | None = None
    axis_elongation: float | None = None


@dataclass(frozen=True)
class TrackedDetection:
    track_id: int
    bbox: tuple[float, float, float, float]
    confidence: float


@dataclass(frozen=True)
class MotionEstimate:
    moving: bool
    velocity: np.ndarray
    speed: float
    net_displacement: float
    turn_candidate: bool | None
    turn_angle: float | None


@dataclass(frozen=True)
class CameraMotionEstimate:
    valid: bool
    affine: np.ndarray | None
    tracked_points: int = 0
    inliers: int = 0
    inlier_ratio: float = 0.0


@dataclass
class TemporalLatch:
    """Consecutive confirmation plus short release hysteresis for one rule."""

    candidate_frames: int = 0
    release_frames: int = 0
    active: bool = False
    activations: int = 0

    def invalidate(self) -> None:
        """Suppress an alert immediately when current evidence is UNKNOWN."""

        self.candidate_frames = 0
        self.release_frames = 0
        self.active = False

    def update(self, condition: bool, confirm_frames: int, release_frames: int) -> bool:
        """Update the latch and return True only on an inactive->active edge."""

        was_active = self.active
        if condition:
            self.candidate_frames += 1
            self.release_frames = 0
            if not self.active and self.candidate_frames >= confirm_frames:
                self.active = True
                self.activations += 1
        else:
            # UNKNOWN/missing evidence must never advance a candidate.
            self.candidate_frames = 0
            if self.active:
                self.release_frames += 1
                if self.release_frames >= release_frames:
                    self.active = False
                    self.release_frames = 0
            else:
                self.release_frames = 0
        return (not was_active) and self.active


@dataclass
class TrackState:
    """All temporal state is isolated by ByteTrack track ID."""

    track_id: int
    motion_history: deque[MotionSample] = field(
        default_factory=lambda: deque(maxlen=90)
    )
    fork_history: deque[HeightSample] = field(default_factory=lambda: deque(maxlen=120))
    direction_evidence: deque[str] = field(default_factory=lambda: deque(maxlen=90))
    last_seen_frame: int = -1
    last_motion_frame: int = -1
    last_raw_center: np.ndarray | None = None
    compensated_center: np.ndarray | None = None

    moving_state: bool | None = None
    turning_state: bool | None = None
    turn_true_frames: int = 0
    turn_false_frames: int = 0
    stable_direction: str = UNKNOWN
    stable_fork_state: str = UNKNOWN

    current_fork_state: str = UNKNOWN
    current_direction: str = UNKNOWN
    current_is_moving: bool | None = None
    current_is_turning: bool | None = None
    fork_relative_height: float | None = None
    fork_slope: float | None = None
    movement_speed: float | None = None
    movement_displacement: float | None = None
    turn_angle: float | None = None
    direction_cosine: float | None = None

    reverse_latch: TemporalLatch = field(default_factory=TemporalLatch)
    turn_latch: TemporalLatch = field(default_factory=TemporalLatch)

    def mark_missing(self, release_frames: int) -> None:
        """Invalidate current evidence and require a contiguous warm-up again."""

        del release_frames  # UNKNOWN suppresses immediately; no release grace.
        self.motion_history.clear()
        self.fork_history.clear()
        self.direction_evidence.clear()
        self.last_motion_frame = -1
        self.last_raw_center = None
        self.compensated_center = None
        self.moving_state = None
        self.turning_state = None
        self.turn_true_frames = 0
        self.turn_false_frames = 0
        self.stable_direction = UNKNOWN
        self.stable_fork_state = UNKNOWN
        self.current_fork_state = UNKNOWN
        self.current_direction = UNKNOWN
        self.current_is_moving = None
        self.current_is_turning = None
        self.fork_relative_height = None
        self.fork_slope = None
        self.movement_speed = None
        self.movement_displacement = None
        self.turn_angle = None
        self.direction_cosine = None
        self.reverse_latch.invalidate()
        self.turn_latch.invalidate()


@dataclass
class RunStats:
    frames_processed: int = 0
    tracked_rows: int = 0
    frames_with_tracks: int = 0
    segmentation_crops: int = 0
    fork_hits: int = 0
    mast_hits: int = 0
    paired_hits: int = 0
    segmentation_errors: int = 0
    camera_motion_valid_frames: int = 0
    camera_motion_errors: int = 0
    violation_frames: int = 0
    reverse_events: int = 0
    turn_events: int = 0


class GlobalMotionEstimator:
    """Estimate previous->current camera motion from background optical flow."""

    def __init__(self) -> None:
        self.previous_gray: np.ndarray | None = None
        self.previous_boxes: list[tuple[float, float, float, float]] = []

    @staticmethod
    def _background_mask(
        shape: tuple[int, int],
        boxes: Iterable[tuple[float, float, float, float]],
    ) -> np.ndarray:
        height, width = shape
        mask = np.full((height, width), 255, dtype=np.uint8)
        for raw_box in boxes:
            values = np.asarray(raw_box, dtype=np.float64)
            if values.shape != (4,) or not np.isfinite(values).all():
                continue
            x1, y1, x2, y2 = map(float, values)
            box_width, box_height = x2 - x1, y2 - y1
            if box_width <= 0.0 or box_height <= 0.0:
                continue
            pad_x, pad_y = 0.12 * box_width, 0.12 * box_height
            ix1 = max(0, min(width - 1, int(math.floor(x1 - pad_x))))
            iy1 = max(0, min(height - 1, int(math.floor(y1 - pad_y))))
            ix2 = max(0, min(width - 1, int(math.ceil(x2 + pad_x))))
            iy2 = max(0, min(height - 1, int(math.ceil(y2 + pad_y))))
            if ix2 > ix1 and iy2 > iy1:
                cv2.rectangle(mask, (ix1, iy1), (ix2, iy2), 0, -1)
        return mask

    def update(
        self,
        frame: np.ndarray,
        current_boxes: list[tuple[float, float, float, float]],
    ) -> CameraMotionEstimate:
        current_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        previous_gray = self.previous_gray
        previous_boxes = self.previous_boxes
        self.previous_gray = current_gray
        self.previous_boxes = list(current_boxes)
        if previous_gray is None or previous_gray.shape != current_gray.shape:
            return CameraMotionEstimate(False, None)

        previous_mask = self._background_mask(previous_gray.shape, previous_boxes)
        points = cv2.goodFeaturesToTrack(
            previous_gray,
            maxCorners=CAMERA_MAX_CORNERS,
            qualityLevel=0.01,
            minDistance=8,
            mask=previous_mask,
            blockSize=7,
        )
        if points is None or len(points) < CAMERA_MIN_TRACKED_POINTS:
            return CameraMotionEstimate(False, None, 0)

        next_points, status, errors = cv2.calcOpticalFlowPyrLK(
            previous_gray,
            current_gray,
            points,
            None,
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )
        if next_points is None or status is None:
            return CameraMotionEstimate(False, None, 0)

        old_points = points.reshape(-1, 2)
        new_points = next_points.reshape(-1, 2)
        valid = status.reshape(-1).astype(bool)
        valid &= np.isfinite(old_points).all(axis=1)
        valid &= np.isfinite(new_points).all(axis=1)
        if errors is not None:
            flat_errors = errors.reshape(-1)
            valid &= np.isfinite(flat_errors) & (flat_errors < 40.0)

        current_mask = self._background_mask(current_gray.shape, current_boxes)
        rounded = np.rint(new_points).astype(np.int64)
        inside = (
            (rounded[:, 0] >= 0)
            & (rounded[:, 0] < current_gray.shape[1])
            & (rounded[:, 1] >= 0)
            & (rounded[:, 1] < current_gray.shape[0])
        )
        valid &= inside
        valid_indices = np.flatnonzero(valid)
        if valid_indices.size:
            coords = rounded[valid_indices]
            background = current_mask[coords[:, 1], coords[:, 0]] > 0
            valid[valid_indices] &= background

        old_points = old_points[valid]
        new_points = new_points[valid]
        tracked_count = int(len(old_points))
        if tracked_count < CAMERA_MIN_TRACKED_POINTS:
            return CameraMotionEstimate(False, None, tracked_count)

        affine, inlier_mask = cv2.estimateAffinePartial2D(
            old_points,
            new_points,
            method=cv2.RANSAC,
            ransacReprojThreshold=2.5,
            maxIters=2000,
            confidence=0.99,
            refineIters=10,
        )
        if affine is None or inlier_mask is None or not np.isfinite(affine).all():
            return CameraMotionEstimate(False, None, tracked_count)

        inliers = int(np.count_nonzero(inlier_mask))
        ratio = inliers / max(tracked_count, 1)
        scale = math.hypot(float(affine[0, 0]), float(affine[1, 0]))
        rotation = abs(
            math.degrees(math.atan2(float(affine[1, 0]), float(affine[0, 0])))
        )
        translation = math.hypot(float(affine[0, 2]), float(affine[1, 2]))
        max_translation = 0.20 * math.hypot(
            current_gray.shape[1], current_gray.shape[0]
        )
        transform_is_sane = (
            abs(scale - 1.0) <= CAMERA_MAX_FRAME_SCALE_CHANGE
            and rotation <= CAMERA_MAX_FRAME_ROTATION_DEGREES
            and translation <= max_translation
        )
        is_valid = (
            inliers >= CAMERA_MIN_INLIERS
            and ratio >= CAMERA_MIN_INLIER_RATIO
            and transform_is_sane
        )
        return CameraMotionEstimate(
            is_valid,
            affine.astype(np.float64) if is_valid else None,
            tracked_count,
            inliers,
            ratio,
        )


CSV_FIELDS = [
    "frame",
    "timestamp_sec",
    "track_id",
    "fork_state",
    "fork_relative_height",
    "fork_slope",
    "is_moving",
    "is_turning",
    "turn_angle",
    "drive_direction",
    "direction_cosine",
    "reverse_lowering",
    "turn_lowering",
    "r7_violation",
    "forklift_confidence",
    "fork_confidence",
    "mast_confidence",
    "camera_motion_valid",
    "camera_motion_inliers",
    "camera_motion_inlier_ratio",
]


def positive_unit_interval(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or not 0.0 <= parsed <= 1.0:
        raise argparse.ArgumentTypeError("must be a finite number in [0, 1]")
    return parsed


def non_negative_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0.0:
        raise argparse.ArgumentTypeError("must be a finite non-negative number")
    return parsed


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Track forklifts and detect temporal R7 unsafe double actions "
            "(reverse+lowering or turning+lowering)."
        )
    )
    parser.add_argument(
        "--input",
        default=str(VIDEO_PATH),
        help=f"Input video path (configured default: {VIDEO_PATH}).",
    )
    parser.add_argument(
        "--output",
        help=f"Output MP4 (configured default: {OUTPUT_VIDEO_PATH}).",
    )
    parser.add_argument(
        "--output-csv",
        help=f"Per-track event CSV (configured default: {OUTPUT_CSV_PATH}).",
    )
    parser.add_argument(
        "--forklift-model",
        default=str(FORKLIFT_MODEL_PATH),
        help="Forklift detection checkpoint.",
    )
    parser.add_argument(
        "--r7-model",
        default=str(R7_SEGMENTATION_MODEL_PATH),
        help="Fork/mast segmentation checkpoint.",
    )
    parser.add_argument(
        "--tracker",
        default=str(TRACKER_CONFIG_PATH),
        help="ByteTrack YAML path (generated from the constants above by default).",
    )
    parser.add_argument(
        "--device",
        default=str(DEVICE),
        help="Ultralytics device such as auto, cpu, or 0.",
    )
    parser.add_argument(
        "--forklift-conf",
        type=positive_unit_interval,
        default=FORKLIFT_CONF_THRESHOLD,
        help=f"Detector pre-filter confidence (default: {FORKLIFT_CONF_THRESHOLD}).",
    )
    parser.add_argument(
        "--seg-conf",
        type=positive_unit_interval,
        default=SEGMENTATION_CONF_THRESHOLD,
        help=f"Fork/mast segmentation confidence (default: {SEGMENTATION_CONF_THRESHOLD}).",
    )
    parser.add_argument(
        "--crop-padding",
        type=non_negative_float,
        default=CROP_PADDING,
        help=f"Symmetric bbox padding as a bbox-size fraction (default: {CROP_PADDING}).",
    )
    parser.add_argument(
        "--crop-top-padding",
        type=non_negative_float,
        default=CROP_TOP_PADDING,
        help=(
            "Top padding as a bbox-height fraction; helps retain tall masts "
            f"when the detector covers only the body (default: {CROP_TOP_PADDING})."
        ),
    )
    parser.add_argument(
        "--max-frames",
        type=positive_int,
        default=MAX_FRAMES,
        help="Optional frame limit for a quick smoke test.",
    )
    parser.add_argument(
        "--overwrite",
        action=argparse.BooleanOptionalAction,
        default=OVERWRITE_OUTPUT,
        help="Replace existing outputs; use --no-overwrite for refusal mode.",
    )
    parser.add_argument(
        "--disable-camera-compensation",
        action="store_true",
        default=not ENABLE_CAMERA_COMPENSATION,
        help=(
            "Use raw bbox-center motion. Only appropriate for a known fixed "
            "camera; background optical-flow compensation is enabled by default."
        ),
    )
    parser.add_argument(
        "--debug",
        action=argparse.BooleanOptionalAction,
        default=DEBUG_OVERLAY,
        help="Draw numeric diagnostics.",
    )
    parser.add_argument(
        "--show",
        action=argparse.BooleanOptionalAction,
        default=SHOW_PREVIEW,
        help="Show a live OpenCV window.",
    )
    return parser


def normalize_class_name(value: Any) -> str:
    return "".join(char for char in str(value).casefold() if char.isalnum())


def class_items(names: Any) -> list[tuple[int, str]]:
    if isinstance(names, Mapping):
        items = names.items()
    elif isinstance(names, Sequence) and not isinstance(names, (str, bytes)):
        items = enumerate(names)
    else:
        raise ValueError(f"Unsupported model.names value: {type(names).__name__}")
    converted: list[tuple[int, str]] = []
    for raw_id, raw_name in items:
        try:
            class_id = int(raw_id)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid class ID {raw_id!r} in model.names") from exc
        converted.append((class_id, str(raw_name)))
    return converted


def resolve_class_id(names: Any, aliases: Iterable[str], role: str) -> int:
    normalized_aliases = {normalize_class_name(alias) for alias in aliases}
    matches = [
        (class_id, name)
        for class_id, name in class_items(names)
        if normalize_class_name(name) in normalized_aliases
    ]
    if len(matches) != 1:
        available = ", ".join(f"{idx}:{name}" for idx, name in class_items(names))
        raise ValueError(
            f"Expected exactly one {role} class, found {matches or 'none'}. "
            f"Available classes: {available}"
        )
    return matches[0][0]


def resolve_cli_path(value: str) -> Path:
    return Path(value).expanduser().resolve()


def write_default_tracker_config(path: Path, fps: float) -> None:
    """Materialize ByteTrack YAML from the single source of truth above."""

    if not 0.0 <= TRACK_LOW_THRESHOLD <= TRACK_HIGH_THRESHOLD <= 1.0:
        raise ValueError(
            "Tracker thresholds must satisfy 0 <= low <= high <= 1; got "
            f"{TRACK_LOW_THRESHOLD}, {TRACK_HIGH_THRESHOLD}"
        )
    if not 0.0 <= NEW_TRACK_THRESHOLD <= 1.0:
        raise ValueError(f"Invalid NEW_TRACK_THRESHOLD: {NEW_TRACK_THRESHOLD}")
    if TRACK_BUFFER_SECONDS <= 0.0 or not math.isfinite(TRACK_BUFFER_SECONDS):
        raise ValueError(f"Invalid TRACK_BUFFER_SECONDS: {TRACK_BUFFER_SECONDS}")
    if not 0.0 <= MATCH_THRESHOLD <= 1.0:
        raise ValueError(f"Invalid MATCH_THRESHOLD: {MATCH_THRESHOLD}")

    track_buffer_frames = max(1, int(math.ceil(TRACK_BUFFER_SECONDS * fps)))
    content = (
        "# Generated by r7_demo.py from its top-level tracker constants.\n"
        "tracker_type: bytetrack\n"
        f"track_high_thresh: {TRACK_HIGH_THRESHOLD}\n"
        f"track_low_thresh: {TRACK_LOW_THRESHOLD}\n"
        f"new_track_thresh: {NEW_TRACK_THRESHOLD}\n"
        f"track_buffer: {track_buffer_frames}\n"
        f"match_thresh: {MATCH_THRESHOLD}\n"
        f"fuse_score: {str(FUSE_SCORE).lower()}\n"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    current = path.read_text(encoding="utf-8") if path.is_file() else None
    if current != content:
        path.write_text(content, encoding="utf-8")


def resolve_tracker(value: str) -> str:
    candidate = Path(value).expanduser()
    if candidate.exists():
        return str(candidate.resolve())
    # Ultralytics ships these short built-in config names.
    if value in {"bytetrack.yaml", "botsort.yaml"}:
        return value
    raise FileNotFoundError(f"Tracker config not found: {candidate.resolve()}")


def validate_model(model: YOLO, expected_task: str, path: Path) -> None:
    actual = str(getattr(model, "task", "")).casefold()
    if actual != expected_task:
        raise ValueError(
            f"Wrong model task for {path}: expected {expected_task!r}, got {actual!r}"
        )


def validate_tracking_runtime() -> None:
    """Fail clearly instead of relying on Ultralytics auto-install while offline."""

    try:
        __import__("lap")
    except ImportError as exc:
        raise RuntimeError(
            "ByteTrack requires the optional package 'lap>=0.5.12'. "
            "Install it before running: python -m pip install 'lap>=0.5.12'"
        ) from exc


def derive_output_paths(
    input_path: Path,
    output_arg: str | None,
    csv_arg: str | None,
) -> tuple[Path, Path]:
    configured_input = VIDEO_PATH.resolve()
    if output_arg:
        output_path = resolve_cli_path(output_arg)
    elif input_path == configured_input:
        output_path = OUTPUT_VIDEO_PATH.resolve()
    else:
        output_path = (PROJECT_ROOT / "outputs" / f"{input_path.stem}_r7.mp4").resolve()
    if csv_arg:
        csv_path = resolve_cli_path(csv_arg)
    elif output_arg is None and input_path == configured_input:
        csv_path = OUTPUT_CSV_PATH.resolve()
    else:
        csv_path = output_path.with_name(f"{output_path.stem}_events.csv")
    return output_path, csv_path


def sanitize_fps(raw_fps: float) -> tuple[float, bool]:
    if math.isfinite(raw_fps) and raw_fps > 0.0:
        return float(raw_fps), False
    return FALLBACK_FPS, True


def sanitize_total_frames(raw_total: float) -> int | None:
    if math.isfinite(raw_total) and raw_total > 0:
        return int(raw_total)
    return None


def frame_count_for(seconds: float, fps: float) -> int:
    return max(1, int(math.ceil(seconds * fps)))


def extract_tracked_detections(
    result: Any, forklift_class_id: int
) -> list[TrackedDetection]:
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0 or getattr(boxes, "id", None) is None:
        return []

    xyxy = boxes.xyxy.detach().cpu().numpy()
    track_ids = boxes.id.detach().cpu().numpy()
    confidences = boxes.conf.detach().cpu().numpy()
    classes = boxes.cls.detach().cpu().numpy()
    count = min(len(xyxy), len(track_ids), len(confidences), len(classes))

    by_track: dict[int, TrackedDetection] = {}
    for index in range(count):
        raw_values = np.asarray(xyxy[index], dtype=np.float64)
        if raw_values.shape != (4,) or not np.isfinite(raw_values).all():
            continue
        track_value = float(track_ids[index])
        class_value = float(classes[index])
        confidence = float(confidences[index])
        if not all(
            math.isfinite(value) for value in (track_value, class_value, confidence)
        ):
            continue
        class_id = int(round(class_value))
        if class_id != forklift_class_id:
            continue
        x1, y1, x2, y2 = map(float, raw_values)
        if x2 <= x1 or y2 <= y1:
            continue
        track_id = int(round(track_value))
        detection = TrackedDetection(track_id, (x1, y1, x2, y2), confidence)
        previous = by_track.get(track_id)
        if previous is None or detection.confidence > previous.confidence:
            by_track[track_id] = detection
    return [by_track[track_id] for track_id in sorted(by_track)]


def extract_all_forklift_boxes(
    result: Any, forklift_class_id: int
) -> list[tuple[float, float, float, float]]:
    """Return tracked and untracked forklift boxes for background exclusion."""

    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return []
    xyxy = boxes.xyxy.detach().cpu().numpy()
    classes = boxes.cls.detach().cpu().numpy()
    output: list[tuple[float, float, float, float]] = []
    for raw_box, raw_class in zip(xyxy, classes):
        values = np.asarray(raw_box, dtype=np.float64)
        class_value = float(raw_class)
        if (
            values.shape != (4,)
            or not np.isfinite(values).all()
            or not math.isfinite(class_value)
            or int(round(class_value)) != forklift_class_id
        ):
            continue
        x1, y1, x2, y2 = map(float, values)
        if x2 > x1 and y2 > y1:
            output.append((x1, y1, x2, y2))
    return output


def append_motion_sample(
    state: TrackState,
    frame_index: int,
    timestamp: float,
    bbox: tuple[float, float, float, float],
    camera_motion: CameraMotionEstimate,
    compensate_camera: bool,
) -> None:
    """Append a raw or camera-stabilized bbox-center trajectory sample."""

    x1, y1, x2, y2 = bbox
    width, height = x2 - x1, y2 - y1
    raw_center = np.asarray(((x1 + x2) / 2.0, (y1 + y2) / 2.0), dtype=np.float64)
    contiguous = state.last_motion_frame == frame_index - 1

    if not compensate_camera:
        stabilized = raw_center
    elif (
        contiguous
        and state.last_raw_center is not None
        and state.compensated_center is not None
        and camera_motion.valid
        and camera_motion.affine is not None
    ):
        homogeneous_previous = np.asarray(
            (state.last_raw_center[0], state.last_raw_center[1], 1.0),
            dtype=np.float64,
        )
        expected_current = camera_motion.affine @ homogeneous_previous
        residual = raw_center - expected_current
        # Reject a nonsensical single-frame residual before it pollutes turning.
        if float(np.linalg.norm(residual)) > 1.5 * max(1.0, math.hypot(width, height)):
            state.motion_history.clear()
            stabilized = raw_center
        else:
            stabilized = state.compensated_center + residual
    else:
        # Invalid background transform or a tracking gap breaks continuity.  The
        # current point becomes a fresh anchor and needs a new warm-up window.
        state.motion_history.clear()
        stabilized = raw_center

    state.last_raw_center = raw_center
    state.compensated_center = np.asarray(stabilized, dtype=np.float64)
    state.last_motion_frame = frame_index
    state.motion_history.append(
        MotionSample(
            timestamp=timestamp,
            cx=float(stabilized[0]),
            cy=float(stabilized[1]),
            width=width,
            height=height,
        )
    )


def padded_crop_box(
    bbox: tuple[float, float, float, float],
    frame_shape: tuple[int, ...],
    padding: float,
    top_padding: float,
) -> tuple[int, int, int, int] | None:
    frame_height, frame_width = frame_shape[:2]
    values = np.asarray(bbox, dtype=np.float64)
    if values.shape != (4,) or not np.isfinite(values).all():
        return None
    x1, y1, x2, y2 = map(float, values)
    width, height = x2 - x1, y2 - y1
    if width <= 0.0 or height <= 0.0:
        return None

    pad_x = padding * width
    pad_bottom = padding * height
    pad_top = max(padding, top_padding) * height
    crop_x1 = max(0, int(math.floor(x1 - pad_x)))
    crop_y1 = max(0, int(math.floor(y1 - pad_top)))
    crop_x2 = min(frame_width, int(math.ceil(x2 + pad_x)))
    crop_y2 = min(frame_height, int(math.ceil(y2 + pad_bottom)))
    if crop_x2 <= crop_x1 or crop_y2 <= crop_y1:
        return None
    return crop_x1, crop_y1, crop_x2, crop_y2


def mask_geometry(mask: np.ndarray) -> tuple[tuple[float, float], int] | None:
    ys, xs = np.nonzero(mask)
    area = int(xs.size)
    if area < 4:
        return None
    return (float(xs.mean()), float(ys.mean())), area


def principal_mask_axis(mask: np.ndarray) -> tuple[np.ndarray, float, float] | None:
    ys, xs = np.nonzero(mask)
    if xs.size < 8:
        return None
    stride = max(1, xs.size // 6000)
    points = np.column_stack((xs[::stride], ys[::stride])).astype(np.float64)
    centered = points - points.mean(axis=0, keepdims=True)
    covariance = centered.T @ centered / max(1, len(centered) - 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)
    major_value = float(eigenvalues[order[-1]])
    minor_value = max(float(eigenvalues[order[0]]), 1e-9)
    if not math.isfinite(major_value) or major_value <= 0.0:
        return None
    elongation = math.sqrt(major_value / minor_value)
    axis = eigenvectors[:, int(order[-1])]
    norm = float(np.linalg.norm(axis))
    if not math.isfinite(norm) or norm < 1e-9:
        return None
    axis = axis / norm
    if axis[1] < 0.0:
        axis = -axis
    projections = centered @ axis
    extent = float(np.quantile(projections, 0.95) - np.quantile(projections, 0.05))
    if not math.isfinite(extent) or extent <= 1.0:
        return None
    return axis.astype(np.float64), extent, elongation


def resize_binary_mask(mask: np.ndarray, width: int, height: int) -> np.ndarray:
    binary = np.asarray(mask) > 0.5
    if binary.shape != (height, width):
        binary = cv2.resize(
            binary.astype(np.uint8),
            (width, height),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
    return binary


def build_mask_candidates(
    result: Any,
    crop_shape: tuple[int, ...],
    valid_class_ids: set[int],
    seg_conf: float,
) -> list[MaskObservation]:
    boxes = getattr(result, "boxes", None)
    masks = getattr(result, "masks", None)
    mask_data = getattr(masks, "data", None) if masks is not None else None
    if boxes is None or len(boxes) == 0 or mask_data is None:
        return []

    classes = boxes.cls.detach().cpu().numpy()
    confidences = boxes.conf.detach().cpu().numpy()
    tensors = mask_data.detach().cpu().numpy()
    count = min(len(classes), len(confidences), len(tensors))
    crop_height, crop_width = crop_shape[:2]
    crop_area = max(1.0, float(crop_height * crop_width))
    candidates: list[MaskObservation] = []

    for index in range(count):
        class_value = float(classes[index])
        confidence = float(confidences[index])
        if not math.isfinite(class_value) or not math.isfinite(confidence):
            continue
        class_id = int(round(class_value))
        if class_id not in valid_class_ids:
            continue
        if confidence < seg_conf:
            continue
        binary = resize_binary_mask(tensors[index], crop_width, crop_height)
        geometry = mask_geometry(binary)
        if geometry is None:
            continue
        centroid, area = geometry
        area_ratio = area / crop_area
        dx = (centroid[0] - crop_width / 2.0) / max(crop_width, 1)
        dy = (centroid[1] - crop_height / 2.0) / max(crop_height, 1)
        center_score = max(0.0, 1.0 - math.hypot(dx, dy) / math.sqrt(0.5))
        area_score = min(1.0, math.sqrt(max(0.0, area_ratio) / 0.05))
        score = 0.72 * confidence + 0.18 * area_score + 0.10 * center_score
        axis_data = principal_mask_axis(binary)
        axis, extent, elongation = (
            axis_data if axis_data is not None else (None, None, None)
        )
        candidates.append(
            MaskObservation(
                class_id=class_id,
                confidence=confidence,
                mask=binary,
                centroid=centroid,
                area=area,
                score=score,
                principal_axis=axis,
                axis_extent=extent,
                axis_elongation=elongation,
            )
        )
    return candidates


def merge_mask_candidates(candidates: list[MaskObservation]) -> MaskObservation | None:
    if not candidates:
        return None
    merged = np.zeros_like(candidates[0].mask, dtype=bool)
    weighted_confidence = 0.0
    weight_sum = 0.0
    for candidate in candidates:
        merged |= candidate.mask
        weight = max(1, candidate.area)
        weighted_confidence += candidate.confidence * weight
        weight_sum += weight
    geometry = mask_geometry(merged)
    if geometry is None:
        return None
    centroid, area = geometry
    axis_data = principal_mask_axis(merged)
    axis, extent, elongation = (
        axis_data if axis_data is not None else (None, None, None)
    )
    return MaskObservation(
        class_id=candidates[0].class_id,
        confidence=weighted_confidence / max(weight_sum, 1.0),
        mask=merged,
        centroid=centroid,
        area=area,
        score=max(candidate.score for candidate in candidates),
        principal_axis=axis,
        axis_extent=extent,
        axis_elongation=elongation,
    )


def select_segmentation(
    result: Any,
    crop_shape: tuple[int, ...],
    fork_class_id: int,
    mast_class_id: int,
    seg_conf: float,
) -> tuple[MaskObservation | None, MaskObservation | None]:
    candidates = build_mask_candidates(
        result,
        crop_shape,
        {fork_class_id, mast_class_id},
        seg_conf,
    )
    fork_candidates = [item for item in candidates if item.class_id == fork_class_id]
    mast_candidates = [
        item
        for item in candidates
        if item.class_id == mast_class_id
        and item.principal_axis is not None
        and item.axis_extent is not None
        and item.axis_elongation is not None
        and item.axis_elongation >= MIN_MAST_ELONGATION
    ]

    # A crop should contain one forklift mast.  Choosing the dominant scored mask
    # prevents an unrelated duplicate from shifting the mast reference axis.
    selected_mast = (
        max(mast_candidates, key=lambda item: (item.score, item.confidence, item.area))
        if mast_candidates
        else None
    )

    # Fork annotations can contain one instance per tine.  Keep all reasonably
    # confident candidates near the best candidate and merge their binary masks.
    selected_forks: list[MaskObservation] = []
    if fork_candidates:
        crop_height, crop_width = crop_shape[:2]
        crop_diag = max(1.0, math.hypot(crop_width, crop_height))

        def fork_rank(item: MaskObservation) -> tuple[float, float, int]:
            pair_bonus = 0.0
            if selected_mast is not None:
                pair_distance = (
                    math.dist(item.centroid, selected_mast.centroid) / crop_diag
                )
                pair_bonus = 0.10 * max(0.0, 1.0 - pair_distance)
            return item.score + pair_bonus, item.confidence, item.area

        primary = max(fork_candidates, key=fork_rank)
        confidence_floor = max(seg_conf, 0.50 * primary.confidence)
        for candidate in sorted(
            fork_candidates,
            key=lambda item: tuple(-value for value in fork_rank(item)),
        ):
            distance = math.dist(candidate.centroid, primary.centroid) / crop_diag
            if (
                candidate.confidence >= confidence_floor
                and distance <= MAX_FORK_MERGE_DISTANCE
            ):
                selected_forks.append(candidate)
    return merge_mask_candidates(selected_forks), selected_mast


def relative_fork_height(
    fork: MaskObservation | None,
    mast: MaskObservation | None,
    bbox_height: float,
) -> float | None:
    if fork is None or mast is None or bbox_height <= 1.0:
        return None
    axis = mast.principal_axis
    if (
        axis is None
        or mast.axis_extent is None
        or mast.axis_extent < 5.0
        or mast.axis_elongation is None
        or mast.axis_elongation < MIN_MAST_ELONGATION
    ):
        return None
    # A nearly horizontal "mast" is likely an erroneous mask; do not invent a
    # vertical direction in that case.
    if abs(float(axis[1])) < 0.20:
        return None
    offset = np.asarray(fork.centroid, dtype=np.float64) - np.asarray(
        mast.centroid, dtype=np.float64
    )
    axis_distance = float(np.dot(offset, axis))
    perpendicular = offset - axis_distance * axis
    if abs(axis_distance) / bbox_height > MAX_PAIR_AXIS_DISTANCE:
        return None
    if (
        float(np.linalg.norm(perpendicular)) / bbox_height
        > MAX_PAIR_PERPENDICULAR_DISTANCE
    ):
        return None
    value = axis_distance / bbox_height
    return value if math.isfinite(value) else None


def recent_motion_samples(
    samples: deque[MotionSample], timestamp: float
) -> list[MotionSample]:
    cutoff = timestamp - MOTION_WINDOW_SECONDS
    return [sample for sample in samples if sample.timestamp >= cutoff]


def linear_velocity(samples: list[MotionSample]) -> tuple[np.ndarray, float] | None:
    if len(samples) < 3:
        return None
    times = np.asarray([sample.timestamp for sample in samples], dtype=np.float64)
    if times[-1] - times[0] <= 1e-6:
        return None
    positions = np.asarray(
        [(sample.cx, sample.cy) for sample in samples], dtype=np.float64
    )
    centered_times = times - times.mean()
    denominator = float(np.dot(centered_times, centered_times))
    if denominator <= 1e-12:
        return None
    velocity = centered_times @ positions / denominator
    bbox_diagonal = float(
        np.median([math.hypot(sample.width, sample.height) for sample in samples])
    )
    if bbox_diagonal <= 1.0 or not np.isfinite(velocity).all():
        return None
    return velocity.astype(np.float64), bbox_diagonal


def robust_endpoint(samples: list[MotionSample]) -> tuple[np.ndarray, np.ndarray]:
    group_size = max(2, min(4, len(samples) // 3))
    first = np.median(
        np.asarray([(sample.cx, sample.cy) for sample in samples[:group_size]]), axis=0
    )
    last = np.median(
        np.asarray([(sample.cx, sample.cy) for sample in samples[-group_size:]]), axis=0
    )
    return first.astype(np.float64), last.astype(np.float64)


def trajectory_turn(
    samples: list[MotionSample], bbox_diagonal: float
) -> tuple[bool | None, float | None]:
    if len(samples) < MIN_MOTION_SAMPLES:
        return None, None
    midpoint = len(samples) // 2
    first_leg = samples[:midpoint]
    second_leg = samples[midpoint:]
    if len(first_leg) < 3 or len(second_leg) < 3:
        return None, None
    first_fit = linear_velocity(first_leg)
    second_fit = linear_velocity(second_leg)
    if first_fit is None or second_fit is None:
        return None, None

    first_start, first_end = robust_endpoint(first_leg)
    second_start, second_end = robust_endpoint(second_leg)
    first_disp = float(np.linalg.norm(first_end - first_start) / bbox_diagonal)
    second_disp = float(np.linalg.norm(second_end - second_start) / bbox_diagonal)
    if (
        first_disp < MIN_TURN_LEG_DISPLACEMENT
        or second_disp < MIN_TURN_LEG_DISPLACEMENT
    ):
        return False, None

    vector_a = first_fit[0]
    vector_b = second_fit[0]
    norm_a = float(np.linalg.norm(vector_a))
    norm_b = float(np.linalg.norm(vector_b))
    if norm_a <= 1e-9 or norm_b <= 1e-9:
        return False, None
    cosine = float(np.dot(vector_a, vector_b) / (norm_a * norm_b))
    angle = math.degrees(math.acos(float(np.clip(cosine, -1.0, 1.0))))
    # A near-180 degree change is usually a forward/reverse transition, not a
    # spatial turn, so it is deliberately excluded from the turning rule.
    is_turn = TURN_ENTER_DEGREES <= angle < TURN_REVERSAL_CUTOFF_DEGREES
    return is_turn, angle


def estimate_motion(state: TrackState, timestamp: float) -> MotionEstimate | None:
    samples = recent_motion_samples(state.motion_history, timestamp)
    if len(samples) < MIN_MOTION_SAMPLES:
        return None
    timestamps = np.asarray([sample.timestamp for sample in samples], dtype=np.float64)
    if timestamps[-1] - timestamps[0] < MIN_MOTION_SPAN_SECONDS:
        return None
    if (
        len(timestamps) > 1
        and float(np.max(np.diff(timestamps))) > MAX_MOTION_GAP_SECONDS
    ):
        return None
    fit = linear_velocity(samples)
    if fit is None:
        return None
    velocity, bbox_diagonal = fit
    speed = float(np.linalg.norm(velocity) / bbox_diagonal)
    first, last = robust_endpoint(samples)
    net_displacement = float(np.linalg.norm(last - first) / bbox_diagonal)

    if state.moving_state is True:
        moving = (
            speed >= MOVING_EXIT_SPEED
            and net_displacement >= MIN_NET_DISPLACEMENT / 2.0
        )
    else:
        moving = (
            speed >= MOVING_ENTER_SPEED and net_displacement >= MIN_NET_DISPLACEMENT
        )
    turn_candidate, turn_angle = trajectory_turn(samples, bbox_diagonal)
    return MotionEstimate(
        moving=moving,
        velocity=velocity / bbox_diagonal,
        speed=speed,
        net_displacement=net_displacement,
        turn_candidate=turn_candidate,
        turn_angle=turn_angle,
    )


def update_turn_state(
    state: TrackState,
    estimate: MotionEstimate | None,
    fps: float,
) -> bool | None:
    if estimate is None:
        state.turning_state = None
        state.turn_true_frames = 0
        state.turn_false_frames = 0
        return None
    if not estimate.moving:
        state.turning_state = False
        state.turn_true_frames = 0
        state.turn_false_frames = 0
        return False
    if estimate.turn_candidate is None:
        return state.turning_state

    enter_threshold = TURN_ENTER_DEGREES
    if state.turning_state:
        candidate = (
            estimate.turn_angle is not None
            and TURN_EXIT_DEGREES <= estimate.turn_angle < TURN_REVERSAL_CUTOFF_DEGREES
        )
    else:
        candidate = (
            estimate.turn_angle is not None
            and enter_threshold <= estimate.turn_angle < TURN_REVERSAL_CUTOFF_DEGREES
        )

    if candidate:
        state.turn_true_frames += 1
        state.turn_false_frames = 0
        if state.turn_true_frames >= frame_count_for(TURN_CONFIRM_SECONDS, fps):
            state.turning_state = True
    else:
        state.turn_true_frames = 0
        state.turn_false_frames += 1
        if state.turn_false_frames >= frame_count_for(TURN_RELEASE_SECONDS, fps):
            state.turning_state = False
    return state.turning_state


def median_filter(values: np.ndarray) -> np.ndarray:
    if len(values) < 3:
        return values.copy()
    filtered = values.copy()
    for index in range(1, len(values) - 1):
        filtered[index] = float(np.median(values[index - 1 : index + 2]))
    return filtered


def robust_fork_slope(samples: list[HeightSample]) -> float | None:
    if len(samples) < MIN_FORK_SAMPLES:
        return None
    times = np.asarray([sample.timestamp for sample in samples], dtype=np.float64)
    values = median_filter(
        np.asarray([sample.value for sample in samples], dtype=np.float64)
    )
    if times[-1] - times[0] < MIN_FORK_SPAN_SECONDS:
        return None
    gaps = np.diff(times)
    if len(gaps) and float(np.max(gaps)) > FORK_MAX_GAP_SECONDS:
        return None
    centered = times - times.mean()
    denominator = float(np.dot(centered, centered))
    if denominator <= 1e-12:
        return None
    initial_slope = float(np.dot(centered, values - values.mean()) / denominator)
    intercept = float(values.mean() - initial_slope * times.mean())
    residuals = values - (initial_slope * times + intercept)
    median_residual = float(np.median(residuals))
    mad = float(np.median(np.abs(residuals - median_residual)))
    keep = np.ones(len(values), dtype=bool)
    if mad > 1e-8:
        keep = np.abs(residuals - median_residual) <= 3.5 * 1.4826 * mad
    if int(keep.sum()) < MIN_FORK_SAMPLES:
        return None
    kept_times = times[keep]
    kept_values = values[keep]
    kept_centered = kept_times - kept_times.mean()
    kept_denominator = float(np.dot(kept_centered, kept_centered))
    if kept_denominator <= 1e-12:
        return None
    slope = float(
        np.dot(kept_centered, kept_values - kept_values.mean()) / kept_denominator
    )
    fitted_change = abs(slope) * float(kept_times[-1] - kept_times[0])
    if abs(slope) >= min(FORK_LOWER_ENTER_SLOPE, abs(FORK_RAISE_ENTER_SLOPE)):
        if fitted_change < MIN_FORK_DYNAMIC_CHANGE:
            slope = 0.0
    return slope if math.isfinite(slope) else None


def classify_fork_state(state: TrackState, slope: float | None) -> str:
    if slope is None:
        return UNKNOWN
    previous = state.stable_fork_state
    if previous == LOWERING and slope >= FORK_LOWER_EXIT_SLOPE:
        classification = LOWERING
    elif previous == RAISING and slope <= FORK_RAISE_EXIT_SLOPE:
        classification = RAISING
    elif slope >= FORK_LOWER_ENTER_SLOPE:
        classification = LOWERING
    elif slope <= FORK_RAISE_ENTER_SLOPE:
        classification = RAISING
    else:
        classification = STATIC
    state.stable_fork_state = classification
    return classification


def instantaneous_direction(
    bbox: tuple[float, float, float, float],
    crop_box: tuple[int, int, int, int],
    fork: MaskObservation | None,
    mast: MaskObservation | None,
    motion: MotionEstimate | None,
) -> tuple[str, float | None]:
    if fork is None or motion is None or not motion.moving:
        return UNKNOWN, None
    x1, y1, x2, y2 = bbox
    bbox_width, bbox_height = x2 - x1, y2 - y1
    bbox_diagonal = math.hypot(bbox_width, bbox_height)
    if bbox_diagonal <= 1.0:
        return UNKNOWN, None
    crop_x1, crop_y1, _, _ = crop_box
    fork_full = np.asarray(
        (fork.centroid[0] + crop_x1, fork.centroid[1] + crop_y1), dtype=np.float64
    )
    vehicle_center = np.asarray(((x1 + x2) / 2.0, (y1 + y2) / 2.0), dtype=np.float64)
    front_vector = fork_full - vehicle_center

    # Fork lift/lower motion changes the raw front vector.  Remove its component
    # along the mast axis when that axis is trustworthy.
    if mast is not None and mast.principal_axis is not None:
        mast_axis = np.asarray(mast.principal_axis, dtype=np.float64)
        front_vector = front_vector - float(np.dot(front_vector, mast_axis)) * mast_axis

    front_norm = float(np.linalg.norm(front_vector))
    movement_norm = float(np.linalg.norm(motion.velocity))
    if front_norm / bbox_diagonal < MIN_FRONT_VECTOR or movement_norm <= 1e-9:
        return UNKNOWN, None
    cosine = float(np.dot(front_vector, motion.velocity) / (front_norm * movement_norm))
    cosine = float(np.clip(cosine, -1.0, 1.0))
    if cosine >= DIRECTION_FORWARD_COSINE:
        return FORWARD, cosine
    if cosine <= DIRECTION_REVERSE_COSINE:
        return REVERSE, cosine
    return UNKNOWN, cosine


def update_direction_state(
    state: TrackState,
    instantaneous: str,
    current_has_evidence: bool,
    fps: float,
) -> str:
    if not current_has_evidence:
        state.direction_evidence.clear()
        state.stable_direction = UNKNOWN
        return UNKNOWN
    state.direction_evidence.append(instantaneous)
    window_size = frame_count_for(DIRECTION_CONFIRM_SECONDS, fps)
    evidence = list(state.direction_evidence)[-window_size:]
    required = max(2, int(math.ceil(window_size * DIRECTION_REQUIRED_RATIO)))
    counts = Counter(evidence)

    proposed = UNKNOWN
    if len(evidence) >= window_size:
        if counts[FORWARD] >= required:
            proposed = FORWARD
        elif counts[REVERSE] >= required:
            proposed = REVERSE

    if proposed != UNKNOWN:
        # Switching direction requires the same full confirmation as entering it.
        state.stable_direction = proposed
    elif len(evidence) >= window_size and all(value == UNKNOWN for value in evidence):
        state.stable_direction = UNKNOWN

    # Even when historical evidence is stable, a missing current fork/movement
    # observation is shown as UNKNOWN and cannot advance an R7 candidate.
    if instantaneous == UNKNOWN or instantaneous != state.stable_direction:
        return UNKNOWN
    return state.stable_direction


def blend_mask(
    frame: np.ndarray,
    observation: MaskObservation | None,
    crop_box: tuple[int, int, int, int],
    color: tuple[int, int, int],
    alpha: float = 0.34,
) -> None:
    if observation is None:
        return
    x1, y1, x2, y2 = crop_box
    region = frame[y1:y2, x1:x2]
    mask = observation.mask
    if mask.shape != region.shape[:2]:
        mask = cv2.resize(
            mask.astype(np.uint8),
            (region.shape[1], region.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
    if not np.any(mask):
        return
    original = region[mask].astype(np.float32)
    tint = np.asarray(color, dtype=np.float32)
    region[mask] = np.clip(original * (1.0 - alpha) + tint * alpha, 0, 255).astype(
        np.uint8
    )
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(region, contours, -1, color, 1, cv2.LINE_AA)


def format_optional(value: float | None, digits: int = 3) -> str:
    return "N/A" if value is None or not math.isfinite(value) else f"{value:.{digits}f}"


def tri_state(value: bool | None) -> str:
    if value is None:
        return UNKNOWN
    return "TRUE" if value else "FALSE"


def tri_and(*values: bool | None) -> bool | None:
    """Strict evidence AND: any UNKNOWN suppresses the temporal alert."""

    if any(value is None for value in values):
        return None
    return all(bool(value) for value in values)


def draw_label_lines(
    frame: np.ndarray,
    lines: list[str],
    anchor: tuple[int, int],
    color: tuple[int, int, int],
) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.48
    thickness = 1
    line_height = 20
    padding = 5
    widths = [
        cv2.getTextSize(line, font, font_scale, thickness)[0][0] for line in lines
    ]
    box_width = max(widths, default=0) + 2 * padding
    box_height = line_height * len(lines) + 2 * padding
    frame_height, frame_width = frame.shape[:2]
    x = max(0, min(anchor[0], max(0, frame_width - box_width - 1)))
    y = anchor[1] - box_height
    if y < 0:
        y = min(frame_height - box_height - 1, anchor[1] + 4)
    y = max(0, y)
    cv2.rectangle(frame, (x, y), (x + box_width, y + box_height), (18, 18, 18), -1)
    cv2.rectangle(frame, (x, y), (x + box_width, y + box_height), color, 1)
    for index, line in enumerate(lines):
        baseline_y = y + padding + 15 + index * line_height
        cv2.putText(
            frame,
            line,
            (x + padding, baseline_y),
            font,
            font_scale,
            (245, 245, 245),
            thickness,
            cv2.LINE_AA,
        )


def draw_header(
    frame: np.ndarray,
    frame_index: int,
    active_tracks: int,
    active_violations: int,
    violation_frames: int,
) -> None:
    header = (
        f"R7 Unsafe Double Actions | Frame {frame_index} | Active forklifts: {active_tracks} | "
        f"Violations: {active_violations} | Violation frames: {violation_frames}"
    )
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.54
    text_width = cv2.getTextSize(header, font, font_scale, 1)[0][0]
    if text_width > frame.shape[1] - 18:
        font_scale = max(0.32, font_scale * (frame.shape[1] - 18) / max(text_width, 1))
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 34), (12, 12, 12), -1)
    cv2.putText(
        frame,
        header,
        (9, 23),
        font,
        font_scale,
        (245, 245, 245),
        1,
        cv2.LINE_AA,
    )


def csv_row(
    frame_index: int,
    timestamp: float,
    detection: TrackedDetection,
    state: TrackState,
    fork: MaskObservation | None,
    mast: MaskObservation | None,
    reverse_lowering: bool,
    turn_lowering: bool,
    camera_motion: CameraMotionEstimate,
    camera_compensation_enabled: bool,
) -> dict[str, Any]:
    active_rules: list[str] = []
    if state.reverse_latch.active:
        active_rules.append("R7_REVERSE_LOWERING")
    if state.turn_latch.active:
        active_rules.append("R7_TURN_LOWERING")
    return {
        "frame": frame_index,
        "timestamp_sec": f"{timestamp:.3f}",
        "track_id": detection.track_id,
        "fork_state": state.current_fork_state,
        "fork_relative_height": format_optional(state.fork_relative_height, 6)
        if state.fork_relative_height is not None
        else "",
        "fork_slope": format_optional(state.fork_slope, 6)
        if state.fork_slope is not None
        else "",
        "is_moving": tri_state(state.current_is_moving),
        "is_turning": tri_state(state.current_is_turning),
        "turn_angle": format_optional(state.turn_angle, 3)
        if state.turn_angle is not None
        else "",
        "drive_direction": state.current_direction,
        "direction_cosine": format_optional(state.direction_cosine, 5)
        if state.direction_cosine is not None
        else "",
        "reverse_lowering": int(reverse_lowering),
        "turn_lowering": int(turn_lowering),
        "r7_violation": ";".join(active_rules) if active_rules else "NONE",
        "forklift_confidence": f"{detection.confidence:.5f}",
        "fork_confidence": f"{fork.confidence:.5f}" if fork is not None else "",
        "mast_confidence": f"{mast.confidence:.5f}" if mast is not None else "",
        "camera_motion_valid": (
            int(camera_motion.valid) if camera_compensation_enabled else "DISABLED"
        ),
        "camera_motion_inliers": (
            camera_motion.inliers if camera_compensation_enabled else ""
        ),
        "camera_motion_inlier_ratio": (
            f"{camera_motion.inlier_ratio:.4f}" if camera_compensation_enabled else ""
        ),
    }


def print_progress(
    stats: RunStats,
    total_frames: int | None,
    started_at: float,
) -> None:
    elapsed = max(time.monotonic() - started_at, 1e-9)
    processing_fps = stats.frames_processed / elapsed
    total_text = str(total_frames) if total_frames is not None else "?"
    print(
        f"[progress] {stats.frames_processed}/{total_text} frames | "
        f"{processing_fps:.2f} FPS | tracks rows={stats.tracked_rows} | "
        f"R7 frames={stats.violation_frames}",
        flush=True,
    )


def resolve_device(device: str | int) -> str:
    device_text = str(device).strip()
    requested = "" if device_text.casefold() == "auto" else device_text
    selected = select_device(requested, verbose=False)
    return str(selected)


def model_call_kwargs(device: str) -> dict[str, Any]:
    device = str(device)
    normalized = device.strip().casefold()
    kwargs: dict[str, Any] = {"device": device}
    # Ultralytics 8.4 uses quantize=16 instead of the deprecated half=True.
    if normalized.startswith("cuda"):
        kwargs["quantize"] = FP16_PRECISION
    return kwargs


def process_video(args: argparse.Namespace) -> RunStats:
    input_path = resolve_cli_path(str(args.input))
    detector_path = resolve_cli_path(str(args.forklift_model))
    segmenter_path = resolve_cli_path(str(args.r7_model))
    tracker_arg = str(args.tracker)
    output_path, events_path = derive_output_paths(
        input_path, args.output, args.output_csv
    )

    for label, path in (
        ("input video", input_path),
        ("forklift model", detector_path),
        ("R7 segmentation model", segmenter_path),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{label.capitalize()} not found: {path}")
    if input_path == output_path:
        raise ValueError("Input and output must be different files")
    if input_path == events_path or output_path == events_path:
        raise ValueError("Input video, output video, and output CSV must be distinct")
    if output_path.suffix.casefold() != ".mp4":
        raise ValueError(f"Output must use the .mp4 extension: {output_path}")
    if events_path.suffix.casefold() != ".csv":
        raise ValueError(f"Event output must use the .csv extension: {events_path}")
    if not args.overwrite:
        existing = [path for path in (output_path, events_path) if path.exists()]
        if existing:
            joined = ", ".join(str(path) for path in existing)
            raise FileExistsError(
                f"Output already exists ({joined}); pass --overwrite to replace it"
            )
    validate_tracking_runtime()

    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"Cannot open input video: {input_path}")

    writer: cv2.VideoWriter | None = None
    csv_handle: Any = None
    show_enabled = bool(args.show)
    stats = RunStats()
    started_at = time.monotonic()

    try:
        ok, frame = capture.read()
        if not ok or frame is None or frame.size == 0:
            raise RuntimeError(f"Input video contains no readable frames: {input_path}")
        frame_height, frame_width = frame.shape[:2]
        if frame_width <= 0 or frame_height <= 0:
            raise RuntimeError(f"Invalid first-frame dimensions: {frame.shape}")

        fps, used_fallback_fps = sanitize_fps(float(capture.get(cv2.CAP_PROP_FPS)))
        total_frames = sanitize_total_frames(
            float(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        )
        default_tracker_requested = (
            Path(tracker_arg).expanduser().resolve() == TRACKER_CONFIG_PATH.resolve()
        )
        if default_tracker_requested:
            write_default_tracker_config(TRACKER_CONFIG_PATH, fps)
        tracker = resolve_tracker(tracker_arg)
        resolved_device = resolve_device(str(args.device))
        if used_fallback_fps:
            print(
                f"[warning] Invalid source FPS; using fallback {fps:.2f}",
                file=sys.stderr,
            )

        print(f"Input:      {input_path}")
        print(f"Output:     {output_path}")
        print(f"Events CSV: {events_path}")
        print(f"Detector:   {detector_path}")
        print(f"Segmenter:  {segmenter_path}")
        print(f"Tracker:    {tracker}")
        print(
            f"Video:      {frame_width}x{frame_height} @ {fps:.3f} FPS, "
            f"frames={total_frames if total_frames is not None else 'unknown'}"
        )
        print(f"Device:     {resolved_device} (requested: {args.device})")
        print(
            f"Inference:  det_conf={args.forklift_conf}, seg_conf={args.seg_conf}, "
            f"det_imgsz={IMAGE_SIZE}, seg_imgsz={SEGMENTATION_IMAGE_SIZE}"
        )
        print(
            "Safety:     missing/ambiguous mask or motion evidence => UNKNOWN; "
            "no R7 candidate"
        )
        print(
            "Camera:     "
            + (
                "background optical-flow compensation enabled; invalid estimate => UNKNOWN"
                if not args.disable_camera_compensation
                else "compensation DISABLED (use only for a known fixed camera)"
            )
        )
        if default_tracker_requested and args.forklift_conf >= TRACK_HIGH_THRESHOLD:
            print(
                f"[warning] --forklift-conf >= {TRACK_HIGH_THRESHOLD} disables "
                "the bundled ByteTrack low-score recovery band.",
                file=sys.stderr,
            )

        print("Loading models once...")
        detector = YOLO(str(detector_path))
        segmenter = YOLO(str(segmenter_path))
        validate_model(detector, "detect", detector_path)
        validate_model(segmenter, "segment", segmenter_path)
        forklift_class_id = resolve_class_id(
            detector.names, {"forklift", "forklifts"}, "forklift detector"
        )
        fork_class_id = resolve_class_id(
            segmenter.names, {"fork", "forks", "forktine", "forktines"}, "fork mask"
        )
        mast_class_id = resolve_class_id(segmenter.names, {"mast"}, "mast mask")
        if fork_class_id == mast_class_id:
            raise ValueError("Fork and mast classes resolve to the same class ID")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        events_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(
            str(output_path), fourcc, fps, (frame_width, frame_height)
        )
        if not writer.isOpened():
            raise RuntimeError(f"Cannot open output video writer: {output_path}")
        csv_handle = events_path.open("w", newline="", encoding="utf-8")
        csv_writer = csv.DictWriter(csv_handle, fieldnames=CSV_FIELDS)
        csv_writer.writeheader()

        states: dict[int, TrackState] = {}
        camera_estimator = GlobalMotionEstimator()
        max_missing_frames = frame_count_for(TRACK_STATE_TTL_SECONDS, fps)
        release_frames = frame_count_for(R7_RELEASE_SECONDS, fps)
        confirm_frames = frame_count_for(R7_CONFIRM_SECONDS, fps)
        call_kwargs = model_call_kwargs(resolved_device)
        frame_index = 0
        stopped_early = False

        while True:
            timestamp = frame_index / fps
            track_results = detector.track(
                frame,
                persist=True,
                tracker=tracker,
                classes=[forklift_class_id],
                conf=args.forklift_conf,
                imgsz=IMAGE_SIZE,
                verbose=False,
                **call_kwargs,
            )
            track_result = track_results[0] if track_results else None
            detections = (
                extract_tracked_detections(track_result, forklift_class_id)
                if track_result is not None
                else []
            )
            all_forklift_boxes = (
                extract_all_forklift_boxes(track_result, forklift_class_id)
                if track_result is not None
                else []
            )
            if args.disable_camera_compensation:
                camera_motion = CameraMotionEstimate(True, None)
            else:
                try:
                    camera_motion = camera_estimator.update(frame, all_forklift_boxes)
                except cv2.error as exc:
                    stats.camera_motion_errors += 1
                    camera_estimator = GlobalMotionEstimator()
                    camera_motion = CameraMotionEstimate(False, None)
                    if stats.camera_motion_errors <= 3 or args.debug:
                        print(
                            f"[warning] Camera compensation failed at frame "
                            f"{frame_index}: {exc}",
                            file=sys.stderr,
                        )
            if camera_motion.valid and not args.disable_camera_compensation:
                stats.camera_motion_valid_frames += 1
            current_track_ids = {detection.track_id for detection in detections}

            # Any gap breaks a candidate immediately.  Active alerts get only the
            # short configured release grace, never new candidate credit.
            for track_id, state in list(states.items()):
                if track_id not in current_track_ids:
                    state.mark_missing(release_frames)
                if frame_index - state.last_seen_frame > max_missing_frames:
                    del states[track_id]

            crop_records: list[
                tuple[TrackedDetection, tuple[int, int, int, int], np.ndarray]
            ] = []
            for detection in detections:
                crop_box = padded_crop_box(
                    detection.bbox,
                    frame.shape,
                    args.crop_padding,
                    args.crop_top_padding,
                )
                if crop_box is None:
                    continue
                crop_x1, crop_y1, crop_x2, crop_y2 = crop_box
                crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]
                if crop.size == 0:
                    continue
                crop_records.append((detection, crop_box, crop.copy()))

            segmentation_results: list[Any] = []
            if crop_records:
                stats.segmentation_crops += len(crop_records)
                try:
                    raw_segmentation_results = segmenter.predict(
                        [record[2] for record in crop_records],
                        conf=args.seg_conf,
                        imgsz=SEGMENTATION_IMAGE_SIZE,
                        classes=[fork_class_id, mast_class_id],
                        retina_masks=True,
                        verbose=False,
                        **call_kwargs,
                    )
                    segmentation_results = list(raw_segmentation_results)
                except Exception as exc:  # keep the safety pipeline alive as UNKNOWN
                    stats.segmentation_errors += 1
                    if stats.segmentation_errors <= 3 or args.debug:
                        print(
                            f"[warning] Segmentation failed at frame {frame_index}: {exc}",
                            file=sys.stderr,
                        )

            frame_violation_track_ids: set[int] = set()
            processed_detection_ids: set[int] = set()

            for crop_index, (detection, crop_box, crop) in enumerate(crop_records):
                processed_detection_ids.add(detection.track_id)
                seg_result = (
                    segmentation_results[crop_index]
                    if crop_index < len(segmentation_results)
                    else None
                )
                if seg_result is not None:
                    fork, mast = select_segmentation(
                        seg_result,
                        crop.shape,
                        fork_class_id,
                        mast_class_id,
                        args.seg_conf,
                    )
                else:
                    fork, mast = None, None

                if fork is not None:
                    stats.fork_hits += 1
                if mast is not None:
                    stats.mast_hits += 1
                if fork is not None and mast is not None:
                    stats.paired_hits += 1

                state = states.setdefault(
                    detection.track_id, TrackState(detection.track_id)
                )
                x1, y1, x2, y2 = detection.bbox
                bbox_height = y2 - y1
                append_motion_sample(
                    state,
                    frame_index,
                    timestamp,
                    detection.bbox,
                    camera_motion,
                    compensate_camera=not args.disable_camera_compensation,
                )
                state.last_seen_frame = frame_index
                motion = estimate_motion(state, timestamp)
                if motion is None:
                    state.current_is_moving = None
                    state.movement_speed = None
                    state.movement_displacement = None
                else:
                    state.moving_state = motion.moving
                    state.current_is_moving = motion.moving
                    state.movement_speed = motion.speed
                    state.movement_displacement = motion.net_displacement
                state.current_is_turning = update_turn_state(state, motion, fps)
                state.turn_angle = motion.turn_angle if motion is not None else None

                relative_height = relative_fork_height(fork, mast, bbox_height)
                state.fork_relative_height = relative_height
                if relative_height is not None:
                    state.fork_history.append(HeightSample(timestamp, relative_height))
                    recent_heights = [
                        sample
                        for sample in state.fork_history
                        if sample.timestamp >= timestamp - FORK_WINDOW_SECONDS
                    ]
                    state.fork_slope = robust_fork_slope(recent_heights)
                    state.current_fork_state = classify_fork_state(
                        state, state.fork_slope
                    )
                else:
                    state.fork_history.clear()
                    state.stable_fork_state = UNKNOWN
                    state.fork_slope = None
                    state.current_fork_state = UNKNOWN

                instant_direction, direction_cosine = instantaneous_direction(
                    detection.bbox, crop_box, fork, mast, motion
                )
                state.direction_cosine = direction_cosine
                has_direction_evidence = (
                    fork is not None
                    and motion is not None
                    and motion.moving
                    and direction_cosine is not None
                )
                state.current_direction = update_direction_state(
                    state,
                    instant_direction,
                    has_direction_evidence,
                    fps,
                )

                reverse_condition = tri_and(
                    state.current_is_moving,
                    None
                    if state.current_direction == UNKNOWN
                    else state.current_direction == REVERSE,
                    None
                    if state.current_fork_state == UNKNOWN
                    else state.current_fork_state == LOWERING,
                )
                turn_condition = tri_and(
                    state.current_is_moving,
                    state.current_is_turning,
                    None
                    if state.current_fork_state == UNKNOWN
                    else state.current_fork_state == LOWERING,
                )
                reverse_lowering = reverse_condition is True
                turn_lowering = turn_condition is True
                if reverse_condition is None:
                    state.reverse_latch.invalidate()
                elif state.reverse_latch.update(
                    reverse_condition, confirm_frames, release_frames
                ):
                    stats.reverse_events += 1
                if turn_condition is None:
                    state.turn_latch.invalidate()
                elif state.turn_latch.update(
                    turn_condition, confirm_frames, release_frames
                ):
                    stats.turn_events += 1
                if state.reverse_latch.active or state.turn_latch.active:
                    frame_violation_track_ids.add(detection.track_id)

                blend_mask(frame, mast, crop_box, MAST_COLOR)
                blend_mask(frame, fork, crop_box, FORK_COLOR)

                has_violation = state.reverse_latch.active or state.turn_latch.active
                bbox_color = (
                    ALERT_COLOR
                    if has_violation
                    else UNKNOWN_COLOR
                    if (
                        state.current_fork_state == UNKNOWN
                        or state.current_direction == UNKNOWN
                        or state.current_is_turning is None
                    )
                    else CAUTION_COLOR
                    if state.current_direction == REVERSE
                    or state.current_is_turning is True
                    else NORMAL_COLOR
                )
                draw_x1 = max(0, min(frame_width - 1, int(round(x1))))
                draw_y1 = max(0, min(frame_height - 1, int(round(y1))))
                draw_x2 = max(0, min(frame_width - 1, int(round(x2))))
                draw_y2 = max(0, min(frame_height - 1, int(round(y2))))
                cv2.rectangle(
                    frame, (draw_x1, draw_y1), (draw_x2, draw_y2), bbox_color, 2
                )
                lines = [
                    f"ID {detection.track_id} forklift {detection.confidence:.2f}",
                    f"Fork: {state.current_fork_state}",
                    "Motion: "
                    + (
                        "MOVING"
                        if state.current_is_moving is True
                        else "STOPPED"
                        if state.current_is_moving is False
                        else UNKNOWN
                    ),
                    f"Direction: {state.current_direction}",
                    f"Turning: {tri_state(state.current_is_turning)}",
                ]
                if args.debug:
                    lines.extend(
                        [
                            f"fork_rel={format_optional(state.fork_relative_height)} "
                            f"slope={format_optional(state.fork_slope)}",
                            f"speed={format_optional(state.movement_speed)} "
                            f"disp={format_optional(state.movement_displacement)}",
                            f"turn={format_optional(state.turn_angle, 1)}deg "
                            f"cos={format_optional(state.direction_cosine)}",
                            "masks "
                            f"fork={format_optional(fork.confidence if fork else None, 2)} "
                            f"mast={format_optional(mast.confidence if mast else None, 2)}",
                        ]
                    )
                if has_violation:
                    active_names: list[str] = []
                    if state.reverse_latch.active:
                        active_names.append("REVERSE + LOWERING")
                    if state.turn_latch.active:
                        active_names.append("TURN + LOWERING")
                    lines.append("R7 VIOLATION: " + " | ".join(active_names))
                draw_label_lines(frame, lines, (draw_x1, draw_y1), bbox_color)

                csv_writer.writerow(
                    csv_row(
                        frame_index,
                        timestamp,
                        detection,
                        state,
                        fork,
                        mast,
                        reverse_lowering,
                        turn_lowering,
                        camera_motion,
                        not args.disable_camera_compensation,
                    )
                )
                stats.tracked_rows += 1

            # A valid tracked box with an invalid crop still gets its own state
            # invalidated; no stale rule evidence is reused.
            for detection in detections:
                if detection.track_id not in processed_detection_ids:
                    state = states.setdefault(
                        detection.track_id, TrackState(detection.track_id)
                    )
                    state.last_seen_frame = frame_index
                    state.mark_missing(release_frames)
                    raw_x1, raw_y1, raw_x2, raw_y2 = detection.bbox
                    invalid_x1 = max(0, min(frame_width - 1, int(round(raw_x1))))
                    invalid_y1 = max(0, min(frame_height - 1, int(round(raw_y1))))
                    invalid_x2 = max(0, min(frame_width - 1, int(round(raw_x2))))
                    invalid_y2 = max(0, min(frame_height - 1, int(round(raw_y2))))
                    cv2.rectangle(
                        frame,
                        (invalid_x1, invalid_y1),
                        (invalid_x2, invalid_y2),
                        UNKNOWN_COLOR,
                        2,
                    )
                    draw_label_lines(
                        frame,
                        [
                            f"ID {detection.track_id} forklift {detection.confidence:.2f}",
                            "Fork: UNKNOWN",
                            "Motion: UNKNOWN",
                            "Direction: UNKNOWN",
                            "Turning: UNKNOWN",
                            "Invalid crop; state suppressed",
                        ],
                        (invalid_x1, invalid_y1),
                        UNKNOWN_COLOR,
                    )
                    csv_writer.writerow(
                        csv_row(
                            frame_index,
                            timestamp,
                            detection,
                            state,
                            None,
                            None,
                            False,
                            False,
                            camera_motion,
                            not args.disable_camera_compensation,
                        )
                    )
                    stats.tracked_rows += 1

            if detections:
                stats.frames_with_tracks += 1
            if frame_violation_track_ids:
                stats.violation_frames += 1
            draw_header(
                frame,
                frame_index,
                len(detections),
                len(frame_violation_track_ids),
                stats.violation_frames,
            )
            writer.write(frame)
            stats.frames_processed += 1

            if show_enabled:
                try:
                    cv2.imshow("R7 forklift demo", frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (27, ord("q")):
                        print("Stopped by user (q/ESC).")
                        stopped_early = True
                        break
                except cv2.error as exc:
                    print(
                        f"[warning] OpenCV display unavailable: {exc}", file=sys.stderr
                    )
                    show_enabled = False

            if (
                PROGRESS_INTERVAL > 0
                and stats.frames_processed % PROGRESS_INTERVAL == 0
            ):
                csv_handle.flush()
                print_progress(stats, total_frames, started_at)
            if (
                args.max_frames is not None
                and stats.frames_processed >= args.max_frames
            ):
                print(f"Reached --max-frames={args.max_frames}.")
                stopped_early = True
                break

            ok, next_frame = capture.read()
            if not ok or next_frame is None:
                break
            if next_frame.shape[:2] != (frame_height, frame_width):
                raise RuntimeError(
                    "Input resolution changed mid-stream: "
                    f"expected {(frame_height, frame_width)}, got {next_frame.shape[:2]}"
                )
            frame = next_frame
            frame_index += 1

        csv_handle.flush()
        if (
            not stopped_early
            and total_frames is not None
            and stats.frames_processed < total_frames
        ):
            print(
                f"[warning] Video metadata reports {total_frames} frames, but OpenCV "
                f"could decode only {stats.frames_processed}.",
                file=sys.stderr,
            )
        print_progress(stats, total_frames, started_at)
        elapsed = max(time.monotonic() - started_at, 1e-9)
        print("\nCompleted R7 processing")
        print(f"  Frames processed:       {stats.frames_processed}")
        print(f"  Average processing FPS: {stats.frames_processed / elapsed:.2f}")
        print(f"  Track rows:             {stats.tracked_rows}")
        print(f"  Segmentation crops:     {stats.segmentation_crops}")
        print(f"  Fork/mast paired hits:  {stats.paired_hits}")
        if not args.disable_camera_compensation:
            print(
                "  Camera compensation:    "
                f"{stats.camera_motion_valid_frames}/{stats.frames_processed} valid frames"
            )
            if stats.camera_motion_errors:
                print(f"  Camera-motion errors:   {stats.camera_motion_errors}")
        print(f"  R7 violation frames:    {stats.violation_frames}")
        print(
            f"  R7 event activations:   {stats.reverse_events + stats.turn_events} "
            f"(reverse={stats.reverse_events}, turn={stats.turn_events})"
        )
        print(f"  Video:                  {output_path}")
        print(f"  CSV:                    {events_path}")
        if stats.segmentation_crops and stats.paired_hits == 0:
            print(
                "[warning] No trustworthy fork+mast pair was found. Fork state and "
                "R7 alerts correctly remain UNKNOWN/inactive. This usually indicates "
                "a segmentation domain/crop mismatch, not a temporal-rule shortcut.",
                file=sys.stderr,
            )
        return stats
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        if csv_handle is not None:
            csv_handle.close()
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        process_video(args)
    except KeyboardInterrupt:
        print("Interrupted by user.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        if args.debug:
            raise
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
