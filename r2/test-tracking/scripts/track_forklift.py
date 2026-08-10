"""Track forklifts with the fine-tuned YOLO26 model and a tuned ByteTrack.

The detector deliberately keeps low-confidence boxes so ByteTrack can recover
an existing forklift after a weak frame.  New tracks still need a stronger
score, and a temporal confirmation layer suppresses persistent pallet/rack
false positives before they are rendered as forklifts.
"""

from __future__ import annotations

import argparse
import json
import math
import tempfile
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import cv2
import numpy as np
import torch
import ultralytics
from ultralytics import YOLO


# Backward-compatible defaults: running the script without arguments still
# processes test2.mp4 with test-tracking/models/best_fresh.pt.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "best_fresh.pt"
VIDEO_PATH = PROJECT_ROOT / "video" / "test3.mp4"
OUTPUT_PATH = PROJECT_ROOT / "outputs" / "test3.mp4"

# For tracking we use YOLO26's one-to-many head with NMS.  Its scores are
# better calibrated for ByteTrack than the checkpoint's native NMS-free head.
# Boxes from 0.10-0.25 can recover a track, while a new ID needs >= 0.30.
CONF_THRESHOLD = 0.10
IOU_THRESHOLD = 0.55
IMAGE_SIZE = 1920
TRACK_LOW_THRESHOLD = 0.10
TRACK_HIGH_THRESHOLD = 0.25
NEW_TRACK_THRESHOLD = 0.30
TRACK_BUFFER_SECONDS = 2.0
MATCH_THRESHOLD = 0.85
FUSE_SCORE = True

CONFIRMATION_MODE = "motion"
MIN_CONFIRMATION_HITS = 6
CONFIRMATION_EVIDENCE_SECONDS = 0.5
MOTION_WINDOW_SECONDS = 2.0
MIN_MOTION_PIXELS = 30.0
MIN_MOTION_FRAME_DIAGONAL_RATIO = 0.02
VISUAL_MOTION_INTERVAL_SECONDS = 0.15
VISUAL_DIFFERENCE_THRESHOLD = 20
MIN_VISUAL_MOTION_RATIO = 0.10
STATIC_CONFIRMATION_HITS = 3
STATIC_CONFIDENCE_THRESHOLD = 0.50
BOX_SMOOTHING_ALPHA = 0.65
TRAJECTORY_LENGTH = 45
MAX_BOX_AREA_RATIO = 0.30
TOP_CORNER_REJECTION_AREA_RATIO = 0.10
TRACK_DEDUPLICATION_IOU = 0.65
TRACK_DEDUPLICATION_IOS = 0.80

DEVICE = "auto"
FP16_PRECISION = 16
PROGRESS_INTERVAL = 100


@dataclass(frozen=True)
class VideoMetadata:
    """Properties that must be preserved in the output video."""

    width: int
    height: int
    fps: float
    frame_count: int

    @property
    def duration_seconds(self) -> float:
        return self.frame_count / self.fps

    @property
    def frame_diagonal(self) -> float:
        return math.hypot(self.width, self.height)


@dataclass(frozen=True)
class OutputVerification:
    """Properties measured by reopening and decoding the completed output."""

    width: int
    height: int
    fps: float
    metadata_frame_count: int
    decoded_frame_count: int


@dataclass(frozen=True)
class TrackerSettings:
    """ByteTrack settings calibrated for low-confidence forklift boxes."""

    detector_confidence: float = CONF_THRESHOLD
    track_low_threshold: float = TRACK_LOW_THRESHOLD
    track_high_threshold: float = TRACK_HIGH_THRESHOLD
    new_track_threshold: float = NEW_TRACK_THRESHOLD
    buffer_seconds: float = TRACK_BUFFER_SECONDS
    match_threshold: float = MATCH_THRESHOLD
    fuse_score: bool = FUSE_SCORE

    def validate(self) -> None:
        values = {
            "detector confidence": self.detector_confidence,
            "track low threshold": self.track_low_threshold,
            "track high threshold": self.track_high_threshold,
            "new track threshold": self.new_track_threshold,
            "match threshold": self.match_threshold,
        }
        for name, value in values.items():
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {value}.")
        if self.detector_confidence > self.track_low_threshold:
            raise ValueError(
                "Detector confidence must be <= the ByteTrack low threshold; "
                "otherwise weak recovery detections are discarded too early."
            )
        if self.track_low_threshold >= self.track_high_threshold:
            raise ValueError("Track low threshold must be < track high threshold.")
        if self.new_track_threshold < self.track_high_threshold:
            raise ValueError("New track threshold must be >= track high threshold.")
        if self.buffer_seconds <= 0:
            raise ValueError("Track buffer seconds must be positive.")


@dataclass
class TrackState:
    """Temporal evidence and display state for one ByteTrack identity."""

    track_id: int
    positions: deque[tuple[int, float, float]]
    display_positions: deque[tuple[int, float, float]]
    confidences: deque[float]
    visual_motion: deque[float]
    smoothed_box: np.ndarray | None = None
    total_hits: int = 0
    last_seen_frame: int = 0
    confirmed: bool = False
    confirmation_reason: str | None = None


class TrackConfirmation:
    """Confirm real tracks and smooth their boxes without changing ByteTrack IDs."""

    def __init__(
        self,
        *,
        mode: str,
        motion_min_hits: int,
        motion_history: int,
        min_motion_pixels: float,
        min_visual_motion_ratio: float,
        static_hits: int,
        static_confidence: float,
        smoothing_alpha: float,
        trajectory_length: int,
    ) -> None:
        if mode not in {"motion", "hits", "none"}:
            raise ValueError(f"Unsupported confirmation mode: {mode}")
        if motion_min_hits <= 0:
            raise ValueError("Minimum confirmation hits must be positive.")
        if motion_history < motion_min_hits:
            raise ValueError("Motion history must be >= minimum confirmation hits.")
        if static_hits <= 0:
            raise ValueError("Static confirmation hits must be positive.")
        if min_motion_pixels < 0:
            raise ValueError("Minimum motion must be non-negative.")
        if not 0.0 <= min_visual_motion_ratio <= 1.0:
            raise ValueError("Visual motion ratio must be in [0, 1].")
        if not 0.0 <= static_confidence <= 1.0:
            raise ValueError("Static confidence must be in [0, 1].")
        if not 0.0 < smoothing_alpha <= 1.0:
            raise ValueError("Box smoothing alpha must be in (0, 1].")
        if trajectory_length < 2:
            raise ValueError("Trajectory length must be at least 2.")

        self.mode = mode
        self.motion_min_hits = motion_min_hits
        self.motion_history = motion_history
        self.min_motion_pixels = min_motion_pixels
        self.min_visual_motion_ratio = min_visual_motion_ratio
        self.static_hits = static_hits
        self.static_confidence = static_confidence
        self.smoothing_alpha = smoothing_alpha
        self.trajectory_length = trajectory_length
        self.states: dict[int, TrackState] = {}

    def update(
        self,
        track_id: int,
        box: Sequence[float],
        confidence: float,
        visual_motion_ratio: float,
        frame_index: int,
    ) -> TrackState:
        """Add one observation and return its updated temporal state."""
        raw_box = np.asarray(box, dtype=np.float32)
        state = self.states.get(track_id)
        if state is None:
            history_size = max(
                self.motion_history,
                self.static_hits,
                self.trajectory_length,
            )
            state = TrackState(
                track_id=track_id,
                positions=deque(maxlen=history_size),
                display_positions=deque(maxlen=self.trajectory_length),
                confidences=deque(
                    maxlen=max(self.static_hits, self.motion_min_hits)
                ),
                visual_motion=deque(maxlen=self.motion_history),
            )
            self.states[track_id] = state

        state.total_hits += 1
        state.last_seen_frame = frame_index
        ground_x, ground_y = bottom_center(raw_box)
        state.positions.append((frame_index, ground_x, ground_y))
        state.confidences.append(confidence)
        state.visual_motion.append(visual_motion_ratio)

        healthy_current_box = (
            visual_motion_ratio >= self.min_visual_motion_ratio
            or confidence >= self.static_confidence
        )
        if state.smoothed_box is None:
            state.smoothed_box = raw_box.copy()
        elif self.mode != "motion" or not state.confirmed or healthy_current_box:
            state.smoothed_box = (
                self.smoothing_alpha * raw_box
                + (1.0 - self.smoothing_alpha) * state.smoothed_box
            )

        if not state.confirmed:
            reason = self._confirmation_reason(state)
            if reason is not None:
                state.confirmed = True
                state.confirmation_reason = reason
        return state

    def _confirmation_reason(self, state: TrackState) -> str | None:
        if self.mode == "none":
            return "immediate"
        if self.mode == "hits":
            return (
                "temporal"
                if state.total_hits >= self.motion_min_hits
                else None
            )
        if state.total_hits >= self.motion_min_hits:
            motion = robust_net_motion(state.positions, self.motion_history)
            visual_evidence = list(state.visual_motion)[-self.motion_min_hits :]
            visual_motion = (
                float(np.median(visual_evidence))
                if visual_evidence
                else 0.0
            )
            if (
                motion >= self.min_motion_pixels
                and visual_motion >= self.min_visual_motion_ratio
            ):
                return "motion"
        if state.total_hits >= self.static_hits:
            recent = list(state.confidences)[-self.static_hits :]
            if recent and float(np.median(recent)) >= self.static_confidence:
                return "strong-static"
        return None

    @property
    def confirmed_count(self) -> int:
        return sum(state.confirmed for state in self.states.values())


def normalized_class_names(
    names: Mapping[int, str] | Sequence[str],
) -> dict[int, str]:
    """Normalize Ultralytics class-name mappings to integer keys."""
    if isinstance(names, Mapping):
        return {int(class_id): str(name) for class_id, name in names.items()}
    return {class_id: str(name) for class_id, name in enumerate(names)}


def get_forklift_class_id(names: Mapping[int, str] | Sequence[str]) -> int:
    """Find the single class named exactly ``forklift``, ignoring case/space."""
    class_names = normalized_class_names(names)
    matches = [
        class_id
        for class_id, class_name in class_names.items()
        if class_name.strip().casefold() == "forklift"
    ]

    available = ", ".join(
        f"{class_id}: {class_name!r}"
        for class_id, class_name in sorted(class_names.items())
    )
    if not matches:
        raise RuntimeError(
            "No class named exactly 'forklift' (case-insensitive) exists in "
            f"model.names. Available classes: {available}"
        )
    if len(matches) > 1:
        raise RuntimeError(
            "Multiple classes are named 'forklift'; the target is ambiguous. "
            f"Available classes: {available}"
        )
    return matches[0]


def get_video_metadata(capture: cv2.VideoCapture) -> VideoMetadata:
    """Read and validate source video metadata."""
    metadata = VideoMetadata(
        width=int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        height=int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        fps=float(capture.get(cv2.CAP_PROP_FPS)),
        frame_count=int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
    )
    if metadata.width <= 0 or metadata.height <= 0:
        raise RuntimeError(
            f"Invalid input resolution: {metadata.width}x{metadata.height}"
        )
    if metadata.fps <= 0:
        raise RuntimeError(f"Invalid input FPS: {metadata.fps}")
    return metadata


def print_video_metadata(metadata: VideoMetadata) -> None:
    """Print source resolution, rate, frame count, and duration."""
    print("\nInput Video")
    print("--------------------------------")
    print(f"Resolution: {metadata.width}x{metadata.height}")
    print(f"FPS: {metadata.fps:.3f}")
    print(f"Frames (metadata): {metadata.frame_count}")
    print(f"Duration: {metadata.duration_seconds:.1f} seconds")


def resolve_device(requested: str) -> int | str:
    """Resolve ``auto``, ``cpu``, ``mps``, or a CUDA device index."""
    value = requested.strip().casefold()
    if value == "auto":
        return 0 if torch.cuda.is_available() else "cpu"
    if value == "cpu":
        return "cpu"
    if value == "mps":
        mps = getattr(torch.backends, "mps", None)
        if mps is None or not mps.is_available():
            raise RuntimeError("MPS was requested but is unavailable.")
        return "mps"
    try:
        device_index = int(value)
    except ValueError as error:
        raise ValueError(
            f"Invalid device {requested!r}; use auto, cpu, mps, or a CUDA index."
        ) from error
    if not torch.cuda.is_available():
        raise RuntimeError(
            f"CUDA device {device_index} was requested but CUDA is unavailable."
        )
    if not 0 <= device_index < torch.cuda.device_count():
        raise RuntimeError(
            f"CUDA device {device_index} does not exist; available device "
            f"count: {torch.cuda.device_count()}"
        )
    return device_index


def print_environment(device: int | str, use_fp16: bool) -> None:
    """Print versions and the already validated inference device."""
    print("Runtime Environment")
    print("--------------------------------")
    print(f"PyTorch: {torch.__version__}")
    print(f"Ultralytics: {ultralytics.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if isinstance(device, int):
        print(f"Device: cuda:{device} ({torch.cuda.get_device_name(device)})")
        print(f"Inference precision: {'FP16' if use_fp16 else 'FP32'}")
    else:
        print(f"Device: {device}")
        print("Inference precision: FP32")


def write_tracker_config(
    output_path: Path,
    settings: TrackerSettings,
    fps: float,
) -> int:
    """Write a temporary ByteTrack YAML and return its frame-based buffer."""
    settings.validate()
    buffer_frames = max(1, int(round(settings.buffer_seconds * fps)))
    fuse_score = str(settings.fuse_score).lower()
    output_path.write_text(
        "\n".join(
            (
                "tracker_type: bytetrack",
                f"track_high_thresh: {settings.track_high_threshold}",
                f"track_low_thresh: {settings.track_low_threshold}",
                f"new_track_thresh: {settings.new_track_threshold}",
                f"track_buffer: {buffer_frames}",
                f"match_thresh: {settings.match_threshold}",
                f"fuse_score: {fuse_score}",
                "",
            )
        ),
        encoding="utf-8",
    )
    return buffer_frames


def bottom_center(box: Sequence[float]) -> tuple[float, float]:
    """Return the ground-contact proxy for a bounding box."""
    x1, _y1, x2, y2 = box
    return float((x1 + x2) / 2.0), float(y2)


def robust_net_motion(
    positions: Sequence[tuple[int, float, float]],
    window: int,
) -> float:
    """Measure displacement between robust early/late trajectory centers."""
    recent = list(positions)[-window:]
    if len(recent) < 2:
        return 0.0
    edge = min(3, max(1, len(recent) // 3))
    start_x = float(np.median([point[1] for point in recent[:edge]]))
    start_y = float(np.median([point[2] for point in recent[:edge]]))
    end_x = float(np.median([point[1] for point in recent[-edge:]]))
    end_y = float(np.median([point[2] for point in recent[-edge:]]))
    return math.hypot(end_x - start_x, end_y - start_y)


def box_iou(first: Sequence[float], second: Sequence[float]) -> float:
    """Return axis-aligned intersection-over-union for two xyxy boxes."""
    ax1, ay1, ax2, ay2 = first
    bx1, by1, bx2, by2 = second
    intersection_width = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    intersection_height = max(0.0, min(ay2, by2) - max(ay1, by1))
    intersection = intersection_width * intersection_height
    first_area = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    second_area = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = first_area + second_area - intersection
    return intersection / union if union > 0 else 0.0


def box_intersection_over_smaller(
    first: Sequence[float],
    second: Sequence[float],
) -> float:
    """Return intersection divided by the smaller box area."""
    ax1, ay1, ax2, ay2 = first
    bx1, by1, bx2, by2 = second
    intersection_width = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    intersection_height = max(0.0, min(ay2, by2) - max(ay1, by1))
    intersection = intersection_width * intersection_height
    first_area = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    second_area = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    smaller_area = min(first_area, second_area)
    return intersection / smaller_area if smaller_area > 0 else 0.0


def select_track_detections(
    track_data: np.ndarray,
    track_confirmation: TrackConfirmation,
    frame_shape: Sequence[int],
    forklift_class_id: int,
    roi: np.ndarray | None,
    max_box_area_ratio: float,
    reject_top_corner_boxes: bool,
    top_corner_area_ratio: float,
    deduplication_iou: float,
    deduplication_ios: float,
) -> tuple[list[np.ndarray], int, int]:
    """Apply geometry/ROI filters and suppress duplicate tracked boxes.

    Older confirmed IDs are preferred over newly spawned duplicate IDs.  This
    stabilizes labels when the YOLO26 end-to-end head emits overlapping boxes
    for the same forklift.
    """
    frame_area = float(frame_shape[0] * frame_shape[1])
    candidates: list[np.ndarray] = []
    geometry_rejected = 0
    for detection in track_data:
        box = detection[:4]
        class_id = int(detection[5])
        if class_id != forklift_class_id:
            continue
        width = max(0.0, float(box[2] - box[0]))
        height = max(0.0, float(box[3] - box[1]))
        area_ratio = width * height / frame_area
        if area_ratio > max_box_area_ratio:
            geometry_rejected += 1
            continue
        # On fixed, high-angle warehouse cameras, the dominant hallucination
        # from this checkpoint is a large wall/door box anchored to a top
        # corner.  Keep small/distant corner objects, but reject backdrop-sized
        # boxes.  The check can be disabled from the CLI for another camera.
        frame_height, frame_width = frame_shape[:2]
        touches_top = float(box[1]) <= 0.02 * frame_height
        touches_side = (
            float(box[0]) <= 0.02 * frame_width
            or float(box[2]) >= 0.98 * frame_width
        )
        if (
            reject_top_corner_boxes
            and area_ratio >= top_corner_area_ratio
            and touches_top
            and touches_side
        ):
            geometry_rejected += 1
            continue
        if not point_in_roi(bottom_center(box), roi):
            geometry_rejected += 1
            continue
        candidates.append(detection)

    def priority(detection: np.ndarray) -> tuple[int, int, float]:
        state = track_confirmation.states.get(int(detection[6]))
        return (
            int(state.confirmed) if state is not None else 0,
            state.total_hits if state is not None else 0,
            float(detection[4]),
        )

    selected: list[np.ndarray] = []
    duplicates_rejected = 0
    for detection in sorted(candidates, key=priority, reverse=True):
        if any(
            box_iou(detection[:4], kept[:4]) >= deduplication_iou
            or box_intersection_over_smaller(detection[:4], kept[:4])
            >= deduplication_ios
            for kept in selected
        ):
            duplicates_rejected += 1
            continue
        selected.append(detection)
    return selected, geometry_rejected, duplicates_rejected


def visual_motion_ratio(
    change_mask: np.ndarray | None,
    box: Sequence[float],
) -> float:
    """Return the changed-pixel fraction inside a clipped detection box."""
    if change_mask is None:
        return 0.0
    height, width = change_mask.shape[:2]
    x1, y1, x2, y2 = (int(round(value)) for value in box)
    x1 = max(0, min(x1, width))
    x2 = max(0, min(x2, width))
    y1 = max(0, min(y1, height))
    y2 = max(0, min(y2, height))
    if x2 <= x1 or y2 <= y1:
        return 0.0
    return float(np.mean(change_mask[y1:y2, x1:x2]))


def observation_is_renderable(
    state: TrackState,
    *,
    confirmation_mode: str,
    confidence: float,
    current_visual_motion: float,
    min_visual_motion: float,
    static_confidence: float,
) -> bool:
    """Reject a confirmed ID when its current box has drifted onto background.

    Confirmation remains latched, so a healthy box can reappear with the same
    ID.  Only rendering is suspended while both visual motion and detector
    confidence are weak.
    """
    if not state.confirmed:
        return False
    if confirmation_mode != "motion":
        return True
    return (
        current_visual_motion >= min_visual_motion
        or confidence >= static_confidence
    )


def load_tracking_roi(path: Path | None) -> np.ndarray | None:
    """Load a polygon or the ``tracking_roi`` field from a JSON file."""
    if path is None:
        return None
    if not path.is_file():
        raise FileNotFoundError(f"ROI configuration not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    points = data.get("tracking_roi") if isinstance(data, dict) else data
    polygon = np.asarray(points, dtype=np.float32)
    if polygon.ndim != 2 or polygon.shape[0] < 3 or polygon.shape[1] != 2:
        raise ValueError(
            "ROI JSON must be a list of at least three [x, y] points, or an "
            "object containing that list in 'tracking_roi'."
        )
    if not np.isfinite(polygon).all():
        raise ValueError("ROI coordinates must all be finite numbers.")
    return polygon


def point_in_roi(point: tuple[float, float], roi: np.ndarray | None) -> bool:
    """Return whether a point is accepted by the optional tracking ROI."""
    if roi is None:
        return True
    contour = roi.astype(np.float32).reshape((-1, 1, 2))
    return cv2.pointPolygonTest(contour, point, False) >= 0


def draw_roi(frame: np.ndarray, roi: np.ndarray | None) -> None:
    """Draw the configured tracking ROI."""
    if roi is None:
        return
    contour = np.rint(roi).astype(np.int32).reshape((-1, 1, 2))
    color = (255, 220, 0)
    cv2.polylines(frame, [contour], True, color, 2, cv2.LINE_AA)
    x, y = contour[0, 0]
    cv2.putText(
        frame,
        "TRACKING ROI",
        (int(x) + 5, max(22, int(y) - 7)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color,
        2,
        cv2.LINE_AA,
    )


def _clipped_box(frame: np.ndarray, box: Sequence[float]) -> tuple[int, int, int, int]:
    frame_height, frame_width = frame.shape[:2]
    x1, y1, x2, y2 = (int(round(value)) for value in box)
    x1 = max(0, min(x1, frame_width - 1))
    x2 = max(0, min(x2, frame_width - 1))
    y1 = max(0, min(y1, frame_height - 1))
    y2 = max(0, min(y2, frame_height - 1))
    return x1, y1, x2, y2


def draw_track(
    frame: np.ndarray,
    state: TrackState,
    confidence: float,
    *,
    draw_trajectory: bool,
    candidate: bool = False,
    box_override: Sequence[float] | None = None,
) -> None:
    """Draw one confirmed track, or an explicitly requested candidate."""
    box_to_draw = box_override if box_override is not None else state.smoothed_box
    if box_to_draw is None:
        return
    x1, y1, x2, y2 = _clipped_box(frame, box_to_draw)
    color = (145, 145, 145) if candidate else (0, 220, 0)

    if draw_trajectory and not candidate and len(state.display_positions) >= 2:
        recent = list(state.display_positions)[-TRAJECTORY_LENGTH:]
        for first, second in zip(recent, recent[1:]):
            # Do not imply a path through a long interval with no observation.
            if second[0] - first[0] > 2:
                continue
            cv2.line(
                frame,
                (int(round(first[1])), int(round(first[2]))),
                (int(round(second[1])), int(round(second[2]))),
                color,
                2,
                cv2.LINE_AA,
            )

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
    prefix = "Candidate" if candidate else "Forklift"
    label = f"{prefix} #{state.track_id} | {confidence:.2f}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    text_thickness = 2
    (text_width, text_height), baseline = cv2.getTextSize(
        label, font, font_scale, text_thickness
    )
    frame_height, frame_width = frame.shape[:2]
    text_x = max(0, min(x1, frame_width - text_width - 8))
    text_y = y1 - 8
    if text_y - text_height - baseline < 0:
        text_y = min(frame_height - baseline - 1, y1 + text_height + baseline + 8)
    background_top = max(0, text_y - text_height - baseline - 4)
    background_bottom = min(frame_height - 1, text_y + baseline + 3)
    cv2.rectangle(
        frame,
        (text_x, background_top),
        (min(frame_width - 1, text_x + text_width + 8), background_bottom),
        color,
        -1,
    )
    cv2.putText(
        frame,
        label,
        (text_x + 4, text_y),
        font,
        font_scale,
        (0, 0, 0),
        text_thickness,
        cv2.LINE_AA,
    )


def draw_status_panel(
    frame: np.ndarray,
    active_forklifts: int,
    active_candidates: int,
    confirmed_total: int,
    current_frame: int,
    total_frames: int,
    processing_fps: float,
) -> None:
    """Draw compact, explicit detector/tracker status."""
    total_text = str(total_frames) if total_frames > 0 else "?"
    lines = (
        f"Forklifts active: {active_forklifts}",
        f"Candidates active: {active_candidates}",
        f"Confirmed IDs: {confirmed_total}",
        f"Frame: {current_frame} / {total_text}",
        f"Processing FPS: {processing_fps:.1f}",
    )
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.52
    thickness = 1
    sizes = [cv2.getTextSize(line, font, font_scale, thickness)[0] for line in lines]
    panel_width = min(frame.shape[1], max(width for width, _ in sizes) + 22)
    panel_height = min(frame.shape[0], len(lines) * 24 + 12)
    overlay = frame[:panel_height, :panel_width]
    dark_panel = np.full_like(overlay, (20, 20, 20))
    cv2.addWeighted(dark_panel, 0.72, overlay, 0.28, 0, overlay)
    for line_number, line in enumerate(lines):
        cv2.putText(
            frame,
            line,
            (10, 22 + line_number * 24),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )


def inspect_output_video(output_path: Path) -> OutputVerification:
    """Reopen and decode the completed output instead of trusting write calls."""
    output_capture = cv2.VideoCapture(str(output_path))
    try:
        if not output_capture.isOpened():
            raise RuntimeError(f"Could not reopen output video: {output_path}")

        width = int(output_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(output_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(output_capture.get(cv2.CAP_PROP_FPS))
        metadata_frame_count = int(output_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        decoded_frame_count = 0
        while True:
            success, frame = output_capture.read()
            if not success:
                break
            if frame is None or frame.size == 0:
                raise RuntimeError(
                    f"Output decoder returned an empty frame at index {decoded_frame_count}."
                )
            if frame.shape[:2] != (height, width):
                raise RuntimeError(
                    "Output frame resolution changed at index "
                    f"{decoded_frame_count}: expected {width}x{height}, got "
                    f"{frame.shape[1]}x{frame.shape[0]}."
                )
            decoded_frame_count += 1

        return OutputVerification(
            width=width,
            height=height,
            fps=fps,
            metadata_frame_count=metadata_frame_count,
            decoded_frame_count=decoded_frame_count,
        )
    finally:
        output_capture.release()


def process_video(
    model: YOLO,
    forklift_class_id: int,
    *,
    video_path: Path,
    output_path: Path,
    device: int | str,
    use_fp16: bool,
    tracker_settings: TrackerSettings,
    tracker_override: str | None,
    image_size: int,
    iou_threshold: float,
    confirmation_mode: str,
    min_confirmation_hits: int,
    confirmation_evidence_seconds: float,
    motion_window_seconds: float,
    min_motion_pixel_floor: float,
    motion_threshold_ratio: float,
    visual_motion_interval_seconds: float,
    visual_difference_threshold: int,
    min_visual_motion_ratio: float,
    static_confirmation_hits: int,
    static_confidence: float,
    smoothing_alpha: float,
    max_box_area_ratio: float,
    reject_top_corner_boxes: bool,
    top_corner_area_ratio: float,
    deduplication_iou: float,
    deduplication_ios: float,
    roi: np.ndarray | None,
    show_candidates: bool,
    draw_trajectories: bool,
    max_frames: int | None,
) -> None:
    """Track forklifts in every decoded frame and write the output video."""
    if not video_path.is_file():
        raise FileNotFoundError(f"Input video not found: {video_path}")
    if video_path.resolve() == output_path.resolve():
        raise ValueError("Input and output video paths must be different.")

    capture = cv2.VideoCapture(str(video_path))
    writer: cv2.VideoWriter | None = None
    decoded_frames = 0
    written_frames = 0
    frames_with_confirmed_tracks = 0
    confirmed_observations = 0
    geometry_rejected_total = 0
    duplicates_rejected_total = 0
    start_time = time.perf_counter()

    try:
        if not capture.isOpened():
            raise RuntimeError(f"Could not open input video: {video_path}")
        metadata = get_video_metadata(capture)
        print_video_metadata(metadata)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            metadata.fps,
            (metadata.width, metadata.height),
        )
        if not writer.isOpened():
            raise RuntimeError(f"Could not initialize VideoWriter: {output_path}")

        evidence_frames = max(
            1,
            int(math.ceil(metadata.fps * confirmation_evidence_seconds)),
        )
        motion_min_hits = max(min_confirmation_hits, evidence_frames)
        motion_history = max(
            motion_min_hits,
            int(math.ceil(metadata.fps * motion_window_seconds)),
        )
        effective_static_hits = max(static_confirmation_hits, evidence_frames)
        min_motion_pixels = max(
            min_motion_pixel_floor,
            metadata.frame_diagonal * motion_threshold_ratio,
        )
        track_confirmation = TrackConfirmation(
            mode=confirmation_mode,
            motion_min_hits=motion_min_hits,
            motion_history=motion_history,
            min_motion_pixels=min_motion_pixels,
            min_visual_motion_ratio=min_visual_motion_ratio,
            static_hits=effective_static_hits,
            static_confidence=static_confidence,
            smoothing_alpha=smoothing_alpha,
            trajectory_length=TRAJECTORY_LENGTH,
        )
        visual_interval_frames = max(
            1,
            int(round(metadata.fps * visual_motion_interval_seconds)),
        )
        grayscale_history: deque[np.ndarray] = deque(
            maxlen=visual_interval_frames + 1
        )

        with tempfile.TemporaryDirectory(prefix="forklift_bytetrack_") as temp_dir:
            if tracker_override:
                tracker_path = tracker_override
                print(f"Tracker config: {tracker_path} (user supplied)")
            else:
                generated_path = Path(temp_dir) / "bytetrack_forklift.yaml"
                buffer_frames = write_tracker_config(
                    generated_path,
                    tracker_settings,
                    metadata.fps,
                )
                tracker_path = str(generated_path)
                print("\nTuned ByteTrack")
                print("--------------------------------")
                print(
                    "Detector / low / high / new: "
                    f"{tracker_settings.detector_confidence:.2f} / "
                    f"{tracker_settings.track_low_threshold:.2f} / "
                    f"{tracker_settings.track_high_threshold:.2f} / "
                    f"{tracker_settings.new_track_threshold:.2f}"
                )
                print(
                    f"Lost-track buffer: {buffer_frames} frames "
                    f"({tracker_settings.buffer_seconds:.2f}s)"
                )
                print(f"Match threshold: {tracker_settings.match_threshold:.2f}")
                print(f"Fuse detection score: {tracker_settings.fuse_score}")
                print(
                    f"Confirmation: {confirmation_mode}; motion >= "
                    f"{min_motion_pixels:.1f}px or sustained confidence >= "
                    f"{static_confidence:.2f}"
                )
                print(
                    f"Evidence: {motion_history} frame motion window; "
                    f"{motion_min_hits} motion hits / "
                    f"{effective_static_hits} strong hits"
                )
                print(
                    "Visual motion: median changed-pixel ratio >= "
                    f"{min_visual_motion_ratio:.2f} over "
                    f"{visual_interval_frames} frame interval"
                )

            with torch.inference_mode():
                while max_frames is None or decoded_frames < max_frames:
                    success, frame = capture.read()
                    if not success:
                        break
                    decoded_frames += 1

                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    gray = cv2.GaussianBlur(gray, (5, 5), 0)
                    grayscale_history.append(gray)
                    change_mask: np.ndarray | None = None
                    if len(grayscale_history) == grayscale_history.maxlen:
                        difference = cv2.absdiff(
                            grayscale_history[0],
                            grayscale_history[-1],
                        )
                        change_mask = difference >= visual_difference_threshold

                    results = model.track(
                        frame,
                        persist=True,
                        tracker=tracker_path,
                        classes=[forklift_class_id],
                        conf=tracker_settings.detector_confidence,
                        iou=iou_threshold,
                        imgsz=image_size,
                        device=device,
                        quantize=FP16_PRECISION if use_fp16 else None,
                        verbose=False,
                    )

                    active_forklifts = 0
                    active_candidates = 0
                    draw_roi(frame, roi)

                    if results:
                        boxes = results[0].boxes
                        if boxes is not None and len(boxes) > 0 and boxes.id is not None:
                            # Combine fields before a single GPU-to-CPU transfer.
                            track_data = torch.cat(
                                (
                                    boxes.xyxy,
                                    boxes.conf.unsqueeze(1),
                                    boxes.cls.unsqueeze(1),
                                    boxes.id.unsqueeze(1),
                                ),
                                dim=1,
                            ).detach().cpu().numpy()

                            selected, geometry_rejected, duplicates_rejected = (
                                select_track_detections(
                                    track_data,
                                    track_confirmation,
                                    frame.shape,
                                    forklift_class_id,
                                    roi,
                                    max_box_area_ratio,
                                    reject_top_corner_boxes,
                                    top_corner_area_ratio,
                                    deduplication_iou,
                                    deduplication_ios,
                                )
                            )
                            geometry_rejected_total += geometry_rejected
                            duplicates_rejected_total += duplicates_rejected

                            for detection in selected:
                                box = detection[:4]
                                confidence = float(detection[4])
                                track_id = int(detection[6])

                                current_visual_motion = visual_motion_ratio(
                                    change_mask,
                                    box,
                                )
                                state = track_confirmation.update(
                                    track_id,
                                    box,
                                    confidence,
                                    current_visual_motion,
                                    decoded_frames,
                                )
                                if observation_is_renderable(
                                    state,
                                    confirmation_mode=confirmation_mode,
                                    confidence=confidence,
                                    current_visual_motion=current_visual_motion,
                                    min_visual_motion=min_visual_motion_ratio,
                                    static_confidence=static_confidence,
                                ):
                                    ground_x, ground_y = bottom_center(box)
                                    state.display_positions.append(
                                        (decoded_frames, ground_x, ground_y)
                                    )
                                    active_forklifts += 1
                                    confirmed_observations += 1
                                    draw_track(
                                        frame,
                                        state,
                                        confidence,
                                        draw_trajectory=draw_trajectories,
                                    )
                                else:
                                    active_candidates += 1
                                    if show_candidates:
                                        draw_track(
                                            frame,
                                            state,
                                            confidence,
                                            draw_trajectory=False,
                                            candidate=True,
                                            box_override=box,
                                        )

                    if active_forklifts > 0:
                        frames_with_confirmed_tracks += 1
                    elapsed = max(time.perf_counter() - start_time, 1e-9)
                    processing_fps = decoded_frames / elapsed
                    draw_status_panel(
                        frame,
                        active_forklifts,
                        active_candidates,
                        track_confirmation.confirmed_count,
                        decoded_frames,
                        metadata.frame_count,
                        processing_fps,
                    )
                    writer.write(frame)
                    written_frames += 1

                    if decoded_frames % PROGRESS_INTERVAL == 0:
                        total_text = (
                            str(metadata.frame_count)
                            if metadata.frame_count > 0
                            else "?"
                        )
                        print(
                            f"[{decoded_frames}/{total_text}] "
                            f"FPS: {processing_fps:.1f} | "
                            f"confirmed: {active_forklifts} | "
                            f"candidates: {active_candidates}"
                        )
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()

    elapsed = time.perf_counter() - start_time
    average_fps = decoded_frames / elapsed if elapsed > 0 else 0.0
    output_verification = inspect_output_video(output_path)

    print("\nProcessing Summary")
    print("--------------------------------")
    print(f"Input frames (metadata): {metadata.frame_count}")
    print(f"Input frames (decoded): {decoded_frames}")
    print(f"Output frames (written): {written_frames}")
    print(f"Output frames (metadata): {output_verification.metadata_frame_count}")
    print(f"Output frames (decoded): {output_verification.decoded_frame_count}")
    print(f"Confirmed track IDs: {track_confirmation.confirmed_count}")
    print(f"Confirmed observations: {confirmed_observations}")
    print(f"Frames with confirmed tracks: {frames_with_confirmed_tracks}")
    print(f"Oversized/outside-ROI boxes rejected: {geometry_rejected_total}")
    print(f"Overlapping track boxes deduplicated: {duplicates_rejected_total}")
    print(f"Elapsed time: {elapsed:.2f} seconds")
    print(f"Average processing FPS: {average_fps:.2f}")
    print(f"Output path: {output_path}")

    if decoded_frames != written_frames or written_frames != output_verification.decoded_frame_count:
        raise RuntimeError(
            "Frame verification failed: "
            f"input decoded {decoded_frames}, write calls {written_frames}, "
            f"output decoded {output_verification.decoded_frame_count}."
        )
    if (output_verification.width, output_verification.height) != (
        metadata.width,
        metadata.height,
    ):
        raise RuntimeError(
            "Output resolution verification failed: "
            f"expected {metadata.width}x{metadata.height}, got "
            f"{output_verification.width}x{output_verification.height}."
        )
    if not math.isclose(output_verification.fps, metadata.fps, rel_tol=1e-3, abs_tol=1e-3):
        raise RuntimeError(
            "Output FPS verification failed: "
            f"expected {metadata.fps:.6g}, got {output_verification.fps:.6g}."
        )
    print(
        "Frame verification: OK - decoded input/output counts, resolution, and FPS match."
    )
    if max_frames is None and metadata.frame_count != decoded_frames:
        print(
            "NOTE: OpenCV input metadata differs from the actual decoded count "
            f"({metadata.frame_count} vs {decoded_frames})."
        )
    if output_verification.metadata_frame_count != written_frames:
        print(
            "WARNING: OpenCV output metadata differs from the written count "
            f"({output_verification.metadata_frame_count} vs {written_frames}); "
            "the decoded output count above is authoritative."
        )


def probability(value: str) -> float:
    """Argparse type for a probability in [0, 1]."""
    parsed = float(value)
    if not 0.0 <= parsed <= 1.0:
        raise argparse.ArgumentTypeError("must be between 0 and 1")
    return parsed


def positive_int(value: str) -> int:
    """Argparse type for a strictly positive integer."""
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def positive_float(value: str) -> float:
    """Argparse type for a strictly positive float."""
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def byte_value(value: str) -> int:
    """Argparse type for an 8-bit pixel threshold."""
    parsed = int(value)
    if not 1 <= parsed <= 255:
        raise argparse.ArgumentTypeError("must be an integer from 1 to 255")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line interface without loading the model."""
    parser = argparse.ArgumentParser(
        description=(
            "Track forklifts with best_fresh.pt, tuned ByteTrack thresholds, "
            "and temporal false-positive confirmation."
        )
    )
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    parser.add_argument("--source", "--video", dest="video", type=Path, default=VIDEO_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--device", default=DEVICE, help="auto, cpu, mps, or CUDA index")
    parser.add_argument("--imgsz", type=positive_int, default=IMAGE_SIZE)
    parser.add_argument("--conf", type=probability, default=CONF_THRESHOLD)
    parser.add_argument(
        "--iou",
        type=probability,
        default=IOU_THRESHOLD,
        help="NMS IoU used by the default one-to-many tracking head",
    )
    parser.add_argument(
        "--tracker",
        default=None,
        help="Optional Ultralytics tracker YAML/name; overrides tuned ByteTrack",
    )
    parser.add_argument("--track-low", type=probability, default=TRACK_LOW_THRESHOLD)
    parser.add_argument("--track-high", type=probability, default=TRACK_HIGH_THRESHOLD)
    parser.add_argument("--new-track", type=probability, default=NEW_TRACK_THRESHOLD)
    parser.add_argument(
        "--track-buffer-seconds",
        type=positive_float,
        default=TRACK_BUFFER_SECONDS,
    )
    parser.add_argument("--match-threshold", type=probability, default=MATCH_THRESHOLD)
    parser.add_argument(
        "--fuse-score",
        action=argparse.BooleanOptionalAction,
        default=FUSE_SCORE,
        help="Fuse detector confidence into ByteTrack association cost",
    )
    parser.add_argument(
        "--native-end2end",
        action="store_true",
        help="Use the checkpoint's native NMS-free head instead of NMS tracking mode",
    )
    parser.add_argument(
        "--confirmation-mode",
        choices=("motion", "hits", "none"),
        default=CONFIRMATION_MODE,
        help="motion suppresses static false positives; hits/none include parked objects",
    )
    parser.add_argument("--min-hits", type=positive_int, default=MIN_CONFIRMATION_HITS)
    parser.add_argument(
        "--confirmation-seconds",
        type=positive_float,
        default=CONFIRMATION_EVIDENCE_SECONDS,
        help="Minimum recent evidence duration, converted to frames from video FPS",
    )
    parser.add_argument(
        "--motion-window-seconds",
        type=positive_float,
        default=MOTION_WINDOW_SECONDS,
    )
    parser.add_argument(
        "--motion-threshold-pixels",
        type=positive_float,
        default=MIN_MOTION_PIXELS,
        help="Absolute floor for required robust displacement",
    )
    parser.add_argument(
        "--motion-threshold-ratio",
        type=probability,
        default=MIN_MOTION_FRAME_DIAGONAL_RATIO,
        help="Required displacement as a fraction of the frame diagonal",
    )
    parser.add_argument(
        "--visual-motion-interval-seconds",
        type=positive_float,
        default=VISUAL_MOTION_INTERVAL_SECONDS,
    )
    parser.add_argument(
        "--visual-difference-threshold",
        type=byte_value,
        default=VISUAL_DIFFERENCE_THRESHOLD,
    )
    parser.add_argument(
        "--visual-motion-ratio",
        type=probability,
        default=MIN_VISUAL_MOTION_RATIO,
        help="Median changed-pixel fraction required for motion confirmation",
    )
    parser.add_argument(
        "--static-hits",
        type=positive_int,
        default=STATIC_CONFIRMATION_HITS,
    )
    parser.add_argument(
        "--static-confidence",
        type=probability,
        default=STATIC_CONFIDENCE_THRESHOLD,
    )
    parser.add_argument(
        "--smoothing",
        type=probability,
        default=BOX_SMOOTHING_ALPHA,
        help="EMA weight for the current box; 1 disables smoothing",
    )
    parser.add_argument(
        "--max-box-area-ratio",
        type=probability,
        default=MAX_BOX_AREA_RATIO,
        help="Reject implausible boxes occupying more than this frame fraction",
    )
    parser.add_argument(
        "--top-corner-area-ratio",
        type=probability,
        default=TOP_CORNER_REJECTION_AREA_RATIO,
        help="Reject top-corner boxes at or above this backdrop-sized area",
    )
    parser.add_argument(
        "--allow-top-corner-boxes",
        action="store_true",
        help="Disable the large top-corner wall/door false-positive filter",
    )
    parser.add_argument(
        "--dedup-iou",
        type=probability,
        default=TRACK_DEDUPLICATION_IOU,
        help="Suppress overlapping tracked boxes at or above this IoU",
    )
    parser.add_argument(
        "--dedup-ios",
        type=probability,
        default=TRACK_DEDUPLICATION_IOS,
        help="Suppress a tracked box mostly contained inside another box",
    )
    parser.add_argument(
        "--roi",
        type=Path,
        default=None,
        help="Optional JSON polygon or object containing tracking_roi",
    )
    parser.add_argument("--show-candidates", action="store_true")
    parser.add_argument("--no-trajectories", action="store_true")
    parser.add_argument("--fp32", action="store_true", help="Disable FP16 on CUDA")
    parser.add_argument(
        "--max-frames",
        type=positive_int,
        default=None,
        help="Process only the first N frames (useful for smoke tests)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Validate inputs, load the fine-tuned model, and track the video."""
    args = build_parser().parse_args(argv)
    tracker_settings = TrackerSettings(
        detector_confidence=args.conf,
        track_low_threshold=args.track_low,
        track_high_threshold=args.track_high,
        new_track_threshold=args.new_track,
        buffer_seconds=args.track_buffer_seconds,
        match_threshold=args.match_threshold,
        fuse_score=args.fuse_score,
    )
    if args.tracker is None:
        tracker_settings.validate()
    if args.smoothing <= 0:
        raise ValueError("Smoothing must be in (0, 1].")
    if args.max_box_area_ratio <= 0:
        raise ValueError("Maximum box area ratio must be in (0, 1].")
    if args.top_corner_area_ratio <= 0:
        raise ValueError("Top-corner area ratio must be in (0, 1].")
    if args.dedup_iou <= 0 or args.dedup_ios <= 0:
        raise ValueError("Deduplication thresholds must be in (0, 1].")
    if not args.model.is_file():
        raise FileNotFoundError(f"Fine-tuned model not found: {args.model}")
    if not args.video.is_file():
        raise FileNotFoundError(f"Input video not found: {args.video}")

    device = resolve_device(args.device)
    use_fp16 = isinstance(device, int) and not args.fp32
    if isinstance(device, int):
        torch.backends.cudnn.benchmark = True
    print_environment(device, use_fp16)
    print(f"Model path: {args.model.resolve()}")

    model = YOLO(str(args.model))
    if model.task != "detect":
        raise RuntimeError(
            f"Expected a detection checkpoint, but model task is {model.task!r}."
        )
    class_names = normalized_class_names(model.names)
    print(f"Model classes: {class_names}")
    forklift_class_id = get_forklift_class_id(class_names)
    print(f"Forklift class ID: {forklift_class_id}")
    native_end2end = bool(getattr(model.model, "end2end", False))
    if native_end2end and not args.native_end2end:
        model.model.end2end = False
        print(
            "Model head: one-to-many + NMS for tracking "
            f"(iou={args.iou:.2f})"
        )
    elif native_end2end:
        print(
            "Model head: native end-to-end/NMS-free "
            "(--iou is ignored in this mode)"
        )
    else:
        print(f"Model head: standard NMS (iou={args.iou:.2f})")

    roi = load_tracking_roi(args.roi)
    if args.roi is not None:
        print(f"Tracking ROI: {args.roi.resolve()}")

    process_video(
        model,
        forklift_class_id,
        video_path=args.video,
        output_path=args.output,
        device=device,
        use_fp16=use_fp16,
        tracker_settings=tracker_settings,
        tracker_override=args.tracker,
        image_size=args.imgsz,
        iou_threshold=args.iou,
        confirmation_mode=args.confirmation_mode,
        min_confirmation_hits=args.min_hits,
        confirmation_evidence_seconds=args.confirmation_seconds,
        motion_window_seconds=args.motion_window_seconds,
        min_motion_pixel_floor=args.motion_threshold_pixels,
        motion_threshold_ratio=args.motion_threshold_ratio,
        visual_motion_interval_seconds=args.visual_motion_interval_seconds,
        visual_difference_threshold=args.visual_difference_threshold,
        min_visual_motion_ratio=args.visual_motion_ratio,
        static_confirmation_hits=args.static_hits,
        static_confidence=args.static_confidence,
        smoothing_alpha=args.smoothing,
        max_box_area_ratio=args.max_box_area_ratio,
        reject_top_corner_boxes=not args.allow_top_corner_boxes,
        top_corner_area_ratio=args.top_corner_area_ratio,
        deduplication_iou=args.dedup_iou,
        deduplication_ios=args.dedup_ios,
        roi=roi,
        show_candidates=args.show_candidates,
        draw_trajectories=not args.no_trajectories,
        max_frames=args.max_frames,
    )


if __name__ == "__main__":
    main()
