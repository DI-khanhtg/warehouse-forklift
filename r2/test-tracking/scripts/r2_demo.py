"""R2 forklift slowdown PoC for test1.mp4.

Pipeline:
  best_fresh.pt -> Ultralytics YOLO + default ByteTrack -> ROI -> trajectory
  -> relative speed -> Approach Zone -> Intersection Zone -> R2 decision

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
VIDEO_PATH = PROJECT_ROOT / "video" / "test1.mp4"
CONFIG_DIR = PROJECT_ROOT / "configs"
ZONE_CONFIG_PATH = CONFIG_DIR / "test1_r2_zones.json"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_VIDEO_PATH = OUTPUT_DIR / "r2_test1_demo.mp4"
OUTPUT_CSV_PATH = OUTPUT_DIR / "r2_test1_events.csv"

# =============================================================================
# DETECTOR / TRACKER BASELINE -- LOCKED TO BEST EXTERNAL TEST CONFIG
# =============================================================================
CONF_THRESHOLD = 0.15
IOU_THRESHOLD = 0.70
IMAGE_SIZE = 1280
TRACKER = "bytetrack.yaml"
DEVICE = 0
FP16_PRECISION = 16
PROGRESS_INTERVAL = 100

# =============================================================================
# ZONES
# =============================================================================
FORCE_REDRAW_ZONES = False
NUM_APPROACH_ZONES = 2
USE_TRACKING_ROI = True

# =============================================================================
# TRAJECTORY / SPEED
# =============================================================================
TRAJECTORY_LENGTH = 90
SPEED_WINDOW_FRAMES = 5
SPEED_EMA_ALPHA = 0.30
MIN_VALID_SPEED_PX_S = 5.0

# Static false positives may be tracked. They are not allowed to trigger R2
# until they have demonstrated meaningful motion. Once confirmed, a track stays
# confirmed even if it later slows/stops.
ENABLE_MOTION_CONFIRMATION = True
MOTION_HISTORY_FRAMES = 20
MIN_MOTION_OBSERVATIONS = 6
MIN_NET_MOTION_PX = 18.0

# =============================================================================
# R2 RULE
# =============================================================================
MIN_APPROACH_SPEED_SAMPLES = 5
MIN_INTERSECTION_SPEED_SAMPLES = 5
APPROACH_BASELINE_WINDOW = 15
MIN_SLOWDOWN_RATIO = 0.20
EVENT_BANNER_SECONDS = 3.0

# =============================================================================
# DISPLAY
# =============================================================================
DRAW_ZONES = True
DRAW_TRAJECTORY = True
DRAW_SPEED = True
DRAW_PANEL = True
BOX_THICKNESS = 3
FONT_SCALE = 0.58
FONT_THICKNESS = 2
PANEL_FONT_SCALE = 0.42
PANEL_FONT_THICKNESS = 1
PANEL_LINE_HEIGHT = 22
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
        default_factory=lambda: deque(maxlen=TRAJECTORY_LENGTH)
    )
    speed_history: Deque[float] = field(
        default_factory=lambda: deque(maxlen=TRAJECTORY_LENGTH)
    )
    approach_speeds: Deque[float] = field(
        default_factory=lambda: deque(maxlen=100)
    )
    intersection_speeds: Deque[float] = field(
        default_factory=lambda: deque(maxlen=100)
    )
    smoothed_speed_px_s: Optional[float] = None
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
            if len(points) >= 3:
                done = True
            else:
                status = "Need at least 3 points."

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
                2,
                cv2.LINE_AA,
            )

        overlay = canvas.copy()
        cv2.rectangle(overlay, (0, 0), (min(1100, canvas.shape[1]), 115), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.68, canvas, 0.32, 0, canvas)
        cv2.putText(canvas, instruction, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.60,
                    (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(canvas,
                    "Left:add | Right/ENTER/SPACE:finish | U:undo | R:reset | Q/ESC:cancel",
                    (15, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.49, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(canvas, status or f"Points: {len(points)}", (15, 92),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.49,
                    (80, 80, 255) if status else (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imshow(title, canvas)
        key = cv2.waitKey(20) & 0xFF

        if key in (10, 13, 32):
            if len(points) >= 3:
                done = True
            else:
                status = "Need at least 3 points."
        elif key in (8, 127, ord('u'), ord('U')):
            if points:
                points.pop()
            status = ""
        elif key in (ord('r'), ord('R')):
            points.clear()
            status = "Polygon reset."
        elif key in (ord('q'), ord('Q'), 27):
            cv2.destroyAllWindows()
            raise ZoneSelectionAborted(title)

    cv2.destroyWindow(title)
    return points


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
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    ZONE_CONFIG_PATH.write_text(json.dumps(zones, indent=2), encoding="utf-8")
    print(f"Saved zones: {ZONE_CONFIG_PATH}")
    return zones


def load_or_create_zones(first_frame: np.ndarray) -> Dict:
    if ZONE_CONFIG_PATH.exists() and not FORCE_REDRAW_ZONES:
        raw = json.loads(ZONE_CONFIG_PATH.read_text(encoding="utf-8"))
        zones = {
            "tracking_roi": [tuple(p) for p in raw["tracking_roi"]],
            "approach_zones": [[tuple(p) for p in poly] for poly in raw["approach_zones"]],
            "intersection_zone": [tuple(p) for p in raw["intersection_zone"]],
        }
        print(f"Loaded zones: {ZONE_CONFIG_PATH}")
        return zones
    return create_zones(first_frame)


def get_forklift_class_id(model: YOLO) -> int:
    names = model.names
    class_map = (
        {int(k): str(v) for k, v in names.items()}
        if isinstance(names, dict)
        else {i: str(v) for i, v in enumerate(names)}
    )
    matches = [i for i, name in class_map.items() if name.strip().casefold() == "forklift"]
    if len(matches) != 1:
        raise RuntimeError(f"Expected exactly one 'forklift' class. Classes: {class_map}")
    return matches[0]


def robust_net_motion(positions: Deque[Tuple[int, float, float]]) -> float:
    recent = list(positions)[-MOTION_HISTORY_FRAMES:]
    if len(recent) < 2:
        return 0.0
    edge = min(3, max(1, len(recent) // 3))
    sx = float(np.median([p[1] for p in recent[:edge]]))
    sy = float(np.median([p[2] for p in recent[:edge]]))
    ex = float(np.median([p[1] for p in recent[-edge:]]))
    ey = float(np.median([p[2] for p in recent[-edge:]]))
    return math.hypot(ex - sx, ey - sy)


def update_motion_confirmation(state: TrackState) -> None:
    if not ENABLE_MOTION_CONFIRMATION:
        state.motion_confirmed = True
        return
    if state.motion_confirmed or len(state.positions) < MIN_MOTION_OBSERVATIONS:
        return
    if robust_net_motion(state.positions) >= MIN_NET_MOTION_PX:
        state.motion_confirmed = True


def update_speed(state: TrackState, fps: float) -> Optional[float]:
    if len(state.positions) <= SPEED_WINDOW_FRAMES:
        return None
    current = state.positions[-1]
    previous = list(state.positions)[-1 - SPEED_WINDOW_FRAMES]
    frame_delta = current[0] - previous[0]
    if frame_delta <= 0:
        return None
    dt = frame_delta / fps
    distance = math.hypot(current[1] - previous[1], current[2] - previous[2])
    raw = distance / dt
    if raw < MIN_VALID_SPEED_PX_S:
        raw = 0.0
    if state.smoothed_speed_px_s is None:
        smoothed = raw
    else:
        smoothed = SPEED_EMA_ALPHA * raw + (1.0 - SPEED_EMA_ALPHA) * state.smoothed_speed_px_s
    state.smoothed_speed_px_s = smoothed
    state.speed_history.append(smoothed)
    return smoothed


def find_approach_zone(point: Point, zones: Dict) -> Optional[int]:
    for idx, poly in enumerate(zones["approach_zones"]):
        if point_in_polygon(point, poly):
            return idx
    return None


def update_r2(state: TrackState, point: Point, speed: Optional[float], frame_idx: int, zones: Dict):
    if not state.motion_confirmed:
        return None

    approach_idx = find_approach_zone(point, zones)
    in_intersection = point_in_polygon(point, zones["intersection_zone"])

    if approach_idx is not None and not state.entered_intersection:
        state.approach_zone = approach_idx
        state.has_visited_approach = True
        if speed is not None:
            state.approach_speeds.append(speed)

    # R2 is valid only for the intended flow: Approach -> Intersection.
    if in_intersection and not state.entered_intersection and state.has_visited_approach:
        state.entered_intersection = True
        state.intersection_entry_frame = frame_idx
        if len(state.approach_speeds) >= MIN_APPROACH_SPEED_SAMPLES:
            recent = list(state.approach_speeds)[-APPROACH_BASELINE_WINDOW:]
            state.baseline_speed_px_s = float(np.median(recent))

    if state.entered_intersection and in_intersection and speed is not None:
        state.intersection_speeds.append(speed)

    if state.decision is not None:
        return None
    if not state.entered_intersection or state.baseline_speed_px_s is None:
        return None
    if len(state.intersection_speeds) < MIN_INTERSECTION_SPEED_SAMPLES:
        return None

    early = list(state.intersection_speeds)[:MIN_INTERSECTION_SPEED_SAMPLES]
    intersection_speed = float(np.median(early))
    state.intersection_speed_px_s = intersection_speed
    baseline = max(state.baseline_speed_px_s, 1e-6)
    ratio = (baseline - intersection_speed) / baseline
    state.slowdown_ratio = ratio
    state.decision = "SLOWED" if ratio >= MIN_SLOWDOWN_RATIO else "NO_SLOWDOWN"

    return {
        "track_id": state.track_id,
        "approach_zone": state.approach_zone + 1 if state.approach_zone is not None else "",
        "intersection_entry_frame": state.intersection_entry_frame,
        "approach_speed_px_s": round(state.baseline_speed_px_s, 3),
        "intersection_speed_px_s": round(state.intersection_speed_px_s, 3),
        "slowdown_percent": round(ratio * 100.0, 2),
        "result": state.decision,
    }


def draw_polygon(frame, polygon: Polygon, color, label: str, thickness: int = 2):
    if len(polygon) < 3:
        return
    pts = np.asarray(polygon, dtype=np.int32).reshape((-1, 1, 2))
    cv2.polylines(frame, [pts], True, color, thickness, cv2.LINE_AA)
    x, y = polygon[0]
    cv2.putText(frame, label, (x + 4, max(20, y - 7)), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, color, 2, cv2.LINE_AA)


def draw_zones(frame, zones: Dict):
    if not DRAW_ZONES:
        return
    if USE_TRACKING_ROI:
        draw_polygon(frame, zones["tracking_roi"], COLOR_ROI, "TRACKING ROI")
    for idx, poly in enumerate(zones["approach_zones"]):
        draw_polygon(frame, poly, COLOR_APPROACH, f"APPROACH {idx + 1}")
    draw_polygon(frame, zones["intersection_zone"], COLOR_INTERSECTION, "INTERSECTION", 3)


def state_color(state: TrackState):
    if not state.motion_confirmed:
        return COLOR_CANDIDATE
    if state.decision == "NO_SLOWDOWN":
        return COLOR_NO_SLOWDOWN
    if state.decision == "SLOWED":
        return COLOR_SLOWED
    if state.entered_intersection:
        return COLOR_EVALUATING
    return COLOR_NORMAL


def state_text(state: TrackState) -> str:
    if not state.motion_confirmed:
        return "CANDIDATE"
    if state.decision == "NO_SLOWDOWN":
        return "R2: NO SLOWDOWN"
    if state.decision == "SLOWED":
        return "R2: SLOWED"
    if state.entered_intersection:
        return "R2: EVALUATING"
    if state.has_visited_approach:
        return "APPROACH"
    return "TRACKED"


def draw_track(frame, box, confidence: float, state: TrackState):
    color = state_color(state)
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    x1, x2 = max(0, min(x1, w - 1)), max(0, min(x2, w - 1))
    y1, y2 = max(0, min(y1, h - 1)), max(0, min(y2, h - 1))
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, BOX_THICKNESS)

    gx, gy = bottom_center(box)
    cv2.circle(frame, (int(gx), int(gy)), 5, color, -1)

    if DRAW_TRAJECTORY and len(state.positions) >= 2:
        pts = np.asarray([(int(x), int(y)) for _, x, y in state.positions], dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(frame, [pts], False, color, 2, cv2.LINE_AA)

    speed = "-- px/s" if state.smoothed_speed_px_s is None else f"{state.smoothed_speed_px_s:.1f} px/s"
    parts = [f"Forklift #{state.track_id}", f"{confidence:.2f}"]
    if DRAW_SPEED:
        parts.append(speed)
    parts.append(state_text(state))
    label = " | ".join(parts)

    (tw, th), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, FONT_THICKNESS)
    tx = max(0, min(x1, w - tw - 10))
    ty = y1 - 8
    if ty - th - base < 0:
        ty = min(h - base - 2, y1 + th + base + 8)
    top = max(0, ty - th - base - 5)
    bottom = min(h - 1, ty + base + 4)
    cv2.rectangle(frame, (tx, top), (min(w - 1, tx + tw + 10), bottom), color, -1)
    cv2.putText(frame, label, (tx + 5, ty), cv2.FONT_HERSHEY_SIMPLEX,
                FONT_SCALE, (0, 0, 0), FONT_THICKNESS, cv2.LINE_AA)


def draw_panel(frame, frame_idx, total_frames, roi_tracks, confirmed, events, fps_proc):
    if not DRAW_PANEL:
        return
    lines = [
        "R2 Forklift Slowdown PoC",
        f"Frame: {frame_idx} / {total_frames}",
        f"Tracks in ROI: {roi_tracks}",
        f"Motion-confirmed: {confirmed}",
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
    text = (
        f"R2 RESULT | Forklift #{event['track_id']} | "
        f"{event['result'].replace('_', ' ')} | Slowdown: {event['slowdown_percent']:.1f}%"
    )
    (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.76, 2)
    x1, y2 = 20, frame.shape[0] - 20
    y1 = max(0, y2 - th - base - 24)
    x2 = min(frame.shape[1] - 20, x1 + tw + 30)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, -1)
    cv2.putText(frame, text, (x1 + 15, y2 - base - 10), cv2.FONT_HERSHEY_SIMPLEX,
                0.76, (255, 255, 255), 2, cv2.LINE_AA)


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
        print(f"GPU : {torch.cuda.get_device_name(DEVICE)}")
    print(f"Config: conf={CONF_THRESHOLD}, iou={IOU_THRESHOLD}, imgsz={IMAGE_SIZE}, tracker={TRACKER}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(VIDEO_PATH))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open: {VIDEO_PATH}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
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
    forklift_class_id = get_forklift_class_id(model)

    writer = cv2.VideoWriter(
        str(OUTPUT_VIDEO_PATH),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Cannot create: {OUTPUT_VIDEO_PATH}")

    states: Dict[int, TrackState] = {}
    events: List[Dict] = []
    recent_event = None
    recent_event_until = 0
    frame_idx = 0
    written = 0
    start = time.perf_counter()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_idx += 1
            t0 = time.perf_counter()

            # Keep exact detector/tracker baseline used in external testing.
            results = model.track(
                frame,
                persist=True,
                tracker=TRACKER,
                classes=[forklift_class_id],
                conf=CONF_THRESHOLD,
                iou=IOU_THRESHOLD,
                imgsz=IMAGE_SIZE,
                device=device,
                quantize=FP16_PRECISION if use_cuda else None,
                verbose=False,
            )

            roi_tracks = 0
            confirmed_visible = 0

            if results:
                b = results[0].boxes
                if b is not None and len(b) > 0 and b.id is not None:
                    data = torch.cat(
                        (b.xyxy, b.conf.unsqueeze(1), b.cls.unsqueeze(1), b.id.unsqueeze(1)),
                        dim=1,
                    ).detach().cpu().numpy()

                    for det in data:
                        box = det[:4]
                        confidence = float(det[4])
                        class_id = int(det[5])
                        track_id = int(det[6])
                        if class_id != forklift_class_id:
                            continue

                        point = bottom_center(box)
                        if USE_TRACKING_ROI and not point_in_polygon(point, zones["tracking_roi"]):
                            continue

                        roi_tracks += 1
                        state = states.setdefault(track_id, TrackState(track_id=track_id))
                        state.positions.append((frame_idx, point[0], point[1]))
                        update_motion_confirmation(state)
                        speed = update_speed(state, fps)
                        if state.motion_confirmed:
                            confirmed_visible += 1

                        event = update_r2(state, point, speed, frame_idx, zones)
                        if event is not None and not state.event_logged:
                            state.event_logged = True
                            events.append(event)
                            recent_event = event
                            recent_event_until = frame_idx + int(EVENT_BANNER_SECONDS * fps)
                            print("R2 EVENT:", event)

                        draw_track(frame, box, confidence, state)

            draw_zones(frame, zones)
            elapsed_frame = time.perf_counter() - t0
            fps_proc = 1.0 / elapsed_frame if elapsed_frame > 0 else 0.0
            draw_panel(frame, frame_idx, total_frames, roi_tracks, confirmed_visible, len(events), fps_proc)

            if recent_event is not None and frame_idx <= recent_event_until:
                draw_event_banner(frame, recent_event)

            writer.write(frame)
            written += 1

            if frame_idx == 1 or frame_idx % PROGRESS_INTERVAL == 0:
                elapsed = time.perf_counter() - start
                avg = frame_idx / elapsed if elapsed > 0 else 0.0
                print(
                    f"[{frame_idx}/{total_frames}] ROI={roi_tracks} | "
                    f"confirmed={confirmed_visible} | events={len(events)} | avg_fps={avg:.2f}"
                )
    finally:
        cap.release()
        writer.release()
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
    print(f"Video output   : {OUTPUT_VIDEO_PATH}")
    print(f"CSV output     : {OUTPUT_CSV_PATH}")
    print("NOTE: speed is px/s, not calibrated km/h.")


if __name__ == "__main__":
    try:
        main()
    except ZoneSelectionAborted as exc:
        cv2.destroyAllWindows()
        print(f"Zone setup cancelled: {exc}")
