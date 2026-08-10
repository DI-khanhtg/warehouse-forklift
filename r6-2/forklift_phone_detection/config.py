"""Central configuration for the R6.2 phone-use detection PoC."""

import argparse
from dataclasses import dataclass, replace
from typing import Optional, Tuple


DEVICE = "auto"
PHONE_MODEL = "yolo11s.pt"
POSE_MODEL = "yolo11n-pose.pt"

PHONE_CONF_THRESHOLD = 0.10
POSE_CONF_THRESHOLD = 0.25
KEYPOINT_CONF_THRESHOLD = 0.30

PHONE_IMAGE_SIZE = 960
PHONE_CROP_IMAGE_SIZE = 640
POSE_IMAGE_SIZE = 640
ENABLE_DRIVER_TRACKING = True
POSE_SMOOTHING_ALPHA = 0.55
DRIVER_TRACK_MAX_CENTER_JUMP = 0.35
DRIVER_TRACK_MIN_IOU = 0.10
DRIVER_TRACK_MAX_MISSED_FRAMES = 3
POSE_KEYPOINT_MAX_JUMP = 0.85

DEBUG_PHONE_DETECTION = True
PHONE_DEBUG_INTERNAL_CONF_THRESHOLD = 0.01
PHONE_DEBUG_LOG_EVERY = 0
USE_POSE_ASSISTED_PHONE_SEARCH = True
PHONE_ROI_SCALE = 0.75
PHONE_NMS_IOU_THRESHOLD = 0.50
PHONE_CROP_INTERVAL = 1
COMPACT_PHONE_SEARCH_ROIS = False
SHOW_PHONE_SEARCH_ROIS = True
DISPLAY_MODE = "phone_only"
DISPLAY_WIDTH = 1280
DISPLAY_HEIGHT = 720

HAND_PHONE_DISTANCE_THRESHOLD = 0.60
HEAD_PHONE_DISTANCE_THRESHOLD = 0.70

TEMPORAL_WINDOW_SECONDS = 1.5
ALERT_ON_RATIO = 0.60
ALERT_OFF_RATIO = 0.30
MIN_WINDOW_FILL_RATIO = 0.50

USE_DRIVER_ROI = False
DRIVER_ROI = {"x1": 0.0, "y1": 0.0, "x2": 1.0, "y2": 1.0}

IMAGE_SIZE = 640
SHOW_POSE = True
SHOW_DEBUG_LINES = True
REQUIRE_PHONE_IN_BODY_REGION = True
REQUIRE_PHONE_BELOW_SHOULDERS = False


@dataclass(frozen=True)
class Settings:
    device: str = DEVICE
    phone_model: str = PHONE_MODEL
    pose_model: str = POSE_MODEL
    phone_conf_threshold: float = PHONE_CONF_THRESHOLD
    pose_conf_threshold: float = POSE_CONF_THRESHOLD
    keypoint_conf_threshold: float = KEYPOINT_CONF_THRESHOLD
    phone_image_size: int = PHONE_IMAGE_SIZE
    phone_crop_image_size: int = PHONE_CROP_IMAGE_SIZE
    pose_image_size: int = POSE_IMAGE_SIZE
    enable_driver_tracking: bool = ENABLE_DRIVER_TRACKING
    pose_smoothing_alpha: float = POSE_SMOOTHING_ALPHA
    driver_track_max_center_jump: float = DRIVER_TRACK_MAX_CENTER_JUMP
    driver_track_min_iou: float = DRIVER_TRACK_MIN_IOU
    driver_track_max_missed_frames: int = DRIVER_TRACK_MAX_MISSED_FRAMES
    pose_keypoint_max_jump: float = POSE_KEYPOINT_MAX_JUMP
    debug_phone_detection: bool = DEBUG_PHONE_DETECTION
    phone_debug_internal_conf_threshold: float = PHONE_DEBUG_INTERNAL_CONF_THRESHOLD
    phone_debug_log_every: int = PHONE_DEBUG_LOG_EVERY
    use_pose_assisted_phone_search: bool = USE_POSE_ASSISTED_PHONE_SEARCH
    phone_roi_scale: float = PHONE_ROI_SCALE
    phone_nms_iou_threshold: float = PHONE_NMS_IOU_THRESHOLD
    phone_crop_interval: int = PHONE_CROP_INTERVAL
    compact_phone_search_rois: bool = COMPACT_PHONE_SEARCH_ROIS
    show_phone_search_rois: bool = SHOW_PHONE_SEARCH_ROIS
    display_mode: str = DISPLAY_MODE
    display_width: int = DISPLAY_WIDTH
    display_height: int = DISPLAY_HEIGHT
    hand_phone_distance_threshold: float = HAND_PHONE_DISTANCE_THRESHOLD
    head_phone_distance_threshold: float = HEAD_PHONE_DISTANCE_THRESHOLD
    temporal_window_seconds: float = TEMPORAL_WINDOW_SECONDS
    alert_on_ratio: float = ALERT_ON_RATIO
    alert_off_ratio: float = ALERT_OFF_RATIO
    min_window_fill_ratio: float = MIN_WINDOW_FILL_RATIO
    use_driver_roi: bool = USE_DRIVER_ROI
    driver_roi: Tuple[float, float, float, float] = (
        DRIVER_ROI["x1"], DRIVER_ROI["y1"], DRIVER_ROI["x2"], DRIVER_ROI["y2"]
    )
    image_size: int = IMAGE_SIZE
    show_pose: bool = SHOW_POSE
    show_debug_lines: bool = SHOW_DEBUG_LINES
    require_phone_in_body_region: bool = REQUIRE_PHONE_IN_BODY_REGION
    require_phone_below_shoulders: bool = REQUIRE_PHONE_BELOW_SHOULDERS

    def with_overrides(self, **kwargs) -> "Settings":
        return replace(self, **{k: v for k, v in kwargs.items() if v is not None})


def add_common_inference_args(parser) -> None:
    parser.add_argument("--device", default=None, help="auto, cpu, 0, cuda:0, ...")
    parser.add_argument("--phone-model", default=None, help="YOLO COCO detector weights")
    parser.add_argument("--pose-model", default=None, help="YOLO pose weights")
    parser.add_argument("--phone-conf", type=float, default=None)
    parser.add_argument("--pose-conf", type=float, default=None)
    parser.add_argument("--keypoint-conf", type=float, default=None)
    parser.add_argument("--hand-threshold", type=float, default=None)
    parser.add_argument("--head-threshold", type=float, default=None)
    parser.add_argument("--window-seconds", type=float, default=None)
    parser.add_argument("--alert-on-ratio", type=float, default=None)
    parser.add_argument("--alert-off-ratio", type=float, default=None)
    parser.add_argument(
        "--phone-image-size", type=int, choices=(640, 960, 1280), default=None,
        help="Full-frame phone detector size",
    )
    parser.add_argument(
        "--phone-crop-image-size", type=int, choices=(640, 960, 1280), default=None,
        help="Pose-assisted crop detector size",
    )
    parser.add_argument("--pose-image-size", type=int, default=None)
    parser.add_argument(
        "--driver-tracking", action=argparse.BooleanOptionalAction, default=None,
        help="Associate and smooth the same driver across frames",
    )
    parser.add_argument("--pose-smoothing-alpha", type=float, default=None)
    parser.add_argument("--driver-max-center-jump", type=float, default=None)
    parser.add_argument("--driver-min-iou", type=float, default=None)
    parser.add_argument("--driver-max-missed-frames", type=int, default=None)
    parser.add_argument("--pose-keypoint-max-jump", type=float, default=None)
    parser.add_argument(
        "--image-size", type=int, default=None,
        help="Legacy shortcut: set phone, crop, and pose image sizes together",
    )
    parser.add_argument("--phone-roi-scale", type=float, default=None)
    parser.add_argument("--phone-nms-iou", type=float, default=None)
    parser.add_argument("--phone-debug-internal-conf", type=float, default=None)
    parser.add_argument("--phone-debug-log-every", type=int, default=None)
    parser.add_argument("--phone-crop-interval", type=int, default=None)
    parser.add_argument(
        "--compact-phone-search-rois", action=argparse.BooleanOptionalAction, default=None,
        help="When both wrists exist, search one both-hands crop instead of two wrist crops",
    )
    parser.add_argument(
        "--debug-phone-detection", action=argparse.BooleanOptionalAction, default=None,
        help="Retain and expose cell-phone candidates below --phone-conf",
    )
    parser.add_argument(
        "--pose-phone-search", action=argparse.BooleanOptionalAction, default=None,
        help="Run secondary phone detection on pose-generated local crops",
    )
    parser.add_argument(
        "--show-phone-search-rois", action=argparse.BooleanOptionalAction, default=None,
    )
    parser.add_argument(
        "--display-size", default=None, metavar="WIDTHxHEIGHT",
        help="Preview/output resolution; default 1280x720",
    )
    parser.add_argument(
        "--display-mode", choices=("phone_only", "debug"), default=None,
        help="phone_only shows only accepted phone boxes, state, and FPS",
    )
    parser.add_argument(
        "--roi",
        type=str,
        default=None,
        metavar="X1,Y1,X2,Y2",
        help="Normalized driver ROI; enables ROI processing",
    )
    parser.add_argument(
        "--allow-phone-outside-body",
        action="store_true",
        help="Do not require a hand-held phone center to be inside the person box",
    )
    parser.add_argument(
        "--require-below-shoulders",
        action="store_true",
        help="Require the texting/holding phone center to be below the shoulder line",
    )
    parser.add_argument("--hide-pose", action="store_true")
    parser.add_argument("--hide-debug-lines", action="store_true")


def settings_from_args(args) -> Settings:
    overrides = {
        "device": getattr(args, "device", None),
        "phone_model": getattr(args, "phone_model", None),
        "pose_model": getattr(args, "pose_model", None),
        "phone_conf_threshold": getattr(args, "phone_conf", None),
        "pose_conf_threshold": getattr(args, "pose_conf", None),
        "keypoint_conf_threshold": getattr(args, "keypoint_conf", None),
        "phone_image_size": getattr(args, "phone_image_size", None),
        "phone_crop_image_size": getattr(args, "phone_crop_image_size", None),
        "pose_image_size": getattr(args, "pose_image_size", None),
        "enable_driver_tracking": getattr(args, "driver_tracking", None),
        "pose_smoothing_alpha": getattr(args, "pose_smoothing_alpha", None),
        "driver_track_max_center_jump": getattr(args, "driver_max_center_jump", None),
        "driver_track_min_iou": getattr(args, "driver_min_iou", None),
        "driver_track_max_missed_frames": getattr(args, "driver_max_missed_frames", None),
        "pose_keypoint_max_jump": getattr(args, "pose_keypoint_max_jump", None),
        "debug_phone_detection": getattr(args, "debug_phone_detection", None),
        "phone_debug_internal_conf_threshold": getattr(args, "phone_debug_internal_conf", None),
        "phone_debug_log_every": getattr(args, "phone_debug_log_every", None),
        "phone_crop_interval": getattr(args, "phone_crop_interval", None),
        "compact_phone_search_rois": getattr(args, "compact_phone_search_rois", None),
        "use_pose_assisted_phone_search": getattr(args, "pose_phone_search", None),
        "phone_roi_scale": getattr(args, "phone_roi_scale", None),
        "phone_nms_iou_threshold": getattr(args, "phone_nms_iou", None),
        "show_phone_search_rois": getattr(args, "show_phone_search_rois", None),
        "display_mode": getattr(args, "display_mode", None),
        "hand_phone_distance_threshold": getattr(args, "hand_threshold", None),
        "head_phone_distance_threshold": getattr(args, "head_threshold", None),
        "temporal_window_seconds": getattr(args, "window_seconds", None),
        "alert_on_ratio": getattr(args, "alert_on_ratio", None),
        "alert_off_ratio": getattr(args, "alert_off_ratio", None),
    }
    settings = Settings().with_overrides(**overrides)
    legacy_image_size = getattr(args, "image_size", None)
    if legacy_image_size is not None:
        settings = settings.with_overrides(
            image_size=legacy_image_size,
            phone_image_size=legacy_image_size,
            phone_crop_image_size=legacy_image_size,
            pose_image_size=legacy_image_size,
        )
    display_size = getattr(args, "display_size", None)
    if display_size:
        try:
            display_width, display_height = (
                int(value.strip()) for value in display_size.lower().split("x", maxsplit=1)
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("--display-size must use WIDTHxHEIGHT, for example 1280x720") from exc
        if display_width <= 0 or display_height <= 0:
            raise ValueError("--display-size dimensions must be positive")
        settings = settings.with_overrides(
            display_width=display_width, display_height=display_height
        )
    roi = getattr(args, "roi", None)
    if roi:
        values = tuple(float(value.strip()) for value in roi.split(","))
        if len(values) != 4 or not all(0.0 <= value <= 1.0 for value in values):
            raise ValueError("--roi must contain four normalized values between 0 and 1")
        if values[0] >= values[2] or values[1] >= values[3]:
            raise ValueError("--roi must satisfy x1 < x2 and y1 < y2")
        settings = settings.with_overrides(use_driver_roi=True, driver_roi=values)
    if getattr(args, "allow_phone_outside_body", False):
        settings = settings.with_overrides(require_phone_in_body_region=False)
    if getattr(args, "require_below_shoulders", False):
        settings = settings.with_overrides(require_phone_below_shoulders=True)
    if getattr(args, "hide_pose", False):
        settings = settings.with_overrides(show_pose=False)
    if getattr(args, "hide_debug_lines", False):
        settings = settings.with_overrides(show_debug_lines=False)
    if settings.alert_off_ratio > settings.alert_on_ratio:
        raise ValueError("--alert-off-ratio cannot exceed --alert-on-ratio")
    if not 0.0 <= settings.phone_debug_internal_conf_threshold <= settings.phone_conf_threshold <= 1.0:
        raise ValueError("Phone confidences must satisfy 0 <= internal <= normal <= 1")
    if settings.phone_roi_scale <= 0:
        raise ValueError("--phone-roi-scale must be positive")
    if not 0.0 <= settings.phone_nms_iou_threshold <= 1.0:
        raise ValueError("--phone-nms-iou must be between 0 and 1")
    if settings.phone_debug_log_every < 0:
        raise ValueError("--phone-debug-log-every cannot be negative")
    if settings.phone_crop_interval < 1:
        raise ValueError("--phone-crop-interval must be at least 1")
    if not 0.0 < settings.pose_smoothing_alpha <= 1.0:
        raise ValueError("--pose-smoothing-alpha must be in (0, 1]")
    if settings.driver_track_max_center_jump <= 0:
        raise ValueError("--driver-max-center-jump must be positive")
    if not 0.0 <= settings.driver_track_min_iou <= 1.0:
        raise ValueError("--driver-min-iou must be between 0 and 1")
    if settings.driver_track_max_missed_frames < 0:
        raise ValueError("--driver-max-missed-frames cannot be negative")
    if settings.pose_keypoint_max_jump <= 0:
        raise ValueError("--pose-keypoint-max-jump must be positive")
    return settings
