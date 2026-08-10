"""Run the same R6.2 pipeline on a live webcam/cockpit camera."""

import argparse
import time
from pathlib import Path

from forklift_phone_detection.config import add_common_inference_args, settings_from_args
from forklift_phone_detection.models.common import resolve_device
from forklift_phone_detection.pipeline import PhoneUsagePipeline
from forklift_phone_detection.utils.logger import configure_logging
from forklift_phone_detection.utils.video import resize_with_letterbox


def build_parser():
    parser = argparse.ArgumentParser(description="R6.2 live cockpit-camera inference")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--events", default="output/events.csv")
    parser.add_argument("--events-json", default="output/events.json")
    parser.add_argument(
        "--camera-mode",
        choices=("auto", "optimized", "recall"),
        default="auto",
        help="auto uses optimized settings on CPU and recall settings on CUDA",
    )
    parser.add_argument("--verbose", action="store_true")
    add_common_inference_args(parser)
    return parser


def apply_camera_mode(settings, requested_mode: str, args=None):
    mode = requested_mode
    if mode == "auto":
        mode = "optimized" if resolve_device(settings.device) == "cpu" else "recall"
    if mode == "optimized":
        optimized_defaults = {
            "phone_model": "yolo11n.pt",
            "phone_image_size": 640,
            "phone_crop_image_size": 320,
            "pose_image_size": 320,
            "use_pose_assisted_phone_search": True,
            "compact_phone_search_rois": True,
            "phone_crop_interval": 1,
            "display_mode": "phone_only",
        }
        explicit_arguments = {
            "phone_model": ("phone_model",),
            "phone_image_size": ("phone_image_size", "image_size"),
            "phone_crop_image_size": ("phone_crop_image_size", "image_size"),
            "pose_image_size": ("pose_image_size", "image_size"),
            "use_pose_assisted_phone_search": ("pose_phone_search",),
            "compact_phone_search_rois": ("compact_phone_search_rois",),
            "phone_crop_interval": ("phone_crop_interval",),
            "display_mode": ("display_mode",),
        }
        if args is not None:
            optimized_defaults = {
                field: value
                for field, value in optimized_defaults.items()
                if not any(
                    getattr(args, argument, None) is not None
                    for argument in explicit_arguments[field]
                )
            }
        settings = settings.with_overrides(**optimized_defaults)
    return settings, mode


def run(args) -> int:
    import cv2

    log = configure_logging(args.verbose)
    settings = settings_from_args(args)
    settings, camera_mode = apply_camera_mode(settings, args.camera_mode, args)
    capture = cv2.VideoCapture(args.camera)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open camera index {args.camera}")
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    fps = fps if fps > 1 else 30.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    pipeline = PhoneUsagePipeline(settings, fps)
    log.info("R6.2 Forklift Phone Detection")
    log.info("Camera mode: %s", camera_mode.upper())
    log.info(
        "Device: %s | detector: %s | pose: %s",
        resolve_device(settings.device).upper(),
        settings.phone_model,
        settings.pose_model,
    )
    log.info(
        "Phone imgsz: %d | crop imgsz: %d every %d frames | pose imgsz: %d",
        settings.phone_image_size,
        settings.phone_crop_image_size,
        settings.phone_crop_interval,
        settings.pose_image_size,
    )
    log.info("Camera: %d | resolution: %dx%d | assumed FPS: %.2f", args.camera, width, height, fps)
    log.info("Display: %dx%d", settings.display_width, settings.display_height)
    log.info("Temporal filter: timestamp-based %.1f s window", settings.temporal_window_seconds)
    log.info("Press q to exit")
    started = time.perf_counter()
    window_name = "R6.2 Forklift Phone Detection"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, settings.display_width, settings.display_height)
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                log.warning("Camera frame read failed; stopping")
                break
            timestamp = time.perf_counter() - started
            annotated, _ = pipeline.process_frame(frame, timestamp)
            display_frame = resize_with_letterbox(
                annotated, settings.display_width, settings.display_height
            )
            cv2.imshow(window_name, display_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        pipeline.finalize(time.perf_counter() - started)
        capture.release()
        cv2.destroyAllWindows()
    pipeline.events.write(Path(args.events), Path(args.events_json) if args.events_json else None)
    log.info(
        "Processed frames: %d | events: %d",
        pipeline.processed_frames,
        len(pipeline.events.events),
    )
    log.info("Events file: %s", args.events)
    return 0


def main():
    try:
        return run(build_parser().parse_args())
    except (RuntimeError, ValueError) as exc:
        configure_logging().error("%s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
