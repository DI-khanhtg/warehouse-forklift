"""Run R6.2 inference on an MP4 or other OpenCV-readable video file."""

import argparse
import json
import time
from pathlib import Path

from forklift_phone_detection.config import add_common_inference_args, settings_from_args
from forklift_phone_detection.models.common import resolve_device
from forklift_phone_detection.pipeline import PhoneUsagePipeline
from forklift_phone_detection.utils.logger import configure_logging
from forklift_phone_detection.utils.video import (
    create_video_writer,
    default_video_output,
    resize_with_letterbox,
)


def build_parser():
    parser = argparse.ArgumentParser(description="R6.2 forklift cockpit phone-use video inference")
    parser.add_argument("--source", required=True, help="Input video path")
    parser.add_argument("--output", default=None, help="Annotated MP4 path")
    parser.add_argument("--events", default=None, help="Events CSV path")
    parser.add_argument("--events-json", default=None, help="Optional events JSON path")
    parser.add_argument("--report", default=None, help="Inference summary JSON path")
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--preview", action="store_true", help="Show a live preview; q exits")
    parser.add_argument(
        "--save-debug-frames",
        action="store_true",
        help="Save sampled frames where wrists are visible but no phone is detected near them",
    )
    parser.add_argument(
        "--debug-frame-interval", type=float, default=1.0,
        help="Minimum seconds between saved missed-phone frames",
    )
    parser.add_argument("--verbose", action="store_true")
    add_common_inference_args(parser)
    return parser


def run(args) -> int:
    import cv2

    log = configure_logging(args.verbose)
    source = Path(args.source)
    if not source.is_file():
        raise FileNotFoundError(f"Input video does not exist: {source}")
    settings = settings_from_args(args)
    output = Path(args.output) if args.output else default_video_output(source)
    events_path = Path(args.events) if args.events else output.parent / "events.csv"
    events_json = Path(args.events_json) if args.events_json else output.parent / "events.json"
    report_path = Path(args.report) if args.report else output.parent / f"{source.stem}_run_report.json"

    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise RuntimeError(f"OpenCV could not open input video: {source}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    fps = fps if fps > 0 else 25.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    log.info("R6.2 Forklift Phone Detection")
    log.info("Device: %s", resolve_device(settings.device).upper())
    log.info("Detector: %s", settings.phone_model)
    log.info(
        "Phone detector: full imgsz=%d | crop imgsz=%d | accept conf=%.3f | internal conf=%.3f",
        settings.phone_image_size,
        settings.phone_crop_image_size,
        settings.phone_conf_threshold,
        settings.phone_debug_internal_conf_threshold if settings.debug_phone_detection else settings.phone_conf_threshold,
    )
    log.info("Pose-assisted phone search: %s", "ON" if settings.use_pose_assisted_phone_search else "OFF")
    log.info("Pose model: %s", settings.pose_model)
    log.info("Phone class: cell phone (resolved from model.names)")
    log.info("Input: %s", source)
    log.info("Resolution: %dx%d | source FPS: %.2f", width, height, fps)
    log.info(
        "Display/output: %dx%d | mode: %s",
        settings.display_width,
        settings.display_height,
        settings.display_mode,
    )

    pipeline = PhoneUsagePipeline(settings, fps)
    writer = create_video_writer(
        output, fps, settings.display_width, settings.display_height
    )
    started = time.perf_counter()
    frame_index = 0
    saved_debug_frames = 0
    last_debug_save_time = float("-inf")
    debug_directory = output.parent / "debug" / "missed_phone"
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            timestamp = frame_index / fps
            original_frame = frame.copy() if args.save_debug_frames else None
            annotated, result = pipeline.process_frame(frame, timestamp)
            display_frame = resize_with_letterbox(
                annotated, settings.display_width, settings.display_height
            )
            writer.write(display_frame)
            should_save_debug = (
                args.save_debug_frames
                and result["visible_wrist_count"] > 0
                and not result["wrist_phone_detected"]
                and timestamp - last_debug_save_time >= max(0.0, args.debug_frame_interval)
            )
            if should_save_debug:
                debug_directory.mkdir(parents=True, exist_ok=True)
                stem = f"{source.stem}_frame_{frame_index:06d}_{int(timestamp * 1000):010d}ms"
                image_path = debug_directory / f"{stem}.jpg"
                metadata_path = debug_directory / f"{stem}.json"
                if not cv2.imwrite(str(image_path), original_frame):
                    log.warning("Could not save debug frame: %s", image_path)
                else:
                    metadata = {
                        "source_video": str(source),
                        "frame_index": frame_index,
                        "timestamp": round(timestamp, 3),
                        "candidate_status": result["phone_candidate_status"],
                        "visible_wrist_count": result["visible_wrist_count"],
                        "accepted_phones": result["phones"],
                        "raw_phone_candidates": result["raw_phone_candidates"],
                        "search_rois": result["phone_search_rois"],
                        "pose": result["pose"],
                    }
                    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
                    raw_summary = ", ".join(
                        f"{item['confidence']:.3f}@{item['source']}"
                        for item in result["raw_phone_candidates"]
                    ) or "none"
                    log.info(
                        "Saved missed-phone debug frame: %s | status=%s | raw=[%s]",
                        image_path,
                        result["phone_candidate_status"],
                        raw_summary,
                    )
                    saved_debug_frames += 1
                    last_debug_save_time = timestamp
            frame_index += 1
            if args.progress_every > 0 and (
                frame_index == 1 or frame_index % args.progress_every == 0
            ):
                total_text = str(total_frames) if total_frames > 0 else "?"
                log.info(
                    "Frame: %d/%s | processing FPS: %.1f | State: %s",
                    frame_index,
                    total_text,
                    result["processing_fps"],
                    result["temporal"].state,
                )
            if args.preview:
                cv2.imshow("R6.2 Forklift Phone Detection", display_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        final_time = frame_index / fps
        pipeline.finalize(final_time)
        capture.release()
        writer.release()
        if args.preview:
            cv2.destroyAllWindows()

    pipeline.events.write(events_path, events_json)
    elapsed = max(time.perf_counter() - started, 1e-9)
    total_violation = sum(event["duration"] for event in pipeline.events.events)
    report = {
        "video": str(source),
        "duration_seconds": round(frame_index / fps, 3),
        "frames": frame_index,
        "resolution": {"width": width, "height": height},
        "output_resolution": {
            "width": settings.display_width,
            "height": settings.display_height,
        },
        "display_mode": settings.display_mode,
        "source_fps": round(fps, 3),
        "processing_fps": round(frame_index / elapsed, 3),
        "device": resolve_device(settings.device),
        "phone_model": settings.phone_model,
        "pose_model": settings.pose_model,
        "phone_detections": pipeline.phone_detections_count,
        "raw_phone_candidates": pipeline.raw_phone_candidates_count,
        "low_confidence_phone_candidates": pipeline.low_confidence_phone_candidates_count,
        "saved_missed_phone_debug_frames": saved_debug_frames,
        "detected_events": len(pipeline.events.events),
        "total_violation_duration_seconds": round(total_violation, 3),
        "thresholds": {
            "phone_confidence": settings.phone_conf_threshold,
            "phone_debug_internal_confidence": settings.phone_debug_internal_conf_threshold,
            "phone_image_size": settings.phone_image_size,
            "phone_crop_image_size": settings.phone_crop_image_size,
            "phone_roi_scale": settings.phone_roi_scale,
            "phone_nms_iou": settings.phone_nms_iou_threshold,
            "keypoint_confidence": settings.keypoint_conf_threshold,
            "hand_phone_distance": settings.hand_phone_distance_threshold,
            "head_phone_distance": settings.head_phone_distance_threshold,
            "temporal_window_seconds": settings.temporal_window_seconds,
            "alert_on_ratio": settings.alert_on_ratio,
            "alert_off_ratio": settings.alert_off_ratio,
        },
        "driver_roi": list(settings.driver_roi) if settings.use_driver_roi else None,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    log.info("Processed frames: %d", frame_index)
    log.info("Wall-clock average FPS: %.2f", frame_index / elapsed)
    log.info("Phone detections: %d", pipeline.phone_detections_count)
    log.info(
        "Raw phone candidates: %d | low-confidence filtered: %d",
        pipeline.raw_phone_candidates_count,
        pipeline.low_confidence_phone_candidates_count,
    )
    if args.save_debug_frames:
        log.info("Saved missed-phone debug frames: %d | directory: %s", saved_debug_frames, debug_directory)
    log.info("Phone-use events: %d", len(pipeline.events.events))
    log.info("Total violation duration: %.2f s", total_violation)
    log.info("Output video: %s", output)
    log.info("Events file: %s", events_path)
    log.info("Run report: %s", report_path)
    return 0


def main():
    try:
        return run(build_parser().parse_args())
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        configure_logging().error("%s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
