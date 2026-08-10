"""Shared two-stage phone/pose pipeline used by video and live inference."""

import logging
import time
from collections import deque
from typing import Optional

from .config import Settings
from .logic.event_tracker import EventTracker
from .logic.geometry import (
    is_point_inside_bbox,
    normalized_distance,
    nms_detections,
    operator_scale,
)
from .logic.phone_usage import classify_phone_usage
from .logic.pose_tracker import DriverPoseTracker
from .logic.search_rois import generate_phone_search_rois
from .logic.temporal_filter import TemporalFilter
from .models.phone_detector import PhoneDetector
from .models.pose_detector import PoseDetector
from .utils.drawing import draw_annotations


class PhoneUsagePipeline:
    def __init__(
        self,
        settings: Settings,
        source_fps: float,
        phone_detector=None,
        pose_detector=None,
    ):
        self.settings = settings
        self.source_fps = max(1.0, float(source_fps))
        self.phone_detector = phone_detector or PhoneDetector(
            settings.phone_model,
            settings.phone_conf_threshold,
            settings.phone_image_size,
            settings.device,
        )
        self.pose_detector = pose_detector or PoseDetector(
            settings.pose_model,
            settings.pose_conf_threshold,
            settings.keypoint_conf_threshold,
            settings.pose_image_size,
            settings.device,
        )
        self.temporal = TemporalFilter(
            fps=self.source_fps,
            window_seconds=settings.temporal_window_seconds,
            alert_on_ratio=settings.alert_on_ratio,
            alert_off_ratio=settings.alert_off_ratio,
            min_window_fill_ratio=settings.min_window_fill_ratio,
        )
        self.events = EventTracker()
        self.processing_times = deque(maxlen=30)
        self.processed_frames = 0
        self.phone_detections_count = 0
        self.raw_phone_candidates_count = 0
        self.low_confidence_phone_candidates_count = 0
        self.last_timestamp = 0.0
        self.log = logging.getLogger("r6_phone_detection")
        self.active_track_id = None
        self.pose_tracker = None
        if settings.enable_driver_tracking:
            self.pose_tracker = DriverPoseTracker(
                smoothing_alpha=settings.pose_smoothing_alpha,
                max_center_jump=settings.driver_track_max_center_jump,
                minimum_iou=settings.driver_track_min_iou,
                max_missed_frames=settings.driver_track_max_missed_frames,
                keypoint_max_jump=settings.pose_keypoint_max_jump,
            )

    def _roi(self, frame):
        height, width = frame.shape[:2]
        if not self.settings.use_driver_roi:
            return frame, (0, 0), None, None
        x1n, y1n, x2n, y2n = self.settings.driver_roi
        x1, y1 = int(round(x1n * width)), int(round(y1n * height))
        x2, y2 = int(round(x2n * width)), int(round(y2n * height))
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(width, x2), min(height, y2)
        if x2 <= x1 or y2 <= y1:
            raise ValueError("Driver ROI is empty after conversion to pixels")
        center = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
        return frame[y1:y2, x1:x2], (x1, y1), (x1, y1, x2, y2), center

    def _internal_phone_confidence(self) -> float:
        if self.settings.debug_phone_detection:
            return self.settings.phone_debug_internal_conf_threshold
        return self.settings.phone_conf_threshold

    def _full_frame_phone_candidates(self, model_frame, offset):
        return self.phone_detector.infer(
            model_frame,
            offset=offset,
            source="full_frame",
            image_size=self.settings.phone_image_size,
            confidence=self._internal_phone_confidence(),
        )

    def _local_phone_candidates(self, frame, search_rois):
        if not search_rois:
            return []
        crops, offsets, sources = [], [], []
        for roi in search_rois:
            x1, y1, x2, y2 = roi["bbox"]
            crops.append(frame[y1:y2, x1:x2])
            offsets.append((x1, y1))
            sources.append(roi["source"])
        if hasattr(self.phone_detector, "infer_batch"):
            batches = self.phone_detector.infer_batch(
                crops,
                offsets=offsets,
                sources=sources,
                image_size=self.settings.phone_crop_image_size,
                confidence=self._internal_phone_confidence(),
            )
        else:
            batches = [
                self.phone_detector.infer(crop, offset=offset)
                for crop, offset in zip(crops, offsets)
            ]
            for source, detections in zip(sources, batches):
                for detection in detections:
                    detection.setdefault("source", source)
        return [detection for detections in batches for detection in detections]

    def _detection_search_rois(self, search_rois):
        if not self.settings.compact_phone_search_rois:
            return search_rois
        sources = {roi["source"] for roi in search_rois}
        if "both_hands_roi" not in sources:
            return search_rois
        return [
            roi for roi in search_rois
            if roi["source"] not in {"left_wrist_roi", "right_wrist_roi"}
        ]

    def _wrist_phone_detected(self, phones, pose, frame_shape) -> bool:
        if not pose:
            return False
        keypoints = pose.get("keypoints", pose)
        wrists = [keypoints.get(name) for name in ("left_wrist", "right_wrist")]
        wrists = [point for point in wrists if point is not None]
        if not wrists:
            return False
        scale = operator_scale(pose, frame_shape)
        for phone in phones:
            for wrist in wrists:
                if is_point_inside_bbox(wrist, phone["bbox"], margin=2.0):
                    return True
                distance = normalized_distance(phone["center"], wrist, scale)
                if distance is not None and distance < self.settings.hand_phone_distance_threshold:
                    return True
        return False

    def _infer_pose(self, model_frame, offset, roi_center):
        target_bbox = self.pose_tracker.target_bbox if self.pose_tracker else None
        if target_bbox is None:
            return self.pose_detector.infer(
                model_frame, offset=offset, target_center=roi_center
            )
        try:
            return self.pose_detector.infer(
                model_frame,
                offset=offset,
                target_center=roi_center,
                target_bbox=target_bbox,
            )
        except TypeError as exc:
            if "target_bbox" not in str(exc):
                raise
            return self.pose_detector.infer(
                model_frame, offset=offset, target_center=roi_center
            )

    def process_frame(self, frame, timestamp: Optional[float] = None):
        started = time.perf_counter()
        if timestamp is None:
            timestamp = self.processed_frames / self.source_fps
        self.last_timestamp = float(timestamp)
        model_frame, offset, roi_box, roi_center = self._roi(frame)
        pose_candidate = self._infer_pose(model_frame, offset, roi_center)
        if self.pose_tracker:
            pose = self.pose_tracker.update(pose_candidate, frame.shape)
            logic_pose = pose if pose and pose.get("tracking_fresh", False) else None
        else:
            pose = pose_candidate
            logic_pose = pose

        current_track_id = pose.get("track_id") if pose else None
        identity_changed = False
        if self.active_track_id is not None and current_track_id != self.active_track_id:
            identity_changed = True
            self.temporal.reset()
            self.events.update(False, timestamp, "NORMAL", 0.0)
            self.log.info(
                "Driver track changed: %s -> %s; temporal evidence reset",
                self.active_track_id,
                current_track_id,
            )
        self.active_track_id = current_track_id

        raw_candidates = self._full_frame_phone_candidates(model_frame, offset)
        search_rois = []
        if self.settings.use_pose_assisted_phone_search:
            search_rois = generate_phone_search_rois(
                logic_pose, frame.shape, self.settings.phone_roi_scale
            )
            detection_rois = self._detection_search_rois(search_rois)
            run_local_search = (
                bool(detection_rois)
                and self.processed_frames % self.settings.phone_crop_interval == 0
            )
            if run_local_search:
                raw_candidates.extend(
                    self._local_phone_candidates(frame, detection_rois)
                )

        all_raw_candidates = list(raw_candidates)
        merged_raw_candidates = nms_detections(
            all_raw_candidates, self.settings.phone_nms_iou_threshold
        )
        low_confidence_candidates = [
            detection for detection in all_raw_candidates
            if detection["confidence"] < self.settings.phone_conf_threshold
        ]
        phones = nms_detections(
            [
                detection for detection in all_raw_candidates
                if detection["confidence"] >= self.settings.phone_conf_threshold
            ],
            self.settings.phone_nms_iou_threshold,
        )
        if phones:
            candidate_status = "accepted"
        elif low_confidence_candidates:
            candidate_status = "low_confidence_filtered"
        else:
            candidate_status = "no_candidate"

        instant = classify_phone_usage(phones, logic_pose, frame.shape, self.settings)
        temporal = self.temporal.update(
            instant["using_phone"], instant["behavior"], timestamp=timestamp
        )
        self.events.update(
            temporal.state == "USING_PHONE",
            timestamp,
            temporal.behavior,
            instant.get("phone_confidence", 0.0),
        )
        elapsed = max(time.perf_counter() - started, 1e-9)
        self.processing_times.append(elapsed)
        self.processed_frames += 1
        self.phone_detections_count += len(phones)
        self.raw_phone_candidates_count += len(all_raw_candidates)
        self.low_confidence_phone_candidates_count += len(low_confidence_candidates)

        log_every = self.settings.phone_debug_log_every
        if self.settings.debug_phone_detection and log_every > 0 and self.processed_frames % log_every == 0:
            raw_summary = ", ".join(
                f"{item['confidence']:.3f}@{item['source']}" for item in all_raw_candidates
            ) or "none"
            self.log.info(
                "Phone debug frame=%d status=%s raw=[%s] accepted=%d",
                self.processed_frames,
                candidate_status,
                raw_summary,
                len(phones),
            )

        processing_fps = len(self.processing_times) / sum(self.processing_times)
        annotated = draw_annotations(
            frame,
            phones,
            pose,
            instant,
            temporal,
            processing_fps,
            self.events.current_duration,
            roi=roi_box,
            phone_search_rois=search_rois,
            raw_phone_candidates=low_confidence_candidates,
            debug_phone_detection=self.settings.debug_phone_detection,
            show_phone_search_rois=self.settings.show_phone_search_rois,
            display_mode=self.settings.display_mode,
            show_pose=self.settings.show_pose,
            show_debug_lines=self.settings.show_debug_lines,
        )
        keypoints = logic_pose.get("keypoints", logic_pose) if logic_pose else {}
        visible_wrist_count = sum(
            1 for name in ("left_wrist", "right_wrist") if keypoints.get(name) is not None
        )
        wrist_phone_detected = self._wrist_phone_detected(
            phones, logic_pose, frame.shape
        )
        return annotated, {
            "phones": phones,
            "raw_phone_candidates": all_raw_candidates,
            "merged_raw_phone_candidates": merged_raw_candidates,
            "low_confidence_phone_candidates": low_confidence_candidates,
            "phone_candidate_status": candidate_status,
            "phone_search_rois": search_rois,
            "visible_wrist_count": visible_wrist_count,
            "wrist_phone_detected": wrist_phone_detected,
            "pose": pose,
            "pose_candidate": pose_candidate,
            "logic_pose": logic_pose,
            "pose_tracking_status": (
                pose.get("tracking_status", "untracked") if pose else "lost"
            ),
            "driver_track_id": current_track_id,
            "driver_identity_changed": identity_changed,
            "instant": instant,
            "temporal": temporal,
            "processing_fps": processing_fps,
            "timestamp": timestamp,
        }

    def finalize(self, timestamp: Optional[float] = None):
        self.events.finalize(self.last_timestamp if timestamp is None else timestamp)

    @property
    def average_processing_fps(self) -> float:
        return len(self.processing_times) / sum(self.processing_times) if self.processing_times else 0.0
