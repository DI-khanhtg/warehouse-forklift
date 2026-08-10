"""Temporal driver pose association, smoothing, and outlier rejection."""

from copy import deepcopy

from .geometry import bbox_center, bbox_iou, euclidean_distance, operator_scale


SWAPPABLE_KEYPOINT_PAIRS = (
    ("left_elbow", "right_elbow"),
    ("left_wrist", "right_wrist"),
)


class DriverPoseTracker:
    def __init__(
        self,
        smoothing_alpha: float = 0.45,
        max_center_jump: float = 1.0,
        minimum_iou: float = 0.05,
        max_missed_frames: int = 4,
        keypoint_max_jump: float = 1.0,
    ):
        self.smoothing_alpha = float(smoothing_alpha)
        self.max_center_jump = float(max_center_jump)
        self.minimum_iou = float(minimum_iou)
        self.max_missed_frames = int(max_missed_frames)
        self.keypoint_max_jump = float(keypoint_max_jump)
        self.pose = None
        self.trusted_keypoints = {}
        self.missed_frames = 0
        self.track_id = 0

    @property
    def target_bbox(self):
        return self.pose.get("bbox") if self.pose is not None else None

    def reset(self):
        self.pose = None
        self.trusted_keypoints = {}
        self.missed_frames = 0

    def _tag(self, pose, status: str, fresh: bool, **extra):
        tagged = deepcopy(pose)
        tagged.update(
            {
                "track_id": self.track_id,
                "tracking_status": status,
                "tracking_fresh": fresh,
                "tracking_stale": not fresh,
                "tracking_missed_frames": self.missed_frames,
            }
        )
        tagged.update(extra)
        return tagged

    def _acquire(self, candidate, status: str = "acquired"):
        self.track_id += 1
        self.missed_frames = 0
        self.trusted_keypoints = deepcopy(candidate.get("keypoints", {}))
        self.pose = self._tag(candidate, status, True)
        return self.pose

    def _hold_or_lose(self, reason: str):
        self.missed_frames += 1
        if self.pose is None or self.missed_frames > self.max_missed_frames:
            self.reset()
            return None
        held = self._tag(self.pose, "held", False, tracking_reason=reason)
        held["tracking_missed_frames"] = self.missed_frames
        return held

    def _normalize_side_pairs(self, candidate_keypoints, previous_keypoints, scale):
        normalized = dict(candidate_keypoints)
        swapped_pairs = []
        for left_name, right_name in SWAPPABLE_KEYPOINT_PAIRS:
            new_left = normalized.get(left_name)
            new_right = normalized.get(right_name)
            old_left = previous_keypoints.get(left_name)
            old_right = previous_keypoints.get(right_name)
            if any(point is None for point in (new_left, new_right, old_left, old_right)):
                continue
            direct = euclidean_distance(new_left, old_left) + euclidean_distance(new_right, old_right)
            swapped = euclidean_distance(new_left, old_right) + euclidean_distance(new_right, old_left)
            if swapped + 0.15 * scale < direct:
                normalized[left_name], normalized[right_name] = new_right, new_left
                swapped_pairs.append(f"{left_name}/{right_name}")
        return normalized, swapped_pairs

    def _smooth(self, candidate, frame_shape):
        previous = self.pose
        alpha = self.smoothing_alpha
        previous_keypoints = self.trusted_keypoints
        candidate_keypoints = candidate.get("keypoints", {})
        scale_pose = dict(previous)
        scale_pose["keypoints"] = previous_keypoints
        scale = operator_scale(scale_pose, frame_shape)
        candidate_keypoints, swapped_pairs = self._normalize_side_pairs(
            candidate_keypoints, previous_keypoints, scale
        )
        smoothed_keypoints = {}
        next_trusted_keypoints = dict(previous_keypoints)
        rejected_keypoints = []
        for name, current in candidate_keypoints.items():
            old = previous_keypoints.get(name)
            if old is None:
                accepted = tuple(current)
                smoothed_keypoints[name] = accepted
                next_trusted_keypoints[name] = accepted
                continue
            jump = euclidean_distance(current, old) / max(scale, 1.0)
            if jump > self.keypoint_max_jump:
                rejected_keypoints.append(name)
                continue
            confidence = float(current[2]) if len(current) > 2 else 1.0
            accepted = (
                alpha * float(current[0]) + (1.0 - alpha) * float(old[0]),
                alpha * float(current[1]) + (1.0 - alpha) * float(old[1]),
                confidence,
            )
            smoothed_keypoints[name] = accepted
            next_trusted_keypoints[name] = accepted
        self.trusted_keypoints = next_trusted_keypoints
        previous_bbox = previous["bbox"]
        current_bbox = candidate["bbox"]
        smoothed_bbox = [
            alpha * float(current) + (1.0 - alpha) * float(old)
            for current, old in zip(current_bbox, previous_bbox)
        ]
        return {
            "keypoints": smoothed_keypoints,
            "bbox": smoothed_bbox,
            "confidence": float(candidate.get("confidence", 0.0)),
            "tracking_rejected_keypoints": rejected_keypoints,
            "tracking_swapped_pairs": swapped_pairs,
        }

    def update(self, candidate, frame_shape):
        if candidate is None:
            return self._hold_or_lose("pose_missing")
        if self.pose is None:
            return self._acquire(candidate)

        previous_bbox = self.pose["bbox"]
        candidate_bbox = candidate["bbox"]
        previous_width = max(1.0, float(previous_bbox[2]) - float(previous_bbox[0]))
        previous_height = max(1.0, float(previous_bbox[3]) - float(previous_bbox[1]))
        previous_diagonal = (previous_width ** 2 + previous_height ** 2) ** 0.5
        center_jump = euclidean_distance(
            bbox_center(previous_bbox), bbox_center(candidate_bbox)
        ) / previous_diagonal
        overlap = bbox_iou(previous_bbox, candidate_bbox)
        center_jump_rejected = center_jump > self.max_center_jump
        weak_continuity = (
            overlap < self.minimum_iou
            and center_jump > 0.5 * self.max_center_jump
        )
        if center_jump_rejected or weak_continuity:
            held = self._hold_or_lose("driver_jump_rejected")
            if held is not None:
                held["tracking_center_jump"] = center_jump
                held["tracking_iou"] = overlap
                return held
            return self._acquire(candidate, "reacquired")

        self.missed_frames = 0
        smoothed = self._smooth(candidate, frame_shape)
        self.pose = self._tag(
            smoothed,
            "tracked",
            True,
            tracking_center_jump=center_jump,
            tracking_iou=overlap,
        )
        return self.pose
