import numpy as np

from forklift_phone_detection.config import Settings
from forklift_phone_detection.pipeline import PhoneUsagePipeline


POSE = {
    "bbox": [40, 20, 280, 230],
    "keypoints": {
        "left_shoulder": (100, 100, 0.9),
        "right_shoulder": (200, 100, 0.9),
        "left_wrist": (110, 180, 0.9),
        "right_wrist": (190, 180, 0.9),
        "left_ear": (125, 60, 0.9),
        "right_ear": (175, 60, 0.9),
        "nose": (150, 70, 0.9),
    },
}


class FakePoseDetector:
    def infer(self, frame, offset=(0, 0), target_center=None):
        return POSE


class LocalPhoneDetector:
    def infer(self, frame, offset=(0, 0), source="full_frame", image_size=None, confidence=None):
        return []

    def infer_batch(self, frames, offsets, sources, image_size=None, confidence=None):
        outputs = []
        for source in sources:
            if source == "left_wrist_roi":
                outputs.append([{
                    "bbox": [100, 160, 120, 200],
                    "center": [110, 180],
                    "confidence": 0.35,
                    "source": source,
                }])
            else:
                outputs.append([])
        return outputs


class LowConfidenceDetector:
    def infer(self, frame, offset=(0, 0), source="full_frame", image_size=None, confidence=None):
        return [{
            "bbox": [100, 160, 120, 200],
            "center": [110, 180],
            "confidence": 0.05,
            "source": source,
        }]


class CountingEmptyDetector:
    def __init__(self):
        self.batch_calls = 0

    def infer(self, frame, offset=(0, 0), source="full_frame", image_size=None, confidence=None):
        return []

    def infer_batch(self, frames, offsets, sources, image_size=None, confidence=None):
        self.batch_calls += 1
        return [[] for _ in frames]


class FullFramePhoneDetector:
    def infer(self, frame, offset=(0, 0), source="full_frame", image_size=None, confidence=None):
        return [
            {
                "bbox": [100, 160, 120, 200],
                "center": [110, 180],
                "confidence": 0.8,
                "source": "full_frame",
            },
            {
                "bbox": [400, 160, 420, 200],
                "center": [410, 180],
                "confidence": 0.8,
                "source": "full_frame",
            },
        ]


class PoseSequenceDetector:
    def __init__(self, poses):
        self.poses = list(poses)
        self.index = 0

    def infer(self, frame, offset=(0, 0), target_center=None, target_bbox=None):
        pose = self.poses[min(self.index, len(self.poses) - 1)]
        self.index += 1
        return pose


def shifted_pose(x_offset):
    return {
        "bbox": [40 + x_offset, 20, 280 + x_offset, 230],
        "keypoints": {
            name: (point[0] + x_offset, point[1], point[2])
            for name, point in POSE["keypoints"].items()
        },
        "confidence": 0.9,
    }


def test_local_roi_recovers_phone_and_can_trigger_temporal_alert():
    settings = Settings(
        phone_conf_threshold=0.10,
        phone_image_size=640,
        phone_crop_image_size=640,
        pose_image_size=640,
    )
    pipeline = PhoneUsagePipeline(settings, 2, LocalPhoneDetector(), FakePoseDetector())
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    _, first = pipeline.process_frame(frame.copy(), 0.0)
    _, second = pipeline.process_frame(frame.copy(), 0.5)
    _, third = pipeline.process_frame(frame.copy(), 1.0)
    assert first["phones"][0]["source"] == "left_wrist_roi"
    assert first["instant"]["behavior"] == "HANDHELD_PHONE_USE"
    assert first["wrist_phone_detected"]
    assert second["temporal"].state == "NORMAL"
    assert third["temporal"].state == "USING_PHONE"


def test_low_confidence_candidate_is_distinguished_from_no_candidate():
    settings = Settings(
        phone_conf_threshold=0.10,
        use_pose_assisted_phone_search=False,
        phone_image_size=640,
        pose_image_size=640,
    )
    pipeline = PhoneUsagePipeline(settings, 10, LowConfidenceDetector(), FakePoseDetector())
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    _, result = pipeline.process_frame(frame, 0.0)
    assert result["phones"] == []
    assert len(result["low_confidence_phone_candidates"]) == 1
    assert result["phone_candidate_status"] == "low_confidence_filtered"
    assert result["temporal"].state == "NORMAL"


def test_local_crop_interval_is_respected_even_when_crop_finds_nothing():
    detector = CountingEmptyDetector()
    settings = Settings(
        phone_crop_interval=3,
        compact_phone_search_rois=False,
        phone_image_size=640,
        phone_crop_image_size=320,
        pose_image_size=320,
    )
    pipeline = PhoneUsagePipeline(settings, 30, detector, FakePoseDetector())
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    for index in range(4):
        pipeline.process_frame(frame.copy(), index / 30)
    assert detector.batch_calls == 2


def test_skipped_local_crop_does_not_replay_stale_phone_detection():
    detector = LocalPhoneDetector()
    settings = Settings(
        phone_crop_interval=3,
        compact_phone_search_rois=False,
        phone_image_size=640,
        phone_crop_image_size=320,
        pose_image_size=320,
    )
    pipeline = PhoneUsagePipeline(settings, 30, detector, FakePoseDetector())
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    _, first = pipeline.process_frame(frame.copy(), 0.0)
    _, skipped = pipeline.process_frame(frame.copy(), 1 / 30)
    assert first["phones"]
    assert skipped["phones"] == []
    assert skipped["instant"]["using_phone"] is False


def test_held_stale_pose_is_draw_only_and_cannot_create_phone_evidence():
    settings = Settings(
        use_pose_assisted_phone_search=False,
        driver_track_max_center_jump=0.25,
        phone_image_size=640,
        pose_image_size=320,
    )
    detector = FullFramePhoneDetector()
    poses = PoseSequenceDetector([shifted_pose(0), shifted_pose(300)])
    pipeline = PhoneUsagePipeline(settings, 10, detector, poses)
    frame = np.zeros((240, 640, 3), dtype=np.uint8)
    _, first = pipeline.process_frame(frame.copy(), 0.0)
    _, held = pipeline.process_frame(frame.copy(), 0.1)
    assert first["instant"]["using_phone"]
    assert held["pose_tracking_status"] == "held"
    assert held["pose"]["tracking_stale"]
    assert held["logic_pose"] is None
    assert not held["instant"]["using_phone"]


def test_driver_identity_change_resets_temporal_votes():
    settings = Settings(
        use_pose_assisted_phone_search=False,
        driver_track_max_center_jump=0.25,
        driver_track_max_missed_frames=1,
        phone_image_size=640,
        pose_image_size=320,
        temporal_window_seconds=1.0,
        min_window_fill_ratio=1.0,
    )
    detector = FullFramePhoneDetector()
    poses = PoseSequenceDetector([
        shifted_pose(0),
        shifted_pose(0),
        shifted_pose(0),
        shifted_pose(300),
        shifted_pose(300),
    ])
    pipeline = PhoneUsagePipeline(settings, 2, detector, poses)
    frame = np.zeros((240, 640, 3), dtype=np.uint8)
    pipeline.process_frame(frame.copy(), 0.0)
    pipeline.process_frame(frame.copy(), 0.5)
    _, active = pipeline.process_frame(frame.copy(), 1.0)
    assert active["temporal"].state == "USING_PHONE"
    pipeline.process_frame(frame.copy(), 1.5)
    _, changed = pipeline.process_frame(frame.copy(), 2.0)
    assert changed["driver_identity_changed"]
    assert changed["driver_track_id"] != active["driver_track_id"]
    assert changed["temporal"].valid_frames == 1
    assert changed["temporal"].state == "NORMAL"
