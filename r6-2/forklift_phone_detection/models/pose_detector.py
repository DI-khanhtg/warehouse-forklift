"""Pretrained YOLO pose wrapper selecting the cockpit operator."""

from typing import Dict, Optional, Tuple

from ..logic.geometry import bbox_area, bbox_center, bbox_iou, euclidean_distance
from .common import resolve_device, resolve_model_path


COCO_KEYPOINT_NAMES = (
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip", "left_knee",
    "right_knee", "left_ankle", "right_ankle",
)

RELEVANT_KEYPOINTS = {
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
}


def select_pose_index(boxes, confidences=None, target_bbox=None, target_center=None):
    """Select one driver, preferring temporal bbox continuity when available."""
    if not boxes:
        return None
    confidences = confidences or [1.0] * len(boxes)
    if target_bbox is not None:
        target_width = max(1.0, float(target_bbox[2]) - float(target_bbox[0]))
        target_height = max(1.0, float(target_bbox[3]) - float(target_bbox[1]))
        target_diagonal = (target_width ** 2 + target_height ** 2) ** 0.5
        target_area = max(1.0, bbox_area(target_bbox))

        def continuity_score(index):
            box = boxes[index]
            overlap = bbox_iou(box, target_bbox)
            center_distance = euclidean_distance(
                bbox_center(box), bbox_center(target_bbox)
            ) / target_diagonal
            area = max(1.0, bbox_area(box))
            area_similarity = min(area, target_area) / max(area, target_area)
            return (
                3.0 * overlap
                - center_distance
                + 0.5 * area_similarity
                + 0.2 * float(confidences[index])
            )

        return max(range(len(boxes)), key=continuity_score)
    if target_center is not None:
        return min(
            range(len(boxes)),
            key=lambda index: euclidean_distance(bbox_center(boxes[index]), target_center),
        )
    return max(
        range(len(boxes)),
        key=lambda index: bbox_area(boxes[index]) * (0.75 + 0.25 * float(confidences[index])),
    )


class PoseDetector:
    def __init__(
        self,
        model_path: str,
        confidence: float,
        keypoint_confidence: float,
        image_size: int,
        device: str,
    ):
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                "Ultralytics is not installed. Run: pip install -r requirements.txt"
            ) from exc
        self.device = resolve_device(device)
        self.confidence = confidence
        self.keypoint_confidence = keypoint_confidence
        self.image_size = image_size
        self.model_path = resolve_model_path(model_path)
        self.model = YOLO(self.model_path)

    def infer(
        self,
        frame,
        offset: Tuple[int, int] = (0, 0),
        target_center: Optional[Tuple[float, float]] = None,
        target_bbox=None,
    ) -> Optional[Dict]:
        results = self.model.predict(
            frame,
            conf=self.confidence,
            imgsz=self.image_size,
            device=self.device,
            verbose=False,
        )
        if not results or results[0].boxes is None or results[0].keypoints is None:
            return None

        boxes = results[0].boxes.xyxy.cpu().tolist()
        if not boxes:
            return None
        x_offset, y_offset = offset
        full_boxes = [
            [box[0] + x_offset, box[1] + y_offset, box[2] + x_offset, box[3] + y_offset]
            for box in boxes
        ]
        confidences = results[0].boxes.conf.cpu().tolist()
        selected = select_pose_index(
            full_boxes,
            confidences=confidences,
            target_bbox=target_bbox,
            target_center=target_center,
        )

        data = results[0].keypoints.data[selected].cpu().tolist()
        keypoints = {}
        for index, values in enumerate(data):
            if index >= len(COCO_KEYPOINT_NAMES):
                break
            name = COCO_KEYPOINT_NAMES[index]
            if name not in RELEVANT_KEYPOINTS:
                continue
            x, y = float(values[0]), float(values[1])
            conf = float(values[2]) if len(values) > 2 else 1.0
            if conf >= self.keypoint_confidence:
                keypoints[name] = (x + x_offset, y + y_offset, conf)

        box_confidence = float(results[0].boxes.conf[selected].item())
        return {"keypoints": keypoints, "bbox": full_boxes[selected], "confidence": box_confidence}
