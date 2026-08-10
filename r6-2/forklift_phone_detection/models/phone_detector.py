"""Pretrained COCO cell-phone detector with raw-candidate support."""

from typing import Dict, List, Optional, Sequence, Tuple

from .common import resolve_device, resolve_model_path


class PhoneDetector:
    def __init__(self, model_path: str, confidence: float, image_size: int, device: str):
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                "Ultralytics is not installed. Run: pip install -r requirements.txt"
            ) from exc

        self.device = resolve_device(device)
        self.confidence = confidence
        self.image_size = image_size
        self.model_path = resolve_model_path(model_path)
        self.model = YOLO(self.model_path)
        names = self.model.names
        name_items = names.items() if isinstance(names, dict) else enumerate(names)
        matches = [int(index) for index, label in name_items if str(label).strip().lower() == "cell phone"]
        if not matches:
            raise ValueError(f"Model {model_path!r} has no 'cell phone' class: {names}")
        self.phone_class_id = matches[0]
        self.phone_class_name = "cell phone"

    def _parse_result(self, result, offset: Tuple[int, int], source: str) -> List[Dict]:
        detections = []
        x_offset, y_offset = offset
        if result.boxes is None:
            return detections
        for box in result.boxes:
            class_id = int(box.cls[0].item())
            if class_id != self.phone_class_id:
                continue
            x1, y1, x2, y2 = (float(value) for value in box.xyxy[0].tolist())
            bbox = [x1 + x_offset, y1 + y_offset, x2 + x_offset, y2 + y_offset]
            detections.append(
                {
                    "bbox": bbox,
                    "confidence": float(box.conf[0].item()),
                    "center": [(bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0],
                    "source": source,
                }
            )
        return detections

    def infer(
        self,
        frame,
        offset: Tuple[int, int] = (0, 0),
        source: str = "full_frame",
        image_size: Optional[int] = None,
        confidence: Optional[float] = None,
    ) -> List[Dict]:
        batches = self.infer_batch(
            [frame],
            offsets=[offset],
            sources=[source],
            image_size=image_size,
            confidence=confidence,
        )
        return batches[0]

    def infer_batch(
        self,
        frames: Sequence,
        offsets: Optional[Sequence[Tuple[int, int]]] = None,
        sources: Optional[Sequence[str]] = None,
        image_size: Optional[int] = None,
        confidence: Optional[float] = None,
    ) -> List[List[Dict]]:
        if not frames:
            return []
        offsets = list(offsets or [(0, 0)] * len(frames))
        sources = list(sources or ["full_frame"] * len(frames))
        if len(offsets) != len(frames) or len(sources) != len(frames):
            raise ValueError("frames, offsets, and sources must have equal lengths")
        results = self.model.predict(
            list(frames),
            conf=self.confidence if confidence is None else float(confidence),
            imgsz=self.image_size if image_size is None else int(image_size),
            classes=[self.phone_class_id],
            device=self.device,
            verbose=False,
        )
        return [
            self._parse_result(result, offsets[index], sources[index])
            for index, result in enumerate(results)
        ]
