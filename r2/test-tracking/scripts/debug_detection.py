from pathlib import Path

import cv2
import torch
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = PROJECT_ROOT / "models" / "best.pt"
VIDEO_PATH = (
    PROJECT_ROOT
    / "video"
    / "Forklift Accident_ The Blind Corner.mp4"
)

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "debug"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Frame trong screenshot bạn gửi
TARGET_FRAME = 156

# Cố tình để rất thấp để xem raw capability của detector
CONF_THRESHOLD = 0.01

# CCTV high-angle -> thử resolution lớn hơn
IMAGE_SIZE = 1280

DEVICE = 0


def main():
    print("CUDA available:", torch.cuda.is_available())

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    model = YOLO(MODEL_PATH)

    print("\nModel classes:")
    print(model.names)

    cap = cv2.VideoCapture(str(VIDEO_PATH))

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {VIDEO_PATH}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, TARGET_FRAME)

    success, frame = cap.read()
    cap.release()

    if not success:
        raise RuntimeError(
            f"Cannot read frame {TARGET_FRAME}"
        )

    # IMPORTANT:
    # no classes=[...] filter here.
    # We want to inspect ALL predictions.
    results = model.predict(
        source=frame,
        conf=CONF_THRESHOLD,
        iou=0.5,
        imgsz=IMAGE_SIZE,
        device=DEVICE,
        quantize=16 if torch.cuda.is_available() else None,
        verbose=False,
    )

    result = results[0]

    print("\nDetections:")
    print("-" * 70)

    if result.boxes is None or len(result.boxes) == 0:
        print("NO DETECTIONS AT ALL")
    else:
        boxes = result.boxes

        for i in range(len(boxes)):
            cls_id = int(boxes.cls[i].item())
            conf = float(boxes.conf[i].item())
            xyxy = boxes.xyxy[i].cpu().numpy()

            class_name = model.names[cls_id]

            print(
                f"{i:02d} | "
                f"class={class_name:<20} "
                f"id={cls_id:<2} "
                f"conf={conf:.4f} "
                f"box={xyxy}"
            )

    annotated = result.plot()

    output_path = OUTPUT_DIR / f"frame_{TARGET_FRAME}_all_predictions.jpg"

    cv2.imwrite(
        str(output_path),
        annotated,
    )

    print("\nSaved:")
    print(output_path)


if __name__ == "__main__":
    main()