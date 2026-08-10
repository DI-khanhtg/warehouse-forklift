"""Compare pretrained YOLO phone recall across model sizes and resolutions."""

import argparse
import csv
import json
from pathlib import Path

from forklift_phone_detection.models.phone_detector import PhoneDetector


DEFAULT_MODELS = ("yolo11n.pt", "yolo11s.pt", "yolo11m.pt")
DEFAULT_IMAGE_SIZES = (640, 960, 1280)


def build_parser():
    parser = argparse.ArgumentParser(description="Compare raw COCO cell-phone predictions on one image")
    parser.add_argument("--source", required=True, help="Input JPG/PNG path")
    parser.add_argument("--output-dir", default="output/debug/phone_comparison")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument("--sizes", nargs="+", type=int, choices=DEFAULT_IMAGE_SIZES, default=list(DEFAULT_IMAGE_SIZES))
    parser.add_argument("--internal-conf", type=float, default=0.01)
    parser.add_argument("--normal-conf", type=float, default=0.10)
    return parser


def _annotate(image, detections, model_name: str, image_size: int, normal_conf: float):
    import cv2

    output = image.copy()
    for detection in detections:
        x1, y1, x2, y2 = (int(round(value)) for value in detection["bbox"])
        accepted = detection["confidence"] >= normal_conf
        color = (40, 210, 40) if accepted else (80, 130, 255)
        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2 if accepted else 1)
        cv2.putText(
            output,
            f"phone {detection['confidence']:.3f}",
            (x1, max(18, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )
    best = max((item["confidence"] for item in detections), default=None)
    status = "no candidate" if best is None else f"best={best:.3f}"
    cv2.rectangle(output, (5, 5), (min(output.shape[1] - 5, 520), 48), (20, 20, 20), -1)
    cv2.putText(
        output,
        f"{model_name} imgsz={image_size} {status}",
        (14, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (240, 240, 240),
        2,
        cv2.LINE_AA,
    )
    return output


def compare(source, output_dir, models, sizes, device="auto", internal_conf=0.01, normal_conf=0.10):
    import cv2

    source = Path(source)
    image = cv2.imread(str(source))
    if image is None:
        raise ValueError(f"OpenCV could not read image: {source}")
    output_dir = Path(output_dir) / source.stem
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for model_path in models:
        detector = PhoneDetector(model_path, internal_conf, sizes[0], device)
        for image_size in sizes:
            detections = detector.infer(
                image,
                source="full_frame",
                image_size=image_size,
                confidence=internal_conf,
            )
            best = max((item["confidence"] for item in detections), default=None)
            accepted_count = sum(item["confidence"] >= normal_conf for item in detections)
            row = {
                "model": Path(model_path).stem,
                "imgsz": image_size,
                "best_phone_conf": None if best is None else round(best, 6),
                "raw_candidates": len(detections),
                "accepted_candidates": accepted_count,
            }
            rows.append(row)
            annotated = _annotate(image, detections, Path(model_path).stem, image_size, normal_conf)
            output_path = output_dir / f"{Path(model_path).stem}_{image_size}.jpg"
            if not cv2.imwrite(str(output_path), annotated):
                raise RuntimeError(f"Could not save annotated comparison: {output_path}")

    csv_path = output_dir / "comparison.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "comparison.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return rows, output_dir


def _print_table(rows):
    headers = ("model", "imgsz", "best_phone_conf", "raw", "accepted")
    print(f"{headers[0]:<12} {headers[1]:>6} {headers[2]:>17} {headers[3]:>6} {headers[4]:>9}")
    for row in rows:
        best = "none" if row["best_phone_conf"] is None else f"{row['best_phone_conf']:.4f}"
        print(
            f"{row['model']:<12} {row['imgsz']:>6} {best:>17} "
            f"{row['raw_candidates']:>6} {row['accepted_candidates']:>9}"
        )


def main():
    args = build_parser().parse_args()
    if not 0 <= args.internal_conf <= args.normal_conf <= 1:
        print("Confidence values must satisfy 0 <= internal <= normal <= 1")
        return 2
    try:
        rows, output_dir = compare(
            args.source,
            args.output_dir,
            args.models,
            args.sizes,
            args.device,
            args.internal_conf,
            args.normal_conf,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"Comparison failed: {exc}")
        return 1
    _print_table(rows)
    print(f"Annotated outputs: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
