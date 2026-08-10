"""Video I/O helpers."""

from pathlib import Path


def create_video_writer(path, fps: float, width: int, height: int):
    import cv2

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        max(1.0, float(fps)),
        (int(width), int(height)),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not create output video: {path}")
    return writer


def default_video_output(source) -> Path:
    source = Path(source)
    return Path("output") / f"{source.stem}_annotated.mp4"


def resize_with_letterbox(frame, width: int = 1280, height: int = 720):
    """Fit a frame into an exact output size without changing its aspect ratio."""
    import cv2
    import numpy as np

    source_height, source_width = frame.shape[:2]
    if source_width <= 0 or source_height <= 0:
        raise ValueError("Cannot resize an empty frame")
    scale = min(width / source_width, height / source_height)
    resized_width = max(1, int(round(source_width * scale)))
    resized_height = max(1, int(round(source_height * scale)))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(frame, (resized_width, resized_height), interpolation=interpolation)
    canvas = np.zeros((height, width, frame.shape[2]), dtype=frame.dtype)
    x_offset = (width - resized_width) // 2
    y_offset = (height - resized_height) // 2
    canvas[y_offset:y_offset + resized_height, x_offset:x_offset + resized_width] = resized
    return canvas
