"""Shared model helpers."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
STANDARD_WEIGHT_NAMES = {
    "yolo11n.pt",
    "yolo11s.pt",
    "yolo11m.pt",
    "yolo11n-pose.pt",
}


def resolve_model_path(requested: str) -> str:
    """Use a locally cached weight from weights/ when a bare name is supplied."""
    path = Path(requested)
    if path.is_absolute() or path.parent != Path("."):
        return str(path)
    cached_path = PROJECT_ROOT / "weights" / path.name
    if cached_path.is_file() or path.name in STANDARD_WEIGHT_NAMES:
        cached_path.parent.mkdir(parents=True, exist_ok=True)
        return str(cached_path)
    return requested


def resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    try:
        import torch

        return "0" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"
