from pathlib import Path

from forklift_phone_detection.models.common import PROJECT_ROOT, resolve_model_path


def test_standard_weight_names_resolve_to_weights_directory():
    resolved = Path(resolve_model_path("yolo11s.pt"))
    assert resolved == PROJECT_ROOT / "weights" / "yolo11s.pt"


def test_explicit_model_path_is_preserved(tmp_path):
    custom = tmp_path / "custom.pt"
    assert Path(resolve_model_path(str(custom))) == custom

