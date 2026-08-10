from forklift_phone_detection.config import Settings, settings_from_args
from infer_camera import apply_camera_mode, build_parser


def test_auto_camera_mode_optimizes_cpu_inference():
    settings, mode = apply_camera_mode(Settings(device="cpu"), "auto")
    assert mode == "optimized"
    assert settings.phone_model == "yolo11n.pt"
    assert settings.phone_image_size == 640
    assert settings.pose_image_size == 320
    assert settings.phone_crop_interval == 1
    assert settings.compact_phone_search_rois
    assert settings.use_pose_assisted_phone_search
    assert settings.display_mode == "phone_only"


def test_recall_camera_mode_preserves_default_pipeline():
    original = Settings(device="cpu")
    settings, mode = apply_camera_mode(original, "recall")
    assert mode == "recall"
    assert settings == original


def test_optimized_mode_respects_explicit_quality_overrides():
    args = build_parser().parse_args(
        [
            "--camera-mode", "optimized",
            "--phone-model", "yolo11m.pt",
            "--phone-image-size", "1280",
            "--pose-image-size", "960",
        ]
    )
    original = settings_from_args(args).with_overrides(device="cpu")
    settings, mode = apply_camera_mode(original, args.camera_mode, args)
    assert mode == "optimized"
    assert settings.phone_model == "yolo11m.pt"
    assert settings.phone_image_size == 1280
    assert settings.pose_image_size == 960
