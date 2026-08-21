import numpy as np

from forklift_phone_detection.utils.video import downscale_to_width, resize_with_letterbox


def test_downscale_to_width_preserves_aspect_ratio():
    frame = np.full((2160, 3840, 3), 255, dtype=np.uint8)
    output = downscale_to_width(frame, 1920)
    assert output.shape == (1080, 1920, 3)


def test_downscale_to_width_does_not_upscale():
    frame = np.full((720, 1280, 3), 255, dtype=np.uint8)
    assert downscale_to_width(frame, 1920) is frame


def test_resize_with_letterbox_produces_hd_without_distortion():
    frame = np.full((480, 640, 3), 255, dtype=np.uint8)
    output = resize_with_letterbox(frame, 1280, 720)
    assert output.shape == (720, 1280, 3)
    # 4:3 content becomes 960x720, centered with 160-pixel black side bars.
    assert np.all(output[:, :160] == 0)
    assert np.all(output[:, 160:1120] == 255)
    assert np.all(output[:, 1120:] == 0)
