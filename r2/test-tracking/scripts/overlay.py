"""Drawing the pose overlay for the demo video.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

__all__ = [
    "OverlayStyle",
    "PersonRender",
    "draw_frame",
    "draw_head_inset",
]

# COCO keypoint indices used here.
NOSE, LEYE, REYE, LEAR, REAR, LSH, RSH = range(7)

# BGR. Chosen to stay distinguishable for the common red-green colour blindness:
# the status colours are orange vs blue rather than red vs green.
COLOR_LOOKING = (200, 130, 20)      # blue-ish: driver is looking back
COLOR_NOT_LOOKING = (30, 130, 240)  # orange: driver is not looking back
COLOR_UNKNOWN = (150, 150, 150)     # grey: not enough signal to say
COLOR_SKELETON = (240, 240, 240)
COLOR_KP = (60, 230, 250)
COLOR_TEXT = (255, 255, 255)
COLOR_PANEL = (35, 35, 35)

SKELETON_EDGES = ((LSH, RSH), (NOSE, LEYE), (NOSE, REYE), (LEYE, LEAR), (REYE, REAR))


@dataclass(frozen=True)
class OverlayStyle:
    """Tunable look of the overlay. Defaults suit a 1080p source."""

    box_thickness: int = 3
    kp_radius: int = 5
    font_scale: float = 0.7
    font_thickness: int = 2
    inset_scale: int = 6
    """Magnification of the head inset."""
    inset_margin: int = 18
    show_metrics: bool = True
    show_inset: bool = True


@dataclass
class PersonRender:
    """Everything needed to draw one person, already decided upstream."""

    bbox: tuple[float, float, float, float]
    keypoints: np.ndarray
    """(17, 3) array of x, y, confidence."""

    track_id: int | None
    status: str
    """'looking', 'not_looking' or 'unknown'."""

    status_label: str
    """Text drawn on the box, e.g. 'LOOKING BACK'."""

    metrics: dict[str, str] | None = None
    """Optional short key/value pairs rendered under the box."""


def _status_color(status: str) -> tuple[int, int, int]:
    return {
        "looking": COLOR_LOOKING,
        "not_looking": COLOR_NOT_LOOKING,
    }.get(status, COLOR_UNKNOWN)


def _put_label(
    img: np.ndarray,
    text: str,
    org: tuple[int, int],
    color: tuple[int, int, int],
    style: OverlayStyle,
    *,
    filled: bool = True,
) -> None:
    """Draw text on a solid plate so it stays readable over any background."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), base = cv2.getTextSize(text, font, style.font_scale, style.font_thickness)
    x, y = org
    # Keep the plate inside the frame; a label clipped by the right edge is
    # unreadable exactly when it matters most.
    x = max(0, min(x, img.shape[1] - tw - 12))
    y = max(th + base + 6, y)
    if filled:
        cv2.rectangle(img, (x, y - th - base - 4), (x + tw + 10, y + 2), color, -1)
    cv2.putText(
        img, text, (x + 5, y - base + 1), font, style.font_scale,
        COLOR_TEXT, style.font_thickness, cv2.LINE_AA,
    )


def draw_head_inset(
    frame: np.ndarray,
    keypoints: np.ndarray,
    style: OverlayStyle,
    *,
    kp_conf_min: float = 0.3,
) -> tuple[int, int] | None:
    """Draw a magnified crop of the head in the top-right corner.

    This is the most persuasive element for a resolution discussion: it shows
    the reviewer the actual pixels the model is working from. Returns the
    inset's bottom-left corner so callers can place a caption, or None when no
    head keypoint is confident enough to locate.
    """
    head_pts = [keypoints[i] for i in (NOSE, LEYE, REYE, LEAR, REAR)]
    good = [p for p in head_pts if p[2] >= kp_conf_min]
    if not good:
        return None

    cx = float(np.mean([p[0] for p in good]))
    cy = float(np.mean([p[1] for p in good]))

    # Size the crop from the visible head span, with a floor so a single
    # confident keypoint still yields a sensible box.
    xs = [p[0] for p in good]
    ys = [p[1] for p in good]
    span = max(max(xs) - min(xs), max(ys) - min(ys), 14.0)
    half = int(span * 1.6)

    h, w = frame.shape[:2]
    x1, y1 = max(0, int(cx) - half), max(0, int(cy) - half)
    x2, y2 = min(w, int(cx) + half), min(h, int(cy) + half)
    if x2 - x1 < 4 or y2 - y1 < 4:
        return None

    crop = frame[y1:y2, x1:x2].copy()
    # INTER_NEAREST on purpose: smoothing would hide the pixel grid, which is
    # exactly what the reviewer needs to see.
    inset = cv2.resize(crop, None, fx=style.inset_scale, fy=style.inset_scale,
                       interpolation=cv2.INTER_NEAREST)

    ih, iw = inset.shape[:2]
    max_side = min(w // 4, h // 3)
    if max(ih, iw) > max_side:
        scale = max_side / max(ih, iw)
        inset = cv2.resize(inset, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
        ih, iw = inset.shape[:2]

    px, py = w - iw - style.inset_margin, style.inset_margin
    frame[py:py + ih, px:px + iw] = inset
    cv2.rectangle(frame, (px, py), (px + iw, py + ih), COLOR_KP, 2)
    # Tie the inset back to where it came from.
    cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_KP, 1)
    cv2.line(frame, (x2, y1), (px, py + ih), COLOR_KP, 1)
    return px, py + ih


def draw_frame(
    frame: np.ndarray,
    people: list[PersonRender],
    style: OverlayStyle | None = None,
    *,
    banner: str | None = None,
    caption: str | None = None,
    kp_conf_min: float = 0.3,
) -> np.ndarray:
    """Render all overlays onto a copy of ``frame`` and return it."""
    style = style or OverlayStyle()
    out = frame.copy()

    for person in people:
        color = _status_color(person.status)
        x1, y1, x2, y2 = (int(v) for v in person.bbox)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, style.box_thickness)

        kp = person.keypoints
        # An all-zero keypoint array marks a non-person box (the vehicle), so
        # skip the skeleton and the head inset for it.
        has_pose = bool(np.any(kp[:, 2] >= kp_conf_min))
        for a, b in SKELETON_EDGES:
            if kp[a, 2] >= kp_conf_min and kp[b, 2] >= kp_conf_min:
                cv2.line(out, (int(kp[a, 0]), int(kp[a, 1])),
                         (int(kp[b, 0]), int(kp[b, 1])), COLOR_SKELETON, 2, cv2.LINE_AA)
        for i in (NOSE, LEYE, REYE, LEAR, REAR, LSH, RSH):
            if kp[i, 2] >= kp_conf_min:
                cv2.circle(out, (int(kp[i, 0]), int(kp[i, 1])), style.kp_radius,
                           COLOR_KP, -1, cv2.LINE_AA)

        # Inset first: it occupies the top-right corner, and drawing it after
        # the labels would paste over any label that reaches that far.
        inset_anchor = None
        if style.show_inset and has_pose:
            inset_anchor = draw_head_inset(out, kp, style, kp_conf_min=kp_conf_min)

        tag = f"ID:{person.track_id}  " if person.track_id is not None else ""
        label = f"{tag}{person.status_label}".strip()
        label_y = max(y1 - 6, 24)
        # If the box label would collide with the inset, drop it below the box
        # instead of letting the two overlap into an unreadable mess.
        if inset_anchor is not None:
            ix, iy_bottom = inset_anchor
            if x1 > ix - 200 and label_y < iy_bottom + 30:
                label_y = y2 + 24
        _put_label(out, label, (x1, label_y), color, style)

        if style.show_metrics and person.metrics:
            ty = label_y + 26 if label_y > y2 else y2 + 22
            for key, val in person.metrics.items():
                _put_label(out, f"{key}: {val}", (x1, ty), COLOR_PANEL, style)
                ty += 26

    if banner:
        _put_label(out, banner, (18, 40), COLOR_PANEL, style)
    if caption:
        h = out.shape[0]
        _put_label(out, caption, (18, h - 18), COLOR_PANEL, style)
    return out
