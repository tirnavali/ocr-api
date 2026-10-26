"""Exact coordinate geometry transformations between working image, source image, and PDF user space."""

from __future__ import annotations

import math
from dataclasses import dataclass

from ocr_service.domain.models import BBoxPt, BBoxPx


@dataclass(frozen=True, slots=True)
class WorkingToSourceTransform:
    """Parameters needed to map a bounding box from working OCR image pixels back to source pixels."""

    source_width_px: int
    source_height_px: int
    working_width_px: int
    working_height_px: int
    scale_x: float = 1.0  # working_width / source_width (before cardinal rotation)
    scale_y: float = 1.0  # working_height / source_height (before cardinal rotation)
    deskew_angle_deg: float = 0.0  # Counter-clockwise rotation applied to working image
    cardinal_rotation_deg: int = 0  # Clockwise rotation (0, 90, 180, 270) applied to working copy


def map_point_inverse_deskew(
    x: float,
    y: float,
    center_x: float,
    center_y: float,
    deskew_angle_deg: float,
) -> tuple[float, float]:
    """Map point (x, y) from a deskewed image back to the pre-deskewed image coordinate frame.

    OpenCV's getRotationMatrix2D(center, angle_deg, 1.0) rotates counter-clockwise in screen space
    (where Y points down) by:
        x' = cx + (x - cx)*cos(theta) + (y - cy)*sin(theta)
        y' = cy - (x - cx)*sin(theta) + (y - cy)*cos(theta)
    Therefore, the exact inverse mapping (x', y') -> (x, y) is:
        x = cx + (x' - cx)*cos(theta) - (y' - cy)*sin(theta)
        y = cy + (x' - cx)*sin(theta) + (y' - cy)*cos(theta)
    """
    if abs(deskew_angle_deg) < 1e-5:
        return x, y
    theta = math.radians(deskew_angle_deg)
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    dx = x - center_x
    dy = y - center_y
    orig_x = center_x + dx * cos_t - dy * sin_t
    orig_y = center_y + dx * sin_t + dy * cos_t
    return orig_x, orig_y


def map_point_inverse_cardinal(
    x: float,
    y: float,
    rotated_w: float,
    rotated_h: float,
    cardinal_rotation_deg: int,
) -> tuple[float, float]:
    """Map point (x, y) from an image rotated clockwise by cardinal_rotation_deg back to unrotated."""
    rot = cardinal_rotation_deg % 360
    if rot == 0:
        return x, y
    if rot == 90:
        # Clockwise 90: (x_rot, y_rot) = (H_orig - y_orig, x_orig) where rotated_w = H_orig
        return y, rotated_w - x
    if rot == 180:
        return rotated_w - x, rotated_h - y
    if rot == 270:
        # Clockwise 270 (CCW 90): (x_rot, y_rot) = (y_orig, W_orig - x_orig) where rotated_h = W_orig
        return rotated_h - y, x
    return x, y


def working_bbox_to_source_bbox(
    bbox_work: BBoxPx,
    transform: WorkingToSourceTransform,
) -> BBoxPx:
    """Transform a word bounding box from working OCR pixel coordinates to source image pixels (FR-06)."""
    cx = transform.working_width_px / 2.0
    cy = transform.working_height_px / 2.0

    corners = [
        (bbox_work.left, bbox_work.top),
        (bbox_work.right, bbox_work.top),
        (bbox_work.right, bbox_work.bottom),
        (bbox_work.left, bbox_work.bottom),
    ]

    mapped_xs: list[float] = []
    mapped_ys: list[float] = []

    for wx, wy in corners:
        # Step 1: Invert deskew rotation around working image center
        ux, uy = map_point_inverse_deskew(wx, wy, cx, cy, transform.deskew_angle_deg)
        # Step 2: Invert cardinal rotation if applied on working image
        rx, ry = map_point_inverse_cardinal(
            ux,
            uy,
            float(transform.working_width_px),
            float(transform.working_height_px),
            transform.cardinal_rotation_deg,
        )
        # Step 3: Invert DPI scaling factor
        sx = rx / transform.scale_x if transform.scale_x > 0 else rx
        sy = ry / transform.scale_y if transform.scale_y > 0 else ry
        mapped_xs.append(sx)
        mapped_ys.append(sy)

    min_x = max(0.0, min(mapped_xs))
    max_x = min(float(transform.source_width_px), max(mapped_xs))
    min_y = max(0.0, min(mapped_ys))
    max_y = min(float(transform.source_height_px), max(mapped_ys))

    width = max(1.0, max_x - min_x)
    height = max(1.0, max_y - min_y)

    return BBoxPx(left=min_x, top=min_y, width=width, height=height)


def source_bbox_to_pdf_points(
    bbox_src: BBoxPx,
    *,
    source_width_px: int,
    source_height_px: int,
    page_width_pt: float,
    page_height_pt: float,
) -> tuple[BBoxPt, float]:
    """Convert a top-left source pixel bounding box into bottom-left PDF user space points.

    Returns (bbox_pdf_pt, baseline_pdf_y).
    In PDF user space:
      - Origin (0, 0) is at the bottom-left corner of the page.
      - X increases to the right: x_pt = left_px * (page_width_pt / source_width_px)
      - Y increases upward: y_pt = page_height_pt - bottom_px * (page_height_pt / source_height_px)
    """
    scale_x = page_width_pt / float(max(1, source_width_px))
    scale_y = page_height_pt / float(max(1, source_height_px))

    x_pt = bbox_src.left * scale_x
    w_pt = max(0.5, bbox_src.width * scale_x)
    h_pt = max(0.5, bbox_src.height * scale_y)
    y_bottom_pt = max(0.0, page_height_pt - (bbox_src.bottom * scale_y))

    # Standard Latin/Turkish font baseline sits ~18% of bbox height above the bottom descender line
    baseline_y_pt = y_bottom_pt + (0.18 * h_pt)

    return BBoxPt(x=x_pt, y=y_bottom_pt, width=w_pt, height=h_pt), baseline_y_pt


def bbox_iou_pt(a: BBoxPt, b: BBoxPt) -> float:
    """Compute Intersection-over-Union (IoU) of two PDF point bounding boxes for alignment verification."""
    inter_left = max(a.x, b.x)
    inter_bottom = max(a.y, b.y)
    inter_right = min(a.right, b.right)
    inter_top = min(a.top, b.top)

    if inter_right <= inter_left or inter_top <= inter_bottom:
        return 0.0

    inter_area = (inter_right - inter_left) * (inter_top - inter_bottom)
    area_a = a.width * a.height
    area_b = b.width * b.height
    union_area = area_a + area_b - inter_area
    if union_area <= 0:
        return 0.0
    return inter_area / union_area
