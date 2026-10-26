"""Unit tests for coordinate geometry transformations (Working Px -> Source Px -> PDF Pt)."""

from __future__ import annotations

import math

import pytest

from ocr_service.domain.models import BBoxPt, BBoxPx
from ocr_service.pipeline.geometry import (
    WorkingToSourceTransform,
    bbox_iou_pt,
    map_point_inverse_cardinal,
    map_point_inverse_deskew,
    source_bbox_to_pdf_points,
    working_bbox_to_source_bbox,
)


def test_source_px_to_pdf_points_bottom_left_origin() -> None:
    # 300 DPI page: 2400 x 3600 px -> 576 x 864 pt (8 x 12 inches)
    src_w, src_h = 2400, 3600
    w_pt, h_pt = 576.0, 864.0

    # Word box near top-left: left=300px (1 in = 72pt), top=300px (1 in from top), w=600px (144pt), h=150px (36pt)
    # Bottom in px = 300 + 150 = 450px (1.5 in from top -> 12 - 1.5 = 10.5 in = 756 pt from bottom)
    bbox_px = BBoxPx(left=300, top=300, width=600, height=150)
    bbox_pt, baseline_y = source_bbox_to_pdf_points(
        bbox_px,
        source_width_px=src_w,
        source_height_px=src_h,
        page_width_pt=w_pt,
        page_height_pt=h_pt,
    )

    assert bbox_pt.x == pytest.approx(72.0, abs=1e-3)
    assert bbox_pt.width == pytest.approx(144.0, abs=1e-3)
    assert bbox_pt.height == pytest.approx(36.0, abs=1e-3)
    assert bbox_pt.y == pytest.approx(756.0, abs=1e-3)
    assert bbox_pt.y < baseline_y < bbox_pt.top


def test_working_to_source_scaling_inversion() -> None:
    # 600 DPI archival scan (4800 x 6400 px) downsampled to 300 DPI working image (2400 x 3200 px)
    transform = WorkingToSourceTransform(
        source_width_px=4800,
        source_height_px=6400,
        working_width_px=2400,
        working_height_px=3200,
        scale_x=0.5,
        scale_y=0.5,
        deskew_angle_deg=0.0,
        cardinal_rotation_deg=0,
    )
    work_box = BBoxPx(left=200, top=400, width=300, height=50)
    src_box = working_bbox_to_source_bbox(work_box, transform)

    assert src_box.left == pytest.approx(400.0)
    assert src_box.top == pytest.approx(800.0)
    assert src_box.width == pytest.approx(600.0)
    assert src_box.height == pytest.approx(100.0)


def test_inverse_deskew_point_round_trip() -> None:
    cx, cy = 1000.0, 1500.0
    orig_x, orig_y = 420.0, 680.0
    angle_deg = 2.75
    theta = math.radians(angle_deg)

    # Forward rotation matching cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    dx, dy = orig_x - cx, orig_y - cy
    rot_x = cx + dx * math.cos(theta) + dy * math.sin(theta)
    rot_y = cy - dx * math.sin(theta) + dy * math.cos(theta)

    rec_x, rec_y = map_point_inverse_deskew(rot_x, rot_y, cx, cy, angle_deg)
    assert rec_x == pytest.approx(orig_x, abs=1e-4)
    assert rec_y == pytest.approx(orig_y, abs=1e-4)


def test_inverse_cardinal_rotation_all_angles() -> None:
    orig_w, orig_h = 1000.0, 2000.0
    px, py = 250.0, 600.0

    # 90 deg CW: new image has width=2000, height=1000; point becomes (H - y, x) = (1400, 250)
    rx90, ry90 = map_point_inverse_cardinal(1400.0, 250.0, orig_h, orig_w, 90)
    assert (rx90, ry90) == pytest.approx((px, py))

    # 180 deg CW: new image has width=1000, height=2000; point becomes (W - x, H - y) = (750, 1400)
    rx180, ry180 = map_point_inverse_cardinal(750.0, 1400.0, orig_w, orig_h, 180)
    assert (rx180, ry180) == pytest.approx((px, py))

    # 270 deg CW: new image has width=2000, height=1000; point becomes (y, W - x) = (600, 750)
    rx270, ry270 = map_point_inverse_cardinal(600.0, 750.0, orig_h, orig_w, 270)
    assert (rx270, ry270) == pytest.approx((px, py))


def test_bbox_iou_pt_computation() -> None:
    box1 = BBoxPt(x=100.0, y=200.0, width=80.0, height=20.0)
    box2 = BBoxPt(x=100.0, y=200.0, width=80.0, height=20.0)
    assert bbox_iou_pt(box1, box2) == pytest.approx(1.0)

    disjoint = BBoxPt(x=300.0, y=500.0, width=50.0, height=20.0)
    assert bbox_iou_pt(box1, disjoint) == 0.0
