"""Background removal (rembg isnet-general-use), Alpha/RGB quality scoring, and packaging."""

from __future__ import annotations

import os
import sys
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage

from assets_manifest import AssetSpec

_REMBG_SESSION: Any = None


def ensure_onnx_dll_path() -> None:
    """Ensure local venv Scripts DLLs (MSVC runtime) are discoverable on Windows."""
    if os.name != "nt" or not hasattr(os, "add_dll_directory"):
        return
    scripts_dir = Path(sys.prefix) / "Scripts"
    if scripts_dir.is_dir():
        os.add_dll_directory(str(scripts_dir))


def get_rembg_session() -> Any:
    """Lazily initialize and reuse a CPU rembg session for isnet-general-use."""
    global _REMBG_SESSION
    if _REMBG_SESSION is None:
        ensure_onnx_dll_path()
        import rembg

        _REMBG_SESSION = rembg.new_session(
            "isnet-general-use",
            providers=["CPUExecutionProvider"],
        )
    return _REMBG_SESSION


@dataclass(frozen=True)
class AlphaAnalysis:
    """Detailed Alpha/RGB inspection metrics and quality score for one variant."""

    width: int
    height: int
    fg_pixels: int
    fg_area_ratio: float
    main_area_ratio: float
    num_fragments: int
    significant_extra_blobs: int
    micro_speck_count: int
    debris_ratio: float
    touch_top: int
    touch_bottom: int
    touch_left: int
    touch_right: int
    total_edge_touch_px: int
    max_edge_touch_px: int
    min_margin_px: int
    min_margin_ratio: float
    bbox: tuple[int, int, int, int]
    bbox_w_ratio: float
    bbox_h_ratio: float
    cx_bbox_err: float
    cy_bbox_err: float
    cx_mass_err: float
    cy_mass_err: float
    halo_ratio: float
    center_score: float
    cleanliness_score: float
    framing_score: float
    proportion_score: float
    total_score: float
    passed: bool
    reject_reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        """Convert metrics to JSON-serializable dict."""
        data = asdict(self)
        data["reject_reasons"] = list(self.reject_reasons)
        data["bbox"] = list(self.bbox)
        return data


def _clean_enclosed_corner_bg_pockets(
    raw_rgb: Image.Image, cutout_rgba: Image.Image
) -> Image.Image:
    """Remove interior pockets and 1-2px outer fringes matching corner background color."""
    rgb = np.asarray(raw_rgb.convert("RGB"), dtype=np.int16)
    rgba = np.asarray(cutout_rgba.convert("RGBA"), dtype=np.uint8).copy()
    h, w, _ = rgba.shape
    patch = min(16, h // 8, w // 8)
    if patch < 4:
        return cutout_rgba

    alpha = rgba[:, :, 3]
    corners = [
        (slice(0, patch), slice(0, patch)),
        (slice(0, patch), slice(w - patch, w)),
        (slice(h - patch, h), slice(0, patch)),
        (slice(h - patch, h), slice(w - patch, w)),
    ]
    samples = []
    for ys, xs in corners:
        bg_mask = alpha[ys, xs] < 30
        if np.count_nonzero(bg_mask) >= (patch * patch) // 2:
            samples.append(rgb[ys, xs][bg_mask])

    if len(samples) < 3:
        return cutout_rgba

    all_bg = np.concatenate(samples, axis=0)
    if float(np.std(all_bg, axis=0).max()) > 20.0:
        return cutout_rgba

    bg_color = np.median(all_bg, axis=0)
    chroma = float(bg_color.max() - bg_color.min())
    luma = float(bg_color.mean())
    # Never strip near-white (white robes) or near-black (black ink) interiors
    if chroma <= 18.0 and (luma >= 225.0 or luma <= 30.0):
        return cutout_rgba

    dist = np.linalg.norm(rgb - bg_color[None, None, :], axis=2)
    pocket_candidates = (alpha > 20) & (dist < 30.0)
    if np.any(pocket_candidates):
        labels, n_comp = ndimage.label(pocket_candidates)
        if n_comp > 0:
            counts = np.bincount(labels.ravel())
            max_pocket_px = int(0.15 * h * w)
            valid_ids = [
                cid
                for cid in np.where(counts >= 100)[0]
                if cid != 0 and counts[cid] <= max_pocket_px
            ]
            if valid_ids:
                alpha[np.isin(labels, valid_ids)] = 0

    # Trim 1-2px outer boundary fringes matching background color or bottom grey/brown halo
    fg = alpha > 0
    if np.any(fg):
        ys, xs = np.nonzero(fg)
        bbox_ymin, bbox_ymax = int(ys.min()), int(ys.max())
        bbox_h = max(1, bbox_ymax - bbox_ymin + 1)

        transparent = alpha == 0
        near_empty_2px = ndimage.binary_dilation(
            transparent, structure=np.ones((3, 3), dtype=bool), iterations=2
        ) & fg
        near_empty_1px = ndimage.binary_dilation(
            transparent, structure=np.ones((3, 3), dtype=bool), iterations=1
        ) & fg

        raw_luma = rgb.mean(axis=2)
        bg_fringe = near_empty_2px & (dist < 38.0) & (raw_luma > 45.0)
        alpha[bg_fringe] = 0

        bottom_band = np.zeros((h, w), dtype=bool)
        bottom_start = max(0, bbox_ymax - int(0.15 * bbox_h))
        bottom_band[bottom_start:, :] = True
        bottom_fringe = (
            near_empty_1px
            & bottom_band
            & ((alpha < 210) | (dist < 52.0))
            & (raw_luma > 45.0)
        )
        alpha[bottom_fringe] = 0

    rgba[:, :, 3] = alpha
    return Image.fromarray(rgba, mode="RGBA")


def _restore_enclosed_white_robe_details(
    raw_rgb: Image.Image, cutout_rgba: Image.Image, spec: AssetSpec
) -> Image.Image:
    """Restore enclosed pure-white robe details mistakenly eroded by rembg."""
    if spec.mode not in ("sprite_bottom", "icon_center"):
        return cutout_rgba

    rgb_u8 = np.asarray(raw_rgb.convert("RGB"), dtype=np.uint8)
    rgb = rgb_u8.astype(np.float32)
    rgba = np.asarray(cutout_rgba.convert("RGBA"), dtype=np.uint8).copy()
    alpha = rgba[:, :, 3]
    h, w = alpha.shape

    opaque = alpha >= 200
    if not np.any(opaque):
        return cutout_rgba

    ys_fg, xs_fg = np.nonzero(opaque)
    ymin, ymax = int(ys_fg.min()), int(ys_fg.max())
    bbox_h = max(1, ymax - ymin + 1)

    raw_luma = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
    raw_chroma = rgb.max(axis=2) - rgb.min(axis=2)
    is_raw_white = (raw_luma >= 195.0) & (raw_chroma <= 35.0)

    corners = [
        raw_luma[:16, :16],
        raw_luma[:16, -16:],
        raw_luma[-16:, :16],
        raw_luma[-16:, -16:],
    ]
    corner_white = (
        float(np.median(np.concatenate([c.ravel() for c in corners]))) >= 220.0
    )

    white_holes = is_raw_white & (alpha < 235)
    labels, n_comp = ndimage.label(white_holes)
    if n_comp == 0:
        return cutout_rgba

    border_mask = np.zeros((h, w), dtype=bool)
    border_mask[:4, :] = True
    border_mask[-4:, :] = True
    border_mask[:, :4] = True
    border_mask[:, -4:] = True
    if corner_white:
        ext_bg, _ = ndimage.label(alpha < 25)
        ext_ids = set(np.unique(ext_bg[border_mask]).tolist()) - {0}
        if ext_ids:
            ext_mask = np.isin(ext_bg, list(ext_ids))
            touching_ext = ndimage.binary_dilation(ext_mask, iterations=2)
            border_mask = border_mask | touching_ext

    excluded_ids = set(np.unique(labels[border_mask]).tolist())
    counts = np.bincount(labels.ravel())
    max_area = int(0.12 * h * w)
    slices = ndimage.find_objects(labels)
    restore_ids: list[int] = []
    upper_limit_y = ymax - int(0.10 * bbox_h) if corner_white else h

    for cid in range(1, n_comp + 1):
        if cid in excluded_ids:
            continue
        if not (24 <= counts[cid] <= max_area):
            continue
        slc = slices[cid - 1]
        if slc is None:
            continue
        if corner_white and (slc[0].stop - 1) >= upper_limit_y:
            continue
        restore_ids.append(cid)

    if restore_ids:
        mask = np.isin(labels, restore_ids)
        rgba[mask, :3] = rgb_u8[mask]
        rgba[mask, 3] = 255
        return Image.fromarray(rgba, mode="RGBA")
    return cutout_rgba


def _clear_enclosed_center_white(rgba: Image.Image, white_thresh: int = 235) -> Image.Image:
    """For hollow UI borders/bar frames, clear pure-white interior if rembg kept it opaque."""
    arr = np.asarray(rgba, dtype=np.uint8).copy()
    h, w, _ = arr.shape
    cy, cx = h // 2, w // 2
    r, g, b, a = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2], arr[:, :, 3]
    is_white_opaque = (a > 180) & (r >= white_thresh) & (g >= white_thresh) & (b >= white_thresh)
    if not is_white_opaque[cy, cx]:
        return rgba

    visited = np.zeros((h, w), dtype=bool)
    queue: deque[tuple[int, int]] = deque([(cy, cx)])
    visited[cy, cx] = True
    touches_outer = False
    pixels: list[tuple[int, int]] = []

    while queue:
        y, x = queue.popleft()
        pixels.append((y, x))
        if y == 0 or y == h - 1 or x == 0 or x == w - 1:
            touches_outer = True
            break
        for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
            if not visited[ny, nx] and is_white_opaque[ny, nx]:
                visited[ny, nx] = True
                queue.append((ny, nx))

    if not touches_outer and len(pixels) > int(0.05 * h * w):
        ys, xs = zip(*pixels)
        arr[ys, xs, 3] = 0
        return Image.fromarray(arr, mode="RGBA")
    return rgba


def remove_background(raw_image: Image.Image, spec: AssetSpec) -> Image.Image:
    """Run rembg (isnet-general-use) if spec.remove_bg is True, else return RGBA."""
    rgb = raw_image.convert("RGB")
    if not spec.remove_bg:
        return rgb.convert("RGBA")

    session = get_rembg_session()
    import rembg

    cutout = rembg.remove(rgb, session=session).convert("RGBA")
    cutout = _clean_enclosed_corner_bg_pockets(rgb, cutout)
    cutout = _restore_enclosed_white_robe_details(rgb, cutout, spec)
    if spec.mode == "ui_border" or spec.id == "UI_Bar_Frame":
        cutout = _clear_enclosed_center_white(cutout)
    return cutout


def _empty_analysis(w: int, h: int, reason: str = "empty_foreground (0% area)") -> AlphaAnalysis:
    return AlphaAnalysis(
        width=w,
        height=h,
        fg_pixels=0,
        fg_area_ratio=0.0,
        main_area_ratio=0.0,
        num_fragments=0,
        significant_extra_blobs=0,
        micro_speck_count=0,
        debris_ratio=1.0,
        touch_top=0,
        touch_bottom=0,
        touch_left=0,
        touch_right=0,
        total_edge_touch_px=0,
        max_edge_touch_px=0,
        min_margin_px=0,
        min_margin_ratio=0.0,
        bbox=(0, 0, 0, 0),
        bbox_w_ratio=0.0,
        bbox_h_ratio=0.0,
        cx_bbox_err=0.5,
        cy_bbox_err=0.5,
        cx_mass_err=0.5,
        cy_mass_err=0.5,
        halo_ratio=1.0,
        center_score=0.0,
        cleanliness_score=0.0,
        framing_score=0.0,
        proportion_score=0.0,
        total_score=0.0,
        passed=False,
        reject_reasons=(reason,),
    )


def analyze_opaque(rgba_image: Image.Image, spec: AssetSpec) -> AlphaAnalysis:
    """Score non-transparent assets (ArenaFloor, MainMenu_BG, App_Icon, Bar_Fill_White)."""
    rgb = np.asarray(rgba_image.convert("RGB"), dtype=np.float32)
    h, w, _ = rgb.shape
    total_px = float(h * w)
    luma = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]

    if spec.mode == "ui_bar_fill":
        bar_mask = luma >= 185.0
        fg_pixels = int(np.count_nonzero(bar_mask))
        if fg_pixels == 0:
            return _empty_analysis(w, h, "no_white_bar_detected")

        labels, n_comp = ndimage.label(bar_mask, structure=np.ones((3, 3), dtype=np.int32))
        counts = np.bincount(labels.ravel())[1:]
        sorted_areas = np.sort(counts)[::-1]
        main_area = int(sorted_areas[0])
        fg_area_ratio = fg_pixels / total_px
        main_area_ratio = main_area / total_px
        debris_ratio = max(0, fg_pixels - main_area) / float(fg_pixels)

        main_id = int(np.argmax(counts)) + 1
        ys, xs = np.nonzero(labels == main_id)
        ymin, ymax = int(ys.min()), int(ys.max())
        xmin, xmax = int(xs.min()), int(xs.max())
        bbox_w = xmax - xmin + 1
        bbox_h = ymax - ymin + 1
        rect_fill = main_area / float(max(1, bbox_w * bbox_h))

        cx_bbox_err = abs(((xmin + xmax) * 0.5) / float(w - 1) - 0.5)
        cy_bbox_err = abs(((ymin + ymax) * 0.5) / float(h - 1) - 0.5)
        min_margin_px = min(ymin, h - 1 - ymax, xmin, w - 1 - xmax)

        reject_reasons: list[str] = []
        if main_area_ratio < 0.08:
            reject_reasons.append(f"bar_too_small ({main_area_ratio * 100:.1f}% < 8%)")
        if bbox_w <= bbox_h * 1.8:
            reject_reasons.append(f"not_horizontal ({bbox_w}x{bbox_h})")

        center_score = 35.0 * max(0.0, 1.0 - (cx_bbox_err + cy_bbox_err) / 0.25)
        cleanliness_score = 35.0 * (0.6 * rect_fill + 0.4 * max(0.0, 1.0 - debris_ratio * 2.0))
        framing_score = 20.0 * min(1.0, (bbox_w / float(w)) / 0.65)
        proportion_score = 10.0 * min(1.0, rect_fill / 0.85)
        total_score = center_score + cleanliness_score + framing_score + proportion_score

        return AlphaAnalysis(
            width=w,
            height=h,
            fg_pixels=fg_pixels,
            fg_area_ratio=round(fg_area_ratio, 4),
            main_area_ratio=round(main_area_ratio, 4),
            num_fragments=int(np.count_nonzero(sorted_areas >= 32)),
            significant_extra_blobs=0,
            micro_speck_count=0,
            debris_ratio=round(debris_ratio, 4),
            touch_top=0,
            touch_bottom=0,
            touch_left=0,
            touch_right=0,
            total_edge_touch_px=0,
            max_edge_touch_px=0,
            min_margin_px=int(min_margin_px),
            min_margin_ratio=round(min_margin_px / float(min(h, w)), 4),
            bbox=(xmin, ymin, xmax, ymax),
            bbox_w_ratio=round(bbox_w / float(w), 4),
            bbox_h_ratio=round(bbox_h / float(h), 4),
            cx_bbox_err=round(cx_bbox_err, 4),
            cy_bbox_err=round(cy_bbox_err, 4),
            cx_mass_err=round(cx_bbox_err, 4),
            cy_mass_err=round(cy_bbox_err, 4),
            halo_ratio=0.0,
            center_score=round(center_score, 2),
            cleanliness_score=round(cleanliness_score, 2),
            framing_score=round(framing_score, 2),
            proportion_score=round(proportion_score, 2),
            total_score=round(total_score, 2),
            passed=len(reject_reasons) == 0,
            reject_reasons=tuple(reject_reasons),
        )

    # Full-frame scenes/floors/app icons
    luma_std = float(np.std(luma))
    gx = np.abs(np.diff(luma, axis=1, prepend=luma[:, :1]))
    gy = np.abs(np.diff(luma, axis=0, prepend=luma[:1, :]))
    grad = gx + gy

    rim = max(8, int(min(h, w) * 0.03))
    border_mask = np.zeros((h, w), dtype=bool)
    border_mask[:rim, :] = True
    border_mask[-rim:, :] = True
    border_mask[:, :rim] = True
    border_mask[:, -rim:] = True
    white_border_ratio = float(np.count_nonzero((luma > 238.0) & border_mask)) / float(
        np.count_nonzero(border_mask)
    )

    cy0, cy1 = int(h * 0.2), int(h * 0.8)
    cx0, cx1 = int(w * 0.2), int(w * 0.8)
    center_grad = float(np.mean(grad[cy0:cy1, cx0:cx1]))
    outer_grad = float(np.mean(grad[border_mask])) + 1e-5
    focus_ratio = center_grad / outer_grad

    # Gradient center-of-mass balance
    ys, xs = np.indices((h, w))
    g_sum = float(grad.sum()) + 1e-5
    cx_mass_err = abs(float((xs * grad).sum() / (g_sum * (w - 1))) - 0.5)
    cy_mass_err = abs(float((ys * grad).sum() / (g_sum * (h - 1))) - 0.5)

    reject_reasons_op: list[str] = []
    if luma_std < 8.0:
        reject_reasons_op.append(f"flat_low_contrast (std={luma_std:.1f})")
    if white_border_ratio > 0.12:
        reject_reasons_op.append(f"white_border_artifact ({white_border_ratio * 100:.1f}%)")

    center_score = 35.0 * max(0.0, 1.0 - (0.6 * cx_mass_err + 0.4 * cy_mass_err) / 0.20)
    cleanliness_score = 35.0 * max(0.0, 1.0 - white_border_ratio / 0.12)
    framing_score = 20.0 * min(1.0, focus_ratio / 1.5) if spec.mode == "opaque_center" else 18.0
    proportion_score = 10.0 * min(1.0, luma_std / 35.0)
    total_score = center_score + cleanliness_score + framing_score + proportion_score

    return AlphaAnalysis(
        width=w,
        height=h,
        fg_pixels=int(total_px),
        fg_area_ratio=1.0,
        main_area_ratio=1.0,
        num_fragments=1,
        significant_extra_blobs=0,
        micro_speck_count=0,
        debris_ratio=0.0,
        touch_top=0,
        touch_bottom=0,
        touch_left=0,
        touch_right=0,
        total_edge_touch_px=0,
        max_edge_touch_px=0,
        min_margin_px=0,
        min_margin_ratio=0.0,
        bbox=(0, 0, w - 1, h - 1),
        bbox_w_ratio=1.0,
        bbox_h_ratio=1.0,
        cx_bbox_err=0.0,
        cy_bbox_err=0.0,
        cx_mass_err=round(cx_mass_err, 4),
        cy_mass_err=round(cy_mass_err, 4),
        halo_ratio=0.0,
        center_score=round(center_score, 2),
        cleanliness_score=round(cleanliness_score, 2),
        framing_score=round(framing_score, 2),
        proportion_score=round(proportion_score, 2),
        total_score=round(total_score, 2),
        passed=len(reject_reasons_op) == 0,
        reject_reasons=tuple(reject_reasons_op),
    )


def analyze_alpha(rgba_image: Image.Image, spec: AssetSpec) -> AlphaAnalysis:
    """Analyze the Alpha channel (or RGB for opaque assets) to filter defects & score."""
    if not spec.remove_bg:
        return analyze_opaque(rgba_image, spec)

    alpha = np.asarray(rgba_image.getchannel("A"), dtype=np.uint8)
    h, w = alpha.shape
    total_px = float(h * w)

    fg_mask = alpha > 20
    fg_pixels = int(np.count_nonzero(fg_mask))
    if fg_pixels == 0:
        return _empty_analysis(w, h)

    fg_area_ratio = fg_pixels / total_px

    # 8-connected component analysis on foreground mask
    structure = np.ones((3, 3), dtype=np.int32)
    labels, raw_comp_count = ndimage.label(fg_mask, structure=structure)
    if raw_comp_count == 0:
        return _empty_analysis(w, h)

    comp_counts = np.bincount(labels.ravel())[1:]
    sorted_areas = np.sort(comp_counts)[::-1]

    min_frag_px = max(16, int(0.000035 * total_px))
    sig_blob_px = max(120, int(0.0005 * total_px))

    fragment_areas = sorted_areas[sorted_areas >= min_frag_px]
    micro_speck_count = int(np.count_nonzero((sorted_areas >= 4) & (sorted_areas < min_frag_px)))
    num_fragments = int(len(fragment_areas))

    main_area = int(sorted_areas[0])
    main_area_ratio = main_area / total_px
    primary_area = int(sorted_areas[: max(1, spec.primary_parts)].sum())
    debris_pixels = max(0, fg_pixels - primary_area)
    debris_ratio = debris_pixels / float(fg_pixels)
    significant_extra_blobs = int(
        np.count_nonzero(sorted_areas[max(1, spec.primary_parts) :] >= sig_blob_px)
    )

    # Edge-touch / clipping detection along outermost border pixels
    edge_band = 1 if spec.mode.startswith("ui_") else 2
    solid_edge_mask = alpha >= 50
    touch_top = int(np.count_nonzero(solid_edge_mask[:edge_band, :]))
    touch_bottom = int(np.count_nonzero(solid_edge_mask[h - edge_band :, :]))
    touch_left = int(np.count_nonzero(solid_edge_mask[:, :edge_band]))
    touch_right = int(np.count_nonzero(solid_edge_mask[:, w - edge_band :]))
    total_edge_touch_px = touch_top + touch_bottom + touch_left + touch_right
    max_edge_touch_px = max(touch_top, touch_bottom, touch_left, touch_right)

    # Foreground bounding box & margin
    ys, xs = np.nonzero(fg_mask)
    ymin, ymax = int(ys.min()), int(ys.max())
    xmin, xmax = int(xs.min()), int(xs.max())
    min_margin_px = min(ymin, h - 1 - ymax, xmin, w - 1 - xmax)
    min_margin_ratio = min_margin_px / float(min(h, w))

    bbox_w_ratio = (xmax - xmin + 1) / float(w)
    bbox_h_ratio = (ymax - ymin + 1) / float(h)

    # Center balance: bounding box center & alpha-weighted center of mass
    cx_bbox = ((xmin + xmax) * 0.5) / float(w - 1)
    cy_bbox = ((ymin + ymax) * 0.5) / float(h - 1)
    cx_bbox_err = abs(cx_bbox - 0.5)
    cy_bbox_err = abs(cy_bbox - 0.5)

    weights = alpha[ys, xs].astype(np.float64)
    w_sum = float(weights.sum())
    cx_mass = float((xs * weights).sum() / (w_sum * (w - 1))) if w_sum > 0 else cx_bbox
    cy_mass = float((ys * weights).sum() / (w_sum * (h - 1))) if w_sum > 0 else cy_bbox
    cx_mass_err = abs(cx_mass - 0.5)
    cy_mass_err = abs(cy_mass - 0.5)

    # Semi-transparent halo / fuzzy-edge ratio
    semi_mask = (alpha > 10) & (alpha < 220)
    halo_ratio = int(np.count_nonzero(semi_mask)) / float(fg_pixels)

    # --- Rejection Gates ---
    reject_reasons: list[str] = []
    if fg_area_ratio < spec.min_area_ratio:
        reject_reasons.append(
            f"area_too_small ({fg_area_ratio * 100:.1f}% < {spec.min_area_ratio * 100:.0f}%)"
        )
    if fg_area_ratio > spec.max_area_ratio:
        reject_reasons.append(
            f"area_too_large_or_bg_stuck ({fg_area_ratio * 100:.1f}% > {spec.max_area_ratio * 100:.0f}%)"
        )

    edge_limit_single = 25 if spec.mode.startswith("ui_") else 8
    edge_limit_total = 40 if spec.mode.startswith("ui_") else 12
    if max_edge_touch_px >= edge_limit_single or total_edge_touch_px >= edge_limit_total:
        sides = []
        if touch_top > 0:
            sides.append(f"top={touch_top}px")
        if touch_bottom > 0:
            sides.append(f"bottom={touch_bottom}px")
        if touch_left > 0:
            sides.append(f"left={touch_left}px")
        if touch_right > 0:
            sides.append(f"right={touch_right}px")
        reject_reasons.append(f"clipped_at_edge ({', '.join(sides)})")

    if num_fragments > spec.max_fragments:
        reject_reasons.append(
            f"too_many_fragments ({num_fragments} > {spec.max_fragments})"
        )
    if debris_ratio > spec.max_debris_ratio:
        reject_reasons.append(
            f"excessive_debris ({debris_ratio * 100:.1f}% > {spec.max_debris_ratio * 100:.0f}%)"
        )

    # --- Quality Scoring (0 .. 100) ---
    horiz_err = 0.6 * cx_mass_err + 0.4 * cx_bbox_err
    vert_err = 0.6 * cy_mass_err + 0.4 * cy_bbox_err
    combined_center_err = 0.65 * horiz_err + 0.35 * vert_err
    center_score = 35.0 * max(0.0, 1.0 - combined_center_err / 0.22)

    cohesion_pts = 18.0 * max(0.0, 1.0 - debris_ratio / max(0.05, spec.max_debris_ratio))
    extra_fragments = max(0, num_fragments - max(1, spec.primary_parts))
    allowed_extra = max(1, spec.max_fragments - max(1, spec.primary_parts))
    fragment_pts = max(
        0.0,
        10.0 * (1.0 - extra_fragments / float(allowed_extra))
        - min(3.0, micro_speck_count * 0.25),
    )
    edge_crisp_pts = 7.0 * max(0.0, 1.0 - max(0.0, halo_ratio - 0.06) / 0.30)
    cleanliness_score = cohesion_pts + fragment_pts + edge_crisp_pts

    margin_target = 0.015 if spec.mode.startswith("ui_") else 0.045
    margin_pts = 10.0 * min(1.0, min_margin_ratio / margin_target)

    ideal_low, ideal_high = (0.20, 0.72) if spec.mode.startswith("ui_") else (0.18, 0.52)
    if ideal_low <= fg_area_ratio <= ideal_high:
        area_pts = 10.0
    elif fg_area_ratio < ideal_low:
        denom = max(0.05, ideal_low - spec.min_area_ratio * 0.5)
        area_pts = 10.0 * max(0.0, 1.0 - (ideal_low - fg_area_ratio) / denom)
    else:
        denom = max(0.05, spec.max_area_ratio - ideal_high)
        area_pts = 10.0 * max(0.0, 1.0 - (fg_area_ratio - ideal_high) / denom)
    framing_score = margin_pts + area_pts

    if spec.mode == "sprite_bottom":
        proportion_score = 10.0 if bbox_h_ratio >= 0.55 else 10.0 * max(0.0, bbox_h_ratio / 0.55)
    elif spec.mode == "icon_center":
        proportion_score = 10.0 * min(1.0, max(bbox_w_ratio, bbox_h_ratio) / 0.65)
    else:
        proportion_score = 10.0 * min(1.0, (bbox_w_ratio * bbox_h_ratio) / 0.45)

    total_score = center_score + cleanliness_score + framing_score + proportion_score

    return AlphaAnalysis(
        width=w,
        height=h,
        fg_pixels=fg_pixels,
        fg_area_ratio=round(fg_area_ratio, 4),
        main_area_ratio=round(main_area_ratio, 4),
        num_fragments=num_fragments,
        significant_extra_blobs=significant_extra_blobs,
        micro_speck_count=micro_speck_count,
        debris_ratio=round(debris_ratio, 4),
        touch_top=touch_top,
        touch_bottom=touch_bottom,
        touch_left=touch_left,
        touch_right=touch_right,
        total_edge_touch_px=total_edge_touch_px,
        max_edge_touch_px=max_edge_touch_px,
        min_margin_px=int(min_margin_px),
        min_margin_ratio=round(min_margin_ratio, 4),
        bbox=(xmin, ymin, xmax, ymax),
        bbox_w_ratio=round(bbox_w_ratio, 4),
        bbox_h_ratio=round(bbox_h_ratio, 4),
        cx_bbox_err=round(cx_bbox_err, 4),
        cy_bbox_err=round(cy_bbox_err, 4),
        cx_mass_err=round(cx_mass_err, 4),
        cy_mass_err=round(cy_mass_err, 4),
        halo_ratio=round(halo_ratio, 4),
        center_score=round(center_score, 2),
        cleanliness_score=round(cleanliness_score, 2),
        framing_score=round(framing_score, 2),
        proportion_score=round(proportion_score, 2),
        total_score=round(total_score, 2),
        passed=len(reject_reasons) == 0,
        reject_reasons=tuple(reject_reasons),
    )


def _bleed_rgb_into_transparent(arr: np.ndarray, iterations: int = 4) -> np.ndarray:
    """Dilate RGB values from opaque edges into alpha==0 pixels to prevent white fringes."""
    out = arr.copy()
    alpha = out[:, :, 3]
    valid = alpha > 0
    rgb = out[:, :, :3].astype(np.int32)

    for _ in range(iterations):
        if np.all(valid):
            break
        dilated = ndimage.binary_dilation(valid, structure=np.ones((3, 3), dtype=bool))
        frontier = dilated & (~valid)
        if not np.any(frontier):
            break

        sum_rgb = np.zeros_like(rgb, dtype=np.int32)
        count = np.zeros(valid.shape, dtype=np.int32)
        h, w = valid.shape
        valid_i = valid.astype(np.int32)

        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                ys_dst = slice(max(0, dy), h + min(0, dy))
                xs_dst = slice(max(0, dx), w + min(0, dx))
                ys_src = slice(max(0, -dy), h + min(0, -dy))
                xs_src = slice(max(0, -dx), w + min(0, -dx))

                m = valid_i[ys_src, xs_src]
                count[ys_dst, xs_dst] += m
                sum_rgb[ys_dst, xs_dst] += rgb[ys_src, xs_src] * m[:, :, None]

        fill_mask = frontier & (count > 0)
        rgb[fill_mask] = sum_rgb[fill_mask] // count[fill_mask][:, None]
        valid = valid | fill_mask

    out[:, :, :3] = np.clip(rgb, 0, 255).astype(np.uint8)
    return out


def _package_opaque(image: Image.Image, spec: AssetSpec) -> Image.Image:
    """Crop/fit non-transparent assets cleanly and resize with LANCZOS."""
    rgb = image.convert("RGB")
    target_w, target_h = spec.target_size

    if spec.mode == "ui_bar_fill":
        arr = np.asarray(rgb, dtype=np.float32)
        luma = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]
        bar_mask = luma >= 185.0
        labels, n_comp = ndimage.label(bar_mask)
        if n_comp > 0:
            counts = np.bincount(labels.ravel())[1:]
            main_id = int(np.argmax(counts)) + 1
            ys, xs = np.nonzero(labels == main_id)
            ymin, ymax = int(ys.min()), int(ys.max())
            xmin, xmax = int(xs.min()), int(xs.max())
            # Inset slightly inside bar bounds so dark background rim is excluded
            inset_y = min(6, max(0, (ymax - ymin) // 8))
            inset_x = min(6, max(0, (xmax - xmin) // 24))
            crop = rgb.crop((xmin + inset_x, ymin + inset_y, xmax + 1 - inset_x, ymax + 1 - inset_y))
            return crop.resize((target_w, target_h), Image.Resampling.LANCZOS)
        return rgb.resize((target_w, target_h), Image.Resampling.LANCZOS)

    # Aspect-preserving center crop for scenes/floors/app icons
    src_w, src_h = rgb.size
    target_ratio = target_w / float(target_h)
    src_ratio = src_w / float(src_h)
    if abs(src_ratio - target_ratio) > 0.005:
        if src_ratio > target_ratio:
            new_w = int(round(src_h * target_ratio))
            x0 = (src_w - new_w) // 2
            rgb = rgb.crop((x0, 0, x0 + new_w, src_h))
        else:
            new_h = int(round(src_w / target_ratio))
            y0 = (src_h - new_h) // 2
            rgb = rgb.crop((0, y0, src_w, y0 + new_h))

    return rgb.resize((target_w, target_h), Image.Resampling.LANCZOS)


def package_sprite(rgba_image: Image.Image, spec: AssetSpec) -> Image.Image:
    """Clean micro-specks, bleed edge colors, normalize alignment, and resize with LANCZOS."""
    if not spec.remove_bg:
        return _package_opaque(rgba_image, spec)

    arr = np.asarray(rgba_image.convert("RGBA"), dtype=np.uint8).copy()

    alpha = arr[:, :, 3]
    alpha[alpha < 12] = 0
    fg = alpha > 0
    labels, n_comp = ndimage.label(fg, structure=np.ones((3, 3), dtype=np.int32))
    if n_comp > 1:
        counts = np.bincount(labels.ravel())
        tiny_ids = np.where(counts < 25)[0]
        tiny_ids = tiny_ids[tiny_ids != 0]
        if len(tiny_ids) > 0:
            alpha[np.isin(labels, tiny_ids)] = 0

    arr[:, :, 3] = alpha
    arr = _bleed_rgb_into_transparent(arr, iterations=4)
    cleaned = Image.fromarray(arr, mode="RGBA")

    target_w, target_h = spec.target_size
    ys, xs = np.nonzero(alpha > 0)
    if len(ys) == 0:
        return cleaned.resize((target_w, target_h), Image.Resampling.LANCZOS)

    ymin, ymax = int(ys.min()), int(ys.max())
    xmin, xmax = int(xs.min()), int(xs.max())
    cropped = cleaned.crop((xmin, ymin, xmax + 1, ymax + 1))

    if spec.mode == "ui_bar_frame":
        pad_x = max(2, int(round(target_w * spec.padding_ratio)))
        pad_y = max(2, int(round(target_h * spec.padding_ratio)))
        resized = cropped.resize(
            (max(1, target_w - 2 * pad_x), max(1, target_h - 2 * pad_y)),
            Image.Resampling.LANCZOS,
        )
        canvas = Image.new("RGBA", (target_w, target_h), (0, 0, 0, 0))
        canvas.paste(resized, (pad_x, pad_y), resized)
        return canvas

    crop_w, crop_h = cropped.size
    pad_x = max(2, int(round(target_w * spec.padding_ratio)))
    pad_y = max(2, int(round(target_h * spec.padding_ratio)))
    avail_w = max(1, target_w - 2 * pad_x)
    avail_h = max(1, target_h - 2 * pad_y)

    scale = min(avail_w / float(crop_w), avail_h / float(crop_h))
    new_w = max(1, int(round(crop_w * scale)))
    new_h = max(1, int(round(crop_h * scale)))
    resized = cropped.resize((new_w, new_h), Image.Resampling.LANCZOS)

    canvas = Image.new("RGBA", (target_w, target_h), (0, 0, 0, 0))
    paste_x = (target_w - new_w) // 2
    if spec.mode == "sprite_bottom":
        paste_y = target_h - pad_y - new_h
    else:
        paste_y = (target_h - new_h) // 2

    canvas.paste(resized, (paste_x, paste_y), resized)
    return canvas


def _make_checkerboard(w: int, h: int, tile: int = 16) -> Image.Image:
    ys, xs = np.indices((h, w))
    grid = ((ys // tile) + (xs // tile)) % 2
    bg = np.where(grid[:, :, None] == 0, 34, 46).astype(np.uint8)
    rgb = np.repeat(bg, 3, axis=2)
    return Image.fromarray(rgb, mode="RGB")


def build_contact_sheet(
    candidates: list[dict[str, Any]],
    out_path: Path,
    cols: int = 4,
    thumb_size: int = 320,
) -> None:
    """Create a labeled checkerboard contact sheet of all candidates."""
    if not candidates:
        return
    rows = (len(candidates) + cols - 1) // cols
    footer_h = 68
    pad = 12
    cell_w = thumb_size + pad
    cell_h = thumb_size + footer_h + pad
    sheet = Image.new("RGB", (cols * cell_w + pad, rows * cell_h + pad), (18, 20, 26))
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    checker = _make_checkerboard(thumb_size, thumb_size)

    for idx, item in enumerate(candidates):
        r, c = divmod(idx, cols)
        x0 = pad + c * cell_w
        y0 = pad + r * cell_h

        cutout: Image.Image = item["cutout"].convert("RGBA")
        thumb = cutout.copy()
        thumb.thumbnail((thumb_size, thumb_size), Image.Resampling.LANCZOS)
        tile = checker.copy()
        off_x = (thumb_size - thumb.width) // 2
        off_y = (thumb_size - thumb.height) // 2
        tile.paste(thumb, (off_x, off_y), thumb)
        sheet.paste(tile, (x0, y0))

        analysis: AlphaAnalysis = item["analysis"]
        is_winner = item.get("is_winner", False)
        if is_winner:
            border_color = (56, 224, 160)
            status = f"WINNER #{item['rank']}"
        elif analysis.passed:
            border_color = (82, 168, 255)
            status = f"PASS #{item['rank']}"
        else:
            border_color = (239, 83, 80)
            status = "REJECTED"

        draw.rectangle(
            [x0 - 2, y0 - 2, x0 + thumb_size + 1, y0 + thumb_size + 1],
            outline=border_color,
            width=3 if is_winner else 2,
        )

        line1 = f"V{item['variant_idx']:02d} [{status}] Score: {analysis.total_score:.1f}"
        line2 = (
            f"seed={item['seed']} area={analysis.fg_area_ratio * 100:.1f}% "
            f"frag={analysis.num_fragments} deb={analysis.debris_ratio * 100:.1f}%"
        )
        line3 = (
            f"margin={analysis.min_margin_px}px halo={analysis.halo_ratio * 100:.1f}%"
            if analysis.passed
            else f"reason: {', '.join(analysis.reject_reasons)[:44]}"
        )
        draw.text((x0, y0 + thumb_size + 6), line1, fill=border_color, font=font)
        draw.text((x0, y0 + thumb_size + 24), line2, fill=(215, 220, 230), font=font)
        draw.text((x0, y0 + thumb_size + 42), line3, fill=(175, 182, 196), font=font)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path, format="PNG")
