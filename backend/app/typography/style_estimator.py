"""
Typography style estimators — size, weight, slant, spacing, alignment, hierarchy.
"""

from __future__ import annotations

import math
import re
from typing import Any, Optional

import cv2
import numpy as np

from app.typography.models import Alignment, TextHierarchy


def bbox_xyxy(block: dict[str, Any]) -> tuple[float, float, float, float]:
    pts = block.get("bbox") or []
    if pts and isinstance(pts[0], (list, tuple)):
        xs = [float(p[0]) for p in pts]
        ys = [float(p[1]) for p in pts]
        return min(xs), min(ys), max(xs), max(ys)
    return (
        float(block.get("center_x", 0) - block.get("width", 0) / 2),
        float(block.get("center_y", 0) - block.get("height", 0) / 2),
        float(block.get("center_x", 0) + block.get("width", 0) / 2),
        float(block.get("center_y", 0) + block.get("height", 0) / 2),
    )


def estimate_font_size(
    block: dict[str, Any],
    ink_mask: Optional[np.ndarray],
    page_median_height: float,
) -> tuple[float, float]:
    """
    Estimate font size in pixels from bbox height and ink connected-component heights.
    Returns (font_size, confidence_component 0-1).
    """
    x1, y1, x2, y2 = bbox_xyxy(block)
    box_h = max(1.0, y2 - y1)
    char_h = box_h * 0.82  # typical ink vs box padding

    if ink_mask is not None and ink_mask.any():
        mask_u8 = (ink_mask.astype(np.uint8) * 255)
        n, _, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
        heights = []
        for i in range(1, n):
            ch = int(stats[i, cv2.CC_STAT_HEIGHT])
            cw = int(stats[i, cv2.CC_STAT_WIDTH])
            area = int(stats[i, cv2.CC_STAT_AREA])
            if area < 4 or ch < 3:
                continue
            if ch > box_h * 1.2 or cw > (x2 - x1):
                continue
            heights.append(ch)
        if heights:
            char_h = float(np.median(heights))

    # Blend box-based and component-based estimates
    font_size = 0.55 * box_h + 0.45 * char_h
    # Soft normalize vs page median (keep absolute pixels but stabilize)
    if page_median_height > 0:
        font_size = 0.85 * font_size + 0.15 * (font_size / page_median_height) * page_median_height

    conf = 0.9 if ink_mask is not None and ink_mask.any() else 0.7
    return round(float(font_size), 1), conf


def estimate_bold_italic_underline(
    ink_mask: Optional[np.ndarray],
    gray_roi: Optional[np.ndarray],
    font_size: float,
) -> tuple[float, float, float]:
    """
    Return (bold_prob, italic_prob, underline_prob) in 0-1.
    """
    bold = 0.15
    italic = 0.05
    underline = 0.02

    if ink_mask is None or not ink_mask.any():
        return bold, italic, underline

    mask = (ink_mask.astype(np.uint8) * 255)
    # Stroke width via distance transform on ink
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
    ink_dist = dist[ink_mask]
    if ink_dist.size:
        stroke = float(np.median(ink_dist)) * 2.0
        # Bold if stroke is thick relative to font size
        ratio = stroke / max(font_size, 1.0)
        bold = float(np.clip((ratio - 0.08) / 0.18, 0.0, 1.0))

    # Italic: shear of ink moments / row shifts
    ys, xs = np.where(ink_mask)
    if len(xs) > 30:
        # For each row, mean x of ink; slope vs y → italic slant
        row_means = []
        for y in range(ink_mask.shape[0]):
            row = np.where(ink_mask[y])[0]
            if len(row) >= 2:
                row_means.append((y, float(row.mean())))
        if len(row_means) >= 4:
            yy = np.array([p[0] for p in row_means], dtype=np.float64)
            xx = np.array([p[1] for p in row_means], dtype=np.float64)
            # Fit x = a*y + b; |a| large → italic.
            # Serif upright glyphs often yield noisy small slopes — require stronger slant.
            a, _ = np.polyfit(yy, xx, 1)
            slant = abs(float(a))
            if slant < 0.12:
                italic = 0.05
            else:
                italic = float(np.clip((slant - 0.12) / 0.40, 0.0, 1.0))

    # Underline: dark horizontal band near bottom
    if gray_roi is not None and gray_roi.size:
        h = gray_roi.shape[0]
        band = gray_roi[int(h * 0.82) : h, :]
        if band.size:
            # Strong horizontal edge / dark line
            row_mean = band.mean(axis=1)
            if len(row_mean) >= 2:
                darkest = float(row_mean.min())
                overall = float(gray_roi.mean())
                if overall - darkest > 25:
                    # Check continuity
                    dark_row = int(np.argmin(row_mean))
                    line = band[dark_row, :]
                    dark_frac = float((line < overall - 20).mean())
                    underline = float(np.clip(dark_frac, 0.0, 1.0))

    return round(bold, 3), round(italic, 3), round(underline, 3)


def estimate_font_family(
    ink_mask: Optional[np.ndarray],
    gray_roi: Optional[np.ndarray],
    italic: float,
) -> tuple[str, float]:
    """
    Coarse family guess: handwriting | sans | serif.
    Returns (family, confidence 0-1).
    """
    if ink_mask is None or not ink_mask.any():
        return "serif", 0.4

    mask = ink_mask.astype(bool)
    # Stroke-width variance — handwriting tends to vary more
    dist = cv2.distanceTransform((mask.astype(np.uint8) * 255), cv2.DIST_L2, 3)
    ink_d = dist[mask]
    stroke_cv = 0.0
    if ink_d.size > 20:
        med = float(np.median(ink_d)) + 1e-6
        stroke_cv = float(np.std(ink_d) / med)

    # Contour irregularity
    irregularity = 0.0
    try:
        cnts, _ = cv2.findContours(
            (mask.astype(np.uint8) * 255), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if cnts:
            cnt = max(cnts, key=cv2.contourArea)
            peri = cv2.arcLength(cnt, True)
            area = abs(cv2.contourArea(cnt)) + 1e-6
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            # More vertices / higher peri^2/area → more organic (hand) glyphs
            circularity = 4 * np.pi * area / (peri * peri + 1e-6)
            irregularity = float(np.clip(len(approx) / 40.0 + (1.0 - circularity), 0, 2))
    except Exception:
        irregularity = 0.0

    # Edge density near bbox corners suggests serifs
    serif_score = 0.0
    if gray_roi is not None and gray_roi.size and gray_roi.shape[0] > 6 and gray_roi.shape[1] > 6:
        h, w = gray_roi.shape[:2]
        corners = [
            gray_roi[: h // 4, : w // 5],
            gray_roi[: h // 4, -w // 5 :],
            gray_roi[-h // 4 :, : w // 5],
            gray_roi[-h // 4 :, -w // 5 :],
        ]
        for c in corners:
            if c.size:
                serif_score += float((c < np.percentile(gray_roi, 35)).mean())
        serif_score /= 4.0

    hand_score = (
        0.45 * min(stroke_cv / 0.50, 1.6)
        + 0.45 * min(irregularity, 1.6)
        + 0.10 * min(italic, 1.0)
    )
    # Serif corners beat handwriting guess (classic book titles on parchment)
    if serif_score >= 0.28 and stroke_cv < 0.65:
        return "serif", float(np.clip(0.55 + serif_score, 0.55, 0.95))
    # Handwriting: organic strokes without strong serif corners
    if hand_score >= 0.82 and stroke_cv >= 0.42 and serif_score < 0.28:
        return "handwriting", float(np.clip(hand_score / 1.2, 0.55, 0.95))
    if serif_score >= 0.18 and stroke_cv < 0.5:
        return "serif", float(np.clip(0.55 + serif_score, 0.55, 0.92))
    if stroke_cv < 0.38 and irregularity < 0.6:
        return "sans", 0.65
    return "serif", 0.6


def uppercase_ratio(text: str) -> float:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    return round(sum(1 for c in letters if c.isupper()) / len(letters), 3)


def estimate_spacing(
    text: str,
    box_w: float,
    box_h: float,
    font_size: float,
) -> tuple[float, float]:
    """
    Character spacing and word spacing estimates (relative units).
    """
    text = text.strip()
    if not text:
        return 0.0, 1.0

    n_chars = max(1, len(re.sub(r"\s+", "", text)))
    n_words = max(1, len(text.split()))
    # Expected width ≈ n_chars * font_size * 0.55 for typical Latin
    expected = n_chars * font_size * 0.55
    char_spacing = (box_w / expected) if expected > 0 else 1.0
    char_spacing = float(np.clip(char_spacing, 0.4, 2.5))

    words = text.split()
    if len(words) <= 1:
        word_spacing = 1.0
    else:
        # Remaining width after char mass distributed into gaps
        word_spacing = float(np.clip(box_w / (n_words * font_size), 0.3, 3.0))

    return round(char_spacing, 3), round(word_spacing, 3)


def estimate_alignment(
    block: dict[str, Any],
    siblings: list[dict[str, Any]],
    page_w: float,
    column_bounds: Optional[tuple[float, float]] = None,
) -> Alignment:
    """
    Infer alignment from geometry within page/column and sibling consistency.
    """
    x1, y1, x2, y2 = bbox_xyxy(block)
    left_edge, right_edge = (0.0, page_w)
    if column_bounds:
        left_edge, right_edge = column_bounds

    col_w = max(1.0, right_edge - left_edge)
    left_gap = (x1 - left_edge) / col_w
    right_gap = (right_edge - x2) / col_w
    center_off = abs(((x1 + x2) / 2.0) - ((left_edge + right_edge) / 2.0)) / col_w

    # Justified: wide block spanning most of column
    span = (x2 - x1) / col_w
    if span > 0.82 and left_gap < 0.08 and right_gap < 0.08:
        return Alignment.JUSTIFIED

    # Prefer left for typical poster/list copy; only mark center when clearly centered
    if center_off < 0.04 and abs(left_gap - right_gap) < 0.06 and left_gap > 0.18:
        return Alignment.CENTER
    if right_gap < 0.06 and left_gap > 0.22:
        return Alignment.RIGHT
    if left_gap < 0.18:
        return Alignment.LEFT

    # Sibling vote — only if strongly centered as a group
    if siblings:
        centers = []
        for s in siblings:
            sx1, _, sx2, _ = bbox_xyxy(s)
            centers.append(((sx1 + sx2) / 2.0 - left_edge) / col_w)
        if centers and abs(float(np.mean(centers)) - 0.5) < 0.05 and float(np.mean(centers)) > 0.35:
            return Alignment.CENTER

    return Alignment.LEFT if left_gap <= right_gap else Alignment.RIGHT


def classify_hierarchy(
    block: dict[str, Any],
    font_size: float,
    page_h: float,
    size_rank: float,
    layout_type: Optional[str],
    uppercase: float,
) -> TextHierarchy:
    """
    Map to TITLE / HEADING / SUBHEADING / BODY / FOOTER / CAPTION / LABEL.
    """
    _, y1, _, y2 = bbox_xyxy(block)
    cy = (y1 + y2) / 2.0
    rel_y = cy / max(page_h, 1.0)
    text = (block.get("text") or "").strip()
    words = len(text.split())

    if layout_type == "TITLE" or (size_rank >= 0.92 and rel_y < 0.35 and uppercase > 0.6):
        return TextHierarchy.TITLE
    if layout_type == "SUBTITLE" or (size_rank >= 0.75 and rel_y < 0.45):
        return TextHierarchy.HEADING if size_rank >= 0.82 else TextHierarchy.SUBHEADING
    if rel_y > 0.82 or layout_type == "FOOTER":
        return TextHierarchy.FOOTER
    if size_rank < 0.35 and words <= 6:
        return TextHierarchy.LABEL if rel_y < 0.7 else TextHierarchy.CAPTION
    if size_rank < 0.45 and words <= 10:
        return TextHierarchy.CAPTION
    if size_rank >= 0.65:
        return TextHierarchy.SUBHEADING
    return TextHierarchy.BODY


def line_and_paragraph_spacing(
    block: dict[str, Any],
    all_blocks: list[dict[str, Any]],
    font_size: float,
) -> tuple[float, float, float, float]:
    """
    Returns (line_spacing, paragraph_spacing, indentation, paragraph_width).
    Spacing values are multipliers of font size where applicable.
    """
    x1, y1, x2, y2 = bbox_xyxy(block)
    para = block.get("paragraph")
    line = block.get("line")

    same_para = [
        b
        for b in all_blocks
        if b.get("paragraph") == para and para is not None
    ] or [block]
    same_para = sorted(same_para, key=lambda b: bbox_xyxy(b)[1])

    line_spacing = 1.2
    if len(same_para) >= 2:
        gaps = []
        for i in range(1, len(same_para)):
            _, py1, _, py2 = bbox_xyxy(same_para[i - 1])
            _, cy1, _, _ = bbox_xyxy(same_para[i])
            gap = cy1 - py2
            if gap > 0:
                gaps.append(gap / max(font_size, 1.0))
        if gaps:
            line_spacing = float(np.clip(np.median(gaps) + 1.0, 0.8, 3.0))

    # Paragraph spacing: gap to next different paragraph
    paragraph_spacing = 1.5
    others = sorted(all_blocks, key=lambda b: bbox_xyxy(b)[1])
    for b in others:
        if b.get("id") == block.get("id"):
            continue
        _, by1, _, _ = bbox_xyxy(b)
        if by1 > y2:
            paragraph_spacing = float(np.clip((by1 - y2) / max(font_size, 1.0), 0.5, 5.0))
            break

    # Indentation relative to paragraph left
    para_left = min(bbox_xyxy(b)[0] for b in same_para)
    indentation = max(0.0, x1 - para_left)
    paragraph_width = max(bbox_xyxy(b)[2] for b in same_para) - para_left

    return (
        round(line_spacing, 3),
        round(paragraph_spacing, 3),
        round(indentation, 2),
        round(paragraph_width, 2),
    )


def style_confidence(
    components: list[float],
    ocr_conf: float,
) -> float:
    """Aggregate 0-100 confidence score."""
    base = float(np.mean(components)) if components else 0.7
    score = 100.0 * (0.7 * base + 0.3 * float(np.clip(ocr_conf, 0, 1)))
    return round(float(np.clip(score, 0, 100)), 1)
