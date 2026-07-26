"""
Typography analysis service — style metadata for every OCR text block.

Consumes processed image + OCR JSON + layout JSON.
Does not generate PDF / SVG or estimate font families.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
from PIL import Image

from app.config import Settings
from app.ocr.ocr_service import resolve_processed_image
from app.typography.color_detector import (
    contrast_ratio,
    extract_text_and_background_colors,
    rgb_to_hex,
    rgb_to_hsv,
)
from app.typography.models import (
    TextHierarchy,
    TextStyle,
    TypographySummary,
)
from app.typography.style_estimator import (
    bbox_xyxy,
    classify_hierarchy,
    estimate_alignment,
    estimate_bold_italic_underline,
    estimate_font_family,
    estimate_font_size,
    estimate_spacing,
    line_and_paragraph_spacing,
    style_confidence,
    uppercase_ratio,
)
from app.ocr.script_detect import has_devanagari, is_mostly_devanagari

logger = logging.getLogger(__name__)

_HIER_COLORS = {
    TextHierarchy.TITLE: (0, 200, 0),
    TextHierarchy.HEADING: (0, 180, 100),
    TextHierarchy.SUBHEADING: (0, 160, 200),
    TextHierarchy.BODY: (220, 120, 40),
    TextHierarchy.FOOTER: (160, 160, 160),
    TextHierarchy.CAPTION: (180, 100, 200),
    TextHierarchy.LABEL: (100, 100, 220),
}


def _load_json(path: Path) -> Optional[dict[str, Any]]:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        logger.warning("Failed reading JSON %s", path)
        return None


def _layout_type_for_ocr_block(layout: Optional[dict[str, Any]], ocr_id: int) -> Optional[str]:
    if not layout:
        return None
    for obj in layout.get("objects") or []:
        ids = obj.get("ocr_block_ids") or []
        if ocr_id in ids:
            return obj.get("type")
        # Also match by identical text on TITLE/SUBTITLE/etc.
    return None


def analyze_typography(image_id: str, settings: Settings) -> dict[str, Any]:
    image_path = resolve_processed_image(image_id, settings)

    try:
        with Image.open(image_path) as img:
            img.load()
            page_w, page_h = img.size
    except Exception as exc:
        raise ValueError(f"Corrupted or unreadable image: {exc}") from exc

    bgr = cv2.imread(str(image_path))
    if bgr is None:
        pil = Image.open(image_path).convert("RGB")
        bgr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

    ocr = _load_json(settings.results_path / f"{image_id}.json")
    if not ocr or not ocr.get("text_blocks"):
        raise FileNotFoundError(
            f"OCR results not found for image_id={image_id}. Run /api/ocr first."
        )
    layout = _load_json(settings.results_path / f"layout_{image_id}.json")

    blocks: list[dict[str, Any]] = list(ocr["text_blocks"])
    started = time.perf_counter()

    heights = [max(1.0, bbox_xyxy(b)[3] - bbox_xyxy(b)[1]) for b in blocks]
    page_median_h = float(np.median(heights)) if heights else 20.0
    styles: list[TextStyle] = []

    for idx, block in enumerate(blocks):
        text = str(block.get("text") or "")
        x1, y1, x2, y2 = bbox_xyxy(block)
        box_w, box_h = max(1.0, x2 - x1), max(1.0, y2 - y1)
        ocr_id = int(block.get("id") or idx + 1)

        color_info = extract_text_and_background_colors(bgr, (x1, y1, x2, y2))
        font_rgb = color_info["font_rgb"]
        bg_rgb = color_info["bg_rgb"]
        opacity = float(color_info.get("opacity") or 1.0)
        ink_mask = color_info.get("ink_mask")

        gray_roi = None
        ix1, iy1 = int(x1), int(y1)
        ix2, iy2 = int(x2), int(y2)
        if iy2 > iy1 and ix2 > ix1:
            gray_roi = cv2.cvtColor(bgr[iy1:iy2, ix1:ix2], cv2.COLOR_BGR2GRAY)

        font_size, size_conf = estimate_font_size(block, ink_mask, page_median_h)
        bold, italic, underline = estimate_bold_italic_underline(ink_mask, gray_roi, font_size)
        # Slightly skewed OCR quads make upright serif look italic — dampen when rotation is tiny
        try:
            ocr_rot = abs(float(block.get("rotation") or 0.0))
        except (TypeError, ValueError):
            ocr_rot = 0.0
        if ocr_rot < 4.0:
            italic = min(italic, 0.25) * 0.35
        font_family, _fam_conf = estimate_font_family(ink_mask, gray_roi, italic)
        if is_mostly_devanagari(text) or has_devanagari(text):
            font_family = "devanagari"  # resolves to AAText
        # Short tokens / step numbers: bottom serifs ≠ underline
        if len(text.strip()) <= 3 or text.strip().rstrip(".").isdigit():
            underline = min(underline, 0.15)
        up_ratio = uppercase_ratio(text)
        char_sp, word_sp = estimate_spacing(text, box_w, box_h, font_size)

        # Siblings in same paragraph for alignment
        para = block.get("paragraph")
        siblings = [b for b in blocks if b.get("paragraph") == para and b is not block]
        alignment = estimate_alignment(block, siblings, float(page_w))

        line_sp, para_sp, indent, para_w = line_and_paragraph_spacing(block, blocks, font_size)

        # Rank among page sizes (0-1)
        rank = (
            sum(1 for h in heights if h <= box_h) / float(len(heights))
            if heights
            else 1.0
        )
        layout_type = _layout_type_for_ocr_block(layout, ocr_id)
        # Also use layout objects by text match
        if layout and not layout_type:
            for obj in layout.get("objects") or []:
                if (obj.get("text") or "").strip() == text.strip() and obj.get("type") in {
                    "TITLE",
                    "SUBTITLE",
                    "PARAGRAPH",
                    "TEXT_BLOCK",
                    "LIST",
                }:
                    layout_type = obj.get("type")
                    break

        hierarchy = classify_hierarchy(block, font_size, float(page_h), rank, layout_type, up_ratio)
        contrast = contrast_ratio(font_rgb, bg_rgb)
        hsv = rgb_to_hsv(*font_rgb)

        conf = style_confidence(
            [
                size_conf,
                0.95 if contrast >= 3.0 else 0.6,
                0.85,
                1.0 - abs(0.5 - min(bold, 1.0)) * 0.2,
            ],
            float(block.get("confidence") or 0.8),
        )

        margins = {
            "left": round(x1, 2),
            "right": round(page_w - x2, 2),
            "top": round(y1, 2),
            "bottom": round(page_h - y2, 2),
        }

        styles.append(
            TextStyle(
                id=idx + 1,
                ocr_block_id=ocr_id,
                text=text,
                font_size=font_size,
                font_color=rgb_to_hex(*font_rgb),
                font_color_rgb=list(font_rgb),
                font_color_hsv=hsv,
                background_color=rgb_to_hex(*bg_rgb),
                background_color_rgb=list(bg_rgb),
                contrast_ratio=contrast,
                bold=bold,
                italic=italic,
                underline=underline,
                uppercase_ratio=up_ratio,
                alignment=alignment,
                character_spacing=char_sp,
                word_spacing=word_sp,
                line_spacing=line_sp,
                paragraph_spacing=para_sp,
                text_box_width=round(box_w, 2),
                text_box_height=round(box_h, 2),
                rotation=float(block.get("rotation") or 0.0),
                opacity=opacity,
                hierarchy=hierarchy,
                indentation=indent,
                margins=margins,
                paragraph_width=para_w,
                average_line_height=round(font_size * line_sp, 2),
                confidence=conf,
                bbox=[round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)],
                font_family=font_family,
            )
        )

    # Promote top dark headline cluster: bold + near-black (avoid highlight bleed)
    if styles:
        top = sorted(styles, key=lambda s: float((s.bbox or [0, 9999])[1]))
        top_band = [
            s
            for s in top
            if float((s.bbox or [0, 9999])[1]) < float(page_h) * 0.22
            and len((s.text or "").strip()) >= 3
        ]
        if top_band:
            max_h = max(
                float((s.bbox or [0, 0, 0, 0])[3] - (s.bbox or [0, 0, 0, 0])[1]) for s in top_band
            )
            headline = []
            for s in top_band:
                box_h = float((s.bbox or [0, 0, 0, 0])[3] - (s.bbox or [0, 0, 0, 0])[1])
                if box_h < max_h * 0.82:
                    continue
                lum = (
                    0.299 * s.font_color_rgb[0]
                    + 0.587 * s.font_color_rgb[1]
                    + 0.114 * s.font_color_rgb[2]
                )
                # Only dark headline ink — skip gray subheads
                if lum > 90:
                    continue
                headline.append(s)
            for s in headline:
                if s.hierarchy not in {TextHierarchy.TITLE, TextHierarchy.HEADING}:
                    s.hierarchy = TextHierarchy.TITLE
                s.bold = max(float(s.bold or 0), 0.72)
                s.font_color = "#1A1A1A"
                s.font_color_rgb = [26, 26, 26]

    # Soft-snap body bold so mid-range noise doesn't flicker weight in PDF
    for s in styles:
        if s.hierarchy in {TextHierarchy.TITLE, TextHierarchy.HEADING}:
            continue
        b = float(s.bold or 0)
        if 0.35 <= b < 0.55:
            s.bold = 0.2
        elif b >= 0.55 and s.hierarchy in {TextHierarchy.BODY, TextHierarchy.FOOTER, TextHierarchy.CAPTION}:
            # Dense poem/body lines rarely truly bold unless strong evidence
            if b < 0.7:
                s.bold = 0.25
        # Serif baselines / descenders often false-trigger underline
        if float(s.underline or 0) < 0.85:
            s.underline = min(float(s.underline or 0), 0.15)

    # Page-level font family vote — keep one consistent face for poster-like pages
    if styles:
        from collections import Counter

        blob = " ".join((s.text or "").lower() for s in styles)
        from app.classify.document_type import load_document_mode

        mode_info = load_document_mode(image_id, settings) or {}
        doc_mode = str(mode_info.get("mode") or "")
        form_like = doc_mode == "ruled_form" or sum(
            1
            for k in (
                "particulars",
                "quantity",
                "amount",
                "bill",
                "cash memo",
                "invoice",
                "lorry",
                "rate",
                "total",
            )
            if k in blob
        ) >= 3
        designed_invoice = doc_mode == "designed_invoice"

        if form_like and not designed_invoice:
            # Bill books: prefer clean sans; never force handwriting
            for s in styles:
                t = (s.text or "").strip()
                low = t.lower()
                if has_devanagari(t):
                    s.font_family = "devanagari"
                    continue
                if s.font_family == "handwriting":
                    s.font_family = "sans"
                # Large ALL-CAPS shop name stays sans bold
                if t.isupper() and len(t) >= 8:
                    s.font_family = "sans"
                    s.bold = max(float(s.bold or 0), 0.85)
                # Prop badge / italic distributor lines
                elif low.startswith("prop") or "distribut" in low:
                    s.font_family = "serif"
                    s.italic = max(float(s.italic or 0), 0.75)
                    if low.startswith("prop"):
                        s.font_color = "#FFFFFF"
                else:
                    # Labels, headers, signatures → sans for bill clarity
                    s.font_family = "sans"
                # Pure black ink by default
                if low.startswith("prop") or "service station" in low or "reparing" in low or "repairing" in low:
                    if len(t) >= 12:
                        s.font_color = "#FFFFFF"
                else:
                    s.font_color = "#000000"
        else:
            # Keep Devanagari faces; only unify Latin families
            for s in styles:
                if has_devanagari(s.text or ""):
                    s.font_family = "devanagari"
            latin_styles = [s for s in styles if not has_devanagari(s.text or "")]
            # Prefer TITLE signal when clear (handwritten posters vs serif letters)
            title_styles = [
                s
                for s in latin_styles
                if s.hierarchy == TextHierarchy.TITLE and len((s.text or "").strip()) >= 4
            ]
            if not title_styles:
                title_styles = [
                    s
                    for s in latin_styles
                    if s.hierarchy == TextHierarchy.HEADING and len((s.text or "").strip()) >= 4
                ]
            if title_styles:
                tw = Counter()
                for s in title_styles:
                    tw[s.font_family] += float(s.font_size or 12.0)
                title_dom, _ = tw.most_common(1)[0]
                if title_dom == "handwriting" and tw["handwriting"] >= sum(tw.values()) * 0.5:
                    for s in latin_styles:
                        s.font_family = "handwriting"
                elif title_dom in {"serif", "sans"}:
                    for s in latin_styles:
                        s.font_family = title_dom
                else:
                    title_dom = None
            else:
                title_dom = None

            if title_dom is None and latin_styles:
                votes = Counter(s.font_family for s in latin_styles)
                dominant, n = votes.most_common(1)[0]
                need = max(3, int(len(latin_styles) * (0.55 if dominant == "handwriting" else 0.35)))
                if n >= need:
                    for s in latin_styles:
                        s.font_family = dominant
                else:
                    fallback = "serif"
                    if votes.get("sans", 0) > votes.get("serif", 0) and votes.get("sans", 0) >= votes.get(
                        "handwriting", 0
                    ):
                        fallback = "sans"
                    for s in latin_styles:
                        s.font_family = fallback

    elapsed_ms = (time.perf_counter() - started) * 1000
    summary = _build_summary(styles)

    logger.info(
        "Typography complete id=%s styles=%d avg_size=%.1f avg_conf=%.1f colors=%d time=%.1fms",
        image_id,
        summary.total_styles,
        summary.average_font_size,
        summary.average_confidence,
        summary.unique_colors,
        elapsed_ms,
    )
    logger.info("Detected colors: %s", ", ".join(summary.detected_colors[:12]) or "(none)")

    results_path = settings.results_path / f"typography_{image_id}.json"
    debug_path = settings.debug_path / f"typography_{image_id}.png"
    _draw_debug(image_path, styles, debug_path)

    payload = {
        "success": True,
        "image_id": image_id,
        "page": {"width": page_w, "height": page_h},
        "text_styles": [s.model_dump(mode="json") for s in styles],
        "summary": summary.model_dump(mode="json"),
        "processing_time_ms": round(elapsed_ms, 1),
        "results_file": str(results_path),
        "debug_image": str(debug_path),
        "message": "Typography analysis completed successfully.",
    }
    results_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Typography JSON saved -> %s", results_path)
    return payload


def _build_summary(styles: list[TextStyle]) -> TypographySummary:
    if not styles:
        return TypographySummary()

    align_dist: dict[str, int] = {}
    colors: list[str] = []
    for s in styles:
        key = s.alignment.value if hasattr(s.alignment, "value") else str(s.alignment)
        align_dist[key] = align_dist.get(key, 0) + 1
        if s.font_color not in colors:
            colors.append(s.font_color)

    hier_counts = {
        TextHierarchy.TITLE: 0,
        TextHierarchy.HEADING: 0,
        TextHierarchy.SUBHEADING: 0,
        TextHierarchy.BODY: 0,
        TextHierarchy.FOOTER: 0,
        TextHierarchy.CAPTION: 0,
        TextHierarchy.LABEL: 0,
    }
    for s in styles:
        hier_counts[s.hierarchy] = hier_counts.get(s.hierarchy, 0) + 1

    return TypographySummary(
        total_styles=len(styles),
        average_font_size=round(float(np.mean([s.font_size for s in styles])), 2),
        average_confidence=round(float(np.mean([s.confidence for s in styles])), 2),
        titles=hier_counts[TextHierarchy.TITLE],
        headings=hier_counts[TextHierarchy.HEADING],
        subheadings=hier_counts[TextHierarchy.SUBHEADING],
        body_text=hier_counts[TextHierarchy.BODY],
        footers=hier_counts[TextHierarchy.FOOTER],
        captions=hier_counts[TextHierarchy.CAPTION],
        labels=hier_counts[TextHierarchy.LABEL],
        unique_colors=len(colors),
        alignment_distribution=align_dist,
        average_bold=round(float(np.mean([s.bold for s in styles])), 3),
        detected_colors=colors,
    )


def _draw_debug(image_path: Path, styles: list[TextStyle], output_path: Path) -> None:
    image = cv2.imread(str(image_path))
    if image is None:
        pil = Image.open(image_path).convert("RGB")
        image = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

    for s in styles:
        x1, y1, x2, y2 = map(int, s.bbox)
        color = _HIER_COLORS.get(s.hierarchy, (128, 128, 128))
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

        # Color sample swatch
        fr, fg, fb = s.font_color_rgb
        cv2.rectangle(image, (x1, max(0, y1 - 18)), (x1 + 16, max(0, y1 - 2)), (fb, fg, fr), -1)

        label = f"{s.hierarchy.value} {s.font_size:.0f}px {s.font_color}"
        cv2.putText(
            image,
            label[:48],
            (x1 + 20, max(14, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (20, 20, 20),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            image,
            label[:48],
            (x1 + 20, max(14, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            color,
            1,
            cv2.LINE_AA,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), image)
    logger.info("Typography debug image saved -> %s", output_path)
