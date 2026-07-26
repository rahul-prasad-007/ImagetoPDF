"""
Layout analysis service — structured document model from processed images + OCR.

Hybrid OpenCV pipeline (Windows-safe). No PDF / fonts / background reconstruction.
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

from app.classify.document_type import ensure_document_mode
from app.config import Settings
from app.layout.detectors import (
    RawDetection,
    _is_highlight_bgr,
    _text_coverage_ratio,
    detect_background_panels,
    detect_image_regions,
    detect_lines,
    detect_shapes,
    merge_paragraph_groups,
    ocr_blocks_to_text_detections,
    _contains,
    _iou,
)
from app.layout.models import (
    LayoutCounts,
    LayoutObject,
    LayoutTreeNode,
    ObjectType,
    PageInfo,
)
from app.ocr.ocr_service import resolve_processed_image
from app.structure.pp_structure import run_pp_structure

logger = logging.getLogger(__name__)

# Debug draw colors BGR
_DEBUG_COLORS = {
    ObjectType.TITLE: (0, 200, 0),  # Green
    ObjectType.SUBTITLE: (0, 180, 80),
    ObjectType.PARAGRAPH: (255, 140, 0),  # Blue-ish (BGR) → actually orange-blue; use blue
    ObjectType.TEXT_BLOCK: (255, 120, 0),  # Blue
    ObjectType.LIST: (255, 160, 40),
    ObjectType.IMAGE: (180, 0, 180),  # Purple
    ObjectType.PHOTO: (200, 0, 160),
    ObjectType.LOGO: (0, 220, 255),  # Yellow
    ObjectType.ICON: (0, 200, 230),
    ObjectType.QR_CODE: (0, 180, 255),
    ObjectType.RECTANGLE: (0, 140, 255),  # Orange
    ObjectType.ROUNDED_RECTANGLE: (0, 110, 255),
    ObjectType.LINE: (80, 80, 80),
    ObjectType.CIRCLE: (200, 100, 50),
    ObjectType.ELLIPSE: (180, 90, 40),
    ObjectType.TABLE: (100, 50, 200),
    ObjectType.BACKGROUND_SHAPE: (160, 160, 160),
    ObjectType.DECORATIVE_ELEMENT: (120, 120, 200),
    ObjectType.BACKGROUND: (200, 200, 200),
    ObjectType.HEADER: (100, 180, 100),
    ObjectType.MAIN_CONTENT: (180, 160, 100),
    ObjectType.FOOTER: (100, 140, 180),
    ObjectType.PAGE: (50, 50, 50),
}

# Fix TEXT colors to true blue in BGR
_DEBUG_COLORS[ObjectType.PARAGRAPH] = (220, 100, 30)  # blue-ish
_DEBUG_COLORS[ObjectType.TEXT_BLOCK] = (255, 90, 20)  # blue

_KIND_TO_TYPE = {
    "title": ObjectType.TITLE,
    "subtitle": ObjectType.SUBTITLE,
    "paragraph": ObjectType.PARAGRAPH,
    "text_block": ObjectType.TEXT_BLOCK,
    "list": ObjectType.LIST,
    "line": ObjectType.LINE,
    "rectangle": ObjectType.RECTANGLE,
    "rounded_rectangle": ObjectType.ROUNDED_RECTANGLE,
    "circle": ObjectType.CIRCLE,
    "ellipse": ObjectType.ELLIPSE,
    "panel": ObjectType.BACKGROUND_SHAPE,
    "photo": ObjectType.PHOTO,
    "logo": ObjectType.LOGO,
    "icon": ObjectType.ICON,
    "image": ObjectType.IMAGE,
    "qr_code": ObjectType.QR_CODE,
    "table": ObjectType.TABLE,
    "decorative": ObjectType.DECORATIVE_ELEMENT,
}


def _load_ocr_json(image_id: str, settings: Settings) -> Optional[dict[str, Any]]:
    path = settings.results_path / f"{image_id}.json"
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        logger.warning("Could not read OCR results for %s", image_id)
        return None


def _raw_to_object(det: RawDetection, obj_id: int, page_area: float) -> LayoutObject:
    x1, y1, x2, y2 = det.bbox
    w, h = max(0.0, x2 - x1), max(0.0, y2 - y1)
    area = w * h
    # Larger area → lower z (background-ish); small → higher
    z = int(max(0, min(1000, 1000 - (area / (page_area + 1e-6)) * 1000)))
    otype = _KIND_TO_TYPE.get(det.kind, ObjectType.DECORATIVE_ELEMENT)
    return LayoutObject(
        id=obj_id,
        type=otype,
        bbox=[round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)],
        center_x=round((x1 + x2) / 2.0, 2),
        center_y=round((y1 + y2) / 2.0, 2),
        width=round(w, 2),
        height=round(h, 2),
        area=round(area, 2),
        rotation=round(float(det.rotation), 2),
        z_index=z,
        parent=None,
        children=[],
        text=det.text,
        ocr_block_ids=list(det.ocr_block_ids),
        confidence=round(float(det.confidence), 4) if det.confidence is not None else None,
        meta=dict(det.meta or {}),
    )


def _suppress_shapes_overlapping_text(
    shapes: list[RawDetection],
    texts: list[RawDetection],
) -> list[RawDetection]:
    """Drop shape boxes that are essentially OCR text outlines; keep highlights."""
    text_boxes = [t.bbox for t in texts]
    kept: list[RawDetection] = []
    for s in shapes:
        if s.kind in {"line", "circle", "ellipse"}:
            kept.append(s)
            continue
        meta = s.meta or {}
        if meta.get("highlight") or _is_highlight_bgr(meta.get("color_bgr")):
            kept.append(s)
            continue
        # If a shape tightly wraps a single text block, skip it
        overlaps = [t for t in texts if _iou(s.bbox, t.bbox) > 0.65]
        if overlaps and abs(s.area - overlaps[0].area) / (s.area + 1e-6) < 0.35:
            continue
        # Drop gray panels sitting on top of list/body text
        coverage = _text_coverage_ratio(s.bbox, text_boxes)
        if coverage > 0.35 and s.kind in {"rectangle", "rounded_rectangle", "panel"}:
            continue
        kept.append(s)
    return kept


def _assign_parents(objects: list[LayoutObject], page_w: int, page_h: int) -> None:
    """
    Build parent-child via spatial containment.
    Prefer smallest containing non-text structural parent.
    """
    # Structural preference order for containers
    container_types = {
        ObjectType.BACKGROUND_SHAPE,
        ObjectType.RECTANGLE,
        ObjectType.ROUNDED_RECTANGLE,
        ObjectType.PHOTO,
        ObjectType.IMAGE,
        ObjectType.HEADER,
        ObjectType.MAIN_CONTENT,
        ObjectType.FOOTER,
        ObjectType.BACKGROUND,
        ObjectType.PAGE,
    }

    by_id = {o.id: o for o in objects}

    for obj in objects:
        if obj.type == ObjectType.PAGE:
            continue
        candidates: list[LayoutObject] = []
        for other in objects:
            if other.id == obj.id:
                continue
            if other.type not in container_types and other.type not in {
                ObjectType.RECTANGLE,
                ObjectType.ROUNDED_RECTANGLE,
                ObjectType.BACKGROUND_SHAPE,
            }:
                # Text can still be parent of nothing typically
                if other.type in {
                    ObjectType.TITLE,
                    ObjectType.SUBTITLE,
                    ObjectType.PARAGRAPH,
                    ObjectType.TEXT_BLOCK,
                    ObjectType.LIST,
                    ObjectType.LINE,
                }:
                    continue
            ob = (other.bbox[0], other.bbox[1], other.bbox[2], other.bbox[3])
            ib = (obj.bbox[0], obj.bbox[1], obj.bbox[2], obj.bbox[3])
            if _contains(ob, ib, pad=4.0) and other.area > obj.area * 1.05:
                candidates.append(other)

        if not candidates:
            # Attach to PAGE
            page = next((o for o in objects if o.type == ObjectType.PAGE), None)
            if page:
                obj.parent = page.id
            continue

        # Smallest area container wins
        parent = min(candidates, key=lambda c: c.area)
        obj.parent = parent.id

    # Rebuild children lists
    for o in objects:
        o.children = []
    for o in objects:
        if o.parent is not None and o.parent in by_id:
            by_id[o.parent].children.append(o.id)

    # Reading-order sort children (top→bottom, left→right)
    for o in objects:
        o.children.sort(
            key=lambda cid: (
                by_id[cid].center_y if cid in by_id else 0,
                by_id[cid].center_x if cid in by_id else 0,
            )
        )


def _add_section_bands(
    objects: list[LayoutObject],
    page_w: int,
    page_h: int,
    next_id: int,
) -> tuple[list[LayoutObject], int]:
    """Create HEADER / MAIN_CONTENT / FOOTER band containers and re-parent loose items."""
    page = next(o for o in objects if o.type == ObjectType.PAGE)
    header = LayoutObject(
        id=next_id,
        type=ObjectType.HEADER,
        bbox=[0, 0, float(page_w), float(page_h * 0.28)],
        center_x=page_w / 2,
        center_y=page_h * 0.14,
        width=float(page_w),
        height=float(page_h * 0.28),
        area=float(page_w * page_h * 0.28),
        z_index=10,
        parent=page.id,
        children=[],
    )
    next_id += 1
    main = LayoutObject(
        id=next_id,
        type=ObjectType.MAIN_CONTENT,
        bbox=[0, float(page_h * 0.22), float(page_w), float(page_h * 0.82)],
        center_x=page_w / 2,
        center_y=page_h * 0.52,
        width=float(page_w),
        height=float(page_h * 0.60),
        area=float(page_w * page_h * 0.60),
        z_index=10,
        parent=page.id,
        children=[],
    )
    next_id += 1
    footer = LayoutObject(
        id=next_id,
        type=ObjectType.FOOTER,
        bbox=[0, float(page_h * 0.78), float(page_w), float(page_h)],
        center_x=page_w / 2,
        center_y=page_h * 0.89,
        width=float(page_w),
        height=float(page_h * 0.22),
        area=float(page_w * page_h * 0.22),
        z_index=10,
        parent=page.id,
        children=[],
    )
    next_id += 1

    objects.extend([header, main, footer])
    page.children.extend([header.id, main.id, footer.id])
    by_id = {o.id: o for o in objects}

    # Re-parent direct children of PAGE (except bands/background) into bands by Y
    for obj in list(objects):
        if obj.parent != page.id:
            continue
        if obj.type in {
            ObjectType.HEADER,
            ObjectType.MAIN_CONTENT,
            ObjectType.FOOTER,
            ObjectType.BACKGROUND,
            ObjectType.PAGE,
        }:
            continue
        cy = obj.center_y
        if cy < page_h * 0.28:
            band = header
        elif cy > page_h * 0.78:
            band = footer
        else:
            band = main
        if obj.id in page.children:
            page.children.remove(obj.id)
        obj.parent = band.id
        band.children.append(obj.id)

    for band in (header, main, footer):
        band.children.sort(
            key=lambda cid: (by_id[cid].center_y, by_id[cid].center_x)
        )

    return objects, next_id


def _build_tree(objects: list[LayoutObject]) -> LayoutTreeNode:
    by_id = {o.id: o for o in objects}
    page = next(o for o in objects if o.type == ObjectType.PAGE)

    def walk(oid: int) -> LayoutTreeNode:
        obj = by_id[oid]
        label = None
        if obj.text:
            label = obj.text[:60] + ("…" if len(obj.text) > 60 else "")
        elif obj.type == ObjectType.LOGO:
            label = "Logo"
        return LayoutTreeNode(
            id=obj.id,
            type=obj.type,
            label=label,
            children=[walk(cid) for cid in obj.children if cid in by_id],
        )

    return walk(page.id)


def _compute_counts(objects: list[LayoutObject]) -> LayoutCounts:
    counts = LayoutCounts(total=len(objects))
    mapping = {
        ObjectType.TITLE: "titles",
        ObjectType.SUBTITLE: "subtitles",
        ObjectType.PARAGRAPH: "paragraphs",
        ObjectType.TEXT_BLOCK: "text_blocks",
        ObjectType.IMAGE: "images",
        ObjectType.PHOTO: "photos",
        ObjectType.LOGO: "logos",
        ObjectType.ICON: "icons",
        ObjectType.RECTANGLE: "rectangles",
        ObjectType.ROUNDED_RECTANGLE: "rounded_rectangles",
        ObjectType.LINE: "lines",
        ObjectType.CIRCLE: "circles",
        ObjectType.ELLIPSE: "ellipses",
        ObjectType.TABLE: "tables",
        ObjectType.LIST: "lists",
        ObjectType.BACKGROUND_SHAPE: "background_shapes",
        ObjectType.DECORATIVE_ELEMENT: "decorative_elements",
        ObjectType.QR_CODE: "qr_codes",
    }
    for obj in objects:
        attr = mapping.get(obj.type)
        if attr:
            setattr(counts, attr, getattr(counts, attr) + 1)
    return counts


def draw_layout_debug(
    image_path: Path,
    objects: list[LayoutObject],
    output_path: Path,
) -> Path:
    image = cv2.imread(str(image_path))
    if image is None:
        pil = Image.open(image_path).convert("RGB")
        image = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

    # Draw larger containers first (low z), then details
    ordered = sorted(objects, key=lambda o: o.z_index)
    skip_types = {ObjectType.PAGE, ObjectType.HEADER, ObjectType.MAIN_CONTENT, ObjectType.FOOTER}

    for obj in ordered:
        if obj.type in skip_types:
            continue
        color = _DEBUG_COLORS.get(obj.type, (128, 128, 128))
        x1, y1, x2, y2 = map(int, obj.bbox)
        thickness = 3 if obj.type in {ObjectType.TITLE, ObjectType.LOGO, ObjectType.PHOTO} else 2
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
        label = obj.type.value
        if obj.text:
            label = f"{label}: {obj.text[:24]}"
        cv2.putText(
            image,
            label,
            (x1, max(y1 - 6, 14)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (20, 20, 20),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            image,
            label,
            (x1, max(y1 - 6, 14)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            color,
            1,
            cv2.LINE_AA,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), image)
    logger.info("Layout debug image saved -> %s", output_path)
    return output_path


def analyze_layout(image_id: str, settings: Settings) -> dict[str, Any]:
    """
    Full layout pipeline for a processed image (+ optional OCR JSON).
    """
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
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    page_area = float(page_w * page_h)

    started = time.perf_counter()

    # --- Text from OCR ---
    ocr_payload = _load_ocr_json(image_id, settings)
    text_dets: list[RawDetection] = []
    text_boxes: list[tuple[float, float, float, float]] = []
    if ocr_payload:
        text_dets = ocr_blocks_to_text_detections(ocr_payload)
        text_dets = merge_paragraph_groups(text_dets)
        text_boxes = [d.bbox for d in text_dets]

    # --- Document mode (hybrid pipeline) ---
    mode_info = ensure_document_mode(image_id, settings, bgr, ocr_payload)
    doc_mode = str(mode_info.get("mode") or "poster")

    # --- Optional PP-StructureV3 tables / regions ---
    structure = run_pp_structure(bgr, enabled=bool(settings.use_pp_structure))
    try:
        (settings.results_path / f"structure_{image_id}.json").write_text(
            json.dumps(structure, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    except Exception:
        pass
    structure_tables: list[RawDetection] = []
    if structure.get("ok"):
        for t in structure.get("tables") or []:
            bbox = t.get("bbox")
            if not bbox or len(bbox) < 4:
                continue
            structure_tables.append(
                RawDetection(
                    kind="table",
                    bbox=(float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])),
                    confidence=0.85,
                    meta={"source": "pp_structure", "has_html": bool(t.get("html"))},
                )
            )
        for r in structure.get("regions") or []:
            label = str(r.get("label") or "").lower()
            bbox = r.get("bbox")
            if not bbox or len(bbox) < 4:
                continue
            if "table" in label and not structure_tables:
                structure_tables.append(
                    RawDetection(
                        kind="table",
                        bbox=(float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])),
                        confidence=0.75,
                        meta={"source": "pp_structure_region", "label": label},
                    )
                )

    # --- Visual detectors ---
    shape_dets = detect_shapes(bgr, gray)
    shape_dets = _suppress_shapes_overlapping_text(shape_dets, text_dets)
    line_dets = detect_lines(gray, page_w, page_h)
    panel_dets = detect_background_panels(bgr)
    image_dets = detect_image_regions(bgr, gray, text_boxes)

    # Deduplicate panels that heavily overlap large rectangles
    filtered_panels: list[RawDetection] = []
    for p in panel_dets:
        if any(
            _iou(p.bbox, s.bbox) > 0.75
            for s in shape_dets
            if s.kind in {"rectangle", "rounded_rectangle"}
        ):
            continue
        filtered_panels.append(p)

    # Drop lines that hug rectangle borders (frame edges already represented)
    rects = [s for s in shape_dets if s.kind in {"rectangle", "rounded_rectangle"}]
    filtered_lines: list[RawDetection] = []
    for ln in line_dets:
        skip = False
        for rect in rects:
            rx1, ry1, rx2, ry2 = rect.bbox
            lx1, ly1, lx2, ly2 = ln.bbox
            # Near any of the four sides
            if (
                abs(lx1 - rx1) < 8
                or abs(lx2 - rx2) < 8
                or abs(ly1 - ry1) < 8
                or abs(ly2 - ry2) < 8
            ) and _iou(ln.bbox, rect.bbox) > 0.01:
                # Only skip if line spans a large portion of that side
                if ln.meta.get("orientation") == "horizontal" and (lx2 - lx1) > (rx2 - rx1) * 0.4:
                    skip = True
                if ln.meta.get("orientation") == "vertical" and (ly2 - ly1) > (ry2 - ry1) * 0.4:
                    skip = True
        if not skip:
            filtered_lines.append(ln)

    all_raw = (
        text_dets
        + shape_dets
        + filtered_lines
        + filtered_panels
        + image_dets
        + structure_tables
    )

    objects: list[LayoutObject] = []
    next_id = 1

    # PAGE root
    page_obj = LayoutObject(
        id=next_id,
        type=ObjectType.PAGE,
        bbox=[0, 0, float(page_w), float(page_h)],
        center_x=page_w / 2,
        center_y=page_h / 2,
        width=float(page_w),
        height=float(page_h),
        area=page_area,
        z_index=0,
        parent=None,
        children=[],
    )
    next_id += 1
    objects.append(page_obj)

    # Full-page background
    bg = LayoutObject(
        id=next_id,
        type=ObjectType.BACKGROUND,
        bbox=[0, 0, float(page_w), float(page_h)],
        center_x=page_w / 2,
        center_y=page_h / 2,
        width=float(page_w),
        height=float(page_h),
        area=page_area,
        z_index=1,
        parent=page_obj.id,
        children=[],
        meta={"role": "page_background"},
    )
    next_id += 1
    objects.append(bg)
    page_obj.children.append(bg.id)

    for det in all_raw:
        obj = _raw_to_object(det, next_id, page_area)
        next_id += 1
        objects.append(obj)

    _assign_parents(objects, page_w, page_h)
    objects, next_id = _add_section_bands(objects, page_w, page_h, next_id)
    # Re-run parent assignment for any orphans created by band insertion edge cases
    # (bands already assigned)

    tree = _build_tree(objects)
    counts = _compute_counts(objects)
    elapsed_ms = (time.perf_counter() - started) * 1000

    shape_count = (
        counts.rectangles
        + counts.rounded_rectangles
        + counts.circles
        + counts.ellipses
        + counts.lines
        + counts.background_shapes
    )
    image_count = counts.images + counts.photos + counts.logos + counts.icons + counts.qr_codes
    text_count = counts.titles + counts.subtitles + counts.paragraphs + counts.text_blocks + counts.lists

    logger.info(
        "Layout complete id=%s mode=%s objects=%d shapes=%d images=%d text=%d "
        "pp_tables=%d time=%.1fms",
        image_id,
        doc_mode,
        counts.total,
        shape_count,
        image_count,
        text_count,
        len(structure_tables),
        elapsed_ms,
    )

    results_path = settings.results_path / f"layout_{image_id}.json"
    debug_path = settings.debug_path / f"layout_{image_id}.png"
    draw_layout_debug(image_path, objects, debug_path)

    payload = {
        "success": True,
        "image_id": image_id,
        "document_mode": doc_mode,
        "document_mode_info": mode_info,
        "structure": {
            "enabled": bool(structure.get("enabled")),
            "ok": bool(structure.get("ok")),
            "table_count": len(structure.get("tables") or []),
            "region_count": len(structure.get("regions") or []),
            "error": structure.get("error"),
        },
        "page": PageInfo(width=page_w, height=page_h).model_dump(),
        "objects": [o.model_dump(mode="json") for o in objects],
        "tree": tree.model_dump(mode="json"),
        "counts": counts.model_dump(),
        "processing_time_ms": round(elapsed_ms, 1),
        "results_file": str(results_path),
        "debug_image": str(debug_path),
        "message": "Layout analysis completed successfully.",
    }

    results_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Layout JSON saved -> %s", results_path)
    return payload
