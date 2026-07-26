"""
Reconstruction planner — decide how each detected object should be rebuilt.

No PDF / SVG / CDR export. Planning + debug overlay only.
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
from app.reconstruction.models import (
    PlannedObject,
    ReconstructionCounts,
    ReconstructionSummary,
    ReconstructionType,
)
from app.reconstruction.rules import (
    LAYER_BACKGROUND,
    LAYER_IMAGES,
    LAYER_PANELS,
    LAYER_SHAPES,
    LAYER_TEXT,
    decide_reconstruction,
    is_image_type,
    is_vector_type,
)
from app.reconstruction.segmentation import prepare_objects_for_planning

logger = logging.getLogger(__name__)

# Debug overlay colors (OpenCV BGR)
# Green=text, Blue=vector, Purple=image, Yellow=logo, Orange=background
_DEBUG = {
    ReconstructionType.TEXT: (0, 200, 0),
    ReconstructionType.VECTOR_RECTANGLE: (255, 90, 20),
    ReconstructionType.VECTOR_ROUNDED_RECTANGLE: (255, 90, 20),
    ReconstructionType.VECTOR_LINE: (255, 90, 20),
    ReconstructionType.VECTOR_CIRCLE: (255, 90, 20),
    ReconstructionType.VECTOR_ELLIPSE: (255, 90, 20),
    ReconstructionType.VECTOR_POLYGON: (255, 90, 20),
    ReconstructionType.VECTOR_PATH: (255, 90, 20),
    ReconstructionType.IMAGE: (180, 0, 180),
    ReconstructionType.PHOTO_IMAGE: (180, 0, 180),
    ReconstructionType.ICON_IMAGE: (180, 0, 180),
    ReconstructionType.LOGO_IMAGE: (0, 220, 255),
    ReconstructionType.BACKGROUND_IMAGE: (0, 140, 255),
    ReconstructionType.IGNORE: (120, 120, 120),
}


def _load_json(path: Path) -> Optional[dict[str, Any]]:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        logger.warning("Failed reading %s", path)
        return None


def _typography_index(typo: Optional[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    if not typo:
        return out
    for style in typo.get("text_styles") or []:
        oid = style.get("ocr_block_id")
        if oid is not None:
            out[int(oid)] = style
    return out


def _quad_to_bbox(bbox: Any) -> Optional[list[float]]:
    if not bbox:
        return None
    if len(bbox) == 4 and all(isinstance(v, (int, float)) for v in bbox):
        return [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])]
    try:
        xs = [float(p[0]) for p in bbox]
        ys = [float(p[1]) for p in bbox]
        return [min(xs), min(ys), max(xs), max(ys)]
    except Exception:
        return None


def _covered_ocr_ids(objects: list[dict[str, Any]]) -> set[int]:
    covered: set[int] = set()
    for obj in objects:
        for oid in obj.get("ocr_block_ids") or []:
            try:
                covered.add(int(oid))
            except (TypeError, ValueError):
                continue
    return covered


def _inject_missing_ocr_text(
    objects: list[dict[str, Any]],
    ocr: Optional[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Ensure every OCR text block becomes a planning candidate (always editable).
    Layout may merge or drop some blocks; reconstruction still plans them as TEXT.
    """
    if not ocr:
        return objects
    blocks = ocr.get("text_blocks") or []
    if not blocks:
        return objects

    covered = _covered_ocr_ids(objects)
    next_id = max((int(o.get("id") or 0) for o in objects), default=0) + 1
    out = list(objects)

    for block in blocks:
        bid = block.get("id")
        if bid is None:
            continue
        bid = int(bid)
        if bid in covered:
            continue
        box = _quad_to_bbox(block.get("bbox"))
        if not box:
            continue
        conf = float(block.get("confidence") or 0.95)
        if conf <= 1.0:
            conf *= 100.0
        out.append(
            {
                "id": next_id,
                "type": "TEXT_BLOCK",
                "bbox": box,
                "width": box[2] - box[0],
                "height": box[3] - box[1],
                "area": max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1]),
                "confidence": conf / 100.0,
                "text": block.get("text"),
                "ocr_block_ids": [bid],
                "z_index": 50,
                "_merged_from": [],
                "meta": {"from_ocr_gapfill": True},
            }
        )
        next_id += 1
        covered.add(bid)
    return out


def _expand_merged_text_objects(
    objects: list[dict[str, Any]],
    ocr: Optional[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Split LIST/PARAGRAPH/TITLE mega-blocks that cover multiple OCR lines into
    one TEXT_BLOCK per OCR line. Prevents overlapping multi-line PDF text boxes.
    """
    if not ocr:
        return objects
    ocr_by_id: dict[int, dict[str, Any]] = {}
    for b in ocr.get("text_blocks") or []:
        if b.get("id") is not None:
            ocr_by_id[int(b["id"])] = b

    text_types = {
        "TEXT",
        "TEXT_BLOCK",
        "TITLE",
        "SUBTITLE",
        "HEADING",
        "PARAGRAPH",
        "LIST",
        "CAPTION",
        "LABEL",
    }
    out: list[dict[str, Any]] = []
    next_id = max((int(o.get("id") or 0) for o in objects), default=0) + 1

    for obj in objects:
        otype = str(obj.get("type") or "")
        ocr_ids = [int(x) for x in (obj.get("ocr_block_ids") or []) if x is not None]
        if otype not in text_types or len(ocr_ids) <= 1:
            out.append(obj)
            continue

        # Keep structural LIST/PARAGRAPH as IGNORE shell; emit one block per OCR id
        shell = dict(obj)
        shell["meta"] = dict(shell.get("meta") or {})
        shell["meta"]["expanded_to_ocr"] = True
        shell["_force_ignore"] = True
        out.append(shell)

        for oid in ocr_ids:
            block = ocr_by_id.get(oid)
            if not block:
                continue
            box = _quad_to_bbox(block.get("bbox"))
            if not box:
                continue
            conf = float(block.get("confidence") or 0.95)
            if conf <= 1.0:
                conf *= 100.0
            out.append(
                {
                    "id": next_id,
                    "type": "TEXT_BLOCK",
                    "bbox": box,
                    "width": box[2] - box[0],
                    "height": box[3] - box[1],
                    "area": max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1]),
                    "confidence": conf / 100.0,
                    "text": block.get("text"),
                    "ocr_block_ids": [oid],
                    "z_index": int(obj.get("z_index") or 50),
                    "_merged_from": [int(obj.get("id") or 0)],
                    "meta": {"from_ocr_expand": True, "parent_layout_type": otype},
                }
            )
            next_id += 1

    return out


def plan_reconstruction(image_id: str, settings: Settings) -> dict[str, Any]:
    image_path = resolve_processed_image(image_id, settings)

    try:
        with Image.open(image_path) as img:
            img.load()
            page_w, page_h = img.size
    except Exception as exc:
        raise ValueError(f"Corrupted or unreadable image: {exc}") from exc

    layout = _load_json(settings.results_path / f"layout_{image_id}.json")
    if not layout or not layout.get("objects"):
        raise FileNotFoundError(
            f"Layout results not found for image_id={image_id}. Run /api/layout first."
        )

    ocr = _load_json(settings.results_path / f"{image_id}.json")
    typo = _load_json(settings.results_path / f"typography_{image_id}.json")
    typo_by_ocr = _typography_index(typo)

    started = time.perf_counter()

    raw_objects = list(layout["objects"])
    raw_objects = _expand_merged_text_objects(raw_objects, ocr)
    raw_objects = _inject_missing_ocr_text(raw_objects, ocr)
    prepared = prepare_objects_for_planning(raw_objects)

    planned: list[PlannedObject] = []
    plan_id = 1
    text_boxes: list[tuple[float, float, float, float]] = []
    for b in (ocr or {}).get("text_blocks") or []:
        box = _quad_to_bbox(b.get("bbox"))
        if box:
            text_boxes.append((box[0], box[1], box[2], box[3]))

    for obj in prepared:
        if obj.get("_force_ignore"):
            rtype, confidence, reason, layer = (
                ReconstructionType.IGNORE,
                0.99,
                "Expanded multi-OCR text shell",
                0,
            )
        else:
            rtype, confidence, reason, layer = decide_reconstruction(
                obj, typo_by_ocr, text_boxes=text_boxes
            )
        bbox = obj.get("bbox") or [0, 0, 0, 0]
        if len(bbox) < 4:
            continue
        # Skip IGNORE structural noise in output? User wants IGNORE in decisions — keep them
        planned.append(
            PlannedObject(
                id=plan_id,
                source_id=int(obj.get("id") or 0),
                type=str(obj.get("type") or "UNKNOWN"),
                reconstruction=rtype,
                layer=int(layer),
                confidence=round(float(confidence), 2),
                bbox=[float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])],
                reason=reason,
                merged_from=[int(x) for x in (obj.get("_merged_from") or []) if x],
                meta={
                    "text": obj.get("text"),
                    "ocr_block_ids": obj.get("ocr_block_ids") or [],
                    "z_index_layout": obj.get("z_index"),
                },
            )
        )
        plan_id += 1

    # Stable layer ordering: background → panels → shapes → images → text
    # Refine: BACKGROUND_IMAGE at layer 1, panel vectors at 2, etc. already set
    planned.sort(key=lambda p: (p.layer, p.bbox[1], p.bbox[0], p.id))

    # Re-number for reading clarity after sort? Keep ids stable for debug — reassign sequential
    for i, p in enumerate(planned, start=1):
        p.id = i

    summary = _build_summary(planned)
    elapsed_ms = (time.perf_counter() - started) * 1000

    logger.info(
        "Reconstruction plan id=%s objects=%d text=%d vectors=%d images=%d bg=%d paths=%d ignore=%d "
        "avg_conf=%.1f score=%.1f time=%.1fms",
        image_id,
        summary.counts.total,
        summary.counts.editable_text,
        summary.counts.vector_shapes,
        summary.counts.embedded_images,
        summary.counts.background_regions,
        summary.counts.svg_paths,
        summary.counts.ignored,
        summary.average_confidence,
        summary.overall_score,
        elapsed_ms,
    )
    logger.info("Decision breakdown: %s", summary.decision_breakdown)

    results_path = settings.results_path / f"reconstruction_{image_id}.json"
    debug_path = settings.debug_path / f"reconstruction_{image_id}.png"
    _draw_debug(image_path, planned, debug_path)

    payload = {
        "success": True,
        "image_id": image_id,
        "page": {"width": page_w, "height": page_h},
        "objects": [p.model_dump(mode="json") for p in planned],
        "summary": summary.model_dump(mode="json"),
        "processing_time_ms": round(elapsed_ms, 1),
        "results_file": str(results_path),
        "debug_image": str(debug_path),
        "message": "Reconstruction plan completed successfully.",
        # Convenience for OCR presence logging
        "inputs": {
            "has_ocr": bool(ocr),
            "has_typography": bool(typo),
            "layout_objects_in": len(raw_objects),
            "objects_after_merge": len(prepared),
        },
    }
    # Strip inputs from response model — keep in file only? User schema doesn't include inputs.
    # Save full payload then return response-compatible dict
    results_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Reconstruction JSON saved -> %s", results_path)

    return {
        "success": True,
        "image_id": image_id,
        "page": {"width": page_w, "height": page_h},
        "objects": [p.model_dump(mode="json") for p in planned],
        "summary": summary.model_dump(mode="json"),
        "processing_time_ms": round(elapsed_ms, 1),
        "results_file": str(results_path),
        "debug_image": str(debug_path),
        "message": "Reconstruction plan completed successfully.",
    }


def _build_summary(planned: list[PlannedObject]) -> ReconstructionSummary:
    counts = ReconstructionCounts(total=len(planned))
    breakdown: dict[str, int] = {}
    layer_stats: dict[str, int] = {}
    confs: list[float] = []

    for p in planned:
        key = p.reconstruction.value
        breakdown[key] = breakdown.get(key, 0) + 1
        layer_stats[str(p.layer)] = layer_stats.get(str(p.layer), 0) + 1
        confs.append(p.confidence)

        if p.reconstruction == ReconstructionType.TEXT:
            counts.editable_text += 1
        elif p.reconstruction == ReconstructionType.VECTOR_PATH:
            counts.svg_paths += 1
            counts.vector_shapes += 1
        elif is_vector_type(p.reconstruction):
            counts.vector_shapes += 1
            if p.layer <= LAYER_PANELS and p.type in {"BACKGROUND_SHAPE", "BACKGROUND"}:
                counts.background_regions += 1
        elif p.reconstruction == ReconstructionType.BACKGROUND_IMAGE:
            counts.background_regions += 1
            counts.embedded_images += 1
        elif is_image_type(p.reconstruction):
            counts.embedded_images += 1
        elif p.reconstruction == ReconstructionType.IGNORE:
            counts.ignored += 1

    avg_conf = float(np.mean(confs)) if confs else 0.0

    # Overall score: weighted readiness
    drawable = max(1, counts.total - counts.ignored)
    text_ok = counts.editable_text / drawable
    vector_ok = counts.vector_shapes / drawable
    # Prefer high confidence
    score = 100.0 * (0.45 * min(1.0, text_ok * 3) + 0.25 * min(1.0, vector_ok * 2) + 0.30 * (avg_conf / 100.0))
    # Clamp and boost if text coverage is strong
    if counts.editable_text > 0:
        score = max(score, 85.0)
    score = float(np.clip(score, 0, 100))

    return ReconstructionSummary(
        counts=counts,
        average_confidence=round(avg_conf, 2),
        overall_score=round(score, 1),
        decision_breakdown=breakdown,
        layer_stats=layer_stats,
    )


def _draw_debug(image_path: Path, planned: list[PlannedObject], output_path: Path) -> None:
    image = cv2.imread(str(image_path))
    if image is None:
        pil = Image.open(image_path).convert("RGB")
        image = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

    # Draw low layers first
    for p in sorted(planned, key=lambda x: x.layer):
        if p.reconstruction == ReconstructionType.IGNORE:
            continue
        color = _DEBUG.get(p.reconstruction, (128, 128, 128))
        x1, y1, x2, y2 = map(int, p.bbox)
        thickness = 3 if p.reconstruction == ReconstructionType.TEXT else 2
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
        label = f"{p.reconstruction.value} L{p.layer}"
        cv2.putText(
            image,
            label[:40],
            (x1, max(14, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (15, 15, 15),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            image,
            label[:40],
            (x1, max(14, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            color,
            1,
            cv2.LINE_AA,
        )

    # Legend strip
    legend = [
        ("TEXT", (0, 200, 0)),
        ("VECTOR", (255, 90, 20)),
        ("IMAGE", (180, 0, 180)),
        ("LOGO", (0, 220, 255)),
        ("BG", (0, 140, 255)),
    ]
    x = 10
    for name, col in legend:
        cv2.rectangle(image, (x, 10), (x + 14, 24), col, -1)
        cv2.putText(image, name, (x + 18, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (30, 30, 30), 1, cv2.LINE_AA)
        x += 70

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), image)
    logger.info("Reconstruction debug image saved -> %s", output_path)
