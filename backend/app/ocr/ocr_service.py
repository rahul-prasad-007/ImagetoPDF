"""
OCR service: lazy-loaded PaddleOCR singleton, text grouping, debug visualization.

Scope: OCR + metadata only (no PDF, fonts, layout recreation, or background work).
"""

from __future__ import annotations

import json
import logging
import math
import threading
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from app.config import Settings
from app.ocr.models import PageInfo, TextBlock
from app.ocr.script_detect import choose_ocr_lang, ocr_blocks_look_like_wrong_script

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Singleton OCR engines (one per language, load once)
# ---------------------------------------------------------------------------
_ocr_engines: dict[str, Any] = {}
_ocr_lock = threading.Lock()
_LOW_CONFIDENCE_THRESHOLD = 0.60

# Explicit recognition models for stable Windows CPU deploy
_ENGINE_SPECS: dict[str, dict[str, str]] = {
    "en": {
        "lang": "en",
        "det": "PP-OCRv5_mobile_det",
        "rec": "en_PP-OCRv5_mobile_rec",
    },
    "hi": {
        "lang": "hi",
        # Devanagari recognizer; det can stay mobile for speed
        "det": "PP-OCRv5_mobile_det",
        "rec": "devanagari_PP-OCRv5_mobile_rec",
    },
}


def get_ocr_engine(lang: str = "en") -> Any:
    """
    Lazily initialize and return a shared PaddleOCR instance for `lang`.

    Thread-safe. Models load only on first use per language.
    """
    key = (lang or "en").strip().lower()
    if key not in _ENGINE_SPECS:
        key = "en"
    if key in _ocr_engines:
        return _ocr_engines[key]

    with _ocr_lock:
        if key in _ocr_engines:
            return _ocr_engines[key]

        logger.info("Loading PaddleOCR model lang=%s (first request)...", key)
        started = time.perf_counter()

        import os

        os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

        from paddleocr import PaddleOCR

        spec = _ENGINE_SPECS[key]
        engine = PaddleOCR(
            lang=spec["lang"],
            text_detection_model_name=spec["det"],
            text_recognition_model_name=spec["rec"],
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
        )
        _ocr_engines[key] = engine

        elapsed = (time.perf_counter() - started) * 1000
        logger.info("PaddleOCR lang=%s loaded in %.1fms", key, elapsed)
        return engine


def _normalize_page_data(page: Any) -> Any:
    """Unwrap paddlex OCRResult / nested {'res': ...} into a dict with rec_texts."""
    data = page
    if hasattr(page, "json"):
        raw = page.json
        try:
            data = raw() if callable(raw) else raw
        except Exception:
            data = page
    if isinstance(data, dict) and "res" in data and isinstance(data["res"], dict):
        data = data["res"]
    # OCRResult is dict-like
    if not isinstance(data, dict) and hasattr(page, "get"):
        try:
            if page.get("rec_texts") is not None or page.get("dt_polys") is not None:
                return page
            nested = page.get("res")
            if isinstance(nested, dict):
                return nested
        except Exception:
            pass
    return data


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------
def _as_quad(points: Any) -> list[list[float]]:
    """Normalize a polygon / box into [[x,y] x4]."""
    arr = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if arr.shape[0] < 4:
        # Axis-aligned fallback from min/max
        x_min, y_min = arr.min(axis=0)
        x_max, y_max = arr.max(axis=0)
        return [
            [float(x_min), float(y_min)],
            [float(x_max), float(y_min)],
            [float(x_max), float(y_max)],
            [float(x_min), float(y_max)],
        ]
    return [[float(x), float(y)] for x, y in arr[:4]]


def _bbox_metrics(bbox: list[list[float]]) -> tuple[float, float, float, float, float]:
    """
    Returns center_x, center_y, width, height, rotation_degrees.
    Rotation is the angle of the top edge (p0→p1).
    """
    pts = np.asarray(bbox, dtype=np.float64)
    center_x = float(pts[:, 0].mean())
    center_y = float(pts[:, 1].mean())

    # Width / height from edge lengths (average opposite sides)
    w1 = float(np.linalg.norm(pts[1] - pts[0]))
    w2 = float(np.linalg.norm(pts[2] - pts[3]))
    h1 = float(np.linalg.norm(pts[3] - pts[0]))
    h2 = float(np.linalg.norm(pts[2] - pts[1]))
    width = (w1 + w2) / 2.0
    height = (h1 + h2) / 2.0

    dx = pts[1][0] - pts[0][0]
    dy = pts[1][1] - pts[0][1]
    rotation = float(math.degrees(math.atan2(dy, dx)))

    return center_x, center_y, width, height, rotation


# ---------------------------------------------------------------------------
# Parse PaddleOCR 3.x / 2.x result shapes
# ---------------------------------------------------------------------------
def _extract_raw_detections(raw_result: Any) -> list[tuple[list[list[float]], str, float, float]]:
    """
    Normalize OCR engine output into a list of (bbox, text, confidence, rotation_hint).

    Supports:
      - PaddleOCR 3.x predict() → list of result objects / dicts with rec_* fields
      - Legacy ocr() → [[[box], (text, score)], ...]
    """
    detections: list[tuple[list[list[float]], str, float, float]] = []

    if raw_result is None:
        return detections

    # Unwrap single-page list wrappers
    pages = raw_result if isinstance(raw_result, list) else [raw_result]

    for page in pages:
        if page is None:
            continue

        data = _normalize_page_data(page)

        if isinstance(data, dict) and ("rec_texts" in data or "dt_polys" in data or "rec_polys" in data):
            texts = list(data.get("rec_texts") or [])
            scores = list(data.get("rec_scores") or [])
            polys = data.get("rec_polys")
            if polys is None:
                polys = data.get("dt_polys")
            polys = list(polys) if polys is not None else []
            angles = list(data.get("textline_orientation_angles") or [])

            count = max(len(texts), len(polys), len(scores))
            for i in range(count):
                text = str(texts[i]).strip() if i < len(texts) else ""
                if not text:
                    continue
                score = float(scores[i]) if i < len(scores) else 0.0
                poly = polys[i] if i < len(polys) else [[0, 0], [0, 0], [0, 0], [0, 0]]
                bbox = _as_quad(poly)
                angle = float(angles[i]) if i < len(angles) and angles[i] is not None else None
                if angle is not None and angle >= 0:
                    rotation_hint = float(angle)
                else:
                    rotation_hint = _bbox_metrics(bbox)[4]
                detections.append((bbox, text, score, rotation_hint))
            continue

        # Attribute-style result objects (paddlex OCRResult)
        if hasattr(page, "rec_texts") and not isinstance(page, dict):
            texts = list(getattr(page, "rec_texts") or [])
            scores = list(getattr(page, "rec_scores", []) or [])
            polys = getattr(page, "rec_polys", None)
            if polys is None:
                polys = getattr(page, "dt_polys", [])
            polys = list(polys or [])
            angles = list(getattr(page, "textline_orientation_angles", []) or [])
            for i, text in enumerate(texts):
                text = str(text).strip()
                if not text:
                    continue
                score = float(scores[i]) if i < len(scores) else 0.0
                poly = polys[i] if i < len(polys) else [[0, 0], [0, 0], [0, 0], [0, 0]]
                bbox = _as_quad(poly)
                angle = float(angles[i]) if i < len(angles) and angles[i] is not None else None
                rotation_hint = float(angle) if angle is not None and angle >= 0 else _bbox_metrics(bbox)[4]
                detections.append((bbox, text, score, rotation_hint))
            continue

        # Legacy: page is a list of [box, (text, score)]
        if isinstance(page, (list, tuple)):
            for item in page:
                if item is None:
                    continue
                try:
                    box, rec = item
                    if isinstance(rec, (list, tuple)):
                        text, score = rec[0], float(rec[1])
                    else:
                        text, score = str(rec), 0.0
                    text = str(text).strip()
                    if not text:
                        continue
                    bbox = _as_quad(box)
                    detections.append((bbox, text, float(score), _bbox_metrics(bbox)[4]))
                except Exception:
                    logger.debug("Skipping unparsable OCR item: %r", item, exc_info=True)
                    continue

    return detections


# ---------------------------------------------------------------------------
# Line / paragraph grouping (reading order: top→bottom, left→right)
# ---------------------------------------------------------------------------
def group_text_blocks(
    detections: list[tuple[list[list[float]], str, float, float]],
) -> list[TextBlock]:
    """
    Sort detections into reading order and assign line / word / paragraph indices.
    """
    if not detections:
        return []

    enriched: list[dict[str, Any]] = []
    for bbox, text, conf, rotation in detections:
        cx, cy, w, h, rot = _bbox_metrics(bbox)
        enriched.append(
            {
                "bbox": bbox,
                "text": text,
                "confidence": conf,
                "center_x": cx,
                "center_y": cy,
                "width": w,
                "height": h,
                "rotation": rotation if rotation is not None else rot,
            }
        )

    # Sort top→bottom, then left→right
    enriched.sort(key=lambda d: (d["center_y"], d["center_x"]))

    heights = [d["height"] for d in enriched if d["height"] > 0]
    median_h = float(np.median(heights)) if heights else 20.0
    line_thresh = max(median_h * 0.6, 8.0)

    # Cluster into lines by Y proximity
    lines: list[list[dict[str, Any]]] = []
    for item in enriched:
        placed = False
        for line in lines:
            line_y = float(np.mean([x["center_y"] for x in line]))
            if abs(item["center_y"] - line_y) <= line_thresh:
                line.append(item)
                placed = True
                break
        if not placed:
            lines.append([item])

    # Sort words in each line left→right; sort lines by average Y
    for line in lines:
        line.sort(key=lambda d: d["center_x"])
    lines.sort(key=lambda line: float(np.mean([x["center_y"] for x in line])))

    # Paragraph grouping: large vertical gaps between consecutive lines
    line_gaps: list[float] = []
    line_ys = [float(np.mean([x["center_y"] for x in line])) for line in lines]
    line_hs = [float(np.mean([x["height"] for x in line])) for line in lines]
    for i in range(1, len(lines)):
        gap = line_ys[i] - line_ys[i - 1] - (line_hs[i - 1] / 2 + line_hs[i] / 2)
        line_gaps.append(max(gap, 0.0))

    para_thresh = max(median_h * 1.4, 18.0)
    paragraph_ids: list[int] = []
    current_para = 1
    for i in range(len(lines)):
        if i > 0 and line_gaps[i - 1] > para_thresh:
            current_para += 1
        paragraph_ids.append(current_para)

    blocks: list[TextBlock] = []
    block_id = 1
    for line_idx, line in enumerate(lines):
        for word_idx, item in enumerate(line):
            blocks.append(
                TextBlock(
                    id=block_id,
                    text=item["text"],
                    confidence=round(float(item["confidence"]), 4),
                    bbox=[[round(p[0], 2), round(p[1], 2)] for p in item["bbox"]],
                    center_x=round(float(item["center_x"]), 2),
                    center_y=round(float(item["center_y"]), 2),
                    width=round(float(item["width"]), 2),
                    height=round(float(item["height"]), 2),
                    rotation=round(float(item["rotation"]), 2),
                    line=line_idx + 1,
                    word=word_idx + 1,
                    paragraph=paragraph_ids[line_idx],
                )
            )
            block_id += 1

    return blocks


# ---------------------------------------------------------------------------
# Debug visualization
# ---------------------------------------------------------------------------
def draw_debug_image(
    image_path: Path,
    blocks: list[TextBlock],
    output_path: Path,
) -> Path:
    """Draw bounding boxes, text, and confidence on a copy of the image."""
    image = cv2.imread(str(image_path))
    if image is None:
        # Fallback via Pillow for formats OpenCV struggles with
        pil = Image.open(image_path).convert("RGB")
        image = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

    for block in blocks:
        pts = np.array(block.bbox, dtype=np.int32).reshape((-1, 1, 2))
        color = (16, 185, 129) if block.confidence >= _LOW_CONFIDENCE_THRESHOLD else (68, 68, 239)
        cv2.polylines(image, [pts], isClosed=True, color=color, thickness=2)

        label = f"{block.text[:28]} ({block.confidence:.2f})"
        x, y = int(block.bbox[0][0]), max(int(block.bbox[0][1]) - 6, 14)
        cv2.putText(
            image,
            label,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (15, 23, 42),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            image,
            label,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), image)
    logger.info("Debug OCR image saved -> %s", output_path)
    return output_path


def save_ocr_json(payload: dict[str, Any], output_path: Path) -> Path:
    """Persist OCR JSON to results/."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("OCR results saved -> %s", output_path)
    return output_path


# ---------------------------------------------------------------------------
# Public pipeline
# ---------------------------------------------------------------------------
def resolve_processed_image(image_id: str, settings: Settings) -> Path:
    """Locate the processed image for a given upload UUID."""
    # Sanitize: only hex uuid chars
    safe_id = "".join(c for c in image_id if c.isalnum())
    if len(safe_id) < 8:
        raise FileNotFoundError("Invalid image_id")

    candidates = [
        settings.processed_path / f"{safe_id}_processed.png",
        *settings.processed_path.glob(f"{safe_id}*_processed.png"),
        *settings.processed_path.glob(f"{safe_id}*.png"),
        *settings.uploads_path.glob(f"{safe_id}.*"),
    ]
    for path in candidates:
        if path.is_file():
            return path

    raise FileNotFoundError(f"Processed image not found for image_id={image_id}")


def run_ocr(image_id: str, settings: Settings) -> dict[str, Any]:
    """
    Run OCR on a processed image, group text, save JSON + debug image.

    Returns a dict matching OcrSuccessResponse fields (plus internal paths).
    """
    image_path = resolve_processed_image(image_id, settings)

    try:
        with Image.open(image_path) as img:
            img.load()
            page_w, page_h = img.size
    except Exception as exc:
        raise ValueError(f"Corrupted or unreadable image: {exc}") from exc

    preferred = choose_ocr_lang(getattr(settings, "ocr_lang", "auto") or "auto")
    lang_setting = (getattr(settings, "ocr_lang", "auto") or "auto").strip().lower()
    started = time.perf_counter()

    def _infer(lang: str):
        engine = get_ocr_engine(lang)
        if hasattr(engine, "predict"):
            return engine.predict(str(image_path))
        return engine.ocr(str(image_path))

    try:
        raw = _infer(preferred)
        detections = _extract_raw_detections(raw)
        probe_blocks = [{"text": t, "confidence": c} for _, t, c, _ in detections]
        avg_probe = (
            sum(float(b["confidence"]) for b in probe_blocks) / max(1, len(probe_blocks))
            if probe_blocks
            else 0.0
        )
        # Auto mode: English OCR that looks like Devanagari garbage → Hindi
        if preferred == "en" and lang_setting in {"auto", "", "detect"} and (
            ocr_blocks_look_like_wrong_script(probe_blocks)
            or (avg_probe < 0.65 and len(probe_blocks) >= 8)
        ):
            logger.info(
                "OCR auto-switch en→hi for image_id=%s (wrong-script/low-conf heuristic)",
                image_id,
            )
            raw = _infer("hi")
            detections = _extract_raw_detections(raw)
            preferred = "hi"
    except Exception as exc:
        logger.exception("PaddleOCR inference failed for %s", image_id)
        raise RuntimeError(f"OCR failure: {exc}") from exc

    # Drop watermark / diagonal noise (rotation far from axis, weak score)
    filtered: list[tuple[list[list[float]], str, float, float]] = []
    for bbox, text, score, rot in detections:
        ang = abs(float(rot or 0.0)) % 180.0
        near_axis = min(ang, abs(ang - 90), abs(ang - 180)) <= 12.0
        if not near_axis and score < 0.92:
            continue
        # Very short Latin crumbs on Devanagari pages
        if preferred == "hi" and len(text) <= 2 and not any("\u0900" <= ch <= "\u097F" for ch in text):
            if score < 0.95:
                continue
        filtered.append((bbox, text, score, rot))
    detections = filtered

    ocr_ms = (time.perf_counter() - started) * 1000
    blocks = group_text_blocks(detections)

    avg_conf = (
        float(sum(b.confidence for b in blocks) / len(blocks)) if blocks else 0.0
    )

    warning = None
    if not blocks:
        warning = "No text found in the image."
    elif avg_conf < _LOW_CONFIDENCE_THRESHOLD:
        warning = (
            f"Low average confidence ({avg_conf:.3f}). "
            "OCR results may be unreliable for this image."
        )

    logger.info(
        "OCR complete id=%s lang=%s blocks=%d avg_conf=%.4f time=%.1fms",
        image_id,
        preferred,
        len(blocks),
        avg_conf,
        ocr_ms,
    )

    debug_name = f"{image_id}_ocr_debug.png"
    debug_path = settings.debug_path / debug_name
    draw_debug_image(image_path, blocks, debug_path)

    payload = {
        "success": True,
        "image_id": image_id,
        "ocr_lang": preferred,
        "page": PageInfo(width=page_w, height=page_h).model_dump(),
        "text_blocks": [b.model_dump() for b in blocks],
        "total_blocks": len(blocks),
        "average_confidence": round(avg_conf, 4),
        "processing_time_ms": round(ocr_ms, 1),
        "debug_image": str(debug_path),
        "results_file": str(settings.results_path / f"{image_id}.json"),
        "message": "OCR completed successfully." if blocks else "OCR finished — no text detected.",
        "warning": warning,
    }

    save_ocr_json(payload, settings.results_path / f"{image_id}.json")
    return payload
