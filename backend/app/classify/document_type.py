"""
Heuristic document-type classifier (free, CPU-only, no model download).

Modes drive the hybrid pipeline:
  - ruled_form       → bill books / cash memos (vector lattice + logo crop)
  - designed_invoice → colored modern invoices (color fills + underlays + tables)
  - poster           → posters / poems / flyers / cards (text + ornaments)
"""

from __future__ import annotations

import json
import logging
from enum import Enum
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

from app.config import Settings

logger = logging.getLogger(__name__)


class DocumentMode(str, Enum):
    RULED_FORM = "ruled_form"
    DESIGNED_INVOICE = "designed_invoice"
    POSTER = "poster"


_FORM_KEYS = (
    "particulars",
    "quantity",
    "rate",
    "amount",
    "cash memo",
    "bill/cash",
    "bill / cash",
    "sl.no",
    "sl no",
    "s.no",
    "lorry",
    "proprietor",
    "gstin",
    "gst no",
)

_INVOICE_KEYS = (
    "invoice",
    "tax invoice",
    "bill to",
    "ship to",
    "subtotal",
    "due date",
    "payment",
    "unit price",
    "qty",
    "description",
    "balance due",
    "invoice #",
    "invoice no",
)

_POSTER_KEYS = (
    "poem",
    "wedding",
    "save the date",
    "rsvp",
    "certificate",
    "award",
    "congratulations",
)


def _ocr_blob(ocr: Optional[dict[str, Any]]) -> str:
    parts: list[str] = []
    for b in (ocr or {}).get("text_blocks") or []:
        t = str(b.get("text") or "").strip().lower()
        if t:
            parts.append(t)
    return " ".join(parts)


def _keyword_hits(blob: str, keys: tuple[str, ...]) -> int:
    return sum(1 for k in keys if k in blob)


def _colorfulness(bgr: np.ndarray) -> float:
    """Simple Hasler-Susstrunk-ish colorfulness on a downscaled image."""
    small = cv2.resize(bgr, (0, 0), fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
    b, g, r = cv2.split(small.astype(np.float32))
    rg = np.abs(r - g)
    yb = np.abs(0.5 * (r + g) - b)
    return float(np.sqrt(rg.std() ** 2 + yb.std() ** 2) + 0.3 * np.sqrt(rg.mean() ** 2 + yb.mean() ** 2))


def _line_density(bgr: np.ndarray) -> float:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    h, w = gray.shape[:2]
    hk = cv2.getStructuringElement(cv2.MORPH_RECT, (max(20, w // 40), 1))
    vk = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(20, h // 40)))
    horiz = cv2.morphologyEx(edges, cv2.MORPH_OPEN, hk)
    vert = cv2.morphologyEx(edges, cv2.MORPH_OPEN, vk)
    return float((np.count_nonzero(horiz) + np.count_nonzero(vert)) / max(1, h * w))


def classify_document(
    bgr: np.ndarray,
    ocr: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """
    Return mode + confidence + signals. Never raises.
    """
    blob = _ocr_blob(ocr)
    form_hits = _keyword_hits(blob, _FORM_KEYS)
    invoice_hits = _keyword_hits(blob, _INVOICE_KEYS)
    poster_hits = _keyword_hits(blob, _POSTER_KEYS)

    color = 0.0
    lines = 0.0
    try:
        color = _colorfulness(bgr)
        lines = _line_density(bgr)
    except Exception as exc:
        logger.warning("classify visual signals failed: %s", exc)

    # Scores
    scores = {
        DocumentMode.RULED_FORM.value: 0.0,
        DocumentMode.DESIGNED_INVOICE.value: 0.0,
        DocumentMode.POSTER.value: 0.0,
    }

    scores[DocumentMode.RULED_FORM.value] += form_hits * 18.0
    scores[DocumentMode.RULED_FORM.value] += min(35.0, lines * 900.0)
    if form_hits >= 3:
        scores[DocumentMode.RULED_FORM.value] += 25.0
    # Monochrome ruled pages
    if color < 18 and form_hits >= 2:
        scores[DocumentMode.RULED_FORM.value] += 15.0

    scores[DocumentMode.DESIGNED_INVOICE.value] += invoice_hits * 16.0
    scores[DocumentMode.DESIGNED_INVOICE.value] += min(30.0, color * 0.9)
    if invoice_hits >= 2 and color >= 20:
        scores[DocumentMode.DESIGNED_INVOICE.value] += 28.0
    # Colorful page with some invoice words but weak lattice
    if color >= 28 and invoice_hits >= 1 and form_hits < 4:
        scores[DocumentMode.DESIGNED_INVOICE.value] += 12.0
    # Colorful table-ish without classic bill-book headers
    if color >= 35 and lines > 0.012 and form_hits < 3 and poster_hits == 0:
        scores[DocumentMode.DESIGNED_INVOICE.value] += 10.0

    scores[DocumentMode.POSTER.value] += poster_hits * 20.0
    if form_hits == 0 and invoice_hits == 0:
        scores[DocumentMode.POSTER.value] += 28.0
    if color >= 15 and lines < 0.008 and form_hits < 2 and invoice_hits < 2:
        scores[DocumentMode.POSTER.value] += 12.0
    # Poems / cards: fewer text blocks, larger fonts (approx via block count)
    n_blocks = len((ocr or {}).get("text_blocks") or [])
    if 3 <= n_blocks <= 40 and form_hits < 2 and invoice_hits < 2:
        scores[DocumentMode.POSTER.value] += 14.0
    # Color alone should not force designed_invoice without invoice vocabulary
    if invoice_hits == 0 and form_hits == 0:
        scores[DocumentMode.DESIGNED_INVOICE.value] *= 0.35
        scores[DocumentMode.RULED_FORM.value] *= 0.45

    mode = max(scores, key=scores.get)
    best = scores[mode]
    second = sorted(scores.values(), reverse=True)[1] if len(scores) > 1 else 0.0
    confidence = float(np.clip(50.0 + (best - second) * 0.8 + min(best, 40) * 0.3, 45.0, 99.0))

    # Safety: classic bill-book headers win over designed_invoice
    if form_hits >= 3 and ("particulars" in blob or "cash memo" in blob or "bill/cash" in blob):
        mode = DocumentMode.RULED_FORM.value
        confidence = max(confidence, 82.0)

    result = {
        "mode": mode,
        "confidence": round(confidence, 1),
        "signals": {
            "form_keyword_hits": form_hits,
            "invoice_keyword_hits": invoice_hits,
            "poster_keyword_hits": poster_hits,
            "colorfulness": round(color, 2),
            "line_density": round(lines, 5),
            "text_blocks": n_blocks,
            "scores": {k: round(v, 1) for k, v in scores.items()},
        },
        "hybrid": {
            "use_form_grid": mode == DocumentMode.RULED_FORM.value
            or (mode == DocumentMode.DESIGNED_INVOICE.value and form_hits >= 2),
            "use_color_regions": mode in {
                DocumentMode.DESIGNED_INVOICE.value,
                DocumentMode.POSTER.value,
            },
            "use_underlays": mode in {
                DocumentMode.RULED_FORM.value,
                DocumentMode.DESIGNED_INVOICE.value,
            },
            "force_black_text": mode == DocumentMode.RULED_FORM.value,
            "preserve_ornaments": mode == DocumentMode.POSTER.value,
        },
    }
    logger.info(
        "Document classified mode=%s conf=%.1f form=%d invoice=%d color=%.1f lines=%.4f",
        mode,
        confidence,
        form_hits,
        invoice_hits,
        color,
        lines,
    )
    return result


def save_document_mode(image_id: str, settings: Settings, payload: dict[str, Any]) -> Path:
    path = settings.results_path / f"mode_{image_id}.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def load_document_mode(image_id: str, settings: Settings) -> Optional[dict[str, Any]]:
    path = settings.results_path / f"mode_{image_id}.json"
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def ensure_document_mode(
    image_id: str,
    settings: Settings,
    bgr: np.ndarray,
    ocr: Optional[dict[str, Any]],
) -> dict[str, Any]:
    existing = load_document_mode(image_id, settings)
    if existing and existing.get("mode"):
        return existing
    payload = classify_document(bgr, ocr)
    payload["image_id"] = image_id
    save_document_mode(image_id, settings, payload)
    return payload
