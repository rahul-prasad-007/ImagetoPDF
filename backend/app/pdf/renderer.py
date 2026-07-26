"""
Editable PDF renderer — reconstruct scene + vectors as real PDF objects.

Uses ReportLab for text/vector/image XObjects. Never flattens the page to one image.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Optional

from reportlab.pdfgen.canvas import Canvas

from app.config import Settings
from app.ocr.ocr_service import resolve_processed_image
from app.pdf.font_manager import ensure_fonts
from app.pdf.image_renderer import render_image_object
from app.pdf.models import RenderSummary
from app.pdf.page_renderer import build_page_context
from app.pdf.pdf_service import (
    apply_pdf_metadata,
    count_drawn,
    describe_size,
    validate_render_inputs,
)
from app.pdf.shape_renderer import render_vector_object
from app.pdf.text_renderer import render_complex_script_texts, render_text_object
from app.classify.document_type import load_document_mode

logger = logging.getLogger(__name__)


def _load_json(path: Path) -> Optional[dict[str, Any]]:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        logger.warning("Failed reading %s", path)
        return None


def _layer_key(obj: dict[str, Any]) -> tuple[int, float, float, int]:
    return (
        int(obj.get("layer") or 3),
        float(obj.get("y") or 0),
        float(obj.get("x") or 0),
        int(obj.get("id") or 0),
    )


def render_editable_pdf(image_id: str, settings: Settings) -> dict[str, Any]:
    """
    Build output/output_<uuid>.pdf from scene + vector + typography.
    """
    image_path = resolve_processed_image(image_id, settings)

    scene = _load_json(settings.results_path / f"scene_{image_id}.json")
    if not scene or not scene.get("objects"):
        raise FileNotFoundError(
            f"Scene graph not found for image_id={image_id}. Run /api/scene first."
        )

    vectors_doc = _load_json(settings.results_path / f"vector_{image_id}.json")
    if not vectors_doc:
        raise FileNotFoundError(
            f"Vector results not found for image_id={image_id}. Run /api/vector first."
        )

    typo = _load_json(settings.results_path / f"typography_{image_id}.json")

    started = time.perf_counter()
    ensure_fonts()
    ctx = build_page_context(scene)
    validation = validate_render_inputs(scene, vectors_doc, ctx)

    output_path = settings.output_path / f"output_{image_id}.pdf"
    canvas = Canvas(
        str(output_path),
        pagesize=(ctx.pdf_width, ctx.pdf_height),
    )

    text_n = image_n = vector_n = 0

    # ---- Shapes from Vector JSON (background → panels → shapes) ----
    vector_items = list(vectors_doc.get("vectors") or [])
    vector_items.sort(key=_layer_key)
    for v in vector_items:
        if render_vector_object(canvas, v, ctx, from_source=True):
            vector_n += 1

    # ---- Images from scene ----
    for obj in sorted(scene.get("objects") or [], key=_layer_key):
        otype = str(obj.get("type") or "")
        if otype not in {"IMAGE", "LOGO", "ICON"}:
            continue
        if render_image_object(canvas, obj, ctx, default_image_path=image_path):
            image_n += 1

    # ---- Editable text from scene ----
    typo_by_ocr: dict[int, dict[str, Any]] = {}
    if typo:
        for style in typo.get("text_styles") or []:
            oid = style.get("ocr_block_id")
            if oid is not None:
                typo_by_ocr[int(oid)] = style

    enriched_texts: list[dict[str, Any]] = []
    for obj in sorted(scene.get("objects") or [], key=_layer_key):
        if str(obj.get("type")) != "TEXT":
            continue
        meta = dict(obj.get("meta") or {})
        render = dict(meta.get("render") or {})
        text_spec = dict(render.get("text") or {})
        for oid in meta.get("ocr_block_ids") or []:
            style = typo_by_ocr.get(int(oid))
            if style:
                text_spec.setdefault("bold", style.get("bold"))
                text_spec.setdefault("italic", style.get("italic"))
                text_spec.setdefault("underline", style.get("underline"))
                text_spec.setdefault("line_height", style.get("line_spacing"))
                text_spec.setdefault("character_spacing", style.get("character_spacing"))
                text_spec.setdefault("font_family", style.get("font_family"))
                break
        render["text"] = text_spec
        meta["render"] = render
        enriched = dict(obj)
        enriched["meta"] = meta
        enriched_texts.append(enriched)
        if render_text_object(canvas, enriched, ctx):
            text_n += 1

    canvas.showPage()
    canvas.save()

    # Devanagari / complex scripts via PyMuPDF (proper shaping)
    try:
        text_n += render_complex_script_texts(output_path, enriched_texts, ctx)
    except Exception as exc:
        logger.warning("Complex-script text pass failed: %s", exc)

    apply_pdf_metadata(
        output_path,
        title=f"Editable PDF — {image_id[:8]}",
        author="Image to Editable PDF",
        producer="ImgToPDF Editable Renderer / ReportLab+PyMuPDF",
    )

    elapsed = (time.perf_counter() - started) * 1000
    size_bytes, size_label = describe_size(output_path)
    counts = count_drawn(text_n, image_n, vector_n)
    summary = RenderSummary(
        counts=counts,
        validation=validation,
        pdf_size_bytes=size_bytes,
        pdf_size=size_label,
        render_time_ms=round(elapsed, 1),
        page_format=ctx.page_format,
        orientation=ctx.orientation,
    )

    rel_pdf = f"output/output_{image_id}.pdf"
    download_url = f"/api/output/output_{image_id}.pdf"
    mode_info = load_document_mode(image_id, settings) or {}
    doc_mode = mode_info.get("mode") or scene.get("document_mode")

    logger.info(
        "PDF rendered id=%s mode=%s objects=%d text=%d images=%d vectors=%d size=%s time=%.1fms path=%s",
        image_id,
        doc_mode,
        counts.total_objects,
        text_n,
        image_n,
        vector_n,
        size_label,
        elapsed,
        output_path,
    )

    return {
        "success": True,
        "image_id": image_id,
        "document_mode": doc_mode,
        "pdf": rel_pdf,
        "download_url": download_url,
        "preview_url": download_url,
        "summary": summary.model_dump(mode="json"),
        "processing_time_ms": round(elapsed, 1),
        "message": "Editable PDF generated successfully.",
    }
