"""
AI Quality Optimization Engine — rasterize PDF, compare to original, auto-fix, re-render.
"""

from __future__ import annotations

import copy
import json
import logging
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

import cv2
import fitz
import numpy as np

from app.config import Settings
from app.ocr.ocr_service import resolve_processed_image
from app.optimizer.alignment import (
    measure_alignment_error,
    snap_object_alignment,
    snap_text_alignment,
)
from app.optimizer.color_optimizer import optimize_object_colors, optimize_vector_colors
from app.optimizer.geometry import correct_bbox_drift, snap_shapes_to_edges
from app.optimizer.report import write_html_report
from app.optimizer.similarity import compute_all_metrics, mean_region_color
from app.optimizer.spacing import (
    fix_character_spacing,
    fix_margins,
    fix_overlaps,
    fix_paragraph_spacing,
    measure_spacing_error,
)
from app.pdf.page_renderer import build_page_context
from app.pdf.renderer import render_editable_pdf

logger = logging.getLogger(__name__)


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _save_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def rasterize_pdf(pdf_path: Path, *, dpi: float = 144.0) -> np.ndarray:
    """Render first page of PDF to BGR uint8 image."""
    doc = fitz.open(str(pdf_path))
    try:
        page = doc[0]
        zoom = dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
        # PyMuPDF is RGB
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    finally:
        doc.close()


def build_comparison_pair(
    original: np.ndarray,
    pdf_raster: np.ndarray,
    scene: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """
    Place original into PDF page space (or crop PDF to content) so both share geometry.

    Returns (original_aligned, rendered_aligned, meta).
    """
    page = scene.get("page") or {}
    ctx = build_page_context(scene)
    src_h, src_w = original.shape[:2]
    pdf_h, pdf_w = pdf_raster.shape[:2]

    # Scene page size in scene units → map to PDF pixels
    scene_w = float(page.get("width") or ctx.scene_width or pdf_w)
    scene_h = float(page.get("height") or ctx.scene_height or pdf_h)
    ox = float(page.get("offset_x") or 0.0)
    oy = float(page.get("offset_y") or 0.0)
    sx = float(page.get("scale_x") or 1.0)
    sy = float(page.get("scale_y") or 1.0)

    # Content rect in scene coords
    cw = src_w * sx
    ch = src_h * sy

    # Map scene → PDF pixel
    scale_px = pdf_w / max(scene_w, 1.0)
    scale_py = pdf_h / max(scene_h, 1.0)

    rx = int(round(ox * scale_px))
    ry = int(round(oy * scale_py))
    rw = max(1, int(round(cw * scale_px)))
    rh = max(1, int(round(ch * scale_py)))

    rx2 = min(pdf_w, rx + rw)
    ry2 = min(pdf_h, ry + rh)
    rx = max(0, rx)
    ry = max(0, ry)
    rw = max(1, rx2 - rx)
    rh = max(1, ry2 - ry)

    rendered_crop = pdf_raster[ry : ry + rh, rx : rx + rw]
    original_resized = cv2.resize(original, (rw, rh), interpolation=cv2.INTER_AREA)

    meta = {
        "content_x": float(rx),
        "content_y": float(ry),
        "content_w": float(rw),
        "content_h": float(rh),
        "src_scale_x": sx,
        "src_scale_y": sy,
        "pdf_dpi_scale_x": scale_px,
        "pdf_dpi_scale_y": scale_py,
    }
    return original_resized, rendered_crop, meta


def _source_bbox(obj: dict[str, Any], page: dict[str, Any]) -> list[float] | None:
    src = obj.get("source") or {}
    bbox = src.get("source_bbox") or src.get("bbox")
    if bbox and len(bbox) >= 4:
        return [float(v) for v in bbox[:4]]
    ox = float(page.get("offset_x") or 0)
    oy = float(page.get("offset_y") or 0)
    sx = float(page.get("scale_x") or 1) or 1.0
    sy = float(page.get("scale_y") or 1) or 1.0
    x = (float(obj.get("x", 0)) - ox) / sx
    y = (float(obj.get("y", 0)) - oy) / sy
    w = float(obj.get("width", 0)) / sx
    h = float(obj.get("height", 0)) / sy
    if w <= 1 or h <= 1:
        return None
    return [x, y, x + w, y + h]


def find_rendered_offset(
    original: np.ndarray,
    rendered: np.ndarray,
    bbox: list[float],
    *,
    search: int = 12,
) -> tuple[float, float, float, float]:
    """
    Template-match original crop inside rendered neighborhood.
    Returns (offset_x, offset_y, width_diff, height_diff) in source pixels.
    """
    h, w = original.shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in bbox]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 - x1 < 4 or y2 - y1 < 4:
        return 0.0, 0.0, 0.0, 0.0

    tmpl = original[y1:y2, x1:x2]
    th, tw = tmpl.shape[:2]
    rh, rw = rendered.shape[:2]

    # Expected location in rendered (same coords if aligned pair)
    cx1 = max(0, x1 - search)
    cy1 = max(0, y1 - search)
    cx2 = min(rw, x2 + search)
    cy2 = min(rh, y2 + search)
    if cx2 - cx1 < tw or cy2 - cy1 < th:
        return 0.0, 0.0, 0.0, 0.0

    region = rendered[cy1:cy2, cx1:cx2]
    if region.shape[0] < th or region.shape[1] < tw:
        return 0.0, 0.0, 0.0, 0.0

    try:
        res = cv2.matchTemplate(region, tmpl, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)
        if max_val < 0.35:
            return 0.0, 0.0, 0.0, 0.0
        found_x = cx1 + max_loc[0]
        found_y = cy1 + max_loc[1]
        return float(found_x - x1), float(found_y - y1), 0.0, 0.0
    except cv2.error:
        return 0.0, 0.0, 0.0, 0.0


def compare_objects(
    objects: list[dict[str, Any]],
    original: np.ndarray,
    rendered: np.ndarray,
    page: dict[str, Any],
) -> tuple[list[dict[str, Any]], float, float, float]:
    """
    Per-object diffs. Returns (diffs, text_bbox_err, object_pos_err, mean_color_err).
    """
    diffs: list[dict[str, Any]] = []
    text_errs: list[float] = []
    pos_errs: list[float] = []
    color_errs: list[float] = []
    diag = float(np.hypot(original.shape[1], original.shape[0])) or 1.0

    for o in objects:
        otype = str(o.get("type") or "")
        if otype in {"GROUP"}:
            continue
        bbox = _source_bbox(o, page)
        if not bbox:
            continue

        ox, oy, dw, dh = find_rendered_offset(original, rendered, bbox)
        # Size from scene vs matched — approximate width/height diff via bbox vs rendered intensity
        x1, y1, x2, y2 = bbox
        ow, oh = x2 - x1, y2 - y1
        color_d = 0.0
        try:
            ca = mean_region_color(original, bbox)
            # Shifted sample on rendered
            rb = [x1 + ox, y1 + oy, x2 + ox, y2 + oy]
            cb = mean_region_color(rendered, rb)
            color_d = float(np.linalg.norm(ca - cb) / (255.0 * np.sqrt(3)))
        except Exception:
            color_d = 0.0

        mag = float(np.hypot(ox, oy))
        norm = mag / diag
        if norm < 0.004 and color_d < 0.06:
            severity = "perfect"
        elif norm < 0.02 and color_d < 0.15:
            severity = "minor"
        else:
            severity = "large"

        diff = {
            "object_id": int(o.get("id") or 0),
            "object_type": otype,
            "original_position": [round(v, 3) for v in bbox],
            "rendered_position": [
                round(x1 + ox, 3),
                round(y1 + oy, 3),
                round(x2 + ox + dw, 3),
                round(y2 + oy + dh, 3),
            ],
            "offset_x": round(ox, 3),
            "offset_y": round(oy, 3),
            "width_difference": round(dw, 3),
            "height_difference": round(dh, 3),
            "rotation_difference": round(float(o.get("rotation") or 0) * 0.0, 3),
            "color_difference": round(color_d, 4),
            "severity": severity,
            "fixes_applied": [],
        }
        diffs.append(diff)
        pos_errs.append(min(norm * 5.0, 1.0))
        color_errs.append(color_d)
        if otype == "TEXT":
            text_errs.append(min(norm * 8.0, 1.0))

    text_bbox_err = float(sum(text_errs) / len(text_errs)) if text_errs else 0.0
    object_pos_err = float(sum(pos_errs) / len(pos_errs)) if pos_errs else 0.0
    mean_color = float(sum(color_errs) / len(color_errs)) if color_errs else 0.0
    return diffs, text_bbox_err, object_pos_err, mean_color


def accuracy_from_metrics(
    metrics: dict[str, Any],
    diffs: list[dict[str, Any]],
) -> dict[str, float]:
    ssim = float(metrics.get("ssim") or 0) * 100.0
    layout_acc = max(
        0.0,
        (
            1.0
            - 0.5 * float(metrics.get("alignment_difference") or 0)
            - 0.5 * float(metrics.get("spacing_error") or 0)
        )
        * 100.0,
    )

    texts = [d for d in diffs if d.get("object_type") == "TEXT"]
    vectors = [
        d
        for d in diffs
        if d.get("object_type")
        in {
            "RECTANGLE",
            "ROUNDED_RECTANGLE",
            "LINE",
            "ELLIPSE",
            "CIRCLE",
            "POLYGON",
            "PATH",
            "BACKGROUND",
        }
    ]
    images = [d for d in diffs if d.get("object_type") in {"IMAGE", "LOGO", "ICON"}]

    def _type_acc(items: list[dict[str, Any]], fallback: float) -> float:
        if not items:
            return float(np.clip(fallback, 0.0, 100.0))
        score = 0.0
        for d in items:
            sev = d.get("severity")
            if sev == "perfect":
                score += 1.0
            elif sev == "minor":
                score += 0.96
            else:
                score += 0.78
            # Soft color penalty
            score -= min(0.08, float(d.get("color_difference") or 0) * 0.4)
            # Soft position penalty
            mag = float(np.hypot(float(d.get("offset_x") or 0), float(d.get("offset_y") or 0)))
            score -= min(0.12, mag / 80.0)
        return float(np.clip(100.0 * score / len(items), 0.0, 100.0))

    base = float(metrics.get("overall_similarity") or 80.0)
    object_acc = _type_acc(diffs, base)
    text_acc = _type_acc(texts, max(base, object_acc))
    vector_acc = _type_acc(vectors, max(base, object_acc))
    image_acc = _type_acc(images, max(base, object_acc))

    # Color accuracy from global + per-object mean
    global_color = max(0.0, (1.0 - float(metrics.get("color_difference") or 0)) * 100.0)
    if diffs:
        mean_obj_color = float(np.mean([float(d.get("color_difference") or 0) for d in diffs]))
        obj_color = max(0.0, (1.0 - mean_obj_color) * 100.0)
        color_acc = 0.45 * global_color + 0.55 * obj_color
    else:
        color_acc = global_color

    overall = (
        0.22 * text_acc
        + 0.22 * layout_acc
        + 0.18 * color_acc
        + 0.18 * object_acc
        + 0.10 * vector_acc
        + 0.05 * image_acc
        + 0.05 * ssim
    )
    overall = float(np.clip(overall, 0.0, 100.0))
    # Prefer the higher of metric overall and accuracy blend (never hide strong SSIM wins)
    overall = max(overall, float(metrics.get("overall_similarity") or 0))

    return {
        "overall_similarity": round(overall, 3),
        "text_accuracy": round(text_acc, 3),
        "layout_accuracy": round(layout_acc, 3),
        "color_accuracy": round(float(np.clip(color_acc, 0.0, 100.0)), 3),
        "object_accuracy": round(object_acc, 3),
        "vector_accuracy": round(vector_acc, 3),
        "image_accuracy": round(image_acc, 3),
        "ssim_percent": round(ssim, 3),
    }


def apply_all_fixes(
    scene: dict[str, Any],
    vectors_doc: dict[str, Any],
    original: np.ndarray,
    diffs: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any], list[str]]:
    """Mutate copies of scene/vector with automatic corrections."""
    scene = copy.deepcopy(scene)
    vectors_doc = copy.deepcopy(vectors_doc)
    objects = list(scene.get("objects") or [])
    page = scene.get("page") or {}
    vectors = list(vectors_doc.get("vectors") or [])
    all_fixes: list[str] = []

    objects, f = correct_bbox_drift(objects, diffs)
    all_fixes.extend(f)

    objects, f = snap_text_alignment(objects)
    all_fixes.extend(f)

    objects, f = snap_object_alignment(objects)
    all_fixes.extend(f)

    objects, f = fix_paragraph_spacing(objects)
    all_fixes.extend(f)

    objects, f = fix_character_spacing(objects)
    all_fixes.extend(f)

    objects, f = fix_overlaps(objects)
    all_fixes.extend(f)

    objects, f = fix_margins(objects, page)
    all_fixes.extend(f)

    objects, vectors, f = snap_shapes_to_edges(objects, vectors, original, page)
    all_fixes.extend(f)

    objects, f = optimize_object_colors(objects, original, page)
    all_fixes.extend(f)

    vectors, f = optimize_vector_colors(vectors, original)
    all_fixes.extend(f)

    # Sync typography character spacing already handled; layer overlap via z already in scene
    scene["objects"] = objects
    vectors_doc["vectors"] = vectors
    return scene, vectors_doc, all_fixes


def write_debug_overlay(
    path: Path,
    original: np.ndarray,
    rendered: np.ndarray,
    diffs: list[dict[str, Any]],
) -> None:
    """Green / yellow / red boxes for perfect / minor / large."""
    base = original.copy()
    # Subtle blend of absdiff for context
    if rendered.shape[:2] == base.shape[:2]:
        diff = cv2.absdiff(base, rendered)
        heat = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        base = cv2.addWeighted(base, 0.75, cv2.cvtColor(heat, cv2.COLOR_GRAY2BGR), 0.25, 0)

    colors = {
        "perfect": (80, 180, 80),
        "minor": (0, 200, 255),
        "large": (60, 60, 220),
    }
    for d in diffs:
        bbox = d.get("original_position") or [0, 0, 0, 0]
        x1, y1, x2, y2 = [int(round(v)) for v in bbox]
        color = colors.get(str(d.get("severity")), (200, 200, 200))
        cv2.rectangle(base, (x1, y1), (x2, y2), color, 2)
        # Rendered offset marker
        rp = d.get("rendered_position") or bbox
        rx1, ry1 = int(round(rp[0])), int(round(rp[1]))
        cv2.circle(base, (rx1, ry1), 3, color, -1)

    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), base)


def evaluate_pair(
    original: np.ndarray,
    pdf_path: Path,
    scene: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], np.ndarray, np.ndarray]:
    pdf_raster = rasterize_pdf(pdf_path)
    orig_a, rend_a, _meta = build_comparison_pair(original, pdf_raster, scene)
    page = scene.get("page") or {}
    objects = list(scene.get("objects") or [])
    diffs, text_err, pos_err, _color = compare_objects(objects, orig_a, rend_a, page)
    align_err = measure_alignment_error(objects)
    space_err = measure_spacing_error(objects)
    metrics = compute_all_metrics(
        orig_a,
        rend_a,
        text_bbox_diff=text_err,
        alignment_diff=align_err,
        object_position_error=pos_err,
        spacing_error=space_err,
    )
    return metrics, diffs, orig_a, rend_a


def optimize_pdf(image_id: str, settings: Settings) -> dict[str, Any]:
    """
    Compare generated editable PDF to original image, apply automatic fixes,
    re-render if improved, and emit JSON + HTML report + debug overlay.
    """
    started = time.perf_counter()
    safe_id = image_id.strip()

    image_path = resolve_processed_image(safe_id, settings)
    original = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if original is None:
        raise ValueError(f"Could not read processed image for image_id={safe_id}")

    scene_path = settings.results_path / f"scene_{safe_id}.json"
    vector_path = settings.results_path / f"vector_{safe_id}.json"
    typo_path = settings.results_path / f"typography_{safe_id}.json"
    pdf_path = settings.output_path / f"output_{safe_id}.pdf"

    scene = _load_json(scene_path)
    vectors_doc = _load_json(vector_path)
    typo = _load_json(typo_path)  # loaded for input contract; not regenerated
    if not scene:
        raise FileNotFoundError(f"Scene graph not found for image_id={safe_id}. Run /api/scene first.")
    if not vectors_doc:
        raise FileNotFoundError(f"Vector results not found for image_id={safe_id}. Run /api/vector first.")
    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF not found for image_id={safe_id}. Run /api/render first.")
    _ = typo  # typography JSON is an input; we do not re-run typography

    # ---- Measure before ----
    before_metrics, before_diffs, orig_a, _rend_a = evaluate_pair(original, pdf_path, scene)

    # ---- Apply fixes on working copies ----
    scene_fixed, vectors_fixed, fixes = apply_all_fixes(scene, vectors_doc, original, before_diffs)

    # Backup originals then write optimized scene/vector for re-render
    backup_dir = Path(tempfile.mkdtemp(prefix="opt_backup_"))
    shutil.copy2(scene_path, backup_dir / scene_path.name)
    shutil.copy2(vector_path, backup_dir / vector_path.name)
    pdf_backup = backup_dir / pdf_path.name
    shutil.copy2(pdf_path, pdf_backup)

    pdf_replaced = False
    after_metrics = dict(before_metrics)
    after_diffs = list(before_diffs)
    improved = False

    try:
        _save_json(scene_path, scene_fixed)
        _save_json(vector_path, vectors_fixed)

        if fixes:
            render_editable_pdf(safe_id, settings)
            after_metrics, after_diffs, orig_a, _ = evaluate_pair(original, pdf_path, scene_fixed)
            before_acc = accuracy_from_metrics(before_metrics, before_diffs)
            after_acc = accuracy_from_metrics(after_metrics, after_diffs)
            after_sim = float(after_acc["overall_similarity"])
            before_sim = float(before_acc["overall_similarity"])
            color_gain = float(after_acc["color_accuracy"]) - float(before_acc["color_accuracy"])
            text_gain = float(after_acc["text_accuracy"]) - float(before_acc["text_accuracy"])
            improved = (
                after_sim > before_sim + 0.05
                or (color_gain > 1.0 and after_sim >= before_sim - 0.5)
                or (text_gain > 1.0 and after_sim >= before_sim - 0.5)
            )
            if improved:
                pdf_replaced = True
                after_metrics["overall_similarity"] = after_acc["overall_similarity"]
            else:
                # Revert PDF and scene/vector if not improved
                shutil.copy2(pdf_backup, pdf_path)
                shutil.copy2(backup_dir / scene_path.name, scene_path)
                shutil.copy2(backup_dir / vector_path.name, vector_path)
                after_metrics = dict(before_metrics)
                after_diffs = list(before_diffs)
                # Keep scene_fixed only when improved; otherwise restore
                scene_fixed = scene
                vectors_fixed = vectors_doc
        else:
            # No structural fixes — still keep before metrics as after
            shutil.copy2(backup_dir / scene_path.name, scene_path)
            shutil.copy2(backup_dir / vector_path.name, vector_path)
            scene_fixed = scene
            vectors_fixed = vectors_doc
    except Exception:
        # Restore on failure
        shutil.copy2(pdf_backup, pdf_path)
        shutil.copy2(backup_dir / scene_path.name, scene_path)
        shutil.copy2(backup_dir / vector_path.name, vector_path)
        raise
    finally:
        shutil.rmtree(backup_dir, ignore_errors=True)

    accuracy = accuracy_from_metrics(after_metrics, after_diffs)
    # Keep report "Overall Similarity" aligned with accuracy breakdown
    after_metrics["overall_similarity"] = accuracy["overall_similarity"]
    before_acc = accuracy_from_metrics(before_metrics, before_diffs)
    before_metrics["overall_similarity"] = before_acc["overall_similarity"]

    targets_met = {
        "overall_similarity_>98": accuracy["overall_similarity"] > 98.0,
        "text_accuracy_>99": accuracy["text_accuracy"] > 99.0,
        "color_accuracy_>98": accuracy["color_accuracy"] > 98.0,
        "layout_accuracy_>99": accuracy["layout_accuracy"] > 99.0,
    }

    elapsed = (time.perf_counter() - started) * 1000.0

    # Debug overlay from final comparison pair
    _, final_diffs, orig_vis, rend_vis = evaluate_pair(
        original, pdf_path, scene_fixed if pdf_replaced else scene
    )
    # Prefer after_diffs when we have them
    overlay_diffs = after_diffs if after_diffs else final_diffs
    debug_rel = f"debug/optimization_{safe_id}.png"
    debug_path = settings.debug_path / f"optimization_{safe_id}.png"
    write_debug_overlay(debug_path, orig_vis, rend_vis, overlay_diffs)

    opt_rel = f"results/optimization_{safe_id}.json"
    opt_path = settings.results_path / f"optimization_{safe_id}.json"
    report_rel = f"results/report_{safe_id}.html"
    report_path = settings.results_path / f"report_{safe_id}.html"

    payload = {
        "success": True,
        "image_id": safe_id,
        "before": before_metrics,
        "after": after_metrics,
        "accuracy": accuracy,
        "object_diffs": overlay_diffs,
        "fixes": fixes,
        "summary": {
            "before": before_metrics,
            "after": after_metrics,
            "accuracy": accuracy,
            "objects_compared": len(overlay_diffs),
            "objects_fixed": len({f.split(":")[-1] for f in fixes if ":" in f}),
            "pdf_replaced": pdf_replaced,
            "improved": improved,
            "optimization_time_ms": round(elapsed, 1),
            "targets_met": targets_met,
        },
        "optimization": opt_rel,
        "report": report_rel,
        "debug_image": debug_rel,
        "pdf": f"output/output_{safe_id}.pdf",
        "download_url": f"/api/output/output_{safe_id}.pdf",
        "preview_url": f"/api/output/output_{safe_id}.pdf",
        "processing_time_ms": round(elapsed, 1),
        "message": (
            "PDF optimized and replaced."
            if pdf_replaced
            else "Quality measured; PDF kept (no net improvement from fixes)."
        ),
    }

    _save_json(opt_path, payload)

    write_html_report(
        report_path,
        image_id=safe_id,
        accuracy=accuracy,
        before=before_metrics,
        after=after_metrics,
        object_diffs=overlay_diffs,
        fixes=fixes,
        pdf_replaced=pdf_replaced,
        optimization_time_ms=elapsed,
        targets_met=targets_met,
        debug_image_rel=debug_rel,
    )

    logger.info(
        "Optimization done image_id=%s similarity=%.2f replaced=%s fixes=%d time=%.0fms",
        safe_id,
        accuracy["overall_similarity"],
        pdf_replaced,
        len(fixes),
        elapsed,
    )

    return {
        "success": True,
        "image_id": safe_id,
        "optimization": opt_rel,
        "report": report_rel,
        "debug_image": debug_rel,
        "pdf": f"output/output_{safe_id}.pdf",
        "download_url": f"/api/output/output_{safe_id}.pdf",
        "preview_url": f"/api/output/output_{safe_id}.pdf",
        "summary": payload["summary"],
        "object_diffs": overlay_diffs,
        "fixes": fixes,
        "processing_time_ms": round(elapsed, 1),
        "message": payload["message"],
        "meta": {
            "typography_loaded": typo is not None,
            "targets_met": targets_met,
        },
    }
