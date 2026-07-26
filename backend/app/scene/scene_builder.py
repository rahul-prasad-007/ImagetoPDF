"""
Scene graph builder — assemble an editable document model from prior phases.

Inputs: OCR + Layout + Typography + Reconstruction JSON
Output: results/scene_<uuid>.json + debug/scene_<uuid>.png

NO PDF / SVG / CDR / ReportLab / CairoSVG export.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
from PIL import Image

from app.classify.document_type import ensure_document_mode, load_document_mode
from app.config import Settings
from app.hybrid.underlays import detect_hybrid_underlays
from app.ocr.ocr_service import resolve_processed_image
from app.scene.scene_models import (
    Margins,
    PageFormat,
    PageOrientation,
    SceneCounts,
    SceneLayer,
    SceneObject,
    SceneObjectType,
    ScenePage,
    SceneSummary,
    SceneValidationIssue,
    SceneValidationReport,
)
from app.scene.refine import refine_scene_objects
from app.scene.renderer_models import (
    ImageRefSpec,
    PaintMode,
    PaintStyle,
    RenderNodeHint,
    TextStyleSpec,
    Transform2D,
)

logger = logging.getLogger(__name__)

# Standard page sizes at 300 DPI
_A4_PORTRAIT = (2480, 3508)
_A4_LANDSCAPE = (3508, 2480)
_LETTER_PORTRAIT = (2550, 3300)
_LETTER_LANDSCAPE = (3300, 2550)

_LAYER_DEFS = [
    (1, "Background"),
    (2, "Panels"),
    (3, "Shapes"),
    (5, "Images"),
    (8, "Text"),
    (9, "Groups"),
]

_RECON_TO_SCENE: dict[str, SceneObjectType] = {
    "TEXT": SceneObjectType.TEXT,
    "VECTOR_RECTANGLE": SceneObjectType.RECTANGLE,
    "VECTOR_ROUNDED_RECTANGLE": SceneObjectType.ROUNDED_RECTANGLE,
    "VECTOR_LINE": SceneObjectType.LINE,
    "VECTOR_CIRCLE": SceneObjectType.CIRCLE,
    "VECTOR_ELLIPSE": SceneObjectType.ELLIPSE,
    "VECTOR_POLYGON": SceneObjectType.POLYGON,
    "VECTOR_PATH": SceneObjectType.PATH,
    "IMAGE": SceneObjectType.IMAGE,
    "PHOTO_IMAGE": SceneObjectType.IMAGE,
    "LOGO_IMAGE": SceneObjectType.LOGO,
    "ICON_IMAGE": SceneObjectType.ICON,
    "BACKGROUND_IMAGE": SceneObjectType.BACKGROUND,
}

_VECTOR_TYPES = {
    SceneObjectType.RECTANGLE,
    SceneObjectType.ROUNDED_RECTANGLE,
    SceneObjectType.LINE,
    SceneObjectType.CIRCLE,
    SceneObjectType.ELLIPSE,
    SceneObjectType.POLYGON,
    SceneObjectType.PATH,
}

_IMAGE_TYPES = {
    SceneObjectType.IMAGE,
    SceneObjectType.LOGO,
    SceneObjectType.ICON,
}


def _load_json(path: Path) -> Optional[dict[str, Any]]:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        logger.warning("Failed reading %s", path)
        return None


def _rgb_to_hex(rgb: Any) -> Optional[str]:
    if not rgb or len(rgb) < 3:
        return None
    try:
        r, g, b = int(rgb[0]), int(rgb[1]), int(rgb[2])
        return f"#{r:02X}{g:02X}{b:02X}"
    except Exception:
        return None


def _bgr_to_hex(bgr: Any) -> Optional[str]:
    if not bgr or len(bgr) < 3:
        return None
    try:
        b, g, r = int(bgr[0]), int(bgr[1]), int(bgr[2])
        return f"#{r:02X}{g:02X}{b:02X}"
    except Exception:
        return None


def _is_highlight_fill(fill: Optional[str]) -> bool:
    if not fill or not isinstance(fill, str) or not fill.startswith("#") or len(fill) < 7:
        return False
    try:
        r, g, b = int(fill[1:3], 16), int(fill[3:5], 16), int(fill[5:7], 16)
    except ValueError:
        return False
    mx, mn = max(r, g, b), min(r, g, b)
    sat = (mx - mn) / max(mx, 1)
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    return sat > 0.25 and 70 < lum < 245 and g >= r + 20 and g >= b + 10


def _choose_page(src_w: float, src_h: float) -> tuple[PageFormat, PageOrientation, int, int]:
    orientation = PageOrientation.PORTRAIT if src_h >= src_w else PageOrientation.LANDSCAPE
    aspect = (src_w / src_h) if src_h else 1.0

    a4 = _A4_LANDSCAPE if orientation == PageOrientation.LANDSCAPE else _A4_PORTRAIT
    letter = _LETTER_LANDSCAPE if orientation == PageOrientation.LANDSCAPE else _LETTER_PORTRAIT
    a4_aspect = a4[0] / a4[1]
    letter_aspect = letter[0] / letter[1]

    if abs(aspect - letter_aspect) < abs(aspect - a4_aspect):
        return PageFormat.LETTER, orientation, letter[0], letter[1]
    return PageFormat.A4, orientation, a4[0], a4[1]


def _fit_transform(
    src_w: float,
    src_h: float,
    page_w: float,
    page_h: float,
    margin_ratio: float = 0.04,
) -> tuple[float, float, float, float, Margins]:
    """Uniform scale + center into page with proportional margins."""
    mx = page_w * margin_ratio
    my = page_h * margin_ratio
    usable_w = max(1.0, page_w - 2 * mx)
    usable_h = max(1.0, page_h - 2 * my)
    scale = min(usable_w / max(src_w, 1.0), usable_h / max(src_h, 1.0))
    draw_w = src_w * scale
    draw_h = src_h * scale
    ox = (page_w - draw_w) / 2.0
    oy = (page_h - draw_h) / 2.0
    margins = Margins(top=oy, right=page_w - ox - draw_w, bottom=page_h - oy - draw_h, left=ox)
    return scale, scale, ox, oy, margins


def _map_box(
    bbox: list[float],
    scale_x: float,
    scale_y: float,
    ox: float,
    oy: float,
) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = bbox
    x = ox + x1 * scale_x
    y = oy + y1 * scale_y
    w = max(0.0, (x2 - x1) * scale_x)
    h = max(0.0, (y2 - y1) * scale_y)
    return round(x, 2), round(y, 2), round(w, 2), round(h, 2)


def _typo_index(typo: Optional[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    if not typo:
        return out
    for style in typo.get("text_styles") or []:
        oid = style.get("ocr_block_id")
        if oid is not None:
            out[int(oid)] = style
    return out


def _layout_index(layout: Optional[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    if not layout:
        return out
    for obj in layout.get("objects") or []:
        if obj.get("id") is not None:
            out[int(obj["id"])] = obj
    return out


def _region_for_bbox(
    bbox: list[float],
    layout_by_id: dict[int, dict[str, Any]],
    src_h: float,
) -> str:
    """Classify object into Header / Main / Footer / Body for grouping."""
    x1, y1, x2, y2 = bbox
    cy = (y1 + y2) / 2.0
    for obj in layout_by_id.values():
        otype = obj.get("type")
        if otype not in {"HEADER", "MAIN_CONTENT", "FOOTER"}:
            continue
        hb = obj.get("bbox") or []
        if len(hb) < 4:
            continue
        if hb[0] <= (x1 + x2) / 2 <= hb[2] and hb[1] <= cy <= hb[3]:
            if otype == "HEADER":
                return "Header"
            if otype == "FOOTER":
                return "Footer"
            return "Main"
    if src_h > 0 and cy < src_h * 0.28:
        return "Header"
    if src_h > 0 and cy > src_h * 0.78:
        return "Footer"
    return "Main"


def _memory_kb() -> float:
    try:
        if sys.platform == "win32":
            import ctypes
            from ctypes import wintypes

            class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
                _fields_ = [
                    ("cb", wintypes.DWORD),
                    ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                ]

            counters = PROCESS_MEMORY_COUNTERS()
            counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
            handle = ctypes.windll.kernel32.GetCurrentProcess()
            ctypes.windll.psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb)
            return round(counters.WorkingSetSize / 1024.0, 1)
    except Exception:
        pass
    try:
        import resource

        return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0, 1)
    except Exception:
        return 0.0


def validate_scene(
    objects: list[SceneObject],
    page: ScenePage,
) -> SceneValidationReport:
    issues: list[SceneValidationIssue] = []
    ids = [o.id for o in objects]
    id_set = set(ids)
    dup = len(ids) - len(id_set)
    if dup:
        issues.append(
            SceneValidationIssue(code="DUPLICATE_IDS", message=f"{dup} duplicate id(s) detected")
        )

    missing_parents = 0
    for o in objects:
        if o.parent is not None and o.parent not in id_set and o.parent != 0:
            missing_parents += 1
            issues.append(
                SceneValidationIssue(
                    code="MISSING_PARENT",
                    message=f"Object {o.id} references missing parent {o.parent}",
                    object_id=o.id,
                )
            )

    negative_sizes = 0
    invalid_coords = 0
    for o in objects:
        if o.width < 0 or o.height < 0:
            negative_sizes += 1
            issues.append(
                SceneValidationIssue(
                    code="NEGATIVE_SIZE",
                    message=f"Object {o.id} has negative size",
                    object_id=o.id,
                )
            )
        if o.x < -1 or o.y < -1 or o.x + o.width > page.width + 1 or o.y + o.height > page.height + 1:
            # Groups may be empty / synthetic — only flag drawable content
            if o.type != SceneObjectType.GROUP:
                invalid_coords += 1
                issues.append(
                    SceneValidationIssue(
                        code="INVALID_COORDINATES",
                        message=f"Object {o.id} extends outside page bounds",
                        object_id=o.id,
                    )
                )

    # Overlapping same-layer pairs (heavy IoU) — informational
    overlap_pairs = 0
    drawables = [o for o in objects if o.type != SceneObjectType.GROUP]
    for i in range(len(drawables)):
        a = drawables[i]
        for j in range(i + 1, len(drawables)):
            b = drawables[j]
            if a.layer != b.layer:
                continue
            if a.type == SceneObjectType.BACKGROUND or b.type == SceneObjectType.BACKGROUND:
                continue
            ax2, ay2 = a.x + a.width, a.y + a.height
            bx2, by2 = b.x + b.width, b.y + b.height
            ix1, iy1 = max(a.x, b.x), max(a.y, b.y)
            ix2, iy2 = min(ax2, bx2), min(ay2, by2)
            inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
            if inter <= 0:
                continue
            area_a = max(1.0, a.width * a.height)
            area_b = max(1.0, b.width * b.height)
            iou = inter / (area_a + area_b - inter)
            if iou >= 0.85:
                overlap_pairs += 1
                issues.append(
                    SceneValidationIssue(
                        code="OVERLAPPING_LAYERS",
                        message=f"Objects {a.id} and {b.id} heavily overlap on layer {a.layer}",
                        object_id=a.id,
                    )
                )

    # Cap issue list for response size
    trimmed = issues[:80]
    ok = dup == 0 and missing_parents == 0 and negative_sizes == 0
    return SceneValidationReport(
        ok=ok,
        issues=trimmed,
        duplicate_ids=dup,
        missing_parents=missing_parents,
        negative_sizes=negative_sizes,
        invalid_coordinates=invalid_coords,
        overlapping_layer_pairs=overlap_pairs,
    )


def build_scene(image_id: str, settings: Settings) -> dict[str, Any]:
    image_path = resolve_processed_image(image_id, settings)

    try:
        with Image.open(image_path) as img:
            img.load()
            src_w, src_h = float(img.size[0]), float(img.size[1])
    except Exception as exc:
        raise ValueError(f"Corrupted or unreadable image: {exc}") from exc

    recon = _load_json(settings.results_path / f"reconstruction_{image_id}.json")
    if not recon or not recon.get("objects"):
        raise FileNotFoundError(
            f"Reconstruction results not found for image_id={image_id}. Run /api/reconstruction first."
        )

    layout = _load_json(settings.results_path / f"layout_{image_id}.json")
    ocr = _load_json(settings.results_path / f"{image_id}.json")
    typo = _load_json(settings.results_path / f"typography_{image_id}.json")
    typo_by_ocr = _typo_index(typo)
    layout_by_id = _layout_index(layout)

    started = time.perf_counter()
    mem_before = _memory_kb()

    page_format, orientation, page_w, page_h = _choose_page(src_w, src_h)
    scale_x, scale_y, ox, oy, margins = _fit_transform(src_w, src_h, float(page_w), float(page_h))
    page = ScenePage(
        width=float(page_w),
        height=float(page_h),
        source_width=src_w,
        source_height=src_h,
        margins=margins,
        orientation=orientation,
        page_format=page_format,
        dpi=300,
        scale_x=scale_x,
        scale_y=scale_y,
        offset_x=ox,
        offset_y=oy,
    )

    objects: list[SceneObject] = []
    next_id = 1

    # Logical groups
    group_ids = {
        "Header": next_id,
        "Main": next_id + 1,
        "Footer": next_id + 2,
    }
    next_id += 3
    for name, gid in group_ids.items():
        objects.append(
            SceneObject(
                id=gid,
                parent=None,
                children=[],
                layer=9,
                type=SceneObjectType.GROUP,
                x=0.0,
                y=0.0,
                width=page.width,
                height=page.height,
                name=name,
                meta={"role": "logical_group"},
            )
        )

    ocr_by_id: dict[int, dict[str, Any]] = {}
    for b in (ocr or {}).get("text_blocks") or []:
        if b.get("id") is not None:
            ocr_by_id[int(b["id"])] = b

    seen_keys: set[tuple[str, float, float, float, float]] = set()

    for plan in recon.get("objects") or []:
        recon_type = str(plan.get("reconstruction") or "")
        if recon_type == "IGNORE":
            continue

        scene_type = _RECON_TO_SCENE.get(recon_type)
        if scene_type is None:
            continue

        bbox = plan.get("bbox") or [0, 0, 0, 0]
        if len(bbox) < 4:
            continue
        bbox_f = [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])]
        x, y, w, h = _map_box(bbox_f, scale_x, scale_y, ox, oy)

        # Deduplicate identical geometry+type
        key = (scene_type.value, round(x, 1), round(y, 1), round(w, 1), round(h, 1))
        if key in seen_keys:
            continue
        seen_keys.add(key)

        region = _region_for_bbox(bbox_f, layout_by_id, src_h)
        parent_id = group_ids[region]

        layout_src = layout_by_id.get(int(plan.get("source_id") or 0)) or {}
        meta_layout = layout_src.get("meta") or {}
        plan_meta = plan.get("meta") or {}

        layer = int(plan.get("layer") or 3)
        if scene_type == SceneObjectType.BACKGROUND:
            layer = 1
        elif scene_type in _IMAGE_TYPES:
            layer = 5
        elif scene_type == SceneObjectType.TEXT:
            layer = 8
        elif scene_type in {SceneObjectType.RECTANGLE, SceneObjectType.ROUNDED_RECTANGLE} and str(
            plan.get("type")
        ) in {"BACKGROUND_SHAPE", "BACKGROUND"}:
            layer = 2
        elif scene_type in {SceneObjectType.RECTANGLE, SceneObjectType.ROUNDED_RECTANGLE} and (
            bool(meta_layout.get("highlight"))
            or bool(plan_meta.get("highlight"))
            or _is_highlight_fill(_bgr_to_hex(meta_layout.get("color_bgr")))
        ):
            layer = 2

        obj = SceneObject(
            id=next_id,
            parent=parent_id,
            children=[],
            layer=layer,
            type=scene_type,
            x=x,
            y=y,
            width=w,
            height=h,
            rotation=float(layout_src.get("rotation") or 0.0),
            opacity=1.0,
            visibility=True,
            locked=False,
            source={
                "reconstruction_id": plan.get("id"),
                "layout_id": plan.get("source_id"),
                "layout_type": plan.get("type"),
                "reconstruction": recon_type,
                "source_bbox": bbox_f,
            },
        )

        if scene_type == SceneObjectType.TEXT:
            ocr_ids = plan_meta.get("ocr_block_ids") or layout_src.get("ocr_block_ids") or []
            style = None
            for oid in ocr_ids:
                style = typo_by_ocr.get(int(oid))
                if style:
                    break
            content = (
                plan_meta.get("text")
                or layout_src.get("text")
                or (style or {}).get("text")
                or ""
            )
            if not content and ocr_ids:
                parts = []
                for oid in ocr_ids:
                    blk = ocr_by_id.get(int(oid))
                    if blk and blk.get("text"):
                        parts.append(str(blk["text"]))
                content = "\n".join(parts)

            obj.content = content
            obj.font_size = round(float((style or {}).get("font_size") or max(10.0, h * 0.75)), 2)
            # Scale font to page space
            obj.font_size = round(obj.font_size * scale_y, 2)
            # Keep glyphs inside the mapped line box
            if h > 1:
                obj.font_size = round(min(float(obj.font_size), h * 0.92), 2)
            obj.font_color = (style or {}).get("font_color") or "#111111"
            # OCR often reports tiny spurious rotations on upright text
            rot = float(layout_src.get("rotation") or 0.0)
            if abs(rot) < 5.0:
                rot = 0.0
            obj.rotation = rot
            obj.alignment = (style or {}).get("alignment") or "left"
            obj.paragraph = int((style or {}).get("id") or (ocr_ids[0] if ocr_ids else 0) or 0)
            # Do not re-apply OCR/layout micro-rotations from typography
            style_rot = float((style or {}).get("rotation") or 0.0)
            if abs(style_rot) >= 5.0:
                obj.rotation = style_rot
            obj.opacity = float((style or {}).get("opacity") or 1.0)
            if obj.opacity > 1.0:
                obj.opacity = 1.0

            hint = RenderNodeHint(
                scene_object_id=obj.id,
                object_type=obj.type.value,
                transform=Transform2D(
                    x=x, y=y, width=w, height=h, rotation_deg=obj.rotation, opacity=obj.opacity
                ),
                text=TextStyleSpec(
                    content=obj.content or "",
                    font_size=obj.font_size or 12.0,
                    font_color=obj.font_color or "#000000",
                    alignment=obj.alignment or "left",
                    bold=float((style or {}).get("bold") or 0),
                    italic=float((style or {}).get("italic") or 0),
                    underline=float((style or {}).get("underline") or 0),
                    line_height=float((style or {}).get("line_spacing") or 1.2),
                    character_spacing=float((style or {}).get("character_spacing") or 0),
                    word_spacing=float((style or {}).get("word_spacing") or 0),
                    paragraph_id=obj.paragraph,
                    font_family=str((style or {}).get("font_family") or "serif"),
                ),
                layer=layer,
                z_order=layer * 1000 + obj.id,
            )
            obj.meta["render"] = hint.model_dump(mode="json")

        elif scene_type in _VECTOR_TYPES or scene_type == SceneObjectType.BACKGROUND:
            fill = (
                _bgr_to_hex(meta_layout.get("color_bgr"))
                or _rgb_to_hex(meta_layout.get("color_rgb"))
                or ("#FFFFFF" if scene_type == SceneObjectType.BACKGROUND else "#E2E8F0")
            )
            stroke = "#64748B" if scene_type == SceneObjectType.LINE else None
            stroke_w = 2.0 * scale_x if scene_type == SceneObjectType.LINE else (1.0 * scale_x)
            corner = 0.0
            if scene_type == SceneObjectType.ROUNDED_RECTANGLE:
                corner = min(w, h) * 0.12
            obj.fill_color = fill
            obj.stroke_color = stroke
            obj.stroke_width = round(stroke_w, 2)
            obj.corner_radius = round(corner, 2)
            if scene_type == SceneObjectType.BACKGROUND:
                obj.parent = None  # page-level background
                obj.opacity = 1.0

            paint_mode = PaintMode.STROKE if scene_type == SceneObjectType.LINE else PaintMode.FILL
            hint = RenderNodeHint(
                scene_object_id=obj.id,
                object_type=obj.type.value,
                transform=Transform2D(
                    x=x, y=y, width=w, height=h, rotation_deg=obj.rotation, opacity=obj.opacity
                ),
                paint=PaintStyle(
                    fill_color=obj.fill_color,
                    stroke_color=obj.stroke_color,
                    stroke_width=obj.stroke_width or 0.0,
                    corner_radius=obj.corner_radius or 0.0,
                    paint_mode=paint_mode,
                ),
                layer=layer,
                z_order=layer * 1000 + obj.id,
            )
            obj.meta["render"] = hint.model_dump(mode="json")

        elif scene_type in _IMAGE_TYPES:
            crop = {
                "x": bbox_f[0],
                "y": bbox_f[1],
                "width": max(0.0, bbox_f[2] - bbox_f[0]),
                "height": max(0.0, bbox_f[3] - bbox_f[1]),
            }
            obj.crop = crop
            obj.image_path = str(image_path)
            obj.scale = round(scale_x, 4)
            hint = RenderNodeHint(
                scene_object_id=obj.id,
                object_type=obj.type.value,
                transform=Transform2D(
                    x=x, y=y, width=w, height=h, rotation_deg=obj.rotation, opacity=obj.opacity
                ),
                image=ImageRefSpec(
                    image_path=str(image_path),
                    crop_x=crop["x"],
                    crop_y=crop["y"],
                    crop_width=crop["width"],
                    crop_height=crop["height"],
                    scale=obj.scale or 1.0,
                ),
                layer=layer,
                z_order=layer * 1000 + obj.id,
            )
            obj.meta["render"] = hint.model_dump(mode="json")

        objects.append(obj)
        # Link into group children (unless background detached)
        if obj.parent is not None:
            for g in objects:
                if g.id == obj.parent and g.type == SceneObjectType.GROUP:
                    g.children.append(obj.id)
                    break
        next_id += 1

    # --- Hybrid underlays (logos / header art) ---
    mode_info = load_document_mode(image_id, settings)
    if mode_info is None:
        bgr_for_mode = cv2.imread(str(image_path))
        if bgr_for_mode is None:
            pil_m = Image.open(image_path).convert("RGB")
            bgr_for_mode = cv2.cvtColor(np.array(pil_m), cv2.COLOR_RGB2BGR)
        mode_info = ensure_document_mode(image_id, settings, bgr_for_mode, ocr)
    doc_mode = str(mode_info.get("mode") or "poster")
    hybrid = mode_info.get("hybrid") or {}
    if settings.hybrid_underlays and hybrid.get("use_underlays", doc_mode != "poster"):
        bgr = cv2.imread(str(image_path))
        if bgr is None:
            pil = Image.open(image_path).convert("RGB")
            bgr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
        underlays = detect_hybrid_underlays(bgr, ocr, mode=doc_mode)
        existing_image_boxes = [
            (o.x, o.y, o.x + o.width, o.y + o.height)
            for o in objects
            if o.type in _IMAGE_TYPES
        ]

        def _iou_page(a, b):
            ax1, ay1, ax2, ay2 = a
            bx1, by1, bx2, by2 = b
            ix1, iy1 = max(ax1, bx1), max(ay1, by1)
            ix2, iy2 = min(ax2, bx2), min(ay2, by2)
            inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
            if inter <= 0:
                return 0.0
            aa = max(1.0, (ax2 - ax1) * (ay2 - ay1))
            ba = max(1.0, (bx2 - bx1) * (by2 - by1))
            return inter / (aa + ba - inter)

        for u in underlays:
            x1, y1, x2, y2 = u["bbox"]
            px = x1 * scale_x + ox
            py = y1 * scale_y + oy
            pw = max(1.0, (x2 - x1) * scale_x)
            ph = max(1.0, (y2 - y1) * scale_y)
            page_box = (px, py, px + pw, py + ph)
            if any(_iou_page(page_box, eb) >= 0.55 for eb in existing_image_boxes):
                continue
            kind = str(u.get("kind") or "image")
            st = SceneObjectType.LOGO if kind == "logo" else SceneObjectType.IMAGE
            crop = {
                "x": float(x1),
                "y": float(y1),
                "width": float(x2 - x1),
                "height": float(y2 - y1),
            }
            img_obj = SceneObject(
                id=next_id,
                parent=group_ids.get("Main"),
                children=[],
                layer=5,
                type=st,
                x=round(px, 2),
                y=round(py, 2),
                width=round(pw, 2),
                height=round(ph, 2),
                image_path=str(image_path),
                crop=crop,
                source={"hybrid": True, "kind": kind},
                meta={
                    "hybrid_underlay": True,
                    "kind": kind,
                    "confidence": u.get("confidence"),
                    **(u.get("meta") or {}),
                    "render": {
                        "image": {
                            "image_path": str(image_path),
                            "crop_x": crop["x"],
                            "crop_y": crop["y"],
                            "crop_width": crop["width"],
                            "crop_height": crop["height"],
                        }
                    },
                },
            )
            objects.append(img_obj)
            existing_image_boxes.append(page_box)
            if img_obj.parent is not None:
                for g in objects:
                    if g.id == img_obj.parent and g.type == SceneObjectType.GROUP:
                        g.children.append(img_obj.id)
                        break
            next_id += 1
        logger.info("Hybrid underlays injected mode=%s count=%d", doc_mode, len(underlays))

    # Column snap + color/size harmony (mode-aware)
    refine_fixes = refine_scene_objects(objects, document_mode=doc_mode)
    if refine_fixes:
        logger.info("Scene refine applied %d fixes", len(refine_fixes))

    # Tighten group bounds from children
    by_id = {o.id: o for o in objects}
    for gname, gid in group_ids.items():
        g = by_id[gid]
        kids = [by_id[c] for c in g.children if c in by_id]
        if not kids:
            continue
        min_x = min(k.x for k in kids)
        min_y = min(k.y for k in kids)
        max_x = max(k.x + k.width for k in kids)
        max_y = max(k.y + k.height for k in kids)
        g.x, g.y = round(min_x, 2), round(min_y, 2)
        g.width, g.height = round(max_x - min_x, 2), round(max_y - min_y, 2)
        g.name = gname

    # Stable order: layer then y then x
    objects.sort(key=lambda o: (0 if o.type == SceneObjectType.GROUP else 1, o.layer, o.y, o.x, o.id))

    layers: list[SceneLayer] = []
    for lid, lname in _LAYER_DEFS:
        ids = [o.id for o in objects if o.layer == lid]
        layers.append(SceneLayer(id=lid, name=lname, order=lid, object_ids=ids))

    validation = validate_scene(objects, page)

    counts = SceneCounts(
        total_objects=len(objects),
        groups=sum(1 for o in objects if o.type == SceneObjectType.GROUP),
        layers=len([L for L in layers if L.object_ids]),
        text_objects=sum(1 for o in objects if o.type == SceneObjectType.TEXT),
        image_objects=sum(1 for o in objects if o.type in _IMAGE_TYPES),
        vector_objects=sum(1 for o in objects if o.type in _VECTOR_TYPES),
        background_objects=sum(1 for o in objects if o.type == SceneObjectType.BACKGROUND),
    )

    elapsed_ms = (time.perf_counter() - started) * 1000
    mem_after = _memory_kb()
    memory_kb = max(mem_after, mem_before)

    summary = SceneSummary(
        counts=counts,
        validation=validation,
        memory_kb=memory_kb,
        build_time_ms=round(elapsed_ms, 1),
    )

    logger.info(
        "Scene built id=%s objects=%d groups=%d layers=%d text=%d vectors=%d images=%d "
        "time=%.1fms mem=%.1fKB validation_ok=%s",
        image_id,
        counts.total_objects,
        counts.groups,
        counts.layers,
        counts.text_objects,
        counts.vector_objects,
        counts.image_objects,
        elapsed_ms,
        memory_kb,
        validation.ok,
    )

    results_path = settings.results_path / f"scene_{image_id}.json"
    debug_path = settings.debug_path / f"scene_{image_id}.png"
    _draw_debug(image_path, objects, page, debug_path)

    payload = {
        "success": True,
        "image_id": image_id,
        "document_mode": doc_mode,
        "page": page.model_dump(mode="json"),
        "layers": [L.model_dump(mode="json") for L in layers],
        "objects": [o.model_dump(mode="json") for o in objects],
        "summary": summary.model_dump(mode="json"),
        "processing_time_ms": round(elapsed_ms, 1),
        "results_file": str(results_path),
        "debug_image": str(debug_path),
        "message": "Scene graph built successfully.",
    }
    results_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Scene JSON saved -> %s", results_path)
    return payload


def _draw_debug(
    image_path: Path,
    objects: list[SceneObject],
    page: ScenePage,
    output_path: Path,
) -> None:
    """Debug overlay in *source* image space (map back from page coords)."""
    image = cv2.imread(str(image_path))
    if image is None:
        pil = Image.open(image_path).convert("RGB")
        image = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

    sx = page.scale_x or 1.0
    sy = page.scale_y or 1.0
    ox, oy = page.offset_x, page.offset_y

    colors = {
        SceneObjectType.TEXT: (0, 200, 0),
        SceneObjectType.RECTANGLE: (255, 90, 20),
        SceneObjectType.ROUNDED_RECTANGLE: (255, 90, 20),
        SceneObjectType.LINE: (255, 90, 20),
        SceneObjectType.CIRCLE: (255, 90, 20),
        SceneObjectType.ELLIPSE: (255, 90, 20),
        SceneObjectType.POLYGON: (255, 90, 20),
        SceneObjectType.PATH: (255, 90, 20),
        SceneObjectType.IMAGE: (180, 0, 180),
        SceneObjectType.LOGO: (0, 220, 255),
        SceneObjectType.ICON: (180, 0, 180),
        SceneObjectType.BACKGROUND: (0, 140, 255),
        SceneObjectType.GROUP: (160, 160, 160),
    }

    for o in sorted(objects, key=lambda z: z.layer):
        if o.type == SceneObjectType.GROUP:
            continue
        # Inverse map page → source
        x1 = int((o.x - ox) / sx)
        y1 = int((o.y - oy) / sy)
        x2 = int((o.x + o.width - ox) / sx)
        y2 = int((o.y + o.height - oy) / sy)
        color = colors.get(o.type, (128, 128, 128))
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        label = f"#{o.id} L{o.layer} {o.type.value}"
        cv2.putText(
            image,
            label[:48],
            (x1, max(14, y1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (20, 20, 20),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            image,
            label[:48],
            (x1, max(14, y1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            color,
            1,
            cv2.LINE_AA,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), image)
    logger.info("Scene debug image saved -> %s", output_path)
