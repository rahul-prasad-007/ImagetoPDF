"""
Post-build scene refinement — column snap, size/color/bold harmony, no overflow.
"""

from __future__ import annotations

import re
from typing import Any

from app.scene.scene_models import SceneObject, SceneObjectType


def _is_number(text: str | None) -> bool:
    t = (text or "").strip()
    return bool(t) and t.rstrip(".").isdigit() and len(t) <= 3


def _is_brand(text: str | None) -> bool:
    t = (text or "").strip()
    return bool(t) and t.isupper() and 2 <= len(t) <= 16 and " " not in t


_FORM_KEEP_TOKENS = {
    "P",
    "Rs",
    "Rs.",
    "No",
    "No -",
    "No-",
    "SL",
    "SL.",
    "SL.No",
    "Qty",
    "QTY",
}


def _is_junk_text(text: str | None, width: float, height: float) -> bool:
    t = (text or "").strip()
    if not t:
        return True
    if t in _FORM_KEEP_TOKENS:
        return False
    if len(t) == 1 and t in {".", ",", "-", "—", "–", "|", "/", "\\", "'", '"', "S", "l", "I", "|"}:
        return True
    # Hearts often OCR as emoji — hide clear icon tokens
    if t in {"❤", "♥", "♡", "<3"} and max(width, height) < 100:
        return True
    # Lone "3" icon (common OCR for heart dividers) — allow larger boxes
    if t == "3" and max(width, height) < 120 and min(width, height) < 100:
        return True
    if len(t) <= 1 and max(width, height) < 40:
        return True
    return False


def _is_form_like_texts(texts: list[SceneObject]) -> bool:
    blob = " ".join((o.content or "").lower() for o in texts)
    keys = (
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
    return sum(1 for k in keys if k in blob) >= 3


def _lum_hex(color: str | None) -> float:
    rgb = _parse_hex(color)
    if not rgb:
        return 0.0
    r, g, b = rgb
    return 0.299 * r + 0.587 * g + 0.114 * b


def _hex_from_rgb(r: float, g: float, b: float) -> str:
    return f"#{int(round(r)):02X}{int(round(g)):02X}{int(round(b)):02X}"


def _parse_hex(color: str | None) -> tuple[float, float, float] | None:
    if not color or not isinstance(color, str) or not color.startswith("#") or len(color) < 7:
        return None
    try:
        return float(int(color[1:3], 16)), float(int(color[3:5], 16)), float(int(color[5:7], 16))
    except ValueError:
        return None


def _median(vals: list[float]) -> float:
    if not vals:
        return 0.0
    s = sorted(vals)
    return float(s[len(s) // 2])


def _cluster_by_x(items: list[SceneObject], tol: float) -> list[list[SceneObject]]:
    ordered = sorted(items, key=lambda o: o.x)
    clusters: list[list[SceneObject]] = []
    for o in ordered:
        placed = False
        for c in clusters:
            if abs(c[0].x - o.x) <= tol:
                c.append(o)
                placed = True
                break
        if not placed:
            clusters.append([o])
    return clusters


def _cluster_by_size(items: list[SceneObject], rel_gap: float = 0.28) -> list[list[SceneObject]]:
    """Split into size bands where consecutive medians jump by rel_gap."""
    ordered = sorted(items, key=lambda o: float(o.font_size or 0))
    bands: list[list[SceneObject]] = []
    for o in ordered:
        fs = float(o.font_size or 0)
        if not bands:
            bands.append([o])
            continue
        prev = _median([float(x.font_size or 0) for x in bands[-1]])
        if prev > 1 and (fs - prev) / prev >= rel_gap:
            bands.append([o])
        else:
            bands[-1].append(o)
    return bands


def _get_bold(o: SceneObject) -> float:
    render = (o.meta or {}).get("render") or {}
    text = render.get("text") or {}
    try:
        return float(text.get("bold") or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _set_bold(o: SceneObject, bold: float) -> None:
    render = dict((o.meta or {}).get("render") or {})
    text = dict(render.get("text") or {})
    text["bold"] = float(bold)
    render["text"] = text
    o.meta["render"] = render


def _sync_text_render(o: SceneObject) -> None:
    """Keep render hint in sync with refined scene fields (used by PDF text renderer)."""
    render = dict((o.meta or {}).get("render") or {})
    text = dict(render.get("text") or {})
    if o.content is not None:
        text["content"] = o.content
    if o.font_size is not None:
        text["font_size"] = float(o.font_size)
    if o.font_color:
        text["font_color"] = o.font_color
    if o.alignment:
        text["alignment"] = o.alignment
    # Preserve font_family if already set on render hint
    if "font_family" not in text:
        text["font_family"] = "serif"
    render["text"] = text
    transform = dict(render.get("transform") or {})
    transform.update(
        {
            "x": o.x,
            "y": o.y,
            "width": o.width,
            "height": o.height,
            "rotation_deg": o.rotation,
            "opacity": o.opacity,
        }
    )
    render["transform"] = transform
    o.meta["render"] = render


def _ocr_cleanup(text: str) -> str:
    """Light OCR repairs that are safe for body copy."""
    t = text
    t = re.sub(r"\bseo\b", "sea", t, flags=re.IGNORECASE)
    t = re.sub(r"^(\d+)\.(\S)", r"\1. \2", t.strip())
    # Common OCR: wingd / wingèd variants already partially recovered
    t = t.replace("wingéd", "wingèd").replace("winged seraphs", "wingèd seraphs")
    return t


def _unify_band(
    band: list[SceneObject],
    fixes: list[str],
    *,
    snap_bold: bool = True,
    snap_size: bool = True,
) -> None:
    if len(band) < 1:
        return
    sizes = [float(o.font_size or 0) for o in band if o.font_size]
    target_fs = _median(sizes) if sizes else 0.0
    cols = [c for o in band if (c := _parse_hex(o.font_color))]
    if cols:
        lums = [0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2] for c in cols]
        med_l = _median(lums)
        # Dark ink on light pages: prefer darker half (avoids parchment AA washout)
        if med_l < 150:
            ordered = sorted(
                cols, key=lambda c: 0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2]
            )
            cols = ordered[: max(1, (len(ordered) + 1) // 2)]
        mr = _median([c[0] for c in cols])
        mg = _median([c[1] for c in cols])
        mb = _median([c[2] for c in cols])
        # Crush washed brown-gray ink toward true dark for document body
        crush_l = 0.299 * mr + 0.587 * mg + 0.114 * mb
        if crush_l < 130:
            factor = max(0.28, 55.0 / max(crush_l, 1.0))
            mr, mg, mb = mr * factor, mg * factor, mb * factor
        target_color = _hex_from_rgb(mr, mg, mb)
    else:
        target_color = None

    bolds = [_get_bold(o) for o in band]
    med_bold = _median(bolds) if bolds else 0.0
    # Snap: body/title bands should be visually consistent (not flickering)
    if snap_bold:
        target_bold = 1.0 if med_bold >= 0.55 else 0.0
    else:
        target_bold = med_bold

    for o in band:
        if snap_size and target_fs > 0 and o.font_size and abs(float(o.font_size) - target_fs) > 0.4:
            o.font_size = round(target_fs, 2)
            fixes.append(f"norm_size:{o.id}")
        if target_color and o.font_color != target_color:
            # Don't crush white prop-badge text back to black
            if _lum_hex(o.font_color) > 200:
                pass
            else:
                o.font_color = target_color
                fixes.append(f"norm_color:{o.id}")
        if snap_bold and abs(_get_bold(o) - target_bold) > 0.05:
            _set_bold(o, target_bold)
            fixes.append(f"norm_bold:{o.id}")


def refine_scene_objects(
    objects: list[SceneObject],
    document_mode: str | None = None,
) -> list[str]:
    """
    Mutate scene objects in place for tighter alignment and color/size consistency.
    Returns list of applied fix labels.
    """
    fixes: list[str] = []
    texts = [o for o in objects if o.type == SceneObjectType.TEXT and (o.content or "").strip()]
    if len(texts) < 2:
        return fixes

    page_w = max((o.x + o.width for o in texts), default=1000.0)
    page_h = max((o.y + o.height for o in texts), default=1000.0)
    mode = (document_mode or "").strip().lower()
    form_page = mode == "ruled_form" or (not mode and _is_form_like_texts(texts))
    designed_invoice = mode == "designed_invoice"
    # Simple text poster / letter: mostly prose, no invoice/form cues
    blob = " ".join((o.content or "") for o in texts)
    has_deva = any("\u0900" <= ch <= "\u097F" for ch in blob)
    text_poster = (mode == "poster" or not mode) and not form_page and not designed_invoice


    # Hide OCR crumbs (single punctuation / stray letters / illustration noise)
    for o in texts:
        if _is_junk_text(o.content, o.width, o.height):
            o.visibility = False
            fixes.append(f"hide_junk:{o.id}")
            continue
        t = (o.content or "").strip()
        if t in _FORM_KEEP_TOKENS:
            continue
        # On Devanagari pages, hide Latin OCR garbage crumbs
        if has_deva and t and not any("\u0900" <= ch <= "\u097F" for ch in t):
            if len(t) <= 4 or sum(ch.isascii() and ch.isalnum() for ch in t) == len(t.replace(" ", "")):
                # Keep pure punctuation used as bullets only if mid-body and short
                if len(t) <= 3 or re.fullmatch(r"[\W\dA-Za-z♡♥❤$]{1,8}", t or ""):
                    o.visibility = False
                    fixes.append(f"hide_latin_crumb:{o.id}")
                    continue
        # Right-side illustration OCR crumbs (0, 92, &, R, a3, …)
        # Skip on form pages — Rs./P and short labels live on the right
        if form_page:
            continue
        if o.x > page_w * 0.55 and o.width < page_w * 0.15 and o.height < page_h * 0.09:
            if len(t) <= 3 and not (t.isalpha() and len(t) >= 2 and t.isupper() is False and t.islower()):
                # hide non-words / tiny tokens
                if not (t.isalpha() and len(t) >= 3):
                    o.visibility = False
                    fixes.append(f"hide_side_crumb:{o.id}")
    texts = [o for o in texts if o.visibility]

    # Light OCR / list formatting cleanup
    for o in texts:
        raw = o.content or ""
        fixed = _ocr_cleanup(raw)
        if fixed != raw:
            o.content = fixed
            fixes.append(f"ocr_cleanup:{o.id}")

    # Cap font size to box height first (prevents spill into neighbors)
    for o in texts:
        if o.font_size and o.height > 1:
            capped = min(float(o.font_size), float(o.height) * 0.92)
            if capped + 0.5 < float(o.font_size):
                o.font_size = round(capped, 2)
                fixes.append(f"cap_box:{o.id}")

    numbers = [o for o in texts if _is_number(o.content)]
    brands = [o for o in texts if _is_brand(o.content)]
    body = [o for o in texts if o not in numbers and o not in brands]

    # --- Column snap ---
    if numbers and len(numbers) >= 2:
        target_x = _median([o.x for o in numbers])
        for o in numbers:
            if abs(o.x - target_x) > 0.5:
                o.x = round(target_x, 2)
                fixes.append(f"snap_number_col:{o.id}")
        _unify_band(numbers, fixes)

    if body:
        tol = max(22.0, page_w * 0.018)
        clusters = _cluster_by_x(body, tol=tol)
        for cluster in clusters:
            if len(cluster) < 2:
                continue
            # If this looks like a dense column (≥4 lines), snap to shared left edge
            # Skip on form pages — labels sit in different columns
            if len(cluster) >= 4 and not form_page:
                target_x = _median([o.x for o in cluster])
                for o in cluster:
                    if abs(o.x - target_x) > 1.0:
                        o.x = round(target_x, 2)
                        fixes.append(f"snap_text_col:{o.id}")

        # Size bands across all body text (title vs author vs poem body stay separate)
        bands = _cluster_by_size(body, rel_gap=0.22 if form_page else 0.28)
        # Fold thin mid-size bands into the dominant body band (OCR box-height noise)
        if len(bands) >= 2 and not form_page:
            main_idx = max(range(len(bands)), key=lambda i: len(bands[i]))
            main = bands[main_idx]
            main_fs = _median([float(o.font_size or 0) for o in main]) or 1.0
            folded: list[list[SceneObject]] = []
            for i, band in enumerate(bands):
                if i == main_idx:
                    folded.append(band)
                    continue
                if len(band) >= max(5, len(body) // 4):
                    folded.append(band)
                    continue
                bfs = _median([float(o.font_size or 0) for o in band]) or 0.0
                # Keep clear titles (much larger) and captions (much smaller)
                if bfs >= main_fs * 1.55 or bfs <= main_fs * 0.72:
                    folded.append(band)
                    continue
                main.extend(band)
                fixes.append(f"fold_mid_band:{len(band)}")
            bands = folded
            # Re-find main after folds
            bands = [b for b in bands if b]

        for band in bands:
            if len(band) >= 2:
                # Forms: only unify color/bold lightly; keep per-label sizes from boxes
                _unify_band(band, fixes, snap_bold=not form_page, snap_size=not form_page)
                if form_page:
                    for o in band:
                        if o.font_size and o.height > 1:
                            o.font_size = round(min(float(o.font_size), float(o.height) * 0.88), 2)
            elif len(band) == 1:
                pass

    if form_page:
        # Keep small top-left logos / hybrid underlays; hide large table raster patches
        for o in objects:
            if o.type in {SceneObjectType.IMAGE, SceneObjectType.LOGO, SceneObjectType.ICON}:
                is_hybrid = bool((o.meta or {}).get("hybrid_underlay"))
                is_logo = (
                    o.y < page_h * 0.22
                    and o.x < page_w * 0.4
                    and o.width < page_w * 0.4
                    and o.height < page_h * 0.22
                ) or (is_hybrid and o.height < page_h * 0.28)
                if not is_logo:
                    o.visibility = False
                    fixes.append(f"form_hide_image:{o.id}")
                else:
                    fixes.append(f"form_keep_logo:{o.id}")
        # Force pure black text (user requirement for bill books), except white-on-black banners
        for o in texts:
            t = (o.content or "").strip()
            low = t.lower()
            # Detect white-on-black banner / prop badge by sampled color already light
            on_dark = _lum_hex(o.font_color) > 180 and (
                low.startswith("prop")
                or "service station" in low
                or "repair" in low
                or "reparing" in low
            )
            if on_dark or low.startswith("prop"):
                # Prop pill / black banner → white ink
                if low.startswith("prop") or "service station" in low or "repair" in low or "reparing" in low:
                    # Confirm via size: banners are wide
                    if o.width > page_w * 0.35 or low.startswith("prop"):
                        o.font_color = "#FFFFFF"
                        fixes.append(f"form_white_on_dark:{o.id}")
                        if low.startswith("prop"):
                            o.alignment = "center"
                        continue
            o.font_color = "#000000"
            fixes.append(f"form_black:{o.id}")

        # Center business title + memo title; keep table headers centered in their boxes
        for o in texts:
            t = (o.content or "").strip().lower()
            if any(
                k in t
                for k in (
                    "enterprise",
                    "bill/cash",
                    "bill / cash",
                    "cash memo",
                    "invoice",
                    "distributer",
                    "distributor",
                    "motors",
                    "service station",
                )
            ) or (t.isupper() and len(t) >= 8 and o.y < page_h * 0.28):
                o.alignment = "center"
                fixes.append(f"form_center:{o.id}")
            if t in {
                "particulars",
                "quantity",
                "rate",
                "amount",
                "sl.no",
                "s.no",
                "s.no.",
                "total",
                "rs.",
                "rs",
                "p",
                "p.",
                "type",
            } or t.startswith("meter"):
                o.alignment = "center"
                fixes.append(f"form_th_center:{o.id}")
        # Address underlines: keep left labels left-aligned
    elif designed_invoice:
        # Keep hybrid underlays; do not force black (preserve brand colors)
        for o in objects:
            if o.type in {SceneObjectType.IMAGE, SceneObjectType.LOGO} and (o.meta or {}).get(
                "hybrid_underlay"
            ):
                o.visibility = True
                fixes.append(f"invoice_keep_underlay:{o.id}")
            elif o.type == SceneObjectType.IMAGE:
                # Hide huge mid-page rasters that ghost over editable text
                if o.width * o.height > page_w * page_h * 0.35 and o.y > page_h * 0.15:
                    o.visibility = False
                    fixes.append(f"invoice_hide_big_image:{o.id}")
        for o in texts:
            t = (o.content or "").strip().lower()
            if any(
                k in t
                for k in (
                    "customer",
                    "address",
                    "lorry",
                    "date",
                    "signature",
                    "amount in word",
                    "bill no",
                    "vehicle",
                    "total amount",
                    "e.&o.e",
                )
            ):
                o.alignment = "left"
            raw = o.content or ""
            if "rampurchak" in raw.lower() and "shyamchak" in raw.lower():
                cleaned = re.sub(r"[★☆□■▪▫⦁•\*\x00]+", "*", raw)
                cleaned = re.sub(r"\s{2,}", " * ", cleaned.strip())
                cleaned = re.sub(
                    r"(?i)(rampurchak)\s*[\*\-]?\s*(shyamchak)\s*[\*\-]?\s*(paschim)",
                    r"\1 * \2 * \3",
                    cleaned,
                )
                if cleaned != raw:
                    o.content = cleaned
                    fixes.append(f"form_stars:{o.id}")
            if raw.strip().lower().startswith("address"):
                fixed_addr = re.sub(r"(?i)^address\.+$", "Address", raw.strip())
                if fixed_addr != raw:
                    o.content = fixed_addr
                    fixes.append(f"form_address:{o.id}")
            # OCR tidy for numbered terms
            fixed = re.sub(r"^(\d)([A-Za-z])", r"\1. \2", raw.strip())
            fixed = fixed.replace("Authorised Signatory_", "Authorised Signatory")
            if fixed != raw:
                o.content = fixed
                fixes.append(f"form_ocr_tidy:{o.id}")
        fixes.append("form_refine")

    elif text_poster:
        # Clean B&W text pages: no grey panels / photo ghosts
        for o in objects:
            if o.type in {
                SceneObjectType.IMAGE,
                SceneObjectType.LOGO,
                SceneObjectType.ICON,
                SceneObjectType.BACKGROUND,
            }:
                o.visibility = False
                fixes.append(f"text_hide_raster:{o.id}")
            if o.type in {
                SceneObjectType.RECTANGLE,
                SceneObjectType.ROUNDED_RECTANGLE,
            }:
                fill = (o.fill_color or "").upper()
                # Drop washed grey fill boxes (false panels)
                if fill in {"#F0F0F0", "#E8E8E8", "#EEEEEE", "#F5F5F5", "#E0E0E0"} or (
                    fill.startswith("#")
                    and len(fill) >= 7
                    and fill[1:3] == fill[3:5] == fill[5:7]
                    and int(fill[1:3], 16) >= 200
                ):
                    o.visibility = False
                    fixes.append(f"text_hide_gray:{o.id}")

        # Title = top-most relatively large short line → center
        ordered = sorted(texts, key=lambda o: o.y)
        title = None
        if ordered:
            title = ordered[0]
            body_sizes = [float(o.font_size or o.height or 12) for o in ordered[1:6]] or [
                float(title.font_size or 12)
            ]
            med_body = sorted(body_sizes)[len(body_sizes) // 2]
            if float(title.font_size or title.height or 0) >= med_body * 1.15 or len(
                (title.content or "").split()
            ) <= 4:
                title.alignment = "center"
                # Recenter box around page
                cx = page_w / 2.0
                title.x = round(max(0.0, cx - title.width / 2.0), 2)
                # Match book-style title weight
                if has_deva:
                    _set_bold(title, max(_get_bold(title), 0.85))
                fixes.append(f"text_title_center:{title.id}")

        # Body: shared left/right margins; Hindi letters → justify like original
        body_lines = [o for o in ordered[1:] if o.y < page_h * 0.88]
        # Keep a clear title→body gap (original Hindi letters have generous spacing)
        if title and body_lines and has_deva:
            min_gap = max(float(title.height or 20) * 1.35, page_h * 0.035)
            first = body_lines[0]
            cur_gap = float(first.y) - (float(title.y) + float(title.height))
            if cur_gap < min_gap:
                shift = min_gap - cur_gap
                for o in body_lines:
                    o.y = round(float(o.y) + shift, 2)
                fixes.append(f"text_title_gap:{round(shift,1)}")

        if len(body_lines) >= 3:
            left = _median([o.x for o in body_lines])
            right = _median([o.x + o.width for o in body_lines])
            for o in body_lines:
                if abs(o.x - left) <= page_w * 0.08:
                    if abs(o.x - left) > 1.0:
                        o.x = round(left, 2)
                        fixes.append(f"text_body_left:{o.id}")
                    # Stretch line boxes to common right edge for justified look
                    if has_deva and right - left > page_w * 0.55:
                        o.width = round(max(float(o.width), right - left), 2)
                        o.alignment = "justify"
                    else:
                        o.alignment = "left"
        for o in texts:
            t = (o.content or "").strip()
            if t.startswith("-") or t.startswith("–") or t.startswith("—"):
                o.alignment = "right"
                fixes.append(f"text_sig_right:{o.id}")
            # Force near-black ink on B&W letters
            if _lum_hex(o.font_color) < 140:
                if o.font_color != "#111111":
                    o.font_color = "#111111"
                    fixes.append(f"text_ink_black:{o.id}")
        fixes.append("text_poster_refine")

    # Cap font size by gap to next line in same column (stop visual overlap)
    col_tol = max(30.0, page_w * 0.03)
    for cluster in _cluster_by_x(texts, tol=col_tol):
        ordered = sorted(cluster, key=lambda o: o.y)
        for i, o in enumerate(ordered):
            if not o.font_size:
                continue
            if i + 1 < len(ordered):
                nxt = ordered[i + 1]
                gap = float(nxt.y) - float(o.y)
                if gap > 4:
                    capped = min(float(o.font_size), gap * 0.88)
                    if capped + 0.5 < float(o.font_size):
                        o.font_size = round(max(capped, float(o.height) * 0.55), 2)
                        fixes.append(f"cap_gap:{o.id}")

    # Center short header lines near top of page
    top = [o for o in texts if o.y < page_h * 0.28 and len((o.content or "").split()) <= 8]
    page_cx = page_w / 2.0
    for o in top:
        cx = o.x + o.width / 2.0
        if abs(cx - page_cx) <= page_w * 0.12:
            if o.alignment != "center":
                o.alignment = "center"
                fixes.append(f"align_center:{o.id}")
    # Harmonize top title/author colors when both dark
    headerish = sorted(texts, key=lambda o: o.y)[:3]
    dark_top = [o for o in headerish if _lum_hex(o.font_color) < 90 and len((o.content or "").split()) <= 6]
    if len(dark_top) >= 2 and not form_page:
        target = "#1A1A1A"
        for o in dark_top:
            if o.font_color != target:
                o.font_color = target
                fixes.append(f"norm_header_black:{o.id}")

    # Skip unicode heart append — many TTF handwriting faces lack ♡; vectors draw it instead

    # Brand / footer right align (HOWIE CHAN style)
    for o in texts:
        if _is_brand(o.content) and "♡" not in (o.content or ""):
            if o.alignment != "right":
                o.alignment = "right"
                fixes.append(f"align_right:{o.id}")

    # Numbers + body row vertical center (list posters)
    if numbers and body:
        for num in numbers:
            ny1, ny2 = num.y, num.y + num.height
            row = [
                o
                for o in body
                if not (o.y + o.height < ny1 - 8 or o.y > ny2 + 8)
            ]
            if not row:
                continue
            row_top = min(o.y for o in row)
            row_bot = max(o.y + o.height for o in row)
            ideal_y = row_top + ((row_bot - row_top) - num.height) / 2.0
            if abs(num.y - ideal_y) > 2 and abs(num.y - ideal_y) < num.height:
                num.y = round(ideal_y, 2)
                fixes.append(f"vcenter_number:{num.id}")

    for o in texts:
        # Kill false underline from serif baselines unless very confident
        render = dict((o.meta or {}).get("render") or {})
        text = dict(render.get("text") or {})
        if float(text.get("underline") or 0) < 0.88:
            text["underline"] = 0.0
            render["text"] = text
            o.meta["render"] = render
        _sync_text_render(o)

    return fixes


def refine_vector_separators(
    vectors: list[dict[str, Any]],
    scene_objects: list[dict[str, Any]],
    page: dict[str, Any],
) -> list[str]:
    """
    Snap thin vertical accent bars to a shared x and place them between
    number column and body text column.
    """
    fixes: list[str] = []
    sx = float(page.get("scale_x") or 1.0) or 1.0
    ox = float(page.get("offset_x") or 0.0)

    texts = [o for o in scene_objects if str(o.get("type")) == "TEXT"]
    numbers = [o for o in texts if _is_number(str(o.get("content") or ""))]
    body = [
        o
        for o in texts
        if not _is_number(str(o.get("content") or ""))
        and not _is_brand(str(o.get("content") or ""))
        and float(o.get("font_size") or 0) < 140
    ]

    bars = []
    for v in vectors:
        w = float(v.get("width") or 0)
        h = float(v.get("height") or 0)
        bbox = v.get("bbox") or v.get("source_bbox")
        if bbox and len(bbox) >= 4:
            bw = float(bbox[2]) - float(bbox[0]) if bbox[2] > bbox[0] else float(v.get("width") or 0)
            bh = float(bbox[3]) - float(bbox[1]) if bbox[3] > bbox[1] else float(v.get("height") or 0)
        else:
            bw, bh = w, h
        if bw <= 0 or bh <= 0:
            continue
        if bw <= 14 and bh >= bw * 3:
            bars.append(v)

    if len(bars) < 2:
        return fixes

    # Only snap when a clear number→body gutter exists (poster lists).
    # Never collapse every vertical rule to one x (destroys bill/invoice grids).
    target_src_x = None
    if numbers and body:
        num_right = max(
            (float(o.get("x") or 0) + float(o.get("width") or 0) - ox) / sx for o in numbers
        )
        body_left = min((float(o.get("x") or 0) - ox) / sx for o in body)
        if body_left > num_right + 8:
            target_src_x = (num_right + body_left) / 2.0
    if target_src_x is None:
        return fixes

    xs = []
    for v in bars:
        bbox = v.get("bbox") or v.get("source_bbox")
        if bbox and len(bbox) >= 4:
            xs.append(float(bbox[0]))
        elif "x" in v:
            xs.append((float(v["x"]) - ox) / sx)
    if not xs:
        return fixes

    shared = target_src_x
    for v in bars:
        bbox = v.get("bbox")
        if bbox and len(bbox) >= 4:
            bw = float(bbox[2]) - float(bbox[0])
            # Only snap bars already near the gutter
            mid = (float(bbox[0]) + float(bbox[2])) / 2.0
            if abs(mid - shared) > 40:
                continue
            v["bbox"] = [shared, float(bbox[1]), shared + bw, float(bbox[3])]
            fixes.append("snap_accent_bar")
        elif "x" in v:
            v["x"] = shared * sx + ox
            fixes.append("snap_accent_bar")

    return fixes
