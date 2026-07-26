"""
Font manager — resolve editable PDF fonts by family (serif / sans / handwriting).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

logger = logging.getLogger(__name__)

# backend/ root (app/pdf/font_manager.py → ../../..)
_BACKEND_ROOT = Path(__file__).resolve().parent.parent.parent
_PROJECT_FONTS = _BACKEND_ROOT / "fonts"

# Prefer bundled / drop-in AAText for all Hindi documents
_AATEXT_CANDIDATES = [
    str(_PROJECT_FONTS / "AAText.ttf"),
    str(_PROJECT_FONTS / "AAText.otf"),
    str(_PROJECT_FONTS / "aatext.ttf"),
    str(_PROJECT_FONTS / "AA_NAGARI_SHREE_L2.ttf"),
    str(_PROJECT_FONTS / "AA_NAGARI_SHREE_L1.ttf"),
    "C:/Windows/Fonts/AAText.ttf",
    "C:/Windows/Fonts/aatext.ttf",
]

# Family → style → candidate paths (Windows + common installs)
_FONT_CANDIDATES: dict[str, dict[str, list[str]]] = {
    "devanagari": {
        "regular": [
            *_AATEXT_CANDIDATES,
            "C:/Windows/Fonts/Nirmala.ttc",
            "C:/Windows/Fonts/mangal.ttf",
            "C:/Windows/Fonts/Mangal.ttf",
            "C:/Windows/Fonts/aparaj.ttf",
            "/usr/share/fonts/truetype/noto/NotoSansDevanagari-Regular.ttf",
        ],
        "bold": [
            str(_PROJECT_FONTS / "AAText-Bold.ttf"),
            *_AATEXT_CANDIDATES,
            "C:/Windows/Fonts/NirmalaB.ttf",
            "C:/Windows/Fonts/Nirmala.ttc",
            "C:/Windows/Fonts/mangalb.ttf",
            "C:/Windows/Fonts/aparajb.ttf",
            "/usr/share/fonts/truetype/noto/NotoSansDevanagari-Bold.ttf",
        ],
        "italic": [
            *_AATEXT_CANDIDATES,
            "C:/Windows/Fonts/Nirmala.ttc",
            "C:/Windows/Fonts/mangal.ttf",
        ],
        "bold_italic": [
            *_AATEXT_CANDIDATES,
            "C:/Windows/Fonts/Nirmala.ttc",
            "C:/Windows/Fonts/mangalb.ttf",
        ],
    },
    "serif": {
        "regular": [
            "C:/Windows/Fonts/times.ttf",
            "C:/Windows/Fonts/georgia.ttf",
            "C:/Windows/Fonts/GEORGIA.TTF",
            "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf",
        ],
        "bold": [
            "C:/Windows/Fonts/timesbd.ttf",
            "C:/Windows/Fonts/georgiab.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf",
        ],
        "italic": [
            "C:/Windows/Fonts/timesi.ttf",
            "C:/Windows/Fonts/georgiai.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Italic.ttf",
        ],
        "bold_italic": [
            "C:/Windows/Fonts/timesbi.ttf",
            "C:/Windows/Fonts/georgiaz.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSerif-BoldItalic.ttf",
        ],
    },
    "sans": {
        "regular": [
            "C:/Windows/Fonts/arial.ttf",
            "C:/Windows/Fonts/ARIAL.TTF",
            "C:/Windows/Fonts/segoeui.ttf",
            "C:/Windows/Fonts/calibri.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        ],
        "bold": [
            "C:/Windows/Fonts/arialbd.ttf",
            "C:/Windows/Fonts/ARIALBD.TTF",
            "C:/Windows/Fonts/segoeuib.ttf",
            "C:/Windows/Fonts/calibrib.ttf",
        ],
        "italic": [
            "C:/Windows/Fonts/ariali.ttf",
            "C:/Windows/Fonts/ARIALI.TTF",
            "C:/Windows/Fonts/segoeuii.ttf",
        ],
        "bold_italic": [
            "C:/Windows/Fonts/arialbi.ttf",
            "C:/Windows/Fonts/ARIALBI.TTF",
            "C:/Windows/Fonts/segoeuiz.ttf",
        ],
    },
    "handwriting": {
        "regular": [
            "C:/Windows/Fonts/Inkfree.ttf",
            "C:/Windows/Fonts/segoepr.ttf",
            "C:/Windows/Fonts/comic.ttf",
            "C:/Windows/Fonts/LHANDW.TTF",
            "C:/Windows/Fonts/FRSCRIPT.TTF",
            "C:/Windows/Fonts/BRUSHSCI.TTF",
        ],
        "bold": [
            "C:/Windows/Fonts/segoeprb.ttf",
            "C:/Windows/Fonts/comicbd.ttf",
            "C:/Windows/Fonts/Inkfree.ttf",
            "C:/Windows/Fonts/segoepr.ttf",
        ],
        "italic": [
            "C:/Windows/Fonts/comici.ttf",
            "C:/Windows/Fonts/Inkfree.ttf",
            "C:/Windows/Fonts/segoepr.ttf",
            "C:/Windows/Fonts/FRSCRIPT.TTF",
        ],
        "bold_italic": [
            "C:/Windows/Fonts/comicz.ttf",
            "C:/Windows/Fonts/segoeprb.ttf",
            "C:/Windows/Fonts/Inkfree.ttf",
        ],
    },
}

_REGISTERED = False
_FONT_MAPS: dict[str, dict[str, str]] = {}
_MISSING: list[str] = []


def _try_register(name: str, paths: list[str]) -> Optional[str]:
    for p in paths:
        path = Path(p)
        if not path.is_file():
            continue
        try:
            if path.suffix.lower() == ".ttc":
                # Nirmala.ttc and similar collections — try first faces
                registered = None
                for idx in range(0, 4):
                    face = f"{name}" if idx == 0 else f"{name}-{idx}"
                    try:
                        pdfmetrics.registerFont(TTFont(face, str(path), subfontIndex=idx))
                        registered = face if idx == 0 else registered or face
                        if idx == 0:
                            break
                    except Exception:
                        continue
                if registered:
                    logger.info("Registered font %s -> %s", registered, path)
                    return registered
                continue
            pdfmetrics.registerFont(TTFont(name, str(path)))
            logger.info("Registered font %s -> %s", name, path)
            return name
        except Exception as exc:
            logger.warning("Failed registering %s from %s: %s", name, path, exc)
    return None


def ensure_fonts() -> dict[str, dict[str, str]]:
    """Register TrueType fonts once for each family."""
    global _REGISTERED, _FONT_MAPS, _MISSING
    if _REGISTERED:
        return _FONT_MAPS

    for family, styles in _FONT_CANDIDATES.items():
        prefix = f"ImgToPdf-{family}"
        regular = _try_register(f"{prefix}", styles["regular"])
        bold = _try_register(f"{prefix}-Bold", styles["bold"])
        italic = _try_register(f"{prefix}-Italic", styles["italic"])
        bold_italic = _try_register(f"{prefix}-BoldItalic", styles["bold_italic"])

        if not regular:
            _MISSING.append(family)
            # Fallbacks
            if family == "handwriting":
                regular = _FONT_MAPS.get("sans", {}).get("regular") or "Helvetica"
            elif family == "devanagari":
                regular = _FONT_MAPS.get("sans", {}).get("regular") or "Helvetica"
            elif family == "sans":
                regular = "Helvetica"
            else:
                regular = "Times-Roman"
        if not bold:
            bold = regular
        if not italic:
            italic = regular
        if not bold_italic:
            bold_italic = bold

        _FONT_MAPS[family] = {
            "regular": regular,
            "bold": bold,
            "italic": italic,
            "bold_italic": bold_italic,
        }

    # Back-compat aliases used by older callers
    serif = _FONT_MAPS.get("serif") or {}
    _FONT_MAPS["default"] = dict(serif)

    _REGISTERED = True
    return _FONT_MAPS


def normalize_font_family(family: str | None) -> str:
    f = (family or "serif").strip().lower()
    if f in {
        "devanagari",
        "hindi",
        "hi",
        "nirmala",
        "mangal",
        "indic",
        "aatext",
        "aa text",
        "aa_text",
        "aanagari",
        "aa_nagari",
    }:
        return "devanagari"
    if f in {"hand", "script", "cursive", "handwritten", "handwriting", "marker", "ink"}:
        return "handwriting"
    if f in {"sans", "sans-serif", "sansserif", "arial", "helvetica"}:
        return "sans"
    if f in {"serif", "times", "georgia", "roman"}:
        return "serif"
    return "serif"


def resolve_aatext_font_path() -> Optional[Path]:
    """Absolute path to AAText (or bundled AA Nagari) for PyMuPDF / CSS @font-face."""
    for p in _AATEXT_CANDIDATES:
        path = Path(p)
        if path.is_file():
            return path.resolve()
    return None


def resolve_font(
    bold: float = 0.0,
    italic: float = 0.0,
    family: str | None = None,
) -> str:
    maps = ensure_fonts()
    fam = normalize_font_family(family)
    fonts = maps.get(fam) or maps.get("serif") or {"regular": "Helvetica"}
    is_bold = bold >= 0.45
    italic_cut = 0.55 if fam == "handwriting" else 0.72
    is_italic = italic >= italic_cut
    if is_bold and is_italic:
        return fonts["bold_italic"]
    if is_bold:
        return fonts["bold"]
    if is_italic:
        return fonts["italic"]
    return fonts["regular"]


def missing_font_count() -> int:
    ensure_fonts()
    return len(_MISSING)
