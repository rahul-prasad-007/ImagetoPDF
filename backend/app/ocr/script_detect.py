"""
Script / language helpers for OCR engine selection.
"""

from __future__ import annotations

import re
from typing import Any, Optional

# Devanagari block + common signs
_DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")
_LATIN_RE = re.compile(r"[A-Za-z]")


def has_devanagari(text: str) -> bool:
    return bool(_DEVANAGARI_RE.search(text or ""))


def is_mostly_devanagari(text: str) -> bool:
    t = (text or "").strip()
    if not t:
        return False
    dev = len(_DEVANAGARI_RE.findall(t))
    lat = len(_LATIN_RE.findall(t))
    return dev >= 2 and dev >= lat


def ocr_blocks_look_like_wrong_script(blocks: list[Any]) -> bool:
    """
    English OCR on Hindi pages tends to emit short garbage tokens,
    digits, and Latin fragments with low mean length.
    """
    if not blocks:
        return True
    texts: list[str] = []
    for b in blocks:
        if isinstance(b, dict):
            texts.append(str(b.get("text") or ""))
        else:
            texts.append(str(getattr(b, "text", "") or ""))
    if not texts:
        return True

    join = " ".join(texts)
    if has_devanagari(join):
        return False

    short = sum(1 for t in texts if len(t.strip()) <= 2)
    digitish = sum(1 for t in texts if sum(ch.isdigit() for ch in t) >= max(1, len(t) // 2))
    avg_len = sum(len(t.strip()) for t in texts) / max(1, len(texts))
    confs = []
    for b in blocks:
        if isinstance(b, dict):
            confs.append(float(b.get("confidence") or 0))
        else:
            confs.append(float(getattr(b, "confidence", 0) or 0))
    avg_conf = sum(confs) / max(1, len(confs))

    # Strong signals of wrong-script OCR
    if short / len(texts) >= 0.45 and avg_len < 6:
        return True
    if digitish / len(texts) >= 0.35 and avg_len < 8:
        return True
    if avg_conf < 0.72 and avg_len < 7 and short / len(texts) >= 0.3:
        return True
    return False


def choose_ocr_lang(
    settings_lang: str,
    *,
    blocks: Optional[list[Any]] = None,
) -> str:
    """
    Resolve OCR language.
    settings_lang: auto | en | hi | ...
    """
    lang = (settings_lang or "auto").strip().lower()
    if lang in {"en", "english"}:
        return "en"
    if lang in {"hi", "hindi", "devanagari", "mr", "ne", "sa"}:
        return "hi"
    # auto
    if blocks is not None and ocr_blocks_look_like_wrong_script(blocks):
        return "hi"
    return "en"
