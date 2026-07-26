"""Optional PP-StructureV3 table/layout assist (free, local; first run may download models)."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

_pipeline = None
_init_failed = False


def _get_pipeline(enabled: bool):
    global _pipeline, _init_failed
    if not enabled or _init_failed:
        return None
    if _pipeline is not None:
        return _pipeline
    try:
        from paddleocr import PPStructureV3

        # Skip seals/charts/formulas for faster CPU deploy
        _pipeline = PPStructureV3(
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_seal_recognition=False,
            use_formula_recognition=False,
            use_chart_recognition=False,
            use_table_recognition=True,
        )
        logger.info("PPStructureV3 pipeline ready")
        return _pipeline
    except Exception as exc:
        _init_failed = True
        logger.warning("PPStructureV3 unavailable (%s) — continuing without it", exc)
        return None


def run_pp_structure(
    image_bgr: np.ndarray,
    *,
    enabled: bool = False,
) -> dict[str, Any]:
    """
    Run PP-StructureV3 and return normalized tables + layout boxes.
    Always returns a dict; never raises to callers.
    """
    empty: dict[str, Any] = {
        "enabled": enabled,
        "ok": False,
        "tables": [],
        "regions": [],
        "error": None,
    }
    if not enabled:
        empty["error"] = "disabled"
        return empty

    pipe = _get_pipeline(True)
    if pipe is None:
        empty["error"] = "init_failed"
        return empty

    try:
        rgb = image_bgr[:, :, ::-1]
        results = pipe.predict(rgb)
        tables: list[dict[str, Any]] = []
        regions: list[dict[str, Any]] = []

        pages = results if isinstance(results, list) else [results]
        for page in pages:
            data: Any = page
            if hasattr(page, "json"):
                try:
                    raw = page.json
                    data = raw() if callable(raw) else raw
                except Exception:
                    data = page
            if hasattr(page, "res"):
                data = getattr(page, "res")
            if not isinstance(data, dict):
                try:
                    data = dict(page)  # type: ignore[arg-type]
                except Exception:
                    data = {"raw_type": type(page).__name__}

            for key in ("layout_det_res", "parsing_res_list", "layout", "regions"):
                items = data.get(key) if isinstance(data, dict) else None
                if not items:
                    continue
                if isinstance(items, dict) and "boxes" in items:
                    items = items.get("boxes") or []
                for item in items if isinstance(items, list) else []:
                    box = None
                    label = None
                    if isinstance(item, dict):
                        box = item.get("coordinate") or item.get("bbox") or item.get("box")
                        label = item.get("label") or item.get("type") or item.get("cls_id")
                    if box is not None and len(box) >= 4:
                        regions.append(
                            {
                                "label": str(label or "region"),
                                "bbox": [
                                    float(box[0]),
                                    float(box[1]),
                                    float(box[2]),
                                    float(box[3]),
                                ],
                            }
                        )

            for key in ("table_res_list", "tables", "table"):
                items = data.get(key) if isinstance(data, dict) else None
                if not items:
                    continue
                if not isinstance(items, list):
                    items = [items]
                for t in items:
                    if not isinstance(t, dict):
                        continue
                    bbox = t.get("bbox") or t.get("coordinate") or t.get("box")
                    html = t.get("pred_html") or t.get("html") or t.get("table_html")
                    tables.append(
                        {
                            "bbox": (
                                [float(v) for v in bbox[:4]]
                                if bbox and len(bbox) >= 4
                                else None
                            ),
                            "html": html,
                            "cells": t.get("cell_box_list") or t.get("cells"),
                        }
                    )

        return {
            "enabled": True,
            "ok": True,
            "tables": tables,
            "regions": regions,
            "error": None,
        }
    except Exception as exc:
        logger.warning("PPStructureV3 predict failed: %s", exc)
        return {
            "enabled": True,
            "ok": False,
            "tables": [],
            "regions": [],
            "error": str(exc),
        }
