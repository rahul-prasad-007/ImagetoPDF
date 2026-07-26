"""
Renderer-facing descriptors derived from the scene graph.

These describe *how* a future PDF/SVG renderer should paint objects.
This module does NOT render or export anything.
"""

from __future__ import annotations

from enum import Enum
from typing import List, Optional, Tuple

from pydantic import BaseModel, Field


class PaintMode(str, Enum):
    FILL = "fill"
    STROKE = "stroke"
    FILL_STROKE = "fill_stroke"
    NONE = "none"


class BlendMode(str, Enum):
    NORMAL = "normal"
    MULTIPLY = "multiply"
    SCREEN = "screen"
    OVERLAY = "overlay"


class Transform2D(BaseModel):
    """Affine-ready transform (degrees, origin top-left)."""

    x: float = 0.0
    y: float = 0.0
    width: float = 0.0
    height: float = 0.0
    rotation_deg: float = 0.0
    scale_x: float = 1.0
    scale_y: float = 1.0
    opacity: float = 1.0


class PaintStyle(BaseModel):
    fill_color: Optional[str] = None
    stroke_color: Optional[str] = None
    stroke_width: float = 0.0
    corner_radius: float = 0.0
    paint_mode: PaintMode = PaintMode.FILL
    blend_mode: BlendMode = BlendMode.NORMAL


class TextStyleSpec(BaseModel):
    content: str = ""
    font_size: float = 12.0
    font_color: str = "#000000"
    alignment: str = "left"
    bold: float = 0.0
    italic: float = 0.0
    underline: float = 0.0
    line_height: float = 1.2
    character_spacing: float = 0.0
    word_spacing: float = 0.0
    paragraph_id: Optional[int] = None
    font_family: str = "serif"


class ImageRefSpec(BaseModel):
    """Reference to a raster region (crop in source pixel space)."""

    image_path: str
    crop_x: float = 0.0
    crop_y: float = 0.0
    crop_width: float = 0.0
    crop_height: float = 0.0
    scale: float = 1.0
    preserve_aspect: bool = True


class PathSpec(BaseModel):
    """Future path payload (commands deferred until SVG/PDF export)."""

    points: List[Tuple[float, float]] = Field(default_factory=list)
    closed: bool = False
    d: Optional[str] = None  # optional SVG-path string placeholder


class RenderNodeHint(BaseModel):
    """
    Flat hint a future renderer can consume without re-deriving paint rules.
    Attached under scene object meta.render when useful.
    """

    scene_object_id: int
    object_type: str
    transform: Transform2D
    paint: Optional[PaintStyle] = None
    text: Optional[TextStyleSpec] = None
    image: Optional[ImageRefSpec] = None
    path: Optional[PathSpec] = None
    layer: int = 0
    z_order: int = 0
