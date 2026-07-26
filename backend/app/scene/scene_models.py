"""
Scene graph models — editable document tree for a future PDF renderer.

No PDF / SVG / CDR export in this phase.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field


class SceneObjectType(str, Enum):
    TEXT = "TEXT"
    RECTANGLE = "RECTANGLE"
    ROUNDED_RECTANGLE = "ROUNDED_RECTANGLE"
    LINE = "LINE"
    ELLIPSE = "ELLIPSE"
    CIRCLE = "CIRCLE"
    POLYGON = "POLYGON"
    PATH = "PATH"
    IMAGE = "IMAGE"
    LOGO = "LOGO"
    ICON = "ICON"
    GROUP = "GROUP"
    BACKGROUND = "BACKGROUND"


class PageFormat(str, Enum):
    A4 = "A4"
    LETTER = "LETTER"
    CUSTOM = "CUSTOM"


class PageOrientation(str, Enum):
    PORTRAIT = "portrait"
    LANDSCAPE = "landscape"


class SceneRequest(BaseModel):
    image_id: str = Field(..., min_length=8)


class Margins(BaseModel):
    top: float = 0.0
    right: float = 0.0
    bottom: float = 0.0
    left: float = 0.0


class ScenePage(BaseModel):
    width: float
    height: float
    source_width: float
    source_height: float
    margins: Margins = Field(default_factory=Margins)
    orientation: PageOrientation = PageOrientation.PORTRAIT
    page_format: PageFormat = PageFormat.A4
    dpi: int = 300
    scale_x: float = 1.0
    scale_y: float = 1.0
    offset_x: float = 0.0
    offset_y: float = 0.0


class SceneLayer(BaseModel):
    id: int
    name: str
    order: int
    object_ids: List[int] = Field(default_factory=list)


class SceneObject(BaseModel):
    id: int
    parent: Optional[int] = None
    children: List[int] = Field(default_factory=list)
    layer: int
    type: SceneObjectType
    # Flat geometry (renderer-friendly)
    x: float
    y: float
    width: float
    height: float
    rotation: float = 0.0
    opacity: float = 1.0
    visibility: bool = True
    locked: bool = False
    # Optional typed payloads
    content: Optional[str] = None
    font_size: Optional[float] = None
    font_color: Optional[str] = None
    alignment: Optional[str] = None
    paragraph: Optional[int] = None
    fill_color: Optional[str] = None
    stroke_color: Optional[str] = None
    stroke_width: Optional[float] = None
    corner_radius: Optional[float] = None
    image_path: Optional[str] = None
    crop: Optional[Dict[str, float]] = None
    scale: Optional[float] = None
    name: Optional[str] = None
    source: Dict[str, Any] = Field(default_factory=dict)
    meta: Dict[str, Any] = Field(default_factory=dict)


class SceneValidationIssue(BaseModel):
    code: str
    message: str
    object_id: Optional[int] = None


class SceneValidationReport(BaseModel):
    ok: bool = True
    issues: List[SceneValidationIssue] = Field(default_factory=list)
    duplicate_ids: int = 0
    missing_parents: int = 0
    negative_sizes: int = 0
    invalid_coordinates: int = 0
    overlapping_layer_pairs: int = 0


class SceneCounts(BaseModel):
    total_objects: int = 0
    groups: int = 0
    layers: int = 0
    text_objects: int = 0
    image_objects: int = 0
    vector_objects: int = 0
    background_objects: int = 0


class SceneSummary(BaseModel):
    counts: SceneCounts
    validation: SceneValidationReport
    memory_kb: float = 0.0
    build_time_ms: float = 0.0


class SceneSuccessResponse(BaseModel):
    success: bool = True
    image_id: str
    page: ScenePage
    layers: List[SceneLayer]
    objects: List[SceneObject]
    summary: SceneSummary
    processing_time_ms: float
    results_file: Optional[str] = None
    debug_image: Optional[str] = None
    message: str = "Scene graph built successfully."
    document_mode: Optional[str] = None
