"""
Vector reconstruction models — editable graphical elements for a future renderer.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class VectorType(str, Enum):
    RECTANGLE = "RECTANGLE"
    ROUNDED_RECTANGLE = "ROUNDED_RECTANGLE"
    LINE = "LINE"
    BORDER = "BORDER"
    PANEL = "PANEL"
    COLOR_REGION = "COLOR_REGION"
    CIRCLE = "CIRCLE"
    ELLIPSE = "ELLIPSE"
    POLYGON = "POLYGON"
    TRIANGLE = "TRIANGLE"
    ARROW = "ARROW"
    RIBBON = "RIBBON"
    WAVE = "WAVE"
    CURVED_BAND = "CURVED_BAND"
    PATH = "PATH"
    HEART = "HEART"
    GRADIENT_REGION = "GRADIENT_REGION"


class GradientKind(str, Enum):
    LINEAR = "LINEAR"
    RADIAL = "RADIAL"
    NONE = "NONE"


class VectorRequest(BaseModel):
    image_id: str = Field(..., min_length=8)


class ControlPoint(BaseModel):
    x: float
    y: float


class PathData(BaseModel):
    commands: str = Field(description="SVG path command string (data only, not exported)")
    control_points: List[ControlPoint] = Field(default_factory=list)
    closed: bool = False
    confidence: float = 0.0


class GradientSpec(BaseModel):
    kind: GradientKind = GradientKind.NONE
    angle: float = 0.0
    start_color: str = "#000000"
    end_color: str = "#FFFFFF"
    center_x: Optional[float] = None
    center_y: Optional[float] = None
    confidence: float = 0.0


class VectorObject(BaseModel):
    id: int
    type: VectorType
    x: float = 0.0
    y: float = 0.0
    width: float = 0.0
    height: float = 0.0
    rotation: float = 0.0
    fill_color: Optional[str] = None
    stroke_color: Optional[str] = None
    stroke_width: float = 0.0
    corner_radius: float = 0.0
    opacity: float = 1.0
    layer: int = 3
    # Aliases matching the brief JSON example
    fill: Optional[str] = None
    stroke: Optional[str] = None
    path: Optional[PathData] = None
    gradient: Optional[GradientSpec] = None
    points: List[ControlPoint] = Field(default_factory=list)
    confidence: float = 90.0
    merged_from: List[int] = Field(default_factory=list)
    source: Dict[str, Any] = Field(default_factory=dict)
    meta: Dict[str, Any] = Field(default_factory=dict)


class VectorCounts(BaseModel):
    total: int = 0
    rectangles: int = 0
    rounded_rectangles: int = 0
    lines: int = 0
    paths: int = 0
    gradients: int = 0
    color_regions: int = 0
    circles: int = 0
    ellipses: int = 0
    polygons: int = 0
    triangles: int = 0
    arrows: int = 0
    ribbons: int = 0
    waves: int = 0
    merged_shapes: int = 0
    curve_count: int = 0


class VectorSummary(BaseModel):
    counts: VectorCounts
    vector_score: float = Field(default=0.0, description="0-100 vector recovery score")
    average_confidence: float = 0.0
    processing_time_ms: float = 0.0


class VectorSuccessResponse(BaseModel):
    success: bool = True
    image_id: str
    page: Dict[str, float]
    vectors: List[VectorObject]
    summary: VectorSummary
    processing_time_ms: float
    results_file: Optional[str] = None
    debug_image: Optional[str] = None
    message: str = "Vector reconstruction completed successfully."
    document_mode: Optional[str] = None
