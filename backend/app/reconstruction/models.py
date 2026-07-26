"""
Reconstruction planner models — how each layout object should be rebuilt later.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ReconstructionType(str, Enum):
    TEXT = "TEXT"
    VECTOR_RECTANGLE = "VECTOR_RECTANGLE"
    VECTOR_ROUNDED_RECTANGLE = "VECTOR_ROUNDED_RECTANGLE"
    VECTOR_LINE = "VECTOR_LINE"
    VECTOR_CIRCLE = "VECTOR_CIRCLE"
    VECTOR_ELLIPSE = "VECTOR_ELLIPSE"
    VECTOR_POLYGON = "VECTOR_POLYGON"
    VECTOR_PATH = "VECTOR_PATH"
    IMAGE = "IMAGE"
    LOGO_IMAGE = "LOGO_IMAGE"
    PHOTO_IMAGE = "PHOTO_IMAGE"
    ICON_IMAGE = "ICON_IMAGE"
    BACKGROUND_IMAGE = "BACKGROUND_IMAGE"
    IGNORE = "IGNORE"


class ReconstructionRequest(BaseModel):
    image_id: str = Field(..., min_length=8)


class PlannedObject(BaseModel):
    id: int
    source_id: int = Field(description="Original layout object id")
    type: str = Field(description="Layout object type")
    reconstruction: ReconstructionType
    layer: int
    confidence: float = Field(ge=0.0, le=100.0)
    bbox: List[float] = Field(description="[x1,y1,x2,y2]")
    reason: str = ""
    merged_from: List[int] = Field(default_factory=list)
    meta: Dict[str, Any] = Field(default_factory=dict)


class ReconstructionCounts(BaseModel):
    total: int = 0
    editable_text: int = 0
    vector_shapes: int = 0
    embedded_images: int = 0
    background_regions: int = 0
    svg_paths: int = 0
    ignored: int = 0


class ReconstructionSummary(BaseModel):
    counts: ReconstructionCounts
    average_confidence: float = 0.0
    overall_score: float = Field(
        default=0.0,
        description="0-100 reconstruction readiness score",
    )
    decision_breakdown: Dict[str, int] = Field(default_factory=dict)
    layer_stats: Dict[str, int] = Field(default_factory=dict)


class ReconstructionSuccessResponse(BaseModel):
    success: bool = True
    image_id: str
    page: Dict[str, int]
    objects: List[PlannedObject]
    summary: ReconstructionSummary
    processing_time_ms: float
    results_file: Optional[str] = None
    debug_image: Optional[str] = None
    message: str = "Reconstruction plan completed successfully."
