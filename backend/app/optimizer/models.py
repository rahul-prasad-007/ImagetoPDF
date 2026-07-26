"""
Optimization engine models — request/response and metric payloads.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class OptimizeRequest(BaseModel):
    image_id: str = Field(..., min_length=8)


class ObjectDiff(BaseModel):
    object_id: int
    object_type: str = ""
    original_position: List[float] = Field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0])
    rendered_position: List[float] = Field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0])
    offset_x: float = 0.0
    offset_y: float = 0.0
    width_difference: float = 0.0
    height_difference: float = 0.0
    rotation_difference: float = 0.0
    color_difference: float = 0.0
    severity: str = "perfect"  # perfect | minor | large
    fixes_applied: List[str] = Field(default_factory=list)


class SimilarityMetrics(BaseModel):
    ssim: float = 0.0
    psnr: float = 0.0
    pixel_difference: float = 0.0
    edge_difference: float = 0.0
    color_difference: float = 0.0
    text_bbox_difference: float = 0.0
    alignment_difference: float = 0.0
    object_position_error: float = 0.0
    spacing_error: float = 0.0
    overall_similarity: float = 0.0


class AccuracyBreakdown(BaseModel):
    overall_similarity: float = 0.0
    text_accuracy: float = 0.0
    layout_accuracy: float = 0.0
    color_accuracy: float = 0.0
    object_accuracy: float = 0.0
    vector_accuracy: float = 0.0
    image_accuracy: float = 0.0


class OptimizationSummary(BaseModel):
    before: SimilarityMetrics
    after: SimilarityMetrics
    accuracy: AccuracyBreakdown
    objects_compared: int = 0
    objects_fixed: int = 0
    pdf_replaced: bool = False
    improved: bool = False
    optimization_time_ms: float = 0.0
    targets_met: Dict[str, bool] = Field(default_factory=dict)


class OptimizeSuccessResponse(BaseModel):
    success: bool = True
    image_id: str
    optimization: str = Field(description="Relative path e.g. results/optimization_<uuid>.json")
    report: str = Field(description="Relative path e.g. results/report_<uuid>.html")
    debug_image: str = Field(description="Relative path e.g. debug/optimization_<uuid>.png")
    pdf: str = Field(description="Relative path e.g. output/output_<uuid>.pdf")
    download_url: str
    preview_url: Optional[str] = None
    summary: OptimizationSummary
    object_diffs: List[ObjectDiff] = Field(default_factory=list)
    fixes: List[str] = Field(default_factory=list)
    processing_time_ms: float
    message: str = "PDF quality optimization complete."
    meta: Dict[str, Any] = Field(default_factory=dict)
