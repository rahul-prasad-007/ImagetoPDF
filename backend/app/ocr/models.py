"""
Pydantic models for OCR request/response payloads.
"""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class OcrRequest(BaseModel):
    """Request body for POST /api/ocr."""

    image_id: str = Field(..., min_length=8, description="UUID from the upload response")


class PageInfo(BaseModel):
    width: int
    height: int


class TextBlock(BaseModel):
    id: int
    text: str
    confidence: float
    bbox: List[List[float]]  # [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
    center_x: float
    center_y: float
    width: float
    height: float
    rotation: float = 0.0
    line: int
    word: int
    paragraph: int


class OcrSuccessResponse(BaseModel):
    success: bool = True
    image_id: str
    page: PageInfo
    text_blocks: List[TextBlock]
    total_blocks: int
    average_confidence: float
    processing_time_ms: float
    debug_image: Optional[str] = None
    results_file: Optional[str] = None
    message: str = "OCR completed successfully."
    warning: Optional[str] = None
    ocr_lang: Optional[str] = None
