"""
Typography analysis Pydantic models.
"""

from __future__ import annotations

from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class Alignment(str, Enum):
    LEFT = "left"
    CENTER = "center"
    RIGHT = "right"
    JUSTIFIED = "justified"


class TextHierarchy(str, Enum):
    TITLE = "TITLE"
    HEADING = "HEADING"
    SUBHEADING = "SUBHEADING"
    BODY = "BODY"
    FOOTER = "FOOTER"
    CAPTION = "CAPTION"
    LABEL = "LABEL"


class ColorInfo(BaseModel):
    rgb: List[int] = Field(description="[R, G, B]")
    hex: str
    hsv: Optional[List[float]] = None  # H 0-360, S/V 0-1


class TypographyRequest(BaseModel):
    image_id: str = Field(..., min_length=8)


class TextStyle(BaseModel):
    id: int
    ocr_block_id: int
    text: str
    font_size: float
    font_color: str
    font_color_rgb: List[int]
    font_color_hsv: List[float]
    background_color: str
    background_color_rgb: List[int]
    contrast_ratio: float
    bold: float = Field(ge=0.0, le=1.0, description="Bold probability 0-1")
    italic: float = Field(ge=0.0, le=1.0)
    underline: float = Field(ge=0.0, le=1.0)
    uppercase_ratio: float = Field(ge=0.0, le=1.0)
    alignment: Alignment
    character_spacing: float
    word_spacing: float
    line_spacing: float
    paragraph_spacing: float
    text_box_width: float
    text_box_height: float
    rotation: float = 0.0
    opacity: float = Field(ge=0.0, le=1.0, default=1.0)
    hierarchy: TextHierarchy
    indentation: float = 0.0
    margins: Dict[str, float] = Field(default_factory=dict)
    paragraph_width: float = 0.0
    average_line_height: float = 0.0
    confidence: float = Field(ge=0.0, le=100.0, description="Style estimate confidence 0-100")
    bbox: List[float] = Field(description="[x1,y1,x2,y2]")
    font_family: str = Field(default="serif", description="serif | sans | handwriting")


class TypographySummary(BaseModel):
    total_styles: int = 0
    average_font_size: float = 0.0
    average_confidence: float = 0.0
    titles: int = 0
    headings: int = 0
    subheadings: int = 0
    body_text: int = 0
    footers: int = 0
    captions: int = 0
    labels: int = 0
    unique_colors: int = 0
    alignment_distribution: Dict[str, int] = Field(default_factory=dict)
    average_bold: float = 0.0
    detected_colors: List[str] = Field(default_factory=list)


class TypographySuccessResponse(BaseModel):
    success: bool = True
    image_id: str
    page: Dict[str, int]
    text_styles: List[TextStyle]
    summary: TypographySummary
    processing_time_ms: float
    results_file: Optional[str] = None
    debug_image: Optional[str] = None
    message: str = "Typography analysis completed successfully."
