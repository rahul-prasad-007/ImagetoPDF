"""
Layout analysis Pydantic models and object-type constants.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ObjectType(str, Enum):
    TITLE = "TITLE"
    SUBTITLE = "SUBTITLE"
    PARAGRAPH = "PARAGRAPH"
    TEXT_BLOCK = "TEXT_BLOCK"
    IMAGE = "IMAGE"
    PHOTO = "PHOTO"
    LOGO = "LOGO"
    ICON = "ICON"
    RECTANGLE = "RECTANGLE"
    ROUNDED_RECTANGLE = "ROUNDED_RECTANGLE"
    LINE = "LINE"
    CIRCLE = "CIRCLE"
    ELLIPSE = "ELLIPSE"
    TABLE = "TABLE"
    LIST = "LIST"
    BACKGROUND_SHAPE = "BACKGROUND_SHAPE"
    DECORATIVE_ELEMENT = "DECORATIVE_ELEMENT"
    # Structural containers (for the layout tree)
    PAGE = "PAGE"
    BACKGROUND = "BACKGROUND"
    HEADER = "HEADER"
    MAIN_CONTENT = "MAIN_CONTENT"
    FOOTER = "FOOTER"
    QR_CODE = "QR_CODE"


class LayoutRequest(BaseModel):
    image_id: str = Field(..., min_length=8, description="UUID from upload response")


class PageInfo(BaseModel):
    width: int
    height: int


class LayoutObject(BaseModel):
    id: int
    type: ObjectType
    bbox: List[float] = Field(description="[x1, y1, x2, y2] axis-aligned")
    center_x: float
    center_y: float
    width: float
    height: float
    area: float
    rotation: float = 0.0
    z_index: int = 0
    parent: Optional[int] = None
    children: List[int] = Field(default_factory=list)
    # Optional enrichment
    text: Optional[str] = None
    ocr_block_ids: List[int] = Field(default_factory=list)
    confidence: Optional[float] = None
    meta: Dict[str, Any] = Field(default_factory=dict)


class LayoutTreeNode(BaseModel):
    id: int
    type: ObjectType
    label: Optional[str] = None
    children: List["LayoutTreeNode"] = Field(default_factory=list)


LayoutTreeNode.model_rebuild()


class LayoutCounts(BaseModel):
    total: int = 0
    titles: int = 0
    subtitles: int = 0
    paragraphs: int = 0
    text_blocks: int = 0
    images: int = 0
    photos: int = 0
    logos: int = 0
    icons: int = 0
    rectangles: int = 0
    rounded_rectangles: int = 0
    lines: int = 0
    circles: int = 0
    ellipses: int = 0
    tables: int = 0
    lists: int = 0
    background_shapes: int = 0
    decorative_elements: int = 0
    qr_codes: int = 0


class LayoutSuccessResponse(BaseModel):
    success: bool = True
    image_id: str
    page: PageInfo
    objects: List[LayoutObject]
    tree: LayoutTreeNode
    counts: LayoutCounts
    processing_time_ms: float
    results_file: Optional[str] = None
    debug_image: Optional[str] = None
    message: str = "Layout analysis completed successfully."
    document_mode: Optional[str] = None
    document_mode_info: Optional[Dict[str, Any]] = None
    structure: Optional[Dict[str, Any]] = None
