"""
PDF render models — request/response and validation for editable PDF output.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class RenderRequest(BaseModel):
    image_id: str = Field(..., min_length=8)


class RenderValidationIssue(BaseModel):
    code: str
    message: str
    object_id: Optional[int] = None


class RenderValidation(BaseModel):
    ok: bool = True
    issues: List[RenderValidationIssue] = Field(default_factory=list)
    missing_fonts: int = 0
    objects_outside_page: int = 0
    invalid_coordinates: int = 0
    overlapping_text: int = 0
    missing_images: int = 0


class RenderCounts(BaseModel):
    total_objects: int = 0
    text_count: int = 0
    image_count: int = 0
    vector_count: int = 0


class RenderSummary(BaseModel):
    counts: RenderCounts
    validation: RenderValidation
    pdf_size_bytes: int = 0
    pdf_size: str = ""
    render_time_ms: float = 0.0
    page_format: str = ""
    orientation: str = ""


class RenderSuccessResponse(BaseModel):
    success: bool = True
    image_id: str
    pdf: str = Field(description="Relative path e.g. output/output_<uuid>.pdf")
    download_url: str = Field(description="HTTP path to download the PDF")
    preview_url: Optional[str] = None
    summary: RenderSummary
    processing_time_ms: float
    message: str = "Editable PDF generated successfully."
    document_mode: Optional[str] = None
