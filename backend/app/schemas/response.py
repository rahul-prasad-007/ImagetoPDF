"""
Pydantic response and error schemas for the upload API.
"""

from typing import Optional

from pydantic import BaseModel, Field


class UploadSuccessResponse(BaseModel):
    """Successful image upload + preprocess response."""

    success: bool = True
    image_id: str
    original_filename: str
    processed_filename: str
    width: int
    height: int
    channels: int
    size: str = Field(description="Human-readable processed file size")
    size_bytes: int
    original_path: str
    processed_path: str
    message: str = "Image uploaded successfully."


class ErrorResponse(BaseModel):
    """Clean JSON error payload."""

    success: bool = False
    error: str
    detail: Optional[str] = None
    code: Optional[str] = None


class HealthResponse(BaseModel):
    """Health check payload."""

    status: str = "ok"
    service: str = "image-to-editable-pdf"
    version: str = "0.1.0"
