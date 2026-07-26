"""
Upload route — accepts an image, validates, stores, and preprocesses it.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status

from app.config import Settings, get_settings
from app.schemas.response import UploadSuccessResponse
from app.services.image_service import process_upload
from app.utils.validators import validate_image

logger = logging.getLogger(__name__)

router = APIRouter(tags=["upload"])


@router.post(
    "/upload",
    response_model=UploadSuccessResponse,
    status_code=status.HTTP_200_OK,
    summary="Upload and preprocess an image",
    responses={
        400: {"description": "Invalid / corrupted / unreadable image"},
        413: {"description": "File too large"},
        415: {"description": "Unsupported file type"},
        500: {"description": "Internal server error"},
    },
)
async def upload_image(
    file: UploadFile = File(..., description="Image file (PNG, JPEG, WEBP, BMP, TIFF)"),
    settings: Settings = Depends(get_settings),
) -> UploadSuccessResponse:
    """
    Accept a single image upload, validate it, store under uploads/,
    preprocess into processed/, and return metadata.

    Does NOT run OCR, layout detection, or PDF generation.
    """
    logger.info(
        "Upload received filename=%s content_type=%s",
        file.filename,
        file.content_type,
    )

    try:
        content = await file.read()
    except Exception as exc:
        logger.exception("Failed reading upload stream")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "success": False,
                "error": "Cannot read image",
                "detail": "Failed to read uploaded file stream.",
                "code": "CANNOT_READ_IMAGE",
            },
        ) from exc

    extension = validate_image(file, content, settings)

    try:
        result = process_upload(
            content=content,
            extension=extension,
            original_client_name=file.filename,
            settings=settings,
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Processing failed for %s", file.filename)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "success": False,
                "error": "Internal server error",
                "detail": "Image processing failed. Please try again.",
                "code": "INTERNAL_ERROR",
            },
        ) from exc

    return UploadSuccessResponse(
        success=True,
        image_id=result.image_id,
        original_filename=result.original_filename,
        processed_filename=result.processed_filename,
        width=result.width,
        height=result.height,
        channels=result.channels,
        size=result.size,
        size_bytes=result.size_bytes,
        original_path=str(result.original_path),
        processed_path=str(result.processed_path),
        message="Image uploaded successfully.",
    )
