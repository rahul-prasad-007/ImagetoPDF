"""
OCR route — run PaddleOCR on a previously uploaded/processed image.
"""

from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, Depends, HTTPException, status

from app.config import Settings, get_settings
from app.ocr.models import OcrRequest, OcrSuccessResponse
from app.ocr.ocr_service import run_ocr

logger = logging.getLogger(__name__)

router = APIRouter(tags=["ocr"])


@router.post(
    "/ocr",
    response_model=OcrSuccessResponse,
    status_code=status.HTTP_200_OK,
    summary="Run OCR on a processed image",
    responses={
        400: {"description": "Invalid image_id / corrupted image / no usable input"},
        404: {"description": "Processed image not found"},
        500: {"description": "OCR engine failure"},
    },
)
async def run_ocr_endpoint(
    body: OcrRequest,
    settings: Settings = Depends(get_settings),
) -> OcrSuccessResponse:
    """
    Accept an image_id from POST /api/upload, run OCR on the processed image,
    save JSON + debug visualization, and return structured text metadata.

    Does NOT generate PDFs, estimate fonts, or recreate layouts.
    """
    image_id = body.image_id.strip()
    logger.info("OCR requested for image_id=%s", image_id)

    try:
        # Run CPU-heavy OCR off the event loop
        payload = await asyncio.to_thread(run_ocr, image_id, settings)
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "success": False,
                "error": "Image not found",
                "detail": str(exc) or "No processed image exists for this image_id.",
                "code": "IMAGE_NOT_FOUND",
            },
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "success": False,
                "error": "Corrupted image",
                "detail": str(exc),
                "code": "CORRUPTED_IMAGE",
            },
        ) from exc
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "success": False,
                "error": "OCR failure",
                "detail": str(exc),
                "code": "OCR_FAILURE",
            },
        ) from exc
    except Exception as exc:
        logger.exception("Unexpected OCR error for %s", image_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "success": False,
                "error": "OCR failure",
                "detail": "An unexpected error occurred during OCR.",
                "code": "OCR_FAILURE",
            },
        ) from exc

    return OcrSuccessResponse(**payload)
