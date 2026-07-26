"""
Layout analysis route — structured document model for a processed image.
"""

from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, Depends, HTTPException, status

from app.config import Settings, get_settings
from app.layout.layout_service import analyze_layout
from app.layout.models import LayoutRequest, LayoutSuccessResponse

logger = logging.getLogger(__name__)

router = APIRouter(tags=["layout"])


@router.post(
    "/layout",
    response_model=LayoutSuccessResponse,
    status_code=status.HTTP_200_OK,
    summary="Analyze document layout",
    responses={
        400: {"description": "Corrupted / invalid image"},
        404: {"description": "Processed image not found"},
        500: {"description": "Layout analysis failure"},
    },
)
async def run_layout_endpoint(
    body: LayoutRequest,
    settings: Settings = Depends(get_settings),
) -> LayoutSuccessResponse:
    """
    Build a hierarchical document layout model from the processed image
    and existing OCR results (if available).

    Does NOT generate PDFs, estimate fonts, or reconstruct backgrounds.
    """
    image_id = body.image_id.strip()
    logger.info("Layout analysis requested for image_id=%s", image_id)

    try:
        payload = await asyncio.to_thread(analyze_layout, image_id, settings)
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
    except Exception as exc:
        logger.exception("Layout analysis failed for %s", image_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "success": False,
                "error": "Layout analysis failure",
                "detail": "An unexpected error occurred during layout analysis.",
                "code": "LAYOUT_FAILURE",
            },
        ) from exc

    return LayoutSuccessResponse(**payload)
