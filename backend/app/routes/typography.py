"""
Typography route — style metadata for OCR text blocks.
"""

from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, Depends, HTTPException, status

from app.config import Settings, get_settings
from app.typography.models import TypographyRequest, TypographySuccessResponse
from app.typography.typography_service import analyze_typography

logger = logging.getLogger(__name__)

router = APIRouter(tags=["typography"])


@router.post(
    "/typography",
    response_model=TypographySuccessResponse,
    status_code=status.HTTP_200_OK,
    summary="Analyze typography / text styles",
    responses={
        400: {"description": "Corrupted image"},
        404: {"description": "Image or OCR results not found"},
        500: {"description": "Typography analysis failure"},
    },
)
async def run_typography_endpoint(
    body: TypographyRequest,
    settings: Settings = Depends(get_settings),
) -> TypographySuccessResponse:
    """
    Estimate visual style metadata for every OCR text block.

    Does NOT generate PDF/SVG or estimate exact font families.
    """
    image_id = body.image_id.strip()
    logger.info("Typography analysis requested for image_id=%s", image_id)

    try:
        payload = await asyncio.to_thread(analyze_typography, image_id, settings)
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "success": False,
                "error": "Not found",
                "detail": str(exc),
                "code": "NOT_FOUND",
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
        logger.exception("Typography analysis failed for %s", image_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "success": False,
                "error": "Typography analysis failure",
                "detail": "An unexpected error occurred during typography analysis.",
                "code": "TYPOGRAPHY_FAILURE",
            },
        ) from exc

    return TypographySuccessResponse(**payload)
