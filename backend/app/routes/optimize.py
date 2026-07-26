"""Optimization route — compare PDF to original and auto-improve quality."""

from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, Depends, HTTPException, status

from app.config import Settings, get_settings
from app.optimizer.models import OptimizeRequest, OptimizeSuccessResponse
from app.optimizer.optimizer import optimize_pdf

logger = logging.getLogger(__name__)

router = APIRouter(tags=["optimize"])


@router.post(
    "/optimize",
    response_model=OptimizeSuccessResponse,
    status_code=status.HTTP_200_OK,
    summary="Optimize editable PDF quality",
    responses={
        400: {"description": "Invalid image or PDF"},
        404: {"description": "Missing scene, vector, or PDF"},
        500: {"description": "Optimization failure"},
    },
)
async def run_optimize_endpoint(
    body: OptimizeRequest,
    settings: Settings = Depends(get_settings),
) -> OptimizeSuccessResponse:
    """
    Rasterize the generated PDF, compare to the original image, apply automatic
    layout/color/geometry fixes, and replace the PDF when quality improves.

    Does not re-run OCR, layout, or typography.
    """
    image_id = body.image_id.strip()
    logger.info("PDF optimization requested for image_id=%s", image_id)

    try:
        payload = await asyncio.to_thread(optimize_pdf, image_id, settings)
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
                "error": "Invalid optimize input",
                "detail": str(exc),
                "code": "INVALID_OPTIMIZE",
            },
        ) from exc
    except Exception as exc:
        logger.exception("PDF optimization failed for %s", image_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "success": False,
                "error": "Optimization failure",
                "detail": "An unexpected error occurred while optimizing the PDF.",
                "code": "OPTIMIZE_FAILURE",
            },
        ) from exc

    return OptimizeSuccessResponse(**payload)
