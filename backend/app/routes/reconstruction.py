"""
Reconstruction planning route — decide how objects should be rebuilt.
"""

from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, Depends, HTTPException, status

from app.config import Settings, get_settings
from app.reconstruction.models import ReconstructionRequest, ReconstructionSuccessResponse
from app.reconstruction.planner import plan_reconstruction

logger = logging.getLogger(__name__)

router = APIRouter(tags=["reconstruction"])


@router.post(
    "/reconstruction",
    response_model=ReconstructionSuccessResponse,
    status_code=status.HTTP_200_OK,
    summary="Build reconstruction plan",
    responses={
        400: {"description": "Corrupted image"},
        404: {"description": "Image or layout results not found"},
        500: {"description": "Reconstruction planning failure"},
    },
)
async def run_reconstruction_endpoint(
    body: ReconstructionRequest,
    settings: Settings = Depends(get_settings),
) -> ReconstructionSuccessResponse:
    """
    Analyze layout/OCR/typography and produce a reconstruction plan.

    Does NOT generate PDF, SVG, CDR, or redraw the page.
    """
    image_id = body.image_id.strip()
    logger.info("Reconstruction planning requested for image_id=%s", image_id)

    try:
        payload = await asyncio.to_thread(plan_reconstruction, image_id, settings)
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
        logger.exception("Reconstruction planning failed for %s", image_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "success": False,
                "error": "Reconstruction planning failure",
                "detail": "An unexpected error occurred during reconstruction planning.",
                "code": "RECONSTRUCTION_FAILURE",
            },
        ) from exc

    return ReconstructionSuccessResponse(**payload)
