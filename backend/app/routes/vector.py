"""Vector reconstruction route — editable vector data only (no PDF/SVG export)."""

from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, Depends, HTTPException, status

from app.config import Settings, get_settings
from app.vector.models import VectorRequest, VectorSuccessResponse
from app.vector.vector_builder import build_vectors

logger = logging.getLogger(__name__)

router = APIRouter(tags=["vector"])


@router.post(
    "/vector",
    response_model=VectorSuccessResponse,
    status_code=status.HTTP_200_OK,
    summary="Reconstruct background & vector shapes",
    responses={
        400: {"description": "Corrupted image"},
        404: {"description": "Image or scene graph not found"},
        500: {"description": "Vector reconstruction failure"},
    },
)
async def run_vector_endpoint(
    body: VectorRequest,
    settings: Settings = Depends(get_settings),
) -> VectorSuccessResponse:
    """
    Detect and reconstruct simple graphical elements as editable vectors.

    Does NOT generate PDF or export SVG files.
    """
    image_id = body.image_id.strip()
    logger.info("Vector reconstruction requested for image_id=%s", image_id)

    try:
        payload = await asyncio.to_thread(build_vectors, image_id, settings)
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
        logger.exception("Vector reconstruction failed for %s", image_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "success": False,
                "error": "Vector reconstruction failure",
                "detail": "An unexpected error occurred during vector reconstruction.",
                "code": "VECTOR_FAILURE",
            },
        ) from exc

    return VectorSuccessResponse(**payload)
