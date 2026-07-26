"""Scene graph route — build editable document model (no PDF/SVG export)."""

from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, Depends, HTTPException, status

from app.config import Settings, get_settings
from app.scene.scene_builder import build_scene
from app.scene.scene_models import SceneRequest, SceneSuccessResponse

logger = logging.getLogger(__name__)

router = APIRouter(tags=["scene"])


@router.post(
    "/scene",
    response_model=SceneSuccessResponse,
    status_code=status.HTTP_200_OK,
    summary="Build editable scene graph",
    responses={
        400: {"description": "Corrupted image"},
        404: {"description": "Image or reconstruction results not found"},
        500: {"description": "Scene build failure"},
    },
)
async def run_scene_endpoint(
    body: SceneRequest,
    settings: Settings = Depends(get_settings),
) -> SceneSuccessResponse:
    """
    Assemble an editable scene graph from OCR/layout/typography/reconstruction.

    Does NOT generate PDF, SVG, CDR, or export drawings.
    """
    image_id = body.image_id.strip()
    logger.info("Scene build requested for image_id=%s", image_id)

    try:
        payload = await asyncio.to_thread(build_scene, image_id, settings)
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
        logger.exception("Scene build failed for %s", image_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "success": False,
                "error": "Scene build failure",
                "detail": "An unexpected error occurred while building the scene graph.",
                "code": "SCENE_FAILURE",
            },
        ) from exc

    return SceneSuccessResponse(**payload)
