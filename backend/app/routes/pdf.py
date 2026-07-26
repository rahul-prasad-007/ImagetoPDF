"""PDF render route — generate editable PDF from scene + vectors."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import FileResponse

from app.config import Settings, get_settings
from app.pdf.models import RenderRequest, RenderSuccessResponse
from app.pdf.renderer import render_editable_pdf

logger = logging.getLogger(__name__)

router = APIRouter(tags=["pdf"])


@router.post(
    "/render",
    response_model=RenderSuccessResponse,
    status_code=status.HTTP_200_OK,
    summary="Render editable PDF",
    responses={
        400: {"description": "Corrupted image or invalid render input"},
        404: {"description": "Scene or vector results not found"},
        500: {"description": "PDF render failure"},
    },
)
async def run_render_endpoint(
    body: RenderRequest,
    settings: Settings = Depends(get_settings),
) -> RenderSuccessResponse:
    """
    Reconstruct an editable PDF from scene graph + vector JSON.

    Does not flatten the page into a single image.
    """
    image_id = body.image_id.strip()
    logger.info("PDF render requested for image_id=%s", image_id)

    try:
        payload = await asyncio.to_thread(render_editable_pdf, image_id, settings)
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
                "error": "Invalid render input",
                "detail": str(exc),
                "code": "INVALID_RENDER",
            },
        ) from exc
    except Exception as exc:
        logger.exception("PDF render failed for %s", image_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "success": False,
                "error": "PDF render failure",
                "detail": "An unexpected error occurred while rendering the PDF.",
                "code": "RENDER_FAILURE",
            },
        ) from exc

    return RenderSuccessResponse(**payload)


@router.get(
    "/output/{filename}",
    summary="Download generated PDF",
    responses={404: {"description": "PDF not found"}},
)
async def download_pdf(
    filename: str,
    settings: Settings = Depends(get_settings),
):
    """Serve a generated PDF from backend/output/."""
    if (
        ".." in filename
        or "/" in filename
        or "\\" in filename
        or not filename.lower().endswith(".pdf")
    ):
        raise HTTPException(
            status_code=400,
            detail={
                "success": False,
                "error": "Invalid filename",
                "code": "BAD_FILENAME",
            },
        )

    path = settings.output_path / Path(filename).name
    if not path.is_file():
        raise HTTPException(
            status_code=404,
            detail={
                "success": False,
                "error": "Not found",
                "detail": "PDF file not found",
                "code": "NOT_FOUND",
            },
        )
    return FileResponse(
        path=str(path),
        media_type="application/pdf",
        filename=path.name,
        content_disposition_type="inline",
    )
