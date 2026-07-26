"""
Image upload validation helpers.
"""

from __future__ import annotations

import logging
from io import BytesIO
from pathlib import Path

from fastapi import HTTPException, UploadFile, status
from PIL import Image, UnidentifiedImageError

from app.config import Settings

logger = logging.getLogger(__name__)

# Magic-byte signatures for rejected / non-image families
_PDF_MAGIC = b"%PDF"
_ZIP_MAGIC = b"PK\x03\x04"
_SVG_HINTS = (b"<svg", b"<?xml")


def generate_uuid() -> str:
    """Generate a unique identifier string (no hyphens for shorter filenames)."""
    import uuid

    return uuid.uuid4().hex


def format_file_size(num_bytes: int) -> str:
    """Format byte count into a human-readable string."""
    if num_bytes < 1024:
        return f"{num_bytes} B"
    if num_bytes < 1024**2:
        return f"{num_bytes / 1024:.1f} KB"
    return f"{num_bytes / (1024**2):.1f} MB"


def get_extension(filename: str | None) -> str:
    """Return lowercase file extension including the dot, or empty string."""
    if not filename:
        return ""
    return Path(filename).suffix.lower()


def validate_image(file: UploadFile, content: bytes, settings: Settings) -> str:
    """
    Validate uploaded file type, size, and that content is a readable image.

    Returns the normalized extension (e.g. '.png').

    Raises:
        HTTPException: with appropriate status for each failure mode.
    """
    if not content:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "success": False,
                "error": "Invalid image",
                "detail": "Uploaded file is empty.",
                "code": "EMPTY_FILE",
            },
        )

    if len(content) > settings.max_upload_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail={
                "success": False,
                "error": "File too large",
                "detail": f"Maximum allowed size is {settings.max_upload_size_mb} MB.",
                "code": "FILE_TOO_LARGE",
            },
        )

    ext = get_extension(file.filename)
    content_type = (file.content_type or "").lower().strip()

    # Explicit rejects
    if ext in settings.rejected_extensions:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail={
                "success": False,
                "error": "Unsupported file",
                "detail": f"File type '{ext}' is not allowed. Use PNG, JPEG, WEBP, BMP, or TIFF.",
                "code": "UNSUPPORTED_FILE",
            },
        )

    # Magic-byte checks for common non-images disguised by extension
    head = content[:256].lstrip()
    if content.startswith(_PDF_MAGIC) or head.startswith(_PDF_MAGIC):
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail={
                "success": False,
                "error": "Unsupported file",
                "detail": "PDF files are not accepted.",
                "code": "UNSUPPORTED_FILE",
            },
        )
    if content.startswith(_ZIP_MAGIC):
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail={
                "success": False,
                "error": "Unsupported file",
                "detail": "Archive/ZIP files are not accepted.",
                "code": "UNSUPPORTED_FILE",
            },
        )
    if any(hint in head[:200].lower() for hint in _SVG_HINTS) and (
        ext == ".svg" or b"<svg" in head.lower()
    ):
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail={
                "success": False,
                "error": "Unsupported file",
                "detail": "SVG files are not accepted.",
                "code": "UNSUPPORTED_FILE",
            },
        )

    # Extension must be allowed
    if ext not in settings.allowed_extensions:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail={
                "success": False,
                "error": "Unsupported file",
                "detail": f"Extension '{ext or '(none)'}' is not allowed. Use PNG, JPEG, WEBP, BMP, or TIFF.",
                "code": "UNSUPPORTED_FILE",
            },
        )

    # MIME type check when provided by the client
    if content_type and content_type not in settings.allowed_mime_types:
        # Some browsers send application/octet-stream — allow if extension is valid
        if content_type not in {"application/octet-stream", "binary/octet-stream"}:
            logger.warning("Rejected MIME type %s for %s", content_type, file.filename)
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail={
                    "success": False,
                    "error": "Unsupported file",
                    "detail": f"MIME type '{content_type}' is not allowed.",
                    "code": "UNSUPPORTED_FILE",
                },
            )

    # Verify the bytes are a readable raster image (rejects corrupted / non-image)
    try:
        with Image.open(BytesIO(content)) as img:
            img.verify()
    except UnidentifiedImageError as exc:
        logger.warning("Unidentified image: %s", file.filename)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "success": False,
                "error": "Cannot read image",
                "detail": "The file could not be identified as a valid image.",
                "code": "CANNOT_READ_IMAGE",
            },
        ) from exc
    except OSError as exc:
        logger.warning("Corrupted image: %s (%s)", file.filename, exc)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "success": False,
                "error": "Invalid image",
                "detail": "The image appears to be corrupted or unreadable.",
                "code": "INVALID_IMAGE",
            },
        ) from exc

    # Re-open after verify() (verify leaves the file in a closed state)
    try:
        with Image.open(BytesIO(content)) as img:
            img.load()
            # Reject animated GIF-like multi-frame if somehow slipped through
            if getattr(img, "is_animated", False) and img.format == "GIF":
                raise HTTPException(
                    status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                    detail={
                        "success": False,
                        "error": "Unsupported file",
                        "detail": "Animated GIF files are not accepted.",
                        "code": "UNSUPPORTED_FILE",
                    },
                )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Failed to load image after verify: %s", file.filename)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "success": False,
                "error": "Cannot read image",
                "detail": str(exc) or "Unable to load image data.",
                "code": "CANNOT_READ_IMAGE",
            },
        ) from exc

    # Normalize jpeg extension
    if ext == ".jpeg":
        return ".jpg"
    return ext
