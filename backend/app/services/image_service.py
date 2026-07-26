"""
Image persistence and preprocessing service.

No OCR, layout detection, or PDF generation — upload + preprocess only.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageOps

from app.config import Settings
from app.utils.validators import format_file_size, generate_uuid

logger = logging.getLogger(__name__)


@dataclass
class ProcessedImageResult:
    """Result of save + preprocess pipeline."""

    image_id: str
    original_filename: str
    processed_filename: str
    original_path: Path
    processed_path: Path
    width: int
    height: int
    channels: int
    size_bytes: int

    @property
    def size(self) -> str:
        return format_file_size(self.size_bytes)


def save_image(content: bytes, extension: str, settings: Settings) -> tuple[str, Path]:
    """
    Persist raw upload bytes under uploads/ with a unique UUID filename.

    Never overwrites existing files (UUID collision is astronomically unlikely;
    we still guard with a exists-check loop).

    Returns:
        (image_id, absolute_path)
    """
    uploads = settings.uploads_path
    image_id = generate_uuid()

    for _ in range(5):
        filename = f"{image_id}{extension}"
        dest = uploads / filename
        if not dest.exists():
            dest.write_bytes(content)
            logger.info(
                "Saved original image id=%s path=%s bytes=%d",
                image_id,
                dest,
                len(content),
            )
            return image_id, dest
        image_id = generate_uuid()

    raise RuntimeError("Unable to allocate a unique filename for upload.")


def _to_rgb(image: Image.Image) -> Image.Image:
    """Convert any mode to RGB, compositing alpha onto white when needed."""
    if image.mode in ("RGBA", "LA") or (image.mode == "P" and "transparency" in image.info):
        rgba = image.convert("RGBA")
        background = Image.new("RGB", rgba.size, (255, 255, 255))
        background.paste(rgba, mask=rgba.split()[-1])
        return background
    if image.mode == "RGB":
        return image
    return image.convert("RGB")


def _resize_if_needed(image: Image.Image, max_dim: int) -> Image.Image:
    """Downscale so the longest side is at most max_dim, preserving aspect ratio."""
    width, height = image.size
    longest = max(width, height)
    if longest <= max_dim:
        return image

    scale = max_dim / float(longest)
    new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
    logger.info("Resizing image from %sx%s to %sx%s", width, height, *new_size)
    return image.resize(new_size, Image.Resampling.LANCZOS)


def _mild_denoise(rgb: Image.Image) -> Image.Image:
    """Apply a light OpenCV denoising pass suitable for posters/scans."""
    arr = np.array(rgb)
    bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    # Mild settings — preserve text edges for future OCR phases
    denoised = cv2.fastNlMeansDenoisingColored(
        bgr,
        None,
        h=5,
        hColor=5,
        templateWindowSize=7,
        searchWindowSize=21,
    )
    rgb_out = cv2.cvtColor(denoised, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb_out)


def _improve_contrast(rgb: Image.Image, factor: float = 1.12) -> Image.Image:
    """Slight contrast boost without crushing shadows/highlights."""
    return ImageEnhance.Contrast(rgb).enhance(factor)


def preprocess_image(
    original_path: Path,
    image_id: str,
    settings: Settings,
) -> tuple[Path, int, int, int, int]:
    """
    Read → EXIF-correct → RGB → resize → denoise → contrast → save to processed/.

    Returns:
        (processed_path, width, height, channels, size_bytes)
    """
    started = time.perf_counter()

    try:
        with Image.open(original_path) as img:
            # Correct EXIF orientation before any geometry ops
            img = ImageOps.exif_transpose(img)
            rgb = _to_rgb(img)
            rgb = _resize_if_needed(rgb, settings.max_image_dimension)
            rgb = _mild_denoise(rgb)
            rgb = _improve_contrast(rgb)

            processed_name = f"{image_id}_processed.png"
            processed_path = settings.processed_path / processed_name

            # Guard against overwrite
            if processed_path.exists():
                processed_name = f"{image_id}_{generate_uuid()[:8]}_processed.png"
                processed_path = settings.processed_path / processed_name

            rgb.save(processed_path, format="PNG", optimize=True)
            width, height = rgb.size
            channels = len(rgb.getbands())  # RGB → 3
            size_bytes = processed_path.stat().st_size
    except Exception:
        logger.exception("Preprocessing failed for %s", original_path)
        raise

    elapsed_ms = (time.perf_counter() - started) * 1000
    logger.info(
        "Preprocessed image id=%s size=%dx%d channels=%d in %.1fms -> %s",
        image_id,
        width,
        height,
        channels,
        elapsed_ms,
        processed_path.name,
    )
    return processed_path, width, height, channels, size_bytes


def process_upload(
    content: bytes,
    extension: str,
    original_client_name: str | None,
    settings: Settings,
) -> ProcessedImageResult:
    """
    Full pipeline: save original → preprocess → return metadata.
    """
    pipeline_start = time.perf_counter()

    image_id, original_path = save_image(content, extension, settings)
    processed_path, width, height, channels, size_bytes = preprocess_image(
        original_path, image_id, settings
    )

    total_ms = (time.perf_counter() - pipeline_start) * 1000
    logger.info(
        "Upload pipeline complete id=%s client=%s total=%.1fms original=%s processed=%s",
        image_id,
        original_client_name,
        total_ms,
        original_path.name,
        processed_path.name,
    )

    return ProcessedImageResult(
        image_id=image_id,
        original_filename=original_path.name,
        processed_filename=processed_path.name,
        original_path=original_path,
        processed_path=processed_path,
        width=width,
        height=height,
        channels=channels,
        size_bytes=size_bytes,
    )
