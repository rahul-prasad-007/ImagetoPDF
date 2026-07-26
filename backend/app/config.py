"""
Application configuration loaded from environment variables.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# backend/ root (parent of app/)
BACKEND_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Central configuration for the Image to Editable PDF backend."""

    model_config = SettingsConfigDict(
        env_file=str(BACKEND_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = True

    # CORS — comma-separated list in .env, or "*" for all (same-origin Docker deploy)
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173,*"

    # Built frontend (Vite dist) — set STATIC_DIR in Docker / production
    static_dir: str = ""

    # Limits
    max_upload_size_mb: int = 20
    max_image_dimension: int = 3000

    # Storage directories (relative to backend root unless absolute)
    upload_dir: str = "uploads"
    processed_dir: str = "processed"
    results_dir: str = "results"
    debug_dir: str = "debug"
    output_dir: str = "output"

    # OCR — auto | en | hi (auto retries Hindi when English OCR looks like garbage)
    ocr_lang: str = "auto"
    ocr_low_confidence_threshold: float = 0.60

    # Hybrid reconstruction — optional PP-StructureV3 (first run may download models)
    use_pp_structure: bool = False
    hybrid_underlays: bool = True

    # Allowed image extensions (lowercase, with leading dot)
    allowed_extensions: frozenset[str] = frozenset(
        {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
    )

    # Allowed MIME types
    allowed_mime_types: frozenset[str] = frozenset(
        {
            "image/png",
            "image/jpeg",
            "image/jpg",
            "image/webp",
            "image/bmp",
            "image/x-ms-bmp",
            "image/tiff",
            "image/tif",
        }
    )

    # Explicitly rejected extensions / content families
    rejected_extensions: frozenset[str] = frozenset(
        {
            ".pdf",
            ".zip",
            ".svg",
            ".txt",
            ".mp4",
            ".mov",
            ".avi",
            ".mkv",
            ".webm",
            ".gif",  # animated / not in accept list
            ".heic",
            ".doc",
            ".docx",
        }
    )

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_size_mb * 1024 * 1024

    @property
    def cors_origin_list(self) -> list[str]:
        origins = [o.strip() for o in self.cors_origins.split(",") if o.strip()]
        if "*" in origins:
            return ["*"]
        return origins

    @property
    def static_path(self) -> Path | None:
        raw = (self.static_dir or "").strip()
        if not raw:
            # Docker default or local monorepo dist
            for candidate in (
                BACKEND_ROOT / "static",
                BACKEND_ROOT.parent / "dist",
            ):
                if (candidate / "index.html").exists():
                    return candidate
            return None
        path = Path(raw)
        if not path.is_absolute():
            path = BACKEND_ROOT / path
        return path if (path / "index.html").exists() else None

    @property
    def uploads_path(self) -> Path:
        path = Path(self.upload_dir)
        if not path.is_absolute():
            path = BACKEND_ROOT / path
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def processed_path(self) -> Path:
        path = Path(self.processed_dir)
        if not path.is_absolute():
            path = BACKEND_ROOT / path
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def results_path(self) -> Path:
        path = Path(self.results_dir)
        if not path.is_absolute():
            path = BACKEND_ROOT / path
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def debug_path(self) -> Path:
        path = Path(self.debug_dir)
        if not path.is_absolute():
            path = BACKEND_ROOT / path
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def output_path(self) -> Path:
        path = Path(self.output_dir)
        if not path.is_absolute():
            path = BACKEND_ROOT / path
        path.mkdir(parents=True, exist_ok=True)
        return path


@lru_cache
def get_settings() -> Settings:
    """Cached settings singleton."""
    settings = Settings()
    # Ensure directories exist at startup
    _ = settings.uploads_path
    _ = settings.processed_path
    _ = settings.results_path
    _ = settings.debug_path
    _ = settings.output_path
    return settings
