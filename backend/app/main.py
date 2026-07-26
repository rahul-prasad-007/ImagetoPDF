"""
FastAPI application entrypoint — Image to Editable PDF backend.

Phase scope: image upload + preprocessing only.
"""

from __future__ import annotations

import logging
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.config import get_settings
from app.routes.upload import router as upload_router
from app.routes.ocr import router as ocr_router
from app.routes.layout import router as layout_router
from app.routes.typography import router as typography_router
from app.routes.reconstruction import router as reconstruction_router
from app.routes.scene import router as scene_router
from app.routes.vector import router as vector_router
from app.routes.pdf import router as pdf_router
from app.routes.optimize import router as optimize_router
from app.schemas.response import HealthResponse

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("app")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    settings = get_settings()
    logger.info(
        "Starting backend | uploads=%s processed=%s results=%s debug=%s output=%s max_mb=%s",
        settings.uploads_path,
        settings.processed_path,
        settings.results_path,
        settings.debug_path,
        settings.output_path,
        settings.max_upload_size_mb,
    )
    yield
    logger.info("Shutting down backend")


app = FastAPI(
    title="Image to Editable PDF API",
    description="Upload through editable PDF rendering and quality optimization.",
    version="0.10.0",
    lifespan=lifespan,
)

# ---------------------------------------------------------------------------
# CORS — allow React frontend (dev + production-ready origin list)
# ---------------------------------------------------------------------------
_settings = get_settings()
_cors_origins = _settings.cors_origin_list
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=_cors_origins != ["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
app.include_router(upload_router, prefix="/api")
app.include_router(ocr_router, prefix="/api")
app.include_router(layout_router, prefix="/api")
app.include_router(typography_router, prefix="/api")
app.include_router(reconstruction_router, prefix="/api")
app.include_router(scene_router, prefix="/api")
app.include_router(vector_router, prefix="/api")
app.include_router(pdf_router, prefix="/api")
app.include_router(optimize_router, prefix="/api")


@app.get("/health", response_model=HealthResponse, tags=["health"])
async def health() -> HealthResponse:
    return HealthResponse()


# ---------------------------------------------------------------------------
# Production SPA (Vite build) — same origin as /api
# ---------------------------------------------------------------------------
_static = _settings.static_path
if _static is not None:
    assets_dir = _static / "assets"
    if assets_dir.is_dir():
        app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="assets")

    @app.get("/", include_in_schema=False)
    async def spa_index():
        return FileResponse(_static / "index.html")

    @app.get("/{full_path:path}", include_in_schema=False)
    async def spa_fallback(full_path: str):
        # Never shadow API / health / docs
        if full_path.startswith(("api/", "health", "docs", "openapi", "redoc")):
            raise HTTPException(status_code=404, detail="Not found")
        candidate = _static / full_path
        if candidate.is_file():
            return FileResponse(candidate)
        return FileResponse(_static / "index.html")
else:

    @app.get("/", tags=["health"])
    async def root():
        return {
            "success": True,
            "message": "Image to Editable PDF API",
            "docs": "/docs",
            "health": "/health",
            "upload": "POST /api/upload",
            "ocr": "POST /api/ocr",
            "layout": "POST /api/layout",
            "typography": "POST /api/typography",
            "reconstruction": "POST /api/reconstruction",
            "scene": "POST /api/scene",
            "vector": "POST /api/vector",
            "render": "POST /api/render",
            "optimize": "POST /api/optimize",
            "download": "GET /api/output/{filename}",
        }


# ---------------------------------------------------------------------------
# Exception handlers — always return clean JSON
# ---------------------------------------------------------------------------
@app.exception_handler(HTTPException)
async def http_exception_handler(_request: Request, exc: HTTPException):
    detail = exc.detail
    if isinstance(detail, dict):
        payload = detail
    else:
        payload = {
            "success": False,
            "error": "Request error",
            "detail": str(detail),
            "code": "HTTP_ERROR",
        }
    return JSONResponse(status_code=exc.status_code, content=payload)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(_request: Request, exc: RequestValidationError):
    logger.warning("Validation error: %s", exc.errors())
    return JSONResponse(
        status_code=422,
        content={
            "success": False,
            "error": "Invalid request",
            "detail": "Request validation failed. Check required fields and JSON body.",
            "code": "VALIDATION_ERROR",
        },
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(_request: Request, exc: Exception):
    logger.exception("Unhandled error: %s", exc)
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": "Internal server error",
            "detail": "An unexpected error occurred.",
            "code": "INTERNAL_ERROR",
        },
    )
