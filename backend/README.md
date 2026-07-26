# Backend — Image to Editable PDF

FastAPI service for upload, preprocess, OCR, layout, typography, and reconstruction planning.

**This phase:** reconstruction planner only (no PDF / SVG / CDR export).

## Setup

```bash
cd backend
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
copy .env.example .env   # or: cp .env.example .env
```

## Run

From the `backend/` directory:

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

- API docs: http://127.0.0.1:8000/docs  
- Health: http://127.0.0.1:8000/health  
- Upload: `POST http://127.0.0.1:8000/api/upload` (multipart field `file`)

## Layout analysis

```bash
curl -X POST http://127.0.0.1:8000/api/layout ^
  -H "Content-Type: application/json" ^
  -d "{\"image_id\":\"YOUR_IMAGE_UUID\"}"
```

Uses a Windows-safe OpenCV hybrid pipeline (contours, Hough lines, morphology, color panels, texture regions) plus OCR association.

Saves:
- `results/layout_<image_id>.json`
- `debug/layout_<image_id>.png`

## Typography

```bash
curl -X POST http://127.0.0.1:8000/api/typography ^
  -H "Content-Type: application/json" ^
  -d "{\"image_id\":\"YOUR_IMAGE_UUID\"}"
```

Requires OCR results (and uses layout JSON when available). Estimates font size, colors, bold/italic/underline probabilities, alignment, spacing, hierarchy — **not** exact font families.

Saves:
- `results/typography_<image_id>.json`
- `debug/typography_<image_id>.png`

## Reconstruction plan

```bash
curl -X POST http://127.0.0.1:8000/api/reconstruction ^
  -H "Content-Type: application/json" ^
  -d "{\"image_id\":\"YOUR_IMAGE_UUID\"}"
```

Requires layout results (uses OCR + typography when present). Decides how each object should be rebuilt later: editable text, vector shapes, embedded images, SVG paths, or ignore.

**Does not** generate PDF, SVG, CDR, or redraw the page.

Saves:
- `results/reconstruction_<image_id>.json`
- `debug/reconstruction_<image_id>.png` (green=text, blue=vector, purple=image, yellow=logo, orange=background)

## Scene graph

```bash
curl -X POST http://127.0.0.1:8000/api/scene ^
  -H "Content-Type: application/json" ^
  -d "{\"image_id\":\"YOUR_IMAGE_UUID\"}"
```

Requires reconstruction results. Builds an editable intermediate document model (groups, layers, normalized page coords) for a future PDF renderer.

**Does not** generate PDF, SVG, or CDR.

Saves:
- `results/scene_<image_id>.json`
- `debug/scene_<image_id>.png`

## Vector reconstruction

```bash
curl -X POST http://127.0.0.1:8000/api/vector ^
  -H "Content-Type: application/json" ^
  -d "{\"image_id\":\"YOUR_IMAGE_UUID\"}"
```

Requires scene graph. Reconstructs rectangles, lines, panels, paths, gradients, and color regions as editable vector data.

**Does not** generate PDF or export SVG files.

Saves:
- `results/vector_<image_id>.json`
- `debug/vector_<image_id>.png`

## Editable PDF render

```bash
curl -X POST http://127.0.0.1:8000/api/render ^
  -H "Content-Type: application/json" ^
  -d "{\"image_id\":\"YOUR_IMAGE_UUID\"}"
```

Requires scene + vector results. Builds a real editable PDF (ReportLab text/paths + embedded image crops). Download via `GET /api/output/output_<id>.pdf`.

Saves:
- `output/output_<image_id>.pdf`

