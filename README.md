# Image to Editable PDF

Convert document images into editable PDFs (text + vectors). React (Vite) frontend + FastAPI backend with PaddleOCR.

**Live repo:** [rahul-prasad-007/ImagetoPDF](https://github.com/rahul-prasad-007/ImagetoPDF)

## Stack

- React + Vite + Tailwind
- FastAPI + OpenCV + PaddleOCR + ReportLab + PyMuPDF

## Local development

```bash
# Frontend
npm install
npm run dev

# Backend (separate terminal)
cd backend
python -m venv .venv
# Windows: .\.venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Open http://127.0.0.1:5173 (Vite proxies `/api` → backend).

## Docker (production)

```bash
docker build -t imagetopdf .
docker run --rm -p 8000:8000 imagetopdf
```

App: http://127.0.0.1:8000 · Health: http://127.0.0.1:8000/health

Optional: drop real `AAText.ttf` into `backend/fonts/` for Hindi glyph matching.

## Deploy

- **Render:** connect this repo; uses `Dockerfile` + `render.yaml`
- **Railway:** connect this repo; uses `Dockerfile` + `railway.toml`

First OCR request downloads Paddle models (can take several minutes). Prefer a plan with ≥2 GB RAM.
