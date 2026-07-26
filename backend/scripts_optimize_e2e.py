"""E2E: upload → … → optimize for fixture document types."""

from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

BASE = "http://127.0.0.1:8000"
FIXTURES = Path(__file__).resolve().parent / "fixtures"
TYPES = ["business_card", "certificate", "poster", "wedding_card", "flyer"]


def post_json(path: str, payload: dict, timeout: int = 600) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{BASE}{path}",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def upload(path: Path) -> dict:
    boundary = "----ImgToPdfBoundary7MA4YWxkTrZu0gW"
    body = b""
    body += f"--{boundary}\r\n".encode()
    body += (
        f'Content-Disposition: form-data; name="file"; filename="{path.name}"\r\n'
    ).encode()
    body += b"Content-Type: image/png\r\n\r\n"
    body += path.read_bytes()
    body += f"\r\n--{boundary}--\r\n".encode()
    req = urllib.request.Request(
        f"{BASE}/api/upload",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read().decode("utf-8"))


def run_one(name: str) -> dict:
    img = FIXTURES / f"{name}.png"
    if not img.is_file():
        raise FileNotFoundError(img)
    t0 = time.time()
    up = upload(img)
    iid = up["image_id"]
    print(f"[{name}] uploaded {iid}", flush=True)
    for step in (
        "/api/ocr",
        "/api/layout",
        "/api/typography",
        "/api/reconstruction",
        "/api/scene",
        "/api/vector",
        "/api/render",
        "/api/optimize",
    ):
        print(f"[{name}] {step} …", flush=True)
        r = post_json(step, {"image_id": iid}, timeout=900)
        if step == "/api/optimize":
            acc = r.get("summary", {}).get("accuracy", {})
            print(
                f"[{name}] DONE sim={acc.get('overall_similarity')} "
                f"text={acc.get('text_accuracy')} color={acc.get('color_accuracy')} "
                f"layout={acc.get('layout_accuracy')} "
                f"replaced={r.get('summary', {}).get('pdf_replaced')} "
                f"wall={time.time()-t0:.1f}s",
                flush=True,
            )
            return {
                "type": name,
                "image_id": iid,
                "accuracy": acc,
                "pdf_replaced": r.get("summary", {}).get("pdf_replaced"),
                "report": r.get("report"),
                "debug_image": r.get("debug_image"),
                "optimization": r.get("optimization"),
            }
    raise RuntimeError("optimize not reached")


def main() -> int:
    results = []
    for name in TYPES:
        try:
            results.append(run_one(name))
        except Exception as exc:
            print(f"[{name}] FAIL {exc}", flush=True)
            results.append({"type": name, "error": str(exc)})
    out = Path(__file__).resolve().parent / "results" / "optimize_e2e_summary.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("summary ->", out)
    failed = [r for r in results if r.get("error")]
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
