"""
HTML quality report for optimization results.
"""

from __future__ import annotations

from html import escape
from pathlib import Path
from typing import Any


def _pct(v: float) -> str:
    return f"{v:.2f}%"


def _metric_row(label: str, before: float, after: float, *, higher_better: bool = True) -> str:
    delta = after - before
    improved = delta > 0 if higher_better else delta < 0
    color = "#15803d" if improved else ("#b45309" if abs(delta) < 1e-6 else "#b91c1c")
    return f"""
    <tr>
      <td>{escape(label)}</td>
      <td>{before:.4f}</td>
      <td>{after:.4f}</td>
      <td style="color:{color}">{delta:+.4f}</td>
    </tr>"""


def write_html_report(
    path: Path,
    *,
    image_id: str,
    accuracy: dict[str, Any],
    before: dict[str, Any],
    after: dict[str, Any],
    object_diffs: list[dict[str, Any]],
    fixes: list[str],
    pdf_replaced: bool,
    optimization_time_ms: float,
    targets_met: dict[str, bool],
    debug_image_rel: str,
) -> None:
    overall = float(accuracy.get("overall_similarity") or 0)
    rows = "\n".join(
        _metric_row(k, float(before.get(k) or 0), float(after.get(k) or 0), higher_better=k in {"ssim", "psnr", "overall_similarity"})
        for k in (
            "overall_similarity",
            "ssim",
            "psnr",
            "pixel_difference",
            "edge_difference",
            "color_difference",
            "text_bbox_difference",
            "alignment_difference",
            "object_position_error",
            "spacing_error",
        )
    )

    obj_rows = []
    for d in object_diffs[:200]:
        sev = str(d.get("severity") or "perfect")
        color = {"perfect": "#15803d", "minor": "#ca8a04", "large": "#b91c1c"}.get(sev, "#334155")
        obj_rows.append(
            f"""<tr>
            <td>{d.get('object_id')}</td>
            <td>{escape(str(d.get('object_type') or ''))}</td>
            <td style="color:{color};font-weight:600">{escape(sev)}</td>
            <td>{float(d.get('offset_x') or 0):.2f}</td>
            <td>{float(d.get('offset_y') or 0):.2f}</td>
            <td>{float(d.get('width_difference') or 0):.2f}</td>
            <td>{float(d.get('height_difference') or 0):.2f}</td>
            <td>{float(d.get('color_difference') or 0):.3f}</td>
            </tr>"""
        )

    targets_html = "".join(
        f"<li><strong>{escape(k)}</strong>: {'✓ met' if v else '✗ below target'}</li>"
        for k, v in targets_met.items()
    )
    fixes_html = "".join(f"<li>{escape(f)}</li>" for f in fixes[:100]) or "<li>None</li>"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Optimization Report — {escape(image_id[:8])}</title>
  <style>
    :root {{
      --bg: #f8fafc;
      --card: #ffffff;
      --text: #0f172a;
      --muted: #64748b;
      --accent: #0f766e;
      --border: #e2e8f0;
    }}
    body {{
      margin: 0; padding: 32px 20px;
      font-family: "Segoe UI", system-ui, sans-serif;
      background: linear-gradient(160deg, #ecfeff 0%, #f8fafc 40%, #f1f5f9 100%);
      color: var(--text);
    }}
    .wrap {{ max-width: 960px; margin: 0 auto; }}
    h1 {{ font-size: 1.75rem; margin: 0 0 8px; letter-spacing: -0.02em; }}
    .sub {{ color: var(--muted); margin-bottom: 28px; }}
    .hero {{
      display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
      gap: 12px; margin-bottom: 28px;
    }}
    .metric {{
      background: var(--card); border: 1px solid var(--border);
      border-radius: 14px; padding: 16px 14px;
    }}
    .metric .label {{ font-size: 11px; text-transform: uppercase; letter-spacing: 0.06em; color: var(--muted); }}
    .metric .value {{ font-size: 1.55rem; font-weight: 700; margin-top: 6px; color: var(--accent); }}
    section {{
      background: var(--card); border: 1px solid var(--border);
      border-radius: 16px; padding: 20px; margin-bottom: 18px;
    }}
    h2 {{ font-size: 1.05rem; margin: 0 0 14px; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 0.875rem; }}
    th, td {{ text-align: left; padding: 8px 10px; border-bottom: 1px solid var(--border); }}
    th {{ color: var(--muted); font-weight: 600; font-size: 11px; text-transform: uppercase; }}
    img.debug {{ width: 100%; border-radius: 12px; border: 1px solid var(--border); }}
    ul {{ margin: 0; padding-left: 18px; color: var(--muted); }}
    .badge {{
      display: inline-block; padding: 4px 10px; border-radius: 999px;
      background: #ccfbf1; color: #0f766e; font-size: 12px; font-weight: 600;
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <h1>Optimization Report</h1>
    <p class="sub">Image ID <code>{escape(image_id)}</code> ·
      {optimization_time_ms:.0f} ms ·
      PDF {"replaced" if pdf_replaced else "unchanged"}
      <span class="badge">Quality Engine</span>
    </p>

    <div class="hero">
      <div class="metric"><div class="label">Overall Similarity</div><div class="value">{_pct(overall)}</div></div>
      <div class="metric"><div class="label">SSIM</div><div class="value">{float(after.get('ssim') or 0)*100:.2f}%</div></div>
      <div class="metric"><div class="label">Object Accuracy</div><div class="value">{_pct(float(accuracy.get('object_accuracy') or 0))}</div></div>
      <div class="metric"><div class="label">Text Accuracy</div><div class="value">{_pct(float(accuracy.get('text_accuracy') or 0))}</div></div>
      <div class="metric"><div class="label">Color Accuracy</div><div class="value">{_pct(float(accuracy.get('color_accuracy') or 0))}</div></div>
      <div class="metric"><div class="label">Vector Accuracy</div><div class="value">{_pct(float(accuracy.get('vector_accuracy') or 0))}</div></div>
      <div class="metric"><div class="label">Image Accuracy</div><div class="value">{_pct(float(accuracy.get('image_accuracy') or 0))}</div></div>
      <div class="metric"><div class="label">Layout Accuracy</div><div class="value">{_pct(float(accuracy.get('layout_accuracy') or 0))}</div></div>
    </div>

    <section>
      <h2>Quality Targets</h2>
      <ul>{targets_html}</ul>
    </section>

    <section>
      <h2>Before / After Metrics</h2>
      <table>
        <thead><tr><th>Metric</th><th>Before</th><th>After</th><th>Delta</th></tr></thead>
        <tbody>{rows}</tbody>
      </table>
    </section>

    <section>
      <h2>Difference Overlay</h2>
      <p class="sub">Green = perfect · Yellow = minor · Red = large</p>
      <img class="debug" src="../{escape(debug_image_rel)}" alt="Optimization debug overlay" />
    </section>

    <section>
      <h2>Automatic Fixes ({len(fixes)})</h2>
      <ul>{fixes_html}</ul>
    </section>

    <section>
      <h2>Per-Object Diffs</h2>
      <table>
        <thead>
          <tr>
            <th>ID</th><th>Type</th><th>Severity</th>
            <th>ΔX</th><th>ΔY</th><th>ΔW</th><th>ΔH</th><th>Color</th>
          </tr>
        </thead>
        <tbody>{''.join(obj_rows) or '<tr><td colspan="8">No objects compared</td></tr>'}</tbody>
      </table>
    </section>
  </div>
</body>
</html>
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")
