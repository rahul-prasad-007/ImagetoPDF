/**
 * Result area — Optimization Report + editable PDF download/preview.
 */
import { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Download, ExternalLink, Eye, FileCheck2, Gauge, Loader2 } from 'lucide-react'

function pct(value) {
  if (value == null || Number.isNaN(Number(value))) return '—'
  return `${Number(value).toFixed(1)}%`
}

function ResultCard({
  uploadResult = null,
  pdfResult = null,
  optimizeResult = null,
  isRendering = false,
  isOptimizing = false,
}) {
  const [showPreview, setShowPreview] = useState(true)
  const hasOpt = Boolean(optimizeResult?.success)
  const hasPdf = Boolean(
    (optimizeResult?.download_url || pdfResult?.download_url) &&
      (optimizeResult?.success || pdfResult?.success),
  )
  const previewUrl = optimizeResult?.download_url || pdfResult?.download_url || null
  const summary = optimizeResult?.summary ?? {}
  const accuracy = summary.accuracy ?? {}
  const pdfSummary = pdfResult?.summary ?? {}
  const counts = pdfSummary.counts ?? {}
  const busy = isRendering || isOptimizing
  const documentMode =
    pdfResult?.document_mode ||
    optimizeResult?.document_mode ||
    uploadResult?.document_mode ||
    null
  const modeLabel = {
    ruled_form: 'Ruled form / bill book',
    designed_invoice: 'Designed invoice',
    poster: 'Poster / card / flyer',
  }[documentMode] || documentMode

  return (
    <motion.section
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: 0.3, duration: 0.5, ease: [0.22, 1, 0.36, 1] }}
      className="rounded-2xl border border-border bg-card p-6 sm:p-10 shadow-[0_4px_24px_rgba(15,23,42,0.06)]"
    >
      <div className="text-center">
        <motion.div
          initial={{ scale: 0.9, opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
          transition={{ delay: 0.4, duration: 0.45 }}
          className={`mx-auto mb-5 flex h-20 w-20 items-center justify-center rounded-full ${
            hasOpt || busy ? 'bg-accent/15' : 'bg-accent/10'
          }`}
        >
          {busy ? (
            <Loader2 className="h-10 w-10 text-primary animate-spin" strokeWidth={1.5} />
          ) : hasOpt ? (
            <Gauge className="h-10 w-10 text-accent" strokeWidth={1.5} />
          ) : (
            <FileCheck2 className="h-10 w-10 text-accent" strokeWidth={1.5} />
          )}
        </motion.div>

        {isOptimizing ? (
          <>
            <h2 className="font-display text-xl font-semibold text-text">Optimizing quality…</h2>
            <p className="mt-2 mx-auto max-w-md text-sm text-slate-500 leading-relaxed">
              Comparing the PDF to your original image and applying automatic fixes.
            </p>
          </>
        ) : isRendering ? (
          <>
            <h2 className="font-display text-xl font-semibold text-text">Rendering…</h2>
            <p className="mt-2 mx-auto max-w-md text-sm text-slate-500 leading-relaxed">
              Reconstructing editable text, vectors, and images into a PDF.
            </p>
          </>
        ) : hasOpt ? (
          <>
            <h2 className="font-display text-xl font-semibold text-text">Optimization Report</h2>
            <p className="mt-2 mx-auto max-w-md text-sm text-slate-500 leading-relaxed">
              {optimizeResult.message ||
                'Quality compared against the original. PDF updated when improvement is detected.'}
            </p>
            {modeLabel ? (
              <p className="mt-3 text-xs font-medium uppercase tracking-wide text-slate-400">
                Hybrid mode · {modeLabel}
              </p>
            ) : null}

            <dl className="mt-6 mx-auto max-w-2xl grid grid-cols-2 sm:grid-cols-3 gap-3 text-left">
              {[
                { label: 'Overall Similarity', value: pct(accuracy.overall_similarity) },
                { label: 'Text Accuracy', value: pct(accuracy.text_accuracy) },
                { label: 'Layout Accuracy', value: pct(accuracy.layout_accuracy) },
                { label: 'Color Accuracy', value: pct(accuracy.color_accuracy) },
                { label: 'Object Accuracy', value: pct(accuracy.object_accuracy) },
                {
                  label: 'Optimization Time',
                  value:
                    summary.optimization_time_ms != null
                      ? `${Number(summary.optimization_time_ms).toFixed(0)} ms`
                      : '—',
                },
              ].map((item) => (
                <div
                  key={item.label}
                  className="rounded-xl border border-border bg-slate-50 px-3.5 py-2.5"
                >
                  <dt className="text-[11px] font-medium uppercase tracking-wide text-slate-400">
                    {item.label}
                  </dt>
                  <dd className="mt-0.5 text-sm font-medium text-text break-all">{item.value}</dd>
                </div>
              ))}
            </dl>

            {(summary.pdf_replaced || summary.improved) && (
              <p className="mt-4 text-xs font-medium text-accent">
                Optimized PDF replaced the previous render.
              </p>
            )}

            {hasPdf && previewUrl && (
              <div className="mt-6 flex flex-col sm:flex-row items-center justify-center gap-3">
                <a
                  href={previewUrl}
                  download
                  className="inline-flex items-center justify-center gap-2 rounded-xl bg-primary px-6 py-3 text-sm font-semibold text-white shadow-md shadow-primary/25 hover:bg-secondary transition-colors"
                >
                  <Download className="h-4 w-4" />
                  Download PDF
                </a>
                <button
                  type="button"
                  onClick={() => setShowPreview((v) => !v)}
                  className="inline-flex items-center justify-center gap-2 rounded-xl border border-border bg-white px-6 py-3 text-sm font-semibold text-text hover:bg-slate-50 transition-colors"
                >
                  <Eye className="h-4 w-4" />
                  {showPreview ? 'Hide Preview' : 'Preview PDF'}
                </button>
                <a
                  href={previewUrl}
                  target="_blank"
                  rel="noreferrer"
                  className="inline-flex items-center justify-center gap-2 rounded-xl border border-border bg-white px-6 py-3 text-sm font-semibold text-text hover:bg-slate-50 transition-colors"
                >
                  <ExternalLink className="h-4 w-4" />
                  Open in New Tab
                </a>
              </div>
            )}
          </>
        ) : hasPdf ? (
          <>
            <h2 className="font-display text-xl font-semibold text-text">
              Editable PDF Generated Successfully
            </h2>
            <p className="mt-2 mx-auto max-w-md text-sm text-slate-500 leading-relaxed">
              {pdfResult.message || 'Your reconstructed PDF is ready to download and edit.'}
            </p>

            <dl className="mt-6 mx-auto max-w-lg grid grid-cols-2 sm:grid-cols-3 gap-3 text-left">
              {[
                { label: 'PDF Size', value: pdfSummary.pdf_size || '—' },
                { label: 'Text Objects', value: String(counts.text_count ?? 0) },
                { label: 'Vectors', value: String(counts.vector_count ?? 0) },
                { label: 'Images', value: String(counts.image_count ?? 0) },
                {
                  label: 'Page',
                  value: pdfSummary.page_format
                    ? `${pdfSummary.page_format} ${pdfSummary.orientation || ''}`.trim()
                    : '—',
                },
                {
                  label: 'Render Time',
                  value:
                    pdfSummary.render_time_ms != null
                      ? `${Number(pdfSummary.render_time_ms).toFixed(0)} ms`
                      : '—',
                },
              ].map((item) => (
                <div
                  key={item.label}
                  className="rounded-xl border border-border bg-slate-50 px-3.5 py-2.5"
                >
                  <dt className="text-[11px] font-medium uppercase tracking-wide text-slate-400">
                    {item.label}
                  </dt>
                  <dd className="mt-0.5 text-sm font-medium text-text break-all">{item.value}</dd>
                </div>
              ))}
            </dl>

            <div className="mt-6 flex flex-col sm:flex-row items-center justify-center gap-3">
              <a
                href={previewUrl}
                download
                className="inline-flex items-center justify-center gap-2 rounded-xl bg-primary px-6 py-3 text-sm font-semibold text-white shadow-md shadow-primary/25 hover:bg-secondary transition-colors"
              >
                <Download className="h-4 w-4" />
                Download PDF
              </a>
              <button
                type="button"
                onClick={() => setShowPreview((v) => !v)}
                className="inline-flex items-center justify-center gap-2 rounded-xl border border-border bg-white px-6 py-3 text-sm font-semibold text-text hover:bg-slate-50 transition-colors"
              >
                <Eye className="h-4 w-4" />
                {showPreview ? 'Hide Preview' : 'Preview PDF'}
              </button>
              <a
                href={previewUrl}
                target="_blank"
                rel="noreferrer"
                className="inline-flex items-center justify-center gap-2 rounded-xl border border-border bg-white px-6 py-3 text-sm font-semibold text-text hover:bg-slate-50 transition-colors"
              >
                <ExternalLink className="h-4 w-4" />
                Open in New Tab
              </a>
            </div>
          </>
        ) : uploadResult?.success ? (
          <>
            <h2 className="font-display text-xl font-semibold text-text">Image processed</h2>
            <p className="mt-2 mx-auto max-w-md text-sm text-slate-500 leading-relaxed">
              {uploadResult.message} PDF will appear after the full pipeline finishes.
            </p>
          </>
        ) : (
          <>
            <h2 className="font-display text-xl font-semibold text-text">Ready for your PDF</h2>
            <p className="mt-2 mx-auto max-w-md text-sm text-slate-500 leading-relaxed">
              Your editable PDF will appear here after generation.
            </p>
          </>
        )}
      </div>

      <AnimatePresence>
        {hasPdf && showPreview && previewUrl && !busy && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            className="mt-8 overflow-hidden"
          >
            <div className="rounded-xl border border-border overflow-hidden bg-slate-100">
              <iframe
                title="PDF Preview"
                src={previewUrl}
                className="w-full h-[480px] sm:h-[640px] bg-white"
              />
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.section>
  )
}

export default ResultCard
