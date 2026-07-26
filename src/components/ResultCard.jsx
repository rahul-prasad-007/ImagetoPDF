import { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Download, Eye, FileCheck2, Loader2, Sparkles } from 'lucide-react'

function ResultCard({
  uploadResult = null,
  pdfResult = null,
  optimizeResult = null,
  isRendering = false,
  isOptimizing = false,
}) {
  const [showPreview, setShowPreview] = useState(true)
  const hasPdf = Boolean(
    (optimizeResult?.download_url || pdfResult?.download_url) &&
      (optimizeResult?.success || pdfResult?.success),
  )
  const previewUrl = optimizeResult?.download_url || pdfResult?.download_url || null
  const busy = isRendering || isOptimizing
  const similarity = optimizeResult?.summary?.accuracy?.overall_similarity

  if (!busy && !hasPdf && !uploadResult?.success) return null

  return (
    <motion.section
      initial={{ opacity: 0, y: 16 }}
      animate={{ opacity: 1, y: 0 }}
      className="rounded-xl border border-border bg-panel p-5 sm:p-7 shadow-sm shadow-slate-900/5 text-left"
    >
      <div className="flex flex-col sm:flex-row sm:items-start gap-4">
        <div
          className={`flex h-12 w-12 shrink-0 items-center justify-center rounded-lg ${
            busy ? 'bg-accent-soft text-accent' : 'bg-success-soft text-success'
          }`}
        >
          {busy ? (
            <Loader2 className="h-6 w-6 animate-spin" strokeWidth={1.85} />
          ) : (
            <FileCheck2 className="h-6 w-6" strokeWidth={1.85} />
          )}
        </div>

        <div className="flex-1 min-w-0">
          {isOptimizing ? (
            <>
              <h2 className="font-display text-lg font-semibold text-ink">Polishing your PDF…</h2>
              <p className="mt-1 text-sm text-muted">Matching layout and quality to your image.</p>
            </>
          ) : isRendering ? (
            <>
              <h2 className="font-display text-lg font-semibold text-ink">Building editable PDF…</h2>
              <p className="mt-1 text-sm text-muted">Rebuilding text and shapes — almost there.</p>
            </>
          ) : hasPdf ? (
            <>
              <h2 className="font-display text-xl font-semibold text-ink">Your PDF is ready</h2>
              <p className="mt-1 text-sm text-muted">
                Download and open in CorelDRAW, Illustrator, or any PDF editor.
              </p>
              {similarity != null && !Number.isNaN(Number(similarity)) && (
                <p className="mt-3 inline-flex items-center gap-1.5 rounded-md border border-accent/20 bg-accent-soft px-2.5 py-1 text-xs font-semibold text-accent">
                  <Sparkles className="h-3 w-3" />
                  Match quality {Number(similarity).toFixed(0)}%
                </p>
              )}

              <div className="mt-5 flex flex-col sm:flex-row gap-2.5">
                <a
                  href={previewUrl}
                  download
                  className="inline-flex items-center justify-center gap-2 rounded-lg bg-accent px-5 py-3 text-sm font-semibold text-white shadow-sm shadow-accent/20 hover:bg-secondary transition-colors"
                >
                  <Download className="h-4 w-4" />
                  Download PDF
                </a>
                <button
                  type="button"
                  onClick={() => setShowPreview((v) => !v)}
                  className="inline-flex items-center justify-center gap-2 rounded-lg border border-border bg-panel-2 px-5 py-3 text-sm font-semibold text-ink hover:bg-surface transition-colors"
                >
                  <Eye className="h-4 w-4 text-accent" />
                  {showPreview ? 'Hide Preview' : 'Preview'}
                </button>
              </div>
            </>
          ) : (
            <>
              <h2 className="font-display text-lg font-semibold text-ink">Image received</h2>
              <p className="mt-1 text-sm text-muted">Working on your editable PDF…</p>
            </>
          )}
        </div>
      </div>

      <AnimatePresence>
        {hasPdf && showPreview && previewUrl && !busy && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            className="mt-6 overflow-hidden"
          >
            <div className="rounded-lg border border-border overflow-hidden bg-surface">
              <iframe
                title="PDF Preview"
                src={previewUrl}
                className="w-full h-[420px] sm:h-[560px] bg-white"
              />
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.section>
  )
}

export default ResultCard
