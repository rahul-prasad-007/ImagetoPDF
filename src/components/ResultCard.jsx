/**
 * Result area — download + preview only (pipeline stats hidden from end users).
 */
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
  const hasOpt = Boolean(optimizeResult?.success)
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
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.45, ease: [0.22, 1, 0.36, 1] }}
      className="rounded-2xl border border-border bg-panel p-6 sm:p-8"
    >
      <div className="text-center">
        <div
          className={`mx-auto mb-5 flex h-16 w-16 items-center justify-center rounded-2xl ${
            busy ? 'bg-accent/15 ring-1 ring-accent/30' : 'bg-success-soft ring-1 ring-success/30'
          }`}
        >
          {busy ? (
            <Loader2 className="h-8 w-8 text-accent animate-spin" strokeWidth={1.75} />
          ) : (
            <FileCheck2 className="h-8 w-8 text-success" strokeWidth={1.75} />
          )}
        </div>

        {isOptimizing ? (
          <>
            <h2 className="font-display text-xl font-semibold text-white">Polishing your PDF…</h2>
            <p className="mt-2 text-sm text-muted">Matching layout and quality to your image.</p>
          </>
        ) : isRendering ? (
          <>
            <h2 className="font-display text-xl font-semibold text-white">Building editable PDF…</h2>
            <p className="mt-2 text-sm text-muted">Rebuilding text and shapes — almost there.</p>
          </>
        ) : hasPdf ? (
          <>
            <h2 className="font-display text-xl font-semibold text-white sm:text-2xl">
              Your PDF is ready
            </h2>
            <p className="mt-2 mx-auto max-w-md text-sm text-muted leading-relaxed">
              Download and open in CorelDRAW, Illustrator, or any PDF editor.
            </p>
            {similarity != null && !Number.isNaN(Number(similarity)) && (
              <p className="mt-3 inline-flex items-center gap-1.5 rounded-full border border-accent/25 bg-accent/10 px-3 py-1 text-xs font-semibold text-accent">
                <Sparkles className="h-3 w-3" />
                Match quality {Number(similarity).toFixed(0)}%
              </p>
            )}

            <div className="mt-6 flex flex-col sm:flex-row items-center justify-center gap-3">
              <a
                href={previewUrl}
                download
                className="inline-flex w-full sm:w-auto items-center justify-center gap-2 rounded-xl bg-accent px-7 py-3.5 text-sm font-semibold text-white shadow-lg shadow-accent/25 hover:bg-secondary transition-colors"
              >
                <Download className="h-4 w-4" />
                Download PDF
              </a>
              <button
                type="button"
                onClick={() => setShowPreview((v) => !v)}
                className="inline-flex w-full sm:w-auto items-center justify-center gap-2 rounded-xl border border-border bg-panel-2 px-7 py-3.5 text-sm font-semibold text-white hover:bg-border/40 transition-colors"
              >
                <Eye className="h-4 w-4 text-accent" />
                {showPreview ? 'Hide Preview' : 'Preview'}
              </button>
            </div>
          </>
        ) : (
          <>
            <h2 className="font-display text-xl font-semibold text-white">Image received</h2>
            <p className="mt-2 text-sm text-muted">Working on your editable PDF…</p>
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
            <div className="rounded-xl border border-border overflow-hidden bg-ink">
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
