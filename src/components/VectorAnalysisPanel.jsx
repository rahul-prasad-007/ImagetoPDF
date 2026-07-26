/**
 * Collapsible Vector Analysis panel — reconstructed shapes summary.
 */
import { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { ChevronDown, Shapes } from 'lucide-react'

function VectorAnalysisPanel({ vectorResult = null, isLoading = false }) {
  const [open, setOpen] = useState(true)

  if (!vectorResult && !isLoading) return null

  const summary = vectorResult?.summary ?? {}
  const counts = summary.counts ?? {}

  return (
    <motion.section
      initial={{ opacity: 0, y: 16 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4, ease: [0.22, 1, 0.36, 1] }}
      className="rounded-2xl border border-border bg-card shadow-[0_4px_24px_rgba(15,23,42,0.06)] overflow-hidden"
    >
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className="w-full flex items-center justify-between gap-3 px-5 sm:px-6 py-4 text-left hover:bg-slate-50/80 transition-colors"
        aria-expanded={open}
      >
        <div className="flex items-center gap-3 min-w-0">
          <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl bg-primary/10 text-primary">
            <Shapes className="h-5 w-5" strokeWidth={1.75} />
          </div>
          <div className="min-w-0">
            <h2 className="font-display text-lg font-semibold text-text">Vector Analysis</h2>
            <p className="text-xs text-slate-500 truncate">
              {isLoading
                ? 'Reconstructing shapes, panels, and paths…'
                : vectorResult?.message || 'Vector reconstruction ready'}
            </p>
          </div>
        </div>
        <ChevronDown
          className={`h-5 w-5 text-slate-400 transition-transform duration-300 ${
            open ? 'rotate-180' : ''
          }`}
        />
      </button>

      <AnimatePresence initial={false}>
        {open && (
          <motion.div
            key="vector-body"
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.28 }}
            className="overflow-hidden"
          >
            <div className="px-5 sm:px-6 pb-5 sm:pb-6 border-t border-border">
              {isLoading ? (
                <p className="pt-5 text-sm text-slate-500 text-center">
                  Detecting rectangles, curves, gradients, and color regions…
                </p>
              ) : (
                <div className="pt-5 grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-3">
                  {[
                    { label: 'Rectangles', value: String(counts.rectangles ?? 0) },
                    {
                      label: 'Rounded Rectangles',
                      value: String(counts.rounded_rectangles ?? 0),
                    },
                    { label: 'Lines', value: String(counts.lines ?? 0) },
                    { label: 'Paths', value: String(counts.paths ?? 0) },
                    { label: 'Gradients', value: String(counts.gradients ?? 0) },
                    { label: 'Color Regions', value: String(counts.color_regions ?? 0) },
                    {
                      label: 'Vector Score',
                      value:
                        summary.vector_score != null
                          ? `${Number(summary.vector_score).toFixed(0)}`
                          : '—',
                    },
                    {
                      label: 'Merged Shapes',
                      value: String(counts.merged_shapes ?? 0),
                    },
                  ].map((stat) => (
                    <div
                      key={stat.label}
                      className="rounded-xl border border-border bg-slate-50 px-3.5 py-3"
                    >
                      <p className="text-[11px] font-medium uppercase tracking-wide text-slate-400">
                        {stat.label}
                      </p>
                      <p className="mt-1 font-display text-xl font-semibold text-text">
                        {stat.value}
                      </p>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.section>
  )
}

export default VectorAnalysisPanel
