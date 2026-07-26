/**
 * Collapsible Layout Analysis panel — object counts + summary table.
 */
import { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { ChevronDown, LayoutTemplate } from 'lucide-react'

const SUMMARY_ROWS = [
  { key: 'titles', label: 'Titles' },
  { key: 'subtitles', label: 'Subtitles' },
  { key: 'paragraphs', label: 'Paragraphs' },
  { key: 'text_blocks', label: 'Text Blocks' },
  { key: 'lists', label: 'Lists' },
  { key: 'images', label: 'Images' },
  { key: 'photos', label: 'Photos' },
  { key: 'logos', label: 'Logos' },
  { key: 'icons', label: 'Icons' },
  { key: 'rectangles', label: 'Rectangles' },
  { key: 'rounded_rectangles', label: 'Rounded Rectangles' },
  { key: 'lines', label: 'Lines' },
  { key: 'circles', label: 'Circles' },
  { key: 'ellipses', label: 'Ellipses' },
  { key: 'tables', label: 'Tables' },
  { key: 'background_shapes', label: 'Background Shapes' },
  { key: 'decorative_elements', label: 'Decorative Elements' },
  { key: 'qr_codes', label: 'QR Codes' },
]

function LayoutAnalysisPanel({ layoutResult = null, isLoading = false }) {
  const [open, setOpen] = useState(true)

  if (!layoutResult && !isLoading) return null

  const counts = layoutResult?.counts ?? {}
  const objects = layoutResult?.objects ?? []
  const visibleRows = SUMMARY_ROWS.filter((row) => (counts[row.key] ?? 0) > 0)

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
          <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl bg-secondary/15 text-primary">
            <LayoutTemplate className="h-5 w-5" strokeWidth={1.75} />
          </div>
          <div className="min-w-0">
            <h2 className="font-display text-lg font-semibold text-text">Layout Analysis</h2>
            <p className="text-xs text-slate-500 truncate">
              {isLoading
                ? 'Detecting layout objects…'
                : layoutResult?.message || 'Document structure ready'}
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
            key="layout-body"
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.28 }}
            className="overflow-hidden"
          >
            <div className="px-5 sm:px-6 pb-5 sm:pb-6 border-t border-border">
              {isLoading ? (
                <p className="pt-5 text-sm text-slate-500 text-center">
                  Building the document layout model…
                </p>
              ) : (
                <>
                  <div className="pt-5 grid grid-cols-2 sm:grid-cols-4 gap-3">
                    {[
                      { label: 'Objects', value: String(counts.total ?? objects.length) },
                      {
                        label: 'Text Regions',
                        value: String(
                          (counts.titles || 0) +
                            (counts.subtitles || 0) +
                            (counts.paragraphs || 0) +
                            (counts.text_blocks || 0) +
                            (counts.lists || 0),
                        ),
                      },
                      {
                        label: 'Shapes',
                        value: String(
                          (counts.rectangles || 0) +
                            (counts.rounded_rectangles || 0) +
                            (counts.lines || 0) +
                            (counts.circles || 0) +
                            (counts.ellipses || 0) +
                            (counts.background_shapes || 0),
                        ),
                      },
                      {
                        label: 'Time',
                        value:
                          layoutResult?.processing_time_ms != null
                            ? `${Number(layoutResult.processing_time_ms).toFixed(0)} ms`
                            : '—',
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

                  <div className="mt-5 overflow-x-auto rounded-xl border border-border">
                    <table className="w-full min-w-[360px] text-left text-sm">
                      <thead className="bg-slate-50 text-xs uppercase tracking-wide text-slate-500">
                        <tr>
                          <th className="px-3 py-2.5 font-semibold">Object Type</th>
                          <th className="px-3 py-2.5 font-semibold">Count</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-border">
                        {(visibleRows.length ? visibleRows : SUMMARY_ROWS.slice(0, 6)).map(
                          (row) => (
                            <tr key={row.key} className="hover:bg-slate-50/80">
                              <td className="px-3 py-2.5 font-medium text-text">{row.label}</td>
                              <td className="px-3 py-2.5 tabular-nums text-slate-600">
                                {counts[row.key] ?? 0}
                              </td>
                            </tr>
                          ),
                        )}
                      </tbody>
                    </table>
                  </div>
                </>
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.section>
  )
}

export default LayoutAnalysisPanel
