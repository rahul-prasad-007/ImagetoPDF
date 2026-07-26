/**
 * Collapsible OCR results panel — stats + clean text-block table.
 */
import { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { ChevronDown, ScanText } from 'lucide-react'

function OcrResultsPanel({ ocrResult = null, isLoading = false }) {
  const [open, setOpen] = useState(true)

  if (!ocrResult && !isLoading) return null

  const blocks = ocrResult?.text_blocks ?? []
  const avg = ocrResult?.average_confidence
  const timeMs = ocrResult?.processing_time_ms

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
            <ScanText className="h-5 w-5" strokeWidth={1.75} />
          </div>
          <div className="min-w-0">
            <h2 className="font-display text-lg font-semibold text-text">OCR Results</h2>
            <p className="text-xs text-slate-500 truncate">
              {isLoading
                ? 'Analyzing text…'
                : ocrResult?.warning || ocrResult?.message || 'Text extraction complete'}
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
            key="ocr-body"
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.28 }}
            className="overflow-hidden"
          >
            <div className="px-5 sm:px-6 pb-5 sm:pb-6 border-t border-border">
              {isLoading ? (
                <p className="pt-5 text-sm text-slate-500 text-center">
                  Running OCR on the processed image…
                </p>
              ) : (
                <>
                  <div className="pt-5 grid grid-cols-1 sm:grid-cols-3 gap-3">
                    {[
                      {
                        label: 'Total Text Blocks',
                        value: String(ocrResult?.total_blocks ?? blocks.length),
                      },
                      {
                        label: 'Average Confidence',
                        value:
                          avg != null ? `${(Number(avg) * 100).toFixed(1)}%` : '—',
                      },
                      {
                        label: 'OCR Processing Time',
                        value: timeMs != null ? `${Number(timeMs).toFixed(0)} ms` : '—',
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

                  {blocks.length === 0 ? (
                    <p className="mt-5 text-sm text-slate-500 text-center">
                      No text blocks were detected in this image.
                    </p>
                  ) : (
                    <div className="mt-5 overflow-x-auto rounded-xl border border-border">
                      <table className="w-full min-w-[640px] text-left text-sm">
                        <thead className="bg-slate-50 text-xs uppercase tracking-wide text-slate-500">
                          <tr>
                            <th className="px-3 py-2.5 font-semibold">#</th>
                            <th className="px-3 py-2.5 font-semibold">Text</th>
                            <th className="px-3 py-2.5 font-semibold">Confidence</th>
                            <th className="px-3 py-2.5 font-semibold">Size (W×H)</th>
                            <th className="px-3 py-2.5 font-semibold">Line</th>
                            <th className="px-3 py-2.5 font-semibold">Para</th>
                          </tr>
                        </thead>
                        <tbody className="divide-y divide-border">
                          {blocks.map((block) => (
                            <tr key={block.id} className="hover:bg-slate-50/80">
                              <td className="px-3 py-2.5 text-slate-400 tabular-nums">
                                {block.id}
                              </td>
                              <td className="px-3 py-2.5 font-medium text-text max-w-[280px]">
                                <span className="line-clamp-2">{block.text}</span>
                              </td>
                              <td className="px-3 py-2.5 tabular-nums">
                                <span
                                  className={
                                    block.confidence >= 0.6
                                      ? 'text-accent font-semibold'
                                      : 'text-error font-semibold'
                                  }
                                >
                                  {(block.confidence * 100).toFixed(1)}%
                                </span>
                              </td>
                              <td className="px-3 py-2.5 text-slate-600 tabular-nums whitespace-nowrap">
                                {Math.round(block.width)} × {Math.round(block.height)}
                              </td>
                              <td className="px-3 py-2.5 text-slate-600 tabular-nums">
                                {block.line}
                              </td>
                              <td className="px-3 py-2.5 text-slate-600 tabular-nums">
                                {block.paragraph}
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  )}
                </>
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.section>
  )
}

export default OcrResultsPanel
