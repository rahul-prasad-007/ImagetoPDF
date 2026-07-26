/**
 * Collapsible Reconstruction Plan panel — decision summary counts.
 */
import { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { ChevronDown, Layers } from 'lucide-react'

function ReconstructionPanel({ reconstructionResult = null, isLoading = false }) {
  const [open, setOpen] = useState(true)

  if (!reconstructionResult && !isLoading) return null

  const summary = reconstructionResult?.summary ?? {}
  const counts = summary.counts ?? {}
  const objects = reconstructionResult?.objects ?? []

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
            <Layers className="h-5 w-5" strokeWidth={1.75} />
          </div>
          <div className="min-w-0">
            <h2 className="font-display text-lg font-semibold text-text">Reconstruction Plan</h2>
            <p className="text-xs text-slate-500 truncate">
              {isLoading
                ? 'Planning how objects will be rebuilt…'
                : reconstructionResult?.message || 'Reconstruction decisions ready'}
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
            key="recon-body"
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.28 }}
            className="overflow-hidden"
          >
            <div className="px-5 sm:px-6 pb-5 sm:pb-6 border-t border-border">
              {isLoading ? (
                <p className="pt-5 text-sm text-slate-500 text-center">
                  Assigning text, vector, and image reconstruction strategies…
                </p>
              ) : (
                <>
                  <div className="pt-5 grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-3">
                    {[
                      { label: 'Editable Text', value: String(counts.editable_text ?? 0) },
                      { label: 'Vector Shapes', value: String(counts.vector_shapes ?? 0) },
                      { label: 'Embedded Images', value: String(counts.embedded_images ?? 0) },
                      { label: 'Background Regions', value: String(counts.background_regions ?? 0) },
                      { label: 'SVG Paths', value: String(counts.svg_paths ?? 0) },
                      { label: 'Ignored Objects', value: String(counts.ignored ?? 0) },
                      {
                        label: 'Overall Score',
                        value:
                          summary.overall_score != null
                            ? `${Number(summary.overall_score).toFixed(0)}`
                            : '—',
                      },
                      {
                        label: 'Avg Confidence',
                        value:
                          summary.average_confidence != null
                            ? `${Number(summary.average_confidence).toFixed(0)}%`
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

                  {objects.length > 0 && (
                    <div className="mt-5 overflow-x-auto rounded-xl border border-border">
                      <table className="w-full min-w-[640px] text-left text-sm">
                        <thead className="bg-slate-50 text-xs uppercase tracking-wide text-slate-500">
                          <tr>
                            <th className="px-3 py-2.5 font-semibold">#</th>
                            <th className="px-3 py-2.5 font-semibold">Layout Type</th>
                            <th className="px-3 py-2.5 font-semibold">Reconstruction</th>
                            <th className="px-3 py-2.5 font-semibold">Layer</th>
                            <th className="px-3 py-2.5 font-semibold">Confidence</th>
                            <th className="px-3 py-2.5 font-semibold">Reason</th>
                          </tr>
                        </thead>
                        <tbody className="divide-y divide-border">
                          {objects
                            .filter((o) => o.reconstruction !== 'IGNORE')
                            .slice(0, 40)
                            .map((row) => (
                              <tr key={row.id} className="hover:bg-slate-50/80">
                                <td className="px-3 py-2.5 text-slate-400 tabular-nums">
                                  {row.id}
                                </td>
                                <td className="px-3 py-2.5 font-medium text-text">{row.type}</td>
                                <td className="px-3 py-2.5 text-slate-700">
                                  {row.reconstruction}
                                </td>
                                <td className="px-3 py-2.5 tabular-nums text-slate-600">
                                  {row.layer}
                                </td>
                                <td className="px-3 py-2.5 tabular-nums text-slate-600">
                                  {Math.round(row.confidence)}%
                                </td>
                                <td className="px-3 py-2.5 text-slate-500 max-w-[240px]">
                                  <span className="line-clamp-1">{row.reason}</span>
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

export default ReconstructionPanel
