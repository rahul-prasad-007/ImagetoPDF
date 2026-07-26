/**
 * Collapsible Scene Graph panel — editable document model summary.
 */
import { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { ChevronDown, Network } from 'lucide-react'

function SceneGraphPanel({ sceneResult = null, isLoading = false }) {
  const [open, setOpen] = useState(true)

  if (!sceneResult && !isLoading) return null

  const summary = sceneResult?.summary ?? {}
  const counts = summary.counts ?? {}
  const page = sceneResult?.page ?? {}
  const validation = summary.validation ?? {}

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
            <Network className="h-5 w-5" strokeWidth={1.75} />
          </div>
          <div className="min-w-0">
            <h2 className="font-display text-lg font-semibold text-text">Scene Graph</h2>
            <p className="text-xs text-slate-500 truncate">
              {isLoading
                ? 'Building editable document model…'
                : sceneResult?.message || 'Scene graph ready'}
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
            key="scene-body"
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.28 }}
            className="overflow-hidden"
          >
            <div className="px-5 sm:px-6 pb-5 sm:pb-6 border-t border-border">
              {isLoading ? (
                <p className="pt-5 text-sm text-slate-500 text-center">
                  Assembling layers, groups, and editable objects…
                </p>
              ) : (
                <>
                  <div className="pt-5 grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-3">
                    {[
                      { label: 'Total Objects', value: String(counts.total_objects ?? 0) },
                      { label: 'Groups', value: String(counts.groups ?? 0) },
                      { label: 'Layers', value: String(counts.layers ?? 0) },
                      { label: 'Text Objects', value: String(counts.text_objects ?? 0) },
                      { label: 'Image Objects', value: String(counts.image_objects ?? 0) },
                      { label: 'Vector Objects', value: String(counts.vector_objects ?? 0) },
                      {
                        label: 'Overall Build Time',
                        value:
                          summary.build_time_ms != null
                            ? `${Number(summary.build_time_ms).toFixed(1)} ms`
                            : '—',
                      },
                      {
                        label: 'Page',
                        value: page.page_format
                          ? `${page.page_format} ${page.orientation || ''}`.trim()
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

                  {validation && (
                    <p className="mt-4 text-xs text-slate-500">
                      Validation:{' '}
                      <span
                        className={
                          validation.ok ? 'text-accent font-medium' : 'text-amber-600 font-medium'
                        }
                      >
                        {validation.ok ? 'OK' : 'Issues found'}
                      </span>
                      {validation.overlapping_layer_pairs
                        ? ` · ${validation.overlapping_layer_pairs} overlap pair(s)`
                        : ''}
                      {summary.memory_kb
                        ? ` · ~${Number(summary.memory_kb).toFixed(0)} KB process`
                        : ''}
                    </p>
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

export default SceneGraphPanel
