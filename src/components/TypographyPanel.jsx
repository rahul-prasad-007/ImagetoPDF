/**
 * Collapsible Typography Analysis panel — style summary stats.
 */
import { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { ChevronDown, Type } from 'lucide-react'

function TypographyPanel({ typographyResult = null, isLoading = false }) {
  const [open, setOpen] = useState(true)

  if (!typographyResult && !isLoading) return null

  const summary = typographyResult?.summary ?? {}
  const align = summary.alignment_distribution ?? {}
  const styles = typographyResult?.text_styles ?? []

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
          <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl bg-accent/10 text-accent">
            <Type className="h-5 w-5" strokeWidth={1.75} />
          </div>
          <div className="min-w-0">
            <h2 className="font-display text-lg font-semibold text-text">Typography Analysis</h2>
            <p className="text-xs text-slate-500 truncate">
              {isLoading
                ? 'Estimating text styles…'
                : typographyResult?.message || 'Style metadata ready'}
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
            key="typo-body"
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.28 }}
            className="overflow-hidden"
          >
            <div className="px-5 sm:px-6 pb-5 sm:pb-6 border-t border-border">
              {isLoading ? (
                <p className="pt-5 text-sm text-slate-500 text-center">
                  Analyzing font size, color, weight, and alignment…
                </p>
              ) : (
                <>
                  <div className="pt-5 grid grid-cols-2 sm:grid-cols-4 gap-3">
                    {[
                      {
                        label: 'Avg Font Size',
                        value:
                          summary.average_font_size != null
                            ? `${Number(summary.average_font_size).toFixed(0)} px`
                            : '—',
                      },
                      {
                        label: 'Titles',
                        value: String(summary.titles ?? 0),
                      },
                      {
                        label: 'Headings',
                        value: String(
                          (summary.headings ?? 0) + (summary.subheadings ?? 0),
                        ),
                      },
                      {
                        label: 'Body Text',
                        value: String(summary.body_text ?? 0),
                      },
                      {
                        label: 'Color Count',
                        value: String(summary.unique_colors ?? 0),
                      },
                      {
                        label: 'Avg Bold',
                        value:
                          summary.average_bold != null
                            ? `${(Number(summary.average_bold) * 100).toFixed(0)}%`
                            : '—',
                      },
                      {
                        label: 'Avg Confidence',
                        value:
                          summary.average_confidence != null
                            ? `${Number(summary.average_confidence).toFixed(0)}`
                            : '—',
                      },
                      {
                        label: 'Time',
                        value:
                          typographyResult?.processing_time_ms != null
                            ? `${Number(typographyResult.processing_time_ms).toFixed(0)} ms`
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

                  {/* Alignment distribution */}
                  <div className="mt-4 rounded-xl border border-border bg-slate-50 px-3.5 py-3">
                    <p className="text-[11px] font-medium uppercase tracking-wide text-slate-400 mb-2">
                      Alignment Distribution
                    </p>
                    <div className="flex flex-wrap gap-3 text-sm">
                      {['left', 'center', 'right', 'justified'].map((key) => (
                        <span key={key} className="text-slate-600">
                          <span className="font-semibold text-text capitalize">{key}</span>
                          {': '}
                          {align[key] ?? 0}
                        </span>
                      ))}
                    </div>
                    {summary.detected_colors?.length > 0 && (
                      <div className="mt-3 flex flex-wrap items-center gap-2">
                        <span className="text-[11px] font-medium uppercase tracking-wide text-slate-400">
                          Colors
                        </span>
                        {summary.detected_colors.slice(0, 8).map((hex) => (
                          <span
                            key={hex}
                            className="inline-flex items-center gap-1.5 rounded-md border border-border bg-white px-2 py-1 text-xs font-medium text-text"
                          >
                            <span
                              className="h-3 w-3 rounded-sm border border-border"
                              style={{ backgroundColor: hex }}
                            />
                            {hex}
                          </span>
                        ))}
                      </div>
                    )}
                  </div>

                  {/* Compact style table */}
                  {styles.length > 0 && (
                    <div className="mt-5 overflow-x-auto rounded-xl border border-border">
                      <table className="w-full min-w-[720px] text-left text-sm">
                        <thead className="bg-slate-50 text-xs uppercase tracking-wide text-slate-500">
                          <tr>
                            <th className="px-3 py-2.5 font-semibold">Text</th>
                            <th className="px-3 py-2.5 font-semibold">Size</th>
                            <th className="px-3 py-2.5 font-semibold">Color</th>
                            <th className="px-3 py-2.5 font-semibold">Hierarchy</th>
                            <th className="px-3 py-2.5 font-semibold">Align</th>
                            <th className="px-3 py-2.5 font-semibold">Bold</th>
                            <th className="px-3 py-2.5 font-semibold">Conf</th>
                          </tr>
                        </thead>
                        <tbody className="divide-y divide-border">
                          {styles.map((row) => (
                            <tr key={row.id} className="hover:bg-slate-50/80">
                              <td className="px-3 py-2.5 font-medium text-text max-w-[220px]">
                                <span className="line-clamp-1">{row.text}</span>
                              </td>
                              <td className="px-3 py-2.5 tabular-nums text-slate-600">
                                {Math.round(row.font_size)}px
                              </td>
                              <td className="px-3 py-2.5">
                                <span className="inline-flex items-center gap-1.5">
                                  <span
                                    className="h-3 w-3 rounded-sm border border-border"
                                    style={{ backgroundColor: row.font_color }}
                                  />
                                  <span className="text-xs text-slate-600">{row.font_color}</span>
                                </span>
                              </td>
                              <td className="px-3 py-2.5 text-slate-600">{row.hierarchy}</td>
                              <td className="px-3 py-2.5 capitalize text-slate-600">
                                {row.alignment}
                              </td>
                              <td className="px-3 py-2.5 tabular-nums text-slate-600">
                                {(row.bold * 100).toFixed(0)}%
                              </td>
                              <td className="px-3 py-2.5 tabular-nums text-slate-600">
                                {Math.round(row.confidence)}
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

export default TypographyPanel
