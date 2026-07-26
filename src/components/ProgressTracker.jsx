import { motion } from 'framer-motion'
import { Check, Loader2 } from 'lucide-react'

const STEPS = [
  { id: 1, label: 'Upload Image' },
  { id: 2, label: 'OCR Analysis' },
  { id: 3, label: 'Layout Detection' },
  { id: 4, label: 'Typography Analysis' },
  { id: 5, label: 'Reconstruction Plan' },
  { id: 6, label: 'Scene Graph' },
  { id: 7, label: 'Vector Analysis' },
  { id: 8, label: 'PDF Generation' },
  { id: 9, label: 'Quality Optimization' },
]

/**
 * Visual progress tracker for the full editable-PDF pipeline.
 *
 * @param {number} completedSteps - how many steps are finished (0–9)
 * @param {number} activeStep - currently running step id, or 0 if idle
 */
function ProgressTracker({ completedSteps = 0, activeStep = 0 }) {
  const getStatus = (stepId) => {
    if (activeStep === stepId) return 'active'
    if (stepId <= completedSteps) return 'complete'
    return 'upcoming'
  }

  return (
    <motion.section
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: 0.25, duration: 0.5, ease: [0.22, 1, 0.36, 1] }}
      className="rounded-2xl border border-border bg-card p-5 sm:p-7 shadow-[0_4px_24px_rgba(15,23,42,0.06)]"
    >
      <h2 className="font-display text-lg font-semibold text-text mb-6 text-center sm:text-left">
        Processing Pipeline
      </h2>

      <ol className="flex flex-col sm:flex-row sm:justify-between gap-0">
        {STEPS.map((step, index) => {
          const status = getStatus(step.id)
          const isLast = index === STEPS.length - 1
          const lineDone =
            completedSteps >= step.id - 1 || (activeStep > 0 && activeStep >= step.id)

          return (
            <li
              key={step.id}
              className="relative flex sm:flex-1 sm:flex-col items-start sm:items-center gap-3 sm:gap-0 sm:text-center"
            >
              {index > 0 && (
                <div
                  className={`hidden sm:block absolute top-5 right-1/2 h-0.5 w-full -translate-y-1/2 ${
                    lineDone ? 'bg-primary' : 'bg-border'
                  }`}
                  aria-hidden
                />
              )}

              <div className="flex flex-col items-center">
                <div
                  className={`
                    relative z-10 flex h-9 w-9 sm:h-10 sm:w-10 shrink-0 items-center justify-center rounded-full
                    border-2 text-sm font-bold transition-all duration-300
                    ${
                      status === 'complete'
                        ? 'border-accent bg-accent text-white'
                        : status === 'active'
                          ? 'border-primary bg-primary text-white shadow-md shadow-primary/30'
                          : 'border-border bg-white text-slate-400'
                    }
                  `}
                >
                  {status === 'complete' ? (
                    <Check className="h-3.5 w-3.5 sm:h-4 sm:w-4" strokeWidth={2.5} />
                  ) : status === 'active' ? (
                    <Loader2 className="h-3.5 w-3.5 sm:h-4 sm:w-4 animate-spin" />
                  ) : (
                    step.id
                  )}
                </div>

                {!isLast && (
                  <div
                    className={`sm:hidden w-0.5 flex-1 min-h-6 ${
                      completedSteps >= step.id ? 'bg-primary' : 'bg-border'
                    }`}
                    aria-hidden
                  />
                )}
              </div>

              <span
                className={`pt-1.5 sm:pt-3 sm:mt-0 pb-5 sm:pb-0 max-w-none sm:max-w-[6.5rem] text-sm sm:text-[11px] font-medium leading-snug ${
                  status === 'upcoming' ? 'text-slate-400' : 'text-text'
                }`}
              >
                {step.label}
              </span>
            </li>
          )
        })}
      </ol>

      <p className="mt-5 sm:mt-6 text-center text-xs text-slate-400">
        Upload → analysis → scene → vectors → editable PDF.
      </p>
    </motion.section>
  )
}

export default ProgressTracker
