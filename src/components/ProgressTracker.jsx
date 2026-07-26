import { motion } from 'framer-motion'
import { Check, Loader2 } from 'lucide-react'

/** User-facing stages mapped from internal pipeline step ids (1–9). */
const STAGES = [
  { id: 1, label: 'Upload' },
  { id: 2, label: 'Analyze' },
  { id: 3, label: 'Create PDF' },
  { id: 4, label: 'Finish' },
]

function stageStatus(stageId, completedSteps, activeStep) {
  const stage = STAGES.find((s) => s.id === stageId)
  const active =
    (stageId === 1 && activeStep === 1) ||
    (stageId === 2 && activeStep >= 2 && activeStep <= 7) ||
    (stageId === 3 && activeStep === 8) ||
    (stageId === 4 && activeStep === 9)
  if (active) return 'active'

  const done =
    (stageId === 1 && completedSteps >= 1 && activeStep !== 1) ||
    (stageId === 2 && completedSteps >= 7) ||
    (stageId === 3 && completedSteps >= 8) ||
    (stageId === 4 && completedSteps >= 9)
  if (done) return 'complete'
  return 'upcoming'
}

/**
 * Compact progress for end users (internal 9-step pipeline unchanged).
 */
function ProgressTracker({ completedSteps = 0, activeStep = 0 }) {
  if (completedSteps === 0 && activeStep === 0) return null

  return (
    <motion.section
      initial={{ opacity: 0, y: 16 }}
      animate={{ opacity: 1, y: 0 }}
      className="rounded-2xl border border-border bg-panel px-4 py-5 sm:px-6"
    >
      <ol className="flex items-center justify-between gap-2">
        {STAGES.map((stage, index) => {
          const status = stageStatus(stage.id, completedSteps, activeStep)
          return (
            <li key={stage.id} className="flex flex-1 flex-col items-center gap-2">
              <div className="flex w-full items-center">
                {index > 0 && (
                  <div
                    className={`h-0.5 flex-1 ${
                      completedSteps >= (stage.id === 2 ? 1 : stage.id === 3 ? 7 : 8)
                        ? 'bg-accent'
                        : 'bg-border'
                    }`}
                    aria-hidden
                  />
                )}
                <div
                  className={`
                    relative z-10 flex h-9 w-9 shrink-0 items-center justify-center rounded-full
                    border-2 text-xs font-bold
                    ${
                      status === 'complete'
                        ? 'border-success bg-success text-ink'
                        : status === 'active'
                          ? 'border-accent bg-accent text-white shadow-md shadow-accent/30'
                          : 'border-border bg-panel-2 text-muted'
                    }
                  `}
                >
                  {status === 'complete' ? (
                    <Check className="h-4 w-4" strokeWidth={2.5} />
                  ) : status === 'active' ? (
                    <Loader2 className="h-4 w-4 animate-spin" />
                  ) : (
                    stage.id
                  )}
                </div>
                {index < STAGES.length - 1 && (
                  <div
                    className={`h-0.5 flex-1 ${
                      status === 'complete' ? 'bg-accent' : 'bg-border'
                    }`}
                    aria-hidden
                  />
                )}
              </div>
              <span
                className={`text-[11px] font-semibold sm:text-xs ${
                  status === 'upcoming' ? 'text-muted' : 'text-white'
                }`}
              >
                {stage.label}
              </span>
            </li>
          )
        })}
      </ol>
    </motion.section>
  )
}

export default ProgressTracker
