import { motion } from 'framer-motion'
import { Check, Loader2 } from 'lucide-react'

const STAGES = [
  { id: 1, label: 'Upload' },
  { id: 2, label: 'Analyze' },
  { id: 3, label: 'Create PDF' },
  { id: 4, label: 'Finish' },
]

function stageStatus(stageId, completedSteps, activeStep) {
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

function ProgressTracker({ completedSteps = 0, activeStep = 0 }) {
  if (completedSteps === 0 && activeStep === 0) return null

  return (
    <motion.section
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      className="rounded-xl border border-border bg-panel px-3 py-4 sm:px-5 shadow-sm shadow-slate-900/5"
    >
      <p className="mb-3 text-left text-[11px] font-bold uppercase tracking-[0.16em] text-muted">
        Progress
      </p>
      <ol className="flex items-center justify-between gap-1">
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
                    relative z-10 flex h-8 w-8 shrink-0 items-center justify-center rounded-md
                    border text-xs font-bold
                    ${
                      status === 'complete'
                        ? 'border-success bg-success text-white'
                        : status === 'active'
                          ? 'border-accent bg-accent text-white'
                          : 'border-border bg-panel-2 text-muted'
                    }
                  `}
                >
                  {status === 'complete' ? (
                    <Check className="h-3.5 w-3.5" strokeWidth={2.5} />
                  ) : status === 'active' ? (
                    <Loader2 className="h-3.5 w-3.5 animate-spin" />
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
                className={`text-[10px] font-semibold sm:text-[11px] ${
                  status === 'upcoming' ? 'text-muted' : 'text-ink'
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
