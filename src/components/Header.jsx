import { FileText, Lock } from 'lucide-react'
import { motion } from 'framer-motion'
import { FileOutput, ScanText, LayoutTemplate } from 'lucide-react'

const FEATURES = [
  { icon: ScanText, label: 'Smart OCR' },
  { icon: LayoutTemplate, label: 'Layout Rebuild' },
  { icon: FileOutput, label: 'Editable PDF' },
]

/**
 * Light, left-aligned Shankar Card header — distinct from dark centered Enhancer.
 */
function Header() {
  return (
    <>
      <header className="container-app flex items-center justify-between gap-4 pt-6 pb-1 sm:pt-8">
        <div className="flex items-center gap-3">
          <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-accent text-white shadow-sm shadow-accent/20">
            <FileText className="h-5 w-5" strokeWidth={2.1} />
          </div>
          <div className="text-left leading-tight">
            <p className="font-brand text-[15px] text-ink sm:text-base">Shankar Card</p>
            <p className="text-[11px] font-medium uppercase tracking-[0.14em] text-muted">
              PDF Studio
            </p>
          </div>
        </div>

        <div className="inline-flex items-center gap-1.5 rounded-md border border-border bg-panel px-3 py-1.5 text-[11px] font-semibold text-accent sm:text-xs">
          <Lock className="h-3 w-3" />
          <span>Free · No login</span>
        </div>
      </header>

      <section className="hero-stage relative">
        <div className="container-app relative z-10 pt-8 pb-6 text-left sm:pt-10 sm:pb-8">
          <motion.p
            initial={{ opacity: 0, x: -8 }}
            animate={{ opacity: 1, x: 0 }}
            className="mb-3 text-[11px] font-bold uppercase tracking-[0.2em] text-highlight sm:text-xs"
          >
            Document tools
          </motion.p>

          <motion.h1
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.45 }}
            className="max-w-2xl"
          >
            <span className="font-brand hero-brand block text-4xl leading-[1.1] sm:text-5xl md:text-[3.25rem]">
              Shankar Card
            </span>
            <span className="hero-underline mt-3 block" aria-hidden />
            <span className="heading-accent mt-4 block font-display text-lg font-semibold tracking-tight sm:text-2xl">
              Image → Editable PDF
            </span>
          </motion.h1>

          <motion.p
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.12 }}
            className="mt-4 max-w-lg text-[15px] leading-relaxed text-muted sm:text-base"
          >
            Upload a poster, flyer, or document — get an editable PDF you can fix and print.
          </motion.p>

          <ul className="mt-7 flex flex-col gap-2.5 sm:flex-row sm:flex-wrap sm:gap-4">
            {FEATURES.map(({ icon: Icon, label }) => (
              <li
                key={label}
                className="inline-flex items-center gap-2.5 rounded-lg border border-border bg-panel px-3.5 py-2 text-sm font-semibold text-ink shadow-sm shadow-slate-900/5"
              >
                <span className="flex h-7 w-7 items-center justify-center rounded-md bg-accent-soft text-accent">
                  <Icon className="h-3.5 w-3.5" strokeWidth={2.25} />
                </span>
                {label}
              </li>
            ))}
          </ul>
        </div>
      </section>
    </>
  )
}

export default Header
