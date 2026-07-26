import { Lock, Sparkles } from 'lucide-react'
import { motion } from 'framer-motion'
import { FileOutput, ScanText, LayoutTemplate } from 'lucide-react'

const FEATURES = [
  { icon: ScanText, label: 'Smart OCR' },
  { icon: LayoutTemplate, label: 'Layout Rebuild' },
  { icon: FileOutput, label: 'Editable PDF' },
]

/**
 * Shankar Card brand header + hero (matches Background Remover product family).
 */
function Header() {
  return (
    <>
      <header className="container-app flex items-center justify-between gap-4 pt-6 pb-2 sm:pt-8">
        <div className="flex items-center gap-2.5">
          <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-accent/15 ring-1 ring-accent/30">
            <Sparkles className="h-4 w-4 text-accent" strokeWidth={2.25} />
          </div>
          <p className="text-[15px] font-bold tracking-tight text-white sm:text-base">
            Shankar <span className="text-accent">Card</span>
          </p>
        </div>

        <div className="inline-flex items-center gap-1.5 rounded-full border border-success/25 bg-success-soft px-3 py-1.5 text-[11px] font-semibold text-success sm:text-xs">
          <Lock className="h-3 w-3" />
          <span>100% Free · No Sign Up</span>
        </div>
      </header>

      <section className="hero-stage relative overflow-hidden">
        <div className="hero-glow" aria-hidden />
        <div className="hero-ring" aria-hidden />

        <div className="container-app relative z-10 pt-10 pb-8 text-center sm:pt-14 sm:pb-10">
          <motion.p
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.4 }}
            className="mb-5 inline-flex items-center gap-2 text-[11px] font-semibold tracking-[0.22em] text-accent uppercase sm:text-xs"
          >
            <span className="h-px w-6 bg-accent/60" />
            Print Ready Studio
            <span className="h-px w-6 bg-accent/60" />
          </motion.p>

          <motion.h1
            initial={{ opacity: 0, y: 18 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.55, ease: [0.22, 1, 0.36, 1] }}
            className="mx-auto max-w-4xl"
          >
            <span className="font-brand hero-brand block text-[2.75rem] leading-[1.05] text-white sm:text-6xl md:text-7xl">
              Shankar Card
            </span>

            <motion.span
              initial={{ scaleX: 0 }}
              animate={{ scaleX: 1 }}
              transition={{ duration: 0.55, delay: 0.28, ease: [0.22, 1, 0.36, 1] }}
              className="hero-underline mx-auto mt-4 block origin-center"
              aria-hidden
            />

            <span className="heading-accent mt-5 block text-xl font-semibold tracking-[0.04em] sm:text-3xl">
              Image to Editable PDF
            </span>
          </motion.h1>

          <motion.p
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.45, delay: 0.18 }}
            className="mx-auto mt-5 max-w-xl text-[15px] leading-relaxed text-slate-300 sm:text-lg"
          >
            Upload a poster, flyer, or document — get an editable PDF you can fix and print.
          </motion.p>

          <motion.ul
            initial="hidden"
            animate="show"
            variants={{
              hidden: {},
              show: { transition: { staggerChildren: 0.07, delayChildren: 0.28 } },
            }}
            className="mx-auto mt-8 flex max-w-3xl flex-wrap items-center justify-center gap-x-1 gap-y-3 sm:gap-x-0"
          >
            {FEATURES.map(({ icon: Icon, label }, i) => (
              <motion.li
                key={label}
                variants={{
                  hidden: { opacity: 0, y: 10 },
                  show: { opacity: 1, y: 0 },
                }}
                className="flex items-center"
              >
                {i > 0 && (
                  <span
                    className="mx-2 hidden h-4 w-px bg-white/15 sm:mx-4 sm:block"
                    aria-hidden
                  />
                )}
                <span className="inline-flex items-center gap-2 text-sm font-medium text-slate-200">
                  <span className="flex h-8 w-8 items-center justify-center rounded-lg bg-accent/15 text-accent ring-1 ring-accent/25">
                    <Icon className="h-3.5 w-3.5" strokeWidth={2.25} />
                  </span>
                  {label}
                </span>
              </motion.li>
            ))}
          </motion.ul>
        </div>
      </section>
    </>
  )
}

export default Header
