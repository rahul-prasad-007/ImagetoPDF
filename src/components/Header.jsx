import { motion } from 'framer-motion'
import { FileImage } from 'lucide-react'

/**
 * App header with brand mark, title, and supporting subtitle.
 */
function Header() {
  return (
    <motion.header
      initial={{ opacity: 0, y: -16 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, ease: [0.22, 1, 0.36, 1] }}
      className="text-center px-4 pt-10 sm:pt-14 pb-8 sm:pb-10"
    >
      <motion.div
        initial={{ scale: 0.85, opacity: 0 }}
        animate={{ scale: 1, opacity: 1 }}
        transition={{ delay: 0.1, duration: 0.45, ease: [0.22, 1, 0.36, 1] }}
        className="mx-auto mb-5 flex h-16 w-16 sm:h-20 sm:w-20 items-center justify-center rounded-2xl bg-primary shadow-lg shadow-primary/25"
      >
        <FileImage className="h-8 w-8 sm:h-10 sm:w-10 text-white" strokeWidth={1.75} />
      </motion.div>

      <h1 className="font-display text-3xl sm:text-4xl md:text-5xl font-bold tracking-tight text-text">
        Image to Editable PDF
      </h1>

      <p className="mt-3 sm:mt-4 mx-auto max-w-xl text-sm sm:text-base text-slate-500 leading-relaxed">
        Upload any poster, flyer, certificate, banner or image and generate an editable PDF.
      </p>
    </motion.header>
  )
}

export default Header
