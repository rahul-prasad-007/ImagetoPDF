import { motion } from 'framer-motion'

/**
 * Simple site footer.
 */
function Footer() {
  return (
    <motion.footer
      initial={{ opacity: 0 }}
      whileInView={{ opacity: 1 }}
      viewport={{ once: true }}
      transition={{ duration: 0.5 }}
      className="mt-12 sm:mt-16 pb-10 text-center"
    >
      <p className="text-sm text-slate-400">
        Made with <span className="text-error">❤️</span> using AI
      </p>
    </motion.footer>
  )
}

export default Footer
