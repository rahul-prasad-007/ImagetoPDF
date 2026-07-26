import { motion } from 'framer-motion'

function Footer() {
  return (
    <motion.footer
      initial={{ opacity: 0 }}
      whileInView={{ opacity: 1 }}
      viewport={{ once: true }}
      className="container-app mt-10 sm:mt-14 pb-10 text-left border-t border-border pt-6"
    >
      <p className="text-sm text-muted">
        <span className="font-brand text-ink">Shankar Card</span>
        <span className="mx-2 text-border">|</span>
        Image to Editable PDF
      </p>
    </motion.footer>
  )
}

export default Footer
