import { motion } from 'framer-motion'

function Footer() {
  return (
    <motion.footer
      initial={{ opacity: 0 }}
      whileInView={{ opacity: 1 }}
      viewport={{ once: true }}
      className="container-app mt-12 sm:mt-16 pb-10 text-center"
    >
      <p className="text-sm text-muted">
        <span className="font-brand text-white/90">Shankar Card</span>
        <span className="mx-2 text-border">·</span>
        Image to Editable PDF
      </p>
    </motion.footer>
  )
}

export default Footer
