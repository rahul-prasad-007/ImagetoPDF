import { motion } from 'framer-motion'
import { ImageIcon, RefreshCw, Trash2 } from 'lucide-react'
import { formatFileSize, getFileTypeLabel } from '../hooks/useImageUpload'

/**
 * Preview panel shown after a successful image upload.
 */
function ImagePreview({ file, previewUrl, dimensions, onChange, onRemove }) {
  if (!file || !previewUrl) return null

  const resolution =
    dimensions != null
      ? `${dimensions.width} × ${dimensions.height} px`
      : 'Reading…'

  const meta = [
    { label: 'Name', value: file.name },
    { label: 'Resolution', value: resolution },
    { label: 'Size', value: formatFileSize(file.size) },
    { label: 'Type', value: getFileTypeLabel(file.type) },
  ]

  return (
    <motion.section
      initial={{ opacity: 0, scale: 0.96, y: 16 }}
      animate={{ opacity: 1, scale: 1, y: 0 }}
      exit={{ opacity: 0, scale: 0.96, y: 8 }}
      transition={{ duration: 0.4, ease: [0.22, 1, 0.36, 1] }}
      className="rounded-2xl border border-border bg-card p-5 sm:p-6 shadow-[0_4px_24px_rgba(15,23,42,0.06)]"
    >
      <div className="flex items-center gap-2 mb-4">
        <ImageIcon className="h-5 w-5 text-primary" />
        <h2 className="font-display text-lg font-semibold text-text">Image Preview</h2>
      </div>

      <div className="grid gap-5 md:grid-cols-[1.1fr_0.9fr] items-start">
        <div className="overflow-hidden rounded-xl border border-border bg-slate-50">
          <img
            src={previewUrl}
            alt={file.name}
            className="mx-auto max-h-72 w-full object-contain p-3"
          />
        </div>

        <div className="flex flex-col gap-4">
          <dl className="space-y-3">
            {meta.map((item) => (
              <div key={item.label} className="flex flex-col gap-0.5">
                <dt className="text-xs font-medium uppercase tracking-wide text-slate-400">
                  {item.label}
                </dt>
                <dd className="text-sm font-medium text-text break-all">{item.value}</dd>
              </div>
            ))}
          </dl>

          <div className="mt-auto flex flex-col sm:flex-row gap-2.5 pt-2">
            <motion.button
              type="button"
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.97 }}
              onClick={onChange}
              className="inline-flex flex-1 items-center justify-center gap-2 rounded-xl border border-border bg-white px-4 py-2.5 text-sm font-semibold text-text transition-colors hover:bg-slate-50"
            >
              <RefreshCw className="h-4 w-4 text-primary" />
              Change Image
            </motion.button>

            <motion.button
              type="button"
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.97 }}
              onClick={onRemove}
              className="inline-flex flex-1 items-center justify-center gap-2 rounded-xl border border-error/20 bg-error/5 px-4 py-2.5 text-sm font-semibold text-error transition-colors hover:bg-error/10"
            >
              <Trash2 className="h-4 w-4" />
              Remove Image
            </motion.button>
          </div>
        </div>
      </div>
    </motion.section>
  )
}

export default ImagePreview
