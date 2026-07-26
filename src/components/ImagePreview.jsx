import { motion } from 'framer-motion'
import { RefreshCw, Trash2 } from 'lucide-react'
import { formatFileSize } from '../hooks/useImageUpload'

function ImagePreview({ file, previewUrl, dimensions, onChange, onRemove }) {
  if (!file || !previewUrl) return null

  const resolution =
    dimensions != null ? `${dimensions.width} × ${dimensions.height}` : '…'

  return (
    <motion.section
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: 8 }}
      className="rounded-xl border border-border bg-panel p-4 shadow-sm shadow-slate-900/5"
    >
      <div className="grid gap-4 sm:grid-cols-[1.2fr_0.8fr] items-center">
        <div className="overflow-hidden rounded-lg border border-border bg-surface">
          <img
            src={previewUrl}
            alt={file.name}
            className="mx-auto max-h-52 w-full object-contain p-3"
          />
        </div>

        <div className="flex flex-col gap-3 text-left">
          <div>
            <p className="text-sm font-semibold text-ink truncate" title={file.name}>
              {file.name}
            </p>
            <p className="mt-1 text-xs text-muted">
              {resolution} px · {formatFileSize(file.size)}
            </p>
          </div>

          <div className="flex gap-2">
            <button
              type="button"
              onClick={onChange}
              className="inline-flex flex-1 items-center justify-center gap-2 rounded-lg border border-border bg-panel-2 px-3 py-2.5 text-sm font-semibold text-ink hover:bg-surface transition-colors"
            >
              <RefreshCw className="h-3.5 w-3.5 text-accent" />
              Change
            </button>
            <button
              type="button"
              onClick={onRemove}
              className="inline-flex items-center justify-center gap-2 rounded-lg border border-error/25 bg-error/5 px-3 py-2.5 text-sm font-semibold text-error hover:bg-error/10 transition-colors"
            >
              <Trash2 className="h-3.5 w-3.5" />
            </button>
          </div>
        </div>
      </div>
    </motion.section>
  )
}

export default ImagePreview
