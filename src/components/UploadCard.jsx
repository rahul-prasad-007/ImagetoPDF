import { useCallback } from 'react'
import { useDropzone } from 'react-dropzone'
import { motion } from 'framer-motion'
import { Upload, FolderOpen } from 'lucide-react'
import toast from 'react-hot-toast'

const FORMAT_LABELS = ['PNG', 'JPEG', 'WEBP', 'BMP', 'TIFF']

/**
 * Large drag-and-drop upload card with browse support and format hints.
 */
function UploadCard({ acceptedTypes, onSelect, disabled = false }) {
  const onDrop = useCallback(
    (acceptedFiles) => {
      if (acceptedFiles?.[0]) onSelect(acceptedFiles[0])
    },
    [onSelect],
  )

  const onDropRejected = useCallback((fileRejections) => {
    const error = fileRejections[0]?.errors?.[0]
    if (!error) return

    if (error.code === 'file-too-large') {
      toast.error('File is too large. Maximum size is 20 MB.')
      return
    }

    if (error.code === 'file-invalid-type') {
      toast.error('Unsupported format. Use PNG, JPEG, WEBP, BMP, or TIFF.')
      return
    }

    toast.error(error.message || 'Unable to upload this file.')
  }, [])

  const { getRootProps, getInputProps, isDragActive, open } = useDropzone({
    onDrop,
    onDropRejected,
    accept: acceptedTypes,
    multiple: false,
    noClick: true,
    disabled,
    maxSize: 20 * 1024 * 1024,
  })

  return (
    <motion.section
      initial={{ opacity: 0, y: 24 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: 0.15, duration: 0.5, ease: [0.22, 1, 0.36, 1] }}
      className="rounded-2xl border border-border bg-card p-5 sm:p-8 shadow-[0_4px_24px_rgba(15,23,42,0.06)]"
    >
      <div
        {...getRootProps()}
        onClick={open}
        role="button"
        tabIndex={0}
        onKeyDown={(e) => {
          if (e.key === 'Enter' || e.key === ' ') {
            e.preventDefault()
            open()
          }
        }}
        aria-label="Upload image by drag and drop or browse"
        className={`
          relative group cursor-pointer rounded-xl border-2 border-dashed
          transition-all duration-300 ease-out
          px-4 py-10 sm:py-14 text-center
          ${
            isDragActive
              ? 'border-primary bg-primary/5 scale-[1.01]'
              : 'border-border hover:border-secondary hover:bg-slate-50/80'
          }
          ${disabled ? 'pointer-events-none opacity-60' : ''}
        `}
      >
        <input {...getInputProps()} />

        <div
          className={`
            pointer-events-none absolute inset-0 rounded-xl
            ring-0 ring-primary/0 transition-all duration-300
            group-hover:ring-4 group-hover:ring-primary/10
            ${isDragActive ? 'ring-4 ring-primary/20' : ''}
          `}
        />

        <motion.div
          animate={isDragActive ? { scale: 1.08, y: -4 } : { scale: 1, y: 0 }}
          transition={{ type: 'spring', stiffness: 320, damping: 22 }}
          className="mx-auto mb-5 flex h-16 w-16 items-center justify-center rounded-2xl bg-primary/10 text-primary"
        >
          <Upload className="h-7 w-7" strokeWidth={1.75} />
        </motion.div>

        <p className="font-display text-lg sm:text-xl font-semibold text-text">
          {isDragActive ? 'Drop your image here' : 'Drag & drop your image here'}
        </p>
        <p className="mt-1.5 text-sm text-slate-500">or click anywhere to browse</p>

        {/* Visual CTA only — whole zone is clickable; avoid nested <button> */}
        <span className="mt-6 inline-flex items-center gap-2 rounded-xl bg-primary px-5 py-2.5 text-sm font-semibold text-white shadow-md shadow-primary/25 pointer-events-none select-none">
          <FolderOpen className="h-4 w-4" />
          Browse Files
        </span>

        <div className="mt-8 flex flex-col sm:flex-row items-center justify-center gap-3 sm:gap-6 text-xs sm:text-sm text-slate-500">
          <div className="flex flex-wrap items-center justify-center gap-1.5">
            <span className="font-medium text-slate-600">Supported:</span>
            {FORMAT_LABELS.map((label) => (
              <span
                key={label}
                className="rounded-md border border-border bg-white px-2 py-0.5 font-medium text-slate-600"
              >
                {label}
              </span>
            ))}
          </div>
          <span className="hidden sm:inline text-border">|</span>
          <p>
            Max size: <span className="font-semibold text-slate-700">20 MB</span>
          </p>
        </div>
      </div>
    </motion.section>
  )
}

export default UploadCard
