import { useCallback } from 'react'
import { useDropzone } from 'react-dropzone'
import { motion } from 'framer-motion'
import { Upload, FolderOpen } from 'lucide-react'
import toast from 'react-hot-toast'

/**
 * Drag-and-drop upload — Shankar Card dark panel style.
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
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: 0.12, duration: 0.45 }}
      className="rounded-2xl border border-border bg-panel p-4 sm:p-6"
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
        aria-label="Upload image"
        className={`
          relative cursor-pointer rounded-xl border-2 border-dashed
          transition-all duration-300 px-4 py-12 sm:py-16 text-center
          ${
            isDragActive
              ? 'border-accent bg-accent/10 scale-[1.01]'
              : 'border-border hover:border-accent/50 hover:bg-panel-2/80'
          }
          ${disabled ? 'pointer-events-none opacity-50' : ''}
        `}
      >
        <input {...getInputProps()} />

        <motion.div
          animate={isDragActive ? { scale: 1.08, y: -4 } : { scale: 1, y: 0 }}
          className="mx-auto mb-5 flex h-16 w-16 items-center justify-center rounded-2xl bg-accent/15 text-accent ring-1 ring-accent/30"
        >
          <Upload className="h-7 w-7" strokeWidth={1.75} />
        </motion.div>

        <p className="font-display text-lg sm:text-xl font-semibold text-white">
          {isDragActive ? 'Drop your image here' : 'Drag & drop your image'}
        </p>
        <p className="mt-1.5 text-sm text-muted">or browse from your device</p>

        <span className="mt-6 inline-flex items-center gap-2 rounded-xl bg-accent px-5 py-2.5 text-sm font-semibold text-white shadow-md shadow-accent/25 pointer-events-none select-none">
          <FolderOpen className="h-4 w-4" />
          Browse Files
        </span>

        <p className="mt-6 text-xs text-muted">
          PNG, JPEG, WEBP, BMP, TIFF · Max 20 MB
        </p>
      </div>
    </motion.section>
  )
}

export default UploadCard
