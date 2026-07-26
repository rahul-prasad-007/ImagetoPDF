import { useCallback } from 'react'
import { useDropzone } from 'react-dropzone'
import { motion } from 'framer-motion'
import { Upload, FolderOpen } from 'lucide-react'
import toast from 'react-hot-toast'

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
      initial={{ opacity: 0, y: 16 }}
      animate={{ opacity: 1, y: 0 }}
      className="rounded-xl border border-border bg-panel p-4 sm:p-5 shadow-sm shadow-slate-900/5"
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
          relative cursor-pointer rounded-lg border-2 border-dashed
          transition-all duration-300 px-4 py-10 sm:py-12 text-left sm:text-center
          ${
            isDragActive
              ? 'border-accent bg-accent-soft scale-[1.01]'
              : 'border-border hover:border-accent/40 hover:bg-panel-2'
          }
          ${disabled ? 'pointer-events-none opacity-50' : ''}
        `}
      >
        <input {...getInputProps()} />

        <div className="flex flex-col sm:items-center gap-4 sm:gap-5">
          <div className="flex h-14 w-14 items-center justify-center rounded-lg bg-accent-soft text-accent">
            <Upload className="h-6 w-6" strokeWidth={1.85} />
          </div>

          <div>
            <p className="font-display text-lg font-semibold text-ink sm:text-xl">
              {isDragActive ? 'Drop your image here' : 'Drop an image to convert'}
            </p>
            <p className="mt-1 text-sm text-muted">PNG, JPEG, WEBP, BMP, TIFF · Max 20 MB</p>
          </div>

          <span className="inline-flex w-fit items-center gap-2 rounded-lg bg-accent px-5 py-2.5 text-sm font-semibold text-white shadow-sm shadow-accent/25 pointer-events-none select-none">
            <FolderOpen className="h-4 w-4" />
            Browse files
          </span>
        </div>
      </div>
    </motion.section>
  )
}

export default UploadCard
