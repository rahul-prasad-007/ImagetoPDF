import { AnimatePresence, motion } from 'framer-motion'
import { Loader2, Sparkles } from 'lucide-react'
import { useDropzone } from 'react-dropzone'
import toast from 'react-hot-toast'

import Header from '../components/Header.jsx'
import UploadCard from '../components/UploadCard.jsx'
import ImagePreview from '../components/ImagePreview.jsx'
import ProgressTracker from '../components/ProgressTracker.jsx'
import ResultCard from '../components/ResultCard.jsx'
import Footer from '../components/Footer.jsx'
import useImageUpload from '../hooks/useImageUpload.js'

/**
 * Main experience — same pipeline, cleaner Shankar Card UI.
 */
function Home() {
  const {
    file,
    previewUrl,
    dimensions,
    isGenerating,
    isOcrRunning,
    isLayoutRunning,
    isTypographyRunning,
    isReconstructionRunning,
    isSceneRunning,
    isVectorRunning,
    isRendering,
    isOptimizing,
    uploadResult,
    ocrResult,
    layoutResult,
    typographyResult,
    reconstructionResult,
    sceneResult,
    vectorResult,
    pdfResult,
    optimizeResult,
    hasImage,
    acceptedTypes,
    selectImage,
    removeImage,
    generatePdf,
  } = useImageUpload()

  const { getInputProps: getChangeInputProps, open: openChangeDialog } = useDropzone({
    accept: acceptedTypes,
    multiple: false,
    noClick: true,
    noKeyboard: true,
    noDrag: true,
    maxSize: 20 * 1024 * 1024,
    onDrop: (accepted) => {
      if (accepted?.[0]) selectImage(accepted[0])
    },
    onDropRejected: (rejections) => {
      const code = rejections[0]?.errors?.[0]?.code
      if (code === 'file-too-large') {
        toast.error('File is too large. Maximum size is 20 MB.')
      } else if (code === 'file-invalid-type') {
        toast.error('Unsupported format. Use PNG, JPEG, WEBP, BMP, or TIFF.')
      } else {
        toast.error('Unable to upload this file.')
      }
    },
  })

  // Keep full pipeline step mapping (functionality unchanged)
  const completedSteps = optimizeResult
    ? 9
    : pdfResult
      ? 8
      : vectorResult
        ? 7
        : sceneResult
          ? 6
          : reconstructionResult
            ? 5
            : typographyResult
              ? 4
              : layoutResult
                ? 3
                : ocrResult
                  ? 2
                  : uploadResult
                    ? 1
                    : hasImage
                      ? 1
                      : 0
  const activeStep = isOptimizing
    ? 9
    : isRendering
      ? 8
      : isVectorRunning
        ? 7
        : isSceneRunning
          ? 6
          : isReconstructionRunning
            ? 5
            : isTypographyRunning
              ? 4
              : isLayoutRunning
                ? 3
                : isOcrRunning
                  ? 2
                  : isGenerating && !uploadResult
                    ? 1
                    : 0
  const busy =
    isGenerating ||
    isOcrRunning ||
    isLayoutRunning ||
    isTypographyRunning ||
    isReconstructionRunning ||
    isSceneRunning ||
    isVectorRunning ||
    isRendering ||
    isOptimizing

  return (
    <main className="pb-4">
      <Header />

      <div className="container-app flex flex-col gap-5 sm:gap-6">
        {!hasImage && (
          <UploadCard
            acceptedTypes={acceptedTypes}
            onSelect={selectImage}
            disabled={busy}
          />
        )}

        <AnimatePresence mode="wait">
          {hasImage && (
            <ImagePreview
              key="preview"
              file={file}
              previewUrl={previewUrl}
              dimensions={dimensions}
              onChange={openChangeDialog}
              onRemove={removeImage}
            />
          )}
        </AnimatePresence>

        <input {...getChangeInputProps()} className="sr-only" aria-hidden tabIndex={-1} />

        <motion.button
          type="button"
          disabled={!hasImage || busy}
          onClick={generatePdf}
          whileHover={hasImage && !busy ? { scale: 1.01 } : undefined}
          whileTap={hasImage && !busy ? { scale: 0.985 } : undefined}
          className={`
            w-full rounded-xl px-6 py-4
            font-display font-semibold text-base sm:text-lg
            flex items-center justify-center gap-2.5
            transition-all duration-300
            ${
              hasImage && !busy
                ? 'bg-accent text-white shadow-md shadow-accent/25 hover:bg-secondary cursor-pointer'
                : 'bg-panel-2 text-muted cursor-not-allowed border border-border'
            }
          `}
        >
          {busy ? (
            <>
              <Loader2 className="h-5 w-5 animate-spin" />
              {isOptimizing
                ? 'Finishing…'
                : isRendering
                  ? 'Creating PDF…'
                  : isVectorRunning ||
                      isSceneRunning ||
                      isReconstructionRunning ||
                      isTypographyRunning ||
                      isLayoutRunning ||
                      isOcrRunning
                    ? 'Analyzing…'
                    : 'Uploading…'}
            </>
          ) : (
            <>
              <Sparkles className="h-5 w-5" />
              Generate Editable PDF
            </>
          )}
        </motion.button>

        <ProgressTracker completedSteps={completedSteps} activeStep={activeStep} />

        <ResultCard
          uploadResult={uploadResult}
          pdfResult={pdfResult}
          optimizeResult={optimizeResult}
          isRendering={isRendering}
          isOptimizing={isOptimizing}
        />
      </div>

      <Footer />
    </main>
  )
}

export default Home
