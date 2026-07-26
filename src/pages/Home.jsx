import { AnimatePresence, motion } from 'framer-motion'
import { ScanText, LayoutTemplate, FileOutput, Loader2, Sparkles } from 'lucide-react'
import { useDropzone } from 'react-dropzone'
import toast from 'react-hot-toast'

import Header from '../components/Header.jsx'
import UploadCard from '../components/UploadCard.jsx'
import ImagePreview from '../components/ImagePreview.jsx'
import ProgressTracker from '../components/ProgressTracker.jsx'
import ResultCard from '../components/ResultCard.jsx'
import OcrResultsPanel from '../components/OcrResultsPanel.jsx'
import LayoutAnalysisPanel from '../components/LayoutAnalysisPanel.jsx'
import TypographyPanel from '../components/TypographyPanel.jsx'
import ReconstructionPanel from '../components/ReconstructionPanel.jsx'
import SceneGraphPanel from '../components/SceneGraphPanel.jsx'
import VectorAnalysisPanel from '../components/VectorAnalysisPanel.jsx'
import FeatureCard from '../components/FeatureCard.jsx'
import Footer from '../components/Footer.jsx'
import useImageUpload from '../hooks/useImageUpload.js'

const FEATURES = [
  {
    icon: ScanText,
    title: 'OCR Text Detection',
    description:
      'Detect and extract text from posters, flyers, and documents with high accuracy — ready for editing.',
  },
  {
    icon: LayoutTemplate,
    title: 'Layout Reconstruction',
    description:
      'Rebuild visual structure, spacing, and hierarchy so your PDF mirrors the original design.',
  },
  {
    icon: FileOutput,
    title: 'Editable PDF Export',
    description:
      'Export a fully editable PDF you can refine, restyle, and share — without starting from scratch.',
  },
]

/**
 * Main single-page experience: upload → preprocess → OCR → results.
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

  // Pipeline: 1–8 analysis/render, 9 quality optimization
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
    <main className="mx-auto w-full max-w-[1200px] px-4 sm:px-6 lg:px-8">
      <Header />

      <div className="flex flex-col gap-6 sm:gap-8">
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

        <motion.div
          initial={{ opacity: 0, y: 16 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2, duration: 0.45 }}
        >
          <motion.button
            type="button"
            disabled={!hasImage || busy}
            onClick={generatePdf}
            whileHover={hasImage && !busy ? { scale: 1.01 } : undefined}
            whileTap={hasImage && !busy ? { scale: 0.985 } : undefined}
            className={`
              w-full rounded-2xl px-6 py-4
              font-display text-base sm:text-lg font-semibold
              flex items-center justify-center gap-2.5
              transition-all duration-300
              ${
                hasImage && !busy
                  ? 'bg-primary text-white shadow-lg shadow-primary/30 hover:bg-secondary cursor-pointer'
                  : 'bg-slate-200 text-slate-400 cursor-not-allowed shadow-none'
              }
            `}
          >
            {busy ? (
              <>
                <Loader2 className="h-5 w-5 animate-spin" />
                {isOptimizing
                  ? 'Optimizing quality…'
                  : isRendering
                  ? 'Rendering…'
                  : isVectorRunning
                  ? 'Reconstructing vectors…'
                  : isSceneRunning
                  ? 'Building scene graph…'
                  : isReconstructionRunning
                  ? 'Planning reconstruction…'
                  : isTypographyRunning
                    ? 'Analyzing typography…'
                    : isLayoutRunning
                      ? 'Analyzing layout…'
                      : isOcrRunning
                        ? 'Running OCR…'
                        : 'Uploading…'}
              </>
            ) : (
              <>
                <Sparkles className="h-5 w-5" />
                Generate Editable PDF
              </>
            )}
          </motion.button>
        </motion.div>

        <ProgressTracker completedSteps={completedSteps} activeStep={activeStep} />

        <ResultCard
          uploadResult={uploadResult}
          pdfResult={pdfResult}
          optimizeResult={optimizeResult}
          isRendering={isRendering}
          isOptimizing={isOptimizing}
        />

        <OcrResultsPanel ocrResult={ocrResult} isLoading={isOcrRunning} />

        <LayoutAnalysisPanel layoutResult={layoutResult} isLoading={isLayoutRunning} />

        <TypographyPanel
          typographyResult={typographyResult}
          isLoading={isTypographyRunning}
        />

        <ReconstructionPanel
          reconstructionResult={reconstructionResult}
          isLoading={isReconstructionRunning}
        />

        <SceneGraphPanel sceneResult={sceneResult} isLoading={isSceneRunning} />

        <VectorAnalysisPanel vectorResult={vectorResult} isLoading={isVectorRunning} />

        <section className="pt-4 sm:pt-6">
          <motion.h2
            initial={{ opacity: 0, y: 12 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="font-display text-xl sm:text-2xl font-bold text-text text-center mb-6 sm:mb-8"
          >
            Built for precision editing
          </motion.h2>

          <div className="grid gap-4 sm:gap-5 sm:grid-cols-2 lg:grid-cols-3">
            {FEATURES.map((feature, index) => (
              <FeatureCard
                key={feature.title}
                icon={feature.icon}
                title={feature.title}
                description={feature.description}
                delay={index * 0.08}
              />
            ))}
          </div>
        </section>
      </div>

      <Footer />
    </main>
  )
}

export default Home
