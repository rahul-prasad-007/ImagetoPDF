import { useCallback, useEffect, useState } from 'react'
import toast from 'react-hot-toast'
import { uploadImage, runOcr, runLayout, runTypography, runReconstruction, runScene, runVector, runRender, runOptimize } from '../lib/api.js'

/** Allowed MIME types for image upload (dropzone accept map) */
const ACCEPTED_TYPES = {
  'image/png': ['.png'],
  'image/jpeg': ['.jpg', '.jpeg'],
  'image/webp': ['.webp'],
  'image/bmp': ['.bmp'],
  'image/x-ms-bmp': ['.bmp'],
  'image/tiff': ['.tif', '.tiff'],
}

const ACCEPTED_MIME_SET = new Set([
  'image/png',
  'image/jpeg',
  'image/webp',
  'image/bmp',
  'image/x-ms-bmp',
  'image/tiff',
])

const ACCEPTED_EXTENSIONS = new Set(['.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tif', '.tiff'])
const MAX_FILE_SIZE = 20 * 1024 * 1024 // 20 MB

/**
 * Format bytes into a human-readable string.
 * @param {number} bytes
 * @returns {string}
 */
export function formatFileSize(bytes) {
  if (bytes === 0) return '0 B'
  const units = ['B', 'KB', 'MB', 'GB']
  const i = Math.floor(Math.log(bytes) / Math.log(1024))
  const value = bytes / 1024 ** i
  return `${value.toFixed(i === 0 ? 0 : 1)} ${units[i]}`
}

/**
 * Map MIME type to a short display label.
 * @param {string} mime
 * @returns {string}
 */
export function getFileTypeLabel(mime) {
  const map = {
    'image/png': 'PNG',
    'image/jpeg': 'JPEG',
    'image/webp': 'WEBP',
    'image/bmp': 'BMP',
    'image/x-ms-bmp': 'BMP',
    'image/tiff': 'TIFF',
  }
  return map[mime] || mime.split('/')[1]?.toUpperCase() || 'Unknown'
}

/**
 * @param {File} file
 * @returns {boolean}
 */
function hasAcceptedExtension(file) {
  const name = file.name?.toLowerCase() ?? ''
  const dot = name.lastIndexOf('.')
  if (dot < 0) return false
  return ACCEPTED_EXTENSIONS.has(name.slice(dot))
}

/**
 * Extract a clean error message from an Axios / API error payload.
 * @param {unknown} err
 * @returns {string}
 */
function getApiErrorMessage(err) {
  const detail = err?.response?.data?.detail
  if (detail && typeof detail === 'object' && detail.error) {
    return detail.detail ? `${detail.error}: ${detail.detail}` : detail.error
  }
  if (typeof detail === 'string') return detail
  if (err?.response?.data?.error) return err.response.data.error
  if (err?.code === 'ERR_NETWORK') {
    return 'Cannot reach the server. Is the backend running?'
  }
  return err?.message || 'Request failed. Please try again.'
}

/**
 * Custom hook: local image selection → upload/preprocess → OCR.
 */
export default function useImageUpload() {
  const [file, setFile] = useState(null)
  const [previewUrl, setPreviewUrl] = useState(null)
  const [dimensions, setDimensions] = useState(null)
  const [isGenerating, setIsGenerating] = useState(false)
  const [isOcrRunning, setIsOcrRunning] = useState(false)
  const [isLayoutRunning, setIsLayoutRunning] = useState(false)
  const [isTypographyRunning, setIsTypographyRunning] = useState(false)
  const [isReconstructionRunning, setIsReconstructionRunning] = useState(false)
  const [isSceneRunning, setIsSceneRunning] = useState(false)
  const [isVectorRunning, setIsVectorRunning] = useState(false)
  const [isRendering, setIsRendering] = useState(false)
  const [isOptimizing, setIsOptimizing] = useState(false)
  const [uploadResult, setUploadResult] = useState(null)
  const [ocrResult, setOcrResult] = useState(null)
  const [layoutResult, setLayoutResult] = useState(null)
  const [typographyResult, setTypographyResult] = useState(null)
  const [reconstructionResult, setReconstructionResult] = useState(null)
  const [sceneResult, setSceneResult] = useState(null)
  const [vectorResult, setVectorResult] = useState(null)
  const [pdfResult, setPdfResult] = useState(null)
  const [optimizeResult, setOptimizeResult] = useState(null)

  useEffect(() => {
    return () => {
      if (previewUrl) URL.revokeObjectURL(previewUrl)
    }
  }, [previewUrl])

  const validateFile = useCallback((candidate) => {
    if (!candidate) return false

    const mimeOk = ACCEPTED_MIME_SET.has(candidate.type)
    const extOk = hasAcceptedExtension(candidate)
    if (!mimeOk && !extOk) {
      toast.error('Unsupported format. Use PNG, JPEG, WEBP, BMP, or TIFF.')
      return false
    }

    if (candidate.size > MAX_FILE_SIZE) {
      toast.error('File is too large. Maximum size is 20 MB.')
      return false
    }

    return true
  }, [])

  const loadDimensions = useCallback((url) => {
    const img = new Image()
    img.onload = () => {
      setDimensions({ width: img.naturalWidth, height: img.naturalHeight })
    }
    img.onerror = () => {
      setDimensions(null)
    }
    img.src = url
  }, [])

  const selectImage = useCallback(
    (nextFile) => {
      if (!validateFile(nextFile)) return

      setPreviewUrl((prev) => {
        if (prev) URL.revokeObjectURL(prev)
        const url = URL.createObjectURL(nextFile)
        loadDimensions(url)
        return url
      })

      setFile(nextFile)
      setUploadResult(null)
      setOcrResult(null)
      setLayoutResult(null)
      setTypographyResult(null)
      setReconstructionResult(null)
      setSceneResult(null)
      setVectorResult(null)
      setPdfResult(null)
      setOptimizeResult(null)
      toast.success('Image selected')
    },
    [validateFile, loadDimensions],
  )

  const removeImage = useCallback(() => {
    setPreviewUrl((prev) => {
      if (prev) URL.revokeObjectURL(prev)
      return null
    })
    setFile(null)
    setDimensions(null)
    setUploadResult(null)
    setOcrResult(null)
    setLayoutResult(null)
    setTypographyResult(null)
    setReconstructionResult(null)
    setSceneResult(null)
    setVectorResult(null)
    setPdfResult(null)
    setOptimizeResult(null)
    toast.success('Image removed')
  }, [])

  /**
   * Upload → OCR → layout → typography → reconstruction → scene → vectors (no PDF).
   */
  const generatePdf = useCallback(async () => {
    if (!file) {
      toast.error('Please upload an image first')
      return
    }

    setIsGenerating(true)
    setUploadResult(null)
    setOcrResult(null)
    setLayoutResult(null)
    setTypographyResult(null)
    setReconstructionResult(null)
    setSceneResult(null)
    setVectorResult(null)
    setPdfResult(null)
    setOptimizeResult(null)

    try {
      const data = await uploadImage(file)
      setUploadResult(data)
      toast.success(data.message || 'Image uploaded successfully.')

      setIsOcrRunning(true)
      try {
        const ocr = await runOcr(data.image_id)
        setOcrResult(ocr)
        if (ocr.warning) {
          toast(ocr.warning, { icon: '⚠️' })
        } else {
          toast.success(
            `OCR complete — ${ocr.total_blocks} text block${ocr.total_blocks === 1 ? '' : 's'}`,
          )
        }

        setIsLayoutRunning(true)
        try {
          const layout = await runLayout(data.image_id)
          setLayoutResult(layout)
          toast.success(
            `Layout complete — ${layout.counts?.total ?? layout.objects?.length ?? 0} objects`,
          )

          setIsTypographyRunning(true)
          try {
            const typo = await runTypography(data.image_id)
            setTypographyResult(typo)
            toast.success(
              `Typography complete — ${typo.summary?.total_styles ?? typo.text_styles?.length ?? 0} styles`,
            )

            setIsReconstructionRunning(true)
            try {
              const plan = await runReconstruction(data.image_id)
              setReconstructionResult(plan)
              toast.success(
                `Plan ready — score ${Math.round(plan.summary?.overall_score ?? 0)}`,
              )

              setIsSceneRunning(true)
              try {
                const scene = await runScene(data.image_id)
                setSceneResult(scene)
                toast.success(
                  `Scene ready — ${scene.summary?.counts?.total_objects ?? scene.objects?.length ?? 0} objects`,
                )

                setIsVectorRunning(true)
                try {
                  const vectors = await runVector(data.image_id)
                  setVectorResult(vectors)
                  toast.success(
                    `Vectors ready — score ${Math.round(vectors.summary?.vector_score ?? 0)}`,
                  )

                  setIsRendering(true)
                  try {
                    const pdf = await runRender(data.image_id)
                    setPdfResult(pdf)
                    toast.success('Editable PDF generated successfully')

                    setIsOptimizing(true)
                    try {
                      const opt = await runOptimize(data.image_id)
                      setOptimizeResult(opt)
                      const sim = opt.summary?.accuracy?.overall_similarity
                      toast.success(
                        sim != null
                          ? `Optimization complete — ${Number(sim).toFixed(1)}% similarity`
                          : 'Optimization complete',
                      )
                    } catch (optErr) {
                      console.error('Optimization failed:', optErr)
                      toast.error(getApiErrorMessage(optErr))
                    } finally {
                      setIsOptimizing(false)
                    }
                  } catch (pdfErr) {
                    console.error('PDF render failed:', pdfErr)
                    toast.error(getApiErrorMessage(pdfErr))
                  } finally {
                    setIsRendering(false)
                  }
                } catch (vectorErr) {
                  console.error('Vector reconstruction failed:', vectorErr)
                  toast.error(getApiErrorMessage(vectorErr))
                } finally {
                  setIsVectorRunning(false)
                }
              } catch (sceneErr) {
                console.error('Scene build failed:', sceneErr)
                toast.error(getApiErrorMessage(sceneErr))
              } finally {
                setIsSceneRunning(false)
              }
            } catch (reconErr) {
              console.error('Reconstruction planning failed:', reconErr)
              toast.error(getApiErrorMessage(reconErr))
            } finally {
              setIsReconstructionRunning(false)
            }
          } catch (typoErr) {
            console.error('Typography failed:', typoErr)
            toast.error(getApiErrorMessage(typoErr))
          } finally {
            setIsTypographyRunning(false)
          }
        } catch (layoutErr) {
          console.error('Layout failed:', layoutErr)
          toast.error(getApiErrorMessage(layoutErr))
        } finally {
          setIsLayoutRunning(false)
        }
      } catch (ocrErr) {
        console.error('OCR failed:', ocrErr)
        toast.error(getApiErrorMessage(ocrErr))
      } finally {
        setIsOcrRunning(false)
      }
    } catch (err) {
      console.error('Upload failed:', err)
      toast.error(getApiErrorMessage(err))
    } finally {
      setIsGenerating(false)
    }
  }, [file])

  return {
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
    hasImage: Boolean(file && previewUrl),
    acceptedTypes: ACCEPTED_TYPES,
    maxFileSize: MAX_FILE_SIZE,
    selectImage,
    removeImage,
    generatePdf,
    validateFile,
    formatFileSize,
    getFileTypeLabel,
  }
}
