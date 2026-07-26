/**
 * Axios client for the FastAPI backend.
 * In development, Vite proxies /api → http://127.0.0.1:8000/api
 */
import axios from 'axios'

const api = axios.create({
  baseURL: import.meta.env.VITE_API_BASE_URL || '/api',
  timeout: 300_000,
})

/**
 * Upload an image for server-side preprocessing.
 * @param {File} file
 * @returns {Promise<object>} UploadSuccessResponse from the API
 */
export async function uploadImage(file) {
  const formData = new FormData()
  formData.append('file', file)

  // Do not set Content-Type manually — Axios adds multipart boundary
  const { data } = await api.post('/upload', formData)
  return data
}

/**
 * Run OCR on a previously uploaded/processed image.
 * @param {string} imageId
 * @returns {Promise<object>} OcrSuccessResponse from the API
 */
export async function runOcr(imageId) {
  const { data } = await api.post('/ocr', { image_id: imageId })
  return data
}

/**
 * Analyze document layout for a processed image.
 * @param {string} imageId
 * @returns {Promise<object>} LayoutSuccessResponse from the API
 */
export async function runLayout(imageId) {
  const { data } = await api.post('/layout', { image_id: imageId })
  return data
}

/**
 * Analyze typography / text styles for a processed image (requires OCR).
 * @param {string} imageId
 * @returns {Promise<object>} TypographySuccessResponse from the API
 */
export async function runTypography(imageId) {
  const { data } = await api.post('/typography', { image_id: imageId })
  return data
}

/**
 * Build a reconstruction plan (text/vector/image decisions).
 * @param {string} imageId
 * @returns {Promise<object>} ReconstructionSuccessResponse
 */
export async function runReconstruction(imageId) {
  const { data } = await api.post('/reconstruction', { image_id: imageId })
  return data
}

/**
 * Build an editable scene graph from prior analysis (no PDF).
 * @param {string} imageId
 * @returns {Promise<object>} SceneSuccessResponse
 */
export async function runScene(imageId) {
  const { data } = await api.post('/scene', { image_id: imageId })
  return data
}

/**
 * Reconstruct background & vector shapes (no PDF/SVG export).
 * @param {string} imageId
 * @returns {Promise<object>} VectorSuccessResponse
 */
export async function runVector(imageId) {
  const { data } = await api.post('/vector', { image_id: imageId })
  return data
}

/**
 * Render an editable PDF from scene + vector data.
 * @param {string} imageId
 * @returns {Promise<object>} RenderSuccessResponse
 */
export async function runRender(imageId) {
  const { data } = await api.post('/render', { image_id: imageId })
  return data
}

/**
 * Optimize rendered PDF quality vs original image (no OCR re-run).
 * @param {string} imageId
 * @returns {Promise<object>} OptimizeSuccessResponse
 */
export async function runOptimize(imageId) {
  const { data } = await api.post('/optimize', { image_id: imageId })
  return data
}

export default api
