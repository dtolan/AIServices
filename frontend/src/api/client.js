import axios from 'axios'

const API_BASE = '/api'

const client = axios.create({
  baseURL: API_BASE,
  timeout: 300000, // 5 minutes for generation
})

export const api = {
  // Health & Info
  getHealth: () => client.get('/health'),
  getHardware: () => client.get('/hardware'),
  getKnowledgeBase: () => client.get('/knowledge-base'),

  // Prompt Enhancement
  enhancePrompt: (data) => client.post('/enhance-prompt', data),

  // Generation
  generate: (data) => client.post('/generate', data),
  iterate: (data) => client.post('/iterate', data),

  // Interactive Mode
  askQuestions: (data) => client.post('/interactive/ask', data),
  generateFromConversation: (data) => client.post('/interactive/generate', data),

  // Stable Diffusion Info
  getSamplers: () => client.get('/samplers'),
  getModels: () => client.get('/models'),

  // Prompt Library
  savePrompt: (data) => client.post('/prompts/save', data),
  getPrompts: () => client.get('/prompts'),
  deletePrompt: (id) => client.delete(`/prompts/${id}`),

  // Settings
  getSettings: () => client.get('/settings'),
  validateSettings: (data) => client.post('/settings/validate', data),
  updateSettings: (data) => client.post('/settings/update', data),
  testConnection: (data) => client.post('/settings/test-connection', data),
  getAvailableModels: (data) => client.post('/settings/available-models', data),

  // Model Management
  getRecommendedModels: () => client.get('/sd-models/recommended'),
  getInstalledModels: () => client.get('/sd-models/installed'),
  getModelsDirectory: () => client.get('/sd-models/directory'),
  getDownloadsModels: () => client.get('/sd-models/downloads'),
  setModelsDirectory: (path) => client.post('/sd-models/directory', { path }),
  deleteModel: (filename) => client.delete(`/sd-models/${filename}`),
  importModel: (sourcePath, move = true) => client.post('/sd-models/import', { source_path: sourcePath, move }),
  recommendModel: (data) => client.post('/sd-models/recommend', data),
  downloadModel: (data) => client.post('/sd-models/download', data),
  getDownloadProgress: (downloadId) => client.get(`/sd-models/download/${downloadId}/progress`),

  // Plan/Act Workflow
  planGeneration: (data) => client.post('/plan-generation', data),
  executeGeneration: (data) => client.post('/execute-generation', data),

  // Img2Img Plan/Act Workflow
  planImg2Img: (data) => client.post('/plan-img2img', data),
  executeImg2Img: (data) => client.post('/execute-img2img', data),
}

export default client
