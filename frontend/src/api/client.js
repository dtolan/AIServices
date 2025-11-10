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
}

export default client
