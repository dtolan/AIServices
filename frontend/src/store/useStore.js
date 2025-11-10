import { create } from 'zustand'

const useStore = create((set, get) => ({
  // Generation history
  generations: [],
  addGeneration: (generation) => set((state) => ({
    generations: [generation, ...state.generations]
  })),
  clearGenerations: () => set({ generations: [] }),

  // Current generation state
  currentGeneration: null,
  setCurrentGeneration: (generation) => set({ currentGeneration: generation }),

  // UI state
  isGenerating: false,
  setIsGenerating: (value) => set({ isGenerating: value }),

  // Conversation state (for interactive mode)
  conversation: [],
  addMessage: (message) => set((state) => ({
    conversation: [...state.conversation, message]
  })),
  clearConversation: () => set({ conversation: [] }),

  // LLM provider info
  providerInfo: null,
  setProviderInfo: (info) => set({ providerInfo: info }),

  // Knowledge base info
  knowledgeBase: null,
  setKnowledgeBase: (info) => set({ knowledgeBase: info }),

  // Settings (UI-only)
  settings: {
    autoSave: true,
    showParameters: true,
    theme: 'dark',
  },
  updateSettings: (newSettings) => set((state) => ({
    settings: { ...state.settings, ...newSettings }
  })),

  // App Settings (from backend)
  appSettings: null,
  setAppSettings: (settings) => set({ appSettings: settings }),

  // Pending settings changes (not yet saved)
  pendingSettings: null,
  setPendingSettings: (settings) => set({ pendingSettings: settings }),

  // Track if settings have unsaved changes
  settingsDirty: false,
  setSettingsDirty: (dirty) => set({ settingsDirty: dirty }),

  // Prompt library
  savedPrompts: [],
  savePrompt: (prompt) => set((state) => ({
    savedPrompts: [...state.savedPrompts, { ...prompt, id: Date.now(), savedAt: new Date().toISOString() }]
  })),
  deletePrompt: (id) => set((state) => ({
    savedPrompts: state.savedPrompts.filter(p => p.id !== id)
  })),

  // Search/filter
  searchQuery: '',
  setSearchQuery: (query) => set({ searchQuery: query }),

  // View mode
  viewMode: 'grid', // 'grid' or 'list'
  setViewMode: (mode) => set({ viewMode: mode }),
}))

export default useStore
