import { useState, useEffect } from 'react'
import { FiZap, FiBook, FiSettings, FiPackage, FiCpu } from 'react-icons/fi'
import { api } from '../api/client'

export default function Header({ activeTab, onTabChange }) {
  const [llmConfig, setLlmConfig] = useState(null)

  const tabs = [
    { id: 'generate', name: 'Generate', icon: FiZap },
    { id: 'library', name: 'Library', icon: FiBook },
    { id: 'models', name: 'Models', icon: FiPackage },
    { id: 'settings', name: 'Settings', icon: FiSettings },
  ]

  useEffect(() => {
    loadLlmConfig()
  }, [])

  const loadLlmConfig = async () => {
    try {
      const response = await api.getSettings()
      setLlmConfig(response.data)
    } catch (error) {
      console.error('Failed to load LLM config:', error)
    }
  }

  const getLlmDisplayText = () => {
    if (!llmConfig) return ''

    if (llmConfig.use_dual_llm) {
      const planProvider = llmConfig.planning_llm_provider.charAt(0).toUpperCase() + llmConfig.planning_llm_provider.slice(1)
      const execProvider = llmConfig.execution_llm_provider.charAt(0).toUpperCase() + llmConfig.execution_llm_provider.slice(1)

      let planModel = ''
      let execModel = ''

      if (llmConfig.planning_llm_provider === 'ollama') planModel = llmConfig.planning_ollama_model
      else if (llmConfig.planning_llm_provider === 'claude') planModel = llmConfig.planning_claude_model.split('-').slice(-1)[0] // e.g. "20250929" -> show model
      else if (llmConfig.planning_llm_provider === 'gemini') planModel = llmConfig.planning_gemini_model

      if (llmConfig.execution_llm_provider === 'ollama') execModel = llmConfig.execution_ollama_model
      else if (llmConfig.execution_llm_provider === 'claude') execModel = llmConfig.execution_claude_model.split('-').slice(-1)[0]
      else if (llmConfig.execution_llm_provider === 'gemini') execModel = llmConfig.execution_gemini_model

      return `Plan: ${planProvider} | Act: ${execProvider}`
    } else {
      const provider = llmConfig.llm_provider.charAt(0).toUpperCase() + llmConfig.llm_provider.slice(1)
      let model = ''

      if (llmConfig.llm_provider === 'ollama') model = llmConfig.ollama_model
      else if (llmConfig.llm_provider === 'claude') model = llmConfig.claude_model
      else if (llmConfig.llm_provider === 'gemini') model = llmConfig.gemini_model

      return `LLM: ${provider}`
    }
  }

  return (
    <header className="glass border-b border-white/20">
      <div className="container mx-auto px-4">
        <div className="flex items-center justify-between h-16">
          <div className="flex items-center space-x-3">
            <div className="w-10 h-10 bg-gradient-to-br from-primary-500 to-purple-500 rounded-lg flex items-center justify-center">
              <FiZap className="w-6 h-6" />
            </div>
            <div>
              <h1 className="text-xl font-bold">SD Prompt Assistant</h1>
              <div className="flex items-center gap-2 text-xs text-white/60">
                <span>AI-Powered Prompt Engineering</span>
                {llmConfig && (
                  <>
                    <span>•</span>
                    <div className="flex items-center gap-1">
                      <FiCpu className="w-3 h-3" />
                      <span>{getLlmDisplayText()}</span>
                    </div>
                  </>
                )}
              </div>
            </div>
          </div>

          <nav className="flex space-x-2">
            {tabs.map((tab) => {
              const Icon = tab.icon
              const isActive = activeTab === tab.id

              return (
                <button
                  key={tab.id}
                  onClick={() => onTabChange(tab.id)}
                  className={`
                    flex items-center space-x-2 px-4 py-2 rounded-lg transition-all duration-200
                    ${isActive
                      ? 'bg-white/20 text-white'
                      : 'text-white/60 hover:bg-white/10 hover:text-white'}
                  `}
                >
                  <Icon className="w-5 h-5" />
                  <span className="font-medium">{tab.name}</span>
                </button>
              )
            })}
          </nav>
        </div>
      </div>
    </header>
  )
}
