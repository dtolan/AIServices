import React, { useState, useEffect } from 'react'
import { FiSave, FiRefreshCw, FiAlertCircle, FiCheckCircle, FiEye, FiEyeOff } from 'react-icons/fi'
import { api } from '../api/client'
import useStore from '../store/useStore'

export default function Settings() {
  const [settings, setSettings] = useState(null)
  const [originalSettings, setOriginalSettings] = useState(null)
  const [loading, setLoading] = useState(true)
  const [saving, setSaving] = useState(false)
  const [message, setMessage] = useState(null)
  const [showApiKeys, setShowApiKeys] = useState(false)
  const [testingConnection, setTestingConnection] = useState(null)
  const [availableModels, setAvailableModels] = useState({
    ollama: [],
    claude: [],
    gemini: []
  })
  const [loadingModels, setLoadingModels] = useState({
    ollama: false,
    claude: false,
    gemini: false
  })
  const [vramInfo, setVramInfo] = useState(null)
  const [detectingVram, setDetectingVram] = useState(false)

  useEffect(() => {
    loadSettings()
  }, [])

  const loadSettings = async () => {
    try {
      setLoading(true)
      const response = await api.getSettings()
      setSettings(response.data)
      setOriginalSettings(response.data)
      // Auto-load Ollama models since it doesn't require API key
      fetchAvailableModels('ollama')
      // Load VRAM info if auto-detection is enabled
      if (response.data.vram_detection_mode === 'auto') {
        fetchVramInfo()
      }
    } catch (error) {
      setMessage({ type: 'error', text: `Failed to load settings: ${error.message}` })
    } finally {
      setLoading(false)
    }
  }

  const fetchVramInfo = async () => {
    try {
      setDetectingVram(true)
      const response = await api.getGpuMemory()
      setVramInfo(response.data)
    } catch (error) {
      console.error('Failed to fetch VRAM info:', error)
      setVramInfo({
        detection_successful: false,
        error: error.response?.data?.detail || error.message
      })
    } finally {
      setDetectingVram(false)
    }
  }

  const fetchAvailableModels = async (provider) => {
    try {
      setLoadingModels(prev => ({ ...prev, [provider]: true }))

      const requestData = { provider }
      // Add API key if needed for cloud providers
      // Check if key exists and is not just asterisks (masked)
      if (provider === 'claude') {
        const key = settings?.anthropic_api_key
        if (key && key.length > 0 && !key.match(/^\*+$/)) {
          requestData.api_key = key
        }
      } else if (provider === 'gemini') {
        const key = settings?.google_api_key
        if (key && key.length > 0 && !key.match(/^\*+$/)) {
          requestData.api_key = key
        }
      }

      console.log(`Fetching ${provider} models with API key:`, requestData.api_key ? 'provided' : 'not provided')
      const response = await api.getAvailableModels(requestData)
      setAvailableModels(prev => ({
        ...prev,
        [provider]: response.data.models
      }))
    } catch (error) {
      console.error(`Failed to fetch ${provider} models:`, error)
      setMessage({
        type: 'error',
        text: `Failed to fetch ${provider} models: ${error.response?.data?.detail || error.message}`
      })
    } finally {
      setLoadingModels(prev => ({ ...prev, [provider]: false }))
    }
  }

  const handleChange = (key, value) => {
    setSettings(prev => ({ ...prev, [key]: value }))
    // Auto-fetch VRAM info when switching to auto mode
    if (key === 'vram_detection_mode' && value === 'auto') {
      fetchVramInfo()
    }
  }

  const hasChanges = () => {
    return JSON.stringify(settings) !== JSON.stringify(originalSettings)
  }

  const handleSave = async () => {
    try {
      setSaving(true)

      // Save settings to .env file
      await api.updateSettings(settings)

      // Reload settings in backend to apply changes immediately
      try {
        const reloadResponse = await api.reloadSettings()
        setMessage({
          type: 'success',
          text: 'Settings saved and applied successfully! Changes are now active.'
        })
        console.log('Settings reloaded:', reloadResponse.data)
      } catch (reloadError) {
        // Settings were saved but reload failed
        console.error('Failed to reload settings:', reloadError)
        setMessage({
          type: 'success',
          text: 'Settings saved! Please restart the application for changes to take effect.'
        })
      }

      setOriginalSettings(settings)
    } catch (error) {
      setMessage({ type: 'error', text: `Failed to save settings: ${error.message}` })
    } finally {
      setSaving(false)
    }
  }

  const handleReset = () => {
    setSettings(originalSettings)
    setMessage(null)
  }

  const handleTestConnection = async (serviceType, config) => {
    try {
      setTestingConnection(serviceType)
      const response = await api.testConnection({
        service_type: serviceType,
        config
      })
      setMessage({
        type: response.data.success ? 'success' : 'error',
        text: response.data.message
      })
    } catch (error) {
      setMessage({ type: 'error', text: `Connection test failed: ${error.message}` })
    } finally {
      setTestingConnection(null)
    }
  }

  // Render model selection dropdown or text input with refresh button
  const renderModelSelector = (provider, currentValue, onChange) => {
    const models = availableModels[provider] || []
    const isLoading = loadingModels[provider]

    return (
      <div className="flex gap-2">
        <div className="flex-1">
          {models.length > 0 ? (
            <select
              value={currentValue}
              onChange={(e) => onChange(e.target.value)}
              className="input w-full"
              disabled={isLoading}
            >
              {models.map(model => (
                <option key={model} value={model}>{model}</option>
              ))}
            </select>
          ) : (
            <input
              type="text"
              value={currentValue}
              onChange={(e) => onChange(e.target.value)}
              className="input w-full"
              placeholder={isLoading ? 'Loading models...' : `Enter ${provider} model name`}
              disabled={isLoading}
            />
          )}
        </div>
        <button
          onClick={() => fetchAvailableModels(provider)}
          className="btn-secondary px-3"
          disabled={isLoading}
          title="Refresh available models"
        >
          <FiRefreshCw className={isLoading ? 'animate-spin' : ''} />
        </button>
      </div>
    )
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="loading-spinner"></div>
      </div>
    )
  }

  if (!settings) {
    return (
      <div className="p-6 text-center">
        <p className="text-red-500">Failed to load settings</p>
        <button onClick={loadSettings} className="btn-primary mt-4">
          Retry
        </button>
      </div>
    )
  }

  return (
    <div className="max-w-6xl mx-auto p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <h1 className="text-3xl font-bold">Application Settings</h1>
        <div className="flex gap-2">
          {hasChanges() && (
            <button
              onClick={handleReset}
              className="btn-secondary"
              disabled={saving}
            >
              <FiRefreshCw className="inline mr-2" />
              Reset
            </button>
          )}
          <button
            onClick={handleSave}
            className="btn-primary"
            disabled={!hasChanges() || saving}
          >
            {saving ? (
              <>
                <div className="loading-spinner inline-block mr-2"></div>
                Saving...
              </>
            ) : (
              <>
                <FiSave className="inline mr-2" />
                Save Settings
              </>
            )}
          </button>
        </div>
      </div>

      {/* Messages */}
      {message && (
        <div className={`p-4 rounded-lg flex items-start gap-3 ${
          message.type === 'success' ? 'bg-green-500/20 text-green-300' : 'bg-red-500/20 text-red-300'
        }`}>
          {message.type === 'success' ? <FiCheckCircle className="flex-shrink-0 mt-0.5" /> : <FiAlertCircle className="flex-shrink-0 mt-0.5" />}
          <p className="flex-1">{message.text}</p>
        </div>
      )}

      {/* LLM Configuration */}
      <div className="card">
        <h2 className="text-xl font-bold mb-4">LLM Configuration</h2>

        {/* Dual-LLM Toggle */}
        <div className="mb-6 p-4 bg-white/5 rounded-lg">
          <label className="flex items-center gap-3 cursor-pointer">
            <input
              type="checkbox"
              checked={settings.use_dual_llm}
              onChange={(e) => handleChange('use_dual_llm', e.target.checked)}
              className="w-5 h-5"
            />
            <div>
              <div className="font-semibold">Use Dual-LLM Mode</div>
              <div className="text-sm text-gray-400">Use different LLMs for planning (quality) vs execution (speed)</div>
            </div>
          </label>
        </div>

        {settings.use_dual_llm ? (
          <div className="grid md:grid-cols-2 gap-6">
            {/* Planning LLM */}
            <div>
              <h3 className="font-semibold mb-3 text-purple-400">Planning LLM (Quality)</h3>
              <div className="space-y-3">
                <div>
                  <label className="block text-sm mb-1">Provider</label>
                  <select
                    value={settings.planning_llm_provider}
                    onChange={(e) => handleChange('planning_llm_provider', e.target.value)}
                    className="input w-full"
                  >
                    <option value="ollama">Ollama (Local)</option>
                    <option value="claude">Claude (Cloud)</option>
                    <option value="gemini">Gemini (Cloud)</option>
                  </select>
                </div>
                {settings.planning_llm_provider === 'ollama' && (
                  <div>
                    <label className="block text-sm mb-1">Model</label>
                    {renderModelSelector('ollama', settings.planning_ollama_model, (value) => handleChange('planning_ollama_model', value))}
                  </div>
                )}
                {settings.planning_llm_provider === 'claude' && (
                  <div>
                    <label className="block text-sm mb-1">Model</label>
                    {renderModelSelector('claude', settings.planning_claude_model, (value) => handleChange('planning_claude_model', value))}
                  </div>
                )}
                {settings.planning_llm_provider === 'gemini' && (
                  <div>
                    <label className="block text-sm mb-1">Model</label>
                    {renderModelSelector('gemini', settings.planning_gemini_model, (value) => handleChange('planning_gemini_model', value))}
                  </div>
                )}
              </div>
            </div>

            {/* Execution LLM */}
            <div>
              <h3 className="font-semibold mb-3 text-blue-400">Execution LLM (Speed)</h3>
              <div className="space-y-3">
                <div>
                  <label className="block text-sm mb-1">Provider</label>
                  <select
                    value={settings.execution_llm_provider}
                    onChange={(e) => handleChange('execution_llm_provider', e.target.value)}
                    className="input w-full"
                  >
                    <option value="ollama">Ollama (Local)</option>
                    <option value="claude">Claude (Cloud)</option>
                    <option value="gemini">Gemini (Cloud)</option>
                  </select>
                </div>
                {settings.execution_llm_provider === 'ollama' && (
                  <div>
                    <label className="block text-sm mb-1">Model</label>
                    {renderModelSelector('ollama', settings.execution_ollama_model, (value) => handleChange('execution_ollama_model', value))}
                  </div>
                )}
                {settings.execution_llm_provider === 'claude' && (
                  <div>
                    <label className="block text-sm mb-1">Model</label>
                    {renderModelSelector('claude', settings.execution_claude_model, (value) => handleChange('execution_claude_model', value))}
                  </div>
                )}
                {settings.execution_llm_provider === 'gemini' && (
                  <div>
                    <label className="block text-sm mb-1">Model</label>
                    {renderModelSelector('gemini', settings.execution_gemini_model, (value) => handleChange('execution_gemini_model', value))}
                  </div>
                )}
              </div>
            </div>
          </div>
        ) : (
          /* Single LLM */
          <div className="space-y-3">
            <div>
              <label className="block text-sm mb-1">Provider</label>
              <select
                value={settings.llm_provider}
                onChange={(e) => handleChange('llm_provider', e.target.value)}
                className="input w-full"
              >
                <option value="ollama">Ollama (Local)</option>
                <option value="claude">Claude (Cloud)</option>
                <option value="gemini">Gemini (Cloud)</option>
              </select>
            </div>
            {settings.llm_provider === 'ollama' && (
              <div>
                <label className="block text-sm mb-1">Model</label>
                {renderModelSelector('ollama', settings.ollama_model, (value) => handleChange('ollama_model', value))}
              </div>
            )}
            {settings.llm_provider === 'claude' && (
              <div>
                <label className="block text-sm mb-1">Model</label>
                {renderModelSelector('claude', settings.claude_model, (value) => handleChange('claude_model', value))}
              </div>
            )}
            {settings.llm_provider === 'gemini' && (
              <div>
                <label className="block text-sm mb-1">Model</label>
                {renderModelSelector('gemini', settings.gemini_model, (value) => handleChange('gemini_model', value))}
              </div>
            )}
          </div>
        )}
      </div>

      {/* Provider Settings */}
      <div className="card">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-xl font-bold">Provider Settings</h2>
          <button
            onClick={() => setShowApiKeys(!showApiKeys)}
            className="btn-secondary text-sm"
          >
            {showApiKeys ? <FiEyeOff className="inline mr-1" /> : <FiEye className="inline mr-1" />}
            {showApiKeys ? 'Hide' : 'Show'} API Keys
          </button>
        </div>

        <div className="grid md:grid-cols-2 gap-4">
          {/* Ollama */}
          <div className="p-4 bg-white/5 rounded-lg space-y-3">
            <h3 className="font-semibold">Ollama (Local)</h3>
            <div>
              <label className="block text-sm mb-1">Host URL</label>
              <input
                type="text"
                value={settings.ollama_host}
                onChange={(e) => handleChange('ollama_host', e.target.value)}
                className="input w-full"
                placeholder="http://localhost:11434"
              />
            </div>
            <label className="flex items-center gap-2 cursor-pointer">
              <input
                type="checkbox"
                checked={settings.ollama_auto_configure}
                onChange={(e) => handleChange('ollama_auto_configure', e.target.checked)}
                className="w-4 h-4"
              />
              <span className="text-sm">Auto-configure based on VRAM</span>
            </label>
            <button
              onClick={() => handleTestConnection('ollama', { host: settings.ollama_host })}
              className="btn-secondary w-full text-sm"
              disabled={testingConnection === 'ollama'}
            >
              {testingConnection === 'ollama' ? 'Testing...' : 'Test Connection'}
            </button>
          </div>

          {/* Claude */}
          <div className="p-4 bg-white/5 rounded-lg space-y-3">
            <h3 className="font-semibold">Anthropic Claude</h3>
            <div>
              <label className="block text-sm mb-1">API Key</label>
              <input
                type={showApiKeys ? "text" : "password"}
                value={settings.anthropic_api_key}
                onChange={(e) => handleChange('anthropic_api_key', e.target.value)}
                className="input w-full"
                placeholder="sk-ant-..."
              />
            </div>
            <div className="flex gap-2">
              <button
                onClick={() => handleTestConnection('claude', { api_key: settings.anthropic_api_key })}
                className="btn-secondary flex-1 text-sm"
                disabled={testingConnection === 'claude' || !settings.anthropic_api_key || settings.anthropic_api_key.match(/^\*+$/)}
              >
                {testingConnection === 'claude' ? 'Testing...' : 'Test Connection'}
              </button>
              <button
                onClick={() => fetchAvailableModels('claude')}
                className="btn-secondary px-3 text-sm"
                disabled={loadingModels.claude}
                title="Fetch available models"
              >
                <FiRefreshCw className={loadingModels.claude ? 'animate-spin' : ''} />
              </button>
            </div>
          </div>

          {/* Gemini */}
          <div className="p-4 bg-white/5 rounded-lg space-y-3">
            <h3 className="font-semibold">Google Gemini</h3>
            <div>
              <label className="block text-sm mb-1">API Key</label>
              <input
                type={showApiKeys ? "text" : "password"}
                value={settings.google_api_key}
                onChange={(e) => handleChange('google_api_key', e.target.value)}
                className="input w-full"
                placeholder="AIza..."
              />
            </div>
            <div className="flex gap-2">
              <button
                onClick={() => handleTestConnection('gemini', { api_key: settings.google_api_key })}
                className="btn-secondary flex-1 text-sm"
                disabled={testingConnection === 'gemini' || !settings.google_api_key || settings.google_api_key.match(/^\*+$/)}
              >
                {testingConnection === 'gemini' ? 'Testing...' : 'Test Connection'}
              </button>
              <button
                onClick={() => fetchAvailableModels('gemini')}
                className="btn-secondary px-3 text-sm"
                disabled={loadingModels.gemini}
                title="Fetch available models"
              >
                <FiRefreshCw className={loadingModels.gemini ? 'animate-spin' : ''} />
              </button>
            </div>
          </div>

          {/* CivitAI */}
          <div className="p-4 bg-white/5 rounded-lg space-y-3">
            <h3 className="font-semibold">CivitAI</h3>
            <div>
              <label className="block text-sm mb-1">API Key</label>
              <input
                type={showApiKeys ? "text" : "password"}
                value={settings.civitai_api_key || ''}
                onChange={(e) => handleChange('civitai_api_key', e.target.value)}
                className="input w-full"
                placeholder="Enter your CivitAI API key"
              />
            </div>
            <p className="text-xs text-gray-400">
              Required for downloading models from CivitAI. Get your API key from{' '}
              <a
                href="https://civitai.com/user/account"
                target="_blank"
                rel="noopener noreferrer"
                className="text-primary-400 hover:text-primary-300 underline"
              >
                your CivitAI account
              </a>
            </p>
          </div>
        </div>
      </div>

      {/* Stable Diffusion Settings */}
      <div className="card">
        <h2 className="text-xl font-bold mb-4">Stable Diffusion</h2>
        <div className="grid md:grid-cols-2 gap-4">
          <div>
            <label className="block text-sm mb-1">API URL</label>
            <input
              type="text"
              value={settings.sd_api_url}
              onChange={(e) => handleChange('sd_api_url', e.target.value)}
              className="input w-full"
              placeholder="http://localhost:7860"
            />
          </div>
          <div>
            <label className="block text-sm mb-1">Timeout (seconds)</label>
            <input
              type="number"
              value={settings.sd_api_timeout}
              onChange={(e) => handleChange('sd_api_timeout', parseInt(e.target.value))}
              className="input w-full"
              min="30"
              max="600"
            />
          </div>
        </div>
        <button
          onClick={() => handleTestConnection('sd', { api_url: settings.sd_api_url })}
          className="btn-secondary mt-3"
          disabled={testingConnection === 'sd'}
        >
          {testingConnection === 'sd' ? 'Testing...' : 'Test Connection'}
        </button>
      </div>

      {/* GPU VRAM Detection */}
      <div className="card">
        <h2 className="text-xl font-bold mb-4">GPU VRAM Detection</h2>
        <p className="text-sm text-gray-400 mb-4">
          VRAM information helps the AI recommend appropriate resolutions and models for your GPU.
        </p>

        <div className="space-y-4">
          {/* Detection Mode */}
          <div>
            <label className="block text-sm mb-2 font-semibold">Detection Mode</label>
            <div className="space-y-2">
              <label className="flex items-center gap-3 p-3 bg-white/5 rounded-lg cursor-pointer hover:bg-white/10 transition-colors">
                <input
                  type="radio"
                  name="vram_detection_mode"
                  value="auto"
                  checked={settings.vram_detection_mode === 'auto'}
                  onChange={(e) => handleChange('vram_detection_mode', e.target.value)}
                  className="w-4 h-4"
                />
                <div className="flex-1">
                  <div className="font-medium">Auto-detect from Stable Diffusion API</div>
                  <div className="text-xs text-gray-400">Recommended - Automatically detects GPU VRAM</div>
                </div>
              </label>

              <label className="flex items-center gap-3 p-3 bg-white/5 rounded-lg cursor-pointer hover:bg-white/10 transition-colors">
                <input
                  type="radio"
                  name="vram_detection_mode"
                  value="manual"
                  checked={settings.vram_detection_mode === 'manual'}
                  onChange={(e) => handleChange('vram_detection_mode', e.target.value)}
                  className="w-4 h-4"
                />
                <div className="flex-1">
                  <div className="font-medium">Manual Entry</div>
                  <div className="text-xs text-gray-400">Manually specify your GPU VRAM amount</div>
                </div>
              </label>

              <label className="flex items-center gap-3 p-3 bg-white/5 rounded-lg cursor-pointer hover:bg-white/10 transition-colors">
                <input
                  type="radio"
                  name="vram_detection_mode"
                  value="disabled"
                  checked={settings.vram_detection_mode === 'disabled'}
                  onChange={(e) => handleChange('vram_detection_mode', e.target.value)}
                  className="w-4 h-4"
                />
                <div className="flex-1">
                  <div className="font-medium">Disabled</div>
                  <div className="text-xs text-gray-400">Don't use VRAM information</div>
                </div>
              </label>
            </div>
          </div>

          {/* Manual VRAM Input */}
          {settings.vram_detection_mode === 'manual' && (
            <div className="p-4 bg-white/5 rounded-lg">
              <label className="block text-sm mb-2">GPU VRAM (GB)</label>
              <input
                type="number"
                step="0.5"
                min="1"
                max="80"
                value={settings.vram_manual_gb}
                onChange={(e) => handleChange('vram_manual_gb', parseFloat(e.target.value))}
                className="input w-full md:w-48"
                placeholder="e.g., 8.0"
              />
              <p className="text-xs text-gray-400 mt-2">
                Enter your GPU's VRAM in gigabytes (e.g., 8.0 for 8GB)
              </p>
            </div>
          )}

          {/* Detection Status */}
          {settings.vram_detection_mode === 'auto' && (
            <div className="p-4 bg-white/5 rounded-lg space-y-3">
              <div className="flex items-center justify-between">
                <h3 className="font-semibold">Detection Status</h3>
                <button
                  onClick={fetchVramInfo}
                  className="btn-secondary text-sm px-3 py-1"
                  disabled={detectingVram}
                >
                  {detectingVram ? (
                    <>
                      <div className="loading-spinner inline-block mr-2 w-3 h-3"></div>
                      Detecting...
                    </>
                  ) : (
                    <>
                      <FiRefreshCw className="inline mr-1" />
                      Detect Now
                    </>
                  )}
                </button>
              </div>

              {vramInfo && (
                <div className="space-y-2 text-sm">
                  {vramInfo.detection_successful !== false ? (
                    <>
                      <div className="flex items-center gap-2 text-green-400">
                        <FiCheckCircle />
                        <span>Detection Successful</span>
                      </div>
                      <div className="grid grid-cols-2 gap-2 text-gray-300">
                        <div>
                          <span className="text-gray-400">Total VRAM:</span>
                          <span className="ml-2 font-semibold">{vramInfo.vram_total_gb?.toFixed(1)}GB</span>
                        </div>
                        <div>
                          <span className="text-gray-400">Free:</span>
                          <span className="ml-2 font-semibold">{vramInfo.vram_free_gb?.toFixed(1)}GB</span>
                        </div>
                        <div>
                          <span className="text-gray-400">Used:</span>
                          <span className="ml-2 font-semibold">{vramInfo.vram_used_gb?.toFixed(1)}GB</span>
                        </div>
                        <div>
                          <span className="text-gray-400">Method:</span>
                          <span className="ml-2 font-mono text-xs">{vramInfo.detection_method}</span>
                        </div>
                      </div>
                      {vramInfo.gpu_name && (
                        <div className="pt-2 border-t border-white/10">
                          <span className="text-gray-400">GPU:</span>
                          <span className="ml-2 font-medium">{vramInfo.gpu_name}</span>
                        </div>
                      )}
                    </>
                  ) : (
                    <div className="flex items-start gap-2 text-red-400">
                      <FiAlertCircle className="flex-shrink-0 mt-0.5" />
                      <div>
                        <div className="font-medium">Detection Failed</div>
                        <div className="text-xs text-gray-400 mt-1">
                          {vramInfo.error || 'Unable to detect GPU VRAM'}
                        </div>
                      </div>
                    </div>
                  )}
                </div>
              )}
            </div>
          )}
        </div>
      </div>

      {/* Generation Defaults */}
      <div className="card">
        <h2 className="text-xl font-bold mb-4">Generation Defaults</h2>
        <div className="grid md:grid-cols-2 gap-4">
          <div>
            <label className="block text-sm mb-1">Steps (1-150)</label>
            <input
              type="number"
              value={settings.default_steps}
              onChange={(e) => handleChange('default_steps', parseInt(e.target.value))}
              className="input w-full"
              min="1"
              max="150"
            />
          </div>
          <div>
            <label className="block text-sm mb-1">CFG Scale (1.0-30.0)</label>
            <input
              type="number"
              step="0.5"
              value={settings.default_cfg_scale}
              onChange={(e) => handleChange('default_cfg_scale', parseFloat(e.target.value))}
              className="input w-full"
              min="1"
              max="30"
            />
          </div>
          <div>
            <label className="block text-sm mb-1">Width</label>
            <input
              type="number"
              value={settings.default_width}
              onChange={(e) => handleChange('default_width', parseInt(e.target.value))}
              className="input w-full"
              min="64"
              max="2048"
              step="8"
            />
          </div>
          <div>
            <label className="block text-sm mb-1">Height</label>
            <input
              type="number"
              value={settings.default_height}
              onChange={(e) => handleChange('default_height', parseInt(e.target.value))}
              className="input w-full"
              min="64"
              max="2048"
              step="8"
            />
          </div>
          <div className="md:col-span-2">
            <label className="block text-sm mb-1">Sampler</label>
            <input
              type="text"
              value={settings.default_sampler}
              onChange={(e) => handleChange('default_sampler', e.target.value)}
              className="input w-full"
              placeholder="DPM++ 2M Karras"
            />
          </div>
        </div>
      </div>

      {/* Save reminder if changes exist */}
      {hasChanges() && (
        <div className="fixed bottom-6 right-6 card shadow-2xl border-2 border-yellow-500/50">
          <div className="flex items-center gap-3">
            <FiAlertCircle className="text-yellow-500 flex-shrink-0" />
            <div>
              <p className="font-semibold">Unsaved Changes</p>
              <p className="text-sm text-gray-400">Don't forget to save your settings</p>
            </div>
            <button
              onClick={handleSave}
              className="btn-primary ml-4"
              disabled={saving}
            >
              {saving ? 'Saving...' : 'Save Now'}
            </button>
          </div>
        </div>
      )}
    </div>
  )
}
