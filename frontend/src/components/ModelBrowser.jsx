import React, { useState, useEffect } from 'react'
import { FiDownload, FiTrash2, FiExternalLink, FiPackage, FiStar, FiRefreshCw, FiZap, FiUpload } from 'react-icons/fi'
import { api } from '../api/client'

export default function ModelBrowser() {
  const [recommendedModels, setRecommendedModels] = useState([])
  const [installedModels, setInstalledModels] = useState([])
  const [modelsDirectory, setModelsDirectory] = useState('')
  const [downloadsModels, setDownloadsModels] = useState([])
  const [downloadsDirectory, setDownloadsDirectory] = useState('')
  const [loading, setLoading] = useState(true)
  const [activeTab, setActiveTab] = useState('recommended') // 'recommended', 'installed'
  const [message, setMessage] = useState(null)
  const [aiRecommendation, setAiRecommendation] = useState(null)
  const [promptForRecommendation, setPromptForRecommendation] = useState('')
  const [loadingRecommendation, setLoadingRecommendation] = useState(false)
  const [downloadProgress, setDownloadProgress] = useState(null) // { downloadId, status, progress, model_name }
  const [cardDownloads, setCardDownloads] = useState({}) // Track downloads for each model card by model.id
  const [importPath, setImportPath] = useState('')
  const [importing, setImporting] = useState(false)
  const [moveFile, setMoveFile] = useState(true) // Default to moving (removing from Downloads)
  const [selectedModels, setSelectedModels] = useState([]) // For multi-select

  useEffect(() => {
    loadModels()

    // Auto-refresh Downloads folder every 5 seconds
    const refreshInterval = setInterval(() => {
      // Only refresh Downloads list, not everything
      api.getDownloadsModels().then(response => {
        setDownloadsModels(response.data.models)
      }).catch(error => {
        console.error('Failed to refresh Downloads:', error)
      })
    }, 5000) // 5 seconds

    // Cleanup interval on unmount
    return () => clearInterval(refreshInterval)
  }, [])

  const loadModels = async () => {
    try {
      setLoading(true)
      const [recommended, installed, directory, downloads] = await Promise.all([
        api.getRecommendedModels(),
        api.getInstalledModels(),
        api.getModelsDirectory(),
        api.getDownloadsModels()
      ])
      setRecommendedModels(recommended.data)
      setInstalledModels(installed.data)
      setModelsDirectory(directory.data.path)
      setDownloadsModels(downloads.data.models)
      setDownloadsDirectory(downloads.data.downloads_directory)
    } catch (error) {
      setMessage({ type: 'error', text: `Failed to load models: ${error.message}` })
    } finally {
      setLoading(false)
    }
  }

  const handleGetAiRecommendation = async () => {
    if (!promptForRecommendation.trim()) {
      setMessage({ type: 'error', text: 'Please enter a prompt first' })
      return
    }

    try {
      setLoadingRecommendation(true)
      setMessage(null)
      const response = await api.recommendModel({ prompt: promptForRecommendation })
      setAiRecommendation(response.data)
      setMessage({ type: 'success', text: 'AI recommendation generated!' })
    } catch (error) {
      setMessage({ type: 'error', text: `Failed to get recommendation: ${error.message}` })
    } finally {
      setLoadingRecommendation(false)
    }
  }

  const handleDeleteModel = async (filename) => {
    if (!confirm(`Are you sure you want to delete ${filename}?`)) {
      return
    }

    try {
      await api.deleteModel(filename)
      setMessage({ type: 'success', text: `Deleted ${filename}` })
      loadModels() // Reload the list
    } catch (error) {
      setMessage({ type: 'error', text: `Failed to delete model: ${error.message}` })
    }
  }

  const handleImportModel = async () => {
    if (!importPath.trim()) {
      setMessage({ type: 'error', text: 'Please enter a file path' })
      return
    }

    try {
      setImporting(true)
      setMessage(null)
      const response = await api.importModel(importPath, moveFile)
      const action = response.data.moved ? 'moved' : 'copied'

      // Check if this was an upgrade
      if (response.data.upgraded && response.data.replaced_versions) {
        const oldVersions = response.data.replaced_versions.join(', ')
        setMessage({
          type: 'success',
          text: `Upgraded! Replaced ${response.data.replaced_versions.length} old version(s): ${oldVersions}`
        })
      } else {
        setMessage({ type: 'success', text: `Successfully ${action} ${response.data.filename} to models directory` })
      }

      setImportPath('') // Clear the input
      loadModels() // Reload the list
    } catch (error) {
      const errorMsg = error.response?.data?.detail || error.message
      const errorData = error.response?.data

      // Check if the file was cleaned up even though import failed (duplicate)
      // Show this as info/notice rather than error since it's expected behavior
      if (errorData?.cleaned_up && moveFile) {
        setMessage({
          type: 'info',
          text: `${errorMsg} (File removed from Downloads)`
        })
        setImportPath('') // Clear the input since file was removed
        loadModels() // Reload to refresh Downloads list
      } else {
        setMessage({ type: 'error', text: `Failed to import model: ${errorMsg}` })
      }
    } finally {
      setImporting(false)
    }
  }

  const handleImportMultiple = async () => {
    if (selectedModels.length === 0) {
      setMessage({ type: 'error', text: 'Please select at least one model to import' })
      return
    }

    try {
      setImporting(true)
      setMessage(null)

      let imported = 0
      let upgraded = 0
      let skipped = 0
      let failed = 0
      const errors = []

      for (const modelPath of selectedModels) {
        try {
          const response = await api.importModel(modelPath, moveFile)
          if (response.data.upgraded) {
            upgraded++
          } else {
            imported++
          }
        } catch (error) {
          const errorData = error.response?.data
          // If it was a duplicate that got cleaned up, count as skipped
          if (errorData?.cleaned_up && moveFile) {
            skipped++
          } else {
            failed++
            errors.push(error.response?.data?.detail || error.message)
          }
        }
      }

      // Build summary message
      const parts = []
      if (imported > 0) parts.push(`${imported} imported`)
      if (upgraded > 0) parts.push(`${upgraded} upgraded`)
      if (skipped > 0) parts.push(`${skipped} skipped (already installed)`)
      if (failed > 0) parts.push(`${failed} failed`)

      const messageText = parts.join(', ')
      const messageType = failed > 0 ? 'warning' : 'success'

      setMessage({ type: messageType, text: messageText })
      setSelectedModels([]) // Clear selection
      setImportPath('') // Clear manual input
      loadModels() // Reload the list
    } catch (error) {
      setMessage({ type: 'error', text: `Import failed: ${error.message}` })
    } finally {
      setImporting(false)
    }
  }

  const toggleModelSelection = (modelPath) => {
    setSelectedModels(prev => {
      if (prev.includes(modelPath)) {
        return prev.filter(p => p !== modelPath)
      } else {
        return [...prev, modelPath]
      }
    })
  }

  const selectAllModels = () => {
    if (selectedModels.length === downloadsModels.length) {
      setSelectedModels([])
    } else {
      setSelectedModels(downloadsModels.map(m => m.path))
    }
  }

  const handleDownloadModel = async (model, isFromCard = false) => {
    console.log('[FRONTEND] Starting download for model:', model.name, 'id:', model.id, 'isFromCard:', isFromCard)

    try {
      // Start the download
      console.log('[FRONTEND] Calling api.downloadModel with:', { model_id: model.id, model_name: model.name })
      const response = await api.downloadModel({
        model_id: model.id,
        model_name: model.name
      })

      console.log('[FRONTEND] Download API response:', response.data)
      const downloadId = response.data.download_id

      // Initialize progress tracking
      const progressState = {
        downloadId,
        status: 'started',
        progress: 0,
        model_name: model.name
      }

      console.log('[FRONTEND] Setting initial progress state:', progressState)

      if (isFromCard) {
        setCardDownloads(prev => ({
          ...prev,
          [model.id]: progressState
        }))
      } else {
        setDownloadProgress(progressState)
      }

      setMessage({ type: 'success', text: `Download started for ${model.name}` })

      // Poll for progress updates
      let pollCount = 0
      const pollInterval = setInterval(async () => {
        pollCount++
        console.log(`[FRONTEND] Poll #${pollCount} - Checking progress for downloadId:`, downloadId)

        try {
          const progressResponse = await api.getDownloadProgress(downloadId)
          const progressData = progressResponse.data

          console.log(`[FRONTEND] Poll #${pollCount} - Progress data:`, progressData)

          const updatedProgress = {
            downloadId,
            ...progressData
          }

          if (isFromCard) {
            setCardDownloads(prev => ({
              ...prev,
              [model.id]: updatedProgress
            }))
          } else {
            setDownloadProgress(updatedProgress)
          }

          // Stop polling if completed or error
          if (progressData.status === 'completed') {
            console.log('[FRONTEND] Download completed!')
            clearInterval(pollInterval)
            setMessage({ type: 'success', text: `${model.name} downloaded successfully!` })

            if (isFromCard) {
              setCardDownloads(prev => {
                const updated = { ...prev }
                delete updated[model.id]
                return updated
              })
            } else {
              setDownloadProgress(null)
            }

            loadModels() // Reload to show the newly installed model
          } else if (progressData.status === 'error') {
            console.error('[FRONTEND] Download error:', progressData.error)
            clearInterval(pollInterval)
            setMessage({ type: 'error', text: `Download failed: ${progressData.error}` })

            if (isFromCard) {
              setCardDownloads(prev => {
                const updated = { ...prev }
                delete updated[model.id]
                return updated
              })
            } else {
              setDownloadProgress(null)
            }
          }
        } catch (error) {
          console.error('[FRONTEND] Error checking progress:', error)
          clearInterval(pollInterval)
          setMessage({ type: 'error', text: `Failed to check download progress: ${error.message}` })

          if (isFromCard) {
            setCardDownloads(prev => {
              const updated = { ...prev }
              delete updated[model.id]
              return updated
            })
          } else {
            setDownloadProgress(null)
          }
        }
      }, 1000) // Poll every second

    } catch (error) {
      console.error('[FRONTEND] Error starting download:', error)
      setMessage({ type: 'error', text: `Failed to start download: ${error.message}` })

      if (isFromCard) {
        setCardDownloads(prev => {
          const updated = { ...prev }
          delete updated[model.id]
          return updated
        })
      } else {
        setDownloadProgress(null)
      }
    }
  }

  const handleDownloadRecommendedModel = async () => {
    if (!aiRecommendation || !aiRecommendation.model_details) {
      setMessage({ type: 'error', text: 'No model recommendation available' })
      return
    }

    await handleDownloadModel(aiRecommendation.model_details, false)
  }

  const isModelInstalled = (modelName) => {
    return installedModels.some(m =>
      m.name.toLowerCase().includes(modelName.toLowerCase()) ||
      modelName.toLowerCase().includes(m.name.toLowerCase())
    )
  }

  const getStyleColor = (style) => {
    const colors = {
      'Realistic': 'text-green-400 bg-green-500/20',
      'Photorealistic': 'text-green-400 bg-green-500/20',
      'Anime': 'text-pink-400 bg-pink-500/20',
      'Artistic': 'text-purple-400 bg-purple-500/20',
      'General': 'text-blue-400 bg-blue-500/20'
    }
    return colors[style] || 'text-gray-400 bg-gray-500/20'
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="loading-spinner"></div>
      </div>
    )
  }

  return (
    <div className="max-w-7xl mx-auto p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold flex items-center gap-3">
            <FiPackage className="text-primary-500" />
            Model Browser
          </h1>
          <p className="text-gray-400 mt-1">Manage your Stable Diffusion models</p>
        </div>
        <button onClick={loadModels} className="btn-secondary">
          <FiRefreshCw className="inline mr-2" />
          Refresh
        </button>
      </div>

      {/* Messages */}
      {message && (
        <div className={`p-4 rounded-lg ${
          message.type === 'success' ? 'bg-green-500/20 text-green-300' : 'bg-red-500/20 text-red-300'
        }`}>
          {message.text}
        </div>
      )}

      {/* Models Directory */}
      <div className="card">
        <h3 className="font-semibold mb-2">Models Directory</h3>
        <p className="text-sm text-gray-400 font-mono">{modelsDirectory}</p>
      </div>

      {/* Import Model Section */}
      <div className="card bg-gradient-to-br from-blue-500/10 to-cyan-500/10 border-blue-500/30">
        <div className="flex items-start gap-3 mb-4">
          <FiUpload className="text-blue-400 w-6 h-6 flex-shrink-0 mt-1" />
          <div className="flex-1">
            <h2 className="text-xl font-bold mb-1">Import Downloaded Model</h2>
            <p className="text-sm text-gray-400">Import a .safetensors model file you've downloaded manually</p>
          </div>
        </div>

        <div className="space-y-3">
          {/* Available models in Downloads folder */}
          {downloadsModels.length > 0 && (
            <div>
              <div className="flex items-center justify-between mb-2">
                <label className="block text-sm">Models found in Downloads folder:</label>
                <button
                  onClick={selectAllModels}
                  className="text-xs text-primary-400 hover:text-primary-300 transition-colors"
                >
                  {selectedModels.length === downloadsModels.length ? 'Deselect All' : 'Select All'}
                </button>
              </div>
              <div className="space-y-2 max-h-64 overflow-y-auto">
                {downloadsModels.map((model) => {
                  const isSelected = selectedModels.includes(model.path)
                  return (
                    <div
                      key={model.path}
                      onClick={() => toggleModelSelection(model.path)}
                      className={`p-3 rounded-lg border cursor-pointer transition-all ${
                        isSelected
                          ? 'bg-blue-500/20 border-blue-500'
                          : 'bg-white/5 border-white/10 hover:bg-white/10 hover:border-white/20'
                      }`}
                    >
                      <div className="flex items-center gap-3">
                        <input
                          type="checkbox"
                          checked={isSelected}
                          onChange={() => toggleModelSelection(model.path)}
                          onClick={(e) => e.stopPropagation()}
                          className="w-4 h-4 rounded border-gray-600 bg-gray-700 text-primary-500 focus:ring-primary-500 focus:ring-2"
                        />
                        <div className="flex items-center justify-between flex-1">
                          <div className="flex-1">
                            <p className="font-semibold text-sm">{model.name}</p>
                            <p className="text-xs text-gray-500 font-mono">{model.filename}</p>
                          </div>
                          <span className="text-xs text-gray-400">{model.size}</span>
                        </div>
                      </div>
                    </div>
                  )
                })}
              </div>
              <p className="text-xs text-gray-500 mt-2">
                {selectedModels.length > 0
                  ? `${selectedModels.length} model(s) selected`
                  : 'Select models to import multiple at once'
                }
              </p>
            </div>
          )}

          <div>
            <label className="block text-sm mb-2">
              {downloadsModels.length > 0 ? 'Or enter path manually:' : 'Full path to .safetensors file:'}
            </label>
            <input
              type="text"
              value={importPath}
              onChange={(e) => setImportPath(e.target.value)}
              className="input w-full font-mono text-sm"
              placeholder={downloadsDirectory ? `${downloadsDirectory}\\model.safetensors` : "C:\\Users\\YourName\\Downloads\\model.safetensors"}
            />
          </div>

          {/* Move vs Copy option */}
          <div className="flex items-center gap-2 p-3 bg-white/5 rounded-lg">
            <input
              type="checkbox"
              id="moveFile"
              checked={moveFile}
              onChange={(e) => setMoveFile(e.target.checked)}
              className="w-4 h-4 rounded border-gray-600 bg-gray-700 text-primary-500 focus:ring-primary-500 focus:ring-2"
            />
            <label htmlFor="moveFile" className="text-sm cursor-pointer flex-1">
              Remove file from Downloads after import (recommended to keep Downloads clean)
            </label>
          </div>

          {/* Import buttons */}
          {selectedModels.length > 0 ? (
            <button
              onClick={handleImportMultiple}
              className="btn-primary w-full"
              disabled={importing}
            >
              {importing ? (
                <>
                  <div className="loading-spinner inline-block mr-2"></div>
                  Importing {selectedModels.length} model(s)...
                </>
              ) : (
                <>
                  <FiUpload className="inline mr-2" />
                  Import {selectedModels.length} Selected Model{selectedModels.length > 1 ? 's' : ''}
                </>
              )}
            </button>
          ) : (
            <button
              onClick={handleImportModel}
              className="btn-primary w-full"
              disabled={importing || !importPath.trim()}
            >
              {importing ? (
                <>
                  <div className="loading-spinner inline-block mr-2"></div>
                  Importing...
                </>
              ) : (
                <>
                  <FiUpload className="inline mr-2" />
                  {moveFile ? 'Move & Import Model' : 'Copy & Import Model'}
                </>
              )}
            </button>
          )}
        </div>
      </div>

      {/* AI Recommendation Section */}
      <div className="card bg-gradient-to-br from-purple-500/10 to-blue-500/10 border-purple-500/30">
        <div className="flex items-start gap-3 mb-4">
          <FiZap className="text-purple-400 w-6 h-6 flex-shrink-0 mt-1" />
          <div className="flex-1">
            <h2 className="text-xl font-bold mb-1">AI-Powered Model Recommendation</h2>
            <p className="text-sm text-gray-400">Let AI analyze your prompt and suggest the best model</p>
          </div>
        </div>

        <div className="space-y-3">
          <div>
            <label className="block text-sm mb-2">Enter your prompt:</label>
            <textarea
              value={promptForRecommendation}
              onChange={(e) => setPromptForRecommendation(e.target.value)}
              className="input w-full h-24 resize-none"
              placeholder="e.g., A photorealistic portrait of a woman in sunlight..."
            />
          </div>

          <button
            onClick={handleGetAiRecommendation}
            className="btn-primary w-full"
            disabled={loadingRecommendation || !promptForRecommendation.trim()}
          >
            {loadingRecommendation ? (
              <>
                <div className="loading-spinner inline-block mr-2"></div>
                Analyzing...
              </>
            ) : (
              <>
                <FiZap className="inline mr-2" />
                Get AI Recommendation
              </>
            )}
          </button>
        </div>

        {/* AI Recommendation Result */}
        {aiRecommendation && (
          <div className="mt-4 p-4 bg-white/5 rounded-lg border border-purple-500/30">
            <h3 className="font-semibold text-purple-400 mb-3">Recommendation:</h3>
            <div className="space-y-3">
              <div className="flex items-center gap-2">
                <FiStar className="text-yellow-400" />
                <span className="font-semibold">{aiRecommendation.recommended_model}</span>
                {aiRecommendation.is_installed ? (
                  <span className="text-xs px-2 py-1 bg-green-500/20 text-green-400 rounded">Installed</span>
                ) : (
                  <span className="text-xs px-2 py-1 bg-orange-500/20 text-orange-400 rounded">Not Installed</span>
                )}
              </div>
              <p className="text-sm text-gray-300">{aiRecommendation.reason}</p>
              {aiRecommendation.alternative && !aiRecommendation.is_installed && (
                <p className="text-sm text-gray-400">
                  <span className="font-semibold">Alternative:</span> {aiRecommendation.alternative}
                </p>
              )}

              {/* Download button if not installed */}
              {!aiRecommendation.is_installed && aiRecommendation.model_details && (
                <div className="mt-3">
                  <a
                    href={aiRecommendation.model_details.civitai_url}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="btn-primary w-full inline-flex items-center justify-center"
                  >
                    <FiDownload className="inline mr-2" />
                    Download {aiRecommendation.recommended_model} from CivitAI
                  </a>
                  <p className="text-xs text-gray-500 mt-2 text-center">
                    After downloading, use the Import section above to add it to your models
                  </p>
                </div>
              )}
            </div>
          </div>
        )}
      </div>

      {/* Tabs */}
      <div className="flex gap-2 border-b border-white/20">
        <button
          onClick={() => setActiveTab('recommended')}
          className={`px-4 py-2 font-medium transition-all ${
            activeTab === 'recommended'
              ? 'text-primary-400 border-b-2 border-primary-400'
              : 'text-gray-400 hover:text-white'
          }`}
        >
          <FiStar className="inline mr-2" />
          Recommended Models ({recommendedModels.length})
        </button>
        <button
          onClick={() => setActiveTab('installed')}
          className={`px-4 py-2 font-medium transition-all ${
            activeTab === 'installed'
              ? 'text-primary-400 border-b-2 border-primary-400'
              : 'text-gray-400 hover:text-white'
          }`}
        >
          <FiPackage className="inline mr-2" />
          Installed Models ({installedModels.length})
        </button>
      </div>

      {/* Recommended Models Tab */}
      {activeTab === 'recommended' && (
        <div className="grid md:grid-cols-2 gap-4">
          {recommendedModels.map((model) => {
            const installed = isModelInstalled(model.name)
            const downloading = cardDownloads[model.id]

            return (
              <div key={model.id} className="card hover:border-primary-500/50 transition-all">
                <div className="flex items-start justify-between mb-3">
                  <div>
                    <h3 className="text-lg font-bold">{model.name}</h3>
                    <p className="text-sm text-gray-400">{model.type}</p>
                  </div>
                  {installed && (
                    <span className="text-xs px-2 py-1 bg-green-500/20 text-green-400 rounded flex items-center gap-1">
                      <FiPackage className="w-3 h-3" />
                      Installed
                    </span>
                  )}
                </div>

                <p className="text-sm text-gray-300 mb-3">{model.description}</p>

                <div className="flex flex-wrap gap-2 mb-3">
                  <span className={`text-xs px-2 py-1 rounded ${getStyleColor(model.style)}`}>
                    {model.style}
                  </span>
                  <span className="text-xs px-2 py-1 bg-white/10 text-gray-300 rounded">
                    {model.size}
                  </span>
                </div>

                <div className="mb-3">
                  <p className="text-xs text-gray-400 mb-1">Best for:</p>
                  <div className="flex flex-wrap gap-1">
                    {model.recommended_for.map((use) => (
                      <span key={use} className="text-xs px-2 py-1 bg-primary-500/20 text-primary-300 rounded">
                        {use}
                      </span>
                    ))}
                  </div>
                </div>

                {model.notes && (
                  <div className="mb-3 p-2 bg-yellow-500/10 border border-yellow-500/30 rounded">
                    <p className="text-xs text-yellow-300">{model.notes}</p>
                  </div>
                )}

                {/* Action buttons */}
                <div className="flex gap-2 mt-4">
                  <a
                    href={model.civitai_url}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="btn-secondary flex-1 text-sm justify-center"
                  >
                    <FiExternalLink className="inline mr-2" />
                    View on CivitAI
                  </a>
                  {!installed && (
                    <a
                      href={model.civitai_url}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="btn-primary text-sm"
                    >
                      <FiDownload className="inline mr-2" />
                      Download
                    </a>
                  )}
                </div>
              </div>
            )
          })}
        </div>
      )}

      {/* Installed Models Tab */}
      {activeTab === 'installed' && (
        <div>
          {installedModels.length === 0 ? (
            <div className="card text-center py-12">
              <FiPackage className="w-12 h-12 mx-auto mb-4 text-gray-500" />
              <p className="text-gray-400 mb-2">No models installed yet</p>
              <p className="text-sm text-gray-500">
                Download models from the Recommended tab to get started
              </p>
            </div>
          ) : (
            <div className="space-y-3">
              {installedModels.map((model) => (
                <div key={model.filename} className="card flex items-center justify-between">
                  <div className="flex-1">
                    <h3 className="font-semibold">{model.name}</h3>
                    <div className="flex items-center gap-4 text-sm text-gray-400 mt-1">
                      <span>{model.size}</span>
                      <span className="font-mono text-xs">{model.filename}</span>
                    </div>
                  </div>
                  <button
                    onClick={() => handleDeleteModel(model.filename)}
                    className="btn-secondary text-red-400 hover:bg-red-500/20"
                  >
                    <FiTrash2 className="inline mr-2" />
                    Delete
                  </button>
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  )
}
