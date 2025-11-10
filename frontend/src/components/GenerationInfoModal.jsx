import { FiX, FiCopy, FiCheck, FiDownload } from 'react-icons/fi'
import { useState } from 'react'

export default function GenerationInfoModal({ generation, onClose }) {
  const [copiedField, setCopiedField] = useState(null)

  const copyToClipboard = (text, field) => {
    navigator.clipboard.writeText(text)
    setCopiedField(field)
    setTimeout(() => setCopiedField(null), 2000)
  }

  const downloadMetadata = () => {
    const metadata = {
      timestamp: generation.timestamp,
      user_input: generation.userInput,
      positive_prompt: generation.prompt_used?.positive_prompt,
      negative_prompt: generation.prompt_used?.negative_prompt,
      parameters: {
        steps: generation.prompt_used?.steps,
        cfg_scale: generation.prompt_used?.cfg_scale,
        width: generation.prompt_used?.width,
        height: generation.prompt_used?.height,
        sampler: generation.prompt_used?.sampler_name,
        seed: generation.seed_used
      },
      generation_time: generation.generation_time,
      explanation: generation.llm_explanation
    }

    const blob = new Blob([JSON.stringify(metadata, null, 2)], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const link = document.createElement('a')
    link.href = url
    link.download = `generation-${generation.id}-metadata.json`
    link.click()
    URL.revokeObjectURL(url)
  }

  const CopyButton = ({ text, field }) => (
    <button
      onClick={() => copyToClipboard(text, field)}
      className="btn-secondary text-xs px-2 py-1 flex items-center gap-1"
      title="Copy to clipboard"
    >
      {copiedField === field ? (
        <>
          <FiCheck className="w-3 h-3" />
          Copied
        </>
      ) : (
        <>
          <FiCopy className="w-3 h-3" />
          Copy
        </>
      )}
    </button>
  )

  return (
    <div className="fixed inset-0 bg-black/70 flex items-center justify-center z-50 p-4">
      <div className="bg-gradient-to-br from-gray-900 to-gray-800 rounded-lg shadow-2xl border border-white/20 max-w-4xl w-full max-h-[90vh] overflow-hidden flex flex-col">
        {/* Header */}
        <div className="flex items-center justify-between p-6 border-b border-white/20">
          <h2 className="text-2xl font-bold">Generation Details</h2>
          <button
            onClick={onClose}
            className="btn-secondary p-2 hover:bg-white/20"
          >
            <FiX className="w-5 h-5" />
          </button>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-y-auto p-6 space-y-6">
          {/* User Input */}
          <div className="card bg-white/5">
            <div className="flex items-center justify-between mb-2">
              <h3 className="font-semibold text-sm text-white/70">Original Request</h3>
            </div>
            <p className="text-sm">{generation.userInput || 'N/A'}</p>
          </div>

          {/* AI Explanation */}
          {generation.llm_explanation && (
            <div className="card bg-gradient-to-br from-blue-500/10 to-purple-500/10 border-blue-500/30">
              <div className="flex items-center justify-between mb-2">
                <h3 className="font-semibold text-sm text-blue-300">AI Analysis</h3>
              </div>
              <p className="text-sm text-white/80">{generation.llm_explanation}</p>
            </div>
          )}

          {/* Positive Prompt */}
          <div className="card bg-white/5">
            <div className="flex items-center justify-between mb-2">
              <h3 className="font-semibold text-sm text-green-300">Positive Prompt</h3>
              <CopyButton text={generation.prompt_used?.positive_prompt} field="positive" />
            </div>
            <p className="text-sm font-mono bg-black/30 p-3 rounded">
              {generation.prompt_used?.positive_prompt || 'N/A'}
            </p>
          </div>

          {/* Negative Prompt */}
          <div className="card bg-white/5">
            <div className="flex items-center justify-between mb-2">
              <h3 className="font-semibold text-sm text-red-300">Negative Prompt</h3>
              <CopyButton text={generation.prompt_used?.negative_prompt} field="negative" />
            </div>
            <p className="text-sm font-mono bg-black/30 p-3 rounded">
              {generation.prompt_used?.negative_prompt || 'None'}
            </p>
          </div>

          {/* Parameters Grid */}
          <div className="card bg-white/5">
            <h3 className="font-semibold text-sm text-white/70 mb-3">Generation Parameters</h3>
            <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
              <div className="p-3 bg-black/30 rounded">
                <div className="text-xs text-white/50">Steps</div>
                <div className="text-lg font-bold">{generation.prompt_used?.steps || 'N/A'}</div>
              </div>
              <div className="p-3 bg-black/30 rounded">
                <div className="text-xs text-white/50">CFG Scale</div>
                <div className="text-lg font-bold">{generation.prompt_used?.cfg_scale || 'N/A'}</div>
              </div>
              <div className="p-3 bg-black/30 rounded">
                <div className="text-xs text-white/50">Resolution</div>
                <div className="text-lg font-bold">
                  {generation.prompt_used?.width}x{generation.prompt_used?.height}
                </div>
              </div>
              <div className="p-3 bg-black/30 rounded">
                <div className="text-xs text-white/50">Sampler</div>
                <div className="text-sm font-bold">{generation.prompt_used?.sampler_name || 'N/A'}</div>
              </div>
              <div className="p-3 bg-black/30 rounded">
                <div className="text-xs text-white/50">Seed</div>
                <div className="text-sm font-mono">{generation.seed_used || 'N/A'}</div>
              </div>
              <div className="p-3 bg-black/30 rounded">
                <div className="text-xs text-white/50">Generation Time</div>
                <div className="text-lg font-bold">{generation.generation_time?.toFixed(2)}s</div>
              </div>
            </div>
          </div>

          {/* Metadata */}
          <div className="card bg-white/5">
            <h3 className="font-semibold text-sm text-white/70 mb-3">Metadata</h3>
            <div className="space-y-2 text-sm">
              <div className="flex justify-between">
                <span className="text-white/50">Generation ID:</span>
                <span className="font-mono">{generation.id}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-white/50">Timestamp:</span>
                <span>{new Date(generation.timestamp).toLocaleString()}</span>
              </div>
            </div>
          </div>
        </div>

        {/* Footer Actions */}
        <div className="p-6 border-t border-white/20 flex gap-3">
          <button
            onClick={downloadMetadata}
            className="btn-secondary flex items-center gap-2"
          >
            <FiDownload className="w-4 h-4" />
            Download Metadata (JSON)
          </button>
          <button
            onClick={onClose}
            className="btn-primary ml-auto"
          >
            Close
          </button>
        </div>
      </div>
    </div>
  )
}
