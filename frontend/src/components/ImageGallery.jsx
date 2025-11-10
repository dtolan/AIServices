import { FiImage, FiDownload, FiRefreshCw, FiHeart, FiInfo, FiEdit3 } from 'react-icons/fi'
import { useState } from 'react'
import useStore from '../store/useStore'
import GenerationInfoModal from './GenerationInfoModal'

export default function ImageGallery({ onIterate }) {
  const { generations, currentGeneration, savePrompt } = useStore()
  const [showInfoModal, setShowInfoModal] = useState(false)

  const handleDownload = (generation) => {
    const link = document.createElement('a')
    link.href = `data:image/png;base64,${generation.image_base64}`
    link.download = `sd-${generation.id}.png`
    link.click()
  }

  const handleSavePrompt = (generation) => {
    savePrompt({
      name: generation.userInput?.substring(0, 50) || 'Saved Prompt',
      prompt: generation.prompt_used,
      thumbnail: generation.image_base64
    })
  }

  return (
    <div className="card flex flex-col h-[calc(100vh-12rem)]">
      <h2 className="text-xl font-bold mb-4 flex items-center">
        <FiImage className="mr-2" />
        Generated Images
      </h2>

      {!currentGeneration && generations.length === 0 && (
        <div className="flex-1 flex items-center justify-center text-white/50">
          <div className="text-center">
            <FiImage className="w-20 h-20 mx-auto mb-4 opacity-30" />
            <p className="text-lg">No images yet</p>
            <p className="text-sm">Start a conversation to generate</p>
          </div>
        </div>
      )}

      {currentGeneration && (
        <div className="flex-1 flex flex-col overflow-hidden">
          <div className="flex-1 relative bg-black/20 rounded-lg overflow-hidden mb-3 min-h-0">
            <img
              src={`data:image/png;base64,${currentGeneration.image_base64}`}
              alt="Generated"
              className="w-full h-full object-contain"
            />
          </div>

          <div className="space-y-2 flex-shrink-0">
            <div className="grid grid-cols-2 gap-2">
              <button onClick={() => setShowInfoModal(true)} className="btn-primary flex items-center justify-center space-x-2">
                <FiInfo className="w-4 h-4" />
                <span>Info</span>
              </button>
              <button onClick={() => onIterate && onIterate(currentGeneration)} className="btn-primary flex items-center justify-center space-x-2 bg-gradient-to-r from-purple-500 to-blue-500 hover:from-purple-600 hover:to-blue-600">
                <FiEdit3 className="w-4 h-4" />
                <span>Iterate</span>
              </button>
            </div>
            <div className="grid grid-cols-2 gap-2">
              <button onClick={() => handleDownload(currentGeneration)} className="btn-secondary flex items-center justify-center space-x-2">
                <FiDownload className="w-4 h-4" />
                <span>Download</span>
              </button>
              <button onClick={() => handleSavePrompt(currentGeneration)} className="btn-secondary flex items-center justify-center space-x-2">
                <FiHeart className="w-4 h-4" />
                <span>Save</span>
              </button>
            </div>
          </div>
        </div>
      )}

      {showInfoModal && currentGeneration && (
        <GenerationInfoModal
          generation={currentGeneration}
          onClose={() => setShowInfoModal(false)}
        />
      )}

      {generations.length > 1 && (
        <div className="mt-4 pt-4 border-t border-white/20">
          <p className="text-sm text-white/70 mb-2">Recent ({generations.length})</p>
          <div className="grid grid-cols-4 gap-2">
            {generations.slice(0, 8).map((gen) => (
              <button
                key={gen.id}
                onClick={() => useStore.getState().setCurrentGeneration(gen)}
                className="aspect-square bg-black/20 rounded-lg overflow-hidden hover:ring-2 hover:ring-primary-500 transition-all"
              >
                <img
                  src={`data:image/png;base64,${gen.image_base64}`}
                  alt="Thumbnail"
                  className="w-full h-full object-cover"
                />
              </button>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}
