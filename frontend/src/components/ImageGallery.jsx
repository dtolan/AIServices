import { FiImage, FiDownload, FiRefreshCw, FiHeart } from 'react-icons/fi'
import useStore from '../store/useStore'

export default function ImageGallery() {
  const { generations, currentGeneration, savePrompt } = useStore()

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
        <div className="flex-1 flex flex-col">
          <div className="flex-1 relative bg-black/20 rounded-lg overflow-hidden mb-4">
            <img
              src={`data:image/png;base64,${currentGeneration.image_base64}`}
              alt="Generated"
              className="w-full h-full object-contain"
            />
          </div>

          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <button onClick={() => handleDownload(currentGeneration)} className="btn-secondary flex items-center space-x-2">
                <FiDownload className="w-4 h-4" />
                <span>Download</span>
              </button>
              <button onClick={() => handleSavePrompt(currentGeneration)} className="btn-secondary flex items-center space-x-2">
                <FiHeart className="w-4 h-4" />
                <span>Save</span>
              </button>
            </div>

            <div className="glass rounded-lg p-3 text-sm">
              <p className="text-white/70 mb-1">Positive Prompt:</p>
              <p className="text-xs">{currentGeneration.prompt_used.positive_prompt}</p>
            </div>
          </div>
        </div>
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
