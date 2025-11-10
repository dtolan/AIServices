import { FiSearch, FiTrash2, FiCopy } from 'react-icons/fi'
import { useState } from 'react'
import useStore from '../store/useStore'

export default function PromptLibrary() {
  const { savedPrompts, deletePrompt } = useStore()
  const [searchQuery, setSearchQuery] = useState('')

  const filteredPrompts = savedPrompts.filter(p =>
    p.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
    p.prompt.positive_prompt.toLowerCase().includes(searchQuery.toLowerCase())
  )

  const copyPrompt = (prompt) => {
    navigator.clipboard.writeText(prompt.positive_prompt)
  }

  return (
    <div className="card">
      <div className="flex items-center justify-between mb-6">
        <h2 className="text-2xl font-bold">Prompt Library</h2>
        <div className="relative">
          <FiSearch className="absolute left-3 top-1/2 transform -translate-y-1/2 text-white/50" />
          <input
            type="text"
            placeholder="Search prompts..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="input-field pl-10 w-64"
          />
        </div>
      </div>

      {filteredPrompts.length === 0 ? (
        <div className="text-center py-12 text-white/50">
          <p className="text-lg">No saved prompts yet</p>
          <p className="text-sm">Save prompts from the gallery to build your library</p>
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {filteredPrompts.map((prompt) => (
            <div key={prompt.id} className="glass rounded-lg overflow-hidden">
              {prompt.thumbnail && (
                <div className="aspect-square bg-black/20">
                  <img
                    src={`data:image/png;base64,${prompt.thumbnail}`}
                    alt={prompt.name}
                    className="w-full h-full object-cover"
                  />
                </div>
              )}
              <div className="p-4">
                <h3 className="font-semibold mb-2">{prompt.name}</h3>
                <p className="text-xs text-white/70 line-clamp-2 mb-3">
                  {prompt.prompt.positive_prompt}
                </p>
                <div className="flex space-x-2">
                  <button
                    onClick={() => copyPrompt(prompt.prompt)}
                    className="btn-secondary flex-1 text-sm flex items-center justify-center space-x-1"
                  >
                    <FiCopy className="w-3 h-3" />
                    <span>Copy</span>
                  </button>
                  <button
                    onClick={() => deletePrompt(prompt.id)}
                    className="btn-secondary text-sm p-2"
                  >
                    <FiTrash2 className="w-4 h-4 text-red-400" />
                  </button>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
