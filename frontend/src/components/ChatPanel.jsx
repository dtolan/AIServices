import { useState, useRef, useEffect } from 'react'
import { FiSend, FiLoader, FiMessageSquare } from 'react-icons/fi'
import useStore from '../store/useStore'
import { api } from '../api/client'

export default function ChatPanel() {
  const [input, setInput] = useState('')
  const [isInteractiveMode, setIsInteractiveMode] = useState(true)
  const messagesEndRef = useRef(null)

  const {
    conversation,
    addMessage,
    isGenerating,
    setIsGenerating,
    addGeneration,
    setCurrentGeneration
  } = useStore()

  useEffect(() => {
    scrollToBottom()
  }, [conversation])

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  const handleSend = async () => {
    if (!input.trim() || isGenerating) return

    const userMessage = { role: 'user', content: input }
    addMessage(userMessage)
    setInput('')
    setIsGenerating(true)

    try {
      if (isInteractiveMode && conversation.length === 0) {
        // First message - ask questions
        const response = await api.askQuestions({
          user_input: input,
          conversation_history: []
        })

        if (response.data.needs_clarification) {
          const questionsText = response.data.questions.join('\n')
          addMessage({
            role: 'assistant',
            content: `I need more details:\n${questionsText}`,
            questions: response.data.questions
          })
        } else {
          // No questions needed, generate directly
          await generateImage(input)
        }
      } else if (conversation.length > 0 && conversation[conversation.length - 2]?.questions) {
        // Answering questions - generate from conversation
        const fullConversation = [...conversation, userMessage]
        const response = await api.generateFromConversation({
          conversation_history: fullConversation
        })

        await generateWithPrompt(response.data)
      } else {
        // Direct generation
        await generateImage(input)
      }
    } catch (error) {
      console.error('Error:', error)
      addMessage({
        role: 'system',
        content: `Error: ${error.response?.data?.detail || error.message}`
      })
    } finally {
      setIsGenerating(false)
    }
  }

  const generateImage = async (userInput) => {
    try {
      const response = await api.generate({
        user_input: userInput,
        conversation_history: conversation
      })

      const generation = {
        id: Date.now(),
        timestamp: new Date().toISOString(),
        userInput,
        ...response.data
      }

      addGeneration(generation)
      setCurrentGeneration(generation)

      addMessage({
        role: 'assistant',
        content: response.data.llm_explanation,
        generation
      })
    } catch (error) {
      throw error
    }
  }

  const generateWithPrompt = async (promptData) => {
    try {
      const response = await api.generate({
        user_input: promptData.enhanced_prompt.positive_prompt,
        skip_questions: true
      })

      const generation = {
        id: Date.now(),
        timestamp: new Date().toISOString(),
        ...response.data
      }

      addGeneration(generation)
      setCurrentGeneration(generation)

      addMessage({
        role: 'assistant',
        content: promptData.explanation,
        generation
      })
    } catch (error) {
      throw error
    }
  }

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  return (
    <div className="card flex flex-col h-[calc(100vh-12rem)]">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-xl font-bold flex items-center">
          <FiMessageSquare className="mr-2" />
          Chat
        </h2>
        <label className="flex items-center space-x-2 text-sm">
          <input
            type="checkbox"
            checked={isInteractiveMode}
            onChange={(e) => setIsInteractiveMode(e.target.checked)}
            className="rounded"
          />
          <span className="text-white/70">Ask questions</span>
        </label>
      </div>

      <div className="flex-1 overflow-y-auto space-y-4 mb-4">
        {conversation.length === 0 && (
          <div className="text-center py-12 text-white/50">
            <FiMessageSquare className="w-16 h-16 mx-auto mb-4 opacity-30" />
            <p className="text-lg">Describe the image you want to create</p>
            <p className="text-sm">I'll help you craft the perfect prompt</p>
          </div>
        )}

        {conversation.map((message, index) => (
          <div
            key={index}
            className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}
          >
            <div
              className={`max-w-[80%] rounded-lg px-4 py-2 ${
                message.role === 'user'
                  ? 'bg-gradient-to-r from-primary-500 to-purple-500'
                  : message.role === 'system'
                  ? 'bg-yellow-500/20 border border-yellow-500/50'
                  : 'bg-white/10'
              }`}
            >
              <p className="whitespace-pre-wrap">{message.content}</p>
              {message.generation && (
                <div className="mt-2 pt-2 border-t border-white/20 text-xs text-white/70">
                  <p>Seed: {message.generation.seed_used}</p>
                  <p>Time: {message.generation.generation_time.toFixed(2)}s</p>
                </div>
              )}
            </div>
          </div>
        ))}

        {isGenerating && (
          <div className="flex justify-start">
            <div className="bg-white/10 rounded-lg px-4 py-2 flex items-center space-x-2">
              <FiLoader className="w-4 h-4 animate-spin" />
              <span className="text-white/70">Generating...</span>
            </div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      <div className="flex space-x-2">
        <textarea
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyPress={handleKeyPress}
          placeholder="Describe the image you want..."
          className="input-field flex-1 resize-none"
          rows="3"
          disabled={isGenerating}
        />
        <button
          onClick={handleSend}
          disabled={!input.trim() || isGenerating}
          className="btn-primary px-6"
        >
          {isGenerating ? (
            <FiLoader className="w-5 h-5 animate-spin" />
          ) : (
            <FiSend className="w-5 h-5" />
          )}
        </button>
      </div>
    </div>
  )
}
