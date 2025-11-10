import { useState, useRef, useEffect, forwardRef, useImperativeHandle } from 'react'
import { FiSend, FiLoader, FiMessageSquare, FiCheckCircle, FiEdit3, FiZap, FiX, FiDownload, FiImage } from 'react-icons/fi'
import useStore from '../store/useStore'
import { api } from '../api/client'
import ImageUpload from './ImageUpload'

const ChatPanel = forwardRef(({ onSwitchTab }, ref) => {
  const [input, setInput] = useState('')
  const [generationPlan, setGenerationPlan] = useState(null) // The PLAN from backend
  const [planningMode, setPlanningMode] = useState(true) // true = PLAN mode, false = ACT mode
  const [selectedModel, setSelectedModel] = useState(null) // User's model override
  const [installedModels, setInstalledModels] = useState([])
  const [iteratingFrom, setIteratingFrom] = useState(null) // The generation we're iterating from
  const [currentActivity, setCurrentActivity] = useState(null) // 'planning', 'refining', 'generating', 'iterating'

  // Img2Img state
  const [img2imgMode, setImg2imgMode] = useState(false)
  const [sourceImage, setSourceImage] = useState(null)
  const [sourceImageName, setSourceImageName] = useState('')

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

  useEffect(() => {
    // Load installed models for dropdown
    loadInstalledModels()
  }, [])

  const loadInstalledModels = async () => {
    try {
      const response = await api.getInstalledModels()
      setInstalledModels(response.data)
    } catch (error) {
      console.error('Failed to load installed models:', error)
    }
  }

  // Expose methods to parent via ref
  useImperativeHandle(ref, () => ({
    startIteration: (generation) => {
      setIteratingFrom(generation)
      setInput(`Refine this: ${generation.userInput}`)
      addMessage({
        role: 'system',
        content: `Iterating from previous generation. Describe how you'd like to refine it.`
      })
    }
  }))

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  const detectGoCommand = (text) => {
    // Check for #go trigger (case insensitive, can be anywhere in text)
    return /#go\b/i.test(text)
  }

  const handleSend = async () => {
    if (!input.trim() || isGenerating) return

    const userMessage = { role: 'user', content: input }
    const hasGoCommand = detectGoCommand(input)

    // Remove #go from display text
    const displayText = input.replace(/#go\b/gi, '').trim()
    const displayMessage = { ...userMessage, content: displayText || input }

    addMessage(displayMessage)
    setInput('')
    setIsGenerating(true)

    try {
      // Check if we're iterating from a previous generation
      if (iteratingFrom) {
        // ITERATE: Use the /iterate endpoint
        setCurrentActivity('iterating')
        const response = await api.iterate({
          previous_prompt: iteratingFrom.prompt_used,
          previous_image_base64: iteratingFrom.image_base64,
          user_feedback: displayText || input
        })

        const generation = {
          id: Date.now(),
          timestamp: new Date().toISOString(),
          userInput: displayText || input,
          ...response.data,
          iteratedFrom: iteratingFrom.id  // Track iteration history
        }

        addGeneration(generation)
        setCurrentGeneration(generation)

        addMessage({
          role: 'assistant',
          content: response.data.llm_explanation,
          generation
        })

        // Clear iteration mode
        setIteratingFrom(null)
      } else if (planningMode && !generationPlan) {
        // PLAN PHASE: Create generation plan (or img2img plan)
        setCurrentActivity('planning')

        let response, plan
        if (img2imgMode && sourceImage) {
          // IMG2IMG PLAN
          response = await api.planImg2Img({
            user_input: displayText || input,
            init_image_base64: sourceImage,
            conversation_history: conversation
          })
        } else {
          // TXT2IMG PLAN
          response = await api.planGeneration({
            user_input: displayText || input,
            conversation_history: conversation
          })
        }

        plan = response.data
        setGenerationPlan(plan)

        // Set default model selection to recommended model
        setSelectedModel(plan.model_recommendation.recommended_model_name)

        // Add plan message
        addMessage({
          role: 'assistant',
          content: 'plan',
          plan: plan
        })

        // If user included #go, automatically proceed to ACT phase
        if (hasGoCommand) {
          console.log('[PLAN/ACT] User included #go, proceeding to generation...')
          setTimeout(() => executeGeneration(plan), 500)
        } else {
          setPlanningMode(false) // Move to ACT mode (waiting for user confirmation)
        }
      } else if (!planningMode && generationPlan && hasGoCommand) {
        // ACT PHASE: Execute the plan
        setCurrentActivity('generating')
        await executeGeneration(generationPlan)
      } else if (!planningMode && generationPlan) {
        // User is refining the plan with additional input
        // Re-run planning with the refinement feedback
        setCurrentActivity('refining')
        const refinedPrompt = `${generationPlan.user_input}. ${displayText || input}`

        const response = await api.planGeneration({
          user_input: refinedPrompt,
          conversation_history: conversation
        })

        const plan = response.data
        setGenerationPlan(plan)

        // Update model selection to new recommendation
        setSelectedModel(plan.model_recommendation.recommended_model_name)

        // Add updated plan message
        addMessage({
          role: 'assistant',
          content: 'plan',
          plan: plan
        })
      } else {
        // Fallback to direct generation (legacy behavior)
        await generateImageDirect(displayText || input)
      }
    } catch (error) {
      console.error('Error:', error)
      addMessage({
        role: 'system',
        content: `Error: ${error.response?.data?.detail || error.message}`
      })
    } finally {
      setIsGenerating(false)
      setCurrentActivity(null)
    }
  }

  const executeGeneration = async (plan) => {
    setIsGenerating(true)
    try {
      let response

      if (img2imgMode && sourceImage) {
        // IMG2IMG EXECUTION
        response = await api.executeImg2Img({
          plan: plan,
          init_image_base64: sourceImage,
          model_override: selectedModel !== plan.model_recommendation.recommended_model_name ? selectedModel : null
        })
      } else {
        // TXT2IMG EXECUTION
        response = await api.executeGeneration({
          plan: plan,
          model_override: selectedModel !== plan.model_recommendation.recommended_model_name ? selectedModel : null
        })
      }

      const generation = {
        id: Date.now(),
        timestamp: new Date().toISOString(),
        userInput: plan.user_input,
        ...response.data,
        isImg2Img: img2imgMode,
        sourceImageName: sourceImageName
      }

      addGeneration(generation)
      setCurrentGeneration(generation)

      addMessage({
        role: 'assistant',
        content: response.data.llm_explanation,
        generation
      })

      // Reset for next generation
      setGenerationPlan(null)
      setPlanningMode(true)

      // Reset img2img mode if it was used
      if (img2imgMode) {
        setImg2imgMode(false)
        setSourceImage(null)
        setSourceImageName('')
      }
      setSelectedModel(null)
    } catch (error) {
      throw error
    } finally {
      setIsGenerating(false)
    }
  }

  const generateImageDirect = async (userInput) => {
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

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  const handleQuickGo = () => {
    if (generationPlan && !planningMode) {
      executeGeneration(generationPlan)
    }
  }

  return (
    <div className="card flex flex-col h-[calc(100vh-12rem)]">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-xl font-bold flex items-center">
          <FiMessageSquare className="mr-2" />
          {planningMode ? 'Plan Mode' : 'Ready to Generate'}
        </h2>
        {!planningMode && generationPlan && (
          <span className="text-xs px-3 py-1 bg-green-500/20 text-green-400 rounded-full border border-green-500/30">
            Plan Ready - Type #go or click Generate
          </span>
        )}
      </div>

      {/* Iteration Mode Indicator */}
      {iteratingFrom && (
        <div className="mb-4 bg-gradient-to-r from-purple-500/20 to-blue-500/20 border border-purple-500 rounded-lg p-3 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <FiEdit3 className="text-purple-400" />
            <div>
              <div className="text-sm font-semibold text-purple-300">Iterating from previous generation</div>
              <div className="text-xs text-white/60">Describe how you'd like to refine it</div>
            </div>
          </div>
          <button
            onClick={() => setIteratingFrom(null)}
            className="btn-secondary text-xs px-2 py-1 hover:bg-white/20"
            title="Cancel iteration"
          >
            <FiX className="inline mr-1" />
            Cancel
          </button>
        </div>
      )}

      {/* Img2Img Mode Toggle & Upload */}
      <div className="mb-4">
        <div className="flex items-center justify-between mb-3">
          <button
            onClick={() => {
              const newMode = !img2imgMode
              setImg2imgMode(newMode)
              if (!newMode) {
                setSourceImage(null)
                setSourceImageName('')
              }
            }}
            className={`btn-secondary flex items-center space-x-2 ${img2imgMode ? 'bg-primary-500/20 border-primary-500' : ''}`}
          >
            <FiImage className="w-4 h-4" />
            <span>{img2imgMode ? 'Img2Img Mode (ON)' : 'Enable Img2Img'}</span>
          </button>
          {img2imgMode && sourceImage && (
            <span className="text-xs text-green-400">Source image loaded: {sourceImageName}</span>
          )}
        </div>

        {img2imgMode && (
          <ImageUpload
            selectedImage={sourceImage}
            onImageSelect={(base64, name) => {
              setSourceImage(base64)
              setSourceImageName(name)
            }}
            onClear={() => {
              setSourceImage(null)
              setSourceImageName('')
            }}
          />
        )}
      </div>

      <div className="flex-1 overflow-y-auto space-y-4 mb-4">
        {conversation.length === 0 && (
          <div className="text-center py-12 text-white/50">
            <FiMessageSquare className="w-16 h-16 mx-auto mb-4 opacity-30" />
            <p className="text-lg">Describe the image you want to create</p>
            <p className="text-sm">I'll analyze and create a detailed plan first</p>
            <p className="text-xs mt-2 text-white/30">Add #go to your message to generate immediately</p>
          </div>
        )}

        {conversation.map((message, index) => (
          <div
            key={index}
            className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}
          >
            {message.role === 'assistant' && message.content === 'plan' && message.plan ? (
              // PLAN DISPLAY
              <div className="max-w-[90%] rounded-lg px-4 py-3 bg-gradient-to-br from-purple-500/10 to-blue-500/10 border border-purple-500/30">
                <div className="flex items-center gap-2 mb-3">
                  <FiCheckCircle className="text-purple-400" />
                  <h3 className="font-bold text-purple-300">Generation Plan Ready</h3>
                </div>

                {/* Model Recommendation */}
                <div className="mb-4 p-3 bg-white/5 rounded-lg">
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-sm font-semibold text-white/70">Recommended Model:</span>
                    {message.plan.model_recommendation.is_installed ? (
                      <span className="text-xs px-2 py-0.5 bg-green-500/20 text-green-400 rounded border border-green-500/30">
                        Installed
                      </span>
                    ) : (
                      <span className="text-xs px-2 py-0.5 bg-yellow-500/20 text-yellow-400 rounded border border-yellow-500/30">
                        Not Installed
                      </span>
                    )}
                  </div>
                  <p className="text-lg font-bold text-primary-300 mb-1">
                    {message.plan.model_recommendation.recommended_model_name}
                  </p>
                  <p className="text-xs text-white/60 mb-2">{message.plan.model_recommendation.reason}</p>

                  {!message.plan.model_recommendation.is_installed && (
                    <button
                      onClick={() => onSwitchTab && onSwitchTab('models')}
                      className="btn-secondary text-xs px-3 py-1.5 flex items-center gap-2 bg-yellow-500/20 hover:bg-yellow-500/30 border-yellow-500/50 mt-2"
                    >
                      <FiDownload className="w-3 h-3" />
                      Download Model
                    </button>
                  )}

                  {/* Model Selector Dropdown */}
                  {installedModels.length > 0 && (
                    <div className="mt-3">
                      <label className="text-xs text-white/50 block mb-1">Override model (optional):</label>
                      <select
                        value={selectedModel || message.plan.model_recommendation.recommended_model_name}
                        onChange={(e) => setSelectedModel(e.target.value)}
                        className="w-full bg-white/10 border border-white/20 rounded px-2 py-1 text-sm focus:outline-none focus:border-primary-500"
                      >
                        <option value={message.plan.model_recommendation.recommended_model_name}>
                          {message.plan.model_recommendation.recommended_model_name} (Recommended)
                        </option>
                        {installedModels.map(model => (
                          <option key={model.name} value={model.name}>
                            {model.name}
                          </option>
                        ))}
                      </select>
                    </div>
                  )}
                </div>

                {/* Enhanced Prompt Preview */}
                <div className="mb-4">
                  <div className="text-xs font-semibold text-white/50 mb-1">Enhanced Prompt:</div>
                  <div className="p-2 bg-black/20 rounded text-xs font-mono text-white/80 max-h-24 overflow-y-auto">
                    {message.plan.enhanced_prompt.positive_prompt}
                  </div>
                  <div className="text-xs font-semibold text-white/50 mt-2 mb-1">Negative Prompt:</div>
                  <div className="p-2 bg-black/20 rounded text-xs font-mono text-white/60 max-h-20 overflow-y-auto">
                    {message.plan.enhanced_prompt.negative_prompt}
                  </div>
                </div>

                {/* Parameters */}
                <div className="grid grid-cols-2 gap-2 mb-4">
                  <div className="p-2 bg-white/5 rounded">
                    <div className="text-xs text-white/50">Steps</div>
                    <div className="font-bold">{message.plan.enhanced_prompt.steps}</div>
                    <div className="text-xs text-white/40">{message.plan.parameter_reasoning.steps_reason}</div>
                  </div>
                  <div className="p-2 bg-white/5 rounded">
                    <div className="text-xs text-white/50">CFG Scale</div>
                    <div className="font-bold">{message.plan.enhanced_prompt.cfg_scale}</div>
                    <div className="text-xs text-white/40">{message.plan.parameter_reasoning.cfg_reason}</div>
                  </div>
                  <div className="p-2 bg-white/5 rounded">
                    <div className="text-xs text-white/50">Resolution</div>
                    <div className="font-bold">{message.plan.enhanced_prompt.width}x{message.plan.enhanced_prompt.height}</div>
                    <div className="text-xs text-white/40">{message.plan.parameter_reasoning.aspect_ratio}</div>
                  </div>
                  <div className="p-2 bg-white/5 rounded">
                    <div className="text-xs text-white/50">Sampler</div>
                    <div className="font-bold text-xs">{message.plan.enhanced_prompt.sampler_name}</div>
                    <div className="text-xs text-white/40 truncate">{message.plan.parameter_reasoning.sampler_reason}</div>
                  </div>
                </div>

                {/* Quality Analysis */}
                <div className="mb-4 p-3 bg-white/5 rounded-lg">
                  <div className="text-xs font-semibold text-white/70 mb-2">Quality Analysis:</div>
                  <div className="space-y-1">
                    <div className="flex items-center justify-between text-xs">
                      <span className="text-white/60">Specificity:</span>
                      <div className="flex items-center gap-2">
                        <div className="w-24 h-1.5 bg-black/30 rounded-full overflow-hidden">
                          <div
                            className="h-full bg-gradient-to-r from-yellow-500 to-green-500"
                            style={{ width: `${message.plan.quality_analysis.specificity_score * 100}%` }}
                          ></div>
                        </div>
                        <span className="text-white/80 font-mono">{(message.plan.quality_analysis.specificity_score * 100).toFixed(0)}%</span>
                      </div>
                    </div>
                    {message.plan.quality_analysis.strengths.length > 0 && (
                      <div className="text-xs">
                        <span className="text-green-400">✓ </span>
                        <span className="text-white/60">{message.plan.quality_analysis.strengths.join(', ')}</span>
                      </div>
                    )}
                    {message.plan.quality_analysis.missing_elements.length > 0 && (
                      <div className="text-xs">
                        <span className="text-yellow-400">⚠ </span>
                        <span className="text-white/60">Consider adding: {message.plan.quality_analysis.missing_elements.join(', ')}</span>
                      </div>
                    )}
                  </div>
                </div>

                {/* Tips */}
                {message.plan.tips && message.plan.tips.length > 0 && (
                  <div className="mb-4">
                    <div className="text-xs font-semibold text-white/50 mb-1">Tips:</div>
                    <ul className="list-disc list-inside text-xs text-white/70 space-y-1">
                      {message.plan.tips.map((tip, i) => (
                        <li key={i}>{tip}</li>
                      ))}
                    </ul>
                  </div>
                )}

                {/* Action */}
                <div className="flex items-center gap-2 pt-3 border-t border-white/10">
                  <button
                    onClick={handleQuickGo}
                    className="btn-primary flex-1 text-sm"
                    disabled={isGenerating}
                  >
                    {isGenerating ? (
                      <>
                        <FiLoader className="inline animate-spin mr-2" />
                        Generating...
                      </>
                    ) : (
                      <>
                        <FiZap className="inline mr-2" />
                        Generate Image
                      </>
                    )}
                  </button>
                  <span className="text-xs text-white/40">or type #go</span>
                </div>
              </div>
            ) : (
              // REGULAR MESSAGE
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
            )}
          </div>
        ))}

        {isGenerating && (
          <div className="flex justify-start">
            <div className="bg-white/10 rounded-lg px-4 py-2 flex items-center space-x-2">
              <FiLoader className="w-4 h-4 animate-spin" />
              <span className="text-white/70">
                {currentActivity === 'planning' && 'Creating plan...'}
                {currentActivity === 'refining' && 'Refining plan...'}
                {currentActivity === 'generating' && 'Generating image...'}
                {currentActivity === 'iterating' && 'Iterating on image...'}
                {!currentActivity && 'Processing...'}
              </span>
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
          placeholder={
            planningMode
              ? "Describe the image you want (add #go to generate immediately)..."
              : "Refine your prompt or type #go to generate..."
          }
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
})

export default ChatPanel
