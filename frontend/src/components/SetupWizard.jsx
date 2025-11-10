import { useState } from 'react'
import { FiCheck, FiLoader } from 'react-icons/fi'
import { api } from '../api/client'

export default function SetupWizard({ onComplete }) {
  const [step, setStep] = useState(0)
  const [checking, setChecking] = useState(false)
  const [status, setStatus] = useState({})

  const steps = [
    { title: 'Welcome', desc: 'Setup your AI prompt assistant' },
    { title: 'Check Services', desc: 'Verify LLM and SD are running' },
    { title: 'Configure', desc: 'Choose your preferences' },
    { title: 'Ready', desc: 'Start creating!' }
  ]

  const checkServices = async () => {
    setChecking(true)
    try {
      const [health, hardware] = await Promise.all([
        api.getHealth(),
        api.getHardware()
      ])
      setStatus({ health: health.data, hardware: hardware.data })
      setStep(2)
    } catch (error) {
      setStatus({ error: error.message })
    } finally {
      setChecking(false)
    }
  }

  return (
    <div className="min-h-screen flex items-center justify-center p-4">
      <div className="card max-w-2xl w-full">
        <div className="mb-8">
          <div className="flex items-center justify-between mb-4">
            {steps.map((s, i) => (
              <div key={i} className="flex items-center">
                <div className={`w-8 h-8 rounded-full flex items-center justify-center ${
                  i <= step ? 'bg-primary-500' : 'bg-white/10'
                }`}>
                  {i < step ? <FiCheck /> : i + 1}
                </div>
                {i < steps.length - 1 && (
                  <div className={`w-12 h-0.5 ${i < step ? 'bg-primary-500' : 'bg-white/10'}`} />
                )}
              </div>
            ))}
          </div>
          <h2 className="text-2xl font-bold">{steps[step].title}</h2>
          <p className="text-white/70">{steps[step].desc}</p>
        </div>

        {step === 0 && (
          <div className="space-y-4">
            <p>Welcome to SD Prompt Assistant! This wizard will help you get started.</p>
            <p className="text-white/70 text-sm">Make sure you have:</p>
            <ul className="list-disc list-inside space-y-2 text-sm text-white/70">
              <li>Ollama or Cloud LLM configured</li>
              <li>Stable Diffusion (Automatic1111) running with --api</li>
            </ul>
            <button onClick={() => setStep(1)} className="btn-primary w-full mt-6">
              Get Started
            </button>
          </div>
        )}

        {step === 1 && (
          <div className="space-y-4">
            <p>Let's check if your services are configured correctly.</p>
            <button onClick={checkServices} disabled={checking} className="btn-primary w-full">
              {checking ? <FiLoader className="w-5 h-5 animate-spin mx-auto" /> : 'Check Services'}
            </button>
            {status.error && (
              <div className="bg-red-500/20 border border-red-500 rounded-lg p-4">
                <p className="text-red-200">{status.error}</p>
                <p className="text-xs text-red-300 mt-2">Make sure services are running and try again</p>
              </div>
            )}
          </div>
        )}

        {step === 2 && (
          <div className="space-y-4">
            <div className="glass rounded-lg p-4">
              <p className="font-semibold mb-2">Service Status</p>
              <div className="space-y-2 text-sm">
                <div className="flex justify-between">
                  <span>LLM:</span>
                  <span className={status.health?.llm === 'healthy' ? 'text-green-400' : 'text-red-400'}>
                    {status.health?.llm || 'Unknown'}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span>Stable Diffusion:</span>
                  <span className={status.health?.sd === 'healthy' ? 'text-green-400' : 'text-red-400'}>
                    {status.health?.sd || 'Unknown'}
                  </span>
                </div>
              </div>
            </div>
            <button onClick={() => setStep(3)} className="btn-primary w-full">
              Continue
            </button>
          </div>
        )}

        {step === 3 && (
          <div className="space-y-4">
            <p className="text-lg">You're all set! 🎉</p>
            <p className="text-white/70">Start creating amazing images with AI-powered prompts.</p>
            <button onClick={onComplete} className="btn-primary w-full">
              Start Creating
            </button>
          </div>
        )}
      </div>
    </div>
  )
}
