import { useEffect } from 'react'
import { FiCpu, FiZap } from 'react-icons/fi'
import useStore from '../store/useStore'
import { api } from '../api/client'

export default function StatusBar() {
  const { providerInfo } = useStore()

  useEffect(() => {
    const loadProviderInfo = async () => {
      try {
        const { data } = await api.getHealth()
        useStore.getState().setProviderInfo(data)
      } catch (error) {
        console.error('Failed to load provider info:', error)
      }
    }

    loadProviderInfo()
    const interval = setInterval(loadProviderInfo, 30000)
    return () => clearInterval(interval)
  }, [])

  if (!providerInfo) return null

  return (
    <div className="glass border-t border-white/20 py-2">
      <div className="container mx-auto px-4">
        <div className="flex items-center justify-between text-xs text-white/70">
          <div className="flex items-center space-x-4">
            <div className="flex items-center space-x-2">
              <FiCpu className="w-4 h-4" />
              <span>
                {providerInfo.mode === 'dual'
                  ? `Planning: ${providerInfo.planning_model} | Execution: ${providerInfo.execution_model}`
                  : providerInfo.model}
              </span>
            </div>
          </div>
          <div className="flex items-center space-x-2">
            <div className={`w-2 h-2 rounded-full ${
              providerInfo.llm === 'healthy' ? 'bg-green-500' : 'bg-red-500'
            }`} />
            <span>LLM {providerInfo.llm}</span>
            <div className={`w-2 h-2 rounded-full ml-4 ${
              providerInfo.sd === 'healthy' ? 'bg-green-500' : 'bg-red-500'
            }`} />
            <span>SD {providerInfo.sd}</span>
          </div>
        </div>
      </div>
    </div>
  )
}
