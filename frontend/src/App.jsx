import { useState, useEffect, useRef } from 'react'
import useStore from './store/useStore'
import { api } from './api/client'
import Header from './components/Header'
import ChatPanel from './components/ChatPanel'
import ImageGallery from './components/ImageGallery'
import PromptLibrary from './components/PromptLibrary'
import ModelBrowser from './components/ModelBrowser'
import SetupWizard from './components/SetupWizard'
import StatusBar from './components/StatusBar'
import Settings from './components/Settings'

function App() {
  const [activeTab, setActiveTab] = useState('generate') // 'generate', 'library', 'models', 'settings'
  const [showSetup, setShowSetup] = useState(false)
  const chatPanelRef = useRef(null)
  const { setProviderInfo, setKnowledgeBase } = useStore()

  useEffect(() => {
    // Load initial data
    loadAppData()

    // Check if first run
    const hasCompletedSetup = localStorage.getItem('setup_completed')
    if (!hasCompletedSetup) {
      setShowSetup(true)
    }
  }, [])

  const loadAppData = async () => {
    try {
      const [health, kb] = await Promise.all([
        api.getHealth(),
        api.getKnowledgeBase()
      ])

      setProviderInfo(health.data)
      setKnowledgeBase(kb.data)
    } catch (error) {
      console.error('Failed to load app data:', error)
    }
  }

  const handleSetupComplete = () => {
    setShowSetup(false)
    localStorage.setItem('setup_completed', 'true')
    loadAppData()
  }

  const handleIterate = (generation) => {
    // Call the ChatPanel's iterate method
    if (chatPanelRef.current && chatPanelRef.current.startIteration) {
      chatPanelRef.current.startIteration(generation)
    }
  }

  if (showSetup) {
    return <SetupWizard onComplete={handleSetupComplete} />
  }

  return (
    <div className="min-h-screen flex flex-col">
      <Header activeTab={activeTab} onTabChange={setActiveTab} />

      <div className="flex-1 container mx-auto px-4 py-6">
        {activeTab === 'generate' && (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 h-full">
            <ChatPanel ref={chatPanelRef} onSwitchTab={setActiveTab} />
            <ImageGallery onIterate={handleIterate} />
          </div>
        )}

        {activeTab === 'library' && (
          <PromptLibrary />
        )}

        {activeTab === 'models' && (
          <ModelBrowser />
        )}

        {activeTab === 'settings' && (
          <Settings />
        )}
      </div>

      <StatusBar />
    </div>
  )
}

export default App
