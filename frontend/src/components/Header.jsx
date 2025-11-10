import { FiZap, FiBook, FiSettings, FiPackage } from 'react-icons/fi'

export default function Header({ activeTab, onTabChange }) {
  const tabs = [
    { id: 'generate', name: 'Generate', icon: FiZap },
    { id: 'library', name: 'Library', icon: FiBook },
    { id: 'models', name: 'Models', icon: FiPackage },
    { id: 'settings', name: 'Settings', icon: FiSettings },
  ]

  return (
    <header className="glass border-b border-white/20">
      <div className="container mx-auto px-4">
        <div className="flex items-center justify-between h-16">
          <div className="flex items-center space-x-3">
            <div className="w-10 h-10 bg-gradient-to-br from-primary-500 to-purple-500 rounded-lg flex items-center justify-center">
              <FiZap className="w-6 h-6" />
            </div>
            <div>
              <h1 className="text-xl font-bold">SD Prompt Assistant</h1>
              <p className="text-xs text-white/60">AI-Powered Prompt Engineering</p>
            </div>
          </div>

          <nav className="flex space-x-2">
            {tabs.map((tab) => {
              const Icon = tab.icon
              const isActive = activeTab === tab.id

              return (
                <button
                  key={tab.id}
                  onClick={() => onTabChange(tab.id)}
                  className={`
                    flex items-center space-x-2 px-4 py-2 rounded-lg transition-all duration-200
                    ${isActive
                      ? 'bg-white/20 text-white'
                      : 'text-white/60 hover:bg-white/10 hover:text-white'}
                  `}
                >
                  <Icon className="w-5 h-5" />
                  <span className="font-medium">{tab.name}</span>
                </button>
              )
            })}
          </nav>
        </div>
      </div>
    </header>
  )
}
