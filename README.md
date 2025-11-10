# SD Prompt Assistant

A self-hosted AI prompt engineering assistant for Stable Diffusion with **flexible LLM support** - use local models (Ollama) or cloud providers (Claude, Gemini) to craft better prompts and iteratively refine images.

## ✨ Key Features

### Core Features
- **🎨 UI-Based Settings Management** - Configure everything from the web interface, no manual .env editing needed
- **🔄 Dual-LLM Mode** - Use different LLMs for planning (quality) vs execution (speed) tasks
- **🤖 Multiple LLM Providers** - Choose between Ollama (free, local), Anthropic Claude (premium quality), or Google Gemini (fast & capable)
- **📋 Model Dropdown Menus** - Automatically fetch and select from available models for each provider
- **💬 Conversational Prompt Refinement** - Describe what you want in natural language, AI suggests optimized SD prompts
- **❓ Interactive Question Mode** - AI asks clarifying questions to understand your vision better
- **🔌 Automatic1111 Integration** - Connects to your existing SD Web UI
- **⚡ Real-time Generation** - See results immediately with side-by-side chat and image view
- **🎯 Smart Parameter Suggestions** - AI recommends optimal steps, CFG, samplers based on desired output
- **📚 Knowledge Base** - Built-in guides and templates for Stable Diffusion
- **🔄 Prompt History** - Track your prompt evolution and iterations

### Advanced Features (NEW!)
- **🧠 Plan/Act Workflow** - AI creates detailed generation plans before executing (Phase 2)
- **🎭 Model Recommendations** - AI suggests the best SD model for your prompt
- **📊 Quality Analysis** - Get specificity scores and missing element warnings
- **💡 Parameter Reasoning** - Understand why each parameter was chosen
- **🖼️ Img2Img Support** - Transform existing images with AI-guided denoising strength
- **📥 Drag-and-Drop Image Upload** - Easy image upload for img2img transformations
- **🎨 Smart Model Detection** - Automatic detection and matching of installed SD models
- **📁 Model Browser** - Download and manage SD models from CivitAI directly in the UI

## 🏗️ Architecture

```
┌─────────────────┐      ┌─────────────────┐      ┌──────────────────────┐
│   React UI      │─────>│   FastAPI       │─────>│  LLM Provider        │
│   (Vite + TW)   │      │   Backend       │      │  • Ollama (local)    │
└─────────────────┘      └─────────────────┘      │  • Claude (cloud)    │
                                │                  │  • Gemini (cloud)    │
                                │                  └──────────────────────┘
                                v
                         ┌─────────────────┐
                         │  Automatic1111  │
                         │(Stable Diffusion)│
                         └─────────────────┘
```

## 📋 Prerequisites

1. **Python 3.9+** (tested with Python 3.10)
2. **Node.js 16+** (for frontend development)
3. **LLM Provider** (choose one or configure multiple):
   - **Ollama** (local, free) - [Install Ollama](https://ollama.ai/)
   - **Anthropic Claude** (cloud, API key) - [Get API key](https://console.anthropic.com/)
   - **Google Gemini** (cloud, API key) - [Get API key](https://aistudio.google.com/app/apikey)
4. **Automatic1111 Stable Diffusion Web UI** - [Setup guide](https://github.com/AUTOMATIC1111/stable-diffusion-webui)

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone <repository-url>
cd AIServices
```

### 2. Setup LLM Provider (Choose at least one)

#### Option A: Ollama (Local - Free)

```bash
# Install Ollama from https://ollama.ai/

# Pull recommended models:
ollama pull llama3.2:3b    # Fast, lightweight (recommended for execution)
ollama pull llama3.1:8b    # More capable (recommended for planning)
ollama pull qwen3:8b       # Alternative: Great for creative tasks
```

#### Option B: Anthropic Claude (Cloud - Paid)

1. Sign up at [Anthropic Console](https://console.anthropic.com/)
2. Create an API key
3. You'll add this in the Settings UI later

**Available models:**
- `claude-sonnet-4-5-20250929` - Best coding model, strongest for complex agents (recommended for planning)
- `claude-haiku-4-5-20251001` - Fast and cost-effective (recommended for execution)
- `claude-opus-4-1-20250805` - Highest quality for agentic tasks
- `claude-3-7-sonnet-20250219` - Hybrid AI reasoning model (legacy)
- `claude-3-5-haiku-20241022` - Legacy fast version

#### Option C: Google Gemini (Cloud - Free tier available)

1. Get API key from [Google AI Studio](https://aistudio.google.com/app/apikey)
2. You'll add this in the Settings UI later

**Available models (as of 2025):**
- `gemini-2.5-flash` - Fast and capable (recommended for execution)
- `gemini-2.5-pro` - Higher quality (recommended for planning)
- `gemini-flash-latest` - Always uses latest flash model
- `gemini-pro-latest` - Always uses latest pro model
- `gemini-2.0-flash-exp` - Experimental fast model

### 3. Start Automatic1111 with API

Launch your Stable Diffusion Web UI with API enabled:

```bash
# Windows
webui.bat --api

# Linux/Mac
./webui.sh --api
```

The API should be available at `http://localhost:7860`

### 4. Install Backend Dependencies

```bash
cd AIServices
pip install -r requirements.txt
```

### 5. Create .env File

Copy the example environment file:

```bash
# Windows
copy .env.example .env

# Linux/Mac
cp .env.example .env
```

**Basic .env configuration** (you can adjust most settings via the UI later):

```env
# ===========================================
# DUAL-LLM CONFIGURATION
# ===========================================
USE_DUAL_LLM=True

# Planning LLM (for initial prompt engineering)
PLANNING_LLM_PROVIDER=gemini
PLANNING_GEMINI_MODEL=gemini-2.5-pro

# Execution LLM (for quick iterations)
EXECUTION_LLM_PROVIDER=gemini
EXECUTION_GEMINI_MODEL=gemini-2.5-flash

# ===========================================
# SINGLE LLM CONFIGURATION (when USE_DUAL_LLM=false)
# ===========================================
LLM_PROVIDER=ollama
OLLAMA_MODEL=llama3.2:latest

# ===========================================
# API KEYS (Add your keys here or via Settings UI)
# ===========================================
# Anthropic Claude Configuration
ANTHROPIC_API_KEY=your-api-key-here

# Google Gemini Configuration
GOOGLE_API_KEY=your-api-key-here

# ===========================================
# STABLE DIFFUSION
# ===========================================
SD_API_URL=http://localhost:7860
SD_API_TIMEOUT=300

# ===========================================
# APPLICATION
# ===========================================
APP_HOST=0.0.0.0
APP_PORT=8000
DEBUG=True

# Generation defaults
DEFAULT_STEPS=30
DEFAULT_CFG_SCALE=7.0
DEFAULT_WIDTH=512
DEFAULT_HEIGHT=512
DEFAULT_SAMPLER=DPM++ 2M Karras
```

### 6. Install Frontend Dependencies

```bash
cd frontend
npm install
```

### 7. Start the Application

**Terminal 1 - Backend:**
```bash
# From AIServices directory
python -m backend.main
```

**Terminal 2 - Frontend:**
```bash
# From AIServices/frontend directory
npm run dev
```

The application will be available at:
- Frontend: `http://localhost:3000`
- Backend API: `http://localhost:8000`
- API Docs: `http://localhost:8000/docs`

### 8. Configure via Settings UI

1. Open `http://localhost:3000` in your browser
2. Click the **Settings** tab
3. Configure your LLM providers:
   - Add API keys for Claude/Gemini
   - Configure Ollama host if needed
   - Select models from dropdown menus (click refresh button to fetch available models)
4. Test connections to verify setup
5. Save settings

## 💡 Usage

### Basic Workflow

1. **Navigate to the Home tab**
2. **Type a natural language description:**
   - "a majestic dragon flying over a medieval castle at sunset"
3. **Choose mode:**
   - **Ask Questions** - AI asks clarifying questions first
   - **Generate Directly** - Skip questions and generate immediately
4. **AI enhances your prompt** with optimal SD syntax and parameters
5. **Image generates automatically** with explanation of prompt choices
6. **Iterate** by describing changes: "make it more dramatic with storm clouds"

### Interactive Question Mode

The AI can ask clarifying questions to better understand your vision:

1. Type your basic idea: "a fantasy character"
2. Click **"Ask Questions"**
3. Answer questions like:
   - "What species or type of character?"
   - "What setting or environment?"
   - "What mood or atmosphere?"
4. Get a highly refined prompt based on your answers

### Dual-LLM Mode Benefits

When enabled (default), the application uses two LLMs:

- **Planning LLM** - High-quality model for initial prompt engineering and question generation
- **Execution LLM** - Fast model for iterations and refinements

This gives you the best of both worlds: quality where it matters, speed for quick iterations.

**Recommended combinations:**
- Planning: `gemini-2.5-pro` or `claude-sonnet-4-5-20250929`
- Execution: `gemini-2.5-flash` or `claude-haiku-4-5-20251001`

## 🎨 Example Prompts

**User Input:**
> "a cozy coffee shop"

**AI Enhanced Prompt:**
```
Positive: cozy coffee shop interior, warm lighting, wooden furniture,
plants, books on shelves, steam from coffee cup, morning sunlight through
windows, inviting atmosphere, detailed textures, masterpiece, best quality,
highly detailed, 8k, photorealistic

Negative: blurry, low quality, distorted, ugly, bad anatomy, dark, gloomy,
empty, crowded

Settings: 30 steps, CFG 7.5, 512x512, DPM++ 2M Karras
```

## 📚 Knowledge Base

The application includes a built-in knowledge base with:

- **Style Guides** - Best practices for different art styles
- **Quality Tags** - How to use quality modifiers
- **Negative Prompts** - Common negative prompts to avoid issues
- **Parameter Guides** - Understanding steps, CFG, samplers
- **Prompt Templates** - Ready-to-use templates for common scenarios

Access via the **Knowledge Base** tab in the UI.

## ⚙️ Settings Management

### UI Settings (Recommended)

Configure everything from the Settings tab:

1. **LLM Configuration**
   - Toggle Dual-LLM mode
   - Select providers (Ollama/Claude/Gemini)
   - Choose models from dropdown menus
   - Test connections

2. **API Keys**
   - Add Claude API key
   - Add Gemini API key
   - Keys are masked in UI for security

3. **Stable Diffusion**
   - Configure SD API URL
   - Set API timeout

4. **Generation Defaults**
   - Default steps, CFG scale
   - Default resolution
   - Default sampler

Changes are saved to `.env` file automatically.

### Manual .env Configuration

You can also edit `.env` directly if preferred. The backend reads settings on startup.

**Important:** After changing `.env` manually, restart the backend for changes to take effect.

## 🔧 API Endpoints

### Health Check
```bash
GET http://localhost:8000/health
```

### Get Settings
```bash
GET http://localhost:8000/settings
```

### Update Settings
```bash
POST http://localhost:8000/settings/update
Content-Type: application/json

{
  "use_dual_llm": true,
  "planning_llm_provider": "gemini",
  "execution_llm_provider": "gemini"
}
```

### Get Available Models
```bash
POST http://localhost:8000/settings/available-models
Content-Type: application/json

{
  "provider": "gemini"
}
```

### Test Connection
```bash
POST http://localhost:8000/settings/test-connection
Content-Type: application/json

{
  "service_type": "gemini",
  "config": {
    "api_key": "your-key-here"
  }
}
```

### Generate with Questions
```bash
POST http://localhost:8000/interactive/questions
Content-Type: application/json

{
  "user_input": "a fantasy character",
  "conversation_history": []
}
```

### Generate Image
```bash
POST http://localhost:8000/generate
Content-Type: application/json

{
  "user_input": "a futuristic city at night",
  "conversation_history": []
}
```

### Iterate on Image
```bash
POST http://localhost:8000/iterate
Content-Type: application/json

{
  "previous_prompt": { ... },
  "previous_image_base64": "...",
  "user_feedback": "make it more colorful"
}
```

## 🐛 Troubleshooting

### Model Not Found Error (404)

**Error:** `Error code: 404 - {'type': 'error', 'error': {'type': 'not_found_error', 'message': 'model: ...'}`

**Solutions:**

**For Gemini:**
- Old: `gemini-1.5-pro` → New: `gemini-2.5-pro`
- Old: `gemini-1.5-flash` → New: `gemini-2.5-flash`

**For Claude:**
- Old: `claude-3-5-sonnet-20241022` → New: `claude-sonnet-4-5-20250929`
- Old: `claude-3-5-haiku-20241022` → New: `claude-haiku-4-5-20251001`

Update your model names in [.env](.env) or use the dropdown menus in Settings to select from current available models.

### API Key Not Working

**Claude:**
- Verify API key at [Anthropic Console](https://console.anthropic.com/)
- Ensure billing is set up
- Check for typos in the key

**Gemini:**
- Verify API key at [Google AI Studio](https://aistudio.google.com/app/apikey)
- Ensure API is enabled for your project
- Try regenerating the key if issues persist

### Can't Fetch Available Models

**Issue:** "Failed to fetch gemini models: 400: Valid API key required"

**Solutions:**
1. Make sure you've added your API key in Settings
2. Click "Test Connection" to verify the key works
3. Restart the backend after adding keys to `.env` manually
4. Check backend logs for detailed error messages

### Ollama Connection Issues

- Ensure Ollama is running: `ollama serve`
- Check model is downloaded: `ollama list`
- Verify `OLLAMA_HOST` is correct (default: `http://localhost:11434`)
- Try restarting Ollama service

### Stable Diffusion Connection Issues

- Ensure Automatic1111 is running with `--api` flag
- Verify SD API URL: `http://localhost:7860`
- Test API manually: `curl http://localhost:7860/sdapi/v1/sd-models`
- Check firewall settings

### Frontend Not Loading

- Ensure both backend and frontend are running
- Check frontend is on port 3000: `npm run dev`
- Clear browser cache and reload
- Check browser console for errors (F12)

### Multiple Backend Instances

If you see intermittent errors:
1. Check for multiple backend processes: `tasklist | findstr python` (Windows) or `ps aux | grep python` (Linux/Mac)
2. Kill old processes
3. Start only one backend instance

## 🔬 Advanced Configuration

### Custom System Prompts

Edit `backend/prompt_engine.py` to modify how the AI generates prompts.

### Change Models Dynamically

Use the Settings UI to switch models without restarting (for most settings).

For `.env` changes, restart the backend:
```bash
# Stop backend (Ctrl+C)
python -m backend.main
```

### Adjust LLM Temperature

Edit `backend/llm_providers/` files to adjust temperature and other generation parameters.

### Add Custom Knowledge Base Content

Add markdown files to:
- `backend/knowledge_base/guides/` - For guides
- `backend/knowledge_base/templates/` - For templates

## 📁 Project Structure

```
AIServices/
├── backend/
│   ├── main.py                    # FastAPI server & endpoints
│   ├── config.py                  # Configuration management with pydantic-settings
│   ├── llm_service.py            # Unified LLM service (provider switcher)
│   ├── llm_providers/            # LLM provider implementations
│   │   ├── base.py               # Base provider interface
│   │   ├── ollama_provider.py   # Ollama integration
│   │   ├── claude_provider.py   # Claude integration
│   │   └── gemini_provider.py   # Gemini integration
│   ├── sd_service.py             # Stable Diffusion API client
│   ├── prompt_engine.py          # Core prompt enhancement logic
│   ├── interactive_prompter.py  # Question-based prompt creation
│   ├── knowledge_base/           # Built-in guides and templates
│   │   ├── guides/
│   │   └── templates/
│   └── models/
│       └── schemas.py            # Pydantic models for API
├── frontend/
│   ├── src/
│   │   ├── components/          # React components
│   │   │   ├── Home.jsx         # Main generation interface
│   │   │   ├── Settings.jsx     # Settings management UI
│   │   │   └── KnowledgeBase.jsx # Knowledge base viewer
│   │   ├── api/
│   │   │   └── client.js        # API client with axios
│   │   ├── store/
│   │   │   └── useStore.js      # Zustand state management
│   │   ├── App.jsx              # Main app component
│   │   ├── main.jsx             # Entry point
│   │   └── index.css            # Tailwind CSS styles
│   ├── index.html               # HTML entry point
│   ├── package.json             # Frontend dependencies
│   ├── vite.config.js           # Vite configuration
│   └── tailwind.config.js       # Tailwind configuration
├── .env.example                 # Example environment configuration
├── .env                         # Your environment configuration (git-ignored)
├── requirements.txt             # Python dependencies
├── .gitignore                   # Git ignore rules
└── README.md                    # This file
```

## 🚀 LLM Provider Comparison

| Provider | Speed | Quality | Cost | Privacy | Setup |
|----------|-------|---------|------|---------|-------|
| **Ollama** | Medium | Good | Free | Private | Local install |
| **Claude Haiku 4.5** | Very Fast | Excellent | $1/$5 per 1M tokens | Cloud | API key |
| **Claude Sonnet 4.5** | Fast | Outstanding | $3/$15 per 1M tokens | Cloud | API key |
| **Gemini Flash** | Very Fast | Very Good | Free tier available | Cloud | API key |
| **Gemini Pro** | Medium | Excellent | ~$1.25/1M tokens | Cloud | API key |

**Recommendations:**

**Best Quality Prompts:**
- Planning: `claude-sonnet-4-5-20250929` or `gemini-2.5-pro`
- Execution: `claude-haiku-4-5-20251001` or `gemini-2.5-flash`

**Best Speed & Cost:**
- Both: `gemini-2.5-flash` (very fast, free tier available)

**Privacy Conscious:**
- Both: Ollama (everything runs locally, no API calls)

**Starting Out:**
- Planning: `gemini-2.5-pro` (free tier)
- Execution: `gemini-2.5-flash` (free tier)

## 🔄 Recent Updates

### Latest Changes (Current Session)

1. **Img2Img (Image-to-Image) Support** ✨ NEW!
   - Complete Plan/Act workflow for image transformations
   - AI-guided denoising strength recommendations with reasoning
   - Drag-and-drop image upload with preview
   - Base64 image handling and validation
   - Intelligent transformation analysis based on user intent
   - Side-by-side source/result comparison in image gallery

2. **ImageUpload Component**
   - Reusable drag-and-drop file upload component
   - Visual feedback for drag states
   - Image preview with hover controls
   - File type validation
   - Clean state management

3. **Enhanced ChatPanel Integration**
   - Seamless img2img mode toggle in chat interface
   - Conditional workflow routing (txt2img vs img2img)
   - State cleanup after generation
   - Source image tracking with filename display

4. **UI-Based Settings Management**
   - Complete settings interface in the web UI
   - No need to manually edit .env files
   - Real-time validation and testing

5. **Model Dropdown Menus**
   - Automatically fetch available models from each provider
   - Easy model selection with refresh buttons
   - See all available models before choosing

6. **Updated Gemini Model Names**
   - Now using latest Gemini 2.5 models
   - `gemini-2.5-pro` for quality
   - `gemini-2.5-flash` for speed

7. **Fixed Multiple Backend Instances**
   - Resolved issue with conflicting backend processes
   - Better startup and shutdown handling

8. **Improved API Key Handling**
   - Backend now correctly uses API keys from settings
   - Proper masking in UI for security

## 📝 Development

### Running in Development Mode

**Backend with auto-reload:**
```bash
python -m backend.main
# Uvicorn will reload on code changes when DEBUG=True
```

**Frontend with hot-reload:**
```bash
cd frontend
npm run dev
# Vite will hot-reload on file changes
```

### Building for Production

**Frontend:**
```bash
cd frontend
npm run build
# Outputs to frontend/dist/
```

Serve the built frontend with any static file server.

**Backend:**
Already production-ready. Set `DEBUG=False` in `.env` for production.

## 🎯 Future Enhancements

### Completed ✅
- [x] Image iteration with img2img (Phase 2 - Complete)
- [x] Plan/Act workflow with AI reasoning (Phase 2 - Complete)
- [x] Model recommendations and management (Phase 2 - Complete)

### In Progress 🚧
- [ ] Advanced img2img controls
  - [ ] Inpainting with mask editor
  - [ ] Sketch-guided img2img
  - [ ] Multi-step img2img workflows
  - [ ] Batch img2img transformations

### Planned 📋
- [ ] Prompt template library with sharing
- [ ] Style presets (anime, photorealistic, artistic, etc.)
- [ ] Batch generation with variations
- [ ] Advanced parameter tuning UI
- [ ] ControlNet integration
  - [ ] Canny edge detection
  - [ ] Depth map control
  - [ ] Pose guidance
  - [ ] Line art control
- [ ] Multi-image comparison view
- [ ] Prompt remixing from generated images
- [ ] Integration with other SD implementations (ComfyUI, InvokeAI)
- [ ] Video generation support
- [ ] Upscaling and enhancement workflows
- [ ] Generation history with search and filtering
- [ ] Export/import project configurations

## 📄 License

MIT License - feel free to modify and use as you wish!

## 🙏 Credits

- Built with [FastAPI](https://fastapi.tiangolo.com/) and [React](https://react.dev/)
- UI with [Tailwind CSS](https://tailwindcss.com/) and [Vite](https://vitejs.dev/)
- State management with [Zustand](https://github.com/pmndrs/zustand)
- LLM providers: [Ollama](https://ollama.ai/), [Anthropic Claude](https://anthropic.com/), [Google Gemini](https://ai.google.dev/)
- Image generation via [Automatic1111 SD Web UI](https://github.com/AUTOMATIC1111/stable-diffusion-webui)

---

**Privacy Note:** When using Ollama, everything runs locally on your machine. When using Claude or Gemini, prompts are sent to their respective APIs (but your generated images stay local on your machine).

**Important:** Make sure to keep your `.env` file secure and never commit it to version control. It contains sensitive API keys.
