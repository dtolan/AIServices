# 🚀 Quickstart Guide

Get up and running with SD Prompt Assistant in 5 minutes!

## Prerequisites

- Python 3.9+
- Node.js 18+ (for modern UI)
- [Ollama](https://ollama.ai/) OR Claude/Gemini API keys
- [Automatic1111 Stable Diffusion](https://github.com/AUTOMATIC1111/stable-diffusion-webui)

## Fast Setup (5 Steps)

### 1. Clone & Install Backend

```bash
cd AIServices
pip install -r requirements.txt
```

### 2. Setup LLM (Choose One)

**Option A: Local (Ollama - Free)**
```bash
# Install from https://ollama.ai/
ollama pull llama3.2
```

**Option B: Cloud (Claude/Gemini)**
```bash
# Get API keys:
# Claude: https://console.anthropic.com/
# Gemini: https://aistudio.google.com/app/apikey
```

### 3. Configure Environment

```bash
cp .env.example .env
# Edit .env with your preferences
```

**Minimal .env for Ollama:**
```env
LLM_PROVIDER=ollama
OLLAMA_AUTO_CONFIGURE=true
SD_API_URL=http://localhost:7860
```

**For Claude + Gemini (Recommended):**
```env
USE_DUAL_LLM=true
PLANNING_LLM_PROVIDER=claude
PLANNING_CLAUDE_MODEL=claude-3-5-sonnet-20241022
EXECUTION_LLM_PROVIDER=gemini
EXECUTION_GEMINI_MODEL=gemini-1.5-flash

ANTHROPIC_API_KEY=sk-ant-your-key-here
GOOGLE_API_KEY=your-google-key-here
```

### 4. Start Services

**Terminal 1 - Stable Diffusion:**
```bash
cd your-sd-webui-folder
webui.bat --api  # Windows
./webui.sh --api # Linux/Mac
```

**Terminal 2 - Backend:**
```bash
cd AIServices
python -m backend.main
```

**Terminal 3 - Modern UI (Optional but Recommended):**
```bash
cd AIServices/frontend
npm install
npm run dev
```

### 5. Start Creating!

Open http://localhost:3000 (modern UI) or http://localhost:8000 (basic UI)

## First Generation

1. Type: **"a cozy coffee shop"**
2. Click **Generate**
3. Watch the AI:
   - Ask clarifying questions (if enabled)
   - Create an optimized prompt
   - Generate your image

## Quick Tips

### Enable Dual-LLM (Recommended)
```env
USE_DUAL_LLM=true
PLANNING_LLM_PROVIDER=claude  # Best prompts
EXECUTION_LLM_PROVIDER=gemini # Fast iterations
```

**Why?** Claude creates better initial prompts, Gemini handles fast iterations.

### Add Custom Knowledge
```bash
# Create your style guide
echo "# My Style
Cyberpunk, neon lights, rain..." > knowledge_base/my_style.md
```

Restart server, and it'll use your knowledge!

### Save Successful Prompts
Click the ❤️ button to save prompts to your library for reuse.

## Common Issues

### LLM Service Offline
```bash
# For Ollama:
ollama serve
ollama pull llama3.2

# For Claude/Gemini:
# Check API keys in .env
```

### SD Not Connected
```bash
# Make sure running with --api flag
webui.bat --api

# Test: curl http://localhost:7860/sdapi/v1/sd-models
```

### No GPU Detected
```bash
# Check hardware recommendations:
curl http://localhost:8000/hardware
```

## Next Steps

1. **Try Interactive Mode**: Enable "Ask questions" for better prompts
2. **Explore Knowledge Base**: Add custom style guides
3. **Save Your Best Prompts**: Build your library
4. **Experiment with Dual-LLM**: Try different provider combinations

## Project Structure

```
AIServices/
├── backend/           # FastAPI server
├── frontend/          # Modern React UI
├── knowledge_base/    # Style guides (auto-created)
├── prompts.json       # Saved prompts (auto-created)
└── .env              # Your configuration
```

## Documentation

- **Dual-LLM**: See [DUAL_LLM_GUIDE.md](DUAL_LLM_GUIDE.md)
- **Knowledge Base**: See [KNOWLEDGE_BASE_GUIDE.md](KNOWLEDGE_BASE_GUIDE.md)
- **Full README**: See [README.md](README.md)

## Support

- Issues: https://github.com/your-repo/issues
- Discussions: https://github.com/your-repo/discussions

---

**That's it!** You're ready to create amazing AI-generated images! 🎨
