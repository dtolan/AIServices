# SD Prompt Assistant

A self-hosted AI prompt engineering assistant for Stable Diffusion with **flexible LLM support** - use local models (Ollama) or cloud providers (Claude, Gemini) to craft better prompts and iteratively refine images.

## Features

- **Multiple LLM Providers** - Choose between Ollama (free, local), Anthropic Claude (premium quality), or Google Gemini (fast & capable)
- **Conversational prompt refinement** - Describe what you want in natural language, AI suggests optimized SD prompts
- **Automatic1111 integration** - Connects to your existing SD Web UI
- **Real-time generation** - See results immediately with side-by-side chat and image view
- **Smart parameter suggestions** - AI recommends optimal steps, CFG, samplers based on desired output
- **Prompt history** - Track your prompt evolution and iterations
- **Easy provider switching** - Change LLM providers with a single config setting

## Architecture

```
┌─────────────┐      ┌─────────────┐      ┌──────────────────────┐
│   Web UI    │─────>│   FastAPI   │─────>│  LLM Provider        │
│  (Browser)  │      │   Backend   │      │  • Ollama (local)    │
└─────────────┘      └─────────────┘      │  • Claude (cloud)    │
                            │              │  • Gemini (cloud)    │
                            │              └──────────────────────┘
                            v
                     ┌─────────────────┐
                     │  Automatic1111  │
                     │ (Stable Diffusion)│
                     └─────────────────┘
```

## Prerequisites

1. **Python 3.9+**
2. **LLM Provider** (choose one or configure multiple):
   - **Ollama** (local, free) - [Install Ollama](https://ollama.ai/)
   - **Anthropic Claude** (cloud, API key) - [Get API key](https://console.anthropic.com/)
   - **Google Gemini** (cloud, API key) - [Get API key](https://aistudio.google.com/app/apikey)
3. **Automatic1111 Stable Diffusion Web UI** - [Setup guide](https://github.com/AUTOMATIC1111/stable-diffusion-webui)

## Installation

### 1. Choose and Setup LLM Provider

#### Option A: Ollama (Local - Free)

```bash
# Install Ollama from https://ollama.ai/

# Pull a recommended model (choose one):
ollama pull llama3.2      # Recommended: Fast and smart (3B)
ollama pull llama3.1      # Alternative: More capable (8B)
ollama pull mistral       # Alternative: Great for creative tasks
```

#### Option B: Anthropic Claude (Cloud - Paid)

1. Sign up at [Anthropic Console](https://console.anthropic.com/)
2. Create an API key
3. Add to `.env`: `ANTHROPIC_API_KEY=your-key-here`

**Recommended models:**
- `claude-3-5-sonnet-20241022` - Best quality, slower
- `claude-3-5-haiku-20241022` - Fast and good quality

#### Option C: Google Gemini (Cloud - Free tier available)

1. Get API key from [Google AI Studio](https://aistudio.google.com/app/apikey)
2. Add to `.env`: `GOOGLE_API_KEY=your-key-here`

**Recommended models:**
- `gemini-1.5-flash` - Fast and capable (recommended)
- `gemini-1.5-pro` - Higher quality, slower

### 2. Start Automatic1111 with API

Launch your Stable Diffusion Web UI with API enabled:

```bash
# Windows
webui.bat --api

# Linux/Mac
./webui.sh --api
```

The API should be available at `http://localhost:7860`

### 3. Install Python Dependencies

```bash
cd AIServices
pip install -r requirements.txt
```

### 4. Configure Environment

Copy the example environment file and adjust if needed:

```bash
copy .env.example .env  # Windows
cp .env.example .env    # Linux/Mac
```

Edit `.env` to configure your chosen LLM provider:

```env
# Choose your LLM provider: "ollama", "claude", or "gemini"
LLM_PROVIDER=ollama

# Ollama Configuration (if using Ollama)
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=llama3.2:latest

# Claude Configuration (if using Claude)
ANTHROPIC_API_KEY=your-api-key-here
CLAUDE_MODEL=claude-3-5-sonnet-20241022

# Gemini Configuration (if using Gemini)
GOOGLE_API_KEY=your-api-key-here
GEMINI_MODEL=gemini-1.5-flash

# Stable Diffusion API
SD_API_URL=http://localhost:7860

# Application Configuration
APP_HOST=0.0.0.0
APP_PORT=8000
```

## Usage

### 1. Start the Backend Server

```bash
# From the AIServices directory
python -m backend.main
```

The API will start on `http://localhost:8000`

### 2. Open the Web Interface

Open [frontend/index.html](frontend/index.html) in your browser, or serve it:

```bash
# Simple HTTP server
python -m http.server 8080
# Then open http://localhost:8080/frontend/
```

### 3. Generate Images

1. Type a natural language description: "a majestic dragon flying over a medieval castle at sunset"
2. The AI will enhance your prompt with optimal SD syntax and parameters
3. Image generates automatically with explanation of prompt choices
4. Iterate by describing changes: "make it more dramatic with storm clouds"

## API Endpoints

### Health Check
```bash
GET http://localhost:8000/health
```

### Enhance Prompt Only
```bash
POST http://localhost:8000/enhance-prompt
Content-Type: application/json

{
  "user_input": "a cat wearing sunglasses",
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

## How It Works

1. **User describes desired image** in natural language
2. **LLM analyzes** the description and applies Stable Diffusion best practices (via your chosen provider)
3. **Prompt Engine** structures the enhanced prompt with:
   - Detailed descriptors and style tags
   - Quality tags (masterpiece, highly detailed, etc.)
   - Negative prompts to avoid unwanted elements
   - Optimal parameters (steps, CFG, sampler, resolution)
4. **SD generates the image** via Automatic1111 API
5. **Results displayed** with explanation of prompt choices
6. **Iterate** by providing feedback to refine the result

## LLM Provider Comparison

| Provider | Speed | Quality | Cost | Setup |
|----------|-------|---------|------|-------|
| **Ollama** | Medium | Good | Free | Local install |
| **Claude Haiku** | Very Fast | Excellent | ~$0.25/1M tokens | API key |
| **Claude Sonnet** | Fast | Outstanding | ~$3/1M tokens | API key |
| **Gemini Flash** | Very Fast | Very Good | Free tier available | API key |
| **Gemini Pro** | Medium | Excellent | ~$1.25/1M tokens | API key |

**Recommendations:**
- **Starting out?** Use Ollama (free, private, no API keys needed)
- **Best quality prompts?** Claude 3.5 Sonnet (significantly better prompt engineering)
- **Speed & cost balance?** Gemini 1.5 Flash (very fast, free tier available)
- **Privacy conscious?** Ollama (everything runs locally)

## Example Prompts

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

## Troubleshooting

### LLM Service Offline

**Ollama:**
- Ensure Ollama is running: `ollama serve`
- Check model is downloaded: `ollama list`
- Verify `OLLAMA_HOST` in `.env`

**Claude:**
- Verify API key is correct in `.env`
- Check your API key at [Anthropic Console](https://console.anthropic.com/)
- Ensure you have credits/billing set up

**Gemini:**
- Verify API key is correct in `.env`
- Check your API key at [Google AI Studio](https://aistudio.google.com/app/apikey)
- Ensure API is enabled for your project

### Stable Diffusion Offline
- Check Automatic1111 is running with `--api` flag
- Verify `SD_API_URL` in `.env`
- Test API: `curl http://localhost:7860/sdapi/v1/sd-models`

### Slow Generation
- Reduce image size (512x512 recommended for speed)
- Reduce steps (20-30 is usually sufficient)
- Use faster samplers like "Euler a"
- Ensure GPU is being used by SD
- **Try a faster LLM**: Switch to Claude Haiku or Gemini Flash

### JSON Parsing Errors
- LLM might return malformed JSON occasionally
- The app has fallback logic to handle this
- Claude and Gemini are more reliable at JSON than local models
- Try adjusting temperature in [backend/prompt_engine.py](backend/prompt_engine.py)

## Advanced Configuration

### Switching LLM Providers

Simply change `LLM_PROVIDER` in `.env`:

```env
# Switch to Claude for better prompts
LLM_PROVIDER=claude
ANTHROPIC_API_KEY=your-key-here

# Or switch to Gemini for speed
LLM_PROVIDER=gemini
GOOGLE_API_KEY=your-key-here

# Or back to Ollama for privacy
LLM_PROVIDER=ollama
```

Then restart the server. No code changes needed!

### Change LLM Model

Edit `.env`:
```env
# For Ollama
OLLAMA_MODEL=mistral:latest

# For Claude
CLAUDE_MODEL=claude-3-5-haiku-20241022

# For Gemini
GEMINI_MODEL=gemini-1.5-pro
```

### Customize System Prompts

Edit [backend/prompt_engine.py](backend/prompt_engine.py) to modify the system prompts that guide the LLM's behavior.

### Adjust Default Parameters

Edit `.env`:
```env
DEFAULT_STEPS=40
DEFAULT_CFG_SCALE=8.0
DEFAULT_WIDTH=768
DEFAULT_HEIGHT=768
```

## Development

### Project Structure

```
AIServices/
├── backend/
│   ├── main.py              # FastAPI server
│   ├── config.py            # Configuration management
│   ├── llm_service.py       # Unified LLM service (provider switcher)
│   ├── llm_providers/       # LLM provider implementations
│   │   ├── base.py          # Base provider interface
│   │   ├── ollama_provider.py
│   │   ├── claude_provider.py
│   │   └── gemini_provider.py
│   ├── sd_service.py        # Stable Diffusion API client
│   ├── prompt_engine.py     # Core prompt enhancement logic
│   └── models/
│       └── schemas.py       # Pydantic models
├── frontend/
│   └── index.html           # Web interface
├── requirements.txt         # Python dependencies
├── .env.example            # Example configuration
└── README.md               # This file
```

### Running in Development Mode

```bash
# Backend with auto-reload
python -m backend.main

# The FastAPI server will reload on code changes when DEBUG=True
```

## Future Enhancements

- [ ] Image iteration with img2img
- [ ] Prompt template library
- [ ] Style presets (anime, photorealistic, artistic, etc.)
- [ ] Batch generation with variations
- [ ] Prompt history and favorites
- [ ] Advanced parameter tuning UI
- [ ] ControlNet integration
- [ ] Multi-image comparison view

## License

MIT License - feel free to modify and use as you wish!

## Credits

- Built with [FastAPI](https://fastapi.tiangolo.com/)
- LLM providers: [Ollama](https://ollama.ai/), [Anthropic Claude](https://anthropic.com/), [Google Gemini](https://ai.google.dev/)
- Image generation via [Automatic1111 SD Web UI](https://github.com/AUTOMATIC1111/stable-diffusion-webui)

---

**Note:** When using Ollama, everything runs locally on your machine. When using Claude or Gemini, prompts are sent to their respective APIs (but your generated images stay local).
