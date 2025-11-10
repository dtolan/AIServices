# AIServices Project State

**Last Updated:** 2025-11-10
**Current Phase:** Phase 2 Complete - Img2Img Implementation
**Status:** Ready for deployment and continued development

---

## 📊 Project Overview

AIServices is a full-stack web application that combines LLM-powered prompt engineering with Stable Diffusion image generation. The project uses a Plan/Act workflow where an LLM analyzes user requests, creates detailed generation plans with reasoning, and then executes them through Stable Diffusion.

### Tech Stack
- **Backend:** Python 3.10, FastAPI, Pydantic, httpx
- **Frontend:** React 18, Vite, Tailwind CSS, Zustand
- **LLM Providers:** Anthropic Claude, Google Gemini, Ollama
- **Image Generation:** Stable Diffusion (via Automatic1111 Web UI API)

---

## ✅ Completed Features

### Core Infrastructure
- [x] FastAPI backend with async support
- [x] React frontend with modern tooling (Vite)
- [x] Zustand state management
- [x] Environment-based configuration with Pydantic
- [x] Comprehensive error handling and logging
- [x] API client abstraction layer

### LLM Integration (Phase 1)
- [x] Multi-provider support (Claude, Gemini, Ollama)
- [x] Dual-LLM mode (Planning + Execution LLMs)
- [x] Settings UI with real-time validation
- [x] API key management with masking
- [x] Model selection dropdowns with auto-fetch
- [x] Provider-specific model configurations

### Stable Diffusion Integration (Phase 1)
- [x] SD Web UI API integration
- [x] Text-to-image generation
- [x] Image-to-image transformation
- [x] Sampler and model listing
- [x] Parameter customization (steps, CFG, dimensions)
- [x] Generation progress tracking

### Plan/Act Workflow (Phase 2)
- [x] Two-phase generation system (Plan → Execute)
- [x] Comprehensive generation planning with reasoning
- [x] Quality analysis and specificity scoring
- [x] Missing element detection
- [x] Parameter reasoning explanations
- [x] Model recommendations with justification
- [x] Intelligent negative prompt generation
- [x] Tips and suggestions for best results

### Model Management (Phase 2)
- [x] SD model browser with CivitAI integration
- [x] Model recommendation system
- [x] Installed model detection
- [x] Model directory management
- [x] Download progress tracking
- [x] Model import from filesystem
- [x] Model deletion with safety checks
- [x] Fuzzy model name matching

### Img2Img Support (Phase 2) ✨ LATEST
- [x] Complete img2img Plan/Act workflow
- [x] AI-guided denoising strength recommendations
- [x] Denoising reasoning explanations
- [x] Drag-and-drop image upload component
- [x] Image preview with hover controls
- [x] Base64 image encoding/handling
- [x] File type validation
- [x] ChatPanel img2img mode toggle
- [x] Conditional workflow routing (txt2img vs img2img)
- [x] Source image tracking and display
- [x] Side-by-side comparison in image gallery

### User Interface
- [x] Modern chat-based interface
- [x] Image gallery with generation history
- [x] Settings panel with live validation
- [x] Model management UI
- [x] Generation plan review interface
- [x] Model override capability
- [x] Responsive design with Tailwind CSS
- [x] Dark theme with gradient accents

---

## 🏗️ Project Architecture

### Backend Structure
```
backend/
├── main.py                 # FastAPI app and route definitions
├── config.py               # Pydantic settings and environment config
├── prompt_engine.py        # LLM prompt engineering and planning logic
├── sd_service.py           # Stable Diffusion API integration
├── model_manager.py        # SD model detection and management
├── civitai_service.py      # CivitAI API integration
├── llm_providers/          # LLM provider implementations
│   ├── base.py            # Abstract base provider
│   ├── ollama.py          # Ollama local LLM
│   ├── anthropic.py       # Claude API
│   └── gemini.py          # Google Gemini API
└── models/
    └── schemas.py         # Pydantic models for API contracts
```

### Frontend Structure
```
frontend/
├── src/
│   ├── App.jsx                    # Main app component
│   ├── api/
│   │   └── client.js             # Axios API client
│   ├── components/
│   │   ├── ChatPanel.jsx         # Main chat interface
│   │   ├── ImageGallery.jsx      # Generated images display
│   │   ├── ImageUpload.jsx       # Drag-and-drop image upload
│   │   ├── Settings.jsx          # Settings management UI
│   │   └── ModelManager.jsx      # SD model management UI
│   └── store/
│       └── useStore.js           # Zustand global state
├── vite.config.js                # Vite configuration with proxy
└── tailwind.config.js            # Tailwind CSS configuration
```

### Key Files Modified in Latest Session

#### Backend Files
1. **backend/models/schemas.py** (Lines 254-295)
   - Added `Img2ImgPlanRequest` schema
   - Added `Img2ImgGenerationPlan` schema
   - Added `ExecuteImg2ImgRequest` schema
   - Added `Img2ImgResponse` schema

2. **backend/prompt_engine.py** (Lines 626-902)
   - Added `create_img2img_plan()` method
   - Implemented LLM-guided denoising strength analysis
   - Added transformation type detection
   - Integrated quality analysis for img2img

3. **backend/main.py** (Lines 282-349)
   - Added `/plan-img2img` endpoint
   - Added `/execute-img2img` endpoint
   - Implemented model override support

#### Frontend Files
1. **frontend/src/api/client.js** (Lines 59-61)
   - Added `planImg2Img()` API method
   - Added `executeImg2Img()` API method

2. **frontend/src/components/ImageUpload.jsx** (New file)
   - Reusable drag-and-drop upload component
   - Base64 encoding logic
   - Image preview with controls
   - File validation

3. **frontend/src/components/ChatPanel.jsx** (Multiple sections)
   - Added img2img state management (lines 16-19)
   - Modified `handleSend()` for img2img planning (lines 111-127)
   - Modified `executeGeneration()` for img2img execution (lines 181-223)
   - Added img2img mode toggle UI (lines 329-364)
   - Added source image tracking

---

## 🔄 Current Workflow

### Text-to-Image (txt2img)
1. User enters prompt in chat
2. Frontend calls `/plan-generation`
3. Backend LLM analyzes prompt and creates plan:
   - Model recommendation with reasoning
   - Enhanced prompt with negative prompts
   - Quality analysis and specificity score
   - Parameter reasoning (steps, CFG, dimensions)
   - Tips for best results
4. User reviews plan (can override model)
5. User approves execution
6. Frontend calls `/execute-generation`
7. Backend generates image via SD API
8. Result displayed in image gallery

### Image-to-Image (img2img)
1. User enables img2img mode
2. User uploads source image (drag-and-drop)
3. User enters transformation prompt
4. Frontend calls `/plan-img2img` with source image
5. Backend LLM analyzes transformation and creates plan:
   - Model recommendation
   - Enhanced prompt
   - Denoising strength recommendation with reasoning
   - Quality analysis
   - Transformation-specific tips
6. User reviews plan (can override model)
7. User approves execution
8. Frontend calls `/execute-img2img` with source image
9. Backend transforms image via SD API
10. Result displayed alongside source image

---

## 🔧 Configuration

### Environment Variables
Located in `.env` file (see `.env.example` for template):

**Critical Settings:**
- `USE_DUAL_LLM` - Enable separate planning/execution LLMs
- `PLANNING_LLM_PROVIDER` - Provider for planning (claude/gemini/ollama)
- `EXECUTION_LLM_PROVIDER` - Provider for execution
- `SD_API_URL` - Stable Diffusion API endpoint (default: http://localhost:7860)

**API Keys:**
- `ANTHROPIC_API_KEY` - Claude API key
- `GOOGLE_API_KEY` - Gemini API key
- `CIVITAI_API_KEY` - CivitAI API key (for model downloads)

**Application:**
- `APP_HOST` - Backend host (default: 0.0.0.0)
- `APP_PORT` - Backend port (default: 8000)
- `DEBUG` - Debug mode (default: True)

### Model Configuration
Most settings can be configured via the Settings UI:
- LLM provider selection
- Model selection with auto-fetch
- Dual-LLM configuration
- SD connection testing
- Default generation parameters

---

## 🚧 Known Issues & Limitations

### Backend
- Multiple backend processes can run simultaneously if not properly managed
  - **Workaround:** Check for existing processes before starting
  - **Status:** Documented in README, needs proper fix

### Frontend
- No generation history persistence (lost on page refresh)
  - **Status:** Planned for future enhancement

### Img2Img
- No advanced controls (inpainting, sketch, masks)
  - **Status:** Marked as "In Progress" in roadmap
- No batch img2img transformations
  - **Status:** Planned feature

### Model Management
- CivitAI downloads don't show real-time progress in UI
  - **Status:** Backend polling endpoint exists, needs frontend integration

---

## 📋 Next Steps (Priority Order)

### High Priority
1. **Inpainting Support**
   - Add mask editor component
   - Implement inpainting endpoints
   - Update schemas for inpainting requests

2. **Generation History Persistence**
   - Add database layer (SQLite or file-based)
   - Store generation metadata
   - Implement search and filtering

3. **Download Progress UI Integration**
   - Connect frontend to progress polling endpoint
   - Add real-time progress bars
   - Handle download cancellation

### Medium Priority
4. **Batch Generation**
   - Multiple variations from single prompt
   - Batch img2img transformations
   - Queue management system

5. **ControlNet Integration**
   - Detect ControlNet models
   - Add ControlNet parameter UI
   - Implement preprocessing pipelines

6. **Prompt Library Enhancement**
   - Currently basic implementation
   - Add tagging and categorization
   - Enable sharing/import/export

### Low Priority
7. **Multi-Image Comparison**
   - Side-by-side comparison view
   - A/B testing interface
   - Generation parameter diff view

8. **Video Generation**
   - AnimateDiff integration
   - Frame-by-frame generation
   - Video export capabilities

---

## 🔐 Security Considerations

### Current Implementation
- API keys masked in UI
- `.env` file excluded from git
- CORS properly configured for development
- Input validation via Pydantic

### Production Recommendations
- Set `DEBUG=False` in production
- Use proper secrets management (not .env)
- Implement rate limiting
- Add authentication/authorization layer
- Use HTTPS for all API calls
- Implement CSRF protection

---

## 📦 Deployment Notes

### Backend Deployment
- Already production-ready with `DEBUG=False`
- Use proper ASGI server (Uvicorn with Gunicorn)
- Set appropriate `APP_HOST` and `APP_PORT`
- Ensure SD API is accessible from backend

### Frontend Deployment
- Run `npm run build` to create production bundle
- Serve `dist/` directory via Nginx or similar
- Configure reverse proxy to backend
- Update `API_BASE` in client.js if needed

### Docker Considerations
- No Dockerfile currently provided
- Would need to:
  - Create multi-stage build for frontend
  - Set up Python environment for backend
  - Configure networking between services
  - Handle SD API connection

---

## 🧪 Testing Status

### Current Testing
- Manual testing via UI
- Development testing with real LLM providers
- Integration testing with SD API

### Testing Gaps
- No unit tests for backend logic
- No component tests for React components
- No integration test suite
- No E2E tests

### Testing Recommendations
- Add pytest for backend testing
- Add Vitest for frontend unit tests
- Add React Testing Library for component tests
- Add Playwright for E2E tests
- Mock LLM and SD API responses for testing

---

## 📚 Documentation Status

### Completed Documentation
- ✅ Comprehensive README.md with setup instructions
- ✅ Feature list with Phase 2 additions
- ✅ Recent updates section (current session)
- ✅ Future enhancements roadmap (completed/in-progress/planned)
- ✅ .env.example with all configuration options
- ✅ This project state document

### Documentation Gaps
- No API documentation (consider OpenAPI/Swagger)
- No architecture diagrams
- No contribution guidelines
- No code comments in some complex sections
- No troubleshooting guide

---

## 🎓 Learning Resources

### For Continued Development
- **FastAPI:** https://fastapi.tiangolo.com/
- **React:** https://react.dev/
- **Stable Diffusion API:** https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/API
- **CivitAI API:** https://github.com/civitai/civitai/wiki/REST-API-Reference
- **Pydantic:** https://docs.pydantic.dev/
- **Zustand:** https://github.com/pmndrs/zustand

### Key Concepts to Understand
- **Plan/Act Pattern:** Two-phase AI workflow (planning + execution)
- **Denoising Strength:** Controls how much img2img changes source image
- **Negative Prompts:** Tell SD what NOT to include in generation
- **CFG Scale:** How closely SD follows the prompt (7-15 typical)
- **Sampling Steps:** More steps = more refined (but slower)

---

## 💡 Tips for Continuing Development

### When Adding New Features
1. Update schemas in `backend/models/schemas.py` first
2. Add endpoint to `backend/main.py`
3. Implement logic in appropriate service file
4. Add API client method in `frontend/src/api/client.js`
5. Update UI components as needed
6. Test with real LLM and SD
7. Update documentation

### When Debugging
- Check browser console for frontend errors
- Check terminal for backend logs (with traceback)
- Verify SD API is running (`http://localhost:7860`)
- Check `.env` configuration
- Validate API keys are correct
- Test API endpoints with curl/Postman

### Best Practices
- Always use Pydantic for request/response models
- Keep components focused (single responsibility)
- Use Zustand for global state only
- Handle errors gracefully with try/catch
- Log important operations for debugging
- Keep functions small and testable

---

## 🔄 Git Status

**Current Branch:** main
**Recent Commits:**
- 5f144f5 - Updates for moving to another machine: Readme.md updated
- c27e288 - Initial Push, Broken Code - but getting there

**Untracked Files:**
- `.env_backups/.env.backup_20251110_174354`

**Modified Files:**
- `frontend/package-lock.json`

**Recommendation:** Commit current changes before pushing to GitHub and continuing on another machine.

---

## 📞 Support & Issues

### Current Known Issues
See "Known Issues & Limitations" section above

### Reporting New Issues
When encountering issues, document:
1. What you were trying to do
2. What you expected to happen
3. What actually happened
4. Error messages (frontend console + backend terminal)
5. Configuration details (LLM provider, SD model, etc.)
6. Steps to reproduce

---

**End of Project State Document**

This document should be updated whenever significant changes are made to the project. It serves as a comprehensive reference for understanding the current state, architecture, and future direction of AIServices.
