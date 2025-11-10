# AIServices - Improvement Suggestions

**Last Updated:** 2025-11-10
**Project Phase:** Post Phase 2 - Ready for Phase 3

---

## 🎯 Quick Wins (Easy, High Impact)

### 1. Add Comprehensive Testing
**Priority:** CRITICAL
**Effort:** Medium
**Impact:** High

**Why:** Zero test coverage is the biggest risk to project stability and future development.

**Implementation:**
```bash
# Install testing dependencies
pip install pytest pytest-asyncio pytest-cov pytest-mock
cd frontend && npm install --save-dev vitest @testing-library/react @testing-library/jest-dom

# Create test structure
mkdir -p tests/unit tests/integration
mkdir -p frontend/tests/components frontend/tests/utils
```

**Example Tests:**
- `tests/unit/test_prompt_engine.py` - Test prompt generation logic
- `tests/integration/test_generation_workflow.py` - Test full plan/act flow
- `frontend/tests/components/ImageUpload.test.jsx` - Test upload component

**Success Metrics:**
- 80% backend code coverage
- 70% frontend code coverage
- CI/CD pipeline passing

---

### 2. Implement Redis Caching
**Priority:** HIGH
**Effort:** Low
**Impact:** High

**Why:** Reduce redundant LLM API calls and SD model list fetches.

**What to Cache:**
- Installed SD model lists (TTL: 5 minutes)
- Sampler lists (TTL: 10 minutes)
- Similar prompt plans (TTL: 1 hour)

**Implementation:**
```python
# Install Redis
pip install redis

# Add caching decorator
from redis import Redis
import json

redis_client = Redis(host='localhost', port=6379, db=0)

def cache(ttl=300):
    def decorator(func):
        async def wrapper(*args, **kwargs):
            key = f"{func.__name__}:{hash(str(args))}"
            cached = redis_client.get(key)
            if cached:
                return json.loads(cached)

            result = await func(*args, **kwargs)
            redis_client.setex(key, ttl, json.dumps(result))
            return result
        return wrapper
    return decorator

# Usage
@cache(ttl=600)
async def get_installed_models():
    return await model_manager.get_installed_models()
```

**Expected Improvements:**
- 50-70% reduction in SD API calls
- Faster UI responsiveness
- Lower API costs

---

### 3. Add Toast Notifications
**Priority:** HIGH
**Effort:** Low
**Impact:** Medium

**Why:** Replace alert() with better UX.

**Implementation:**
```bash
npm install react-hot-toast
```

```javascript
// frontend/src/App.jsx
import { Toaster } from 'react-hot-toast'

function App() {
  return (
    <>
      <Toaster position="top-right" />
      {/* rest of app */}
    </>
  )
}

// Usage in components
import toast from 'react-hot-toast'

// Success
toast.success('Image generated successfully!')

// Error with action
toast.error('Generation failed', {
  action: {
    label: 'Retry',
    onClick: () => handleRetry()
  }
})

// Loading with promise
const promise = api.planGeneration(data)
toast.promise(promise, {
  loading: 'Creating generation plan...',
  success: 'Plan ready! Review and generate.',
  error: 'Failed to create plan'
})
```

---

### 4. Split ChatPanel Component
**Priority:** HIGH
**Effort:** Medium
**Impact:** Medium

**Why:** ChatPanel.jsx is 400+ lines and handles too many concerns.

**Proposed Structure:**
```
components/
  ChatPanel/
    index.jsx               # Main container (orchestration)
    ChatMessages.jsx        # Message list display
    MessageInput.jsx        # Input textarea + send button
    GenerationControls.jsx  # Model selection, parameters
    PlanReview.jsx          # Plan display and approval
    Img2ImgControls.jsx     # Img2img toggle and upload
```

**Benefits:**
- Easier to test individual components
- Reduced re-renders (better performance)
- Clearer separation of concerns
- Easier to maintain and extend

---

### 5. Add Keyboard Shortcuts
**Priority:** MEDIUM
**Effort:** Low
**Impact:** Medium

**Why:** Power users love keyboard shortcuts.

**Suggested Shortcuts:**
```javascript
// frontend/src/hooks/useKeyboardShortcuts.js
export function useKeyboardShortcuts() {
  useEffect(() => {
    const handleKeyboard = (e) => {
      // Ctrl+Enter: Send message
      if (e.ctrlKey && e.key === 'Enter') {
        handleSend()
      }

      // Ctrl+K: Focus input
      if (e.ctrlKey && e.key === 'k') {
        e.preventDefault()
        inputRef.current?.focus()
      }

      // Escape: Cancel generation/clear input
      if (e.key === 'Escape') {
        handleCancel()
      }

      // Ctrl+Shift+I: Toggle img2img mode
      if (e.ctrlKey && e.shiftKey && e.key === 'I') {
        setImg2imgMode(!img2imgMode)
      }
    }

    window.addEventListener('keydown', handleKeyboard)
    return () => window.removeEventListener('keydown', handleKeyboard)
  }, [])
}
```

**Include Help Modal:**
- Press `?` to show keyboard shortcuts overlay
- List all available shortcuts
- Show shortcut hints on hover

---

## 🚀 Performance Optimizations

### 6. Implement Image Thumbnails
**Priority:** MEDIUM
**Effort:** Medium
**Impact:** High

**Why:** Full-resolution base64 images slow down the UI.

**Implementation:**
```javascript
// frontend/src/utils/imageUtils.js
export async function createThumbnail(base64, maxWidth = 400, maxHeight = 400) {
  return new Promise((resolve) => {
    const img = new Image()
    img.onload = () => {
      const canvas = document.createElement('canvas')
      let width = img.width
      let height = img.height

      if (width > height) {
        if (width > maxWidth) {
          height *= maxWidth / width
          width = maxWidth
        }
      } else {
        if (height > maxHeight) {
          width *= maxHeight / height
          height = maxHeight
        }
      }

      canvas.width = width
      canvas.height = height
      const ctx = canvas.getContext('2d')
      ctx.drawImage(img, 0, 0, width, height)

      resolve(canvas.toDataURL().split(',')[1])
    }
    img.src = `data:image/png;base64,${base64}`
  })
}

// Usage
const thumbnail = await createThumbnail(image_base64)
// Store both: full image for display, thumbnail for gallery
```

**Benefits:**
- 10x faster gallery loading
- Reduced memory usage
- Smoother scrolling

---

### 7. Add React.memo Optimizations
**Priority:** MEDIUM
**Effort:** Low
**Impact:** Medium

**Why:** Prevent unnecessary re-renders.

**Implementation:**
```javascript
// Wrap expensive components
export const ImageUpload = React.memo(function ImageUpload({ onImageSelect, selectedImage, onClear }) {
  // ...
}, (prevProps, nextProps) => {
  // Custom comparison
  return prevProps.selectedImage === nextProps.selectedImage
})

// Use useMemo for expensive calculations
const sortedImages = useMemo(() => {
  return images.sort((a, b) => b.timestamp - a.timestamp)
}, [images])

// Use useCallback for callbacks passed to children
const handleImageSelect = useCallback((base64, name) => {
  setSourceImage(base64)
  setSourceImageName(name)
}, [])
```

---

### 8. Parallel API Calls
**Priority:** LOW
**Effort:** Low
**Impact:** Low

**Why:** Fetch multiple independent resources simultaneously.

**Implementation:**
```python
# Backend: Use asyncio.gather
import asyncio

async def get_initial_data():
    samplers, models, settings = await asyncio.gather(
        sd_service.get_samplers(),
        model_manager.get_installed_models(),
        get_current_settings()
    )
    return {
        "samplers": samplers,
        "models": models,
        "settings": settings
    }
```

```javascript
// Frontend: Use Promise.all
async function loadInitialData() {
  const [health, hardware, models] = await Promise.all([
    api.getHealth(),
    api.getHardware(),
    api.getModels()
  ])
  return { health, hardware, models }
}
```

---

## 🔒 Security Improvements

### 9. Implement Rate Limiting
**Priority:** HIGH
**Effort:** Low
**Impact:** High

**Why:** Prevent API abuse and control costs.

**Implementation:**
```python
# Install slowapi
pip install slowapi

# In main.py
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Apply to endpoints
@app.post("/plan-generation")
@limiter.limit("10/minute")  # 10 requests per minute per IP
async def plan_generation(request: Request, data: GenerationPlanRequest):
    # ...

@app.post("/execute-generation")
@limiter.limit("5/minute")  # Stricter for expensive operations
async def execute_generation(request: Request, data: ExecuteGenerationRequest):
    # ...
```

**Rate Limit Suggestions:**
- Planning: 10-20 per minute
- Execution: 5-10 per minute
- Model list: 30 per minute
- Settings: 60 per minute

---

### 10. Backend Image Validation
**Priority:** MEDIUM
**Effort:** Low
**Impact:** Medium

**Why:** Don't trust frontend MIME type validation.

**Implementation:**
```python
# Install python-magic
pip install python-magic-bin  # Windows
pip install python-magic      # Linux/Mac

import magic
import base64

def validate_image(base64_data: str) -> bool:
    """Validate image using magic bytes"""
    try:
        image_data = base64.b64decode(base64_data)
        mime = magic.from_buffer(image_data, mime=True)

        allowed_types = ['image/jpeg', 'image/png', 'image/webp', 'image/gif']
        return mime in allowed_types
    except Exception:
        return False

# In endpoint
@app.post("/plan-img2img")
async def plan_img2img(request: Img2ImgPlanRequest):
    if not validate_image(request.init_image_base64):
        raise HTTPException(status_code=400, detail="Invalid image format")
    # ...
```

---

### 11. Add CSRF Protection
**Priority:** LOW
**Effort:** Low
**Impact:** Low

**Why:** Security best practice for production.

**Implementation:**
```python
# Install itsdangerous for token generation
from itsdangerous import URLSafeTimedSerializer

# Generate CSRF tokens
csrf_serializer = URLSafeTimedSerializer(settings.SECRET_KEY)

@app.get("/csrf-token")
async def get_csrf_token():
    token = csrf_serializer.dumps("csrf_token")
    return {"csrf_token": token}

# Validate CSRF tokens
def validate_csrf(token: str) -> bool:
    try:
        csrf_serializer.loads(token, max_age=3600)  # 1 hour
        return True
    except:
        return False
```

---

## 💾 Data Persistence

### 12. Add SQLite Database for History
**Priority:** HIGH
**Effort:** Medium
**Impact:** High

**Why:** Users lose all history on page refresh.

**Implementation:**
```python
# Install SQLAlchemy
pip install sqlalchemy alembic

# Create models
from sqlalchemy import Column, Integer, String, DateTime, JSON, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime

Base = declarative_base()

class Generation(Base):
    __tablename__ = 'generations'

    id = Column(Integer, primary_key=True)
    prompt_input = Column(String, nullable=False)
    enhanced_prompt = Column(JSON, nullable=False)
    plan = Column(JSON, nullable=False)
    image_path = Column(String, nullable=False)  # Store images as files
    thumbnail_path = Column(String)
    generation_type = Column(String)  # 'txt2img' or 'img2img'
    source_image_path = Column(String)  # For img2img
    created_at = Column(DateTime, default=datetime.utcnow)

# Create endpoints
@app.get("/history")
async def get_history(limit: int = 50, offset: int = 0):
    generations = session.query(Generation)\
        .order_by(Generation.created_at.desc())\
        .limit(limit)\
        .offset(offset)\
        .all()
    return generations

@app.delete("/history/{generation_id}")
async def delete_generation(generation_id: int):
    generation = session.query(Generation).get(generation_id)
    if generation:
        # Delete image files
        os.remove(generation.image_path)
        if generation.thumbnail_path:
            os.remove(generation.thumbnail_path)
        session.delete(generation)
        session.commit()
    return {"status": "deleted"}
```

**Database Schema Migration:**
```bash
# Initialize Alembic
alembic init alembic

# Create migration
alembic revision --autogenerate -m "Add generations table"

# Apply migration
alembic upgrade head
```

---

### 13. Store Images as Files (Not Base64)
**Priority:** HIGH
**Effort:** Medium
**Impact:** High

**Why:** Base64 images in database bloat size significantly.

**Implementation:**
```python
# Create images directory
import os
from datetime import datetime
from pathlib import Path

IMAGES_DIR = Path("./generated_images")
IMAGES_DIR.mkdir(exist_ok=True)

def save_image(base64_data: str, generation_id: int) -> str:
    """Save base64 image to disk"""
    image_data = base64.b64decode(base64_data)

    # Organize by date
    date_dir = IMAGES_DIR / datetime.now().strftime("%Y%m%d")
    date_dir.mkdir(exist_ok=True)

    filepath = date_dir / f"generation_{generation_id}.png"
    with open(filepath, 'wb') as f:
        f.write(image_data)

    return str(filepath)

# Serve images via static route
from fastapi.staticfiles import StaticFiles
app.mount("/images", StaticFiles(directory="generated_images"), name="images")

# Return URL instead of base64
return {
    "image_url": f"/images/{date}/generation_{id}.png",
    "thumbnail_url": f"/images/{date}/generation_{id}_thumb.png"
}
```

**Benefits:**
- Smaller API responses
- Faster page loads
- Easier backup/export
- Database stays manageable

---

## 🎨 UI/UX Enhancements

### 14. Add Loading Skeletons
**Priority:** MEDIUM
**Effort:** Low
**Impact:** Medium

**Why:** Better perceived performance than blank space.

**Implementation:**
```javascript
// Create Skeleton component
function Skeleton({ width, height, className = '' }) {
  return (
    <div
      className={`animate-pulse bg-white/10 rounded ${className}`}
      style={{ width, height }}
    />
  )
}

// Use in ImageGallery
{isLoading ? (
  <div className="grid grid-cols-3 gap-4">
    {[...Array(6)].map((_, i) => (
      <Skeleton key={i} width="100%" height="300px" />
    ))}
  </div>
) : (
  <ImageGrid images={images} />
)}

// Use in ChatPanel
{isLoadingPlan && (
  <div className="plan-skeleton">
    <Skeleton width="80%" height="24px" className="mb-2" />
    <Skeleton width="100%" height="100px" className="mb-4" />
    <Skeleton width="60%" height="24px" />
  </div>
)}
```

---

### 15. Add Image Comparison Slider
**Priority:** LOW
**Effort:** Medium
**Impact:** Medium

**Why:** Great for img2img before/after comparison.

**Implementation:**
```bash
npm install react-compare-image
```

```javascript
import ReactCompareImage from 'react-compare-image'

function Img2ImgResult({ sourceImage, resultImage }) {
  return (
    <div className="comparison-view">
      <ReactCompareImage
        leftImage={`data:image/png;base64,${sourceImage}`}
        rightImage={`data:image/png;base64,${resultImage}`}
        sliderLineWidth={3}
        sliderLineColor="#8B5CF6"
      />
      <div className="labels">
        <span>Before</span>
        <span>After</span>
      </div>
    </div>
  )
}
```

---

### 16. Implement Prompt Library Enhancement
**Priority:** MEDIUM
**Effort:** Medium
**Impact:** Medium

**Why:** Current prompt library is basic.

**Enhancements:**
- Tags/categories (portrait, landscape, anime, realistic, etc.)
- Search functionality
- Favorites
- Public/private prompts
- Import/export collections

**Implementation:**
```python
# Enhanced prompt schema
class PromptTemplate(Base):
    __tablename__ = 'prompt_templates'

    id = Column(Integer, primary_key=True)
    title = Column(String, nullable=False)
    description = Column(String)
    prompt_text = Column(String, nullable=False)
    negative_prompt = Column(String)
    tags = Column(JSON)  # ['portrait', 'anime', 'detailed']
    is_favorite = Column(Boolean, default=False)
    is_public = Column(Boolean, default=False)
    usage_count = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)

# Search endpoint
@app.get("/prompts/search")
async def search_prompts(
    q: str = None,
    tags: List[str] = [],
    favorites_only: bool = False
):
    query = session.query(PromptTemplate)

    if q:
        query = query.filter(
            PromptTemplate.title.contains(q) |
            PromptTemplate.prompt_text.contains(q)
        )

    if tags:
        query = query.filter(PromptTemplate.tags.contains(tags))

    if favorites_only:
        query = query.filter(PromptTemplate.is_favorite == True)

    return query.all()
```

---

## 🏗️ Infrastructure & DevOps

### 17. Docker Compose Setup
**Priority:** HIGH
**Effort:** Medium
**Impact:** High

**Why:** Simplify deployment and development setup.

**Implementation:**
See [.claude/comprehensive-review.md](comprehensive-review.md#1-docker-support-priority-high) for complete Docker setup.

**Quick Start After Implementation:**
```bash
# Start everything
docker-compose up

# Development mode with hot reload
docker-compose -f docker-compose.dev.yml up

# Production
docker-compose -f docker-compose.prod.yml up -d
```

---

### 18. GitHub Actions CI/CD
**Priority:** MEDIUM
**Effort:** Low
**Impact:** High

**Why:** Automated testing and deployment.

**Implementation:**
```yaml
# .github/workflows/ci.yml
name: CI/CD Pipeline

on: [push, pull_request]

jobs:
  test-backend:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pytest pytest-asyncio pytest-cov
      - name: Run tests
        run: pytest --cov=backend --cov-report=xml
      - name: Upload coverage
        uses: codecov/codecov-action@v3

  test-frontend:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-node@v3
        with:
          node-version: '18'
      - name: Install dependencies
        run: cd frontend && npm ci
      - name: Run tests
        run: cd frontend && npm test
      - name: Build
        run: cd frontend && npm run build

  deploy:
    needs: [test-backend, test-frontend]
    if: github.ref == 'refs/heads/main'
    runs-on: ubuntu-latest
    steps:
      - name: Deploy to production
        run: echo "Deploy script here"
```

---

### 19. Add Structured Logging
**Priority:** MEDIUM
**Effort:** Low
**Impact:** Medium

**Why:** Better debugging and monitoring in production.

**Implementation:**
```python
# Install structlog
pip install structlog

# Configure in main.py
import structlog

structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

# Usage
logger.info("generation_started", user_input=request.user_input, model=model_name)
logger.error("generation_failed", error=str(e), user_input=request.user_input)
```

---

### 20. Add Health Check Endpoints
**Priority:** LOW
**Effort:** Low
**Impact:** Low

**Why:** Essential for production monitoring.

**Implementation:**
```python
from datetime import datetime
import psutil

@app.get("/health/live")
async def liveness():
    """Basic liveness check - is the app running?"""
    return {"status": "alive", "timestamp": datetime.now().isoformat()}

@app.get("/health/ready")
async def readiness():
    """Readiness check - is the app ready to serve traffic?"""
    checks = {
        "sd_api": await check_sd_connection(),
        "llm_provider": await check_llm_connection(),
        "disk_space": check_disk_space(),
        "memory": check_memory()
    }

    all_ready = all(checks.values())

    return {
        "status": "ready" if all_ready else "not_ready",
        "checks": checks,
        "timestamp": datetime.now().isoformat()
    }

def check_disk_space() -> bool:
    """Check if enough disk space available"""
    usage = psutil.disk_usage('/')
    return usage.percent < 90  # Less than 90% full

def check_memory() -> bool:
    """Check if enough memory available"""
    memory = psutil.virtual_memory()
    return memory.percent < 90  # Less than 90% used
```

---

## 🎯 Feature Additions

### 21. Inpainting Support
**Priority:** HIGH
**Effort:** High
**Impact:** High

**Why:** One of most requested SD features.

**Requirements:**
1. Canvas-based mask editor
2. Brush size/opacity controls
3. Inpainting-specific models
4. Backend endpoint for inpainting

**Recommended Library:**
```bash
npm install react-sketch-canvas
```

**Backend Implementation:**
```python
class InpaintRequest(BaseModel):
    init_image_base64: str
    mask_image_base64: str  # White = inpaint, Black = keep
    prompt: SDPrompt
    denoising_strength: float = 0.75

@app.post("/inpaint")
async def inpaint(request: InpaintRequest):
    result = await sd_service.img2img(
        prompt=request.prompt,
        init_images=[request.init_image_base64],
        mask=request.mask_image_base64,
        denoising_strength=request.denoising_strength,
        inpainting_fill=1,  # original
        inpaint_full_res=True
    )
    return result
```

---

### 22. Batch Generation Queue
**Priority:** MEDIUM
**Effort:** High
**Impact:** Medium

**Why:** Generate multiple variations efficiently.

**Implementation:**
```python
# Install Celery
pip install celery redis

# Create celery_app.py
from celery import Celery

celery_app = Celery(
    'aiservices',
    broker='redis://localhost:6379/0',
    backend='redis://localhost:6379/1'
)

@celery_app.task(bind=True)
def generate_batch_task(self, prompts: List[dict]):
    """Generate multiple images in background"""
    results = []
    for i, prompt_data in enumerate(prompts):
        # Update progress
        self.update_state(
            state='PROGRESS',
            meta={'current': i, 'total': len(prompts)}
        )

        result = sd_service.generate_image(prompt_data)
        results.append(result)

    return results

# API endpoint
@app.post("/batch-generate")
async def batch_generate(prompts: List[GenerationPlanRequest]):
    task = generate_batch_task.delay([p.dict() for p in prompts])
    return {"task_id": task.id}

@app.get("/batch-status/{task_id}")
async def batch_status(task_id: str):
    task = celery_app.AsyncResult(task_id)
    return {
        "status": task.state,
        "progress": task.info.get('current', 0) if task.state == 'PROGRESS' else None,
        "total": task.info.get('total', 0) if task.state == 'PROGRESS' else None,
        "results": task.result if task.state == 'SUCCESS' else None
    }
```

---

### 23. ControlNet Integration
**Priority:** LOW
**Effort:** Very High
**Impact:** High

**Why:** Advanced control over generation.

**Prerequisites:**
- SD Web UI with ControlNet extension
- ControlNet models installed

**Types to Support:**
1. Canny edge detection
2. Depth map
3. Pose detection (OpenPose)
4. Scribble/sketch

**Implementation Overview:**
```python
class ControlNetRequest(BaseModel):
    prompt: SDPrompt
    control_image_base64: str
    control_type: str  # 'canny', 'depth', 'openpose', 'scribble'
    control_weight: float = 1.0
    guidance_start: float = 0.0
    guidance_end: float = 1.0

@app.post("/controlnet-generate")
async def controlnet_generate(request: ControlNetRequest):
    # Preprocess control image based on type
    if request.control_type == 'canny':
        control_image = await preprocess_canny(request.control_image_base64)
    elif request.control_type == 'depth':
        control_image = await preprocess_depth(request.control_image_base64)
    # ... etc

    result = await sd_service.generate_with_controlnet(
        prompt=request.prompt,
        control_image=control_image,
        control_type=request.control_type,
        control_weight=request.control_weight
    )
    return result
```

---

## 📚 Documentation Improvements

### 24. Add API Documentation
**Priority:** MEDIUM
**Effort:** Low
**Impact:** Medium

**Why:** FastAPI has built-in OpenAPI support.

**Implementation:**
```python
# Already works! Just improve descriptions

@app.post(
    "/plan-generation",
    response_model=GenerationPlan,
    summary="Create generation plan",
    description="""
    Creates a comprehensive generation plan using AI analysis.

    The plan includes:
    - Enhanced prompt with negative prompts
    - Model recommendation with reasoning
    - Quality analysis and specificity score
    - Parameter recommendations
    - Tips for best results

    This is the PLAN phase of the Plan/Act workflow.
    """,
    responses={
        200: {"description": "Plan created successfully"},
        500: {"description": "LLM or internal error"}
    }
)
async def plan_generation(request: GenerationPlanRequest):
    pass
```

**Access Documentation:**
- OpenAPI JSON: `http://localhost:8000/openapi.json`
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

---

### 25. Create Architecture Diagrams
**Priority:** LOW
**Effort:** Low
**Impact:** Low

**Why:** Visual documentation helps new developers.

**Tools:**
- Mermaid (embedded in Markdown)
- Draw.io
- Excalidraw

**Example Diagrams to Create:**
```mermaid
# Plan/Act Workflow
sequenceDiagram
    User->>Frontend: Enter prompt
    Frontend->>Backend: POST /plan-generation
    Backend->>LLM: Analyze prompt
    LLM-->>Backend: Plan with reasoning
    Backend-->>Frontend: Generation plan
    Frontend->>User: Review plan
    User->>Frontend: Approve
    Frontend->>Backend: POST /execute-generation
    Backend->>SD API: Generate image
    SD API-->>Backend: Generated image
    Backend-->>Frontend: Result
    Frontend->>User: Display image
```

---

## 🎓 Developer Experience

### 26. Add Pre-commit Hooks
**Priority:** LOW
**Effort:** Low
**Impact:** Low

**Why:** Catch issues before commit.

**Implementation:**
```bash
# Install pre-commit
pip install pre-commit

# Create .pre-commit-config.yaml
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.4.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files

  - repo: https://github.com/psf/black
    rev: 23.3.0
    hooks:
      - id: black
        language_version: python3.10

  - repo: https://github.com/pycqa/isort
    rev: 5.12.0
    hooks:
      - id: isort

  - repo: https://github.com/pycqa/flake8
    rev: 6.0.0
    hooks:
      - id: flake8

# Install hooks
pre-commit install
```

---

### 27. Add Development VS Code Config
**Priority:** LOW
**Effort:** Low
**Impact:** Low

**Implementation:**
```json
// .vscode/settings.json
{
  "python.linting.enabled": true,
  "python.linting.pylintEnabled": true,
  "python.formatting.provider": "black",
  "editor.formatOnSave": true,
  "editor.codeActionsOnSave": {
    "source.organizeImports": true
  },
  "[javascript]": {
    "editor.defaultFormatter": "esbenp.prettier-vscode"
  },
  "[python]": {
    "editor.defaultFormatter": "ms-python.black-formatter"
  }
}
```

```json
// .vscode/launch.json
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "Python: FastAPI",
      "type": "python",
      "request": "launch",
      "module": "backend.main",
      "console": "integratedTerminal",
      "justMyCode": false
    },
    {
      "name": "Debug Frontend",
      "type": "chrome",
      "request": "launch",
      "url": "http://localhost:5173",
      "webRoot": "${workspaceFolder}/frontend/src"
    }
  ]
}
```

---

## 📊 Summary & Prioritization

### Immediate Priority (Do This Week)
1. ✅ Add comprehensive testing
2. ✅ Implement Redis caching
3. ✅ Add toast notifications
4. ✅ Implement rate limiting

### Short Term (Do This Month)
5. ✅ Split ChatPanel component
6. ✅ Add SQLite database for history
7. ✅ Store images as files
8. ✅ Implement image thumbnails
9. ✅ Docker Compose setup

### Medium Term (Next 2-3 Months)
10. ✅ Add keyboard shortcuts
11. ✅ Enhance prompt library
12. ✅ Implement batch generation
13. ✅ Add inpainting support
14. ✅ GitHub Actions CI/CD

### Long Term (Future Roadmap)
15. ✅ ControlNet integration
16. ✅ Advanced monitoring/logging
17. ✅ Mobile app considerations
18. ✅ Multi-user support

---

## 🏆 Success Metrics

Track these metrics to measure improvements:

**Performance:**
- Average page load time < 2 seconds
- Time to first image < 30 seconds
- API response time < 200ms (excluding LLM/SD)

**Quality:**
- Test coverage > 80%
- Zero critical bugs
- < 5 minor bugs

**User Experience:**
- Task completion rate > 95%
- User satisfaction score > 4.5/5
- Feature adoption rate > 70%

**Technical:**
- Deployment time < 5 minutes
- Server uptime > 99.5%
- API error rate < 1%

---

**Suggestions Document Complete**

This document should be reviewed and updated quarterly or after major feature additions. Priority levels may change based on user feedback and business requirements.
