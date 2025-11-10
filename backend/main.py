from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
from contextlib import asynccontextmanager

from backend.config import get_settings
from backend.models.schemas import (
    PromptRequest,
    GenerationResponse,
    IterationRequest,
    PromptEnhancementResponse,
    InteractivePromptRequest,
    QuestionResponse,
    AnswerRequest,
    KnowledgeBaseInfo,
    SettingsResponse,
    SettingsUpdate,
    SettingsValidationResponse,
    ConnectionTestRequest,
    ConnectionTestResponse,
    AvailableModelsRequest,
    AvailableModelsResponse
)
from backend.llm_service import LLMService
from backend.sd_service import StableDiffusionService
from backend.prompt_engine import PromptEngine
from backend.knowledge_base import KnowledgeBase
from backend.interactive_prompter import InteractivePrompter
from backend.prompt_library import PromptLibrary
from backend.settings_service import SettingsService

settings = get_settings()


# Initialize services
llm_service = LLMService()
sd_service = StableDiffusionService()
knowledge_base = KnowledgeBase()
prompt_engine = PromptEngine(llm_service)
interactive_prompter = InteractivePrompter(llm_service, knowledge_base)
prompt_library = PromptLibrary()
settings_service = SettingsService()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    # Startup
    print("[INFO] Starting SD Prompt Assistant...")

    # Initialize knowledge base with examples
    knowledge_base.create_example_guides()
    print("[INFO] Knowledge Base initialized")

    # Display active LLM provider(s)
    provider_info = llm_service.get_provider_info()
    if provider_info.get("mode") == "dual":
        print(f"[LLM] Mode: Dual")
        print(f"   Planning: {provider_info['planning_provider']} ({provider_info['planning_model']})")
        print(f"   Execution: {provider_info['execution_provider']} ({provider_info['execution_model']})")
    else:
        print(f"[LLM] Provider: {provider_info['provider']} ({provider_info['model']})")
    print(f"[SD] Stable Diffusion: {settings.sd_api_url}")

    # Check services
    try:
        await llm_service.health_check()
        if provider_info.get("mode") == "dual":
            print(f"[OK] Planning LLM ({provider_info['planning_provider']}) connected")
            print(f"[OK] Execution LLM ({provider_info['execution_provider']}) connected")
        else:
            print(f"[OK] {provider_info['provider']} connected")
    except Exception as e:
        print(f"[WARNING] LLM Service not available: {e}")

    try:
        await sd_service.health_check()
        print("[OK] Stable Diffusion connected")
    except Exception as e:
        print(f"[WARNING] Stable Diffusion not available: {e}")

    yield

    # Shutdown
    print("[INFO] Shutting down...")


app = FastAPI(
    title="SD Prompt Assistant",
    description="Self-hosted AI prompt engineering assistant for Stable Diffusion",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "online",
        "service": "SD Prompt Assistant",
        "version": "1.0.0"
    }


@app.get("/health")
async def health():
    """Check health of all services"""
    provider_info = llm_service.get_provider_info()
    health_status = {
        "api": "healthy",
        "llm": "unknown",
        "llm_provider": provider_info["provider"],
        "llm_model": provider_info["model"],
        "sd": "unknown"
    }

    try:
        await llm_service.health_check()
        health_status["llm"] = "healthy"
    except Exception as e:
        health_status["llm"] = f"unhealthy: {str(e)}"

    try:
        await sd_service.health_check()
        health_status["sd"] = "healthy"
    except Exception as e:
        health_status["sd"] = f"unhealthy: {str(e)}"

    return health_status


@app.post("/enhance-prompt", response_model=PromptEnhancementResponse)
async def enhance_prompt(request: PromptRequest):
    """
    Enhance a natural language prompt into a structured SD prompt
    """
    try:
        enhancement = await prompt_engine.enhance_prompt(
            user_input=request.user_input,
            conversation_history=request.conversation_history
        )
        return enhancement
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prompt enhancement failed: {str(e)}")


@app.post("/generate", response_model=GenerationResponse)
async def generate_image(request: PromptRequest):
    """
    Full pipeline: enhance prompt and generate image
    """
    try:
        # Enhance the prompt
        enhancement = await prompt_engine.enhance_prompt(
            user_input=request.user_input,
            conversation_history=request.conversation_history
        )

        # Generate image
        result = await sd_service.generate_image(enhancement.enhanced_prompt)

        return GenerationResponse(
            image_base64=result["image_base64"],
            prompt_used=enhancement.enhanced_prompt,
            llm_explanation=enhancement.explanation,
            generation_time=result["generation_time"],
            seed_used=result["seed"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")


@app.post("/iterate", response_model=GenerationResponse)
async def iterate_image(request: IterationRequest):
    """
    Iterate on a previous generation based on user feedback
    """
    try:
        # Get iteration suggestions from LLM
        enhancement = await prompt_engine.iterate_prompt(
            previous_prompt=request.previous_prompt,
            user_feedback=request.user_feedback,
            previous_image_base64=request.previous_image_base64
        )

        # Generate new image
        result = await sd_service.generate_image(enhancement.enhanced_prompt)

        return GenerationResponse(
            image_base64=result["image_base64"],
            prompt_used=enhancement.enhanced_prompt,
            llm_explanation=enhancement.explanation,
            generation_time=result["generation_time"],
            seed_used=result["seed"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Iteration failed: {str(e)}")


@app.get("/knowledge-base", response_model=KnowledgeBaseInfo)
async def get_knowledge_base_info():
    """Get information about available knowledge base"""
    try:
        guides = knowledge_base.get_style_guides()
        templates = knowledge_base.get_prompt_templates()

        return KnowledgeBaseInfo(
            guides_count=len(guides),
            templates_count=len(templates),
            available_guides=guides,
            available_templates=templates
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get knowledge base info: {str(e)}")


@app.post("/interactive/ask", response_model=QuestionResponse)
async def ask_clarifying_questions(request: InteractivePromptRequest):
    """
    Analyze user input and ask clarifying questions if needed
    """
    try:
        result = await interactive_prompter.analyze_and_ask_questions(
            user_input=request.user_input,
            conversation_history=request.conversation_history
        )
        return QuestionResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate questions: {str(e)}")


@app.post("/interactive/generate", response_model=PromptEnhancementResponse)
async def generate_from_conversation(request: AnswerRequest):
    """
    Generate final prompt from conversation with Q&A
    """
    try:
        result = await interactive_prompter.create_prompt_from_conversation(
            conversation_history=request.conversation_history
        )

        from backend.models.schemas import SDPrompt
        sd_prompt = SDPrompt(
            positive_prompt=result["positive_prompt"],
            negative_prompt=result.get("negative_prompt", ""),
            steps=result.get("steps", 30),
            cfg_scale=result.get("cfg_scale", 7.0),
            width=result.get("width", 512),
            height=result.get("height", 512),
            sampler_name=result.get("sampler_name", "DPM++ 2M Karras"),
            seed=result.get("seed", -1)
        )

        return PromptEnhancementResponse(
            enhanced_prompt=sd_prompt,
            explanation=result.get("explanation", ""),
            suggestions=result.get("suggestions", [])
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate prompt: {str(e)}")


@app.get("/hardware")
async def get_hardware_info():
    """Get hardware information and LLM recommendations"""
    try:
        from backend.hardware_detector import HardwareDetector
        hardware_info = HardwareDetector.get_hardware_summary()
        return hardware_info
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get hardware info: {str(e)}")


@app.get("/samplers")
async def get_samplers():
    """Get available samplers from SD"""
    try:
        samplers = await sd_service.get_samplers()
        return {"samplers": samplers}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get samplers: {str(e)}")


@app.get("/models")
async def get_models():
    """Get available SD models"""
    try:
        models = await sd_service.get_models()
        return {"models": models}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get models: {str(e)}")


# Prompt Library Endpoints
@app.post("/prompts/save")
async def save_prompt_endpoint(data: dict):
    """Save a prompt to the library"""
    try:
        saved = prompt_library.save_prompt(data)
        return saved
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save prompt: {str(e)}")


@app.get("/prompts")
async def get_prompts():
    """Get all saved prompts"""
    try:
        prompts = prompt_library.get_all_prompts()
        return {"prompts": prompts}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get prompts: {str(e)}")


@app.delete("/prompts/{prompt_id}")
async def delete_prompt_endpoint(prompt_id: str):
    """Delete a saved prompt"""
    try:
        success = prompt_library.delete_prompt(prompt_id)
        if success:
            return {"message": "Prompt deleted"}
        else:
            raise HTTPException(status_code=404, detail="Prompt not found")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete prompt: {str(e)}")


# ============================================
# SETTINGS ENDPOINTS
# ============================================

@app.get("/settings", response_model=SettingsResponse)
async def get_settings_endpoint():
    """
    Get current application settings (API keys are masked)
    """
    try:
        # Get all settings from .env
        all_settings = settings_service.get_safe_settings()

        # Convert to response model with defaults for missing values
        return SettingsResponse(
            # Dual-LLM Configuration
            use_dual_llm=all_settings.get("USE_DUAL_LLM", "false").lower() == "true",
            planning_llm_provider=all_settings.get("PLANNING_LLM_PROVIDER", "claude"),
            planning_ollama_model=all_settings.get("PLANNING_OLLAMA_MODEL", "llama3.1:8b"),
            planning_claude_model=all_settings.get("PLANNING_CLAUDE_MODEL", "claude-3-5-sonnet-20241022"),
            planning_gemini_model=all_settings.get("PLANNING_GEMINI_MODEL", "gemini-1.5-pro"),
            execution_llm_provider=all_settings.get("EXECUTION_LLM_PROVIDER", "gemini"),
            execution_ollama_model=all_settings.get("EXECUTION_OLLAMA_MODEL", "llama3.2:3b"),
            execution_claude_model=all_settings.get("EXECUTION_CLAUDE_MODEL", "claude-3-5-haiku-20241022"),
            execution_gemini_model=all_settings.get("EXECUTION_GEMINI_MODEL", "gemini-1.5-flash"),

            # Single LLM Configuration
            llm_provider=all_settings.get("LLM_PROVIDER", "ollama"),
            ollama_model=all_settings.get("OLLAMA_MODEL", "llama3.2:latest"),
            claude_model=all_settings.get("CLAUDE_MODEL", "claude-3-5-sonnet-20241022"),
            gemini_model=all_settings.get("GEMINI_MODEL", "gemini-1.5-flash"),

            # Provider-Specific Settings
            ollama_host=all_settings.get("OLLAMA_HOST", "http://localhost:11434"),
            ollama_auto_configure=all_settings.get("OLLAMA_AUTO_CONFIGURE", "true").lower() == "true",
            anthropic_api_key=all_settings.get("ANTHROPIC_API_KEY", ""),
            google_api_key=all_settings.get("GOOGLE_API_KEY", ""),

            # Stable Diffusion API
            sd_api_url=all_settings.get("SD_API_URL", "http://localhost:7860"),
            sd_api_timeout=int(all_settings.get("SD_API_TIMEOUT", "300")),

            # Application Settings
            app_host=all_settings.get("APP_HOST", "0.0.0.0"),
            app_port=int(all_settings.get("APP_PORT", "8000")),
            debug=all_settings.get("DEBUG", "true").lower() == "true",

            # Generation Defaults
            default_steps=int(all_settings.get("DEFAULT_STEPS", "30")),
            default_cfg_scale=float(all_settings.get("DEFAULT_CFG_SCALE", "7.0")),
            default_width=int(all_settings.get("DEFAULT_WIDTH", "512")),
            default_height=int(all_settings.get("DEFAULT_HEIGHT", "512")),
            default_sampler=all_settings.get("DEFAULT_SAMPLER", "DPM++ 2M Karras")
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get settings: {str(e)}")


@app.post("/settings/validate", response_model=SettingsValidationResponse)
async def validate_settings_endpoint(updates: SettingsUpdate):
    """
    Validate settings without applying them
    """
    try:
        # Convert SettingsUpdate to dict, excluding None values
        updates_dict = {k.upper(): str(v) for k, v in updates.dict(exclude_none=True).items()}

        # Validate
        validation_result = settings_service.validate_all_settings(updates_dict)

        return SettingsValidationResponse(
            valid=validation_result['valid'],
            results=validation_result['results']
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Validation failed: {str(e)}")


@app.post("/settings/update")
async def update_settings_endpoint(updates: SettingsUpdate):
    """
    Update application settings and save to .env file
    Note: Some changes may require application restart to take effect
    """
    try:
        # Convert SettingsUpdate to dict, excluding None values
        updates_dict = {k.upper(): str(v) for k, v in updates.dict(exclude_none=True).items()}

        # Validate first
        validation_result = settings_service.validate_all_settings(updates_dict)
        if not validation_result['valid']:
            return JSONResponse(
                status_code=400,
                content={
                    "message": "Validation failed",
                    "validation": validation_result
                }
            )

        # Update settings
        success = settings_service.update_settings(updates_dict)

        if success:
            return {
                "message": "Settings updated successfully",
                "restart_required": True,
                "note": "Please restart the application for changes to take effect"
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to update settings")

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update settings: {str(e)}")


@app.post("/settings/test-connection", response_model=ConnectionTestResponse)
async def test_connection_endpoint(request: ConnectionTestRequest):
    """
    Test a connection to a service (Ollama, Claude, Gemini, or SD)
    """
    try:
        service_type = request.service_type.lower()
        config = request.config

        if service_type == "ollama":
            # Test Ollama connection
            try:
                import httpx
                host = config.get("host", "http://localhost:11434")
                async with httpx.AsyncClient(timeout=5.0) as client:
                    response = await client.get(f"{host}/api/tags")
                    if response.status_code == 200:
                        models = response.json().get("models", [])
                        return ConnectionTestResponse(
                            success=True,
                            message=f"Ollama connected successfully",
                            details={"models_count": len(models), "models": [m["name"] for m in models]}
                        )
                    else:
                        return ConnectionTestResponse(
                            success=False,
                            message=f"Ollama returned status {response.status_code}"
                        )
            except Exception as e:
                return ConnectionTestResponse(
                    success=False,
                    message=f"Failed to connect to Ollama: {str(e)}"
                )

        elif service_type == "claude":
            # Test Claude API connection
            try:
                from anthropic import AsyncAnthropic
                api_key = config.get("api_key", "")
                if not api_key or api_key.startswith("*"):
                    return ConnectionTestResponse(
                        success=False,
                        message="Valid API key required for testing"
                    )

                client = AsyncAnthropic(api_key=api_key)
                # Test with a simple message
                response = await client.messages.create(
                    model="claude-3-5-sonnet-20241022",
                    max_tokens=10,
                    messages=[{"role": "user", "content": "test"}]
                )
                return ConnectionTestResponse(
                    success=True,
                    message="Claude API connected successfully"
                )
            except Exception as e:
                return ConnectionTestResponse(
                    success=False,
                    message=f"Failed to connect to Claude: {str(e)}"
                )

        elif service_type == "gemini":
            # Test Gemini API connection
            try:
                import google.generativeai as genai
                api_key = config.get("api_key", "")
                if not api_key or api_key.startswith("*"):
                    return ConnectionTestResponse(
                        success=False,
                        message="Valid API key required for testing"
                    )

                genai.configure(api_key=api_key)

                # List available models to verify connection
                models = genai.list_models()
                available_models = [m.name for m in models if 'generateContent' in m.supported_generation_methods]

                if available_models:
                    return ConnectionTestResponse(
                        success=True,
                        message="Gemini API connected successfully",
                        details={"models": available_models[:10]}  # Return first 10 models
                    )
                else:
                    return ConnectionTestResponse(
                        success=False,
                        message="No Gemini models available"
                    )
            except Exception as e:
                return ConnectionTestResponse(
                    success=False,
                    message=f"Failed to connect to Gemini: {str(e)}"
                )

        elif service_type == "sd":
            # Test Stable Diffusion API connection
            try:
                import httpx
                api_url = config.get("api_url", "http://localhost:7860")
                async with httpx.AsyncClient(timeout=5.0) as client:
                    response = await client.get(f"{api_url}/sdapi/v1/sd-models")
                    if response.status_code == 200:
                        models = response.json()
                        return ConnectionTestResponse(
                            success=True,
                            message=f"Stable Diffusion connected successfully",
                            details={"models_count": len(models)}
                        )
                    else:
                        return ConnectionTestResponse(
                            success=False,
                            message=f"SD API returned status {response.status_code}"
                        )
            except Exception as e:
                return ConnectionTestResponse(
                    success=False,
                    message=f"Failed to connect to Stable Diffusion: {str(e)}"
                )

        else:
            return ConnectionTestResponse(
                success=False,
                message=f"Unknown service type: {service_type}"
            )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Connection test failed: {str(e)}")


@app.post("/settings/available-models", response_model=AvailableModelsResponse)
async def get_available_models(request: AvailableModelsRequest):
    """
    Get available models for a specific provider
    """
    try:
        provider = request.provider.lower()

        if provider == "ollama":
            # Get Ollama models
            try:
                import httpx
                host = settings.ollama_host
                async with httpx.AsyncClient(timeout=5.0) as client:
                    response = await client.get(f"{host}/api/tags")
                    if response.status_code == 200:
                        models_data = response.json().get("models", [])
                        models = [m["name"] for m in models_data]
                        return AvailableModelsResponse(
                            provider="ollama",
                            models=models
                        )
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Failed to fetch Ollama models: {str(e)}")

        elif provider == "gemini":
            # Get Gemini models
            try:
                import google.generativeai as genai

                # Always use the API key from settings (loaded at startup)
                api_key = settings.google_api_key

                if not api_key or api_key == "your-api-key-here":
                    raise HTTPException(status_code=400, detail="Valid API key required. Please configure GOOGLE_API_KEY in .env file")

                genai.configure(api_key=api_key)
                models_list = genai.list_models()
                models = [m.name.replace('models/', '') for m in models_list if 'generateContent' in m.supported_generation_methods]

                return AvailableModelsResponse(
                    provider="gemini",
                    models=models
                )
            except HTTPException:
                raise
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Failed to fetch Gemini models: {str(e)}")

        elif provider == "claude":
            # Claude doesn't have a list models endpoint, return known models
            known_models = [
                "claude-3-5-sonnet-20241022",
                "claude-3-5-haiku-20241022",
                "claude-3-opus-20240229",
                "claude-3-sonnet-20240229",
                "claude-3-haiku-20240307"
            ]
            return AvailableModelsResponse(
                provider="claude",
                models=known_models
            )

        else:
            raise HTTPException(status_code=400, detail=f"Unknown provider: {provider}")

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get models: {str(e)}")


def start():
    """Start the application"""
    uvicorn.run(
        "backend.main:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=settings.debug
    )


if __name__ == "__main__":
    start()
