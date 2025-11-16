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
    AvailableModelsResponse,
    GenerationPlanRequest,
    GenerationPlan,
    ExecuteGenerationRequest,
    Img2ImgPlanRequest,
    Img2ImgGenerationPlan,
    ExecuteImg2ImgRequest,
    Img2ImgResponse
)
from backend.llm_service import LLMService
from backend.sd_service import StableDiffusionService
from backend.prompt_engine import PromptEngine
from backend.knowledge_base import KnowledgeBase
from backend.interactive_prompter import InteractivePrompter
from backend.prompt_library import PromptLibrary
from backend.settings_service import SettingsService
from backend.model_manager import ModelManager

settings = get_settings()


# Initialize services
llm_service = LLMService()
sd_service = StableDiffusionService()
knowledge_base = KnowledgeBase()
prompt_engine = PromptEngine(llm_service)
interactive_prompter = InteractivePrompter(llm_service, knowledge_base)
prompt_library = PromptLibrary()
settings_service = SettingsService()
model_manager = ModelManager()


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
        "llm_provider": provider_info.get("provider") or f"{provider_info.get('planning_provider')}/{provider_info.get('execution_provider')}",
        "llm_model": provider_info.get("model") or f"{provider_info.get('planning_model')}/{provider_info.get('execution_model')}",
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


@app.post("/plan-generation", response_model=GenerationPlan)
async def plan_generation(request: GenerationPlanRequest):
    """
    PLAN PHASE: Create a comprehensive generation plan without executing

    Returns:
    - Model recommendation with reasoning
    - Enhanced prompt with intelligent negative prompts
    - Quality analysis (specificity score, missing elements, warnings)
    - Parameter reasoning (why each parameter was chosen)
    - Tips for best results
    """
    try:
        # Get installed models
        installed_models = model_manager.get_installed_models()

        # Create comprehensive plan
        plan = await prompt_engine.create_generation_plan(
            user_input=request.user_input,
            installed_models=installed_models,
            conversation_history=request.conversation_history
        )

        return plan
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to create generation plan: {str(e)}")


@app.post("/execute-generation", response_model=GenerationResponse)
async def execute_generation(request: ExecuteGenerationRequest):
    """
    ACT PHASE: Execute a generation plan

    Takes the plan from /plan-generation and generates the image.
    Allows model override if user wants to use a different model than recommended.
    """
    try:
        plan = request.plan

        # Use model override if provided, otherwise use recommended model
        # Note: This would integrate with SD API to switch models
        # For now, we'll just use whatever model is currently loaded in SD

        if request.model_override:
            print(f"[INFO] User override: using model '{request.model_override}' instead of '{plan.model_recommendation.recommended_model_name}'")

        # Generate image using the planned prompt
        result = await sd_service.generate_image(plan.enhanced_prompt)

        return GenerationResponse(
            image_base64=result["image_base64"],
            prompt_used=plan.enhanced_prompt,
            llm_explanation=plan.explanation,
            generation_time=result["generation_time"],
            seed_used=result["seed"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Execution failed: {str(e)}")


# Img2Img Endpoints

@app.post("/plan-img2img", response_model=Img2ImgGenerationPlan)
async def plan_img2img(request: Img2ImgPlanRequest):
    """
    PLAN PHASE: Create a comprehensive img2img transformation plan without executing

    Returns:
    - Model recommendation with reasoning
    - Enhanced prompt with intelligent negative prompts
    - Denoising strength recommendation with reasoning
    - Quality analysis (specificity score, missing elements, warnings)
    - Parameter reasoning (why each parameter was chosen)
    - Tips for best transformation results
    """
    try:
        # Get installed models
        installed_models = model_manager.get_installed_models()

        # Create comprehensive img2img plan
        plan = await prompt_engine.create_img2img_plan(
            user_input=request.user_input,
            init_image_base64=request.init_image_base64,
            installed_models=installed_models,
            conversation_history=request.conversation_history
        )

        return plan
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to create img2img plan: {str(e)}")


@app.post("/execute-img2img", response_model=Img2ImgResponse)
async def execute_img2img(request: ExecuteImg2ImgRequest):
    """
    ACT PHASE: Execute an img2img transformation plan

    Takes the plan from /plan-img2img and generates the transformed image.
    Allows model override if user wants to use a different model than recommended.
    """
    try:
        plan = request.plan

        if request.model_override:
            print(f"[INFO] User override: using model '{request.model_override}' instead of '{plan.model_recommendation.recommended_model_name}'")

        # Generate image using img2img with the planned prompt and denoising strength
        result = await sd_service.img2img(
            prompt=plan.enhanced_prompt,
            init_image_base64=request.init_image_base64,
            denoising_strength=plan.denoising_strength
        )

        return Img2ImgResponse(
            image_base64=result["image_base64"],
            prompt_used=plan.enhanced_prompt,
            denoising_strength=plan.denoising_strength,
            llm_explanation=plan.explanation,
            generation_time=result["generation_time"],
            seed_used=result["seed"],
            source_image_base64=request.init_image_base64
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Img2img execution failed: {str(e)}")


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


@app.get("/loras")
async def get_loras():
    """Get available LoRAs from SD"""
    try:
        loras = await sd_service.get_loras()
        return {"loras": loras}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get LoRAs: {str(e)}")


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
            planning_claude_model=all_settings.get("PLANNING_CLAUDE_MODEL", "claude-sonnet-4-5-20250929"),
            planning_gemini_model=all_settings.get("PLANNING_GEMINI_MODEL", "gemini-2.5-pro"),
            execution_llm_provider=all_settings.get("EXECUTION_LLM_PROVIDER", "gemini"),
            execution_ollama_model=all_settings.get("EXECUTION_OLLAMA_MODEL", "llama3.2:3b"),
            execution_claude_model=all_settings.get("EXECUTION_CLAUDE_MODEL", "claude-haiku-4-5-20251001"),
            execution_gemini_model=all_settings.get("EXECUTION_GEMINI_MODEL", "gemini-2.5-flash"),

            # Single LLM Configuration
            llm_provider=all_settings.get("LLM_PROVIDER", "ollama"),
            ollama_model=all_settings.get("OLLAMA_MODEL", "llama3.2:latest"),
            claude_model=all_settings.get("CLAUDE_MODEL", "claude-sonnet-4-5-20250929"),
            gemini_model=all_settings.get("GEMINI_MODEL", "gemini-2.5-flash"),

            # Provider-Specific Settings
            ollama_host=all_settings.get("OLLAMA_HOST", "http://localhost:11434"),
            ollama_auto_configure=all_settings.get("OLLAMA_AUTO_CONFIGURE", "true").lower() == "true",
            anthropic_api_key=all_settings.get("ANTHROPIC_API_KEY", ""),
            google_api_key=all_settings.get("GOOGLE_API_KEY", ""),

            # Stable Diffusion API
            sd_api_url=all_settings.get("SD_API_URL", "http://localhost:7860"),
            sd_api_timeout=int(all_settings.get("SD_API_TIMEOUT", "300")),

            # GPU VRAM Detection
            vram_detection_mode=all_settings.get("VRAM_DETECTION_MODE", "auto"),
            vram_manual_gb=float(all_settings.get("VRAM_MANUAL_GB", "8.0")),

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

        # Filter out masked API keys (values that start with asterisks)
        # These are display-only values and should not be saved
        filtered_updates = {}
        for key, value in updates_dict.items():
            # Skip masked values (they start with asterisks)
            if value and value.startswith('*'):
                continue
            filtered_updates[key] = value

        # If no valid updates remain, return early
        if not filtered_updates:
            return {
                "success": True,
                "message": "No changes to save (all values were masked)"
            }

        # Validate first
        validation_result = settings_service.validate_all_settings(filtered_updates)
        if not validation_result['valid']:
            return JSONResponse(
                status_code=400,
                content={
                    "message": "Validation failed",
                    "validation": validation_result
                }
            )

        # Update settings
        success = settings_service.update_settings(filtered_updates)

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


@app.post("/settings/reload")
async def reload_settings_endpoint():
    """
    Reload settings from .env file without restarting the application

    This forces all services to re-read configuration from the .env file.
    Useful after updating settings to apply changes immediately.
    """
    try:
        global settings, llm_service, sd_service, prompt_engine

        # Reload settings from .env file
        settings = get_settings()

        # Reinitialize LLM service with new settings
        llm_service = LLMService()

        # Reinitialize prompt engine with new LLM service
        prompt_engine = PromptEngine(llm_service)

        # SD service doesn't need reinitialization as it reads settings dynamically
        # but we'll recreate it to ensure consistency
        sd_service = StableDiffusionService()

        # Get provider info to return to user
        provider_info = llm_service.get_provider_info()

        print("[INFO] Settings reloaded successfully")
        if provider_info.get("mode") == "dual":
            print(f"   Planning: {provider_info['planning_provider']} ({provider_info['planning_model']})")
            print(f"   Execution: {provider_info['execution_provider']} ({provider_info['execution_model']})")
        else:
            print(f"   LLM: {provider_info['provider']} ({provider_info['model']})")

        return {
            "success": True,
            "message": "Settings reloaded successfully",
            "provider_info": provider_info
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Failed to reload settings: {str(e)}"
        )


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

                # If the API key is masked (starts with asterisks), use the one from settings
                if not api_key or api_key.startswith("*"):
                    api_key = settings.anthropic_api_key

                # Check if we have a valid key now
                if not api_key or api_key == "your-api-key-here":
                    return ConnectionTestResponse(
                        success=False,
                        message="Valid API key required for testing. Please configure ANTHROPIC_API_KEY in .env file."
                    )

                client = AsyncAnthropic(api_key=api_key)
                # Test with a simple message using the latest Claude model
                response = await client.messages.create(
                    model="claude-sonnet-4-5-20250929",
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

                # If the API key is masked (starts with asterisks), use the one from settings
                if not api_key or api_key.startswith("*"):
                    api_key = settings.google_api_key

                # Check if we have a valid key now
                if not api_key or api_key == "your-api-key-here":
                    return ConnectionTestResponse(
                        success=False,
                        message="Valid API key required for testing. Please configure GOOGLE_API_KEY in .env file."
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
                "claude-sonnet-4-5-20250929",
                "claude-haiku-4-5-20251001",
                "claude-opus-4-1-20250805",
                "claude-3-7-sonnet-20250219",
                "claude-3-5-haiku-20241022"
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


# ========================================
# MODEL MANAGEMENT ENDPOINTS
# ========================================
# SD Model Management Endpoints
# ========================================

@app.get("/sd-models/recommended")
async def get_recommended_models():
    """Get curated list of recommended SD models"""
    return model_manager.get_recommended_models()


@app.get("/sd-models/installed")
async def get_installed_models():
    """Get list of installed SD models"""
    return model_manager.get_installed_models()


@app.get("/gpu/memory")
async def get_gpu_memory():
    """
    Get GPU VRAM information from Stable Diffusion API

    Returns VRAM total, free, used in GB, plus system RAM info
    """
    try:
        memory_info = await sd_service.get_memory_info()
        return memory_info
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get GPU memory info: {str(e)}"
        )


@app.get("/gpu/vram")
async def get_vram_settings():
    """
    Get current VRAM detection settings and detected/configured VRAM

    Returns:
    {
        "mode": "auto" | "manual" | "disabled",
        "manual_gb": 8.0,
        "detected_gb": 8.0 (if mode is auto and detection successful),
        "effective_gb": 8.0 (the value that will be used),
        "detection_successful": true/false
    }
    """
    try:
        settings = get_settings()
        mode = settings.vram_detection_mode
        manual_gb = settings.vram_manual_gb

        result = {
            "mode": mode,
            "manual_gb": manual_gb,
            "detected_gb": None,
            "effective_gb": None,
            "detection_successful": False
        }

        if mode == "auto":
            try:
                memory_info = await sd_service.get_memory_info()
                detected_vram = memory_info["vram_total_gb"]
                result["detected_gb"] = detected_vram
                result["effective_gb"] = detected_vram
                result["detection_successful"] = True
            except Exception as e:
                print(f"[VRAM] Auto-detection failed: {e}")
                result["detection_successful"] = False
                result["effective_gb"] = None
        elif mode == "manual":
            result["effective_gb"] = manual_gb
        else:  # disabled
            result["effective_gb"] = None

        return result
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get VRAM settings: {str(e)}"
        )


@app.get("/sd-models/directory")
async def get_models_directory():
    """Get the current models directory path"""
    return {"path": model_manager.get_models_directory()}


@app.post("/sd-models/directory")
async def set_models_directory(request: dict):
    """Set a custom models directory"""
    path = request.get("path")
    if not path:
        raise HTTPException(status_code=400, detail="Path is required")

    success = model_manager.set_models_directory(path)
    if success:
        return {"success": True, "path": model_manager.get_models_directory()}
    else:
        raise HTTPException(status_code=400, detail="Invalid path")


@app.delete("/sd-models/{filename}")
async def delete_model(filename: str):
    """Delete an installed model"""
    success = model_manager.delete_model(filename)
    if success:
        return {"success": True, "message": f"Model {filename} deleted"}
    else:
        raise HTTPException(status_code=404, detail="Model not found")


@app.get("/sd-models/downloads")
async def get_downloads_models():
    """
    Get list of .safetensors files in the user's Downloads folder
    """
    from pathlib import Path

    # Get user's Downloads directory
    downloads_dir = Path.home() / "Downloads"

    if not downloads_dir.exists():
        return {
            "downloads_directory": str(downloads_dir),
            "models": []
        }

    # Find all .safetensors files
    models = []
    for file in downloads_dir.glob("*.safetensors"):
        size_mb = file.stat().st_size / (1024 * 1024)
        models.append({
            "name": file.stem,
            "filename": file.name,
            "path": str(file),
            "size": f"{size_mb:.1f} MB",
            "size_bytes": file.stat().st_size
        })

    # Sort by modification time (newest first)
    models.sort(key=lambda x: Path(x["path"]).stat().st_mtime, reverse=True)

    return {
        "downloads_directory": str(downloads_dir),
        "models": models
    }


@app.post("/sd-models/import")
async def import_model(request: dict):
    """
    Import a model from an external location (e.g., Downloads folder) to the models directory

    Request body:
    {
        "source_path": "/path/to/downloaded/model.safetensors",
        "move": true  // Optional: if true, moves file (removes from source). If false, copies it. Default: true
    }
    """
    source_path = request.get("source_path")
    if not source_path:
        raise HTTPException(status_code=400, detail="source_path is required")

    move = request.get("move", True)  # Default to moving (cleaning up Downloads)

    result = model_manager.import_model(source_path, move=move)

    if result["success"]:
        return result
    else:
        raise HTTPException(status_code=400, detail=result["error"])


@app.post("/sd-models/recommend")
async def recommend_model(request: dict):
    """
    Use AI to recommend the best SD model for a given prompt

    Request body:
    {
        "prompt": "user's image description"
    }
    """
    import json

    prompt = request.get("prompt")
    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt is required")

    try:
        # Get installed models
        installed_models = model_manager.get_installed_models()

        # Generate recommendation prompt
        system_prompt = model_manager.get_model_recommendation_prompt(prompt, installed_models)

        # Use execution LLM for speed (this is a quick task)
        response = await llm_service.generate(
            prompt="Analyze and recommend",
            system_prompt=system_prompt,
            temperature=0.3,  # Lower temperature for more consistent recommendations
            use_planning_llm=False  # Use fast execution LLM
        )

        # Parse JSON response
        response = response.strip()
        if "```json" in response:
            response = response.split("```json")[1].split("```")[0].strip()
        elif "```" in response:
            response = response.split("```")[1].split("```")[0].strip()

        recommendation = json.loads(response)

        # Find the recommended model details
        recommended_model = None
        for model in model_manager.RECOMMENDED_MODELS:
            if model["name"].lower() in recommendation["recommended_model"].lower():
                recommended_model = model
                break

        return {
            "recommended_model": recommendation.get("recommended_model"),
            "is_installed": recommendation.get("is_installed", False),
            "reason": recommendation.get("reason"),
            "alternative": recommendation.get("alternative"),
            "model_details": recommended_model,
            "installed_models": installed_models
        }

    except json.JSONDecodeError as e:
        print(f"Failed to parse LLM recommendation: {e}")
        print(f"Response was: {response}")

        # Fallback recommendation
        return {
            "recommended_model": "DreamShaper 8",
            "is_installed": False,
            "reason": "General-purpose model recommended as fallback",
            "model_details": model_manager.RECOMMENDED_MODELS[0],
            "installed_models": installed_models
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate recommendation: {str(e)}")


# Global dictionary to track download progress
download_progress = {}


@app.post("/sd-models/download")
async def download_model_endpoint(request: dict):
    """
    Download a model from CivitAI

    Request body:
    {
        "model_id": "4384",  # CivitAI model ID
        "model_name": "DreamShaper 8"  # For user feedback
    }
    """
    from fastapi import BackgroundTasks
    import asyncio
    import uuid

    model_id = request.get("model_id")
    model_name = request.get("model_name", "Unknown Model")

    if not model_id:
        raise HTTPException(status_code=400, detail="Model ID is required")

    # Generate a unique download ID
    download_id = str(uuid.uuid4())

    # Initialize progress tracking
    download_progress[download_id] = {
        "status": "fetching_info",
        "model_name": model_name,
        "progress": 0,
        "downloaded": 0,
        "total": 0,
        "error": None
    }

    # Start download in background
    async def download_task():
        try:
            # Get download URL from CivitAI
            print(f"[DOWNLOAD] Starting download for model_id: {model_id}, download_id: {download_id}")
            download_progress[download_id]["status"] = "fetching_url"
            download_info = await model_manager.get_download_url_for_model(model_id)

            if not download_info:
                print(f"[DOWNLOAD] Failed to get download info for model_id: {model_id}")
                download_progress[download_id]["status"] = "error"
                download_progress[download_id]["error"] = "Failed to get download URL from CivitAI"
                return

            download_url = download_info["download_url"]
            filename = download_info["filename"]
            total_size = download_info["size"]

            print(f"[DOWNLOAD] Got download URL, filename: {filename}, size: {total_size} bytes")
            download_progress[download_id]["status"] = "downloading"
            download_progress[download_id]["total"] = total_size
            download_progress[download_id]["filename"] = filename

            # Progress callback
            def progress_callback(downloaded, total):
                download_progress[download_id]["downloaded"] = downloaded
                download_progress[download_id]["total"] = total
                download_progress[download_id]["progress"] = int((downloaded / total) * 100) if total > 0 else 0
                if downloaded % (10 * 1024 * 1024) == 0:  # Log every 10MB
                    print(f"[DOWNLOAD] Progress: {downloaded}/{total} bytes ({download_progress[download_id]['progress']}%)")

            # Download the model
            print(f"[DOWNLOAD] Starting actual download...")
            result = await model_manager.download_model(download_url, filename, progress_callback)

            if result["success"]:
                print(f"[DOWNLOAD] Download completed successfully: {filename}")
                download_progress[download_id]["status"] = "completed"
                download_progress[download_id]["progress"] = 100
                download_progress[download_id]["path"] = result["path"]
            else:
                print(f"[DOWNLOAD] Download failed: {result.get('error', 'Unknown error')}")
                download_progress[download_id]["status"] = "error"
                download_progress[download_id]["error"] = result.get("error", "Unknown error")

        except Exception as e:
            print(f"[DOWNLOAD] Exception during download: {str(e)}")
            import traceback
            traceback.print_exc()
            download_progress[download_id]["status"] = "error"
            download_progress[download_id]["error"] = str(e)

    # Start the download task
    asyncio.create_task(download_task())

    return {
        "download_id": download_id,
        "message": f"Download started for {model_name}",
        "status": "started"
    }


@app.get("/sd-models/download/{download_id}/progress")
async def get_download_progress_endpoint(download_id: str):
    """Get the progress of a model download"""
    print(f"[PROGRESS] Request for download_id: {download_id}")
    print(f"[PROGRESS] Available downloads: {list(download_progress.keys())}")

    if download_id not in download_progress:
        print(f"[PROGRESS] Download not found: {download_id}")
        raise HTTPException(status_code=404, detail="Download not found")

    progress_data = download_progress[download_id]
    print(f"[PROGRESS] Returning progress: {progress_data}")
    return progress_data


def start():
    """Start the application"""
    # When in debug mode, watch .env file for changes and auto-reload
    reload_includes = None
    if settings.debug:
        reload_includes = ["*.env"]

    uvicorn.run(
        "backend.main:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=settings.debug,
        reload_includes=reload_includes
    )


if __name__ == "__main__":
    start()
