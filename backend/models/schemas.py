from pydantic import BaseModel, Field
from typing import Optional, List
from enum import Enum


class PromptRequest(BaseModel):
    """User's natural language request for image generation"""
    user_input: str = Field(..., description="Natural language description of desired image")
    conversation_history: Optional[List[dict]] = Field(default=None, description="Previous conversation context")


class SDPrompt(BaseModel):
    """Structured Stable Diffusion prompt"""
    positive_prompt: str = Field(..., description="Positive prompt for SD")
    negative_prompt: str = Field(default="", description="Negative prompt for SD")
    steps: int = Field(default=30, ge=1, le=150)
    cfg_scale: float = Field(default=7.0, ge=1.0, le=30.0)
    width: int = Field(default=512, ge=64, le=2048)
    height: int = Field(default=512, ge=64, le=2048)
    sampler_name: str = Field(default="DPM++ 2M Karras")
    seed: int = Field(default=-1)


class GenerationResponse(BaseModel):
    """Response containing the generated image and metadata"""
    image_base64: str
    prompt_used: SDPrompt
    llm_explanation: str = Field(..., description="LLM's explanation of prompt choices")
    generation_time: float
    seed_used: int


class PromptEnhancementResponse(BaseModel):
    """LLM's enhanced prompt suggestion"""
    enhanced_prompt: SDPrompt
    explanation: str = Field(..., description="Why these prompt choices were made")
    suggestions: List[str] = Field(default=[], description="Additional suggestions for iteration")


class IterationRequest(BaseModel):
    """Request to iterate on a previous generation"""
    previous_prompt: SDPrompt
    previous_image_base64: str
    user_feedback: str = Field(..., description="What the user wants to change or improve")


class ConversationMessage(BaseModel):
    """Single message in conversation history"""
    role: str = Field(..., description="'user' or 'assistant'")
    content: str
    prompt: Optional[SDPrompt] = None
    image_base64: Optional[str] = None


class InteractivePromptRequest(BaseModel):
    """Request for interactive prompt creation with questions"""
    user_input: str = Field(..., description="Natural language description")
    conversation_history: Optional[List[dict]] = Field(default=None, description="Previous Q&A context")
    skip_questions: bool = Field(default=False, description="Skip questions and generate directly")


class QuestionResponse(BaseModel):
    """Response with clarifying questions"""
    needs_clarification: bool
    questions: List[str] = Field(default=[], description="Clarifying questions to ask")
    reasoning: str = Field(default="", description="Why these questions are being asked")


class AnswerRequest(BaseModel):
    """User's answers to clarifying questions"""
    conversation_history: List[dict] = Field(..., description="Full conversation including questions and answers")


class KnowledgeBaseInfo(BaseModel):
    """Information about available knowledge base"""
    guides_count: int
    templates_count: int
    available_guides: List[str]
    available_templates: List[str]


class SettingsResponse(BaseModel):
    """Current application settings (safe for client display)"""
    # Dual-LLM Configuration
    use_dual_llm: bool
    planning_llm_provider: str
    planning_ollama_model: str
    planning_claude_model: str
    planning_gemini_model: str
    execution_llm_provider: str
    execution_ollama_model: str
    execution_claude_model: str
    execution_gemini_model: str

    # Single LLM Configuration
    llm_provider: str
    ollama_model: str
    claude_model: str
    gemini_model: str

    # Provider-Specific Settings
    ollama_host: str
    ollama_auto_configure: bool
    anthropic_api_key: str  # Will be masked
    google_api_key: str  # Will be masked

    # Stable Diffusion API
    sd_api_url: str
    sd_api_timeout: int

    # Application Settings
    app_host: str
    app_port: int
    debug: bool

    # Generation Defaults
    default_steps: int
    default_cfg_scale: float
    default_width: int
    default_height: int
    default_sampler: str


class SettingsUpdate(BaseModel):
    """Settings update request"""
    # All fields optional to allow partial updates
    use_dual_llm: Optional[bool] = None
    planning_llm_provider: Optional[str] = None
    planning_ollama_model: Optional[str] = None
    planning_claude_model: Optional[str] = None
    planning_gemini_model: Optional[str] = None
    execution_llm_provider: Optional[str] = None
    execution_ollama_model: Optional[str] = None
    execution_claude_model: Optional[str] = None
    execution_gemini_model: Optional[str] = None

    llm_provider: Optional[str] = None
    ollama_model: Optional[str] = None
    claude_model: Optional[str] = None
    gemini_model: Optional[str] = None

    ollama_host: Optional[str] = None
    ollama_auto_configure: Optional[bool] = None
    anthropic_api_key: Optional[str] = None
    google_api_key: Optional[str] = None

    sd_api_url: Optional[str] = None
    sd_api_timeout: Optional[int] = None

    app_host: Optional[str] = None
    app_port: Optional[int] = None
    debug: Optional[bool] = None

    default_steps: Optional[int] = None
    default_cfg_scale: Optional[float] = None
    default_width: Optional[int] = None
    default_height: Optional[int] = None
    default_sampler: Optional[str] = None


class ValidationResult(BaseModel):
    """Validation result for a single setting"""
    valid: bool
    errors: List[str] = []
    warnings: List[str] = []


class SettingsValidationResponse(BaseModel):
    """Response from settings validation"""
    valid: bool
    results: dict  # Dict[str, ValidationResult]


class ConnectionTestRequest(BaseModel):
    """Request to test a service connection"""
    service_type: str  # 'ollama', 'claude', 'gemini', 'sd'
    config: dict  # Configuration to test


class ConnectionTestResponse(BaseModel):
    """Response from connection test"""
    success: bool
    message: str
    details: Optional[dict] = None


class AvailableModelsRequest(BaseModel):
    """Request to get available models for a provider"""
    provider: str  # 'ollama', 'claude', 'gemini'
    api_key: Optional[str] = None  # Required for claude/gemini


class AvailableModelsResponse(BaseModel):
    """Response with available models"""
    provider: str
    models: List[str]


# New schemas for Plan/Act workflow
class ModelRecommendation(BaseModel):
    """AI recommendation for best SD model"""
    model_config = {"protected_namespaces": ()}  # Disable model_ namespace protection

    recommended_model_name: str
    model_filename: Optional[str] = None
    is_installed: bool
    reason: str
    alternative: Optional[str] = None
    model_details: Optional[dict] = None  # Full model info from RECOMMENDED_MODELS


class QualityAnalysis(BaseModel):
    """Analysis of prompt quality and suggestions"""
    specificity_score: float = Field(..., ge=0.0, le=1.0, description="How specific is the prompt (0-1)")
    missing_elements: List[str] = Field(default=[], description="Important missing elements")
    warnings: List[str] = Field(default=[], description="Potential issues or conflicts")
    strengths: List[str] = Field(default=[], description="What's good about the prompt")
    category: str = Field(..., description="Detected category: portrait, landscape, anime, etc.")


class ParameterReasoning(BaseModel):
    """Reasoning for each parameter choice"""
    steps_reason: str
    cfg_reason: str
    resolution_reason: str
    sampler_reason: str
    aspect_ratio: str = Field(..., description="portrait, landscape, or square")


class GenerationPlan(BaseModel):
    """Complete generation plan before execution"""
    model_config = {"protected_namespaces": ()}  # Disable model_ namespace protection

    user_input: str
    model_recommendation: ModelRecommendation
    enhanced_prompt: SDPrompt
    quality_analysis: QualityAnalysis
    parameter_reasoning: ParameterReasoning
    explanation: str
    tips: List[str] = Field(default=[], description="Helpful tips for this generation")
    estimated_time: Optional[float] = Field(default=None, description="Estimated generation time in seconds")


class GenerationPlanRequest(BaseModel):
    """Request for generation plan (PLAN phase)"""
    user_input: str = Field(..., description="Natural language description of desired image")
    conversation_history: Optional[List[dict]] = Field(default=None, description="Previous conversation context")


class ExecuteGenerationRequest(BaseModel):
    """Request to execute a generation plan (ACT phase)"""
    model_config = {"protected_namespaces": ()}  # Disable model_ namespace protection

    plan: GenerationPlan
    model_override: Optional[str] = Field(default=None, description="Override recommended model")


# Img2Img Schemas
class Img2ImgPlanRequest(BaseModel):
    """Request for img2img generation plan (PLAN phase)"""
    user_input: str = Field(..., description="Natural language description of desired transformation")
    init_image_base64: str = Field(..., description="Base64 encoded source image")
    conversation_history: Optional[List[dict]] = Field(default=None, description="Previous conversation context")


class Img2ImgGenerationPlan(BaseModel):
    """Complete img2img generation plan before execution"""
    model_config = {"protected_namespaces": ()}  # Disable model_ namespace protection

    user_input: str
    model_recommendation: ModelRecommendation
    enhanced_prompt: SDPrompt
    quality_analysis: QualityAnalysis
    parameter_reasoning: ParameterReasoning
    denoising_strength: float = Field(default=0.7, ge=0.0, le=1.0, description="How much to transform the image")
    denoising_reason: str = Field(..., description="Why this denoising strength was chosen")
    explanation: str
    tips: List[str] = Field(default=[], description="Helpful tips for this img2img generation")
    estimated_time: Optional[float] = Field(default=None, description="Estimated generation time in seconds")


class ExecuteImg2ImgRequest(BaseModel):
    """Request to execute an img2img generation plan (ACT phase)"""
    model_config = {"protected_namespaces": ()}  # Disable model_ namespace protection

    plan: Img2ImgGenerationPlan
    init_image_base64: str = Field(..., description="Base64 encoded source image")
    model_override: Optional[str] = Field(default=None, description="Override recommended model")


class Img2ImgResponse(BaseModel):
    """Response from img2img generation"""
    image_base64: str
    prompt_used: SDPrompt
    denoising_strength: float
    llm_explanation: str = Field(..., description="LLM's explanation of choices")
    generation_time: float
    seed_used: int
    source_image_base64: str = Field(..., description="Original source image for reference")
