from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    """Application configuration"""

    # Dual-LLM Configuration
    # Set to True to use different LLMs for planning vs execution
    use_dual_llm: bool = False

    # Planning LLM (for prompt engineering/refinement)
    # Options: "ollama", "claude", "gemini"
    planning_llm_provider: str = "claude"
    planning_ollama_model: str = "llama3.2:latest"
    planning_claude_model: str = "claude-3-5-sonnet-20241022"
    planning_gemini_model: str = "gemini-1.5-pro"

    # Execution LLM (for quick tasks/iterations)
    # Options: "ollama", "claude", "gemini"
    execution_llm_provider: str = "gemini"
    execution_ollama_model: str = "llama3.2:latest"
    execution_claude_model: str = "claude-3-5-haiku-20241022"
    execution_gemini_model: str = "gemini-1.5-flash"

    # Single LLM Configuration (when use_dual_llm=False)
    # Options: "ollama", "claude", "gemini"
    llm_provider: str = "ollama"

    # Ollama Configuration (local LLM)
    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "llama3.2:latest"
    ollama_auto_configure: bool = True  # Auto-select model based on VRAM

    # Anthropic Claude Configuration
    anthropic_api_key: str = "your-api-key-here"
    claude_model: str = "claude-3-5-sonnet-20241022"  # or claude-3-5-haiku-20241022 for speed

    # Google Gemini Configuration
    google_api_key: str = "your-api-key-here"
    gemini_model: str = "gemini-1.5-flash"  # or gemini-1.5-pro for better quality

    # Stable Diffusion API
    sd_api_url: str = "http://localhost:7860"
    sd_api_timeout: int = 300

    # Application
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    debug: bool = True

    # Generation defaults
    default_steps: int = 30
    default_cfg_scale: float = 7.0
    default_width: int = 512
    default_height: int = 512
    default_sampler: str = "DPM++ 2M Karras"

    class Config:
        env_file = ".env"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()
