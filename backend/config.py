from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import dotenv_values
import os


class Settings(BaseSettings):
    """Application configuration"""

    # Dual-LLM Configuration
    # Set to True to use different LLMs for planning vs execution
    use_dual_llm: bool = False

    # Planning LLM (for prompt engineering/refinement)
    # Options: "ollama", "claude", "gemini"
    planning_llm_provider: str = "claude"
    planning_ollama_model: str = "llama3.2:latest"
    planning_claude_model: str = "claude-sonnet-4-5-20250929"
    planning_gemini_model: str = "gemini-2.5-pro"

    # Execution LLM (for quick tasks/iterations)
    # Options: "ollama", "claude", "gemini"
    execution_llm_provider: str = "gemini"
    execution_ollama_model: str = "llama3.2:latest"
    execution_claude_model: str = "claude-haiku-4-5-20251001"
    execution_gemini_model: str = "gemini-2.5-flash"

    # Single LLM Configuration (when use_dual_llm=False)
    # Options: "ollama", "claude", "gemini"
    llm_provider: str = "ollama"

    # Ollama Configuration (local LLM)
    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "llama3.2:latest"
    ollama_auto_configure: bool = True  # Auto-select model based on VRAM

    # Anthropic Claude Configuration
    anthropic_api_key: str = "your-api-key-here"
    claude_model: str = "claude-sonnet-4-5-20250929"  # or claude-haiku-4-5-20251001 for speed

    # Google Gemini Configuration
    google_api_key: str = "your-api-key-here"
    gemini_model: str = "gemini-2.5-flash"  # or gemini-2.5-pro for better quality

    # Stable Diffusion API
    sd_api_url: str = "http://localhost:7860"
    sd_api_timeout: int = 300

    # CivitAI API (for model downloads)
    civitai_api_key: str = ""

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

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        # Prioritize .env file over environment variables
        env_ignore_empty=True,
        validate_default=True
    )


def get_settings() -> Settings:
    """Get settings instance (reloads from .env each time)"""
    # Load directly from .env file, bypassing environment variables
    # This is necessary because environment variables may have stale values
    env_values = dotenv_values(".env")

    # Temporarily clear the problematic environment variables
    env_backup = {}
    for key in env_values.keys():
        if key in os.environ:
            env_backup[key] = os.environ[key]
            del os.environ[key]

    # Create settings instance (will now use .env values)
    settings = Settings(**env_values)

    # Restore environment variables
    for key, value in env_backup.items():
        os.environ[key] = value

    return settings
