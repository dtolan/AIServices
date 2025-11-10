from typing import List, Dict, Optional
from enum import Enum
from backend.config import get_settings
from backend.llm_providers import (
    BaseLLMProvider,
    OllamaProvider,
    ClaudeProvider,
    GeminiProvider
)
from backend.hardware_detector import HardwareDetector


class LLMRole(Enum):
    """Role of the LLM in dual-LLM setup"""
    PLANNING = "planning"      # For initial prompt engineering and complex reasoning
    EXECUTION = "execution"    # For quick iterations and refinements
    SINGLE = "single"          # Single LLM mode


class LLMService:
    """
    Unified LLM service that supports multiple providers (Ollama, Claude, Gemini)
    with dual-LLM capability for planning vs execution
    """

    def __init__(self):
        # Load settings fresh for each instance
        self.settings = get_settings()

        # Auto-configure Ollama if enabled
        if self.settings.ollama_auto_configure:
            self._auto_configure_ollama()

        # Initialize providers based on mode
        if self.settings.use_dual_llm:
            self.planning_provider = self._initialize_provider(LLMRole.PLANNING)
            self.execution_provider = self._initialize_provider(LLMRole.EXECUTION)
            self.provider = self.planning_provider  # Default to planning
        else:
            self.provider = self._initialize_provider(LLMRole.SINGLE)
            self.planning_provider = self.provider
            self.execution_provider = self.provider

    def _auto_configure_ollama(self):
        """Auto-configure Ollama models based on available hardware"""
        hardware = HardwareDetector.get_hardware_summary()

        if hardware["gpu_detected"]:
            if self.settings.use_dual_llm:
                # Configure dual LLM with recommended models
                planning_rec = hardware["recommendations"]["dual_llm"]["planning"]
                execution_rec = hardware["recommendations"]["dual_llm"]["execution"]

                self.settings.planning_ollama_model = planning_rec["model"]
                self.settings.execution_ollama_model = execution_rec["model"]

                print(f"[AUTO-CONFIG] Ollama (Dual-LLM):")
                print(f"   Planning: {planning_rec['model']} ({planning_rec['size']}) - {planning_rec['reason']}")
                print(f"   Execution: {execution_rec['model']} ({execution_rec['size']}) - {execution_rec['reason']}")
            else:
                # Configure single LLM
                single_rec = hardware["recommendations"]["single_llm"]
                self.settings.ollama_model = single_rec["model"]

                print(f"[AUTO-CONFIG] Ollama: {single_rec['model']} ({single_rec['size']}) - {single_rec['reason']}")
        else:
            print("[WARNING] No GPU detected. Ollama will use CPU (slower). Consider using cloud LLMs.")

    def _initialize_provider(self, role: LLMRole) -> BaseLLMProvider:
        """Initialize the configured LLM provider for a specific role"""

        # Determine which provider and model to use
        if role == LLMRole.PLANNING:
            provider_name = self.settings.planning_llm_provider.lower()
            ollama_model = self.settings.planning_ollama_model
            claude_model = self.settings.planning_claude_model
            gemini_model = self.settings.planning_gemini_model
        elif role == LLMRole.EXECUTION:
            provider_name = self.settings.execution_llm_provider.lower()
            ollama_model = self.settings.execution_ollama_model
            claude_model = self.settings.execution_claude_model
            gemini_model = self.settings.execution_gemini_model
        else:  # SINGLE
            provider_name = self.settings.llm_provider.lower()
            ollama_model = self.settings.ollama_model
            claude_model = self.settings.claude_model
            gemini_model = self.settings.gemini_model

        # Create provider
        if provider_name == "ollama":
            return OllamaProvider(
                base_url=self.settings.ollama_host,
                model=ollama_model,
                timeout=120.0
            )
        elif provider_name == "claude":
            return ClaudeProvider(
                api_key=self.settings.anthropic_api_key,
                model=claude_model,
                timeout=120.0
            )
        elif provider_name == "gemini":
            return GeminiProvider(
                api_key=self.settings.google_api_key,
                model=gemini_model,
                timeout=120.0
            )
        else:
            raise ValueError(f"Unknown LLM provider: {provider_name}")

    async def health_check(self) -> bool:
        """Check if the active LLM provider is available"""
        if self.settings.use_dual_llm:
            # Check both providers
            planning_ok = await self.planning_provider.health_check()
            execution_ok = await self.execution_provider.health_check()
            return planning_ok and execution_ok
        else:
            return await self.provider.health_check()

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        use_planning_llm: bool = True
    ) -> str:
        """
        Generate text using the appropriate LLM provider

        Args:
            prompt: The user prompt
            system_prompt: Optional system prompt for context
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum tokens to generate
            use_planning_llm: If True and dual-LLM enabled, use planning LLM. Otherwise execution LLM.

        Returns:
            Generated text response
        """
        # Select provider based on mode and preference
        if self.settings.use_dual_llm:
            active_provider = self.planning_provider if use_planning_llm else self.execution_provider
        else:
            active_provider = self.provider

        return await active_provider.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens
        )

    async def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        use_planning_llm: bool = True
    ) -> str:
        """
        Chat with the LLM using conversation history

        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            use_planning_llm: If True and dual-LLM enabled, use planning LLM. Otherwise execution LLM.

        Returns:
            Generated response
        """
        # Select provider based on mode and preference
        if self.settings.use_dual_llm:
            active_provider = self.planning_provider if use_planning_llm else self.execution_provider
        else:
            active_provider = self.provider

        return await active_provider.chat(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )

    def get_provider_info(self) -> Dict[str, str]:
        """Get information about the current provider(s)"""
        if self.settings.use_dual_llm:
            return {
                "mode": "dual",
                "planning_provider": self.planning_provider.get_provider_name(),
                "planning_model": self.planning_provider.get_model_name(),
                "execution_provider": self.execution_provider.get_provider_name(),
                "execution_model": self.execution_provider.get_model_name()
            }
        else:
            return {
                "mode": "single",
                "provider": self.provider.get_provider_name(),
                "model": self.provider.get_model_name()
            }

    async def get_available_models(self) -> List[str]:
        """Get list of available models (Ollama only)"""
        if hasattr(self.provider, 'get_available_models'):
            return await self.provider.get_available_models()
        return []
