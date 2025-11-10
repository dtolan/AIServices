import httpx
from typing import List, Dict, Optional
from .base import BaseLLMProvider


class OllamaProvider(BaseLLMProvider):
    """Ollama (local) LLM provider"""

    def __init__(self, base_url: str, model: str, timeout: float = 120.0):
        self.base_url = base_url
        self.model = model
        self.timeout = timeout

    async def health_check(self) -> bool:
        """Check if Ollama is running"""
        async with httpx.AsyncClient(timeout=5.0) as client:
            try:
                response = await client.get(f"{self.base_url}/api/tags")
                return response.status_code == 200
            except Exception as e:
                raise Exception(f"Ollama not available: {e}")

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None
    ) -> str:
        """Generate text using Ollama"""
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": temperature,
                }
            }

            if system_prompt:
                payload["system"] = system_prompt

            if max_tokens:
                payload["options"]["num_predict"] = max_tokens

            try:
                response = await client.post(
                    f"{self.base_url}/api/generate",
                    json=payload
                )
                response.raise_for_status()
                result = response.json()
                return result.get("response", "")
            except httpx.HTTPError as e:
                raise Exception(f"Ollama generation failed: {e}")

    async def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None
    ) -> str:
        """Chat with Ollama using conversation history"""
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            payload = {
                "model": self.model,
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": temperature,
                }
            }

            if max_tokens:
                payload["options"]["num_predict"] = max_tokens

            try:
                response = await client.post(
                    f"{self.base_url}/api/chat",
                    json=payload
                )
                response.raise_for_status()
                result = response.json()
                return result.get("message", {}).get("content", "")
            except httpx.HTTPError as e:
                raise Exception(f"Ollama chat failed: {e}")

    def get_provider_name(self) -> str:
        return "Ollama (Local)"

    def get_model_name(self) -> str:
        return self.model

    async def get_available_models(self) -> List[str]:
        """Get list of available Ollama models"""
        async with httpx.AsyncClient(timeout=5.0) as client:
            try:
                response = await client.get(f"{self.base_url}/api/tags")
                response.raise_for_status()
                models = response.json().get("models", [])
                return [model["name"] for model in models]
            except httpx.HTTPError as e:
                raise Exception(f"Failed to get models: {e}")
