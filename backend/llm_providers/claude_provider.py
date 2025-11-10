import httpx
from typing import List, Dict, Optional
from .base import BaseLLMProvider


class ClaudeProvider(BaseLLMProvider):
    """Anthropic Claude LLM provider"""

    def __init__(self, api_key: str, model: str = "claude-3-5-sonnet-20241022", timeout: float = 120.0):
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.base_url = "https://api.anthropic.com/v1"

    async def health_check(self) -> bool:
        """Check if Claude API is available"""
        if not self.api_key or self.api_key == "your-api-key-here":
            raise Exception("Claude API key not configured")

        # Simple check - if we have an API key, assume it's healthy
        # Actual validation happens on first request
        return True

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None
    ) -> str:
        """Generate text using Claude"""
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            headers = {
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json"
            }

            messages = [{"role": "user", "content": prompt}]

            payload = {
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens or 4096
            }

            if system_prompt:
                payload["system"] = system_prompt

            try:
                response = await client.post(
                    f"{self.base_url}/messages",
                    headers=headers,
                    json=payload
                )
                response.raise_for_status()
                result = response.json()

                # Extract text from Claude's response format
                if result.get("content") and len(result["content"]) > 0:
                    return result["content"][0]["text"]
                return ""

            except httpx.HTTPError as e:
                error_detail = ""
                try:
                    error_detail = e.response.json() if hasattr(e, 'response') else str(e)
                except:
                    error_detail = str(e)
                raise Exception(f"Claude API failed: {error_detail}")

    async def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None
    ) -> str:
        """Chat with Claude using conversation history"""
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            headers = {
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json"
            }

            # Extract system message if present
            system_prompt = None
            claude_messages = []

            for msg in messages:
                if msg["role"] == "system":
                    system_prompt = msg["content"]
                else:
                    # Claude uses "user" and "assistant" roles
                    claude_messages.append({
                        "role": msg["role"],
                        "content": msg["content"]
                    })

            payload = {
                "model": self.model,
                "messages": claude_messages,
                "temperature": temperature,
                "max_tokens": max_tokens or 4096
            }

            if system_prompt:
                payload["system"] = system_prompt

            try:
                response = await client.post(
                    f"{self.base_url}/messages",
                    headers=headers,
                    json=payload
                )
                response.raise_for_status()
                result = response.json()

                if result.get("content") and len(result["content"]) > 0:
                    return result["content"][0]["text"]
                return ""

            except httpx.HTTPError as e:
                error_detail = ""
                try:
                    error_detail = e.response.json() if hasattr(e, 'response') else str(e)
                except:
                    error_detail = str(e)
                raise Exception(f"Claude API failed: {error_detail}")

    def get_provider_name(self) -> str:
        return "Anthropic Claude"

    def get_model_name(self) -> str:
        return self.model
