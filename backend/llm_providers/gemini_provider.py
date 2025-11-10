import httpx
from typing import List, Dict, Optional
from .base import BaseLLMProvider


class GeminiProvider(BaseLLMProvider):
    """Google Gemini LLM provider"""

    def __init__(self, api_key: str, model: str = "gemini-1.5-flash", timeout: float = 120.0):
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.base_url = "https://generativelanguage.googleapis.com/v1beta"

    async def health_check(self) -> bool:
        """Check if Gemini API is available"""
        if not self.api_key or self.api_key == "your-api-key-here":
            raise Exception("Gemini API key not configured")

        # Simple check - if we have an API key, assume it's healthy
        return True

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None
    ) -> str:
        """Generate text using Gemini"""
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            # Build the full prompt with system context if provided
            full_prompt = prompt
            if system_prompt:
                full_prompt = f"{system_prompt}\n\n{prompt}"

            payload = {
                "contents": [{
                    "parts": [{
                        "text": full_prompt
                    }]
                }],
                "generationConfig": {
                    "temperature": temperature,
                    "maxOutputTokens": max_tokens or 8192,
                }
            }

            try:
                response = await client.post(
                    f"{self.base_url}/models/{self.model}:generateContent?key={self.api_key}",
                    json=payload
                )
                response.raise_for_status()
                result = response.json()

                # Extract text from Gemini's response format
                if result.get("candidates") and len(result["candidates"]) > 0:
                    candidate = result["candidates"][0]
                    if candidate.get("content") and candidate["content"].get("parts"):
                        return candidate["content"]["parts"][0].get("text", "")
                return ""

            except httpx.HTTPError as e:
                error_detail = ""
                try:
                    error_detail = e.response.json() if hasattr(e, 'response') else str(e)
                except:
                    error_detail = str(e)
                raise Exception(f"Gemini API failed: {error_detail}")

    async def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None
    ) -> str:
        """Chat with Gemini using conversation history"""
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            # Convert messages to Gemini format
            # Gemini uses "user" and "model" roles
            system_instruction = None
            gemini_contents = []

            for msg in messages:
                if msg["role"] == "system":
                    system_instruction = msg["content"]
                else:
                    role = "model" if msg["role"] == "assistant" else "user"
                    gemini_contents.append({
                        "role": role,
                        "parts": [{"text": msg["content"]}]
                    })

            payload = {
                "contents": gemini_contents,
                "generationConfig": {
                    "temperature": temperature,
                    "maxOutputTokens": max_tokens or 8192,
                }
            }

            # Add system instruction if present
            if system_instruction:
                payload["systemInstruction"] = {
                    "parts": [{"text": system_instruction}]
                }

            try:
                response = await client.post(
                    f"{self.base_url}/models/{self.model}:generateContent?key={self.api_key}",
                    json=payload
                )
                response.raise_for_status()
                result = response.json()

                if result.get("candidates") and len(result["candidates"]) > 0:
                    candidate = result["candidates"][0]
                    if candidate.get("content") and candidate["content"].get("parts"):
                        return candidate["content"]["parts"][0].get("text", "")
                return ""

            except httpx.HTTPError as e:
                error_detail = ""
                try:
                    error_detail = e.response.json() if hasattr(e, 'response') else str(e)
                except:
                    error_detail = str(e)
                raise Exception(f"Gemini API failed: {error_detail}")

    def get_provider_name(self) -> str:
        return "Google Gemini"

    def get_model_name(self) -> str:
        return self.model
