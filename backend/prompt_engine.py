import json
from typing import List, Dict, Optional
from backend.llm_service import LLMService
from backend.models.schemas import SDPrompt, PromptEnhancementResponse
from backend.config import get_settings

settings = get_settings()


class PromptEngine:
    """
    Core engine for enhancing prompts using LLM knowledge
    """

    def __init__(self, llm_service: LLMService):
        self.llm = llm_service

    def _build_system_prompt(self) -> str:
        """Build the system prompt with SD knowledge"""
        return """You are an expert Stable Diffusion prompt engineer. Your role is to help users create high-quality image prompts.

Key Stable Diffusion prompt principles:
1. Be specific and descriptive - use concrete details
2. Structure: [subject], [style], [composition], [lighting], [quality tags]
3. Quality tags boost image quality: "masterpiece, best quality, highly detailed, 8k, professional"
4. Style modifiers: "digital art, oil painting, photograph, 3D render, anime, watercolor"
5. Lighting matters: "dramatic lighting, soft lighting, golden hour, studio lighting"
6. Camera/composition: "close-up, wide angle, portrait, aerial view, cinematic"
7. Artists/styles: "in the style of [artist name]", "trending on artstation"
8. Negative prompts remove unwanted elements: "blurry, low quality, distorted, ugly, bad anatomy"

Common negative prompt template: "blurry, low quality, distorted, deformed, ugly, bad anatomy, extra limbs, poorly drawn, text, watermark"

Parameter guidelines:
- Steps: 20-30 for most, 40-50 for high detail
- CFG Scale: 7-11 (higher = more prompt adherence, lower = more creative freedom)
- Samplers: "DPM++ 2M Karras" (balanced), "Euler a" (creative), "DPM++ SDE Karras" (detailed)
- Resolution: 512x512 baseline, 768x768 for more detail (requires more VRAM)

When enhancing prompts:
1. Expand vague descriptions into specific, detailed prompts
2. Add appropriate quality tags and style modifiers
3. Suggest optimal parameters based on the desired output
4. Include relevant negative prompts
5. Explain your choices briefly

Respond ONLY with valid JSON in this exact format:
{
  "positive_prompt": "detailed prompt here",
  "negative_prompt": "negative tags here",
  "steps": 30,
  "cfg_scale": 7.5,
  "width": 512,
  "height": 512,
  "sampler_name": "DPM++ 2M Karras",
  "explanation": "brief explanation of choices",
  "suggestions": ["tip 1", "tip 2"]
}"""

    def _build_iteration_system_prompt(self) -> str:
        """System prompt for iterating on existing images"""
        return """You are an expert at iterating and refining Stable Diffusion prompts based on feedback.

When the user provides feedback on an image:
1. Analyze what they want to change/improve
2. Modify the prompt strategically:
   - Add/remove specific elements
   - Adjust style descriptors
   - Change emphasis with () for more, [] for less weight
   - Modify parameters if needed (steps, CFG, sampler)
3. Consider if seed should change (-1 for variation, keep same for subtle changes)
4. Update negative prompt to exclude unwanted elements

Common feedback patterns:
- "More detailed" → increase steps, add detail tags, higher resolution
- "Less realistic" → adjust style, lower CFG
- "Different composition" → change seed, modify composition keywords
- "Better quality" → add quality tags, increase steps
- "Darker/lighter" → adjust lighting keywords

Respond ONLY with valid JSON in the same format as before."""

    async def enhance_prompt(
        self,
        user_input: str,
        conversation_history: Optional[List[dict]] = None
    ) -> PromptEnhancementResponse:
        """
        Enhance a user's natural language input into a structured SD prompt
        Uses PLANNING LLM for initial prompt engineering

        Args:
            user_input: Natural language description from user
            conversation_history: Previous conversation context

        Returns:
            PromptEnhancementResponse with enhanced prompt and explanation
        """
        # Build the prompt for the LLM
        user_prompt = f"""User wants to generate: "{user_input}"

Create an optimized Stable Diffusion prompt with appropriate parameters."""

        if conversation_history:
            user_prompt = "Previous context:\n"
            for msg in conversation_history[-3:]:  # Last 3 messages for context
                user_prompt += f"{msg['role']}: {msg['content']}\n"
            user_prompt += f"\nNow the user says: \"{user_input}\"\n\nCreate an optimized Stable Diffusion prompt."

        # Get LLM response - USE PLANNING LLM (better quality for initial prompt engineering)
        try:
            response = await self.llm.generate(
                prompt=user_prompt,
                system_prompt=self._build_system_prompt(),
                temperature=0.7,
                use_planning_llm=True  # Use planning LLM for initial prompt creation
            )

            # Parse JSON response
            # Try to extract JSON from markdown code blocks if present
            response = response.strip()
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0].strip()
            elif "```" in response:
                response = response.split("```")[1].split("```")[0].strip()

            data = json.loads(response)

            # Build SDPrompt
            sd_prompt = SDPrompt(
                positive_prompt=data["positive_prompt"],
                negative_prompt=data.get("negative_prompt", ""),
                steps=data.get("steps", settings.default_steps),
                cfg_scale=data.get("cfg_scale", settings.default_cfg_scale),
                width=data.get("width", settings.default_width),
                height=data.get("height", settings.default_height),
                sampler_name=data.get("sampler_name", settings.default_sampler),
                seed=data.get("seed", -1)
            )

            return PromptEnhancementResponse(
                enhanced_prompt=sd_prompt,
                explanation=data.get("explanation", "Prompt enhanced with SD best practices"),
                suggestions=data.get("suggestions", [])
            )

        except json.JSONDecodeError as e:
            # Fallback if JSON parsing fails
            print(f"Failed to parse LLM response: {e}")
            print(f"Response was: {response}")

            # Create a basic prompt from user input
            sd_prompt = SDPrompt(
                positive_prompt=f"{user_input}, masterpiece, best quality, highly detailed",
                negative_prompt="blurry, low quality, distorted, ugly, bad anatomy",
                steps=settings.default_steps,
                cfg_scale=settings.default_cfg_scale,
                width=settings.default_width,
                height=settings.default_height,
                sampler_name=settings.default_sampler
            )

            return PromptEnhancementResponse(
                enhanced_prompt=sd_prompt,
                explanation="Using basic prompt enhancement (LLM response parsing failed)",
                suggestions=["Try being more specific with your description"]
            )

    async def iterate_prompt(
        self,
        previous_prompt: SDPrompt,
        user_feedback: str,
        previous_image_base64: Optional[str] = None
    ) -> PromptEnhancementResponse:
        """
        Iterate on a previous prompt based on user feedback
        Uses EXECUTION LLM for faster iterations

        Args:
            previous_prompt: The previous SDPrompt used
            user_feedback: What the user wants to change
            previous_image_base64: Base64 of previous image (for context)

        Returns:
            PromptEnhancementResponse with iterated prompt
        """
        user_prompt = f"""Previous prompt:
Positive: {previous_prompt.positive_prompt}
Negative: {previous_prompt.negative_prompt}
Steps: {previous_prompt.steps}, CFG: {previous_prompt.cfg_scale}
Size: {previous_prompt.width}x{previous_prompt.height}
Sampler: {previous_prompt.sampler_name}
Seed: {previous_prompt.seed}

User feedback: "{user_feedback}"

Modify the prompt to address the user's feedback."""

        try:
            # Use EXECUTION LLM for faster iterations
            response = await self.llm.generate(
                prompt=user_prompt,
                system_prompt=self._build_iteration_system_prompt(),
                temperature=0.7,
                use_planning_llm=False  # Use execution LLM for speed
            )

            # Parse JSON response
            response = response.strip()
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0].strip()
            elif "```" in response:
                response = response.split("```")[1].split("```")[0].strip()

            data = json.loads(response)

            sd_prompt = SDPrompt(
                positive_prompt=data["positive_prompt"],
                negative_prompt=data.get("negative_prompt", previous_prompt.negative_prompt),
                steps=data.get("steps", previous_prompt.steps),
                cfg_scale=data.get("cfg_scale", previous_prompt.cfg_scale),
                width=data.get("width", previous_prompt.width),
                height=data.get("height", previous_prompt.height),
                sampler_name=data.get("sampler_name", previous_prompt.sampler_name),
                seed=data.get("seed", -1)  # -1 for new variation by default
            )

            return PromptEnhancementResponse(
                enhanced_prompt=sd_prompt,
                explanation=data.get("explanation", "Prompt iterated based on feedback"),
                suggestions=data.get("suggestions", [])
            )

        except json.JSONDecodeError as e:
            print(f"Failed to parse iteration response: {e}")

            # Simple fallback - append feedback to prompt
            sd_prompt = SDPrompt(
                positive_prompt=f"{previous_prompt.positive_prompt}, {user_feedback}",
                negative_prompt=previous_prompt.negative_prompt,
                steps=previous_prompt.steps,
                cfg_scale=previous_prompt.cfg_scale,
                width=previous_prompt.width,
                height=previous_prompt.height,
                sampler_name=previous_prompt.sampler_name,
                seed=-1
            )

            return PromptEnhancementResponse(
                enhanced_prompt=sd_prompt,
                explanation="Applied basic iteration (LLM parsing failed)",
                suggestions=[]
            )
