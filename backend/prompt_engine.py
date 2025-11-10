import json
from typing import List, Dict, Optional
from backend.llm_service import LLMService
from backend.models.schemas import SDPrompt, PromptEnhancementResponse
from backend.config import get_settings


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

            # Get fresh settings for defaults
            settings = get_settings()

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

            # Get fresh settings for defaults
            settings = get_settings()

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

    # Negative Prompt Intelligence System
    NEGATIVE_PROMPT_TEMPLATES = {
        "portrait": [
            "deformed face", "bad anatomy", "extra limbs", "extra fingers", "missing fingers",
            "poorly drawn hands", "poorly drawn face", "mutation", "deformed", "ugly",
            "bad proportions", "gross proportions", "disfigured", "malformed limbs", "fused fingers"
        ],
        "landscape": [
            "cluttered", "busy", "oversaturated", "unnatural colors", "distorted perspective",
            "unbalanced composition", "blurry", "low detail"
        ],
        "anime": [
            "realistic", "3d", "western cartoon", "photo", "photograph",
            "bad anatomy", "bad hands", "missing fingers", "extra digit", "fewer digits"
        ],
        "photorealistic": [
            "illustration", "cartoon", "anime", "painting", "drawing", "sketch",
            "unrealistic", "CGI", "3D render", "blurry", "low quality"
        ],
        "artistic": [
            "photograph", "photo", "realistic", "amateur", "low quality",
            "blurry", "out of focus"
        ],
        "general": [
            "blurry", "low quality", "distorted", "deformed", "ugly", "bad anatomy",
            "poorly drawn", "extra limbs", "text", "watermark", "signature", "username"
        ]
    }

    def _get_smart_negative_prompt(self, category: str, positive_prompt: str) -> str:
        """
        Generate an intelligent negative prompt based on category and positive prompt content

        Args:
            category: Detected image category (portrait, landscape, anime, etc.)
            positive_prompt: The positive prompt to analyze

        Returns:
            Optimized negative prompt string
        """
        # Start with category-specific negatives
        base_negatives = self.NEGATIVE_PROMPT_TEMPLATES.get(category, self.NEGATIVE_PROMPT_TEMPLATES["general"])

        # Always include general quality negatives
        negatives = list(self.NEGATIVE_PROMPT_TEMPLATES["general"])

        # Add category-specific negatives
        negatives.extend(base_negatives)

        # Smart additions based on positive prompt analysis
        positive_lower = positive_prompt.lower()

        # If prompt mentions people/characters, add hand/anatomy negatives
        if any(word in positive_lower for word in ["person", "people", "character", "man", "woman", "child", "portrait"]):
            negatives.extend([
                "bad hands", "poorly drawn hands", "malformed hands",
                "extra fingers", "fused fingers", "missing fingers"
            ])

        # If prompt mentions eyes/face, add face-specific negatives
        if any(word in positive_lower for word in ["face", "eyes", "portrait", "headshot", "closeup"]):
            negatives.extend([
                "asymmetric eyes", "crooked eyes", "dead eyes",
                "poorly drawn eyes", "uneven eyes"
            ])

        # If prompt is anime/manga style
        if any(word in positive_lower for word in ["anime", "manga", "cel shaded", "japanese art"]):
            negatives.extend(self.NEGATIVE_PROMPT_TEMPLATES["anime"])

        # If prompt is photorealistic
        if any(word in positive_lower for word in ["photorealistic", "photo", "photograph", "camera", "lens"]):
            negatives.extend(self.NEGATIVE_PROMPT_TEMPLATES["photorealistic"])

        # Remove duplicates while preserving order
        seen = set()
        unique_negatives = []
        for neg in negatives:
            if neg.lower() not in seen:
                seen.add(neg.lower())
                unique_negatives.append(neg)

        return ", ".join(unique_negatives)

    async def create_generation_plan(
        self,
        user_input: str,
        installed_models: List[Dict],
        conversation_history: Optional[List[dict]] = None
    ):
        """
        Create a comprehensive generation plan with model recommendation, quality analysis, and reasoning

        This is the PLAN phase - analyzes the request and provides detailed recommendations
        without executing the generation

        Args:
            user_input: User's natural language description
            installed_models: List of installed SD models with metadata
            conversation_history: Previous conversation context

        Returns:
            GenerationPlan with all recommendations and analysis
        """
        from backend.models.schemas import (
            GenerationPlan, ModelRecommendation, QualityAnalysis,
            ParameterReasoning
        )
        from backend.model_manager import ModelManager

        model_manager = ModelManager()

        print("=" * 80, flush=True)
        print("[PLAN GENERATION CALLED]", flush=True)
        print(f"User input: '{user_input}'", flush=True)
        print("=" * 80, flush=True)

        # Build comprehensive planning prompt
        system_prompt = """You are an expert at analyzing image generation requests and providing detailed, actionable plans.

Analyze the user's request and provide:
1. Image category (portrait, landscape, anime, photorealistic, artistic, concept art, etc.)
2. Quality analysis (specificity score 0-1, missing elements, warnings, strengths)
3. Enhanced prompt with optimal parameters
4. Reasoning for each parameter choice
5. Helpful tips for best results

Consider:
- Aspect ratio from content (portrait of person → vertical, landscape scene → horizontal)
- Complexity for steps (simple → 25-30, detailed → 35-45)
- Style-aware sampling recommendations
- Potential conflicts or issues

Respond ONLY with valid JSON in this format:
{
  "category": "portrait",
  "specificity_score": 0.8,
  "missing_elements": ["lighting direction", "mood"],
  "warnings": [],
  "strengths": ["clear subject", "good style description"],
  "positive_prompt": "enhanced detailed prompt here",
  "steps": 35,
  "cfg_scale": 7.5,
  "width": 512,
  "height": 768,
  "sampler_name": "DPM++ 2M Karras",
  "aspect_ratio": "portrait",
  "steps_reason": "35 steps for moderate detail with good quality",
  "cfg_reason": "7.5 for balanced creativity and prompt adherence",
  "resolution_reason": "512x768 portrait aspect for person-focused composition",
  "sampler_reason": "DPM++ 2M Karras for balanced, high-quality results",
  "explanation": "Overall plan explanation",
  "tips": ["tip1", "tip2"]
}"""

        user_prompt = f"""User wants to generate: "{user_input}"

Analyze this request and create a detailed generation plan."""

        try:
            # Get LLM analysis
            response = await self.llm.generate(
                prompt=user_prompt,
                system_prompt=system_prompt,
                temperature=0.7,
                use_planning_llm=True  # Use best LLM for planning
            )

            # Parse JSON
            response = response.strip()
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0].strip()
            elif "```" in response:
                response = response.split("```")[1].split("```")[0].strip()

            data = json.loads(response)

            # Get model recommendation
            model_rec_prompt = model_manager.get_model_recommendation_prompt(
                user_prompt=user_input,
                installed_models=installed_models
            )

            model_rec_response = await self.llm.generate(
                prompt=model_rec_prompt,
                system_prompt="",
                temperature=0.5,
                use_planning_llm=True
            )

            # Parse model recommendation
            model_rec_response = model_rec_response.strip()
            if "```json" in model_rec_response:
                model_rec_response = model_rec_response.split("```json")[1].split("```")[0].strip()
            elif "```" in model_rec_response:
                model_rec_response = model_rec_response.split("```")[1].split("```")[0].strip()

            model_rec_data = json.loads(model_rec_response)

            # Find model details from RECOMMENDED_MODELS
            model_details = None
            for model in model_manager.RECOMMENDED_MODELS:
                if model["name"].lower() == model_rec_data["recommended_model"].lower():
                    model_details = model
                    break

            # Check if recommended model is installed with improved matching
            recommended_name = model_rec_data["recommended_model"].lower()

            # Debug logging with flush to ensure immediate output
            print(f"\n[MODEL DETECTION DEBUG]", flush=True)
            print(f"  Recommended model: '{model_rec_data['recommended_model']}'", flush=True)
            print(f"  Installed models count: {len(installed_models)}", flush=True)
            print(f"  Installed model names:", flush=True)
            for m in installed_models:
                print(f"    - {m.get('name', m.get('filename', 'Unknown'))}", flush=True)

            # More flexible matching logic
            def model_matches(installed_name: str, recommended_name: str) -> bool:
                """Check if installed model matches recommended model name"""
                import re

                installed_lower = installed_name.lower()
                recommended_lower = recommended_name.lower()

                # Direct substring match (existing logic)
                if recommended_lower in installed_lower or installed_lower in recommended_lower:
                    return True

                # Extract base names by removing common patterns
                # Patterns: version numbers, file extensions, special characters
                def normalize_name(name: str) -> str:
                    # Remove file extension
                    name = re.sub(r'\.(safetensors|ckpt|pt)$', '', name, flags=re.IGNORECASE)
                    # Remove version patterns like _v1, -v2.0, _8, etc.
                    name = re.sub(r'[_-]v?\d+(\.\d+)?', '', name)
                    # Remove common suffixes
                    name = re.sub(r'[_-](pruned|fp16|fp32|ema|inpainting|no-ema)', '', name, flags=re.IGNORECASE)
                    # Remove special characters and extra spaces
                    name = re.sub(r'[_-]', ' ', name)
                    name = re.sub(r'\s+', ' ', name).strip()
                    return name

                installed_normalized = normalize_name(installed_lower)
                recommended_normalized = normalize_name(recommended_lower)

                # Check normalized names
                if installed_normalized == recommended_normalized:
                    return True
                if recommended_normalized in installed_normalized or installed_normalized in recommended_normalized:
                    return True

                return False

            is_installed = False
            matched_model = None
            for m in installed_models:
                model_name = m.get("name", m.get("filename", ""))
                if model_matches(model_name, recommended_name):
                    is_installed = True
                    matched_model = model_name
                    break

            print(f"  Match result: {'FOUND' if is_installed else 'NOT FOUND'}", flush=True)
            if is_installed:
                print(f"  Matched with: '{matched_model}'", flush=True)
            print("", flush=True)

            # Get smart negative prompt
            category = data.get("category", "general")
            smart_negative = self._get_smart_negative_prompt(category, data["positive_prompt"])

            # Get settings for defaults
            settings = get_settings()

            # Build SDPrompt with smart negative prompt
            sd_prompt = SDPrompt(
                positive_prompt=data["positive_prompt"],
                negative_prompt=smart_negative,
                steps=data.get("steps", settings.default_steps),
                cfg_scale=data.get("cfg_scale", settings.default_cfg_scale),
                width=data.get("width", settings.default_width),
                height=data.get("height", settings.default_height),
                sampler_name=data.get("sampler_name", settings.default_sampler),
                seed=-1
            )

            # Build complete plan
            plan = GenerationPlan(
                user_input=user_input,
                model_recommendation=ModelRecommendation(
                    recommended_model_name=model_rec_data["recommended_model"],
                    model_filename=None,  # Will be set when model is selected
                    is_installed=is_installed,
                    reason=model_rec_data.get("reason", "Best match for your prompt"),
                    alternative=model_rec_data.get("alternative"),
                    model_details=model_details
                ),
                enhanced_prompt=sd_prompt,
                quality_analysis=QualityAnalysis(
                    specificity_score=data.get("specificity_score", 0.5),
                    missing_elements=data.get("missing_elements", []),
                    warnings=data.get("warnings", []),
                    strengths=data.get("strengths", []),
                    category=category
                ),
                parameter_reasoning=ParameterReasoning(
                    steps_reason=data.get("steps_reason", "Standard quality"),
                    cfg_reason=data.get("cfg_reason", "Balanced adherence"),
                    resolution_reason=data.get("resolution_reason", "Standard resolution"),
                    sampler_reason=data.get("sampler_reason", "Balanced sampler"),
                    aspect_ratio=data.get("aspect_ratio", "square")
                ),
                explanation=data.get("explanation", "Generated based on your prompt"),
                tips=data.get("tips", []),
                estimated_time=data.get("steps", 30) * 0.15  # Rough estimate: ~0.15s per step
            )

            return plan

        except Exception as e:
            print(f"Error creating generation plan: {e}")
            import traceback
            traceback.print_exc()

            # Fallback plan
            settings = get_settings()
            category = "general"
            basic_prompt = f"{user_input}, masterpiece, best quality, highly detailed"
            smart_negative = self._get_smart_negative_prompt(category, basic_prompt)

            return GenerationPlan(
                user_input=user_input,
                model_recommendation=ModelRecommendation(
                    recommended_model_name="DreamShaper 8",
                    is_installed=False,
                    reason="General-purpose model suitable for most prompts",
                    alternative="Any installed model"
                ),
                enhanced_prompt=SDPrompt(
                    positive_prompt=basic_prompt,
                    negative_prompt=smart_negative,
                    steps=settings.default_steps,
                    cfg_scale=settings.default_cfg_scale,
                    width=settings.default_width,
                    height=settings.default_height,
                    sampler_name=settings.default_sampler
                ),
                quality_analysis=QualityAnalysis(
                    specificity_score=0.5,
                    missing_elements=["More specific details would help"],
                    warnings=[],
                    strengths=["Basic prompt structure"],
                    category=category
                ),
                parameter_reasoning=ParameterReasoning(
                    steps_reason="Default quality settings",
                    cfg_reason="Balanced configuration",
                    resolution_reason="Standard resolution",
                    sampler_reason="Reliable default sampler",
                    aspect_ratio="square"
                ),
                explanation="Using fallback plan due to analysis error",
                tips=["Try being more specific with your description"],
                estimated_time=settings.default_steps * 0.15
            )

    async def create_img2img_plan(
        self,
        user_input: str,
        init_image_base64: str,
        installed_models: List[Dict],
        conversation_history: Optional[List[dict]] = None
    ):
        """
        Create a comprehensive img2img generation plan with model recommendation and denoising analysis

        This is the PLAN phase for img2img - analyzes the request and source image to provide
        detailed recommendations for transformation

        Args:
            user_input: User's description of desired transformation
            init_image_base64: Base64 encoded source image
            installed_models: List of installed SD models with metadata
            conversation_history: Previous conversation context

        Returns:
            Img2ImgGenerationPlan with all recommendations and analysis
        """
        from backend.models.schemas import (
            Img2ImgGenerationPlan, ModelRecommendation, QualityAnalysis,
            ParameterReasoning
        )
        from backend.model_manager import ModelManager

        model_manager = ModelManager()

        print("=" * 80, flush=True)
        print("[IMG2IMG PLAN GENERATION CALLED]", flush=True)
        print(f"User input: '{user_input}'", flush=True)
        print("=" * 80, flush=True)

        # Build comprehensive planning prompt for img2img
        system_prompt = """You are an expert at analyzing img2img requests and providing detailed transformation plans.

For img2img generation, analyze:
1. What transformation is requested (style change, detail addition, modification, etc.)
2. How much the source image should be changed (affects denoising strength)
3. Quality analysis and optimal parameters
4. Reasoning for denoising strength selection

Denoising Strength Guidelines:
- 0.3-0.4: Minor tweaks (color correction, small details, subtle style)
- 0.5-0.6: Moderate changes (style transfer, lighting changes)
- 0.7-0.8: Significant transformation (major style change, composition alterations)
- 0.9+: Almost complete redraw (only basic structure preserved)

Consider:
- Keep aspect ratio from source image in most cases
- Lower denoising preserves more of original image
- Higher steps help with complex transformations
- Style-specific sampling recommendations

Respond ONLY with valid JSON in this format:
{
  "category": "style_transfer",
  "specificity_score": 0.8,
  "missing_elements": [],
  "warnings": ["High denoising will significantly alter original"],
  "strengths": ["clear transformation goal"],
  "positive_prompt": "enhanced detailed prompt here",
  "steps": 35,
  "cfg_scale": 7.5,
  "width": 512,
  "height": 512,
  "sampler_name": "DPM++ 2M Karras",
  "aspect_ratio": "square",
  "denoising_strength": 0.7,
  "denoising_reason": "0.7 allows significant style change while preserving composition",
  "steps_reason": "35 steps for quality transformation",
  "cfg_reason": "7.5 for balanced creativity and prompt adherence",
  "resolution_reason": "Match source image dimensions",
  "sampler_reason": "DPM++ 2M Karras for high-quality img2img",
  "explanation": "Overall transformation plan explanation",
  "tips": ["tip1", "tip2"]
}"""

        user_prompt = f"""User wants to transform an image: "{user_input}"

Analyze this img2img request and create a detailed transformation plan."""

        try:
            # Get LLM analysis
            response = await self.llm.generate(
                prompt=user_prompt,
                system_prompt=system_prompt,
                temperature=0.7,
                use_planning_llm=True  # Use best LLM for planning
            )

            # Parse JSON
            response = response.strip()
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0].strip()
            elif "```" in response:
                response = response.split("```")[1].split("```")[0].strip()

            data = json.loads(response)

            # Get model recommendation (same logic as txt2img)
            model_rec_prompt = model_manager.get_model_recommendation_prompt(
                user_prompt=user_input,
                installed_models=installed_models
            )

            model_rec_response = await self.llm.generate(
                prompt=model_rec_prompt,
                system_prompt="",
                temperature=0.5,
                use_planning_llm=True
            )

            # Parse model recommendation
            model_rec_response = model_rec_response.strip()
            if "```json" in model_rec_response:
                model_rec_response = model_rec_response.split("```json")[1].split("```")[0].strip()
            elif "```" in model_rec_response:
                model_rec_response = model_rec_response.split("```")[1].split("```")[0].strip()

            model_rec_data = json.loads(model_rec_response)

            # Find model details from RECOMMENDED_MODELS
            model_details = None
            for model in model_manager.RECOMMENDED_MODELS:
                if model["name"].lower() == model_rec_data["recommended_model"].lower():
                    model_details = model
                    break

            # Check if recommended model is installed (reuse existing matching logic)
            recommended_name = model_rec_data["recommended_model"].lower()

            def model_matches(installed_name: str, recommended_name: str) -> bool:
                """Check if installed model matches recommended model name"""
                import re

                installed_lower = installed_name.lower()
                recommended_lower = recommended_name.lower()

                if recommended_lower in installed_lower or installed_lower in recommended_lower:
                    return True

                def normalize_name(name: str) -> str:
                    name = re.sub(r'\.(safetensors|ckpt|pt)$', '', name, flags=re.IGNORECASE)
                    name = re.sub(r'[_-]v?\d+(\.\d+)?', '', name)
                    name = re.sub(r'[_-](pruned|fp16|fp32|ema|inpainting|no-ema)', '', name, flags=re.IGNORECASE)
                    name = re.sub(r'[_-]', ' ', name)
                    name = re.sub(r'\s+', ' ', name).strip()
                    return name

                installed_normalized = normalize_name(installed_lower)
                recommended_normalized = normalize_name(recommended_lower)

                if installed_normalized == recommended_normalized:
                    return True
                if recommended_normalized in installed_normalized or installed_normalized in recommended_normalized:
                    return True

                return False

            is_installed = False
            matched_model = None
            for m in installed_models:
                model_name = m.get("name", m.get("filename", ""))
                if model_matches(model_name, recommended_name):
                    is_installed = True
                    matched_model = model_name
                    break

            print(f"  Img2Img Model match: {'FOUND' if is_installed else 'NOT FOUND'}", flush=True)
            if is_installed:
                print(f"  Matched with: '{matched_model}'", flush=True)

            # Get smart negative prompt
            category = data.get("category", "general")
            smart_negative = self._get_smart_negative_prompt(category, data["positive_prompt"])

            # Get settings for defaults
            settings = get_settings()

            # Build SDPrompt
            sd_prompt = SDPrompt(
                positive_prompt=data["positive_prompt"],
                negative_prompt=smart_negative,
                steps=data.get("steps", settings.default_steps),
                cfg_scale=data.get("cfg_scale", settings.default_cfg_scale),
                width=data.get("width", settings.default_width),
                height=data.get("height", settings.default_height),
                sampler_name=data.get("sampler_name", settings.default_sampler),
                seed=-1
            )

            # Build complete img2img plan
            plan = Img2ImgGenerationPlan(
                user_input=user_input,
                model_recommendation=ModelRecommendation(
                    recommended_model_name=model_rec_data["recommended_model"],
                    model_filename=None,
                    is_installed=is_installed,
                    reason=model_rec_data.get("reason", "Best match for your transformation"),
                    alternative=model_rec_data.get("alternative"),
                    model_details=model_details
                ),
                enhanced_prompt=sd_prompt,
                quality_analysis=QualityAnalysis(
                    specificity_score=data.get("specificity_score", 0.5),
                    missing_elements=data.get("missing_elements", []),
                    warnings=data.get("warnings", []),
                    strengths=data.get("strengths", []),
                    category=category
                ),
                parameter_reasoning=ParameterReasoning(
                    steps_reason=data.get("steps_reason", "Standard quality"),
                    cfg_reason=data.get("cfg_reason", "Balanced adherence"),
                    resolution_reason=data.get("resolution_reason", "Standard resolution"),
                    sampler_reason=data.get("sampler_reason", "Balanced sampler"),
                    aspect_ratio=data.get("aspect_ratio", "square")
                ),
                denoising_strength=data.get("denoising_strength", 0.7),
                denoising_reason=data.get("denoising_reason", "Balanced transformation strength"),
                explanation=data.get("explanation", "Generated transformation plan based on your prompt"),
                tips=data.get("tips", []),
                estimated_time=data.get("steps", 30) * 0.15
            )

            return plan

        except Exception as e:
            print(f"Error creating img2img plan: {e}")
            import traceback
            traceback.print_exc()

            # Fallback plan
            settings = get_settings()
            category = "general"
            basic_prompt = f"{user_input}, masterpiece, best quality, highly detailed"
            smart_negative = self._get_smart_negative_prompt(category, basic_prompt)

            return Img2ImgGenerationPlan(
                user_input=user_input,
                model_recommendation=ModelRecommendation(
                    recommended_model_name="DreamShaper 8",
                    is_installed=False,
                    reason="General-purpose model suitable for img2img",
                    alternative="Any installed model"
                ),
                enhanced_prompt=SDPrompt(
                    positive_prompt=basic_prompt,
                    negative_prompt=smart_negative,
                    steps=settings.default_steps,
                    cfg_scale=settings.default_cfg_scale,
                    width=settings.default_width,
                    height=settings.default_height,
                    sampler_name=settings.default_sampler
                ),
                quality_analysis=QualityAnalysis(
                    specificity_score=0.5,
                    missing_elements=["More specific transformation details would help"],
                    warnings=[],
                    strengths=["Basic transformation goal"],
                    category=category
                ),
                parameter_reasoning=ParameterReasoning(
                    steps_reason="Default quality settings",
                    cfg_reason="Balanced configuration",
                    resolution_reason="Standard resolution",
                    sampler_reason="Reliable default sampler",
                    aspect_ratio="square"
                ),
                denoising_strength=0.7,
                denoising_reason="0.7 provides balanced transformation",
                explanation="Using fallback plan due to analysis error",
                tips=["Try being more specific about the transformation you want"],
                estimated_time=settings.default_steps * 0.15
            )
