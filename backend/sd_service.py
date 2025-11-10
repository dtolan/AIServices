import httpx
import base64
import time
from typing import Dict, List, Optional
from backend.config import get_settings
from backend.models.schemas import SDPrompt

settings = get_settings()


class StableDiffusionService:
    """Service for interacting with Automatic1111 Stable Diffusion API"""

    def __init__(self):
        self.base_url = settings.sd_api_url
        self.timeout = settings.sd_api_timeout

    async def health_check(self) -> bool:
        """Check if SD API is available"""
        async with httpx.AsyncClient(timeout=5.0) as client:
            try:
                response = await client.get(f"{self.base_url}/sdapi/v1/sd-models")
                return response.status_code == 200
            except Exception as e:
                raise Exception(f"Stable Diffusion API not available: {e}")

    async def generate_image(self, prompt: SDPrompt) -> Dict:
        """
        Generate an image using Stable Diffusion

        Args:
            prompt: SDPrompt object with generation parameters

        Returns:
            Dict with image_base64, generation_time, and seed
        """
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            payload = {
                "prompt": prompt.positive_prompt,
                "negative_prompt": prompt.negative_prompt,
                "steps": prompt.steps,
                "cfg_scale": prompt.cfg_scale,
                "width": prompt.width,
                "height": prompt.height,
                "sampler_name": prompt.sampler_name,
                "seed": prompt.seed,
                "batch_size": 1,
                "n_iter": 1,
            }

            try:
                start_time = time.time()
                response = await client.post(
                    f"{self.base_url}/sdapi/v1/txt2img",
                    json=payload
                )
                response.raise_for_status()
                generation_time = time.time() - start_time

                result = response.json()

                return {
                    "image_base64": result["images"][0],
                    "generation_time": generation_time,
                    "seed": result["info"].get("seed", prompt.seed) if isinstance(result.get("info"), dict) else prompt.seed,
                    "parameters": result.get("parameters", {})
                }
            except httpx.HTTPError as e:
                raise Exception(f"Image generation failed: {e}")

    async def img2img(
        self,
        prompt: SDPrompt,
        init_image_base64: str,
        denoising_strength: float = 0.7
    ) -> Dict:
        """
        Generate image from image (img2img)

        Args:
            prompt: SDPrompt object
            init_image_base64: Base64 encoded initial image
            denoising_strength: How much to transform the image (0.0-1.0)

        Returns:
            Dict with image_base64, generation_time, and seed
        """
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            payload = {
                "init_images": [init_image_base64],
                "prompt": prompt.positive_prompt,
                "negative_prompt": prompt.negative_prompt,
                "steps": prompt.steps,
                "cfg_scale": prompt.cfg_scale,
                "width": prompt.width,
                "height": prompt.height,
                "sampler_name": prompt.sampler_name,
                "seed": prompt.seed,
                "denoising_strength": denoising_strength,
                "batch_size": 1,
                "n_iter": 1,
            }

            try:
                start_time = time.time()
                response = await client.post(
                    f"{self.base_url}/sdapi/v1/img2img",
                    json=payload
                )
                response.raise_for_status()
                generation_time = time.time() - start_time

                result = response.json()

                return {
                    "image_base64": result["images"][0],
                    "generation_time": generation_time,
                    "seed": result.get("seed", prompt.seed),
                    "parameters": result.get("parameters", {})
                }
            except httpx.HTTPError as e:
                raise Exception(f"img2img generation failed: {e}")

    async def get_samplers(self) -> List[str]:
        """Get list of available samplers"""
        async with httpx.AsyncClient(timeout=5.0) as client:
            try:
                response = await client.get(f"{self.base_url}/sdapi/v1/samplers")
                response.raise_for_status()
                samplers = response.json()
                return [sampler["name"] for sampler in samplers]
            except httpx.HTTPError as e:
                raise Exception(f"Failed to get samplers: {e}")

    async def get_models(self) -> List[Dict]:
        """Get list of available SD models"""
        async with httpx.AsyncClient(timeout=5.0) as client:
            try:
                response = await client.get(f"{self.base_url}/sdapi/v1/sd-models")
                response.raise_for_status()
                models = response.json()
                return [
                    {
                        "title": model["title"],
                        "model_name": model["model_name"],
                        "filename": model.get("filename", "")
                    }
                    for model in models
                ]
            except httpx.HTTPError as e:
                raise Exception(f"Failed to get models: {e}")

    async def get_current_model(self) -> Optional[str]:
        """Get currently loaded model"""
        async with httpx.AsyncClient(timeout=5.0) as client:
            try:
                response = await client.get(f"{self.base_url}/sdapi/v1/options")
                response.raise_for_status()
                options = response.json()
                return options.get("sd_model_checkpoint")
            except httpx.HTTPError as e:
                raise Exception(f"Failed to get current model: {e}")

    async def set_model(self, model_name: str) -> bool:
        """Set the active SD model"""
        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                response = await client.post(
                    f"{self.base_url}/sdapi/v1/options",
                    json={"sd_model_checkpoint": model_name}
                )
                response.raise_for_status()
                return True
            except httpx.HTTPError as e:
                raise Exception(f"Failed to set model: {e}")
