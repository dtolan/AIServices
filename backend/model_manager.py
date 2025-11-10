import httpx
import asyncio
from pathlib import Path
from typing import List, Dict, Optional
import hashlib
import shutil
from backend.config import get_settings


class ModelManager:
    """
    Manages Stable Diffusion model downloads and installation
    """

    # Curated list of recommended models
    RECOMMENDED_MODELS = [
        {
            "id": "4384",
            "name": "DreamShaper 8",
            "type": "General Purpose",
            "description": "Excellent all-around model with great balance of quality and versatility",
            "size": "2.0 GB",
            "style": "Realistic",
            "civitai_url": "https://civitai.com/models/4384/dreamshaper",
            "recommended_for": ["Beginners", "General Use"]
        },
        {
            "id": "4201",
            "name": "Realistic Vision V5.1",
            "type": "Photorealistic",
            "description": "Best for photorealistic images with amazing detail",
            "size": "2.1 GB",
            "style": "Photorealistic",
            "civitai_url": "https://civitai.com/models/4201/realistic-vision-v51",
            "recommended_for": ["Photography", "Portraits", "Realism"]
        },
        {
            "id": "9942",
            "name": "AbyssOrangeMix3",
            "type": "Anime",
            "description": "High-quality anime and illustration style",
            "size": "2.0 GB",
            "style": "Anime",
            "civitai_url": "https://civitai.com/models/9942/abyssorangemix3-aom3",
            "recommended_for": ["Anime", "Manga", "Illustrations"]
        },
        {
            "id": "43331",
            "name": "Deliberate V2",
            "type": "Artistic",
            "description": "Great for artistic and creative generations",
            "size": "2.1 GB",
            "style": "Artistic",
            "civitai_url": "https://civitai.com/models/4823/deliberate",
            "recommended_for": ["Art", "Creative", "Stylized"]
        },
        {
            "id": "101055",
            "name": "SDXL 1.0",
            "type": "SDXL Base",
            "description": "Official SDXL base model - requires more VRAM but produces 1024x1024 images",
            "size": "6.9 GB",
            "style": "General",
            "civitai_url": "https://civitai.com/models/101055/sd-xl",
            "recommended_for": ["High Resolution", "Quality", "Advanced Users"],
            "notes": "Requires 8GB+ VRAM"
        },
        {
            "id": "112902",
            "name": "DreamShaper XL",
            "type": "SDXL",
            "description": "SDXL version of DreamShaper with amazing quality",
            "size": "6.5 GB",
            "style": "Realistic",
            "civitai_url": "https://civitai.com/models/112902/dreamshaper-xl",
            "recommended_for": ["High Quality", "Versatile", "SDXL"],
            "notes": "Requires 8GB+ VRAM"
        }
    ]

    def __init__(self):
        settings = get_settings()
        self.models_dir = self._find_sd_models_dir()
        self.api_base = "https://civitai.com/api/v1"
        self.civitai_api_key = settings.civitai_api_key

    def _find_sd_models_dir(self) -> Path:
        """Find the Stable Diffusion models directory"""
        # Common locations for SD WebUI
        possible_paths = [
            Path("stable-diffusion-webui/models/Stable-diffusion"),
            Path("../stable-diffusion-webui/models/Stable-diffusion"),
            Path.home() / "stable-diffusion-webui/models/Stable-diffusion",
            Path("C:/stable-diffusion-webui/models/Stable-diffusion"),
        ]

        for path in possible_paths:
            if path.exists():
                return path

        # Default fallback
        return Path("models/Stable-diffusion")

    def get_recommended_models(self) -> List[Dict]:
        """Get curated list of recommended models"""
        return self.RECOMMENDED_MODELS

    def get_installed_models(self) -> List[Dict]:
        """Get list of installed models"""
        if not self.models_dir.exists():
            return []

        models = []
        for file in self.models_dir.glob("*.safetensors"):
            size_mb = file.stat().st_size / (1024 * 1024)
            models.append({
                "name": file.stem,
                "filename": file.name,
                "size": f"{size_mb:.1f} MB",
                "size_bytes": file.stat().st_size,
                "path": str(file)
            })

        # Sort by name
        models.sort(key=lambda x: x["name"].lower())
        return models

    async def download_model(
        self,
        download_url: str,
        filename: str,
        progress_callback: Optional[callable] = None
    ) -> Dict:
        """
        Download a model file with progress tracking

        Args:
            download_url: Direct download URL
            filename: Target filename (e.g., "dreamshaper_8.safetensors")
            progress_callback: Optional callback(bytes_downloaded, total_bytes)

        Returns:
            Dict with success status and file path
        """
        if not self.models_dir.exists():
            self.models_dir.mkdir(parents=True, exist_ok=True)

        target_path = self.models_dir / filename

        try:
            # First request to CivitAI to get redirect URL and cookies
            async with httpx.AsyncClient(timeout=30, follow_redirects=False) as client:
                initial_response = await client.get(download_url)

                if initial_response.status_code in (307, 302, 303):
                    # Get the redirect location
                    redirect_url = initial_response.headers.get("location")
                    print(f"[DOWNLOAD] Following redirect to: {redirect_url[:80]}...")

                    # Now download from the redirect URL with a longer timeout
                    async with httpx.AsyncClient(timeout=None, follow_redirects=True) as download_client:
                        async with download_client.stream("GET", redirect_url) as response:
                            response.raise_for_status()

                            total_size = int(response.headers.get("content-length", 0))
                            downloaded = 0

                            with open(target_path, "wb") as f:
                                async for chunk in response.aiter_bytes(chunk_size=8192):
                                    f.write(chunk)
                                    downloaded += len(chunk)

                                    if progress_callback:
                                        progress_callback(downloaded, total_size)
                else:
                    # No redirect, download directly
                    async with httpx.AsyncClient(timeout=None, follow_redirects=True) as download_client:
                        async with download_client.stream("GET", download_url) as response:
                            response.raise_for_status()

                            total_size = int(response.headers.get("content-length", 0))
                            downloaded = 0

                            with open(target_path, "wb") as f:
                                async for chunk in response.aiter_bytes(chunk_size=8192):
                                    f.write(chunk)
                                    downloaded += len(chunk)

                                    if progress_callback:
                                        progress_callback(downloaded, total_size)

            return {
                "success": True,
                "path": str(target_path),
                "size": target_path.stat().st_size
            }

        except Exception as e:
            # Clean up partial download
            if target_path.exists():
                target_path.unlink()

            return {
                "success": False,
                "error": str(e)
            }

    async def get_model_info_from_civitai(self, model_id: str) -> Optional[Dict]:
        """Fetch model information from CivitAI API"""
        try:
            headers = {}
            if self.civitai_api_key:
                headers["Authorization"] = f"Bearer {self.civitai_api_key}"
                print(f"[API] Using CivitAI API key for model info request")

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.api_base}/models/{model_id}",
                    headers=headers
                )
                response.raise_for_status()
                return response.json()
        except Exception as e:
            print(f"Error fetching model info: {e}")
            return None

    async def get_download_url_for_model(self, model_id: str) -> Optional[Dict]:
        """
        Get the download URL for a model from CivitAI

        Returns:
            Dict with download_url and filename, or None if failed
        """
        try:
            model_info = await self.get_model_info_from_civitai(model_id)
            if not model_info:
                return None

            # Get the latest version
            if not model_info.get("modelVersions"):
                return None

            latest_version = model_info["modelVersions"][0]

            # Get the primary file (usually the safetensors file)
            if not latest_version.get("files"):
                return None

            # Prefer safetensors format
            primary_file = None
            for file in latest_version["files"]:
                if file.get("primary") or file["name"].endswith(".safetensors"):
                    primary_file = file
                    break

            if not primary_file:
                primary_file = latest_version["files"][0]

            return {
                "download_url": primary_file.get("downloadUrl"),
                "filename": primary_file.get("name"),
                "size": primary_file.get("sizeKB", 0) * 1024,
                "model_name": model_info.get("name"),
                "version_name": latest_version.get("name")
            }

        except Exception as e:
            print(f"Error getting download URL: {e}")
            return None

    def delete_model(self, filename: str) -> bool:
        """Delete an installed model"""
        try:
            model_path = self.models_dir / filename
            if model_path.exists():
                model_path.unlink()
                return True
            return False
        except Exception as e:
            print(f"Error deleting model: {e}")
            return False

    def get_models_directory(self) -> str:
        """Get the current models directory path"""
        return str(self.models_dir)

    def set_models_directory(self, path: str) -> bool:
        """Set a custom models directory"""
        try:
            new_path = Path(path)
            if new_path.exists() or new_path.parent.exists():
                self.models_dir = new_path
                return True
            return False
        except Exception:
            return False

    def get_model_recommendation_prompt(self, user_prompt: str, installed_models: List[Dict]) -> str:
        """
        Generate a system prompt for LLM to recommend the best model

        Args:
            user_prompt: The user's image generation prompt
            installed_models: List of installed models with their names

        Returns:
            System prompt for model recommendation
        """
        installed_names = [m["name"] for m in installed_models] if installed_models else []

        system_prompt = f"""You are an expert at Stable Diffusion models. Analyze the user's prompt and recommend the best SD model.

Available curated models:
{self._format_models_for_llm()}

Currently installed models:
{installed_names if installed_names else "None - suggest one to download"}

Based on the user's prompt, recommend:
1. The BEST model for their specific use case (from curated list)
2. Whether they should use an installed model or download a new one
3. Brief explanation (1-2 sentences) why this model is ideal

User's prompt: "{user_prompt}"

Respond in JSON format:
{{
  "recommended_model": "model name",
  "is_installed": true/false,
  "reason": "explanation",
  "alternative": "backup suggestion if primary not installed"
}}"""

        return system_prompt

    def _format_models_for_llm(self) -> str:
        """Format model list for LLM prompt"""
        lines = []
        for model in self.RECOMMENDED_MODELS:
            lines.append(
                f"- {model['name']} ({model['type']}): {model['description']}"
                f" - Best for: {', '.join(model['recommended_for'])}"
            )
        return "\n".join(lines)

    def _find_model_variants(self, filename: str) -> List[Path]:
        """
        Find installed models that are likely different versions of the same base model.

        For example:
        - dreamshaper_8.safetensors and dreamshaper_7.safetensors
        - realisticVision_v5.safetensors and realisticVision_v4.safetensors

        Returns list of Path objects for variant models
        """
        import re

        # Extract base name by removing version patterns
        # Common patterns: _v1, _v2, _8, -v1.0, etc.
        base_name = re.sub(r'[_-]v?\d+(\.\d+)?', '', filename.lower())
        base_name = re.sub(r'\.safetensors$', '', base_name)

        variants = []
        if self.models_dir.exists():
            for file in self.models_dir.glob("*.safetensors"):
                if file.name == filename:
                    continue  # Skip exact match

                # Check if this file matches the base name pattern
                file_base = re.sub(r'[_-]v?\d+(\.\d+)?', '', file.name.lower())
                file_base = re.sub(r'\.safetensors$', '', file_base)

                if file_base == base_name:
                    variants.append(file)

        return variants

    def import_model(self, source_path: str, move: bool = True) -> Dict:
        """
        Import a model from an external location (e.g., Downloads folder) to the models directory

        Args:
            source_path: Full path to the model file to import
            move: If True, move the file (remove from source). If False, copy it (default: True)

        Returns:
            Dict with success status, target path, and optional error message
        """
        try:
            source = Path(source_path)

            # Validate source file exists
            if not source.exists():
                return {
                    "success": False,
                    "error": f"File not found: {source_path}"
                }

            # Validate it's a safetensors file
            if source.suffix.lower() != '.safetensors':
                return {
                    "success": False,
                    "error": "Only .safetensors files are supported"
                }

            # Ensure models directory exists
            if not self.models_dir.exists():
                self.models_dir.mkdir(parents=True, exist_ok=True)

            # Target path in models directory
            target = self.models_dir / source.name

            # Check if this exact file already exists
            if target.exists():
                # If move=True, clean up the source file from Downloads even though import failed
                if move and source.exists():
                    try:
                        source.unlink()
                        print(f"[CLEANUP] Removed duplicate model from Downloads: {source.name}")
                    except Exception as e:
                        print(f"[WARNING] Failed to clean up duplicate from Downloads: {e}")

                return {
                    "success": False,
                    "error": f"Model '{source.name}' already exists in models directory",
                    "cleaned_up": move  # Indicate whether we cleaned up the Downloads file
                }

            # Check for older/newer versions of the same model
            variants = self._find_model_variants(source.name)
            upgraded_from = None
            if variants:
                # Found variant(s) - likely an upgrade/downgrade scenario
                # Delete the old versions before importing the new one
                deleted_variants = []
                for variant in variants:
                    try:
                        variant.unlink()
                        deleted_variants.append(variant.name)
                        print(f"[UPGRADE] Deleted old version: {variant.name}")
                    except Exception as e:
                        print(f"[WARNING] Failed to delete old version {variant.name}: {e}")

                upgraded_from = deleted_variants

            # Move or copy the file
            if move:
                shutil.move(source, target)
            else:
                shutil.copy2(source, target)

            # Get file size for response
            size_mb = target.stat().st_size / (1024 * 1024)

            result = {
                "success": True,
                "path": str(target),
                "filename": target.name,
                "size": f"{size_mb:.1f} MB",
                "size_bytes": target.stat().st_size,
                "moved": move
            }

            # Add upgrade information if we replaced old versions
            if upgraded_from:
                result["upgraded"] = True
                result["replaced_versions"] = upgraded_from

            return result

        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to import model: {str(e)}"
            }
