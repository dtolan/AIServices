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

    async def get_loras(self) -> List[Dict]:
        """
        Get list of available LoRAs from Stable Diffusion

        Returns:
            List of LoRA dictionaries with name, alias, and path
        """
        async with httpx.AsyncClient(timeout=5.0) as client:
            try:
                response = await client.get(f"{self.base_url}/sdapi/v1/loras")
                response.raise_for_status()
                loras = response.json()

                # Format LoRA data for easier consumption
                formatted_loras = []
                for lora in loras:
                    formatted_loras.append({
                        "name": lora.get("name", ""),
                        "alias": lora.get("alias", lora.get("name", "")),
                        "path": lora.get("path", ""),
                        "metadata": lora.get("metadata", {})
                    })

                return formatted_loras
            except httpx.HTTPError as e:
                print(f"[LORA] Failed to get LoRAs: {e}")
                return []

    async def get_memory_info(self) -> Dict:
        """
        Get GPU memory information with multiple detection methods

        Detection order:
        1. SD API (/sdapi/v1/memory) - Most accurate, real-time usage
        2. nvidia-smi - NVIDIA GPUs (full info: total/free/used)
        3. rocm-smi - AMD GPUs with ROCm drivers (total only)
        4. clinfo - AMD/NVIDIA/Intel via OpenCL (total only)

        Returns:
            Dict with VRAM info:
            {
                "vram_total_gb": 8.0,
                "vram_free_gb": 6.5,
                "vram_used_gb": 1.5,
                "system_ram_total_gb": 16.0,
                "system_ram_free_gb": 8.0,
                "detection_method": "sd_api" | "nvidia_smi" | "rocm_smi" | "clinfo" | "failed",
                "gpu_name": "RTX 4070" (optional, if detected)
            }
        """
        vram_total = 0.0
        vram_free = 0.0
        vram_used = 0.0
        system_ram_total = 0.0
        system_ram_free = 0.0
        detection_method = "failed"

        # Try SD API first
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(f"{self.base_url}/sdapi/v1/memory")
                response.raise_for_status()
                memory_data = response.json()

                # Parse memory data (values are typically in bytes)
                # SD WebUI returns memory in different formats depending on version
                # Try to normalize to GB

                # Common format: {"ram": {...}, "cuda": {...}}
                if "cuda" in memory_data:
                    cuda = memory_data["cuda"]

                    # Check if cuda has 'system' key (A1111 format)
                    if "system" in cuda:
                        vram_data = cuda["system"]
                        vram_total = vram_data.get("total", 0)
                        vram_free = vram_data.get("free", 0)
                        vram_used = vram_data.get("used", 0)
                    else:
                        # Fallback to direct cuda keys
                        vram_total = cuda.get("total", 0)
                        vram_free = cuda.get("free", 0)
                        vram_used = cuda.get("used", 0)

                    # If values seem to be in bytes (> 100), convert to GB
                    if vram_total > 100:
                        vram_total = vram_total / (1024**3)
                        vram_free = vram_free / (1024**3)
                        vram_used = vram_used / (1024**3)

                if "ram" in memory_data:
                    ram = memory_data["ram"]
                    system_ram_total = ram.get("total", 0)
                    system_ram_free = ram.get("free", 0)

                    if system_ram_total > 100:
                        system_ram_total = system_ram_total / (1024**3)
                        system_ram_free = system_ram_free / (1024**3)

                # If we got valid VRAM data, mark as successful
                if vram_total > 0:
                    detection_method = "sd_api"
                    return {
                        "vram_total_gb": round(vram_total, 2),
                        "vram_free_gb": round(vram_free, 2),
                        "vram_used_gb": round(vram_used, 2),
                        "system_ram_total_gb": round(system_ram_total, 2),
                        "system_ram_free_gb": round(system_ram_free, 2),
                        "detection_method": detection_method,
                        "raw_data": memory_data
                    }
        except Exception as e:
            print(f"[VRAM] SD API detection failed: {e}")

        # Fallback 1: Try nvidia-smi (NVIDIA GPUs)
        print("[VRAM] Attempting nvidia-smi fallback...")
        try:
            import subprocess
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=memory.total,memory.free,memory.used,gpu_name', '--format=csv,noheader,nounits'],
                capture_output=True,
                text=True,
                timeout=5
            )

            if result.returncode == 0:
                output = result.stdout.strip()
                if output:
                    # Parse first GPU (format: "total, free, used, name")
                    lines = output.split('\n')
                    first_gpu = lines[0].split(',')

                    if len(first_gpu) >= 3:
                        vram_total = float(first_gpu[0].strip()) / 1024  # MB to GB
                        vram_free = float(first_gpu[1].strip()) / 1024
                        vram_used = float(first_gpu[2].strip()) / 1024
                        gpu_name = first_gpu[3].strip() if len(first_gpu) > 3 else "Unknown NVIDIA"

                        detection_method = "nvidia_smi"
                        print(f"[VRAM] ✓ Detected via nvidia-smi: {gpu_name} with {vram_total:.1f}GB")

                        return {
                            "vram_total_gb": round(vram_total, 2),
                            "vram_free_gb": round(vram_free, 2),
                            "vram_used_gb": round(vram_used, 2),
                            "system_ram_total_gb": 0.0,  # Not available via nvidia-smi
                            "system_ram_free_gb": 0.0,
                            "detection_method": detection_method,
                            "gpu_name": gpu_name
                        }
        except FileNotFoundError:
            print("[VRAM] nvidia-smi not found (NVIDIA drivers not installed or not in PATH)")
        except Exception as e:
            print(f"[VRAM] nvidia-smi detection failed: {e}")

        # Fallback 2: Try rocm-smi (AMD GPUs with ROCm)
        print("[VRAM] Attempting rocm-smi fallback...")
        try:
            import subprocess
            result = subprocess.run(
                ['rocm-smi', '--showmeminfo', 'vram', '--json'],
                capture_output=True,
                text=True,
                timeout=5
            )

            if result.returncode == 0:
                import json as json_lib
                try:
                    data = json_lib.loads(result.stdout)

                    # ROCm-smi JSON format varies, try to extract VRAM info
                    # Typical structure: {"card0": {"VRAM Total Memory": "8192 MB", ...}}
                    for card_id, card_data in data.items():
                        if isinstance(card_data, dict):
                            # Try to find VRAM total
                            vram_str = card_data.get("VRAM Total Memory (B)", "")
                            if not vram_str:
                                vram_str = card_data.get("VRAM Total", "")

                            if vram_str:
                                # Parse value (might be "8192 MB" or bytes)
                                vram_value = float(vram_str.split()[0]) if isinstance(vram_str, str) else vram_str

                                # Check units
                                if "MB" in str(vram_str) or vram_value > 100:
                                    vram_total = vram_value / 1024  # MB to GB
                                else:
                                    vram_total = vram_value / (1024**3)  # Bytes to GB

                                # ROCm-smi doesn't always provide free/used in simple format
                                # Set to 0 for now
                                detection_method = "rocm_smi"
                                print(f"[VRAM] ✓ Detected via rocm-smi: AMD GPU with {vram_total:.1f}GB")

                                return {
                                    "vram_total_gb": round(vram_total, 2),
                                    "vram_free_gb": 0.0,  # Not easily available
                                    "vram_used_gb": 0.0,  # Not easily available
                                    "system_ram_total_gb": 0.0,
                                    "system_ram_free_gb": 0.0,
                                    "detection_method": detection_method,
                                    "gpu_name": f"AMD GPU ({card_id})"
                                }
                except json_lib.JSONDecodeError:
                    pass
        except FileNotFoundError:
            print("[VRAM] rocm-smi not found (AMD ROCm drivers not installed)")
        except Exception as e:
            print(f"[VRAM] rocm-smi detection failed: {e}")

        # Fallback 3: Try rocm-smi with simpler output format (alternative)
        try:
            import subprocess
            result = subprocess.run(
                ['rocm-smi', '--showmeminfo', 'vram'],
                capture_output=True,
                text=True,
                timeout=5
            )

            if result.returncode == 0:
                output = result.stdout
                # Parse text output for VRAM info
                # Example: "GPU[0] : VRAM Total Memory (B): 8589934592"
                import re
                vram_match = re.search(r'VRAM Total.*?[:\s]+(\d+)', output)
                if vram_match:
                    vram_bytes = float(vram_match.group(1))
                    vram_total = vram_bytes / (1024**3)  # Bytes to GB

                    detection_method = "rocm_smi"
                    print(f"[VRAM] ✓ Detected via rocm-smi (text): AMD GPU with {vram_total:.1f}GB")

                    return {
                        "vram_total_gb": round(vram_total, 2),
                        "vram_free_gb": 0.0,
                        "vram_used_gb": 0.0,
                        "system_ram_total_gb": 0.0,
                        "system_ram_free_gb": 0.0,
                        "detection_method": detection_method,
                        "gpu_name": "AMD GPU"
                    }
        except FileNotFoundError:
            pass  # Already logged above
        except Exception as e:
            print(f"[VRAM] rocm-smi (text) detection failed: {e}")

        # Fallback 4: Try clinfo (works for AMD/NVIDIA/Intel via OpenCL)
        print("[VRAM] Attempting clinfo fallback...")
        try:
            import subprocess
            result = subprocess.run(
                ['clinfo'],
                capture_output=True,
                text=True,
                timeout=5
            )

            if result.returncode == 0:
                output = result.stdout
                # Parse clinfo output for GPU memory
                # Example: "Global memory size: 8589934592 (8GiB)"
                import re

                # Look for the first discrete GPU (not CPU/integrated)
                lines = output.split('\n')
                current_device_name = None

                for i, line in enumerate(lines):
                    # Find device name
                    if "Device Name" in line:
                        current_device_name = line.split('Device Name')[-1].strip()

                    # Find global memory size
                    if "Global memory size" in line or "Global Memory" in line:
                        # Check if this is a discrete GPU (not Intel HD/UHD)
                        if current_device_name and ("AMD" in current_device_name or "NVIDIA" in current_device_name or "Radeon" in current_device_name):
                            mem_match = re.search(r'(\d+)\s*\((\d+\.?\d*)([GM]i?B)\)', line)
                            if mem_match:
                                value = float(mem_match.group(2))
                                unit = mem_match.group(3)

                                if 'G' in unit:
                                    vram_total = value
                                else:  # MB
                                    vram_total = value / 1024

                                detection_method = "clinfo"
                                print(f"[VRAM] ✓ Detected via clinfo: {current_device_name} with {vram_total:.1f}GB")

                                return {
                                    "vram_total_gb": round(vram_total, 2),
                                    "vram_free_gb": 0.0,  # Not available via clinfo
                                    "vram_used_gb": 0.0,
                                    "system_ram_total_gb": 0.0,
                                    "system_ram_free_gb": 0.0,
                                    "detection_method": detection_method,
                                    "gpu_name": current_device_name
                                }
        except FileNotFoundError:
            print("[VRAM] clinfo not found (OpenCL not installed)")
        except Exception as e:
            print(f"[VRAM] clinfo detection failed: {e}")

        # If all methods failed, return zeros
        print("[VRAM] ✗ All detection methods failed")
        return {
            "vram_total_gb": 0.0,
            "vram_free_gb": 0.0,
            "vram_used_gb": 0.0,
            "system_ram_total_gb": 0.0,
            "system_ram_free_gb": 0.0,
            "detection_method": "failed"
        }
