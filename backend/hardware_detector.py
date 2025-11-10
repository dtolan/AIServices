import subprocess
import platform
import re
from typing import Dict, Optional, List
from dataclasses import dataclass


@dataclass
class GPUInfo:
    """Information about available GPU"""
    name: str
    vram_total_mb: int
    vram_free_mb: int
    driver_version: Optional[str] = None


@dataclass
class ModelRecommendation:
    """Recommended model based on available resources"""
    model_name: str
    model_size: str
    vram_required_mb: int
    reason: str


class HardwareDetector:
    """Detect hardware capabilities and recommend optimal LLM configurations"""

    # Model VRAM requirements (approximate, in MB)
    MODEL_VRAM_REQUIREMENTS = {
        "llama3.2:1b": 1024,      # 1GB - Tiny, fast
        "llama3.2:3b": 2048,      # 2GB - Small, good balance
        "llama3.2:latest": 2048,  # 2GB - Default (3B)
        "llama3.1:8b": 5120,      # 5GB - Medium, capable
        "llama3.1:70b": 40960,    # 40GB - Large, very capable
        "mistral:7b": 4096,       # 4GB - Good quality
        "mistral:latest": 4096,   # 4GB
        "codellama:7b": 4096,     # 4GB
        "phi3:mini": 2048,        # 2GB - Fast, efficient
        "phi3:medium": 8192,      # 8GB - Balanced
        "qwen2.5:0.5b": 512,      # 512MB - Ultra-light
        "qwen2.5:3b": 2048,       # 2GB
        "gemma2:2b": 1536,        # 1.5GB - Fast
        "gemma2:9b": 6144,        # 6GB - Quality
    }

    @staticmethod
    def detect_nvidia_gpu() -> Optional[GPUInfo]:
        """Detect NVIDIA GPU using nvidia-smi"""
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=name,memory.total,memory.free,driver_version',
                 '--format=csv,noheader,nounits'],
                capture_output=True,
                text=True,
                timeout=5
            )

            if result.returncode == 0 and result.stdout:
                lines = result.stdout.strip().split('\n')
                if lines:
                    # Parse first GPU
                    parts = [p.strip() for p in lines[0].split(',')]
                    if len(parts) >= 3:
                        return GPUInfo(
                            name=parts[0],
                            vram_total_mb=int(float(parts[1])),
                            vram_free_mb=int(float(parts[2])),
                            driver_version=parts[3] if len(parts) > 3 else None
                        )
        except (subprocess.TimeoutExpired, FileNotFoundError, ValueError):
            pass
        return None

    @staticmethod
    def detect_amd_gpu() -> Optional[GPUInfo]:
        """Detect AMD GPU using rocm-smi"""
        try:
            result = subprocess.run(
                ['rocm-smi', '--showmeminfo', 'vram'],
                capture_output=True,
                text=True,
                timeout=5
            )

            if result.returncode == 0 and result.stdout:
                # Parse rocm-smi output (format varies)
                # This is a simplified implementation
                vram_match = re.search(r'VRAM Total.*?(\d+)', result.stdout)
                if vram_match:
                    vram_total = int(vram_match.group(1))
                    return GPUInfo(
                        name="AMD GPU",
                        vram_total_mb=vram_total,
                        vram_free_mb=vram_total  # Approximate
                    )
        except (subprocess.TimeoutExpired, FileNotFoundError, ValueError):
            pass
        return None

    @staticmethod
    def detect_metal_gpu() -> Optional[GPUInfo]:
        """Detect Metal GPU on macOS (Apple Silicon)"""
        if platform.system() != "Darwin":
            return None

        try:
            # Use system_profiler to get GPU info on macOS
            result = subprocess.run(
                ['system_profiler', 'SPDisplaysDataType'],
                capture_output=True,
                text=True,
                timeout=5
            )

            if result.returncode == 0 and result.stdout:
                # Look for Metal/GPU info
                if "Apple" in result.stdout or "Metal" in result.stdout:
                    # Estimate based on Mac model (simplified)
                    # M1/M2 have unified memory
                    return GPUInfo(
                        name="Apple Silicon GPU (Metal)",
                        vram_total_mb=8192,  # Conservative estimate
                        vram_free_mb=6144
                    )
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
        return None

    @classmethod
    def detect_gpu(cls) -> Optional[GPUInfo]:
        """Detect any available GPU"""
        # Try NVIDIA first (most common)
        gpu = cls.detect_nvidia_gpu()
        if gpu:
            return gpu

        # Try AMD
        gpu = cls.detect_amd_gpu()
        if gpu:
            return gpu

        # Try Metal (macOS)
        gpu = cls.detect_metal_gpu()
        if gpu:
            return gpu

        return None

    @classmethod
    def recommend_ollama_model(cls, available_vram_mb: int) -> ModelRecommendation:
        """Recommend optimal Ollama model based on available VRAM"""

        # Leave 20% VRAM free for overhead
        usable_vram = int(available_vram_mb * 0.8)

        # Find best model that fits
        recommendations = [
            (70000, "llama3.1:70b", "70B", "Maximum quality, very slow"),
            (40000, "mistral:latest", "7B", "Excellent quality and speed balance"),
            (8000, "phi3:medium", "14B", "Great quality, good speed"),
            (6000, "gemma2:9b", "9B", "High quality, efficient"),
            (5000, "llama3.1:8b", "8B", "Very good quality"),
            (4000, "mistral:7b", "7B", "Good quality, fast"),
            (2500, "llama3.2:3b", "3B", "Balanced quality and speed"),
            (2000, "phi3:mini", "3.8B", "Fast, good for quick tasks"),
            (1500, "gemma2:2b", "2B", "Fast, lightweight"),
            (1000, "llama3.2:1b", "1B", "Very fast, basic quality"),
            (500, "qwen2.5:0.5b", "0.5B", "Ultra-fast, minimal quality"),
        ]

        for vram_required, model, size, reason in recommendations:
            if usable_vram >= vram_required:
                return ModelRecommendation(
                    model_name=model,
                    model_size=size,
                    vram_required_mb=vram_required,
                    reason=reason
                )

        # Fallback to smallest model
        return ModelRecommendation(
            model_name="qwen2.5:0.5b",
            model_size="0.5B",
            vram_required_mb=500,
            reason="Minimal VRAM available, using smallest model"
        )

    @classmethod
    def get_dual_llm_recommendation(cls, available_vram_mb: int) -> Dict[str, ModelRecommendation]:
        """
        Recommend planning and execution models for dual-LLM setup
        Planning gets the better model, execution gets the faster model
        """
        usable_vram = int(available_vram_mb * 0.8)

        # Allocate 60% to planning, 40% to execution
        planning_vram = int(usable_vram * 0.6)
        execution_vram = int(usable_vram * 0.4)

        return {
            "planning": cls.recommend_ollama_model(planning_vram),
            "execution": cls.recommend_ollama_model(execution_vram)
        }

    @classmethod
    def get_hardware_summary(cls) -> Dict:
        """Get complete hardware summary and recommendations"""
        gpu = cls.detect_gpu()

        if gpu:
            single_model = cls.recommend_ollama_model(gpu.vram_free_mb)
            dual_models = cls.get_dual_llm_recommendation(gpu.vram_free_mb)

            return {
                "gpu_detected": True,
                "gpu_info": {
                    "name": gpu.name,
                    "vram_total_mb": gpu.vram_total_mb,
                    "vram_free_mb": gpu.vram_free_mb,
                    "vram_total_gb": round(gpu.vram_total_mb / 1024, 1),
                    "vram_free_gb": round(gpu.vram_free_mb / 1024, 1),
                    "driver_version": gpu.driver_version
                },
                "recommendations": {
                    "single_llm": {
                        "model": single_model.model_name,
                        "size": single_model.model_size,
                        "vram_required_mb": single_model.vram_required_mb,
                        "vram_required_gb": round(single_model.vram_required_mb / 1024, 1),
                        "reason": single_model.reason
                    },
                    "dual_llm": {
                        "planning": {
                            "model": dual_models["planning"].model_name,
                            "size": dual_models["planning"].model_size,
                            "vram_required_mb": dual_models["planning"].vram_required_mb,
                            "vram_required_gb": round(dual_models["planning"].vram_required_mb / 1024, 1),
                            "reason": dual_models["planning"].reason
                        },
                        "execution": {
                            "model": dual_models["execution"].model_name,
                            "size": dual_models["execution"].model_size,
                            "vram_required_mb": dual_models["execution"].vram_required_mb,
                            "vram_required_gb": round(dual_models["execution"].vram_required_mb / 1024, 1),
                            "reason": dual_models["execution"].reason
                        }
                    }
                }
            }
        else:
            return {
                "gpu_detected": False,
                "message": "No GPU detected. Using cloud LLM providers (Claude/Gemini) is recommended.",
                "recommendations": {
                    "cloud_providers": [
                        {
                            "provider": "claude",
                            "model": "claude-3-5-sonnet-20241022",
                            "reason": "Best quality for prompt engineering"
                        },
                        {
                            "provider": "gemini",
                            "model": "gemini-1.5-flash",
                            "reason": "Fast and cost-effective"
                        }
                    ]
                }
            }
