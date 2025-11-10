# Anime Style Guide

## Overview
Creating anime-style images with Stable Diffusion requires specific prompt patterns and parameters.

## Key Prompt Elements
- Style tags: "anime", "manga style", "cel shaded", "anime art"
- Quality tags: "best quality", "masterpiece", "highly detailed"
- Character features: "anime eyes", "expressive face", "detailed hair"
- Common artists: "makoto shinkai", "kyoto animation", "studio ghibli style"

## Recommended Parameters
- Steps: 28-35 (anime benefits from more steps)
- CFG Scale: 7-9 (higher for more prompt adherence)
- Sampler: DPM++ 2M Karras or Euler a
- Resolution: 512x768 (portrait) or 768x512 (landscape)

## Negative Prompts
Always include: "realistic, photorealistic, 3d, blurry, low quality, distorted, ugly"

## Example Prompts
### Character Portrait
```
anime girl, long flowing hair, detailed eyes, school uniform, cherry blossoms,
soft lighting, pastel colors, masterpiece, best quality, highly detailed,
in the style of makoto shinkai
```

### Action Scene
```
anime warrior, dynamic pose, energy effects, dramatic lighting, detailed armor,
motion blur, cinematic composition, best quality, highly detailed anime art
```
