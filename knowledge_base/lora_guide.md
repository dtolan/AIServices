# LoRA Guide

## What are LoRAs?

LoRAs (Low-Rank Adaptation) are small model add-ons that specialize in specific styles, characters, or concepts. They work with your base SD model to add new capabilities without requiring a full model download.

## LoRA Syntax

To use a LoRA in a prompt, use this syntax:
```
<lora:lora_name:weight>
```

- `lora_name`: The filename of the LoRA (without .safetensors extension)
- `weight`: Strength from 0.0 to 2.0 (default: 1.0)

Example:
```
a beautiful landscape <lora:simpsons_style:0.8>
```

## Common LoRA Types

### Style LoRAs
- **Pixel Art**: `<lora:pixel_art:0.8>` - Converts images to pixel art style
- **Anime Styles**: Various anime-specific styles (Studio Ghibli, Makoto Shinkai, etc.)
- **Art Movements**: Impressionism, Cubism, Art Nouveau, etc.
- **Film Styles**: Cinematic, noir, vintage film grain

### Character/Celebrity LoRAs
- Used to generate specific people or characters
- Usually require trigger words mentioned in the LoRA description
- Weight typically 0.7-1.0 for best results

### Concept LoRAs
- **Lighting**: Dramatic lighting, neon lighting, golden hour
- **Camera**: Specific camera angles, lens effects, depth of field
- **Quality**: Detail enhancers, sharpness, texture improvements

### TV Show/Cartoon LoRAs
- **Simpsons Style**: `<lora:simpsons:0.8-1.0>` - Matt Groening's distinctive art style
- **South Park Style**: `<lora:southpark:0.9>` - Cutout animation style
- **Disney Style**: Various Disney-specific art styles
- **Anime Series**: Specific anime art styles (Naruto, One Piece, etc.)

## LoRA Weight Guidelines

- **0.3-0.5**: Subtle influence, blends with base model
- **0.6-0.8**: Moderate influence, balanced result
- **0.9-1.2**: Strong influence, obvious style change
- **1.3-2.0**: Maximum influence, may override base model heavily

## When to Recommend LoRAs

Recommend LoRAs when:
1. User requests a specific art style not in base model
2. User mentions a specific character, celebrity, or franchise
3. User wants a particular visual effect or aesthetic
4. Base model alone won't achieve the desired result

## Multiple LoRAs

You can use multiple LoRAs in one prompt:
```
masterpiece, portrait <lora:ghibli_style:0.7> <lora:detailed_face:0.5>
```

Best practices:
- Keep total combined weight under 2.0
- Test combinations as they may conflict
- Order matters - earlier LoRAs have slightly more influence

## Finding LoRAs

Popular sources:
- CivitAI (https://civitai.com) - Largest LoRA repository
- HuggingFace - Many free LoRAs
- Tensor.Art - Curated LoRA collections

## LoRA Recommendations by Request

### "Make it look like The Simpsons"
- Primary: `<lora:simpsons:0.8>` or `<lora:simpsons_style:1.0>`
- Trigger words: "simpsons style", "matt groening style"
- Increase weight to 1.0-1.2 for stronger effect

### "Anime style"
- Check for specific anime LoRAs (Ghibli, Shinkai, etc.)
- Anime base models work better than LoRAs for general anime

### "Pixar/3D Animation"
- `<lora:pixar_style:0.8>` or `<lora:3d_render:0.7>`
- Works best with "3d render" in prompt

### "Vintage/Retro"
- `<lora:vintage_photo:0.7>` for film photography look
- `<lora:retro_poster:0.8>` for vintage poster aesthetic

## Important Notes

1. **Always check if requested LoRA is installed** before recommending
2. **Provide fallback suggestions** if LoRA not available
3. **Explain what the LoRA does** so user understands the recommendation
4. **Suggest appropriate weights** based on desired strength
5. **Mention trigger words** if the LoRA requires them
6. **Don't add LoRA syntax to prompts** unless the user specifically requested a style that needs it
