# Knowledge Base Guide

## Overview

The SD Prompt Assistant can read local `.md` files to learn about your preferred styles, techniques, and prompt patterns. This allows the LLM to generate better prompts based on your custom knowledge.

## How It Works

### Automatic Knowledge Loading

When the server starts, it:
1. Creates a `knowledge_base/` directory
2. Generates example style guides and templates
3. Loads all `.md` files into memory
4. Uses this knowledge when generating prompts

### Knowledge Base Structure

```
AIServices/
└── knowledge_base/
    ├── anime_style_guide.md
    ├── photorealistic_style_guide.md
    ├── prompt_templates.md
    └── your_custom_guide.md  (add your own!)
```

## Example Knowledge Base Files

The system automatically creates these examples:

### 1. `anime_style_guide.md`
Contains:
- Key anime prompt elements
- Recommended parameters for anime
- Artist names and styles
- Example prompts

### 2. `photorealistic_style_guide.md`
Contains:
- Photorealistic prompt patterns
- Lighting techniques
- Camera and lens terminology
- Professional photography terms

### 3. `prompt_templates.md`
Contains:
- Reusable prompt templates
- Quick modifiers
- Mood/atmosphere patterns

## Creating Custom Knowledge

### Add Your Own Style Guide

Create `knowledge_base/cyberpunk_style.md`:

```markdown
# Cyberpunk Style Guide

## Overview
Cyberpunk aesthetic with neon, tech, and dystopian themes.

## Key Elements
- Neon lighting (pink, blue, cyan)
- High-tech, low-life aesthetic
- Rain-slicked streets
- Holographic displays
- Corporate megastructures

## Recommended Prompts
- Style: "cyberpunk, neon lights, futuristic, dystopian"
- Lighting: "neon glow, dramatic lighting, volumetric fog"
- Quality: "highly detailed, 8k, cyberpunk 2077 style"

## Negative Prompts
Always avoid: "medieval, fantasy, natural, rustic"

## Parameters
- Steps: 35-40 (detail matters)
- CFG: 8-10 (strong adherence)
- Sampler: DPM++ SDE Karras (for glow effects)

## Example
```
cyberpunk street scene, neon signs, rain-slicked pavement,
holographic advertisements, futuristic vehicles, night time,
dramatic neon lighting, highly detailed, 8k, blade runner style,
volumetric fog, cinematic composition
```
```

### Add Character Templates

Create `knowledge_base/character_templates.md`:

```markdown
# Character Creation Templates

## Female Character
```
[age] [ethnicity] woman, [hair style] [hair color] hair,
[eye color] eyes, [clothing], [expression], [pose],
[background], [lighting], [style], highly detailed
```

## Male Character
```
[age] [ethnicity] man, [hairstyle], [facial hair],
[clothing], [expression], [pose], [background],
[lighting], [style], highly detailed
```

## Fantasy Character
```
[race] [class], [armor/clothing description],
[weapon/accessories], [magical effects], [pose],
fantasy setting, [lighting], epic composition,
highly detailed fantasy art
```
```

### Add Project-Specific Knowledge

Create `knowledge_base/my_game_art_style.md`:

```markdown
# My Game Art Style

## Visual Style
Our game uses a hand-painted, stylized aesthetic similar to Overwatch.

## Color Palette
- Vibrant, saturated colors
- Cel-shaded look
- Strong rim lighting
- Painterly textures

## Character Guidelines
- Exaggerated proportions
- Clear silhouettes
- Detailed costumes
- Expressive faces

## Environment Guidelines
- Detailed architecture
- Atmospheric effects
- Dynamic lighting
- Rich textures

## Prompt Formula
```
[subject], hand-painted texture, stylized, cel-shaded,
vibrant colors, strong rim lighting, clear silhouette,
in the style of overwatch, highly detailed, game art,
unreal engine, concept art
```
```

## Interactive Question System

### How Questions Work

When you give a vague description:

**User:** "a warrior"

**AI asks:**
```
1. What style? (photorealistic, anime, fantasy art, etc.)
2. What setting/environment? (medieval battlefield, futuristic city, etc.)
3. What mood/atmosphere? (epic, dark, heroic, etc.)
```

**User answers:**
```
1. Fantasy art
2. Standing on a cliff at sunset
3. Epic and heroic
```

**AI generates:**
```
Positive: fantasy warrior, ornate armor, standing on cliff edge,
dramatic sunset background, epic composition, wind blowing cape,
heroic pose, detailed armor, highly detailed fantasy art,
cinematic lighting, golden hour, trending on artstation

Negative: realistic, photorealistic, modern, blurry, low quality

Settings: 35 steps, CFG 8.5, 768x512, DPM++ 2M Karras
```

### Skipping Questions

If you want direct generation without questions:

```bash
POST /generate
{
  "user_input": "detailed description here",
  "skip_questions": true
}
```

## API Endpoints

### Get Knowledge Base Info
```bash
GET /knowledge-base

Response:
{
  "guides_count": 3,
  "templates_count": 1,
  "available_guides": ["anime_style_guide", "photorealistic_style_guide"],
  "available_templates": ["prompt_templates"]
}
```

### Interactive Mode - Ask Questions
```bash
POST /interactive/ask
{
  "user_input": "a dragon",
  "conversation_history": []
}

Response:
{
  "needs_clarification": true,
  "questions": [
    "What style? (realistic, fantasy art, anime, etc.)",
    "What is the dragon doing?",
    "What environment/setting?"
  ],
  "reasoning": "Description is vague, need style and context"
}
```

### Interactive Mode - Generate from Answers
```bash
POST /interactive/generate
{
  "conversation_history": [
    {"role": "user", "content": "a dragon"},
    {"role": "assistant", "content": "What style?"},
    {"role": "user", "content": "fantasy art"},
    {"role": "assistant", "content": "What is it doing?"},
    {"role": "user", "content": "breathing fire over a castle"}
  ]
}

Response:
{
  "enhanced_prompt": {
    "positive_prompt": "detailed prompt based on full conversation",
    ...
  },
  "explanation": "Created epic fantasy scene based on your answers",
  "suggestions": ["Try different times of day", "Add weather effects"]
}
```

## Advanced Usage

### Organizing Knowledge

**By project:**
```
knowledge_base/
├── project_a_style.md
├── project_a_characters.md
├── project_b_style.md
└── project_b_environments.md
```

**By style:**
```
knowledge_base/
├── anime/
│   ├── shonen_style.md
│   ├── seinen_style.md
│   └── moe_style.md
└── realistic/
    ├── portrait_guide.md
    └── landscape_guide.md
```

Note: Currently only files directly in `knowledge_base/` are loaded (no subdirectories).

### Knowledge Search

The system automatically searches knowledge for relevant information based on:
- Keywords in user input
- Style mentions
- Subject matter

Example:
- User says "anime girl" → loads `anime_style_guide.md`
- User says "cyberpunk" → searches for cyberpunk knowledge
- User says "portrait" → finds relevant photo guides

## Best Practices

### 1. Be Specific in Guides
❌ Bad: "Use good lighting"
✅ Good: "Use dramatic rim lighting with warm key light at 45° angle"

### 2. Include Examples
Always provide concrete example prompts with parameters.

### 3. Document Negatives
Include what NOT to do for each style.

### 4. Update Regularly
As you discover better prompts, add them to your knowledge base.

### 5. Use Sections
Organize guides with markdown headers for easy parsing:
- `## Key Elements`
- `## Parameters`
- `## Examples`

## Tips

1. **Start with examples** - The auto-generated guides are good templates
2. **Copy successful prompts** - Add prompts that work well to your knowledge base
3. **Document your workflow** - Create guides for your specific needs
4. **Use multiple files** - Separate by theme/style for easier management
5. **Regular updates** - Knowledge base is loaded on startup, restart to reload changes

## Troubleshooting

### Knowledge Not Being Used
- Check files are in `knowledge_base/` directory
- Ensure files have `.md` extension
- Restart server to reload knowledge
- Check server logs for "Knowledge Base initialized"

### Questions Too Generic
- Add more specific examples to your knowledge base
- Use clearer section headers
- Include concrete prompt examples

### LLM Ignoring Knowledge
- Make examples more prominent
- Use consistent terminology
- Add quality indicators to examples

## Example Workflow

1. **Create custom guide** for your project
2. **Start server** - knowledge loads automatically
3. **Describe vaguely** - "a character for my game"
4. **Answer questions** - LLM asks style, pose, etc.
5. **Get perfect prompt** - Based on your knowledge + answers
6. **Iterate** - Refine based on results
7. **Update knowledge** - Add successful patterns

## Future Enhancements

Coming soon:
- [ ] Subdirectory support
- [ ] Hot-reload (no restart needed)
- [ ] Web UI for managing knowledge
- [ ] Prompt voting (track what works)
- [ ] AI-suggested knowledge additions
