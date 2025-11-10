import os
from pathlib import Path
from typing import List, Dict, Optional
import re


class KnowledgeBase:
    """
    Manages local knowledge base for prompt hints and style guides
    Reads .md files from knowledge_base directory
    """

    def __init__(self, knowledge_base_path: str = "knowledge_base"):
        self.kb_path = Path(knowledge_base_path)
        self.kb_path.mkdir(exist_ok=True)
        self._cache = {}

    def _read_markdown_file(self, file_path: Path) -> Dict[str, str]:
        """Read a markdown file and extract sections"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Parse markdown structure
            sections = {}
            current_section = "intro"
            current_content = []

            for line in content.split('\n'):
                # Check for headers
                if line.startswith('#'):
                    # Save previous section
                    if current_content:
                        sections[current_section] = '\n'.join(current_content).strip()

                    # Start new section
                    current_section = line.lstrip('#').strip().lower().replace(' ', '_')
                    current_content = []
                else:
                    current_content.append(line)

            # Save last section
            if current_content:
                sections[current_section] = '\n'.join(current_content).strip()

            return {
                "file": file_path.stem,
                "full_content": content,
                "sections": sections
            }
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            return None

    def load_all_knowledge(self) -> Dict[str, Dict]:
        """Load all knowledge base files"""
        knowledge = {}

        if not self.kb_path.exists():
            return knowledge

        for md_file in self.kb_path.glob("*.md"):
            file_data = self._read_markdown_file(md_file)
            if file_data:
                knowledge[file_data["file"]] = file_data

        return knowledge

    def get_style_guides(self) -> List[str]:
        """Get all available style guides"""
        if not self.kb_path.exists():
            return []

        guides = []
        for md_file in self.kb_path.glob("*style*.md"):
            guides.append(md_file.stem)

        return guides

    def get_prompt_templates(self) -> List[str]:
        """Get all prompt templates"""
        if not self.kb_path.exists():
            return []

        templates = []
        for md_file in self.kb_path.glob("*template*.md"):
            templates.append(md_file.stem)

        return templates

    def search_knowledge(self, query: str) -> List[Dict]:
        """Search knowledge base for relevant information"""
        results = []
        knowledge = self.load_all_knowledge()

        query_lower = query.lower()

        for file_name, file_data in knowledge.items():
            # Search in full content
            if query_lower in file_data["full_content"].lower():
                # Find relevant sections
                relevant_sections = []
                for section_name, section_content in file_data["sections"].items():
                    if query_lower in section_content.lower():
                        relevant_sections.append({
                            "section": section_name,
                            "content": section_content[:500]  # First 500 chars
                        })

                results.append({
                    "file": file_name,
                    "sections": relevant_sections
                })

        return results

    def get_context_for_prompt(self, user_intent: str) -> str:
        """
        Get relevant knowledge base context for a prompt
        This is used to augment the LLM's system prompt
        """
        # Search for relevant information
        results = self.search_knowledge(user_intent)

        if not results:
            return ""

        # Build context string
        context_parts = ["# Relevant Knowledge Base Information:\n"]

        for result in results[:3]:  # Top 3 results
            context_parts.append(f"\n## From {result['file']}:")
            for section in result['sections'][:2]:  # Top 2 sections per file
                context_parts.append(f"\n### {section['section']}:")
                context_parts.append(section['content'])

        return '\n'.join(context_parts)

    def get_all_context(self) -> str:
        """Get all knowledge base content as context"""
        knowledge = self.load_all_knowledge()

        if not knowledge:
            return ""

        context_parts = ["# Available Knowledge Base:\n"]

        for file_name, file_data in knowledge.items():
            context_parts.append(f"\n## {file_name}:")
            context_parts.append(file_data["full_content"][:1000])  # First 1000 chars

        return '\n'.join(context_parts)

    def create_example_guides(self):
        """Create example knowledge base files"""

        # Example style guide
        anime_style = """# Anime Style Guide

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
"""

        photorealistic_style = """# Photorealistic Style Guide

## Overview
Creating photorealistic images requires attention to lighting, detail, and natural composition.

## Key Prompt Elements
- Style tags: "photograph", "photorealistic", "realistic", "photo"
- Quality tags: "8k uhd", "dslr", "professional photography", "award winning"
- Lighting: Be very specific - "golden hour", "studio lighting", "natural light"
- Camera details: "bokeh", "shallow depth of field", "50mm lens", "f/1.8"

## Recommended Parameters
- Steps: 30-40 (more for fine details)
- CFG Scale: 6-8 (lower for more natural look)
- Sampler: DPM++ SDE Karras or DPM++ 2M Karras
- Resolution: 768x768 or higher

## Negative Prompts
Always include: "anime, cartoon, illustration, painting, drawing, art,
low quality, blurry, distorted, ugly, bad anatomy"

## Lighting Types
- **Golden Hour**: warm, soft, natural outdoor lighting
- **Studio**: controlled, professional, even lighting
- **Dramatic**: high contrast, moody, cinematic
- **Natural**: soft daylight, window light

## Example Prompts
### Portrait
```
portrait photo of a woman, natural makeup, soft smile, golden hour lighting,
bokeh background, shallow depth of field, 85mm lens, professional photography,
8k uhd, dslr, high quality, photorealistic
```

### Landscape
```
mountain landscape, dramatic sunset, vivid colors, professional photography,
wide angle lens, high detail, 8k uhd, award winning photo, photorealistic,
national geographic style
```
"""

        prompt_templates = """# Common Prompt Templates

## Character Templates

### Fantasy Character
```
[character type] [action/pose], [clothing/armor details], [setting],
[lighting], [mood], fantasy art, highly detailed, masterpiece, best quality
```

### Modern Character
```
[person description] [clothing], [location], [lighting], [camera angle],
photorealistic, professional photography, 8k, highly detailed
```

## Scene Templates

### Interior Scene
```
[room type], [architectural style], [furniture/decor], [lighting type],
[mood/atmosphere], highly detailed, [art style], best quality
```

### Outdoor Scene
```
[landscape type], [time of day], [weather], [lighting], [mood],
[level of detail], [art style], masterpiece
```

## Quick Modifiers

### Quality Boosters
- Ultra quality: "masterpiece, best quality, highly detailed, 8k uhd"
- Good quality: "high quality, detailed, well composed"
- Artistic: "trending on artstation, award winning, professional"

### Style Modifiers
- Anime: "anime style, cel shaded, anime art"
- Realistic: "photorealistic, photo, realistic"
- Painting: "oil painting, digital painting, brush strokes"
- Concept Art: "concept art, digital art, matte painting"

### Mood/Atmosphere
- Dark/Moody: "dark atmosphere, moody lighting, dramatic"
- Bright/Happy: "vibrant colors, bright lighting, cheerful"
- Mysterious: "mysterious atmosphere, fog, dim lighting"
- Epic: "epic composition, cinematic, grand scale"
"""

        # Write example files
        guides = {
            "anime_style_guide.md": anime_style,
            "photorealistic_style_guide.md": photorealistic_style,
            "prompt_templates.md": prompt_templates
        }

        for filename, content in guides.items():
            file_path = self.kb_path / filename
            if not file_path.exists():
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                print(f"Created example guide: {filename}")
