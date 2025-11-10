# Dual-LLM Configuration Guide

## Overview

The SD Prompt Assistant supports using **different LLMs for different tasks**:

- **Planning LLM**: Used for initial prompt engineering (prioritizes quality)
- **Execution LLM**: Used for quick iterations and refinements (prioritizes speed)

This allows you to get the best of both worlds: high-quality initial prompts and fast iterations.

## Why Use Dual-LLM?

### Use Case Examples

**Example 1: Claude Planning + Gemini Execution**
```env
USE_DUAL_LLM=true
PLANNING_LLM_PROVIDER=claude
PLANNING_CLAUDE_MODEL=claude-3-5-sonnet-20241022
EXECUTION_LLM_PROVIDER=gemini
EXECUTION_GEMINI_MODEL=gemini-1.5-flash
```

**Benefits:**
- Initial prompts use Claude Sonnet (best prompt engineering quality)
- Iterations use Gemini Flash (very fast, free tier available)
- Cost-effective: Only use expensive Claude for initial prompts

**Example 2: Local Ollama Dual Models**
```env
USE_DUAL_LLM=true
PLANNING_LLM_PROVIDER=ollama
PLANNING_OLLAMA_MODEL=llama3.1:8b
EXECUTION_LLM_PROVIDER=ollama
EXECUTION_OLLAMA_MODEL=llama3.2:3b
```

**Benefits:**
- Completely local/private
- Planning uses larger 8B model (better quality)
- Execution uses smaller 3B model (faster iterations)
- Auto-configured based on available VRAM

**Example 3: Cloud Planning + Local Execution**
```env
USE_DUAL_LLM=true
PLANNING_LLM_PROVIDER=claude
PLANNING_CLAUDE_MODEL=claude-3-5-sonnet-20241022
EXECUTION_LLM_PROVIDER=ollama
EXECUTION_OLLAMA_MODEL=llama3.2:3b
```

**Benefits:**
- Best quality initial prompts (Claude)
- Private/fast local iterations (Ollama)
- Cost-effective (only pay for initial prompts)

## How It Works

### Planning LLM Tasks
The planning LLM handles:
1. **Initial prompt generation** from user description
2. Complex prompt engineering decisions
3. Style and parameter recommendations

### Execution LLM Tasks
The execution LLM handles:
1. **Prompt iterations** based on feedback
2. Quick refinements ("make it darker", "add more detail")
3. Parameter adjustments

## Hardware Auto-Configuration

When using Ollama with `OLLAMA_AUTO_CONFIGURE=true`, the system:

1. **Detects your GPU VRAM**
2. **Recommends optimal models** for your hardware
3. **Auto-configures dual-LLM** if enabled

### Example Auto-Configuration

**16GB VRAM:**
```
Planning: llama3.1:8b (uses ~5GB)
Execution: phi3:mini (uses ~2GB)
Total: ~7GB, leaving 9GB free for SD
```

**8GB VRAM:**
```
Planning: mistral:7b (uses ~4GB)
Execution: llama3.2:3b (uses ~2GB)
Total: ~6GB, leaving 2GB free for SD
```

**4GB VRAM:**
```
Planning: llama3.2:3b (uses ~2GB)
Execution: gemma2:2b (uses ~1.5GB)
Total: ~3.5GB, leaving 0.5GB free
```

## Setup Guide

### Step 1: Check Your Hardware

Get hardware recommendations:
```bash
curl http://localhost:8000/hardware
```

This shows:
- Detected GPU and VRAM
- Recommended models for single-LLM
- Recommended models for dual-LLM

### Step 2: Configure `.env`

#### Option A: Auto-Configure (Recommended for Ollama)
```env
USE_DUAL_LLM=true
PLANNING_LLM_PROVIDER=ollama
EXECUTION_LLM_PROVIDER=ollama
OLLAMA_AUTO_CONFIGURE=true  # Let the system choose models
```

#### Option B: Manual Configuration
```env
USE_DUAL_LLM=true

# Planning: Use best quality model
PLANNING_LLM_PROVIDER=claude
PLANNING_CLAUDE_MODEL=claude-3-5-sonnet-20241022

# Execution: Use fastest model
EXECUTION_LLM_PROVIDER=gemini
EXECUTION_GEMINI_MODEL=gemini-1.5-flash
```

### Step 3: Download Models (if using Ollama)

```bash
# Planning model (larger, better quality)
ollama pull llama3.1:8b

# Execution model (smaller, faster)
ollama pull llama3.2:3b
```

### Step 4: Start the Server

```bash
python -m backend.main
```

You'll see:
```
🤖 LLM Mode: Dual
   Planning: Claude Sonnet (claude-3-5-sonnet-20241022)
   Execution: Google Gemini (gemini-1.5-flash)
```

## Recommended Configurations

### For Best Quality
```env
USE_DUAL_LLM=true
PLANNING_LLM_PROVIDER=claude
PLANNING_CLAUDE_MODEL=claude-3-5-sonnet-20241022
EXECUTION_LLM_PROVIDER=claude
EXECUTION_CLAUDE_MODEL=claude-3-5-haiku-20241022
```

### For Best Speed
```env
USE_DUAL_LLM=true
PLANNING_LLM_PROVIDER=gemini
PLANNING_GEMINI_MODEL=gemini-1.5-flash
EXECUTION_LLM_PROVIDER=gemini
EXECUTION_GEMINI_MODEL=gemini-1.5-flash
```

### For Best Cost
```env
USE_DUAL_LLM=true
PLANNING_LLM_PROVIDER=gemini
PLANNING_GEMINI_MODEL=gemini-1.5-flash  # Free tier
EXECUTION_LLM_PROVIDER=ollama  # Free local
EXECUTION_OLLAMA_MODEL=llama3.2:3b
```

### For Privacy
```env
USE_DUAL_LLM=true
PLANNING_LLM_PROVIDER=ollama
EXECUTION_LLM_PROVIDER=ollama
OLLAMA_AUTO_CONFIGURE=true  # Everything local
```

## Performance Comparison

| Configuration | Initial Prompt Quality | Iteration Speed | Cost/100 prompts | Privacy |
|---------------|----------------------|-----------------|------------------|---------|
| **Claude + Gemini** | ★★★★★ | ★★★★★ | $1.50 | ❌ |
| **Claude + Ollama** | ★★★★★ | ★★★★ | $1.00 | ⚠️ |
| **Ollama Dual** | ★★★★ | ★★★★ | $0.00 | ✅ |
| **Gemini + Ollama** | ★★★★ | ★★★★★ | $0.00* | ⚠️ |
| **Single Claude** | ★★★★★ | ★★★★ | $3.00 | ❌ |
| **Single Ollama** | ★★★ | ★★★ | $0.00 | ✅ |

*Gemini free tier available

## Troubleshooting

### Both Models Loading Slowly
- Check GPU VRAM usage
- Reduce model sizes
- Use cloud LLMs for planning

### Out of Memory Errors
- Enable `OLLAMA_AUTO_CONFIGURE=true`
- Manually specify smaller models
- Use cloud LLMs for planning

### Planning LLM Not Being Used
- Check that `USE_DUAL_LLM=true`
- Verify planning provider is configured correctly
- Check server startup logs

## FAQ

**Q: Can I mix local and cloud LLMs?**
A: Yes! E.g., Claude for planning, Ollama for execution.

**Q: Does dual-LLM use more VRAM?**
A: For Ollama yes, but auto-configure manages this. Cloud LLMs use no VRAM.

**Q: Which LLM is used when?**
A: Planning LLM for initial prompts, Execution LLM for iterations.

**Q: Can I use the same model twice?**
A: Yes, but there's no benefit. Use single-LLM mode instead.

**Q: How much does dual-LLM cost?**
A: Depends on providers. Claude planning + Gemini execution ≈ $0.01-0.03 per image.
