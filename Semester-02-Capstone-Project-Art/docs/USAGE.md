# Art Capstone Usage Guide

## Quick Start

### Generate Your First Images
```bash
# Use default prompts
python scripts/generate.py --use-defaults --num-variants 3

# Use custom prompt file
python scripts/generate.py --prompt-file my_prompts.txt --num-variants 5
```

### Basic Workflow
1. **Prepare Prompts**: Create or edit `prompt.txt`
2. **Generate**: Run generation script
3. **Review**: Check `data/generated/` for results
4. **Iterate**: Adjust prompts and settings
5. **Document**: Update `cover_generation.md`

## Command Line Interface

### Main Generation Script

```bash
python scripts/generate.py [OPTIONS]
```

#### Required Arguments
- `--prompt-file FILE` or `--use-defaults`: Source of prompts

#### Optional Arguments
- `--output-dir DIR`: Output directory (default: `data/generated`)
- `--num-variants N`: Number of images to generate (default: 3)
- `--steps N`: Inference steps (default: 30)
- `--cfg FLOAT`: Guidance scale (default: 7.5)
- `--seed INT`: Random seed for reproducibility
- `--width INT`: Image width (default: 1024)
- `--height INT`: Image height (default: 1024)
- `--model-id STR`: HuggingFace model ID
- `--verbose`: Enable verbose output

### Utility Scripts

```bash
# Test device compatibility
python -m scripts.utils test_device

# Download models
python -m scripts.utils download_model --model-id "MODEL_NAME"

# Check dependencies
python -m scripts.utils check_deps

# Package project
python -m scripts.utils pack_submission
```

## Prompt Engineering

### Effective Prompt Structure
```
[STYLE] [SUBJECT] [DETAILS] [QUALITY] [COMPOSITION]
```

### Examples

#### Album Cover Prompts
```
masterpiece album cover art, vintage rock aesthetic, electric guitar silhouette, dramatic lighting, professional photography, high detail, iconic composition, bold typography, psychedelic colors

modern minimalist album cover, electronic music theme, geometric patterns, neon accents, clean typography, digital art style, high contrast, contemporary aesthetic
```

#### Book Cover Prompts
```
professional book cover design, fantasy novel theme, mystical forest, ethereal lighting, elegant typography, magical atmosphere, detailed illustration, captivating composition

minimalist book cover, literary fiction, abstract geometric shapes, sophisticated color palette, modern typography, artistic design, clean layout
```

### Prompt Best Practices

1. **Start with quality terms**: `masterpiece`, `professional`, `high detail`
2. **Specify medium**: `album cover art`, `book cover design`, `movie poster`
3. **Add style descriptors**: `vintage`, `modern`, `minimalist`, `psychedelic`
4. **Include composition**: `dramatic lighting`, `bold typography`, `iconic composition`
5. **End with quality**: `professional photography`, `high detail`, `artistic`

### Negative Prompts
Default negative prompt filters out common issues:
```
blurry, low quality, distorted, ugly, bad anatomy, deformed, text errors, watermark
```

## Model Selection

### Available Models

#### Stable Diffusion XL (Recommended)
- **Model ID**: `stabilityai/stable-diffusion-xl-base-1.0`
- **Resolution**: 1024x1024 native
- **Quality**: Highest
- **Speed**: Moderate
- **Memory**: ~8-12 GB

#### Stable Diffusion 1.5 (Faster)
- **Model ID**: `runwayml/stable-diffusion-v1-5`
- **Resolution**: 512x512 native
- **Quality**: Good
- **Speed**: Fast
- **Memory**: ~4-6 GB

#### Stable Diffusion 2.1
- **Model ID**: `stabilityai/stable-diffusion-2-1`
- **Resolution**: 768x768 native
- **Quality**: Very Good
- **Speed**: Moderate
- **Memory**: ~6-8 GB

### Model Selection Guidelines

| Use Case | Recommended Model | Settings |
|----------|-------------------|----------|
| High Quality Covers | SDXL Base 1.0 | 1024x1024, 30-50 steps |
| Quick Iterations | SD 1.5 | 512x512, 20-30 steps |
| Balanced Quality/Speed | SD 2.1 | 768x768, 25-35 steps |
| Low Memory (<16GB) | SD 1.5 | 512x512, 20 steps |

## Generation Settings

### Inference Steps
- **10-15**: Very fast, lower quality
- **20-30**: Balanced speed/quality ⭐ **Recommended**
- **30-50**: High quality, slower
- **50+**: Diminishing returns

### Guidance Scale (CFG)
- **5.0-6.0**: More creative, less prompt adherence
- **7.0-8.0**: Balanced ⭐ **Recommended**
- **8.0-12.0**: Strong prompt adherence
- **12.0+**: May cause artifacts

### Resolution Guidelines

#### For SDXL
- **Native**: 1024x1024
- **Landscape**: 1152x896, 1216x832
- **Portrait**: 896x1152, 832x1216

#### For SD 1.5
- **Native**: 512x512
- **Landscape**: 768x512, 640x512
- **Portrait**: 512x768, 512x640

## Batch Generation

### Multiple Variants
```bash
# Generate 10 variants with random seeds
python scripts/generate.py --prompt-file prompts.txt --num-variants 10

# Generate with specific seeds for reproducibility
for seed in 42 123 456 789; do
  python scripts/generate.py --prompt-file prompts.txt --seed $seed --num-variants 1
done
```

### Different Styles
Create separate prompt files:
```bash
# Generate vintage style
python scripts/generate.py --prompt-file prompts_vintage.txt --num-variants 3

# Generate modern style
python scripts/generate.py --prompt-file prompts_modern.txt --num-variants 3
```

## Output Management

### File Naming Convention
Generated files follow the pattern:
```
variant-{N}_{TIMESTAMP}_seed{SEED}.png
```

Example: `variant-1_20241201_143022_seed42.png`

### Generation Log
Each run creates `generation_log.json` with:
- Timestamp
- Prompt used
- All generation parameters
- Performance metrics
- Device information

### Organizing Results
```bash
# Create organized folders
mkdir -p data/generated/{album_covers,book_covers,movie_posters}

# Generate into specific folders
python scripts/generate.py --output-dir data/generated/album_covers --prompt-file album_prompts.txt
```

## Performance Optimization

### Apple Silicon (M1/M2/M3)
```bash
# Optimal settings for Apple Silicon
python scripts/generate.py \
  --model-id "stabilityai/stable-diffusion-xl-base-1.0" \
  --steps 30 \
  --cfg 7.5 \
  --width 1024 \
  --height 1024
```

### Memory Optimization
```bash
# For systems with <16GB RAM
python scripts/generate.py \
  --model-id "runwayml/stable-diffusion-v1-5" \
  --steps 25 \
  --width 512 \
  --height 512
```

### Speed Optimization
```bash
# Fastest generation
python scripts/generate.py \
  --model-id "runwayml/stable-diffusion-v1-5" \
  --steps 20 \
  --cfg 6.0 \
  --width 512 \
  --height 512
```

## Advanced Usage

### Custom Model Fine-tuning
```bash
# Use community fine-tuned models
python scripts/generate.py \
  --model-id "nitrosocke/Arcane-Diffusion" \
  --prompt-file custom_prompts.txt
```

### Reproducible Generation
```bash
# Generate with fixed seed for reproducibility
python scripts/generate.py \
  --prompt-file prompts.txt \
  --seed 42 \
  --num-variants 1
```

### Experimental Features
```bash
# Use different schedulers (experimental)
python scripts/generate.py \
  --sampler "dpm_multistep" \
  --steps 25
```

## Integration with Makefile

### Convenient Commands
```bash
# Quick generation with defaults
make generate

# Custom generation (prompts for input)
make generate-custom

# Full workflow
make quickstart
```

## Troubleshooting Generation

### Common Issues

#### Poor Quality Images
1. Increase inference steps: `--steps 40`
2. Adjust guidance scale: `--cfg 8.0`
3. Improve prompts with quality terms
4. Use higher resolution: `--width 1024 --height 1024`

#### Out of Memory
1. Reduce resolution: `--width 512 --height 512`
2. Use SD 1.5: `--model-id "runwayml/stable-diffusion-v1-5"`
3. Reduce batch size to 1 variant at a time

#### Slow Generation
1. Reduce steps: `--steps 20`
2. Use smaller model: SD 1.5 instead of SDXL
3. Close other applications
4. Check device usage: `python -m scripts.utils test_device`

### Getting Help
1. Check logs in `data/generated/generation_log.json`
2. Run device test: `make check-device`
3. Verify dependencies: `python -m scripts.utils check_deps`
4. Review setup guide: `docs/SETUP.md`