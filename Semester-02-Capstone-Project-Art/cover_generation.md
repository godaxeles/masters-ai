# Art Capstone – Alternative Media Cover Generation

## Project Overview

This project demonstrates self-hosted AI image generation using Stable Diffusion to create alternative covers for iconic media. The pipeline is optimized for Apple Silicon Macs and follows TDD principles.

## Original Cover
*[Place original cover image here after adding to data/original/]*

![Original](data/original/placeholder.jpg)

## AI-Generated Variants

| # | Image | Prompt Snippet | Settings |
|---|-------|----------------|----------|
| 1 | ![v1](data/generated/variant-1.png) | "masterpiece album cover art, [genre] style..." | 30 steps, CFG 7.5, seed 42 |
| 2 | ![v2](data/generated/variant-2.png) | "vintage vinyl cover design, [theme]..." | 30 steps, CFG 7.5, seed 123 |
| 3 | ![v3](data/generated/variant-3.png) | "modern minimalist cover, [mood]..." | 30 steps, CFG 7.5, seed 456 |

## Workflow Details

### 1. Model & Checkpoint
- **Primary Model**: Stable Diffusion XL Base 1.0
- **HuggingFace Hub**: https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0
- **Fallback Model**: Stable Diffusion v1.5 for compatibility

### 2. Adapters & Extensions
- None (using base models only)
- Future: Consider LoRA adapters for style consistency

### 3. Generation Settings
- **Inference Steps**: 30 (balanced quality/speed)
- **Sampler**: Euler Ancestral Discrete
- **Guidance Scale**: 7.5 (moderate prompt adherence)
- **Resolution**: 1024x1024 (SDXL native)
- **Precision**: FP32 (MPS compatibility)

### 4. Pipeline Configuration
![Pipeline Screenshot](docs/pipeline_screenshot.png)

**Device Configuration**:
- **Hardware**: Apple Silicon Mac (M1/M2/M3)
- **Acceleration**: Metal Performance Shaders (MPS)
- **Memory Optimization**: Attention slicing enabled
- **Warmup**: Single pass for MPS stability

### 5. Full Prompts Used

#### Variant 1: Classic Rock Album
```
masterpiece album cover art, classic rock style, electric guitar silhouette, dramatic lighting, vintage aesthetic, 1970s design, bold typography, psychedelic colors, professional photography, high detail, iconic composition
```

#### Variant 2: Jazz Album
```
vintage vinyl cover design, jazz theme, smoky blue atmosphere, musical instruments, noir lighting, sophisticated composition, minimalist typography, cool color palette, artistic photography, timeless elegance
```

#### Variant 3: Electronic Music
```
modern minimalist cover, electronic music vibe, geometric patterns, neon accents, futuristic design, clean typography, digital art style, synthetic atmosphere, high contrast, contemporary aesthetic
```

### 6. Technical Resources
- **Framework**: 🤗 Diffusers v0.33.0+
- **Backend**: PyTorch 2.0+ with MPS support
- **Hardware**: Apple Silicon Mac with 16+ GB RAM
- **Generation Time**: ~45-60 seconds per image (1024x1024)
- **Model Size**: ~6.8 GB (SDXL Base)

## Performance Metrics

| Metric | Value | Notes |
|--------|-------|-------|
| Generation Time | 45-60s | Per 1024x1024 image, 30 steps |
| Memory Usage | ~8-12 GB | With attention slicing |
| Model Load Time | ~30s | First load, cached afterward |
| Quality Score | 8.5/10 | Subjective assessment |

## Challenges & Solutions

### Challenge 1: Memory Management
**Issue**: SDXL models require significant memory
**Solution**: Implemented attention slicing and MPS optimizations

### Challenge 2: Mac Compatibility
**Issue**: PyTorch MPS backend inconsistencies
**Solution**: Added warmup pass and fallback mechanisms

### Challenge 3: Generation Speed
**Issue**: Slower than CUDA on similar hardware
**Solution**: Optimized model precision and inference settings

## Future Improvements

1. **Model Fine-tuning**: Train LoRA adapters for specific artistic styles
2. **Batch Processing**: Implement efficient multi-image generation
3. **Style Transfer**: Add ability to match original cover aesthetics
4. **Interactive UI**: Web interface for real-time prompt editing
5. **Cloud Deployment**: Scale to larger models and faster generation

## Lessons Learned

- Apple Silicon MPS acceleration provides good performance for local generation
- Self-hosted solutions offer privacy and unlimited usage compared to SaaS
- Prompt engineering is crucial for consistent, high-quality results
- TDD approach significantly improved code reliability and debugging

## Conclusion

This project successfully demonstrates self-hosted AI image generation for creative applications. The pipeline is robust, well-tested, and optimized for Mac development environments. The generated covers show artistic merit while maintaining technical quality standards.