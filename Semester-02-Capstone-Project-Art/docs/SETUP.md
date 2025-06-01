# Art Capstone Setup Guide

## Prerequisites

### Hardware Requirements
- macOS with Apple Silicon (M1/M2/M3) **recommended**
- 16+ GB RAM (32+ GB preferred for SDXL)
- 10+ GB free disk space
- Fast internet connection for model downloads

### Software Requirements
- Python 3.8-3.11
- Git
- Either Conda or pip/venv

## Installation Steps

### Step 1: Clone Repository
```bash
git clone <your-repo-url>
cd 0genaitask
```

### Step 2: Environment Setup

#### Option A: Conda (Recommended)
```bash
# Create environment
conda env create -f env/environment.yml

# Activate environment
conda activate art_capstone

# Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import diffusers; print(f'Diffusers: {diffusers.__version__}')"
```

#### Option B: Virtual Environment + pip
```bash
# Create virtual environment
python -m venv venv

# Activate (macOS/Linux)
source venv/bin/activate

# Activate (Windows)
# venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
```

### Step 3: Device Verification
```bash
# Check device compatibility
make check-device
# or
python -m scripts.utils test_device
```

Expected output for Apple Silicon:
```
Using device: mps
✓ Apple Silicon GPU acceleration available
MPS available: True
MPS built: True
```

### Step 4: Download Models
```bash
# Download default SDXL model (~7GB)
make download-model

# Or download specific model
python -m scripts.utils download_model --model-id "runwayml/stable-diffusion-v1-5"
```

### Step 5: Run Tests
```bash
# Basic test suite
make test

# Verbose testing
make test-verbose
```

### Step 6: Generate Sample Images
```bash
# Quick generation test
make generate

# Custom generation
python scripts/generate.py --prompt-file prompt.txt --num-variants 5 --steps 20
```

## Troubleshooting

### Common Issues

#### 1. MPS Not Available
**Problem**: `RuntimeError: MPS backend is not available`
**Solution**:
- Ensure you're on macOS 12.3+ with Apple Silicon
- Update PyTorch: `pip install --upgrade torch torchvision torchaudio`

#### 2. Out of Memory
**Problem**: Generation fails with memory errors
**Solutions**:
- Reduce image size: `--width 512 --height 512`
- Use fewer inference steps: `--steps 20`
- Enable attention slicing (automatic on MPS)

#### 3. Slow Generation
**Problem**: Takes >5 minutes per image
**Solutions**:
- Verify MPS is being used: `python -m scripts.utils test_device`
- Close other applications to free memory
- Use SD 1.5 instead of SDXL: `--model-id "runwayml/stable-diffusion-v1-5"`

#### 4. Import Errors
**Problem**: `ModuleNotFoundError` or import issues
**Solutions**:
```bash
# Reinstall environment
make clean
make setup

# Or manual reinstall
pip uninstall torch torchvision torchaudio diffusers
pip install -r requirements.txt
```

#### 5. Model Download Fails
**Problem**: HuggingFace download errors
**Solutions**:
- Check internet connection
- Try different model: `--model-id "runwayml/stable-diffusion-v1-5"`
- Set HuggingFace cache: `export HF_HOME=/path/to/cache`

### Performance Optimization

#### For Apple Silicon Macs
```bash
# Use optimized settings
python scripts/generate.py \
  --model-id "stabilityai/stable-diffusion-xl-base-1.0" \
  --steps 30 \
  --cfg 7.5 \
  --width 1024 \
  --height 1024
```

#### For Lower Memory Systems (<16GB)
```bash
# Memory-optimized settings
python scripts/generate.py \
  --model-id "runwayml/stable-diffusion-v1-5" \
  --steps 25 \
  --width 512 \
  --height 512
```

## Development Setup

### Additional Tools
```bash
# Install development dependencies
pip install -e ".[dev]"

# Code formatting
make format

# Linting
make lint

# Coverage testing
make test-coverage
```

### Environment Variables
Create `.env` file for optional settings:
```bash
# Optional: Set HuggingFace token for gated models
HF_TOKEN=your_huggingface_token_here

# Optional: Set custom cache directory
HF_HOME=/path/to/huggingface/cache

# Optional: Disable telemetry
HF_HUB_DISABLE_TELEMETRY=1
```

## Quick Start Workflow

For first-time users:
```bash
# One-command setup and test
make quickstart
```

This will:
1. Set up environment
2. Download models
3. Run tests
4. Generate sample images
5. Display results location

## Next Steps

After successful setup:
1. Place original cover in `data/original/`
2. Customize prompts in `prompt.txt`
3. Generate variants with `make generate`
4. Review results in `data/generated/`
5. Complete `cover_generation.md` report