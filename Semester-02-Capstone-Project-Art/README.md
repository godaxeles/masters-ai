# Art Capstone – Alternative Media Cover Generation

A self-hosted image generation pipeline using Stable Diffusion for creating alternative covers for iconic media (albums, books, movies, etc.).

## Quick Start

```bash
# 1. Clone and setup environment
git clone https://github.com/godaxeles/masters-ai.git && cd Semester-02-Capstone-Project-Art
make setup

# 2. Download base model (≈ 6 GB)
make download-model

# 3. Place original cover in data/original/
cp /path/to/your/cover.jpg data/original/

# 4. Generate variants
make generate

# 5. Run tests
make test

# 6. Create submission
make package
```

## System Requirements

- macOS with Apple Silicon (M1/M2/M3) recommended
- 16+ GB RAM
- 10+ GB free disk space
- Python 3.10+
- Git

## Installation Methods

### Method 1: Conda (Recommended)
```bash
conda env create -f env/environment.yml
conda activate art_capstone
```

### Method 2: Pip + Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows
pip install -r requirements.txt
```

## Project Structure

```
0genaitask/
├── README.md                     # This file
├── cover_generation.md           # Final report template
├── data/
│   ├── original/                 # Place original covers here
│   └── generated/                # AI-generated variants
├── scripts/
│   ├── generate.py               # Main generation script
│   └── utils.py                  # Helper functions
├── tests/                        # Test suite
├── env/environment.yml           # Conda environment
├── requirements.txt              # Pip requirements
├── Makefile                      # Automation commands
└── prompt.txt                    # Sample prompts
```

## Usage Examples

### Basic Generation
```bash
python scripts/generate.py --prompt-file prompt.txt --num-variants 3
```

### Advanced Options
```bash
python scripts/generate.py \
  --prompt-file custom_prompts.txt \
  --num-variants 5 \
  --steps 50 \
  --cfg 8.0 \
  --width 1024 \
  --height 1024 \
  --seed 42
```

### Utility Commands
```bash
# Test device compatibility
python -m scripts.utils test_device

# Download specific model
python -m scripts.utils download_model --model-id "runwayml/stable-diffusion-v1-5"

# Pack for submission
python -m scripts.utils pack_submission
```

## Device Optimization

The pipeline automatically detects and optimizes for your hardware:

- **Apple Silicon (M1/M2/M3)**: Uses MPS acceleration with memory optimizations
- **NVIDIA GPU**: Uses CUDA acceleration
- **CPU Fallback**: Slower but functional on any system

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Out of memory errors | Use `--width 512 --height 512` or enable attention slicing |
| Slow generation | Ensure you're using MPS/CUDA acceleration |
| Model download fails | Check internet connection and HuggingFace access |
| Import errors | Reinstall environment: `make clean && make setup` |

## Development

```bash
# Run tests with coverage
make test-verbose

# Format code
make format

# Run linting
make lint

# Development install
pip install -e .
```

## License

This project uses Stable Diffusion models under CreativeML OpenRAIL M license.