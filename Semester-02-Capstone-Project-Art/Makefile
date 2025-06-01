.PHONY: help setup test clean lint format run generate download-model package

# Default target
help:
	@echo "Art Capstone - Available commands:"
	@echo "  setup           - Set up development environment"
	@echo "  download-model  - Download base Stable Diffusion model"
	@echo "  test            - Run test suite"
	@echo "  test-verbose    - Run tests with verbose output"
	@echo "  generate        - Generate image variants"
	@echo "  lint            - Run code linting"
	@echo "  format          - Format code"
	@echo "  clean           - Clean temporary files"
	@echo "  package         - Create submission package"
	@echo "  check-device    - Check device compatibility"

# Environment setup
setup:
	@echo "Setting up Art Capstone environment..."
	@if command -v conda >/dev/null 2>&1; then \
		echo "Using conda environment..."; \
		conda env create -f env/environment.yml || conda env update -f env/environment.yml; \
		echo "Activate with: conda activate art_capstone"; \
	else \
		echo "Using pip virtual environment..."; \
		python -m venv venv; \
		. venv/bin/activate && pip install -r requirements.txt; \
		echo "Activate with: source venv/bin/activate"; \
	fi
	@echo "✓ Environment setup complete"

# Download model
download-model:
	@echo "Downloading Stable Diffusion model..."
	python -m scripts.utils download_model
	@echo "✓ Model download complete"

# Testing
test:
	@echo "Running test suite..."
	python -m pytest tests/ -v
	@echo "✓ Tests complete"

test-verbose:
	@echo "Running test suite with verbose output..."
	python -m pytest tests/ -v -s --tb=short
	@echo "✓ Tests complete"

# Image generation
generate:
	@echo "Generating image variants..."
	@if [ ! -f prompt.txt ]; then \
		echo "Creating default prompt file..."; \
		echo "masterpiece album cover art, vintage aesthetic, professional photography" > prompt.txt; \
	fi
	python scripts/generate.py --prompt-file prompt.txt --num-variants 3 --use-defaults
	@echo "✓ Generation complete"

# Utilities
check-device:
	@echo "Checking device compatibility..."
	python -m scripts.utils test_device

# Packaging
package:
	@echo "Creating submission package..."
	python -m scripts.utils pack_submission
	@echo "✓ Package created: art_capstone_submission.zip"

# Cleanup
clean:
	@echo "Cleaning temporary files..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name "*.pyo" -delete 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	@echo "✓ Cleanup complete"

# Quick start for new users
quickstart:
	@echo "🚀 Art Capstone Quick Start"
	@echo "1. Setting up environment..."
	@make setup
	@echo "2. Downloading model..."
	@make download-model
	@echo "3. Running tests..."
	@make test
	@echo "4. Generating sample images..."
	@make generate
	@echo "✅ Quick start complete! Check data/generated/ for results."