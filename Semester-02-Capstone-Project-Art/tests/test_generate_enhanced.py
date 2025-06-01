#!/usr/bin/env python3
"""TDD tests for Art Capstone project - 2024."""

import pathlib
import pytest
import sys

# Add scripts directory to path
ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

try:
    from generate import ImageGenerator, create_default_prompts
    from utils import get_device, set_seed
except ImportError as e:
    pytest.skip(f"Cannot import required modules: {e}", allow_module_level=True)


class TestImageGenerator:
    """TDD tests for ImageGenerator class."""
    
    def test_initialization_valid_model_id(self):
        """Test ImageGenerator initializes correctly."""
        model_id = "stabilityai/stable-diffusion-xl-base-1.0"
        generator = ImageGenerator(model_id)
        assert generator.model_id == model_id
        assert generator.device in ["mps", "cuda", "cpu"]

    def test_initialization_empty_model_id_error(self):
        """Test empty model ID raises ValueError."""
        with pytest.raises(ValueError):
            ImageGenerator("")

    def test_validate_params_valid(self):
        """Test parameter validation works."""
        generator = ImageGenerator()
        # Should not raise
        generator.validate_generation_params("test", 30, 7.5, 512, 512)

    def test_validate_params_empty_prompt_error(self):
        """Test empty prompt raises ValueError."""
        generator = ImageGenerator()
        with pytest.raises(ValueError):
            generator.validate_generation_params("", 30, 7.5, 512, 512)


class TestUtils:
    """Test utility functions."""
    
    def test_get_device_valid(self):
        """Test device detection."""
        device = get_device()
        assert device in ["mps", "cuda", "cpu"]
    
    def test_create_default_prompts(self):
        """Test default prompts."""
        prompts = create_default_prompts()
        assert isinstance(prompts, list)
        assert len(prompts) > 0
