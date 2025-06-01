#!/usr/bin/env python3
"""Smoke tests for generation pipeline."""

import pathlib
import pytest
import sys
from unittest.mock import Mock, patch

# Add scripts directory to path
ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

try:
    from generate import ImageGenerator, create_default_prompts
    from utils import get_device, set_seed, ensure_directory
except ImportError as e:
    pytest.skip(f"Cannot import required modules: {e}", allow_module_level=True)


class TestImageGenerator:
    """Test ImageGenerator class functionality."""
    
    def test_generator_initialization(self):
        """Test that ImageGenerator can be initialized."""
        generator = ImageGenerator()
        assert generator.model_id == "stabilityai/stable-diffusion-xl-base-1.0"
        assert generator.device in ["mps", "cuda", "cpu"]
        assert generator.pipe is None
        assert generator.generation_log == []


class TestUtilityFunctions:
    """Test utility functions."""
    
    def test_get_device(self):
        """Test device detection."""
        device = get_device()
        assert device in ["mps", "cuda", "cpu"]
    
    def test_create_default_prompts(self):
        """Test default prompt creation."""
        prompts = create_default_prompts()
        assert isinstance(prompts, list)
        assert len(prompts) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])