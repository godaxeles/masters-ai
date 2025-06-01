#!/usr/bin/env python3
"""Behavior-Driven Development tests for Art Capstone project - 2024."""

import pathlib
import pytest
import sys
from unittest.mock import Mock, patch
from PIL import Image

# Add scripts directory to path
ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

try:
    from generate import ImageGenerator, create_default_prompts
    from utils import get_device, ensure_directory
except ImportError as e:
    pytest.skip(f"Cannot import required modules: {e}", allow_module_level=True)

@pytest.mark.bdd
class TestImageGenerationBDD:
    """BDD tests following Given-When-Then pattern."""

    def test_user_can_generate_image_with_valid_prompt(self):
        """
        GIVEN a user has a valid prompt
        WHEN they generate an image
        THEN they should receive a valid image
        """
        # GIVEN
        generator = ImageGenerator()
        generator.pipe = Mock()  # Mock the pipeline
        
        # Mock the pipeline to return a fake image result
        mock_image = Mock(spec=Image.Image)
        mock_result = Mock()
        mock_result.images = [mock_image]
        generator.pipe.return_value = mock_result
        prompt = "a beautiful landscape"
        
        # WHEN
        with patch('torch.inference_mode'):
            result_image = generator.generate_image(prompt)
        
        # THEN
        assert result_image == mock_image
        assert len(generator.generation_log) == 1
        assert generator.generation_log[0]['prompt'] == prompt
