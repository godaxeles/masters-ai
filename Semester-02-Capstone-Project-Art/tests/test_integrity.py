#!/usr/bin/env python3
"""Test suite for project integrity and deliverables."""

import json
import pathlib
import pytest
import markdown
from PIL import Image

# Project paths
ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
ORIGINAL_DIR = DATA_DIR / "original"
GENERATED_DIR = DATA_DIR / "generated"
SCRIPTS_DIR = ROOT / "scripts"


class TestProjectStructure:
    """Test that required project structure exists."""

    def test_required_directories_exist(self):
        """Test that all required directories exist."""
        required_dirs = [
            DATA_DIR,
            ORIGINAL_DIR,
            GENERATED_DIR,
            SCRIPTS_DIR,
            ROOT / "tests",
            ROOT / "env"
        ]

        for directory in required_dirs:
            assert directory.exists(), f"Required directory missing: {directory}"
            assert directory.is_dir(), f"Path exists but is not a directory: {directory}"

    def test_required_files_exist(self):
        """Test that required files exist."""
        required_files = [
            ROOT / "README.md",
            ROOT / "cover_generation.md",
            ROOT / "requirements.txt",
            ROOT / "env" / "environment.yml",
            SCRIPTS_DIR / "generate.py",
            SCRIPTS_DIR / "utils.py"
        ]

        for file_path in required_files:
            assert file_path.exists(), f"Required file missing: {file_path}"
            assert file_path.is_file(), f"Path exists but is not a file: {file_path}"

    def test_scripts_are_executable(self):
        """Test that Python scripts have proper structure."""
        script_files = [
            SCRIPTS_DIR / "generate.py",
            SCRIPTS_DIR / "utils.py"
        ]

        for script in script_files:
            content = script.read_text()
            assert content.startswith("#!/usr/bin/env python3"), f"Missing shebang: {script}"
            assert "if __name__ == \"__main__\":" in content, f"Missing main guard: {script}"


class TestMarkdownFiles:
    """Test markdown file validity."""

    def test_readme_valid_markdown(self):
        """Test that README.md is valid markdown."""
        readme_path = ROOT / "README.md"
        content = readme_path.read_text(encoding="utf-8")

        # Should not raise an exception
        markdown.markdown(content)

        # Check for required sections
        assert "# Art Capstone" in content
        assert "## Quick Start" in content

    def test_cover_generation_valid_markdown(self):
        """Test that cover_generation.md is valid markdown."""
        report_path = ROOT / "cover_generation.md"
        content = report_path.read_text(encoding="utf-8")

        # Should not raise an exception
        markdown.markdown(content)

        # Check for required sections
        assert "# Art Capstone" in content
        assert "## Original Cover" in content
        assert "## AI-Generated Variants" in content
        assert "## Workflow Details" in content


class TestGeneratedContent:
    """Test generated content and variants."""

    def test_generated_directory_structure(self):
        """Test that generated directory has proper structure."""
        assert GENERATED_DIR.exists()
        assert GENERATED_DIR.is_dir()

    @pytest.mark.parametrize("variant_pattern", ["variant-1*.png", "variant-2*.png", "variant-3*.png"])
    def test_variant_files_exist(self, variant_pattern):
        """Test that variant files exist when generated."""
        # This test passes if no generation has been run yet
        variant_files = list(GENERATED_DIR.glob(variant_pattern))

        # If files exist, they should be valid images
        for variant_file in variant_files:
            assert variant_file.is_file()
            assert variant_file.suffix.lower() in ['.png', '.jpg', '.jpeg']

            # Test that image can be opened
            try:
                with Image.open(variant_file) as img:
                    img.verify()
            except Exception as e:
                pytest.fail(f"Invalid image file {variant_file}: {e}")

    def test_generation_log_format(self):
        """Test generation log format if it exists."""
        log_file = GENERATED_DIR / "generation_log.json"

        if log_file.exists():
            with open(log_file, 'r') as f:
                log_data = json.load(f)

            assert isinstance(log_data, list), "Generation log should be a list"

            for entry in log_data:
                required_fields = ['timestamp', 'prompt', 'steps', 'cfg', 'seed', 'model_id']
                for field in required_fields:
                    assert field in entry, f"Missing field in log entry: {field}"


class TestDependencies:
    """Test that dependencies are properly configured."""

    def test_requirements_file_format(self):
        """Test that requirements.txt is properly formatted."""
        req_file = ROOT / "requirements.txt"
        content = req_file.read_text()

        lines = [line.strip() for line in content.split('\n') if line.strip() and not line.startswith('#')]

        # Check for required packages
        required_packages = ['torch', 'diffusers', 'transformers', 'pillow', 'numpy']
        content_lower = content.lower()

        for package in required_packages:
            assert package in content_lower, f"Missing required package: {package}"

    def test_conda_environment_file(self):
        """Test that conda environment file is valid."""
        env_file = ROOT / "env" / "environment.yml"
        content = env_file.read_text()

        # Basic YAML structure checks
        assert "name:" in content
        assert "dependencies:" in content
        assert "python=" in content

        # Check for required packages
        required_packages = ['pytorch', 'diffusers', 'transformers']
        for package in required_packages:
            assert package in content, f"Missing package in conda env: {package}"


class TestScriptFunctionality:
    """Test basic script functionality."""

    def test_utils_imports(self):
        """Test that utils.py imports correctly."""
        try:
            import sys
            sys.path.insert(0, str(SCRIPTS_DIR))
            import utils

            # Test key functions exist
            assert hasattr(utils, 'get_device')
            assert hasattr(utils, 'set_seed')
            assert hasattr(utils, 'load_model')
            assert hasattr(utils, 'ensure_directory')

        except ImportError as e:
            pytest.fail(f"Failed to import utils: {e}")

    def test_generate_imports(self):
        """Test that generate.py imports correctly."""
        try:
            import sys
            sys.path.insert(0, str(SCRIPTS_DIR))
            import generate

            # Test key classes/functions exist
            assert hasattr(generate, 'ImageGenerator')
            assert hasattr(generate, 'main')

        except ImportError as e:
            pytest.fail(f"Failed to import generate: {e}")


class TestIntegration:
    """Integration tests for the complete workflow."""

    def test_device_detection(self):
        """Test that device detection works."""
        import sys
        sys.path.insert(0, str(SCRIPTS_DIR))
        from utils import get_device

        device = get_device()
        assert device in ["mps", "cuda", "cpu"], f"Invalid device detected: {device}"

    def test_directory_creation(self):
        """Test that directory creation utility works."""
        import sys
        import tempfile
        sys.path.insert(0, str(SCRIPTS_DIR))
        from utils import ensure_directory

        with tempfile.TemporaryDirectory() as temp_dir:
            test_path = pathlib.Path(temp_dir) / "test" / "nested" / "directory"
            result = ensure_directory(test_path)

            assert result.exists()
            assert result.is_dir()
            assert result == test_path


if __name__ == "__main__":
    pytest.main([__file__, "-v"])