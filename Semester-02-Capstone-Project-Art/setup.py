#!/usr/bin/env python3
"""Setup script for Art Capstone project."""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

# Read requirements
requirements_path = Path(__file__).parent / "requirements.txt"
requirements = []
if requirements_path.exists():
    requirements = [
        line.strip() for line in requirements_path.read_text().splitlines()
        if line.strip() and not line.startswith('#')
    ]

setup(
    name="art-capstone",
    version="1.0.0",
    description="Self-hosted AI image generation pipeline for alternative media covers",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Art Capstone Student",
    author_email="student@example.com",
    url="https://github.com/yourusername/art-capstone",
    packages=find_packages(),
    include_package_data=True,
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-mock",
            "pytest-cov",
            "black",
            "flake8",
            "isort",
        ],
        "docs": [
            "sphinx",
            "sphinx-rtd-theme",
        ],
    },
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Education",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Multimedia :: Graphics",
    ],
    entry_points={
        "console_scripts": [
            "art-capstone-generate=scripts.generate:main",
            "art-capstone-utils=scripts.utils:main",
        ],
    },
    project_urls={
        "Bug Reports": "https://github.com/yourusername/art-capstone/issues",
        "Source": "https://github.com/yourusername/art-capstone",
        "Documentation": "https://github.com/yourusername/art-capstone/wiki",
    },
)