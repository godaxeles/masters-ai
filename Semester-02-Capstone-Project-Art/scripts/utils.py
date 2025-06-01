#!/usr/bin/env python3
"""Utility functions for Art Capstone image generation pipeline - Updated 2024."""

import argparse
import hashlib
import os
import random
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Optional, Union, List, Dict, Any
import warnings

import numpy as np
import torch
from diffusers import (
    StableDiffusionXLPipeline, 
    StableDiffusionPipeline,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    HeunDiscreteScheduler
)
from huggingface_hub import hf_hub_download
from PIL import Image
import psutil
from rich.console import Console
from rich.progress import Progress

console = Console()


def set_seed(seed: int) -> None:
    """Set random seed for reproducible generation."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Apple Silicon specific random seed
    if torch.backends.mps.is_available():
        # MPS doesn't have a specific manual_seed, use general torch
        pass
    
    # CUDA specific
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)



def get_device() -> str:
    """Get the best available device for inference - 2024 optimized."""
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"


def get_device_info() -> Dict[str, Any]:
    """Get detailed device information for optimization."""
    device = get_device()
    info = {
        "device": device,
        "memory_gb": psutil.virtual_memory().total / (1024**3),
        "available_memory_gb": psutil.virtual_memory().available / (1024**3),
    }
    
    if device == "mps":
        info.update({
            "mps_available": torch.backends.mps.is_available(),
            "mps_built": torch.backends.mps.is_built(),
            "recommended_dtype": "float32",  # MPS works best with float32
        })
    elif device == "cuda":
        if torch.cuda.is_available():
            info.update({
                "cuda_version": torch.version.cuda,
                "gpu_name": torch.cuda.get_device_name(0),
                "gpu_memory_gb": torch.cuda.get_device_properties(0).total_memory / (1024**3),
                "recommended_dtype": "float16",
            })
    else:
        info.update({
            "recommended_dtype": "float32",
            "warning": "Using CPU - generation will be slow"
        })
    
    return info


def load_model(
    model_id: str, 
    scheduler: str = "euler_a",
    enable_optimizations: bool = True,
    to_device: bool = True
) -> Union[StableDiffusionXLPipeline, StableDiffusionPipeline]:
    """Load a Stable Diffusion model with 2024 optimizations for Apple Silicon."""
    device_info = get_device_info()
    device = device_info["device"]
    
    console.print(f"[bold blue]Loading model:[/bold blue] {model_id}")
    console.print(f"[bold blue]Device:[/bold blue] {device}")
    console.print(f"[bold blue]Memory:[/bold blue] {device_info['available_memory_gb']:.1f}GB available")
    
    # Determine optimal settings based on device
    torch_dtype = torch.float32  # Default for MPS and CPU
    variant = None
    
    if device == "cuda" and device_info.get("gpu_memory_gb", 0) >= 6:
        torch_dtype = torch.float16
        variant = "fp16"
    
    try:
        # Load appropriate pipeline
        if "xl" in model_id.lower():
            console.print("[yellow]Loading SDXL model...[/yellow]")
            pipe = StableDiffusionXLPipeline.from_pretrained(
                model_id,
                torch_dtype=torch_dtype,
                use_safetensors=True,
                variant=variant,
                add_watermarker=False,  # Disable watermarker for speed
            )
        else:
            console.print("[yellow]Loading SD 1.5/2.x model...[/yellow]")
            pipe = StableDiffusionPipeline.from_pretrained(
                model_id,
                torch_dtype=torch_dtype,
                use_safetensors=True,
                variant=variant,
                safety_checker=None,  # Disable for local use
                requires_safety_checker=False,
            )
            
    except Exception as e:
        console.print(f"[red]Failed to load {model_id}: {e}[/red]")
        console.print("[yellow]Falling back to SD 1.5...[/yellow]")
        pipe = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float32,
            use_safetensors=True,
            safety_checker=None,
            requires_safety_checker=False,
        )

    # Apply scheduler
    scheduler_mapping = {
        "euler_a": EulerAncestralDiscreteScheduler,
        "dpm_multistep": DPMSolverMultistepScheduler,
        "heun": HeunDiscreteScheduler,
    }
    
    if scheduler in scheduler_mapping:
        pipe.scheduler = scheduler_mapping[scheduler].from_config(pipe.scheduler.config)

    # Move to device (if requested)
    if to_device:
        pipe = pipe.to(device)
        
        # Apply optimizations
        if enable_optimizations:
            apply_optimizations(pipe, device, device_info)
    else:
        # Keep on CPU, don't apply optimizations yet
        console.print("[yellow]Keeping model on CPU for custom device handling[/yellow]")
        pipe = pipe.to("cpu")
    
    console.print("[bold green]✓ Model loaded successfully[/bold green]")
    return pipe



def apply_optimizations(
    pipe: Union[StableDiffusionXLPipeline, StableDiffusionPipeline], 
    device: str, 
    device_info: Dict[str, Any]
) -> None:
    """Apply device-specific optimizations based on 2024 best practices."""
    memory_gb = device_info.get("available_memory_gb", 8)
    
    if device == "mps":
        console.print("[cyan]Applying MPS optimizations...[/cyan]")
        
        # Always enable attention slicing for MPS - reduces memory usage
        pipe.enable_attention_slicing()
        
        # Enable memory efficient attention if available
        try:
            pipe.enable_model_cpu_offload()
            console.print("✓ CPU offload enabled")
        except Exception:
            console.print("⚠ CPU offload not available")
        
        # Warmup pass for MPS stability (2024 best practice)
        console.print("[cyan]Performing MPS warmup pass...[/cyan]")
        try:
            is_sdxl = isinstance(pipe, StableDiffusionXLPipeline)
            
            # Use a proper detailed prompt suitable for both SDXL and SD models
            warmup_prompt = "a simple photo of a cat, high quality"
            
            warmup_params = {
                "prompt": warmup_prompt,
                "num_inference_steps": 2,  # Minimal steps for warmup
                "guidance_scale": 1.5,
                "height": 512,
                "width": 512,
            }
            
            # Add SDXL-specific parameters if needed
            if is_sdxl:
                warmup_params.update({
                    "negative_prompt": "low quality, blurry",
                })
            
            with torch.inference_mode():
                _ = pipe(**warmup_params)
                
            console.print("✓ MPS warmup completed")
        except Exception as e:
            # Provide more specific error message without stopping execution
            if "CUDA" in str(e):
                # This is a common misleading error on MPS
                console.print("⚠ MPS warmup skipped: MPS compatibility issue detected")
                console.print("  This won't affect model loading, but first generation may be slower")
            else:
                console.print(f"⚠ MPS warmup skipped: {str(e)}")
    
    elif device == "cuda":
        console.print("[cyan]Applying CUDA optimizations...[/cyan]")
        
        gpu_memory = device_info.get("gpu_memory_gb", 0)
        
        if gpu_memory < 8:
            pipe.enable_attention_slicing()
            console.print("✓ Attention slicing enabled (low VRAM)")
        
        if gpu_memory >= 10:
            # Enable memory efficient attention for high-end cards
            try:
                pipe.enable_xformers_memory_efficient_attention()
                console.print("✓ xFormers memory efficient attention enabled")
            except Exception:
                console.print("⚠ xFormers not available")
    
    else:  # CPU
        console.print("[cyan]Applying CPU optimizations...[/cyan]")
        pipe.enable_attention_slicing()
        
        if memory_gb < 16:
            try:
                pipe.enable_model_cpu_offload()
                console.print("✓ CPU offload enabled for low memory")
            except Exception:
                pass



def download_model(model_id: str = "stabilityai/stable-diffusion-xl-base-1.0") -> None:
    """Download and cache a Stable Diffusion model with progress tracking."""
    console.print(f"[bold blue]Downloading model:[/bold blue] {model_id}")
    
    device_info = get_device_info()
    console.print(f"[cyan]Available memory: {device_info['available_memory_gb']:.1f}GB[/cyan]")
    
    try:
        with Progress() as progress:
            task = progress.add_task("[cyan]Downloading...", total=100)
            pipe = load_model(model_id)
            progress.update(task, completed=100)
        
        console.print(f"[bold green]✓ Model {model_id} successfully downloaded and cached[/bold green]")
        
        # Clean up memory
        del pipe
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()
            
    except Exception as e:
        console.print(f"[bold red]✗ Failed to download model: {e}[/bold red]")
        sys.exit(1)


def validate_image(image_path: Path) -> bool:
    """Validate that an image file is valid and readable."""
    try:
        with Image.open(image_path) as img:
            img.verify()
        return True
    except Exception:
        return False


def create_grid(images: List[Image.Image], rows: int = 2, cols: int = 2) -> Image.Image:
    """Create a grid of images for comparison."""
    if not images:
        raise ValueError("No images provided")

    # Get dimensions from first image
    w, h = images[0].size
    grid = Image.new('RGB', size=(cols * w, rows * h), color='white')

    for i, img in enumerate(images[:rows * cols]):
        if img.size != (w, h):
            img = img.resize((w, h), Image.Resampling.LANCZOS)
        
        row = i // cols
        col = i % cols
        grid.paste(img, (col * w, row * h))

    return grid


def ensure_directory(path: Union[str, Path]) -> Path:
    """Ensure a directory exists, create if it doesn't."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path



def calculate_image_hash(image_path: Path) -> str:
    """Calculate SHA256 hash of an image for integrity checking."""
    with open(image_path, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()


def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent


def pack_submission() -> None:
    """Pack the project for submission with enhanced filtering."""
    project_root = get_project_root()
    output_file = project_root / "art_capstone_submission.zip"

    console.print("[bold blue]Packing submission...[/bold blue]")

    # Enhanced exclusion patterns for 2024
    exclude_patterns = [
        '.git/', '__pycache__/', '.pytest_cache/', 'venv/', 'env/', '.venv/',
        '.DS_Store', '*.pyc', '*.pyo', '*.pyd', '.coverage', 'htmlcov/',
        'node_modules/', '.env', '*.log', 'wandb/', '.huggingface/',
        '.mypy_cache/', '.tox/', '.nox/', 'models/', 'cache/', '.cache/',
        '*.ckpt', '*.safetensors', '*.bin'
    ]

    with zipfile.ZipFile(output_file, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file_path in project_root.rglob('*'):
            if file_path.is_file():
                # Check if file should be excluded
                should_exclude = any(
                    exclude in str(file_path) for exclude in exclude_patterns
                )
                
                # Also exclude very large files (>100MB)
                if file_path.stat().st_size > 100 * 1024 * 1024:
                    should_exclude = True

                if not should_exclude:
                    arcname = file_path.relative_to(project_root)
                    zipf.write(file_path, arcname)

    file_size = output_file.stat().st_size / (1024 * 1024)  # MB
    console.print(f"[bold green]✓ Submission packed:[/bold green] {output_file} ({file_size:.1f} MB)")


def check_dependencies() -> bool:
    """Check if all required dependencies are installed with version info."""
    # Map package names to their import names when different
    package_imports = {
        'torch': 'torch',
        'diffusers': 'diffusers', 
        'transformers': 'transformers',
        'accelerate': 'accelerate',
        'safetensors': 'safetensors',
        'pillow': 'PIL',  # pillow imports as PIL
        'numpy': 'numpy',
        'rich': 'rich',
        'psutil': 'psutil'
    }

    console.print("[bold blue]Checking dependencies...[/bold blue]")
    
    missing = []
    version_info = {}
    
    for package_name, import_name in package_imports.items():
        try:
            module = __import__(import_name)
            version = getattr(module, '__version__', 'unknown')
            version_info[package_name] = version
            console.print(f"✓ {package_name}: {version}")
        except ImportError:
            missing.append(package_name)
            console.print(f"✗ {package_name}: missing")

    if missing:
        console.print(f"[bold red]✗ Missing packages:[/bold red] {', '.join(missing)}")
        console.print("[yellow]Run: pip install -r requirements.txt[/yellow]")
        return False

    console.print("[bold green]✓ All dependencies installed[/bold green]")
    return True



def main():
    """Enhanced command line interface for utility functions."""
    parser = argparse.ArgumentParser(
        description="Art Capstone utilities - 2024 Enhanced",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Download model command
    download_parser = subparsers.add_parser(
        'download_model', 
        help='Download a Stable Diffusion model'
    )
    download_parser.add_argument(
        '--model-id', 
        default='stabilityai/stable-diffusion-xl-base-1.0',
        help='Model ID to download'
    )

    # Pack submission command
    subparsers.add_parser('pack_submission', help='Pack project for submission')

    # Test device command
    subparsers.add_parser('test_device', help='Test device availability and performance')

    # Check dependencies
    subparsers.add_parser('check_deps', help='Check if dependencies are installed')

    args = parser.parse_args()

    if args.command == 'download_model':
        download_model(args.model_id)
    elif args.command == 'pack_submission':
        pack_submission()
    elif args.command == 'test_device':
        device_info = get_device_info()
        console.print("[bold blue]Device Information:[/bold blue]")
        for key, value in device_info.items():
            console.print(f"  {key}: {value}")
    elif args.command == 'check_deps':
        check_dependencies()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
