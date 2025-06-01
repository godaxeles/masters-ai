#!/usr/bin/env python3
"""Generate cover variants locally with Diffusers - Updated 2024 with TDD principles."""

import argparse
import json
import pathlib
import random
import time
from datetime import datetime
from typing import List, Optional, Dict, Any

import torch
from diffusers import StableDiffusionXLPipeline, StableDiffusionPipeline
from PIL import Image
from rich.console import Console

from utils import load_model, set_seed, get_device, ensure_directory, get_device_info

console = Console()


class ImageGenerator:
    """Main class for generating image variants with enhanced 2024 features."""

    def __init__(self, model_id: str = "stabilityai/stable-diffusion-xl-base-1.0"):
        self.model_id = model_id
        self.device_info = get_device_info()
        self.device = self.device_info["device"]
        self.pipe = None
        self.generation_log = []
        self._validate_model_id(model_id)

    def _validate_model_id(self, model_id: str) -> None:
        """Validate model ID format - TDD principle."""
        if not isinstance(model_id, str) or not model_id.strip():
            raise ValueError("Model ID must be a non-empty string")

    def load_pipeline(self, sampler: str = "euler_a", device: Optional[str] = None) -> None:
        """Load the diffusion pipeline with validation."""
        if self.pipe is not None:
            return
            
        # Override device if specified
        if device:
            self.device = device
        
        # Check available memory for MPS device
        if self.device == "mps":
            available_memory = self.device_info.get("available_memory_gb", 0)
            console.print(f"[cyan]MPS device detected with {available_memory:.1f}GB available memory[/cyan]")
            
            # Check if we have enough memory for SDXL on MPS
            if "xl" in self.model_id.lower() and available_memory < 6:
                console.print("[yellow]⚠ Warning: SDXL requires at least 6GB of memory on MPS. Falling back to CPU.[/yellow]")
                self.device = "cpu"
                
        # Check memory before loading model
        initial_memory = torch.mps.current_allocated_memory() / (1024**3) if self.device == "mps" else 0
        console.print(f"[bold blue]Loading model:[/bold blue] {self.model_id}")
        
        # For MPS, we need to customize the loading process
        if self.device == "mps":
            try:
                # Load model but don't move to MPS directly - we'll do it component by component
                console.print("[cyan]Using custom MPS loading approach...[/cyan]")
                self.pipe = load_model(self.model_id, sampler, enable_optimizations=False)
                
                # First move pipeline to CPU explicitly
                self.pipe = self.pipe.to("cpu")
                
                # Clear MPS cache before moving components
                torch.mps.empty_cache()
                console.print(f"[cyan]Memory after loading: {torch.mps.current_allocated_memory() / (1024**3):.2f}GB[/cyan]")
                
                # Move components individually to MPS with specific dtype
                console.print("[cyan]Moving components to MPS one by one...[/cyan]")
                
                # Handle UNet - most critical component
                if hasattr(self.pipe, "unet"):
                    try:
                        # Convert to float32 and contiguous format first
                        self.pipe.unet = self.pipe.unet.to(dtype=torch.float32, memory_format=torch.contiguous_format)
                        # Move to MPS device
                        self.pipe.unet = self.pipe.unet.to(self.device)
                        console.print("✓ UNet moved to MPS successfully")
                    except Exception as unet_error:
                        console.print(f"[yellow]⚠ Failed to move UNet to MPS: {unet_error}[/yellow]")
                        # Critical failure - fall back to CPU
                        self.device = "cpu"
                        self.pipe = self.pipe.to("cpu")
                        console.print("[yellow]⚠ Falling back to CPU due to MPS UNet error[/yellow]")
                        return
                
                # Clear memory after moving UNet
                torch.mps.empty_cache()
                
                # Text encoder(s)
                if self.device == "mps":  # Only continue if we're still using MPS
                    if hasattr(self.pipe, "text_encoder"):
                        try:
                            self.pipe.text_encoder = self.pipe.text_encoder.to(dtype=torch.float32)
                            self.pipe.text_encoder = self.pipe.text_encoder.to(self.device)
                            console.print("✓ Text encoder moved to MPS")
                        except Exception as te_error:
                            console.print(f"[yellow]⚠ Failed to move text encoder to MPS: {te_error}[/yellow]")
                    
                    # SDXL has a second text encoder
                    if hasattr(self.pipe, "text_encoder_2") and self.pipe.text_encoder_2 is not None:
                        try:
                            self.pipe.text_encoder_2 = self.pipe.text_encoder_2.to(dtype=torch.float32)
                            self.pipe.text_encoder_2 = self.pipe.text_encoder_2.to(self.device)
                            console.print("✓ Text encoder 2 moved to MPS")
                        except Exception as te2_error:
                            console.print(f"[yellow]⚠ Failed to move text encoder 2 to MPS: {te2_error}[/yellow]")
                
                # Clear memory again
                torch.mps.empty_cache()
                
                # VAE - can be kept on CPU if needed
                if self.device == "mps" and hasattr(self.pipe, "vae"):
                    try:
                        self.pipe.vae = self.pipe.vae.to(dtype=torch.float32)
                        self.pipe.vae = self.pipe.vae.to(self.device)
                        console.print("✓ VAE moved to MPS")
                    except Exception as vae_error:
                        console.print(f"[yellow]⚠ Failed to move VAE to MPS: {vae_error}[/yellow]")
                        console.print("[cyan]Keeping VAE on CPU - this is fine[/cyan]")
                
                # Special case for SDXL on MPS: additional optimizations
                if self.device == "mps" and "xl" in self.model_id.lower():
                    console.print("[cyan]Applying SDXL-specific MPS optimizations...[/cyan]")
                    
                    # Disable memory efficient attention for SDXL on MPS
                    if hasattr(self.pipe, "enable_attention_slicing"):
                        self.pipe.enable_attention_slicing(1)
                        console.print("✓ Applied aggressive attention slicing")
                
                # Disable safety checker for better performance
                if hasattr(self.pipe, "safety_checker") and self.pipe.safety_checker is not None:
                    self.pipe.safety_checker = None
                    console.print("✓ Safety checker disabled for performance")
                
                # Report final memory usage
                if self.device == "mps":
                    final_memory = torch.mps.current_allocated_memory() / (1024**3)
                    console.print(f"[cyan]MPS memory usage after setup: {final_memory:.2f}GB (change: {final_memory-initial_memory:.2f}GB)[/cyan]")
                
            except Exception as e:
                console.print(f"[yellow]⚠ MPS setup error: {e}[/yellow]")
                console.print("[yellow]Falling back to CPU device[/yellow]")
                self.device = "cpu"
                self.pipe = self.pipe.to("cpu")
        else:
            # For non-MPS devices, use standard loading
            self.pipe = load_model(self.model_id, sampler)
        
        console.print("[bold green]✓ Pipeline loaded successfully[/bold green]")

    def validate_generation_params(self, prompt: str, steps: int, cfg: float, width: int, height: int) -> None:
        """Validate generation parameters - TDD principle."""
        if not prompt or not prompt.strip():
            raise ValueError("Prompt cannot be empty")
        if not 1 <= steps <= 150:
            raise ValueError("Steps must be between 1 and 150")
        if not 0.1 <= cfg <= 30.0:
            raise ValueError("CFG scale must be between 0.1 and 30.0")
        if width % 8 != 0 or height % 8 != 0:
            raise ValueError("Width and height must be multiples of 8")

    def generate_image(
        self,
        prompt: str,
        negative_prompt: str = "blurry, low quality, distorted, ugly, bad anatomy",
        steps: int = 30,
        cfg: float = 7.5,
        width: int = 1024,
        height: int = 1024,
        seed: Optional[int] = None
    ) -> Image.Image:
        """Generate a single image from prompt with enhanced validation."""

        if self.pipe is None:
            raise RuntimeError("Pipeline not loaded. Call load_pipeline() first.")

        # Validate parameters using TDD approach
        self.validate_generation_params(prompt, steps, cfg, width, height)

        # Set seed for reproducibility
        if seed is not None:
            set_seed(seed)
            # Always create generator on CPU for compatibility
            generator = torch.Generator("cpu").manual_seed(seed)
            
            # Special handling for different devices
            if self.device == "mps":
                # For MPS, keep generator on CPU as MPS doesn't fully support generators
                console.print("[cyan]Using CPU-based generator for MPS compatibility[/cyan]")
            elif self.device == "cuda":
                # For CUDA, move generator to GPU
                generator = torch.Generator(self.device).manual_seed(seed)
        else:
            generator = None

        # Generate image with error handling
        start_time = time.time()
        
        try:
            # Prepare device-specific parameters
            generation_params = {
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "num_inference_steps": steps,
                "guidance_scale": cfg,
                "width": width,
                "height": height,
                "generator": generator
            }
            
            # Handle MPS-specific settings
            if self.device == "mps":
                # MPS requires some special handling
                try:
                    # Clear MPS cache before inference
                    torch.mps.empty_cache()
                    
                    # Use inference_mode for better memory handling
                    with torch.inference_mode():
                        # Add a special debugging block
                        console.print(f"[cyan]Starting generation on MPS with {torch.mps.current_allocated_memory() / (1024**3):.2f}GB allocated[/cyan]")
                        
                        if isinstance(self.pipe, StableDiffusionXLPipeline):
                            # For SDXL on MPS - use smaller batch size
                            generation_params["num_images_per_prompt"] = 1
                            result = self.pipe(**generation_params)
                        else:
                            # For SD 1.5/2.0 on MPS
                            result = self.pipe(**generation_params)
                except RuntimeError as e:
                    # Check for common MPS/CUDA errors
                    error_str = str(e).lower()
                    if "cuda" in error_str or "device-side assert" in error_str or "mps" in error_str:
                        # This is a common misleading error on MPS
                        console.print("[yellow]⚠ MPS compatibility issue detected. Trying CPU fallback...[/yellow]")
                        console.print(f"[yellow]Error details: {e}[/yellow]")
                        
                        # Move to CPU as fallback - do it component by component
                        console.print("[cyan]Moving pipeline to CPU component by component...[/cyan]")
                        
                        if hasattr(self.pipe, "unet"):
                            self.pipe.unet = self.pipe.unet.to("cpu")
                        if hasattr(self.pipe, "vae"):
                            self.pipe.vae = self.pipe.vae.to("cpu")
                        if hasattr(self.pipe, "text_encoder"):
                            self.pipe.text_encoder = self.pipe.text_encoder.to("cpu")
                        if hasattr(self.pipe, "text_encoder_2") and self.pipe.text_encoder_2 is not None:
                            self.pipe.text_encoder_2 = self.pipe.text_encoder_2.to("cpu")
                            
                        self.device = "cpu"
                        
                        # Free up MPS memory
                        torch.mps.empty_cache()
                        
                        # Try again on CPU with modified parameters
                        generation_params["num_inference_steps"] = min(generation_params["num_inference_steps"], 20)  # Reduce steps on CPU
                        with torch.inference_mode():
                            console.print("[cyan]Generating on CPU (slower but more compatible)...[/cyan]")
                            result = self.pipe(**generation_params)
                    else:
                        # Re-raise if it's not the CUDA/MPS error
                        raise
            else:
                # Normal processing for CPU or CUDA
                with torch.inference_mode():
                    if isinstance(self.pipe, StableDiffusionXLPipeline):
                        result = self.pipe(**generation_params)
                    else:
                        result = self.pipe(**generation_params)

            image = result.images[0]
            generation_time = time.time() - start_time

            # Log generation details
            self.generation_log.append({
                'timestamp': datetime.now().isoformat(),
                'prompt': prompt,
                'negative_prompt': negative_prompt,
                'steps': steps,
                'cfg': cfg,
                'width': width,
                'height': height,
                'seed': seed,
                'generation_time': generation_time,
                'model_id': self.model_id,
                'device': self.device
            })

            return image

        except Exception as e:
            # Improved error handling with device-specific messages
            if self.device == "mps" and "CUDA" in str(e):
                console.print("[bold red]✗ Generation failed due to MPS compatibility issue[/bold red]")
                console.print("[yellow]This is likely due to a compatibility issue between PyTorch and MPS.[/yellow]")
                console.print("[yellow]Try using --device cpu as a fallback.[/yellow]")
            else:
                console.print(f"[bold red]✗ Generation failed: {e}[/bold red]")
            raise

    def cleanup(self) -> None:
        """Clean up resources."""
        if self.pipe is not None:
            del self.pipe
            self.pipe = None

        if self.device == "mps":
            torch.mps.empty_cache()
        elif self.device == "cuda":
            torch.cuda.empty_cache()


def load_prompts(prompt_file: pathlib.Path) -> List[str]:
    """Load prompts from file with validation."""
    if not prompt_file.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_file}")

    content = prompt_file.read_text().strip()
    prompts = [p.strip() for p in content.split('\n') if p.strip()]

    if not prompts:
        raise ValueError("No valid prompts found in file")

    return prompts


def create_default_prompts() -> List[str]:
    """Create default prompts for testing - 2024 enhanced."""
    return [
        "masterpiece album cover art, vintage aesthetic, professional photography, high detail",
        "modern minimalist cover design, clean typography, contemporary aesthetic, sophisticated",
        "retro vinyl cover art, nostalgic mood, warm colors, classic design, artistic illustration",
        "electronic music cover, futuristic design, neon accents, digital art style, cyberpunk",
        "jazz album cover, smoky atmosphere, musical instruments, noir lighting, vintage elegance"
    ]


def main():
    """Main CLI interface for image generation with enhanced features."""
    parser = argparse.ArgumentParser(description="Generate cover variants with Stable Diffusion - 2024")

    parser.add_argument("--prompt-file", help="File containing prompts")
    parser.add_argument("--output-dir", default="data/generated", help="Output directory")
    parser.add_argument("--num-variants", type=int, default=3, help="Number of variants")
    parser.add_argument("--steps", type=int, default=30, help="Inference steps")
    parser.add_argument("--cfg", type=float, default=7.5, help="Guidance scale")
    parser.add_argument("--seed", type=int, help="Random seed")
    parser.add_argument("--width", type=int, default=1024, help="Image width")
    parser.add_argument("--height", type=int, default=1024, help="Image height")
    parser.add_argument("--model-id", default="stabilityai/stable-diffusion-xl-base-1.0", help="Model ID")
    parser.add_argument("--device", choices=["cuda", "mps", "cpu"], help="Override device selection")
    parser.add_argument("--use-defaults", action="store_true", help="Use default prompts")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")

    args = parser.parse_args()

    # Load prompts with enhanced error handling
    try:
        if args.prompt_file:
            prompts = load_prompts(pathlib.Path(args.prompt_file))
        elif args.use_defaults:
            prompts = create_default_prompts()
            console.print("[cyan]Using enhanced default prompts[/cyan]")
        else:
            # Try default prompt.txt
            default_file = pathlib.Path("prompt.txt")
            if default_file.exists():
                prompts = load_prompts(default_file)
            else:
                console.print("[yellow]No prompt file specified. Use --prompt-file or --use-defaults[/yellow]")
                return 1
    except Exception as e:
        console.print(f"[bold red]Error loading prompts: {e}[/bold red]")
        return 1

    console.print(f"[bold green]Loaded {len(prompts)} prompt(s)[/bold green]")
    if args.verbose:
        for i, prompt in enumerate(prompts, 1):
            console.print(f"  {i}. {prompt[:100]}{'...' if len(prompt) > 100 else ''}")

    # Generate images with enhanced progress tracking
    generator = ImageGenerator(model_id=args.model_id)

    try:
        generator.load_pipeline(device=args.device)

        output_dir = pathlib.Path(args.output_dir)
        ensure_directory(output_dir)

        generated_files = []

        for i in range(args.num_variants):
            prompt = random.choice(prompts)
            seed = args.seed if args.seed else random.randint(0, 2**32 - 1)

            console.print(f"\n[bold cyan]Generating variant {i+1}/{args.num_variants}[/bold cyan]")
            console.print(f"Prompt: {prompt[:100]}{'...' if len(prompt) > 100 else ''}")
            console.print(f"Seed: {seed}")

            try:
                image = generator.generate_image(
                    prompt=prompt,
                    steps=args.steps,
                    cfg=args.cfg,
                    width=args.width,
                    height=args.height,
                    seed=seed
                )

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"variant-{i+1}_{timestamp}_seed{seed}.png"
                output_path = output_dir / filename
                image.save(output_path, quality=95, optimize=True)

                generation_time = generator.generation_log[-1]['generation_time']
                console.print(f"[bold green]✓ Saved {filename} (took {generation_time:.2f}s)[/bold green]")

                generated_files.append(output_path)

            except Exception as e:
                console.print(f"[bold red]✗ Failed to generate variant {i+1}: {e}[/bold red]")
                continue

        # Save generation log
        log_file = output_dir / "generation_log.json"
        with open(log_file, 'w') as f:
            json.dump(generator.generation_log, f, indent=2)

        console.print(f"\n[bold green]✓ Generated {len(generated_files)} images in {output_dir}[/bold green]")
        console.print("\n[bold blue]Generated files:[/bold blue]")
        for file_path in generated_files:
            console.print(f"  - {file_path.name}")

        console.print(f"\n[cyan]Generation log saved to: {log_file}[/cyan]")

        return 0

    except Exception as e:
        console.print(f"\n[bold red]✗ Generation failed: {e}[/bold red]")
        return 1

    finally:
        generator.cleanup()


if __name__ == "__main__":
    exit(main())
