import os
import random
import asyncio
from typing import List, Tuple

import torch
from diffusers import StableDiffusionXLPipeline
from PIL import Image

# Global pipeline instance – loaded lazily on first use
_pipeline: StableDiffusionXLPipeline | None = None

def _load_pipeline() -> StableDiffusionXLPipeline:
    """
    Lazily loads the Stable Diffusion XL pipeline (stabilityai/stable-diffusion-xl-base-1.0)
    onto the appropriate device (GPU if available, otherwise CPU). The pipeline
    is cached in the module‑level ``_pipeline`` variable for reuse.
    """
    global _pipeline
    if _pipeline is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32
        _pipeline = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=dtype,
        )
        _pipeline = _pipeline.to(device)
        if device == "cuda":
            _pipeline.enable_attention_slicing()
    return _pipeline

def _apply_loras(pipeline: StableDiffusionXLPipeline, lora_names: List[str]) -> None:
    """
    Loads LoRA checkpoint files from the pipeline.
    """
    if not lora_names:
        return

    loras_dir = os.path.join(os.path.dirname(__file__), "..", "loras")
    for name in lora_names:
        lora_path = os.path.join(loras_dir, name)
        if not os.path.isfile(lora_path):
            continue
        try:
            pipeline.load_lora_weights(lora_path)
        except Exception as exc:
            print(f"[diffusion] Failed to load LoRA '{name}': {exc}")

async def generate_image(
    prompt: str,
    size: Tuple[int, int] = (512, 512),
    steps: int = 30,
    lora_names: List[str] = None,
) -> Image.Image:
    """
    Generate an image using Stable Diffusion with optional LoRAs.
    Returns a Pillow Image.
    """
    if lora_names is None:
        lora_names = []

    def _run():
        pipeline = _load_pipeline()
        _apply_loras(pipeline, lora_names)
        generator = torch.Generator(device=pipeline.device).manual_seed(random.randint(0, 2**32 - 1))
        output = pipeline(
            prompt,
            height=size[0],
            width=size[1],
            num_inference_steps=steps,
            generator=generator,
        )
        return output.images[0]

    return await asyncio.to_thread(_run)
