"""mflux image models (Z-Image, Qwen-Image, FLUX) on MLX.

All mflux variants share one constructor shape — (quantize, model_path, lora_paths,
lora_scales, model_config) — and one generate_image() entry point, so a single
Loader/Generate pair covers every family in _MODELS.

Progress and cancellation use mflux's own CallbackRegistry; no patching needed.
"""

import time

import numpy as np
import torch

import comfy.model_management
from comfy.utils import ProgressBar

import mlx.core as mx
from mflux.models.common.config.model_config import ModelConfig
from mflux.models.flux.variants.txt2img.flux import Flux1
from mflux.models.qwen.variants.txt2img.qwen_image import QwenImage
from mflux.models.z_image.variants.z_image import ZImage

# display name -> (variant class, ModelConfig factory)
_MODELS = {
    "z-image-turbo": (ZImage, ModelConfig.z_image_turbo),
    "z-image": (ZImage, ModelConfig.z_image),
    "qwen-image": (QwenImage, ModelConfig.qwen_image),
    "qwen-image-edit": (QwenImage, ModelConfig.qwen_image_edit),
    "flux-schnell": (Flux1, ModelConfig.schnell),
    "flux-dev": (Flux1, ModelConfig.dev),
    "flux-krea-dev": (Flux1, ModelConfig.krea_dev),
}

_QUANTIZE = ["none", "3", "4", "5", "6", "8"]


class _ComfyProgress:
    """Bridges mflux's callback protocol to ComfyUI's progress bar and Cancel button."""

    def __init__(self, total_steps):
        self.pbar = ProgressBar(total_steps)
        self.total = total_steps

    def call_in_loop(self, t, seed, prompt, latents, config, time_steps):
        # mflux only catches KeyboardInterrupt, so ComfyUI's interrupt exception
        # propagates out of the loop untouched — which is what we want.
        comfy.model_management.throw_exception_if_processing_interrupted()
        self.pbar.update_absolute(min(t + 1, self.total), self.total)


class MFluxLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (list(_MODELS.keys()),),
                "quantize": (_QUANTIZE, {"default": "8"}),
            },
            "optional": {
                "model_path": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Local weight dir. Leave empty to fetch the model's default repo from Hugging Face.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("MFLUX_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load"
    CATEGORY = "MLX/mflux"

    def load(self, model, quantize, model_path=""):
        variant, config_factory = _MODELS[model]
        bits = None if quantize == "none" else int(quantize)
        start = time.perf_counter()
        instance = variant(
            quantize=bits,
            model_path=model_path.strip() or None,
            model_config=config_factory(),
        )
        print(
            f"[mflux] loaded '{model}' q={quantize} in {time.perf_counter() - start:.1f}s "
            f"(mlx active {mx.get_active_memory() / 1e9:.2f} GB)",
            flush=True,
        )
        return (instance,)


class MFluxGenerate:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MFLUX_MODEL",),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "width": ("INT", {"default": 1024, "min": 256, "max": 4096, "step": 16}),
                "height": ("INT", {"default": 1024, "min": 256, "max": 4096, "step": 16}),
                "steps": (
                    "INT",
                    {"default": 4, "min": 1, "max": 100, "tooltip": "Turbo/schnell models want ~4; dev models ~20-30."},
                ),
                "guidance": (
                    "FLOAT",
                    {"default": 3.5, "min": 0.0, "max": 30.0, "step": 0.1, "tooltip": "Ignored by models that don't support guidance."},
                ),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFF}),
            },
            "optional": {
                "negative_prompt": ("STRING", {"multiline": True, "default": ""}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate"
    CATEGORY = "MLX/mflux"

    def generate(self, model, prompt, width, height, steps, guidance, seed, negative_prompt=""):
        progress = _ComfyProgress(steps)
        model.callbacks.register(progress)

        mx.reset_peak_memory()
        start = time.perf_counter()
        try:
            generated = model.generate_image(
                seed=int(seed),
                prompt=prompt,
                num_inference_steps=int(steps),
                height=int(height),
                width=int(width),
                guidance=float(guidance),
                negative_prompt=negative_prompt.strip() or None,
            )
        finally:
            # The registry lives on the cached model, so stale bars would accumulate.
            model.callbacks.in_loop_callbacks().remove(progress)

        print(
            f"[mflux] {width}x{height} in {time.perf_counter() - start:.1f}s "
            f"| mlx peak {mx.get_peak_memory() / 1e9:.2f} GB",
            flush=True,
        )

        # PIL -> ComfyUI IMAGE: float32 [B, H, W, C] in 0..1
        rgb = generated.image.convert("RGB")
        array = np.asarray(rgb, dtype=np.float32) / 255.0
        return (torch.from_numpy(array).unsqueeze(0),)


NODE_CLASS_MAPPINGS = {
    "MFluxLoader": MFluxLoader,
    "MFluxGenerate": MFluxGenerate,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MFluxLoader": "mflux Loader (MLX)",
    "MFluxGenerate": "mflux Generate (MLX)",
}
