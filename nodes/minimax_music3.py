"""MiniMax Music 3 text-to-music on MLX.

Adapts minimax_music3_mlx (github.com/mikolaj92/minimax-music3-mlx) into ComfyUI nodes.
Loader and Generate are split so ComfyUI's execution cache keeps the loaded modules
alive between queue runs.
"""

import os
import time

import numpy as np
import torch

import comfy.model_management
import folder_paths
from comfy.utils import ProgressBar

import mlx.core as mx
from minimax_music3_mlx.pipeline import generate_audio, load_modules

# Scanned for weight folders. A folder qualifies if it holds mlx_config.json,
# which both the upstream converter and the HF repos write.
_WEIGHT_ROOTS = [
    os.path.join(folder_paths.models_dir, "minimax_music3_mlx"),
    os.path.expanduser("~/Local/minimax-music3-mlx/weights"),
]

# AR decode dominates wall-clock; give it most of the progress bar.
_AR_SHARE = 85


def _find_weights():
    """Map display name -> absolute path for every weight dir we can see."""
    found = {}
    for root in _WEIGHT_ROOTS:
        if not os.path.isdir(root):
            continue
        for entry in sorted(os.listdir(root)):
            path = os.path.join(root, entry)
            if os.path.isfile(os.path.join(path, "mlx_config.json")):
                found.setdefault(entry, path)
    return found


class MiniMaxMusic3MLXLoader:
    @classmethod
    def INPUT_TYPES(cls):
        names = list(_find_weights().keys())
        return {"required": {"weights": (names if names else ["<none found>"],)}}

    RETURN_TYPES = ("MINIMAX_MUSIC3_MLX",)
    RETURN_NAMES = ("modules",)
    FUNCTION = "load"
    CATEGORY = "MLX/MiniMax Music 3"

    def load(self, weights):
        found = _find_weights()
        if weights not in found:
            raise RuntimeError(
                f"weights folder '{weights}' not found. Looked in: {', '.join(_WEIGHT_ROOTS)}"
            )
        start = time.perf_counter()
        modules = load_modules(found[weights], tiny=False, seed=0)
        print(
            f"[MiniMax-MLX] loaded '{weights}' in {time.perf_counter() - start:.2f}s "
            f"(mlx active {mx.get_active_memory() / 1e9:.2f} GB)",
            flush=True,
        )
        return (modules,)


class MiniMaxMusic3MLXGenerate:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "modules": ("MINIMAX_MUSIC3_MLX",),
                "prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "Genre: acoustic pop. BPM: 96. Warm female vocal.",
                        "tooltip": "Structured caption: genre, BPM, instrumentation, vocal style.",
                    },
                ),
                "lyrics": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "[verse]\nMorning light filtering through the pine\n[chorus]\nSoftly the world begins to breathe",
                        "tooltip": "Structure tags like [verse] / [chorus] must be on their own line.",
                    },
                ),
                "duration": (
                    "FLOAT",
                    {
                        "default": 30.0,
                        "min": 1.0,
                        "max": 120.0,
                        "step": 0.5,
                        "tooltip": "Upper bound in seconds (25 AR frames/s). The model often ends earlier.",
                    },
                ),
                "steps": ("INT", {"default": 30, "min": 1, "max": 200}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFF}),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "generate"
    CATEGORY = "MLX/MiniMax Music 3"

    def generate(self, modules, prompt, lyrics, duration, steps, seed):
        pbar = ProgressBar(100)
        state = {"phase": None, "t": time.perf_counter()}

        def on_progress(phase, current, total):
            # Makes the Cancel button work during a multi-minute generation.
            comfy.model_management.throw_exception_if_processing_interrupted()
            if phase != state["phase"]:
                if state["phase"] is not None:
                    print(
                        f"[MiniMax-MLX] {state['phase']} done in {time.perf_counter() - state['t']:.1f}s",
                        flush=True,
                    )
                state["phase"] = phase
                state["t"] = time.perf_counter()
            frac = current / max(1, total)
            pct = frac * _AR_SHARE if phase == "ar" else _AR_SHARE + frac * (100 - _AR_SHARE)
            pbar.update_absolute(int(pct), 100)

        mx.reset_peak_memory()
        start = time.perf_counter()
        wave, rate = generate_audio(
            modules,
            lyrics=lyrics,
            prompt=prompt,
            audio_duration=float(duration),
            num_inference_steps=int(steps),
            seed=int(seed),
            progress_callback=on_progress,
        )
        elapsed = time.perf_counter() - start
        pbar.update_absolute(100, 100)

        audio_seconds = wave.shape[0] / rate
        print(
            f"[MiniMax-MLX] {audio_seconds:.1f}s audio in {elapsed:.1f}s "
            f"({elapsed / max(audio_seconds, 1e-6):.2f}x realtime) | "
            f"mlx peak {mx.get_peak_memory() / 1e9:.2f} GB",
            flush=True,
        )

        # numpy [samples, 2] -> torch [1, 2, samples] to match ComfyUI's AUDIO type.
        waveform = torch.from_numpy(np.ascontiguousarray(wave.T)).unsqueeze(0).float()
        return ({"waveform": waveform, "sample_rate": rate},)


NODE_CLASS_MAPPINGS = {
    "MiniMaxMusic3MLXLoader": MiniMaxMusic3MLXLoader,
    "MiniMaxMusic3MLXGenerate": MiniMaxMusic3MLXGenerate,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MiniMaxMusic3MLXLoader": "MiniMax Music 3 MLX Loader",
    "MiniMaxMusic3MLXGenerate": "MiniMax Music 3 MLX Generate",
}
